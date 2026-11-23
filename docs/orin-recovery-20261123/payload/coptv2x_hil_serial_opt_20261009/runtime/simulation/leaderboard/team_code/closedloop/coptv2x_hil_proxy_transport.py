"""ROS result/trigger transport through the existing Orin Foxy host proxy.

The CARLA Python 3.7 environment on the 4090 host has no ``rclpy``. This
loopback-only TCP endpoint is reached from Orin through an SSH reverse forward;
the Orin ROS proxy translates JSON triggers and framed result wire bytes to the
existing ROS topics. Tensor payloads only flow Orin -> CARLA.
"""

from __future__ import print_function

import copy
import json
import socket
import struct
import sys
from threading import Event, Lock, Thread


_LENGTH = struct.Struct("!I")
_DEFAULT_MAX_WIRE_BYTES = 64 * 1024 * 1024


def _recv_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Orin ROS proxy closed the result stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_result_wire(wire, max_wire_bytes, readonly_results=False):
    if not isinstance(wire, bytes) or not wire or len(wire) > max_wire_bytes:
        raise ValueError("proxy result wire is empty or exceeds its size limit")
    from coptv2x_hil_backend import HilFeatureResult
    from coptv2x_hil_trigger import parse_hil_trigger
    from ditl_edge_protocol import ProtocolLimits, decode_wire_bytes
    from coptv2x_hil_result_buffer import HilResultEnvelope

    limits = ProtocolLimits(
        max_payload_bytes=max_wire_bytes,
        max_raw_payload_bytes=max_wire_bytes,
    )
    header, outputs, _, _ = decode_wire_bytes(
        wire, limits, return_stages=True, readonly=readonly_results,
    )
    if header.get("kind") != "result" or header.get("status") != "ok":
        raise ValueError("proxy result must be a successful HIL result")
    trigger = parse_hil_trigger(header.get("trigger_identity"))
    identity = trigger.to_metadata()
    for field, expected in identity.items():
        if header.get(field) != expected:
            raise ValueError("proxy result identity conflicts at %s" % field)
    batch_order = header.get("batch_order")
    if not isinstance(batch_order, list):
        raise ValueError("proxy result batch_order is missing")
    result = HilFeatureResult(
        outputs=outputs,
        artifact_identity=header.get("artifact_identity", {}),
        trigger_identity=identity,
        payload_semantics=header.get("payload_semantics"),
        payload_sha256=header.get("payload_sha256"),
        remote_input_tensor_bytes=header.get("remote_input_tensor_bytes"),
        timestamps_ns=header.get("edge_timestamps_ns", {}),
        durations_ns=header.get("durations_ns", {}),
        model=header.get("model", {}),
    )
    request_id = header.get("request_id")
    if type(request_id) is not str or not request_id:
        raise ValueError("proxy result request_id is missing")
    return HilResultEnvelope(result, tuple(batch_order)), request_id


class ReverseSshRosProxyTransport(object):
    """AsyncHilClient transport backed by an Orin ROS 2 proxy connection."""

    def __init__(self, listen_host="127.0.0.1", listen_port=56610,
                 connect_timeout_s=60.0, socket_timeout_s=7200.0,
                 max_wire_bytes=_DEFAULT_MAX_WIRE_BYTES, readonly_results=False):
        self.listen_host = str(listen_host)
        self.listen_port = int(listen_port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.socket_timeout_s = float(socket_timeout_s)
        self.max_wire_bytes = int(max_wire_bytes)
        if type(readonly_results) is not bool:
            raise ValueError("readonly_results must be a boolean")
        self.readonly_results = readonly_results
        if self.listen_host not in ("127.0.0.1", "localhost"):
            raise ValueError("HIL proxy listener must stay on loopback")
        if not 0 <= self.listen_port <= 65535:
            raise ValueError("HIL proxy listener port is invalid")
        if (
            self.connect_timeout_s <= 0
            or self.socket_timeout_s <= 0
            or self.max_wire_bytes <= 0
        ):
            raise ValueError("HIL proxy limits must be positive")
        self._handler = None
        self._server = None
        self._connection = None
        self._thread = None
        self._connected = Event()
        self._closed = Event()
        self._send_lock = Lock()
        self._last_error = None

    @property
    def bound_port(self):
        if self._server is None:
            return None
        return int(self._server.getsockname()[1])

    @property
    def last_error(self):
        return self._last_error

    def start(self, handler):
        if not callable(handler):
            raise TypeError("result handler must be callable")
        if self._server is not None:
            raise RuntimeError("HIL proxy transport is already started")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.listen_host, self.listen_port))
        server.listen(1)
        server.settimeout(0.5)
        self._server = server
        self._handler = handler
        self._thread = Thread(target=self._serve, name="coptv2x-hil-proxy", daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            while not self._closed.is_set():
                try:
                    connection, _ = self._server.accept()
                    break
                except socket.timeout:
                    continue
            else:
                return
            connection.settimeout(self.socket_timeout_s)
            self._connection = connection
            self._connected.set()
            while not self._closed.is_set():
                length = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))[0]
                if length <= 0 or length > self.max_wire_bytes:
                    raise ValueError("proxy result frame length violates its limit")
                wire = _recv_exact(connection, length)
                envelope, request_id = _decode_result_wire(
                    wire, self.max_wire_bytes, readonly_results=self.readonly_results,
                )
                self._handler(envelope)
                acknowledgement = ("ACK " + request_id + "\n").encode("ascii")
                with self._send_lock:
                    connection.sendall(acknowledgement)
        except (OSError, EOFError, ValueError, TypeError) as error:
            self._last_error = error
            print(
                "CoptV2X HIL result transport closed: %s: %s"
                % (type(error).__name__, error),
                file=sys.stderr,
                flush=True,
            )
            if not self._closed.is_set() and self._handler is not None:
                try:
                    self._handler(None)
                except Exception:
                    pass
        finally:
            self._connected.clear()
            if self._connection is not None:
                try:
                    self._connection.close()
                except OSError:
                    pass
                self._connection = None

    def publish_trigger(self, payload):
        if not self._connected.wait(self.connect_timeout_s):
            raise RuntimeError("Orin ROS proxy did not connect through SSH forward")
        message = ("TRIGGER " + json.dumps(
            copy.deepcopy(payload), sort_keys=True, separators=(",", ":")
        ) + "\n").encode("utf-8")
        with self._send_lock:
            if self._connection is None:
                raise RuntimeError("Orin ROS proxy connection is closed")
            self._connection.sendall(message)

    def close(self):
        self._closed.set()
        for connection in (self._connection, self._server):
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    connection.close()
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._connection = None
        self._server = None


__all__ = ("ReverseSshRosProxyTransport",)
