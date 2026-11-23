#!/usr/bin/env python3
"""Bridge CARLA's reverse-SSH trigger/result stream through existing ROS topics."""

from __future__ import print_function

import argparse
import json
import queue
import socket
import struct
import threading
import time
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation/leaderboard/team_code/closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_trigger import parse_hil_trigger
from ditl_edge_protocol import FRAME_PREFIX, MAGIC, REQUEST_ID_RE


LENGTH_PREFIX = struct.Struct("!I")
MAX_TRIGGER_BYTES = 16 * 1024
MAX_RESULT_WIRE_BYTES = 64 * 1024 * 1024


def request_id_from_result_wire(wire):
    if not isinstance(wire, bytes) or len(wire) < FRAME_PREFIX.size:
        raise ValueError("ROS result wire is truncated")
    magic, header_size, payload_size = FRAME_PREFIX.unpack(wire[:FRAME_PREFIX.size])
    expected_size = FRAME_PREFIX.size + header_size + payload_size
    if magic != MAGIC or expected_size != len(wire):
        raise ValueError("ROS result wire framing is invalid")
    header_start = FRAME_PREFIX.size
    try:
        header = json.loads(wire[header_start:header_start + header_size].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("ROS result header is invalid JSON") from error
    if header.get("kind") != "result" or header.get("status") != "ok":
        raise ValueError("ROS result is not a successful result frame")
    request_id = header.get("request_id")
    if type(request_id) is not str or not REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError("ROS result request_id is invalid")
    return request_id


def frame_result_for_host(wire, maximum=MAX_RESULT_WIRE_BYTES):
    request_id_from_result_wire(wire)
    if len(wire) > maximum:
        raise ValueError("ROS result wire exceeds configured maximum")
    return LENGTH_PREFIX.pack(len(wire)) + wire


def parse_host_message(line):
    if not isinstance(line, bytes) or len(line) > MAX_TRIGGER_BYTES or not line.endswith(b"\n"):
        raise ValueError("reverse-SSH host message is malformed or too large")
    try:
        text = line[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("reverse-SSH host message is not UTF-8") from error
    if not text.startswith("TRIGGER "):
        raise ValueError("reverse-SSH host message type is unsupported")
    try:
        trigger = parse_hil_trigger(json.loads(text[len("TRIGGER "):]))
    except (TypeError, ValueError) as error:
        raise ValueError("reverse-SSH trigger violates the HIL contract") from error
    return trigger.to_metadata()


def validate_ack_line(line, expected_request_id):
    if not isinstance(line, bytes) or not line.endswith(b"\n"):
        raise ValueError("reverse-SSH ACK is malformed")
    try:
        text = line[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("reverse-SSH ACK is not ASCII") from error
    parts = text.split(" ")
    if len(parts) != 2 or parts[0] != "ACK" or not REQUEST_ID_RE.fullmatch(parts[1]):
        raise ValueError("reverse-SSH ACK format is invalid")
    if parts[1] != expected_request_id:
        raise ValueError("reverse-SSH ACK request_id does not match result")
    return parts[1]


def is_reconnectable_disconnect(error):
    """Episode-scoped host listeners may close between frozen model arms."""

    return isinstance(
        error,
        (EOFError, ConnectionResetError, BrokenPipeError, socket.timeout),
    )


def _read_line(connection, maximum_bytes):
    chunks = bytearray()
    while len(chunks) <= maximum_bytes:
        value = connection.recv(1)
        if not value:
            raise EOFError("reverse-SSH socket closed")
        chunks.extend(value)
        if value == b"\n":
            return bytes(chunks)
    raise ValueError("reverse-SSH line exceeds configured maximum")


def _ros_imports():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String, UInt8MultiArray
    except ImportError as error:
        raise RuntimeError("ROS 2 Foxy Python modules are required") from error
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=16,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    return rclpy, Node, String, UInt8MultiArray, qos


class CoptV2xHilRosProxy(object):
    def __init__(self, args, rclpy, Node, String, UInt8MultiArray, qos):
        self.args = args
        self.rclpy = rclpy
        self.node = Node("coptv2x_hil_ros_proxy")
        self.String = String
        self.UInt8MultiArray = UInt8MultiArray
        self.trigger_publisher = self.node.create_publisher(String, args.trigger_topic, qos)
        self.ack_publisher = self.node.create_publisher(String, args.ack_topic, qos)
        self.result_subscription = self.node.create_subscription(
            UInt8MultiArray, args.result_topic, self._on_result, qos
        )
        self.trace_subscription = self.node.create_subscription(
            String, args.trace_topic, self._on_trace, qos
        )
        self.trigger_subscription = self.node.create_subscription(
            String, args.trigger_topic, self._on_trigger, qos
        )
        self.triggers_to_publish = queue.Queue(maxsize=args.queue_capacity)
        self.triggers_to_worker = queue.Queue(maxsize=args.queue_capacity)
        self.results_to_host = queue.Queue(maxsize=args.queue_capacity)
        self.acks_to_ros = queue.Queue(maxsize=args.queue_capacity)
        self.stop_event = threading.Event()
        self.failed = None
        self.socket = None
        self.send_lock = threading.Lock()
        self.ack_lock = threading.Lock()
        self.awaiting_ack = None
        self.ack_event = None
        self.threads = []

    def _fail(self, error):
        if self.failed is None:
            self.failed = error
        self.stop_event.set()

    def _on_trace(self, _message):
        return None

    def _on_result(self, message):
        try:
            wire = bytes(message.data)
            request_id = request_id_from_result_wire(wire)
            self.results_to_host.put_nowait((request_id, wire))
        except Exception as error:
            self._fail(error)

    def _on_trigger(self, message):
        try:
            trigger = parse_hil_trigger(json.loads(message.data))
            self.triggers_to_worker.put_nowait(trigger.to_metadata())
        except Exception as error:
            self._fail(error)

    def _connect_host_proxy(self):
        deadline = time.monotonic() + self.args.connect_timeout_s
        last_error = None
        while time.monotonic() < deadline and not self.stop_event.is_set():
            try:
                connection = socket.create_connection(
                    ("127.0.0.1", self.args.forward_port), timeout=3.0
                )
                connection.settimeout(self.args.socket_timeout_s)
                with self.send_lock:
                    self.socket = connection
                return connection
            except OSError as error:
                last_error = error
                time.sleep(0.2)
        raise RuntimeError("CARLA reverse-SSH forward is unavailable: %s" % last_error)

    def _detach_socket(self, connection):
        """Close only the reader's connection while excluding concurrent sends."""
        if connection is None:
            return
        with self.send_lock:
            if self.socket is connection:
                self.socket = None
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def _socket_reader(self):
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = self._connect_host_proxy()
                while not self.stop_event.is_set():
                    line = _read_line(connection, MAX_TRIGGER_BYTES)
                    if line.startswith(b"TRIGGER "):
                        trigger = parse_host_message(line)
                        self.triggers_to_publish.put(trigger, timeout=1.0)
                    elif line.startswith(b"ACK "):
                        with self.ack_lock:
                            if self.awaiting_ack is None:
                                raise ValueError("unsolicited reverse-SSH ACK")
                            request_id, event = self.awaiting_ack
                            validate_ack_line(line, request_id)
                            self.acks_to_ros.put(request_id, timeout=1.0)
                            event.set()
                    else:
                        raise ValueError("reverse-SSH host message type is unsupported")
            except Exception as error:
                if self.stop_event.is_set():
                    return
                if not is_reconnectable_disconnect(error):
                    self._fail(error)
                    return
            finally:
                self._detach_socket(connection)
            self.stop_event.wait(0.25)

    def _worker_dispatcher(self):
        try:
            while not self.stop_event.is_set():
                try:
                    trigger = self.triggers_to_worker.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    with socket.create_connection(
                        (self.args.worker_host, self.args.worker_port),
                        timeout=self.args.worker_timeout_s,
                    ) as connection:
                        connection.settimeout(self.args.worker_timeout_s)
                        connection.sendall(
                            (json.dumps(trigger, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                        )
                        reply = _read_line(connection, 2048).decode("utf-8").strip()
                        if not reply.startswith("DONE "):
                            raise RuntimeError("Orin local worker rejected trigger: %s" % reply)
                finally:
                    self.triggers_to_worker.task_done()
        except Exception as error:
            if not self.stop_event.is_set():
                self._fail(error)

    def _host_result_writer(self):
        try:
            while not self.stop_event.is_set():
                try:
                    request_id, wire = self.results_to_host.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if len(wire) > self.args.max_result_mib * 1024 * 1024:
                        raise ValueError("ROS result exceeds configured byte limit")
                    event = threading.Event()
                    with self.ack_lock:
                        if self.awaiting_ack is not None:
                            raise RuntimeError("only one forwarded ROS result may be unacknowledged")
                        self.awaiting_ack = (request_id, event)
                    framed = frame_result_for_host(wire, self.args.max_result_mib * 1024 * 1024)
                    with self.send_lock:
                        if self.stop_event.is_set():
                            return
                        if self.socket is None:
                            raise EOFError("CARLA episode disconnected before result forwarding")
                        self.socket.sendall(framed)
                    if not event.wait(self.args.ack_timeout_s):
                        raise TimeoutError("CARLA did not ACK ROS result %s" % request_id)
                    with self.ack_lock:
                        self.awaiting_ack = None
                finally:
                    self.results_to_host.task_done()
        except Exception as error:
            if not self.stop_event.is_set():
                self._fail(error)

    def start(self):
        targets = (
            (self._socket_reader, "coptv2x-hil-proxy-reader"),
            (self._worker_dispatcher, "coptv2x-hil-worker-dispatch"),
            (self._host_result_writer, "coptv2x-hil-result-writer"),
        )
        for target, name in targets:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def spin(self):
        while self.rclpy.ok() and not self.stop_event.is_set():
            self.rclpy.spin_once(self.node, timeout_sec=0.02)
            for channel, publisher, kind in (
                (self.triggers_to_publish, self.trigger_publisher, "trigger"),
                (self.acks_to_ros, self.ack_publisher, "ack"),
            ):
                try:
                    while True:
                        value = channel.get_nowait()
                        message = self.String()
                        message.data = (
                            json.dumps(value, sort_keys=True, separators=(",", ":"))
                            if kind == "trigger" else value
                        )
                        publisher.publish(message)
                        channel.task_done()
                except queue.Empty:
                    pass
        if self.failed is not None:
            raise RuntimeError("CoptV2X ROS proxy failed: %s" % self.failed)

    def close(self):
        self.stop_event.set()
        with self.ack_lock:
            if self.awaiting_ack is not None:
                self.awaiting_ack[1].set()
        self._detach_socket(self.socket)
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.node.destroy_node()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-port", type=int, default=56611)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=56612)
    parser.add_argument("--trigger-topic", default="/coptv2x/hil/trigger")
    parser.add_argument("--result-topic", default="/ditl/rsu/perception_result")
    parser.add_argument("--trace-topic", default="/ditl/rsu/perception_trace")
    parser.add_argument("--ack-topic", default="/ditl/rsu/perception_result_ack")
    parser.add_argument("--queue-capacity", type=int, default=32)
    parser.add_argument("--max-result-mib", type=int, default=64)
    parser.add_argument("--connect-timeout-s", type=float, default=300.0)
    parser.add_argument("--socket-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--ack-timeout-s", type=float, default=120.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker_host not in ("127.0.0.1", "localhost"):
        raise SystemExit("Orin local worker must use loopback")
    if not 1 <= args.forward_port <= 65535 or not 1 <= args.worker_port <= 65535:
        raise SystemExit("forward and worker ports must be in 1..65535")
    if min(args.queue_capacity, args.max_result_mib) <= 0:
        raise SystemExit("queue capacity and result limit must be positive")
    rclpy, Node, String, UInt8MultiArray, qos = _ros_imports()
    rclpy.init(args=None)
    proxy = CoptV2xHilRosProxy(args, rclpy, Node, String, UInt8MultiArray, qos)
    try:
        proxy.start()
        print(json.dumps({"event": "coptv2x_hil_ros_proxy_ready",
                          "forward_port": args.forward_port,
                          "worker_port": args.worker_port}, sort_keys=True), flush=True)
        proxy.spin()
    finally:
        proxy.close()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
