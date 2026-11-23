from __future__ import print_function

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import sys
from threading import Condition, Lock, Thread
import time
from types import MappingProxyType

import numpy as np

_CLOSEDLOOP_DIR = str(Path(__file__).resolve().parent)
if _CLOSEDLOOP_DIR not in sys.path:
    sys.path.insert(0, _CLOSEDLOOP_DIR)

from coptv2x_hil_result_buffer import (
    AdmissionDecision,
    CoptV2xHilResultBuffer,
    HilResultEnvelope,
    _immutable_bytes_owner,
)
from coptv2x_hil_proxy_transport import ReverseSshRosProxyTransport


_STOP = object()
_OUTPUT_NAMES = ("res0", "res1", "res2")
_RSU_PAYLOAD_SEMANTICS = "codriving_rsu_multiscale_features.v1"


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return value


def _remote_rsu_only(outputs, payload_semantics=None):
    if outputs is None:
        return None
    if not isinstance(outputs, dict) and not hasattr(outputs, "keys"):
        raise TypeError("remote multiscale outputs must be a mapping")
    if set(outputs) != set(_OUTPUT_NAMES):
        raise ValueError("remote multiscale outputs require res0/res1/res2")
    from coptv2x_hil_backend import COPTV2X_PAYLOAD_SEMANTICS
    if payload_semantics is None:
        payload_semantics = COPTV2X_PAYLOAD_SEMANTICS
    if payload_semantics not in (COPTV2X_PAYLOAD_SEMANTICS, _RSU_PAYLOAD_SEMANTICS):
        raise ValueError("unsupported remote payload_semantics")
    rsu_only = payload_semantics == _RSU_PAYLOAD_SEMANTICS
    remote = []
    for name in _OUTPUT_NAMES:
        value = np.asarray(outputs[name])
        if value.ndim != 4 or value.shape[0] != (1 if rsu_only else 2):
            raise ValueError("%s batch must match explicit payload_semantics" % name)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("%s must be finite float32" % name)
        copied = (value if rsu_only and _immutable_bytes_owner(value) is not None
                  else np.array(value if rsu_only else value[1:2], copy=True, order="C"))
        remote.append(copied)
    return tuple(remote)


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def run_hil_prefusion_forward(perception_model, cav_content, remote_outputs,
                              payload_semantics=None):
    """Fuse current local ego with only the explicitly identified remote RSU."""

    remote_rsu_multiscale = _remote_rsu_only(remote_outputs, payload_semantics)
    return perception_model(
        cav_content,
        use_hil=True,
        hil_remote_multiscale=remote_rsu_multiscale,
    )


class JsonlAuditWriter(object):
    """Small synchronized writer for one immutable audit object per fusion frame."""

    def __init__(self, path):
        self.path = Path(path)
        self._stream = None
        self._lock = Lock()
        self._closed = False

    def append(self, record):
        encoded = json.dumps(_plain(record), sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("HIL audit writer is closed")
            if self._stream is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open("a", encoding="utf-8")
            self._stream.write(encoded + "\n")
            self._stream.flush()

    def close(self):
        with self._lock:
            if not self._closed and self._stream is not None:
                self._stream.close()
            self._closed = True


class AsyncHilClient(object):
    """Queue-isolated trigger sender and result validator for the CARLA thread."""

    def __init__(
        self,
        transport,
        result_buffer,
        outbound_capacity=8,
        inbound_capacity=8,
        clock_ns=time.monotonic_ns,
        audit_writer=None,
        result_timeout_s=120.0,
    ):
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.transport = transport
        self.result_buffer = result_buffer
        self.clock_ns = clock_ns
        self.audit_writer = audit_writer
        if (type(result_timeout_s) not in (int, float) or
                not math.isfinite(result_timeout_s) or result_timeout_s <= 0):
            raise ValueError("result_timeout_s must be positive and finite")
        self.result_timeout_s = float(result_timeout_s)
        self._serial_timing = result_buffer.timing_mode == "serial_fixed_comm"
        self._outbound = queue.Queue(
            maxsize=_positive_integer(outbound_capacity, "outbound_capacity")
        )
        self._inbound = queue.Queue(
            maxsize=_positive_integer(inbound_capacity, "inbound_capacity")
        )
        self._state_lock = Lock()
        self._idle = Condition(self._state_lock)
        self._running = False
        self._pending_out = 0
        self._pending_in = 0
        self._sequence_errors = {}
        self._receiver_error = None
        self._stats = {
            "published_triggers": 0,
            "publish_failures": 0,
            "accepted_results": 0,
            "rejected_results": 0,
            "inbound_overflow": 0,
            "outbound_overflow": 0,
        }
        self._sender = None
        self._receiver = None

    @property
    def is_running(self):
        with self._state_lock:
            return self._running

    @property
    def stats(self):
        with self._state_lock:
            return MappingProxyType(dict(self._stats))

    def start(self):
        with self._state_lock:
            if self._running:
                return
            self._running = True
        try:
            self.transport.start(self._receive_from_transport)
            self._sender = Thread(
                target=self._sender_loop,
                name="coptv2x-hil-trigger-sender",
                daemon=True,
            )
            self._receiver = Thread(
                target=self._receiver_loop,
                name="coptv2x-hil-result-validator",
                daemon=True,
            )
            self._sender.start()
            self._receiver.start()
        except Exception:
            with self._state_lock:
                self._running = False
            try:
                self.transport.close()
            finally:
                if self.audit_writer is not None:
                    self.audit_writer.close()
            raise

    def _raise_transport_fatal(self):
        """Expose an explicit asynchronous transport failure on the CARLA thread."""
        if not self.is_running:
            return
        error = getattr(self.transport, "last_error", None)
        if error is not None:
            raise RuntimeError("CoptV2X HIL transport fatal: %s: %s" %
                               (type(error).__name__, error)) from error

    def publish_trigger(self, trigger):
        self._raise_transport_fatal()
        with self._state_lock:
            if not self._running:
                self.result_buffer.record_rejection(
                    "client_not_running", trigger.sequence_no
                )
                return AdmissionDecision(False, "client_not_running", trigger.sequence_no)
        published_ns = 0 if self._serial_timing else int(self.clock_ns())
        try:
            self.result_buffer.register_trigger(trigger, published_ns)
        except Exception as error:
            self.result_buffer.record_rejection(
                "trigger_rejected:%s" % type(error).__name__, trigger.sequence_no
            )
            return AdmissionDecision(
                False, "trigger_rejected:%s" % type(error).__name__, trigger.sequence_no
            )
        payload = copy.deepcopy(trigger.to_metadata())
        with self._state_lock:
            self._pending_out += 1
            try:
                self._outbound.put_nowait((trigger, payload))
            except queue.Full:
                self._pending_out -= 1
                self._stats["outbound_overflow"] += 1
                self.result_buffer.cancel_trigger(trigger)
                self.result_buffer.record_rejection(
                    "outbound_overflow", trigger.sequence_no
                )
                return AdmissionDecision(False, "outbound_overflow", trigger.sequence_no)
        return AdmissionDecision(True, "queued", trigger.sequence_no)

    def _receive_from_transport(self, envelope):
        with self._state_lock:
            if not self._running:
                return
        copied = copy.copy(envelope)
        with self._state_lock:
            self._pending_in += 1
            try:
                arrival_ns = 0 if self._serial_timing else int(self.clock_ns())
                self._inbound.put_nowait((copied, arrival_ns))
            except queue.Full:
                self._pending_in -= 1
                self._stats["inbound_overflow"] += 1
                self._stats["rejected_results"] += 1
                self._receiver_error = RuntimeError("HIL inbound_overflow")
                self._idle.notify_all()
                return

    def _sender_loop(self):
        while True:
            entry = self._outbound.get()
            if entry is _STOP:
                self._outbound.task_done()
                return
            trigger, payload = entry
            try:
                self.transport.publish_trigger(payload)
                with self._state_lock:
                    self._stats["published_triggers"] += 1
            except Exception as error:
                self.result_buffer.record_rejection(
                    "trigger_publish_failed", trigger.sequence_no
                )
                with self._state_lock:
                    self._stats["publish_failures"] += 1
                    self._sequence_errors[trigger.sequence_no] = error
            finally:
                with self._state_lock:
                    self._pending_out -= 1
                    self._idle.notify_all()
                self._outbound.task_done()

    def _receiver_loop(self):
        while True:
            entry = self._inbound.get()
            if entry is _STOP:
                self._inbound.task_done()
                return
            envelope, arrival_ns = entry
            try:
                decision = self.result_buffer.accept(envelope, arrival_ns)
                with self._state_lock:
                    key = "accepted_results" if decision.accepted else "rejected_results"
                    self._stats[key] += 1
                    if not decision.accepted:
                        error = RuntimeError("HIL result rejected: %s" % decision.reason)
                        if decision.sequence_no is None:
                            self._receiver_error = error
                        else:
                            self._sequence_errors[decision.sequence_no] = error
            except Exception as error:
                with self._state_lock:
                    self._receiver_error = error
            finally:
                with self._state_lock:
                    self._pending_in -= 1
                    self._idle.notify_all()
                self._inbound.task_done()

    def select_for_fusion(self, current_frame, current_sim_time_ns, trigger):
        self._raise_transport_fatal()
        selection = self.result_buffer.select_for_fusion(
            current_frame,
            current_sim_time_ns,
            trigger,
            0 if self._serial_timing else int(self.clock_ns()),
        )
        if self.audit_writer is not None:
            self.audit_writer.append(selection.audit)
        return selection

    def wait_for_result(self, trigger, timeout_s=None):
        """Wait for admission of an exact sequence, independent of queue idleness.

        This method runs inside the synchronous agent call, so the scenario
        manager cannot tick CARLA until it returns. Wall waiting is not latency.
        """
        timeout_s = self.result_timeout_s if timeout_s is None else timeout_s
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
                or timeout_s <= 0):
            raise ValueError("timeout_s must be positive and finite")
        deadline = time.monotonic() + timeout_s
        with self._idle:
            while True:
                fatal = getattr(self.transport, "last_error", None)
                error = (fatal or self._sequence_errors.get(trigger.sequence_no)
                         or self._receiver_error)
                if error is not None:
                    raise RuntimeError("CoptV2X HIL exact-result wait failed: %s" % error) from error
                if not self._running:
                    raise RuntimeError("CoptV2X HIL client stopped during result wait")
                result = self.result_buffer.get_exact_result(trigger)
                if result is not None:
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("HIL result timeout for sequence %d" % trigger.sequence_no)
                # Fatal transport changes have no callback; check them finitely.
                self._idle.wait(min(remaining, .1))

    def wait_until_idle(self, timeout_s):
        if not isinstance(timeout_s, (int, float)) or timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + float(timeout_s)
        with self._idle:
            return self._idle.wait_for(
                lambda: self._pending_out == 0 and self._pending_in == 0,
                timeout=max(0.0, deadline - time.monotonic()),
            )

    def shutdown(self, join_timeout_s=2.0):
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            self._idle.notify_all()
        self.transport.close()
        self._outbound.put(_STOP, timeout=join_timeout_s)
        self._inbound.put(_STOP, timeout=join_timeout_s)
        self._sender.join(join_timeout_s)
        self._receiver.join(join_timeout_s)
        if self._sender.is_alive() or self._receiver.is_alive():
            raise RuntimeError("HIL client worker did not shut down")
        if self.audit_writer is not None:
            self.audit_writer.close()


class Ros2HilTransport(object):
    """Lazy ROS 2 String-trigger/UInt8MultiArray-result transport."""

    def __init__(
        self,
        trigger_topic="/coptv2x/hil/trigger",
        result_topic="/ditl/rsu/perception_result",
        ack_topic="/ditl/rsu/perception_result_ack",
        max_payload_mib=64,
        readonly_results=False,
    ):
        self.trigger_topic = str(trigger_topic)
        self.result_topic = str(result_topic)
        self.ack_topic = str(ack_topic)
        self.max_payload_mib = _positive_integer(max_payload_mib, "max_payload_mib")
        if type(readonly_results) is not bool:
            raise ValueError("readonly_results must be a boolean")
        self.readonly_results = readonly_results
        self._handler = None
        self._thread = None
        self._node = None
        self._executor = None
        self._owns_rclpy = False

    def start(self, handler):
        if not callable(handler):
            raise TypeError("result handler must be callable")
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from std_msgs.msg import String, UInt8MultiArray
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 Python is required only when CoptV2X HIL is enabled"
            ) from error
        from coptv2x_hil_backend import HilFeatureResult
        from coptv2x_hil_trigger import parse_hil_trigger
        from ditl_edge_protocol import ProtocolLimits, decode_wire_bytes

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        limits = ProtocolLimits(
            max_payload_bytes=self.max_payload_mib * 1024 * 1024,
            max_raw_payload_bytes=self.max_payload_mib * 1024 * 1024,
        )
        transport = self

        class HilNode(Node):
            def __init__(self):
                super(HilNode, self).__init__("coptv2x_hil_carla_client")
                self.trigger_publisher = self.create_publisher(
                    String, transport.trigger_topic, qos
                )
                self.ack_publisher = self.create_publisher(
                    String, transport.ack_topic, qos
                )
                self.result_subscription = self.create_subscription(
                    UInt8MultiArray,
                    transport.result_topic,
                    self.on_result,
                    qos,
                )

            def on_result(self, message):
                try:
                    wire = bytes(bytearray(message.data))
                    header, outputs, _, _ = decode_wire_bytes(
                        wire, limits, return_stages=True,
                        readonly=transport.readonly_results,
                    )
                    if header.get("kind") != "result" or header.get("status") != "ok":
                        raise ValueError("HIL ROS result must be successful")
                    trigger = parse_hil_trigger(header.get("trigger_identity"))
                    for name, value in trigger.to_metadata().items():
                        if header.get(name) != value:
                            raise ValueError(
                                "HIL ROS result identity conflict: %s" % name
                            )
                    batch_order = header.get("batch_order")
                    if not isinstance(batch_order, list):
                        raise ValueError("HIL ROS result batch_order is required")
                    result = HilFeatureResult(
                        outputs=outputs,
                        artifact_identity=header.get("artifact_identity", {}),
                        trigger_identity=trigger.to_metadata(),
                        payload_semantics=header.get("payload_semantics"),
                        payload_sha256=header.get("payload_sha256"),
                        remote_input_tensor_bytes=header.get(
                            "remote_input_tensor_bytes"
                        ),
                        timestamps_ns=header.get("edge_timestamps_ns", {}),
                        durations_ns=header.get("durations_ns", {}),
                        model=header.get("model", {}),
                    )
                    envelope = HilResultEnvelope(result, tuple(batch_order))
                except Exception:
                    # Treat malformed external results as a rejection in the
                    # async validator thread; never unwind ROS executor.spin().
                    transport._handler(None)
                    return
                transport._handler(envelope)
                ack = String()
                ack.data = str(header["request_id"])
                self.ack_publisher.publish(ack)

        self._handler = handler
        self._node = HilNode()
        self._String = String
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = Thread(
            target=self._executor.spin,
            name="coptv2x-hil-ros2-spin",
            daemon=True,
        )
        self._thread.start()

    def publish_trigger(self, payload):
        message = self._String()
        message.data = json.dumps(
            copy.deepcopy(payload), sort_keys=True, separators=(",", ":")
        )
        self._node.trigger_publisher.publish(message)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
        if self._thread is not None:
            self._thread.join(1.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()


def create_ros2_hil_client(config):
    """Validate an explicit opt-in config and construct an unstarted client."""

    if type(config) is not dict:
        raise TypeError("coptv2x_hil config must be an object")
    from coptv2x_hil_artifacts import HIL_ARM_IDS

    required = ("run_id", "route_id", "scenario_id", "arm_id", "audit_jsonl")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("coptv2x_hil missing fields: %s" % missing)
    if config["arm_id"] not in HIL_ARM_IDS:
        raise ValueError("coptv2x_hil arm_id is not frozen")
    capacity = int(config.get("buffer_capacity", 8))
    writer = JsonlAuditWriter(config["audit_jsonl"])
    transport_kind = config.get("transport", "ros2")
    readonly_results = config.get("timing_mode", "async") == "serial_fixed_comm"
    if transport_kind == "ros2":
        transport = Ros2HilTransport(
            trigger_topic=config.get("trigger_topic", "/coptv2x/hil/trigger"),
            result_topic=config.get("result_topic", "/ditl/rsu/perception_result"),
            ack_topic=config.get("ack_topic", "/ditl/rsu/perception_result_ack"),
            max_payload_mib=int(config.get("max_payload_mib", 64)),
            readonly_results=readonly_results,
        )
    elif transport_kind == "reverse_ssh_ros_proxy":
        transport = ReverseSshRosProxyTransport(
            listen_host=config.get("proxy_listen_host", "127.0.0.1"),
            listen_port=int(config.get("proxy_listen_port", 56610)),
            connect_timeout_s=float(config.get("proxy_connect_timeout_s", 60.0)),
            max_wire_bytes=int(config.get("max_payload_mib", 64)) * 1024 * 1024,
            readonly_results=readonly_results,
        )
    else:
        raise ValueError("unsupported CoptV2X HIL transport %r" % transport_kind)
    result_buffer = CoptV2xHilResultBuffer(
        config["run_id"],
        config["route_id"],
        config["scenario_id"],
        config["arm_id"],
        capacity=capacity,
        trigger_capacity=int(
            config.get("trigger_capacity", max(capacity * 4, 4096))
        ),
        timing_mode=config.get("timing_mode", "async"),
        fixed_comm_delay_ms=config.get("fixed_comm_delay_ms", 100),
    )
    return AsyncHilClient(
        transport,
        result_buffer,
        outbound_capacity=int(config.get("outbound_capacity", capacity)),
        inbound_capacity=int(config.get("inbound_capacity", capacity)),
        audit_writer=writer,
        result_timeout_s=config.get("result_timeout_s", 120.0),
    )


__all__ = (
    "AsyncHilClient",
    "HilResultEnvelope",
    "JsonlAuditWriter",
    "Ros2HilTransport",
    "ReverseSshRosProxyTransport",
    "create_ros2_hil_client",
    "run_hil_prefusion_forward",
)
