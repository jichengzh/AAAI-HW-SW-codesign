#!/usr/bin/env python3
from __future__ import print_function

import argparse
import array
import json
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from ditl_edge_protocol import (
    FRAME_PREFIX,
    MAGIC,
    PROTOCOL_VERSION,
    REQUEST_ID_RE,
    ProtocolError,
    ProtocolLimits,
    build_wire_bytes,
    decode_wire_bytes,
    inspect_envelope_header,
    inspect_wire_bytes_header,
    recv_wire_frame,
)
from ditl_edge_runtime import (
    boot_id,
    build_round_trip_trace,
    monotonic_raw_ns,
    validate_inference_response,
)
from ditl_trace_schema import validate_trace_record, write_latency_csv, write_trace_jsonl


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bridge DitL RSU inference frames through ROS 2 UInt8MultiArray topics."
    )
    subparsers = parser.add_subparsers(dest="mode")

    relay = subparsers.add_parser("edge-relay", help="ROS 2 to native edge service relay")
    relay.add_argument("--edge-host", default="127.0.0.1")
    relay.add_argument("--edge-port", type=int, default=55020)
    relay.add_argument("--request-topic", default="/ditl/rsu/infer_request")
    relay.add_argument("--response-topic", default="/ditl/rsu/infer_response")
    relay.add_argument("--timeout-s", type=float, default=120.0)
    relay.add_argument("--max-requests", type=int, default=0)
    relay.add_argument("--shutdown-grace-s", type=_nonnegative_float, default=1.0)
    relay.add_argument("--max-payload-mib", type=int, default=512)

    client = subparsers.add_parser("probe-client", help="Publish one replay request")
    client.add_argument("--input-npz", required=True)
    client.add_argument("--output-npz", required=True)
    client.add_argument("--trace-jsonl", required=True)
    client.add_argument("--latency-csv", required=True)
    client.add_argument("--request-topic", default="/ditl/rsu/infer_request")
    client.add_argument("--response-topic", default="/ditl/rsu/infer_response")
    client.add_argument("--request-id", default=None)
    client.add_argument("--run-id", default="ditl-ros2-smoke")
    client.add_argument("--route-id", default="tensor-replay")
    client.add_argument("--scenario-id", default="edge-rsu")
    client.add_argument("--source-id", default="rsu-orin-01")
    client.add_argument("--target-id", default="ego-server-01")
    client.add_argument("--frame-id", type=int, default=0)
    client.add_argument("--sequence-id", type=int, default=0)
    client.add_argument("--topic", default="/ditl/rsu/features")
    client.add_argument("--message-type", default="intermediate_feature")
    client.add_argument("--compression-ratio", type=float, default=1.0)
    client.add_argument("--wire-compression", choices=("none", "zlib"), default="zlib")
    client.add_argument("--timeout-s", type=float, default=120.0)
    client.add_argument("--discovery-timeout-s", type=float, default=30.0)
    client.add_argument("--max-payload-mib", type=int, default=512)
    client.add_argument("--observer-host-id", default=socket.gethostname())

    publisher = subparsers.add_parser(
        "result-publisher", help="Publish Orin-local result frames over ROS 2"
    )
    publisher.add_argument("--bind", default="127.0.0.1")
    publisher.add_argument("--result-port", type=int, default=55021)
    publisher.add_argument("--result-topic", default="/ditl/rsu/perception_result")
    publisher.add_argument("--trace-topic", default="/ditl/rsu/perception_trace")
    publisher.add_argument("--max-results", type=int, required=True)
    publisher.add_argument("--discovery-timeout-s", type=float, default=30.0)
    publisher.add_argument("--socket-timeout-s", type=float, default=120.0)
    publisher.add_argument("--shutdown-grace-s", type=_nonnegative_float, default=1.0)
    publisher.add_argument("--max-payload-mib", type=int, default=512)
    publisher.add_argument("--ready-file")

    sink = subparsers.add_parser(
        "result-sink", help="Receive Orin-local ROS 2 results and write v2 traces"
    )
    sink.add_argument("--result-topic", default="/ditl/rsu/perception_result")
    sink.add_argument("--trace-topic", default="/ditl/rsu/perception_trace")
    sink.add_argument("--trace-jsonl", required=True)
    sink.add_argument("--output-npz", required=True)
    sink.add_argument("--max-results", type=int, required=True)
    sink.add_argument("--timeout-s", type=float, default=300.0)
    sink.add_argument("--max-payload-mib", type=int, default=512)
    sink.add_argument("--max-pending-pairs", type=int, default=16)
    sink.add_argument("--max-pending-result-mib", type=int, default=64)
    sink.add_argument("--max-sideband-bytes", type=int, default=4096)
    sink.add_argument("--receiver-host-id", default="orin-agx-01")
    sink.add_argument("--receiver-boot-id", default=None)
    sink.add_argument("--ready-file")
    return parser


def _ros_imports():
    try:
        import rclpy
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
            "ROS 2 Python is required; source /opt/ros/<distro>/setup.bash"
        ) from error
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    return rclpy, Node, UInt8MultiArray, String, qos


def _wire_from_socket(connection, limits):
    header_bytes, payload, _ = recv_wire_frame(connection, limits)
    return FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload)) + header_bytes + payload


def _wire_to_ros_uint8_array(wire):
    """Convert a wire frame to Foxy's efficient uint8[] representation."""
    return array.array("B", wire)


def _bounded_pending_put(
    pending, request_id, value, value_bytes, max_entries, max_bytes
):
    if isinstance(value_bytes, bool) or not isinstance(value_bytes, int) or value_bytes < 0:
        raise ProtocolError("pending value size must be a non-negative integer")
    current = pending.get(request_id)
    current_bytes = 0 if current is None else current["bytes"]
    if current is None and len(pending) >= max_entries:
        raise ProtocolError("pending entry limit exceeded")
    total_bytes = sum(item["bytes"] for item in pending.values())
    if total_bytes - current_bytes + value_bytes > max_bytes:
        raise ProtocolError("pending byte limit exceeded")
    updated = dict(pending)
    updated[request_id] = {"bytes": value_bytes, "value": value}
    return updated


def _parse_result_sideband(value, max_bytes):
    if not isinstance(value, str):
        raise ProtocolError("trace sideband must be text")
    if len(value) > max_bytes or len(value.encode("utf-8")) > max_bytes:
        raise ProtocolError("trace sideband size exceeds limit")
    sideband = json.loads(value)
    if not isinstance(sideband, dict):
        raise ProtocolError("trace sideband must be an object")
    request_id = sideband.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.match(request_id):
        raise ProtocolError("trace sideband request_id is invalid")
    return sideband


def _ros_uint8_array_to_wire(values):
    """Convert Foxy's uint8[] representation back to a wire frame."""
    return bytes(values)


def _build_relay_error_wire(request_id, error, limits):
    return build_wire_bytes(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "error",
            "request_id": request_id,
            "error_code": "relay_error",
            "message": type(error).__name__[:512],
        },
        {},
        compression="none",
        limits=limits,
    )


def _error_wire_for_request_wire(request_wire, code, message, limits):
    request_id = "invalid-request"
    try:
        request, _, _ = decode_wire_bytes(request_wire, limits)
        request_id = request["request_id"]
    except ProtocolError:
        pass
    return build_wire_bytes(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "error",
            "request_id": request_id,
            "error_code": str(code)[:128],
            "message": str(message)[:512],
        },
        {},
        compression="none",
        limits=limits,
    )


def _ros_endpoints_ready(node, request_publisher, response_topic):
    return (
        request_publisher.get_subscription_count() > 0
        and node.count_publishers(response_topic) > 0
    )


def _write_json_atomic(path, payload):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def _finalize_result_wire(
    wire,
    sideband,
    ros_receive_ns,
    receiver_host_id,
    receiver_boot_id,
    limits,
):
    header, arrays, wire_bytes, decode_stages = decode_wire_bytes(
        wire, limits, return_stages=True
    )
    if header.get("kind") != "result" or header.get("status") != "ok":
        raise ProtocolError("formal edge frame must be a successful result")
    if sideband.get("request_id") != header.get("request_id"):
        raise ProtocolError("result and trace sideband request_id mismatch")
    edge_timestamps = {
        str(name): int(value)
        for name, value in header.get("edge_timestamps_ns", {}).items()
    }
    required_sender = (
        "rsu_data_available",
        "source_read_start",
        "source_read_end",
        "preprocess_start",
        "preprocess_end",
        "h2d_start",
        "h2d_end",
        "inference_start",
        "inference_end",
        "d2h_start",
        "d2h_end",
        "postprocess_start",
        "postprocess_end",
        "pack_start",
        "pack_end",
        "hash_start",
        "hash_end",
        "compress_start",
        "compress_end",
        "protocol_serialize_start",
        "protocol_serialize_end",
    )
    for name in required_sender:
        if name not in edge_timestamps:
            raise ProtocolError("result is missing sender timestamp %s" % name)
    for name in (
        "local_handoff_end",
        "ros_message_start",
        "ros_message_end",
        "ros_publish_start",
        "ros_publish_end",
    ):
        if isinstance(sideband.get(name), bool) or not isinstance(sideband.get(name), int):
            raise ProtocolError("trace sideband.%s must be an integer" % name)

    receiver_timestamps = decode_stages["timestamps_ns"]
    fusion_ready = monotonic_raw_ns()
    timestamps = dict(edge_timestamps)
    timestamps.update(
        {
            "ros_message_start": sideband["ros_message_start"],
            "ros_message_end": sideband["ros_message_end"],
            "ros_publish_start": sideband["ros_publish_start"],
            "ros_publish_end": sideband["ros_publish_end"],
            "ros_receive": int(ros_receive_ns),
            "frame_parse_start": receiver_timestamps["frame_parse_start"],
            "frame_parse_end": receiver_timestamps["frame_parse_end"],
            "decompress_start": receiver_timestamps["decompress_start"],
            "decompress_end": receiver_timestamps["decompress_end"],
            "verify_start": receiver_timestamps["verify_start"],
            "verify_end": receiver_timestamps["verify_end"],
            "reconstruct_start": receiver_timestamps["reconstruct_start"],
            "reconstruct_end": receiver_timestamps["reconstruct_end"],
            "fusion_ready": fusion_ready,
        }
    )
    durations = {
        str(name): int(value)
        for name, value in header.get("durations_ns", {}).items()
    }
    handoff_start = edge_timestamps.get(
        "local_handoff_start", edge_timestamps["protocol_serialize_end"]
    )
    durations.update(
        {
            "local_handoff": max(0, sideband["local_handoff_end"] - handoff_start),
            "ros_message": max(
                0, sideband["ros_message_end"] - sideband["ros_message_start"]
            ),
            "ros_publish_call": max(
                0, sideband["ros_publish_end"] - sideband["ros_publish_start"]
            ),
            "frame_parse": int(decode_stages["frame_parse_ns"]),
            "decompress": int(decode_stages["decompress_ns"]),
            "verify": int(decode_stages["verify_ns"]),
            "reconstruct": int(decode_stages["reconstruct_ns"]),
        }
    )
    edge_clock = header.get("edge_clock")
    if not isinstance(edge_clock, dict):
        raise ProtocolError("result edge_clock is required")
    receiver_clock = {
        "host_id": str(receiver_host_id),
        "domain": "monotonic_raw",
        "boot_id": str(receiver_boot_id),
        "sync_state": "locked",
        "offset_ns": 0,
        "uncertainty_ns": 0,
    }
    comparable = (
        edge_clock.get("host_id") == receiver_clock["host_id"]
        and edge_clock.get("domain") == "monotonic_raw"
        and edge_clock.get("boot_id") == receiver_clock["boot_id"]
    )
    bundle = header["tensor_bundle"]
    descriptors = bundle["arrays"]
    first = descriptors[0] if descriptors else {"shape": [1], "dtype": "uint8"}
    raw_bytes = int(bundle["raw_payload_bytes"])
    wire_payload_bytes = int(bundle["wire_payload_bytes"])
    compression_ratio = (
        float(raw_bytes) / float(wire_payload_bytes)
        if wire_payload_bytes
        else 1.0
    )
    sequence_id = int(header["sequence_id"])
    record = {
        "schema_version": "ditl.trace.v2",
        "trace_id": header["request_id"],
        "run_id": header["run_id"],
        "route_id": header["route_id"],
        "scenario_id": header["scenario_id"],
        "measurement_scope": "rsu_to_fusion",
        "headline_eligible": comparable,
        "headline_ineligible_reason": None if comparable else "clock_not_comparable",
        "source": {
            "type": header["source_type"],
            "payload_semantics": header["payload_semantics"],
            "postprocess": header["postprocess"],
        },
        "message": {
            "source_id": header["source_id"],
            "target_id": header["target_id"],
            "frame_id": int(header["frame_id"]),
            "sequence_id": sequence_id,
            "topic": "/ditl/rsu/perception_result",
            "message_type": "dense_perception_heads",
            "payload_bytes": raw_bytes,
            "tensor_shape": list(first["shape"]),
            "tensor_dtype": first["dtype"],
            "compression_ratio": compression_ratio,
        },
        "clock": {"sender": dict(edge_clock), "receiver": receiver_clock},
        "clock_comparability": {
            "comparable": comparable,
            "reason": "same_host_monotonic" if comparable else "clock_not_comparable",
        },
        "status": {
            "outcome": "consumed",
            "duplicate": False,
            "out_of_order": False,
            "reason": None,
        },
        "timestamps_ns": timestamps,
        "network": {
            "transport": "ros2_uint8multiarray_one_way",
            "qos": "reliable",
            "sender_count": sequence_id + 1,
            "receiver_count": sequence_id + 1,
            "wire_bytes": int(wire_bytes),
            "remote_input_tensor_bytes": int(
                header.get("remote_input_tensor_bytes", -1)
            ),
        },
        "codec": {
            "name": bundle["compression"],
            "raw_payload_bytes": raw_bytes,
            "wire_payload_bytes": wire_payload_bytes,
        },
        "durations_ns": durations,
        "testbed_overhead_ns": {"control": 0, "startup": 0},
        "model": header.get("model", {}),
        "device": header.get("device", {}),
        "warmup": bool(header.get("warmup", False)),
    }
    if record["network"]["remote_input_tensor_bytes"] != 0:
        raise ProtocolError("formal result must declare zero remote input tensor bytes")
    return arrays, validate_trace_record(record)


def run_edge_relay(args):
    rclpy, Node, UInt8MultiArray, _, qos = _ros_imports()
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )

    class EdgeRelay(Node):
        def __init__(self):
            super(EdgeRelay, self).__init__("ditl_rsu_edge_relay")
            self.publisher = self.create_publisher(UInt8MultiArray, args.response_topic, qos)
            self.subscription = self.create_subscription(
                UInt8MultiArray, args.request_topic, self.on_request, qos
            )
            self.served = 0
            self.failed = 0

        def on_request(self, message):
            request_wire = _ros_uint8_array_to_wire(message.data)
            request_id = "invalid-request"
            callback_error = None
            try:
                request, _, _ = decode_wire_bytes(request_wire, limits)
                request_id = request["request_id"]
                with socket.create_connection(
                    (args.edge_host, args.edge_port), timeout=args.timeout_s
                ) as connection:
                    connection.settimeout(args.timeout_s)
                    connection.sendall(request_wire)
                    response_wire = _wire_from_socket(connection, limits)
                response, _, _ = decode_wire_bytes(response_wire, limits)
                if response["request_id"] != request_id:
                    raise RuntimeError("edge response request_id mismatch")
            except Exception as error:
                callback_error = error
                self.failed += 1
                response_wire = _build_relay_error_wire(request_id, error, limits)
            output = UInt8MultiArray()
            output.data = _wire_to_ros_uint8_array(response_wire)
            try:
                self.publisher.publish(output)
            except Exception as error:
                self.failed += 1
                self.get_logger().error(
                    json.dumps(
                        {
                            "event": "relay_publish_failed",
                            "request_id": request_id,
                            "error_type": type(error).__name__,
                            "failed": self.failed,
                        },
                        sort_keys=True,
                    )
                )
                return
            self.served += 1
            logger = self.get_logger().error if callback_error else self.get_logger().info
            logger(
                json.dumps(
                    {
                        "event": "relay_error" if callback_error else "relayed",
                        "request_id": request_id,
                        "error_type": (
                            type(callback_error).__name__ if callback_error else None
                        ),
                        "wire_bytes": len(response_wire),
                        "served": self.served,
                        "failed": self.failed,
                    },
                    sort_keys=True,
                )
            )
    rclpy.init(args=None)
    node = EdgeRelay()
    print(
        json.dumps(
            {
                "event": "ros2_edge_relay_ready",
                "request_topic": args.request_topic,
                "response_topic": args.response_topic,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        while rclpy.ok() and (args.max_requests == 0 or node.served < args.max_requests):
            rclpy.spin_once(node, timeout_sec=0.2)
        if args.max_requests > 0 and node.served >= args.max_requests:
            time.sleep(args.shutdown_grace_s)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _load_npz(path):
    with np.load(str(Path(path)), allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("input NPZ contains no arrays")
        return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}


def run_probe_client(args):
    rclpy, Node, UInt8MultiArray, _, qos = _ros_imports()
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )
    arrays = _load_npz(args.input_npz)
    request_id = args.request_id or ("ros-req-" + uuid.uuid4().hex)
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "request",
        "request_id": request_id,
        "operation": "infer",
        "run_id": args.run_id,
        "route_id": args.route_id,
        "scenario_id": args.scenario_id,
        "source_id": args.source_id,
        "target_id": args.target_id,
        "frame_id": args.frame_id,
        "sequence_id": args.sequence_id,
        "topic": args.topic,
        "message_type": args.message_type,
        "compression_ratio": args.compression_ratio,
    }
    request_wire = build_wire_bytes(
        request, arrays, compression=args.wire_compression, limits=limits
    )
    request_envelope, _, _ = decode_wire_bytes(request_wire, limits)

    class ProbeClient(Node):
        def __init__(self):
            super(ProbeClient, self).__init__("ditl_rsu_probe_client")
            self.publisher = self.create_publisher(UInt8MultiArray, args.request_topic, qos)
            self.subscription = self.create_subscription(
                UInt8MultiArray, args.response_topic, self.on_response, qos
            )
            self.response_wire = None
            self.response_ns = None

        def on_response(self, message):
            wire = _ros_uint8_array_to_wire(message.data)
            header, _, _ = decode_wire_bytes(wire, limits)
            if header["request_id"] == request_id:
                self.response_wire = wire
                self.response_ns = monotonic_raw_ns()

    rclpy.init(args=None)
    node = ProbeClient()
    try:
        discovery_deadline = time.monotonic() + args.discovery_timeout_s
        while time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if _ros_endpoints_ready(node, node.publisher, args.response_topic):
                break
        if not _ros_endpoints_ready(node, node.publisher, args.response_topic):
            raise RuntimeError("ROS 2 relay request/response endpoints were not discovered")
        message = UInt8MultiArray()
        message.data = _wire_to_ros_uint8_array(request_wire)
        send_start = monotonic_raw_ns()
        node.publisher.publish(message)
        send_end = monotonic_raw_ns()
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline and node.response_wire is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.response_wire is None:
            raise RuntimeError("timed out waiting for ROS 2 edge response")
        response, outputs, response_wire_bytes = decode_wire_bytes(
            node.response_wire, limits
        )
        validate_inference_response(request, response)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **outputs)
    measurement = {
        "request_send_start_ns": send_start,
        "request_send_end_ns": send_end,
        "response_receive_ns": node.response_ns,
        "round_trip_ns": node.response_ns - send_start,
        "request_wire_bytes": len(request_wire),
        "response_wire_bytes": response_wire_bytes,
        "request_tensor_bundle": request_envelope["tensor_bundle"],
    }
    trace = build_round_trip_trace(
        request, response, measurement, observer_host_id=args.observer_host_id
    )
    trace["network"]["transport"] = "ros2_uint8multiarray_round_trip_probe"
    trace["network"]["request_topic"] = args.request_topic
    trace["network"]["response_topic"] = args.response_topic
    write_trace_jsonl(args.trace_jsonl, [trace])
    write_latency_csv(args.latency_csv, [trace])
    print(
        json.dumps(
            {
                "request_id": request_id,
                "model_id": response["model"]["model_id"],
                "round_trip_ms": measurement["round_trip_ns"] / 1_000_000.0,
                "transport": trace["network"]["transport"],
                "output_names": sorted(outputs),
            },
            sort_keys=True,
        )
    )
    return 0


def run_result_publisher(args):
    rclpy, Node, UInt8MultiArray, String, qos = _ros_imports()
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )
    if args.bind not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("result publisher handoff must bind to localhost")

    class ResultPublisher(Node):
        def __init__(self):
            super(ResultPublisher, self).__init__("ditl_rsu_result_publisher")
            self.result_publisher = self.create_publisher(
                UInt8MultiArray, args.result_topic, qos
            )
            self.trace_publisher = self.create_publisher(String, args.trace_topic, qos)

    rclpy.init(args=None)
    node = ResultPublisher()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.result_port))
    listener.listen(1)
    listener.settimeout(args.socket_timeout_s)
    ready = {
        "event": "result_publisher_ready",
        "bind": args.bind,
        "result_port": listener.getsockname()[1],
        "result_topic": args.result_topic,
        "trace_topic": args.trace_topic,
    }
    _write_json_atomic(args.ready_file, ready)
    print(json.dumps(ready, sort_keys=True), flush=True)
    try:
        deadline = time.monotonic() + args.discovery_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (
                node.result_publisher.get_subscription_count() > 0
                and node.trace_publisher.get_subscription_count() > 0
            ):
                break
        if (
            node.result_publisher.get_subscription_count() <= 0
            or node.trace_publisher.get_subscription_count() <= 0
        ):
            raise RuntimeError("result and trace subscribers were not discovered")
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(args.socket_timeout_s)
            for _ in range(args.max_results):
                header_bytes, payload, _ = recv_wire_frame(connection, limits)
                local_handoff_end = monotonic_raw_ns()
                header = inspect_envelope_header(header_bytes, payload, limits)
                if header.get("kind") != "result" or header.get("status") != "ok":
                    raise ProtocolError("local handoff accepts successful results only")
                wire = (
                    FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload))
                    + header_bytes
                    + payload
                )
                ros_message_start = monotonic_raw_ns()
                message = UInt8MultiArray()
                message.data = _wire_to_ros_uint8_array(wire)
                ros_message_end = monotonic_raw_ns()
                ros_publish_start = monotonic_raw_ns()
                node.result_publisher.publish(message)
                ros_publish_end = monotonic_raw_ns()
                trace_message = String()
                trace_message.data = json.dumps(
                    {
                        "request_id": header["request_id"],
                        "local_handoff_end": local_handoff_end,
                        "ros_message_start": ros_message_start,
                        "ros_message_end": ros_message_end,
                        "ros_publish_start": ros_publish_start,
                        "ros_publish_end": ros_publish_end,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                node.trace_publisher.publish(trace_message)
                rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(args.shutdown_grace_s)
    finally:
        listener.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


def run_result_sink(args):
    rclpy, Node, UInt8MultiArray, String, qos = _ros_imports()
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )
    max_wire_bytes = (
        FRAME_PREFIX.size + limits.max_header_bytes + limits.max_payload_bytes
    )
    max_pending_result_bytes = args.max_pending_result_mib * 1024 * 1024
    receiver_boot = args.receiver_boot_id or boot_id()

    class ResultSink(Node):
        def __init__(self):
            super(ResultSink, self).__init__("ditl_rsu_result_sink")
            self.result_subscription = self.create_subscription(
                UInt8MultiArray, args.result_topic, self.on_result, qos
            )
            self.trace_subscription = self.create_subscription(
                String, args.trace_topic, self.on_trace, qos
            )
            self.pending_results = {}
            self.pending_sidebands = {}
            self.records = []
            self.last_outputs = None
            self.failed = 0

        def on_result(self, message):
            receive_ns = monotonic_raw_ns()
            try:
                if len(message.data) > max_wire_bytes:
                    raise ProtocolError("ROS result frame exceeds wire limit")
                wire = _ros_uint8_array_to_wire(message.data)
                header, _ = inspect_wire_bytes_header(wire, limits)
                request_id = header["request_id"]
                self.pending_results = _bounded_pending_put(
                    self.pending_results,
                    request_id,
                    (wire, receive_ns),
                    len(wire),
                    args.max_pending_pairs,
                    max_pending_result_bytes,
                )
                self._complete(request_id)
            except Exception as error:
                self.failed += 1
                self.get_logger().error("result rejected: %s" % type(error).__name__)

        def on_trace(self, message):
            try:
                sideband = _parse_result_sideband(
                    message.data, args.max_sideband_bytes
                )
                request_id = sideband["request_id"]
                self.pending_sidebands = _bounded_pending_put(
                    self.pending_sidebands,
                    request_id,
                    sideband,
                    len(message.data.encode("utf-8")),
                    args.max_pending_pairs,
                    args.max_pending_pairs * args.max_sideband_bytes,
                )
                self._complete(request_id)
            except Exception as error:
                self.failed += 1
                self.get_logger().error("trace sideband rejected: %s" % type(error).__name__)

        def _complete(self, request_id):
            if request_id not in self.pending_results or request_id not in self.pending_sidebands:
                return
            result_entry = self.pending_results[request_id]
            sideband_entry = self.pending_sidebands[request_id]
            self.pending_results = {
                key: value
                for key, value in self.pending_results.items()
                if key != request_id
            }
            self.pending_sidebands = {
                key: value
                for key, value in self.pending_sidebands.items()
                if key != request_id
            }
            wire, receive_ns = result_entry["value"]
            sideband = sideband_entry["value"]
            try:
                outputs, trace = _finalize_result_wire(
                    wire,
                    sideband,
                    receive_ns,
                    args.receiver_host_id,
                    receiver_boot,
                    limits,
                )
            except Exception:
                self.failed += 1
                raise
            self.last_outputs = outputs
            self.records.append(trace)

    rclpy.init(args=None)
    node = ResultSink()
    ready = {
        "event": "result_sink_ready",
        "result_topic": args.result_topic,
        "trace_topic": args.trace_topic,
        "receiver_host_id": args.receiver_host_id,
        "max_pending_pairs": args.max_pending_pairs,
        "max_pending_result_mib": args.max_pending_result_mib,
        "max_sideband_bytes": args.max_sideband_bytes,
    }
    _write_json_atomic(args.ready_file, ready)
    print(json.dumps(ready, sort_keys=True), flush=True)
    deadline = time.monotonic() + args.timeout_s
    try:
        while rclpy.ok() and len(node.records) < args.max_results:
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for formal RSU results")
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    write_trace_jsonl(args.trace_jsonl, node.records)
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **node.last_outputs)
    print(
        json.dumps(
            {
                "event": "result_sink_complete",
                "records": len(node.records),
                "failed": node.failed,
                "trace_jsonl": str(Path(args.trace_jsonl)),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.error("a mode is required")
    if args.max_payload_mib <= 0:
        parser.error("payload limit must be positive")
    if args.mode == "edge-relay":
        if args.timeout_s <= 0:
            parser.error("timeout must be positive")
        if args.max_requests < 0:
            parser.error("--max-requests must be non-negative")
        return run_edge_relay(args)
    if args.mode == "probe-client":
        if args.timeout_s <= 0:
            parser.error("timeout must be positive")
        if args.frame_id < 0 or args.sequence_id < 0 or args.discovery_timeout_s <= 0:
            parser.error("IDs must be non-negative and discovery timeout positive")
        return run_probe_client(args)
    if args.max_results <= 0:
        parser.error("max-results must be positive")
    if args.mode == "result-publisher":
        if not 1 <= args.result_port <= 65535:
            parser.error("result-port must be between 1 and 65535")
        if args.discovery_timeout_s <= 0 or args.socket_timeout_s <= 0:
            parser.error("publisher timeouts must be positive")
        return run_result_publisher(args)
    if args.timeout_s <= 0:
        parser.error("timeout must be positive")
    if (
        args.max_pending_pairs <= 0
        or args.max_pending_result_mib <= 0
        or args.max_sideband_bytes <= 0
    ):
        parser.error("result-sink pending limits must be positive")
    return run_result_sink(args)


if __name__ == "__main__":
    raise SystemExit(main())
