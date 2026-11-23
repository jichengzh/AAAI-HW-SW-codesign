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
    ProtocolError,
    ProtocolLimits,
    build_wire_bytes,
    decode_wire_bytes,
    recv_wire_frame,
)
from ditl_edge_runtime import build_round_trip_trace, monotonic_raw_ns
from ditl_trace_schema import write_latency_csv, write_trace_jsonl


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
        from std_msgs.msg import UInt8MultiArray
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
    return rclpy, Node, UInt8MultiArray, qos


def _wire_from_socket(connection, limits):
    header_bytes, payload, _ = recv_wire_frame(connection, limits)
    return FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload)) + header_bytes + payload


def _wire_to_ros_uint8_array(wire):
    """Convert a wire frame to Foxy's efficient uint8[] representation."""
    return array.array("B", wire)


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


def run_edge_relay(args):
    rclpy, Node, UInt8MultiArray, qos = _ros_imports()
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

        def on_request(self, message):
            request_wire = _ros_uint8_array_to_wire(message.data)
            try:
                request, _, _ = decode_wire_bytes(request_wire, limits)
                with socket.create_connection(
                    (args.edge_host, args.edge_port), timeout=args.timeout_s
                ) as connection:
                    connection.settimeout(args.timeout_s)
                    connection.sendall(request_wire)
                    response_wire = _wire_from_socket(connection, limits)
                response, _, _ = decode_wire_bytes(response_wire, limits)
                if response["request_id"] != request["request_id"]:
                    raise RuntimeError("edge response request_id mismatch")
            except Exception as error:
                response_wire = _error_wire_for_request_wire(
                    request_wire, "relay_error", type(error).__name__, limits
                )
                self.get_logger().warning(
                    json.dumps(
                        {"event": "relay_error", "error": type(error).__name__},
                        sort_keys=True,
                    )
                )
            output = UInt8MultiArray()
            output.data = _wire_to_ros_uint8_array(response_wire)
            self.publisher.publish(output)
            self.served += 1
            self.get_logger().info(
                json.dumps(
                    {
                        "event": "relayed",
                        "request_id": request["request_id"],
                        "wire_bytes": len(response_wire),
                        "served": self.served,
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
    rclpy, Node, UInt8MultiArray, qos = _ros_imports()
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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.error("a mode is required: edge-relay or probe-client")
    if args.timeout_s <= 0 or args.max_payload_mib <= 0:
        parser.error("timeout and payload limit must be positive")
    if args.mode == "edge-relay":
        if args.max_requests < 0:
            parser.error("--max-requests must be non-negative")
        return run_edge_relay(args)
    if args.frame_id < 0 or args.sequence_id < 0 or args.discovery_timeout_s <= 0:
        parser.error("IDs must be non-negative and discovery timeout positive")
    return run_probe_client(args)


if __name__ == "__main__":
    raise SystemExit(main())
