#!/usr/bin/env python3
"""One real ROS payload round trip, not a network benchmark or driving HIL.

Run inside the existing ROS container using a serial probe --last-wire export.
Monotonic time is used exclusively to bound discovery and receipt deadlines.
"""

import argparse
import array
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = ROOT / "simulation/leaderboard/team_code/closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_artifacts import DEFAULT_CONFIG_DIR
from coptv2x_hil_backend import (
    CODRIVING_BATCH_ORDER, CODRIVING_RSU_BATCH_ORDER,
    COPTV2X_RSU_PAYLOAD_SEMANTICS, HilBackendError,
    hash_feature_payload, validate_feature_payload_contract,
)
from ditl_edge_protocol import ProtocolError, decode_wire_bytes


DEFAULT_TOPIC = "/coptv2x/serial_opt_smoke/result"


def validate_rsu_wire(
    wire, config_dir, *, expected_request_id=None, expected_payload_sha256=None,
):
    """Validate bytes and frozen tensor identity; this alone is not ROS evidence."""
    header, outputs, _ = decode_wire_bytes(wire, readonly=True)
    if header.get("kind") != "result" or header.get("status") != "ok":
        raise ProtocolError("smoke requires a successful result frame")
    if header.get("payload_semantics") != COPTV2X_RSU_PAYLOAD_SEMANTICS:
        raise ProtocolError("smoke requires explicit RSU-only feature semantics")
    if header.get("batch_order") != list(CODRIVING_RSU_BATCH_ORDER):
        raise ProtocolError("smoke requires payload batch_order [rsu]")
    if header.get("source_batch_order") != list(CODRIVING_BATCH_ORDER):
        raise ProtocolError("smoke requires source_batch_order [ego, rsu]")
    if header["tensor_bundle"]["compression"] != "none":
        raise ProtocolError("serial smoke requires uncompressed FP32 features")
    try:
        validate_feature_payload_contract(
            header.get("artifact_identity"), outputs, config_dir,
            payload_semantics=COPTV2X_RSU_PAYLOAD_SEMANTICS,
        )
        actual_hash = hash_feature_payload(outputs)
    except HilBackendError as error:
        raise ProtocolError("smoke frozen feature contract failed") from error
    if header.get("payload_sha256") != actual_hash:
        raise ProtocolError("smoke feature payload hash mismatch")
    if expected_request_id is not None and header["request_id"] != expected_request_id:
        raise ProtocolError("received request_id differs from published frame")
    if expected_payload_sha256 is not None and actual_hash != expected_payload_sha256:
        raise ProtocolError("received feature hash differs from published frame")
    return {
        "request_id": header["request_id"],
        "arm_id": header["artifact_identity"]["arm_id"],
        "payload_bytes": header["tensor_bundle"]["raw_payload_bytes"],
        "payload_semantics": header["payload_semantics"],
        "batch_order": list(header["batch_order"]),
        "source_batch_order": list(header["source_batch_order"]),
        "artifact_sha256": dict(header["artifact_identity"]["artifact_sha256"]),
        "expected_payload_sha256": header["payload_sha256"],
        "actual_payload_sha256": actual_hash,
    }


def _spin_until(executor, rclpy, predicate, deadline, label):
    while rclpy.ok():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ROS smoke timed out waiting for " + label)
        executor.spin_once(timeout_sec=min(0.1, remaining))
        if predicate():
            return
    raise RuntimeError("ROS stopped while waiting for " + label)


def run_ros_smoke(wire, config_dir, topic=DEFAULT_TOPIC, timeout_s=30.0):
    """Publish once after discovery; confirm actual receipt through ROS callbacks."""
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
        raise ValueError("timeout_s must be finite and in (0, 30]")
    expected = validate_rsu_wire(wire, config_dir)
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import UInt8MultiArray
    except ImportError as error:
        raise RuntimeError("run in the existing sourced ROS 2 container") from error

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
    )
    received, errors = [], []
    def on_result(message):
        try:
            received.append(validate_rsu_wire(
                bytes(message.data), config_dir,
                expected_request_id=expected["request_id"],
                expected_payload_sha256=expected["actual_payload_sha256"],
            ))
        except Exception as error:
            errors.append(error)

    rclpy.init(args=[])
    sender = receiver = executor = None
    try:
        sender = Node("coptv2x_serial_smoke_sender_%d" % os.getpid())
        receiver = Node("coptv2x_serial_smoke_receiver_%d" % os.getpid())
        publisher = sender.create_publisher(UInt8MultiArray, topic, qos)
        receiver.create_subscription(UInt8MultiArray, topic, on_result, qos)
        executor = SingleThreadedExecutor()
        executor.add_node(sender)
        executor.add_node(receiver)
        deadline = time.monotonic() + timeout_s
        _spin_until(
            executor, rclpy,
            lambda: publisher.get_subscription_count() > 0 and receiver.count_publishers(topic) > 0,
            deadline, "publisher/subscriber discovery",
        )
        message = UInt8MultiArray()
        message.data = array.array("B", wire)
        publisher.publish(message)
        _spin_until(executor, rclpy, lambda: bool(received or errors), deadline, "received frame")
        if errors:
            raise errors[0]
        return dict(
            received[0], topic=topic, passed=True, is_ros_smoke=True,
            is_carla=False, is_hil_driving=False,
        )
    finally:
        if executor is not None:
            executor.shutdown()
        for node in (sender, receiver):
            if node is not None:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", type=Path, required=True, help="probe --last-wire output")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        result = run_ros_smoke(args.wire.read_bytes(), args.config_dir, args.topic, args.timeout_s)
    except Exception as error:
        result = {
            "topic": args.topic, "passed": False, "is_ros_smoke": True,
            "is_carla": False, "is_hil_driving": False,
            "error_type": type(error).__name__, "error": str(error),
        }
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
