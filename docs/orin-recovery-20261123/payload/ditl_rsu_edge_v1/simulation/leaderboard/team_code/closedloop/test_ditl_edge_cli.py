from __future__ import print_function

import array
import json
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ditl_edge_protocol import ProtocolLimits, build_wire_bytes, decode_wire_bytes
from ditl_trace_schema import read_trace_jsonl


REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE = REPO_ROOT / "scripts" / "ditl_rsu_edge_service.py"
CLIENT = REPO_ROOT / "scripts" / "ditl_rsu_edge_client.py"
ROS2_BRIDGE = REPO_ROOT / "scripts" / "ditl_ros2_edge_bridge.py"


def load_ros2_bridge_module():
    spec = importlib.util.spec_from_file_location("ditl_ros2_edge_bridge", str(ROS2_BRIDGE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class DitlEdgeCliTests(unittest.TestCase):
    def test_ros2_relay_converts_callback_failure_to_error_frame(self):
        bridge = load_ros2_bridge_module()
        limits = bridge.ProtocolLimits(max_payload_bytes=1024, max_raw_payload_bytes=1024)
        wire = bridge._build_relay_error_wire("relay-error-001", OSError(), limits)
        header, arrays, _ = bridge.decode_wire_bytes(wire, limits)
        self.assertEqual("error", header["kind"])
        self.assertEqual("relay_error", header["error_code"])
        self.assertEqual("OSError", header["message"])
        self.assertEqual({}, arrays)

    def test_ros2_relay_has_nonzero_shutdown_delivery_grace(self):
        bridge = load_ros2_bridge_module()
        args = bridge.build_parser().parse_args(["edge-relay"])
        self.assertGreater(args.shutdown_grace_s, 0.0)

    def test_ros2_probe_waits_for_request_and_response_endpoint_matches(self):
        bridge = load_ros2_bridge_module()

        class Publisher(object):
            def __init__(self, subscriptions):
                self.subscriptions = subscriptions

            def get_subscription_count(self):
                return self.subscriptions

        class Node(object):
            def __init__(self, publishers):
                self.publishers = publishers

            def count_publishers(self, topic):
                self.topic = topic
                return self.publishers

        self.assertFalse(
            bridge._ros_endpoints_ready(Node(0), Publisher(1), "/response")
        )
        self.assertFalse(
            bridge._ros_endpoints_ready(Node(1), Publisher(0), "/response")
        )
        self.assertTrue(
            bridge._ros_endpoints_ready(Node(1), Publisher(1), "/response")
        )

    def test_ros2_uint8_multi_array_conversion_preserves_wire_bytes(self):
        bridge = load_ros2_bridge_module()
        wire = b"\x00\x01\x7f\x80\xff"
        ros_values = bridge._wire_to_ros_uint8_array(wire)
        self.assertIsInstance(ros_values, array.array)
        self.assertEqual("B", ros_values.typecode)
        self.assertEqual(wire, bridge._ros_uint8_array_to_wire(ros_values))

    def test_ros2_relay_builds_protocol_error_wire_for_bad_request(self):
        bridge = load_ros2_bridge_module()
        limits = ProtocolLimits(max_payload_bytes=1024, max_raw_payload_bytes=1024)

        wire = bridge._error_wire_for_request_wire(
            b"not-a-ditl-frame", "relay_error", "socket failed", limits
        )
        header, arrays, _ = decode_wire_bytes(wire, limits)

        self.assertEqual("error", header["kind"])
        self.assertEqual("invalid-request", header["request_id"])
        self.assertEqual("relay_error", header["error_code"])
        self.assertEqual({}, arrays)

    def test_ros2_relay_preserves_request_id_in_error_wire_for_valid_request(self):
        bridge = load_ros2_bridge_module()
        request = {
            "protocol_version": "ditl.edge.v1",
            "kind": "request",
            "request_id": "ros-error-001",
            "operation": "infer",
        }
        wire = build_wire_bytes(
            request,
            {"feature": np.zeros((1,), dtype=np.float32)},
            limits=ProtocolLimits(max_payload_bytes=1024, max_raw_payload_bytes=1024),
        )

        error_wire = bridge._error_wire_for_request_wire(
            wire,
            "relay_error",
            "socket failed",
            ProtocolLimits(max_payload_bytes=1024, max_raw_payload_bytes=1024),
        )
        header, arrays, _ = decode_wire_bytes(error_wire)

        self.assertEqual("error", header["kind"])
        self.assertEqual("ros-error-001", header["request_id"])
        self.assertEqual("relay_error", header["error_code"])
        self.assertEqual({}, arrays)

    def test_ros2_bridge_help_is_available_before_importing_rclpy(self):
        result = subprocess.run(
            [sys.executable, str(ROS2_BRIDGE), "--help"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn("edge-relay", result.stdout)
        self.assertIn("probe-client", result.stdout)

    def test_replay_service_and_client_produce_output_trace_and_latency_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.npz"
            output_path = tmp / "output.npz"
            trace_path = tmp / "trace.jsonl"
            csv_path = tmp / "latency.csv"
            ready_path = tmp / "ready.json"
            service_log = tmp / "service.log"
            np.savez(
                str(input_path),
                spatial_features=np.ones((1, 2, 3), dtype=np.float32),
            )
            port = free_tcp_port()
            with service_log.open("w+") as log:
                service = subprocess.Popen(
                    [
                        sys.executable,
                        str(SERVICE),
                        "--backend",
                        "replay",
                        "--replay-scale",
                        "3",
                        "--bind",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--max-requests",
                        "1",
                        "--ready-file",
                        str(ready_path),
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.time() + 5
                    while time.time() < deadline and not ready_path.exists():
                        if service.poll() is not None:
                            break
                        time.sleep(0.02)
                    self.assertTrue(ready_path.exists(), "service did not become ready")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(CLIENT),
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(port),
                            "--input-npz",
                            str(input_path),
                            "--output-npz",
                            str(output_path),
                            "--trace-jsonl",
                            str(trace_path),
                            "--latency-csv",
                            str(csv_path),
                            "--request-id",
                            "cli-req-001",
                        ],
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=10,
                    )
                    service.wait(timeout=5)
                finally:
                    if service.poll() is None:
                        service.terminate()
                        service.wait(timeout=5)
                log.seek(0)
                service_output = log.read()

            self.assertEqual(0, result.returncode, result.stdout)
            self.assertEqual(0, service.returncode, service_output)
            with np.load(str(output_path), allow_pickle=False) as output:
                np.testing.assert_array_equal(
                    np.full((1, 2, 3), 3.0, dtype=np.float32),
                    output["spatial_features"],
                )
            records = read_trace_jsonl(trace_path)
            self.assertEqual(1, len(records))
            self.assertEqual("cli-req-001", records[0]["trace_id"])
            self.assertTrue(csv_path.exists())
            summary = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual("replay-rsu-v1", summary["model_id"])


if __name__ == "__main__":
    unittest.main()
