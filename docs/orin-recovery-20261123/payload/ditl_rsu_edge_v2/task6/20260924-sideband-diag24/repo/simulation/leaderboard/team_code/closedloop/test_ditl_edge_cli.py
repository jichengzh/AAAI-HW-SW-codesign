from __future__ import print_function

import array
import contextlib
import io
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
EDGE_PUBLISHER = REPO_ROOT / "scripts" / "ditl_rsu_edge_publisher.py"
README = Path(__file__).resolve().with_name("README_DITL_EDGE.md")


def load_ros2_bridge_module():
    spec = importlib.util.spec_from_file_location("ditl_ros2_edge_bridge", str(ROS2_BRIDGE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_edge_publisher_module():
    spec = importlib.util.spec_from_file_location(
        "ditl_rsu_edge_publisher", str(EDGE_PUBLISHER)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_edge_service_module():
    spec = importlib.util.spec_from_file_location("ditl_rsu_edge_service", str(SERVICE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def formal_result_wire(request_id="local-run:0"):
    base = 2_000_000_000
    sender_names = (
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
    timestamps = {
        name: base + index * 1_000_000
        for index, name in enumerate(sender_names)
    }
    header = {
        "protocol_version": "ditl.edge.v1",
        "kind": "result",
        "request_id": request_id,
        "status": "ok",
        "run_id": "local-run",
        "route_id": "tensor-replay",
        "scenario_id": "edge-rsu",
        "source_id": "rsu-orin-01",
        "target_id": "ego-server-01",
        "frame_id": 0,
        "sequence_id": 0,
        "source_type": "local_feature_replay",
        "payload_semantics": "dense_head_debug",
        "postprocess": "identity",
        "remote_input_tensor_bytes": 0,
        "edge_clock": {
            "host_id": "orin-agx-01",
            "domain": "monotonic_raw",
            "boot_id": "boot-orin",
            "sync_state": "locked",
            "offset_ns": 0,
            "uncertainty_ns": 0,
        },
        "edge_timestamps_ns": timestamps,
        "durations_ns": {
            "source_read": 1_000_000,
            "preprocess": 1_000_000,
            "h2d": 1_000_000,
            "gpu_inference": 1_000_000,
            "d2h": 1_000_000,
            "postprocess": 1_000_000,
            "pack": 1_000_000,
            "hash": 1_000_000,
            "compress": 0,
            "protocol_serialize": 1_000_000,
        },
        "model": {"backend": "fake", "model_id": "fake-rsu-v1"},
        "device": {"hostname": "orin-agx-01"},
    }
    return build_wire_bytes(
        header,
        {"cls_preds": np.arange(16, dtype=np.float32)},
        compression="none",
    )


def free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class DitlEdgeCliTests(unittest.TestCase):
    def test_legacy_service_requires_allowlist_for_non_loopback_bind(self):
        service = load_edge_service_module()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                service.parse_args(["--backend", "replay", "--bind", "0.0.0.0"])
        args = service.parse_args(
            [
                "--backend",
                "replay",
                "--bind",
                "0.0.0.0",
                "--allow-client",
                "192.0.2.10",
            ]
        )
        self.assertEqual(["192.0.2.10"], args.allow_client)

    def test_readme_canonical_path_is_local_result_only_v2(self):
        text = README.read_text(encoding="utf-8")

        self.assertIn("/home/jichengzhi/ditl_rsu_edge_v2", text)
        self.assertIn("ditl_rsu_edge_publisher.py", text)
        self.assertIn("result-publisher", text)
        self.assertIn("result-sink", text)
        self.assertIn("--compression none", text)
        self.assertIn("legacy testbed probe", text.lower())
        self.assertIn("rsu_to_fusion_latency", text)
    def test_local_publisher_has_no_remote_perception_input(self):
        publisher = load_edge_publisher_module()
        help_text = publisher.build_parser().format_help()

        self.assertIn("--data-root", help_text)
        self.assertIn("--manifest", help_text)
        self.assertIn("--engine", help_text)
        self.assertNotIn("--request-topic", help_text)
        self.assertNotIn("--input-npz", help_text)
        self.assertNotIn("--edge-host", help_text)

    def test_result_sink_builds_valid_v2_trace_from_result_and_sideband(self):
        bridge = load_ros2_bridge_module()
        wire = formal_result_wire()
        sideband = {
            "request_id": "local-run:0",
            "local_handoff_end": 2_021_000_000,
            "ros_message_start": 2_022_000_000,
            "ros_message_end": 2_023_000_000,
            "ros_publish_start": 2_024_000_000,
            "ros_publish_end": 2_025_000_000,
        }

        arrays, trace = bridge._finalize_result_wire(
            wire,
            sideband,
            ros_receive_ns=2_026_000_000,
            receiver_host_id="orin-agx-01",
            receiver_boot_id="boot-orin",
            limits=ProtocolLimits(),
        )

        self.assertEqual("ditl.trace.v2", trace["schema_version"])
        self.assertEqual("rsu_to_fusion", trace["measurement_scope"])
        self.assertTrue(trace["headline_eligible"])
        self.assertNotIn("round_trip", trace["derived_ns"])
        np.testing.assert_array_equal(
            np.arange(16, dtype=np.float32), arrays["cls_preds"]
        )

    def test_result_sink_rejects_error_and_sideband_identity_mismatch(self):
        bridge = load_ros2_bridge_module()
        error_wire = build_wire_bytes(
            {
                "protocol_version": "ditl.edge.v1",
                "kind": "error",
                "request_id": "local-run:0",
                "error_code": "backend_error",
                "message": "RuntimeError",
            },
            {},
        )
        sideband = {
            "request_id": "local-run:0",
            "local_handoff_end": 1,
            "ros_message_start": 2,
            "ros_message_end": 3,
            "ros_publish_start": 4,
            "ros_publish_end": 5,
        }
        with self.assertRaises(bridge.ProtocolError):
            bridge._finalize_result_wire(
                error_wire, sideband, 6, "orin-agx-01", "boot-orin", ProtocolLimits()
            )

        with self.assertRaisesRegex(bridge.ProtocolError, "request_id"):
            bridge._finalize_result_wire(
                formal_result_wire(),
                dict(sideband, request_id="wrong"),
                2_026_000_000,
                "orin-agx-01",
                "boot-orin",
                ProtocolLimits(),
            )

    def test_result_sink_bounds_unmatched_result_and_sideband_state(self):
        bridge = load_ros2_bridge_module()
        original = {}
        pending = bridge._bounded_pending_put(
            original, "run:0", (b"1234", 1), 4, max_entries=1, max_bytes=4
        )

        self.assertEqual({}, original)
        self.assertEqual({"run:0"}, set(pending))
        with self.assertRaisesRegex(bridge.ProtocolError, "entry limit"):
            bridge._bounded_pending_put(
                pending, "run:1", (b"1", 2), 1, max_entries=1, max_bytes=4
            )
        with self.assertRaisesRegex(bridge.ProtocolError, "byte limit"):
            bridge._bounded_pending_put(
                {}, "run:0", (b"12345", 1), 5, max_entries=1, max_bytes=4
            )

        sideband = bridge._parse_result_sideband(
            json.dumps({"request_id": "run:0"}), max_bytes=64
        )
        self.assertEqual("run:0", sideband["request_id"])
        with self.assertRaisesRegex(bridge.ProtocolError, "sideband size"):
            bridge._parse_result_sideband("{}" * 33, max_bytes=64)
        with self.assertRaisesRegex(bridge.ProtocolError, "request_id"):
            bridge._parse_result_sideband(
                json.dumps({"request_id": "../unsafe"}), max_bytes=64
            )

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
        self.assertIn("result-publisher", result.stdout)
        self.assertIn("result-sink", result.stdout)

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
