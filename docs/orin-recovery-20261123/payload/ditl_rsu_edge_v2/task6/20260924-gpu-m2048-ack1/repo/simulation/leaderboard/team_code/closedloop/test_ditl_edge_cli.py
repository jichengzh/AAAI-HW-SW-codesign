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


def formal_result_wire(request_id="local-run:0", sequence_id=0, payload_elements=16):
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
        "sequence_id": sequence_id,
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
        {"cls_preds": np.arange(payload_elements, dtype=np.float32).reshape(
            (payload_elements // 16, 16) if payload_elements > 16 else (payload_elements,)
        )},
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




class ResultAckTests(unittest.TestCase):
    def _publisher(self, behavior="delayed", count=24, ack_endpoint=True):
        from unittest import mock
        bridge = load_ros2_bridge_module()
        events, subscriptions, nodes = [], {}, []
        pending = []
        spins = [0]

        class Message:
            pass

        class Publisher:
            def __init__(self, topic):
                self.topic = topic

            def get_subscription_count(self):
                return 1

            def publish(self, message):
                if self.topic.endswith("perception_result"):
                    header, _ = bridge.inspect_wire_bytes_header(bytes(message.data), ProtocolLimits())
                    request_id = header["request_id"]
                    events.append(("result", request_id))
                    pending.append([request_id, 3])
                else:
                    request_id = json.loads(message.data)["request_id"]
                    self_test.assertIn(("ack", request_id), events)
                    self_test.assertFalse(pending, "sideband overtook the delayed result")
                    events.append(("sideband", request_id))

        class Node:
            def __init__(self, name):
                nodes.append(self)

            def create_publisher(self, message_type, topic, qos):
                return Publisher(topic)

            def create_subscription(self, message_type, topic, callback, qos):
                subscriptions[topic] = callback
                return callback

            def count_publishers(self, topic):
                return int(ack_endpoint)

            def destroy_node(self):
                events.append(("destroy", None))

        def spin_once(node, timeout_sec):
            spins[0] += 1
            if not pending or behavior == "timeout":
                return
            pending[0][1] -= 1
            if pending[0][1] > 0:
                return
            request_id = pending.pop(0)[0]
            message = Message()
            message.data = "wrong:0" if behavior == "wrong" else request_id
            callback = subscriptions["/ditl/rsu/perception_result_ack"]
            callback(message)
            events.append(("ack", request_id))
            if behavior == "duplicate":
                callback(message)

        def frame(connection, limits):
            index = len([event for event in events if event[0] == "result"])
            if index:
                self.assertIn(("sideband", "local-run:%d" % (index - 1)), events)
            wire = formal_result_wire("local-run:%d" % index)
            _, header_size, payload_size = bridge.FRAME_PREFIX.unpack(wire[:bridge.FRAME_PREFIX.size])
            start = bridge.FRAME_PREFIX.size
            return wire[start:start + header_size], wire[start + header_size:], len(wire)

        self_test = self
        ros = mock.Mock()
        ros.spin_once.side_effect = spin_once
        ros.ok.return_value = True
        listener = mock.MagicMock()
        listener.accept.return_value = (mock.MagicMock(), ("127.0.0.1", 1))
        listener.getsockname.return_value = ("127.0.0.1", 55021)
        args = bridge.build_parser().parse_args([
            "result-publisher", "--max-results", str(count),
            "--ack-timeout-s", "0.05", "--discovery-timeout-s", "0.05",
            "--shutdown-grace-s", "0",
        ])
        ticks = iter(index * 0.001 for index in range(10000))
        with mock.patch.object(bridge, "_ros_imports", return_value=(ros, Node, Message, Message, None)), mock.patch.object(bridge.socket, "socket", return_value=listener), mock.patch.object(bridge, "recv_wire_frame", side_effect=frame), mock.patch.object(bridge.time, "monotonic", side_effect=lambda: next(ticks)), contextlib.redirect_stdout(io.StringIO()):
            try:
                bridge.run_result_publisher(args)
            finally:
                listener.close.assert_called_once()
                ros.shutdown.assert_called_once()
        return events, spins[0]

    def test_delayed_results_ack_before_sideband_across_24_frames(self):
        events, spins = self._publisher()
        self.assertGreaterEqual(spins, 24 * 3)
        self.assertEqual(
            [(kind, "local-run:%d" % index) for index in range(24) for kind in ("result", "ack", "sideband")],
            [event for event in events if event[0] != "destroy"],
        )

    def test_wrong_duplicate_missing_ack_and_endpoint_fail_closed(self):
        for behavior, pattern in (("wrong", "ACK"), ("duplicate", "ACK"), ("timeout", "ACK")):
            with self.subTest(behavior=behavior), self.assertRaisesRegex(Exception, pattern):
                self._publisher(behavior=behavior, count=2)
        with self.assertRaisesRegex(RuntimeError, "ACK"):
            self._publisher(ack_endpoint=False)

    def test_ack_state_rejects_unexpected_invalid_reused_and_overlapping_ids(self):
        bridge = load_ros2_bridge_module()
        for request_id in ("run:0", "bad\n", {}, ""):
            gate = bridge.ResultAckGate()
            gate.receive(request_id)
            with self.assertRaisesRegex(bridge.ProtocolError, "ACK"):
                gate.check()
        for request_id in ("bad\n", {}, ""):
            with self.subTest(request_id=request_id), self.assertRaisesRegex(bridge.ProtocolError, "ACK"):
                bridge.ResultAckGate().begin(request_id)
        with self.assertRaisesRegex(bridge.ProtocolError, "ACK"):
            bridge.ResultAckGate().finish()
        gate = bridge.ResultAckGate()
        gate.begin("run:0")
        with self.assertRaisesRegex(bridge.ProtocolError, "ACK"):
            gate.begin("run:1")
        gate = bridge.ResultAckGate()
        gate.begin("run:0")
        gate.receive("run:0")
        gate.finish()
        with self.assertRaisesRegex(bridge.ProtocolError, "ACK"):
            gate.begin("run:0")

    def test_ack_cli_defaults_and_finite_positive_timeout(self):
        bridge = load_ros2_bridge_module()
        publisher = bridge.build_parser().parse_args(["result-publisher", "--max-results", "1"])
        sink = bridge.build_parser().parse_args(["result-sink", "--max-results", "1", "--trace-jsonl", "x", "--output-npz", "y"])
        self.assertEqual(publisher.ack_topic, sink.ack_topic)
        self.assertGreater(publisher.ack_timeout_s, 0)
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bridge.main(["result-publisher", "--max-results", "1", "--ack-timeout-s", value])

    def test_sink_ack_only_after_successful_header_and_pending_admission(self):
        from unittest import mock
        bridge = load_ros2_bridge_module()
        nodes, acknowledgements = [], []

        class Message:
            def __init__(self, data=None):
                self.data = data

        class Node:
            def __init__(self, name):
                self.callbacks = {}
                nodes.append(self)

            def create_subscription(self, message_type, topic, callback, qos):
                self.callbacks[topic] = callback
                return callback

            def create_publisher(self, message_type, topic, qos):
                publisher = mock.Mock()
                def publish(message):
                    self_test.assertIn(message.data, self.pending_results)
                    acknowledgements.append(message.data)
                publisher.publish.side_effect = publish
                return publisher

            def get_logger(self):
                return mock.Mock()

            def destroy_node(self):
                pass

        self_test = self
        sideband = {"request_id": "local-run:0", "local_handoff_end": 2_021_000_000,
                    "ros_message_start": 2_022_000_000, "ros_message_end": 2_023_000_000,
                    "ros_publish_start": 2_024_000_000, "ros_publish_end": 2_025_000_000}
        actions = [
            ("result", b"invalid"),
            ("result", formal_result_wire()),
            ("result", formal_result_wire()),  # duplicate: no second ACK
            ("result", formal_result_wire("local-run:1")),  # pending cap: no ACK
            ("trace", json.dumps(sideband)),
        ]
        def spin_once(node, timeout_sec):
            kind, data = actions.pop(0)
            node.callbacks["/ditl/rsu/perception_" + kind](Message(data))
        ros = mock.Mock()
        ros.ok.return_value = True
        ros.spin_once.side_effect = spin_once
        with tempfile.TemporaryDirectory() as directory:
            args = bridge.build_parser().parse_args([
                "result-sink", "--max-results", "1", "--max-pending-pairs", "1",
                "--receiver-boot-id", "boot-orin", "--trace-jsonl", directory + "/trace.jsonl",
                "--output-npz", directory + "/output.npz", "--ready-file", directory + "/ready.json",
            ])
            with mock.patch.object(bridge, "_ros_imports", return_value=(ros, Node, Message, Message, None)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, bridge.run_result_sink(args))
            ready = json.loads(Path(directory, "ready.json").read_text())
            self.assertEqual(args.ack_topic, ready["ack_topic"])
            self.assertEqual(1, len(read_trace_jsonl(directory + "/trace.jsonl")))
        self.assertEqual(["local-run:0"], acknowledgements)
        self.assertEqual(3, nodes[0].failed)


class RosBridgeRegressionTests(unittest.TestCase):
    def test_legacy_relay_success_mismatch_and_transport_failures_cleanup(self):
        from unittest import mock
        bridge = load_ros2_bridge_module()
        for case in ("success", "mismatch", "invalid", "publish"):
            with self.subTest(case=case):
                nodes = []
                class Message:
                    pass
                class Node:
                    def __init__(self, name):
                        nodes.append(self)
                        self.output = mock.Mock()
                        if case == "publish":
                            self.output.publish.side_effect = RuntimeError("DDS failure")
                    def create_publisher(self, *args):
                        return self.output
                    def create_subscription(self, message_type, topic, callback, qos):
                        self.callback = callback
                        return callback
                    def get_logger(self):
                        return mock.Mock()
                    def destroy_node(self):
                        self.destroyed = True
                def spin(node, timeout_sec):
                    message = Message()
                    message.data = b"invalid" if case == "invalid" else formal_result_wire()
                    node.callback(message)
                ros = mock.Mock()
                ros.ok.side_effect = [True, False]
                ros.spin_once.side_effect = spin
                wire = formal_result_wire("local-run:1" if case == "mismatch" else "local-run:0")
                size = bridge.FRAME_PREFIX.size
                _, header_size, _ = bridge.FRAME_PREFIX.unpack(wire[:size])
                frame = (wire[size:size + header_size], wire[size + header_size:], len(wire))
                args = bridge.build_parser().parse_args(["edge-relay", "--max-requests", "1", "--shutdown-grace-s", "0"])
                with mock.patch.object(bridge, "_ros_imports", return_value=(ros, Node, Message, Message, None)), mock.patch.object(bridge.socket, "create_connection", return_value=mock.MagicMock()), mock.patch.object(bridge, "recv_wire_frame", return_value=frame), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(0, bridge.run_edge_relay(args))
                node = nodes[0]
                self.assertTrue(node.destroyed)
                ros.shutdown.assert_called_once()
                self.assertEqual(int(case != "success"), node.failed)
                self.assertEqual(int(case != "publish"), node.served)
                if case != "publish":
                    response, _, _ = decode_wire_bytes(bytes(node.output.publish.call_args[0][0].data))
                    self.assertEqual("result" if case == "success" else "error", response["kind"])

    def test_ros_qos_remains_reliable_depth_one(self):
        import types
        from unittest import mock
        bridge = load_ros2_bridge_module()
        qos = types.ModuleType("rclpy.qos")
        qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile")
        qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
        qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="reliable")
        qos.QoSProfile = mock.Mock()
        node = types.ModuleType("rclpy.node")
        node.Node = mock.Mock()
        messages = types.ModuleType("std_msgs.msg")
        messages.String = mock.Mock()
        messages.UInt8MultiArray = mock.Mock()
        modules = {"rclpy": types.ModuleType("rclpy"), "rclpy.node": node,
                   "rclpy.qos": qos, "std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": messages}
        with mock.patch.dict(sys.modules, modules):
            bridge._ros_imports()
        qos.QoSProfile.assert_called_once_with(history="keep_last", depth=1, reliability="reliable", durability="volatile")
        with mock.patch.dict(sys.modules, {"rclpy": None}), self.assertRaisesRegex(RuntimeError, "ROS 2 Python"):
            bridge._ros_imports()

    def test_cli_dispatch_and_invalid_boundaries_without_ros(self):
        from unittest import mock
        bridge = load_ros2_bridge_module()
        client = ["probe-client", "--input-npz", "in", "--output-npz", "out", "--trace-jsonl", "trace", "--latency-csv", "csv"]
        sink = ["result-sink", "--max-results", "1", "--output-npz", "out", "--trace-jsonl", "trace"]
        publisher = ["result-publisher", "--max-results", "1"]
        for arguments, function in ((["edge-relay"], "run_edge_relay"), (client, "run_probe_client"), (sink, "run_result_sink"), (publisher, "run_result_publisher")):
            with mock.patch.object(bridge, function, return_value=17) as run:
                self.assertEqual(17, bridge.main(arguments))
                run.assert_called_once()
        invalid = [[], ["edge-relay", "--max-payload-mib", "0"], ["edge-relay", "--timeout-s", "0"],
                   ["edge-relay", "--max-requests", "-1"], client + ["--timeout-s", "0"],
                   client + ["--frame-id", "-1"], publisher + ["--max-results", "0"],
                   publisher + ["--result-port", "0"], publisher + ["--socket-timeout-s", "0"],
                   publisher + ["--shutdown-grace-s", "-1"], sink + ["--timeout-s", "0"],
                   sink + ["--max-pending-pairs", "0"], sink + ["--max-pending-result-mib", "0"],
                   sink + ["--max-sideband-bytes", "0"]]
        for arguments in invalid:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bridge.main(arguments)

    def test_actual_publisher_sink_callbacks_pair_24_delayed_megabyte_results(self):
        import queue
        import threading
        from unittest import mock
        bridge = load_ros2_bridge_module()
        nodes, subscribers, publishers, events, errors = {}, {}, {}, [], []
        maximum_pending = [0]
        class Message:
            pass
        class Node:
            def __init__(self, name):
                self.name, self.queue = name, queue.Queue()
                nodes[name] = self
            def create_subscription(self, kind, topic, callback, qos):
                subscribers[topic] = (self, callback)
                return callback
            def create_publisher(self, kind, topic, qos):
                publishers[topic] = self
                publisher = mock.Mock()
                publisher.get_subscription_count.side_effect = lambda: int(topic in subscribers)
                def publish(message):
                    if topic.endswith("result_ack"):
                        self_test.assertIn(message.data, self.pending_results)
                        maximum_pending[0] = max(maximum_pending[0], len(self.pending_results))
                        events.append(("ack", message.data))
                    elif topic.endswith("perception_trace"):
                        request_id = json.loads(message.data)["request_id"]
                        self_test.assertIn(("ack", request_id), events)
                        events.append(("sideband", request_id))
                    destination, callback = subscribers[topic]
                    destination.queue.put((topic, callback, message))
                    if topic.endswith("perception_result"):
                        # Receiver can run before this publish call returns.
                        time.sleep(0.015)
                publisher.publish.side_effect = publish
                return publisher
            def count_publishers(self, topic):
                return int(topic in publishers)
            def get_logger(self):
                return mock.Mock()
            def destroy_node(self):
                pass
        def spin(node, timeout_sec):
            try:
                topic, callback, message = node.queue.get(timeout=max(timeout_sec, 0.0001))
            except queue.Empty:
                return
            if topic.endswith("perception_result"):
                time.sleep(0.002)  # Delay large result delivery independently of sideband.
            callback(message)
        def background(function):
            try:
                function()
            except Exception as error:
                errors.append(error)
        self_test = self
        ros = mock.Mock()
        ros.ok.return_value = True
        ros.spin_once.side_effect = spin
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            port = free_tcp_port()
            sink = bridge.build_parser().parse_args([
                "result-sink", "--max-results", "24", "--timeout-s", "10",
                "--receiver-boot-id", "boot-orin", "--trace-jsonl", str(root / "trace.jsonl"),
                "--output-npz", str(root / "output.npz"),
            ])
            publisher = bridge.build_parser().parse_args([
                "result-publisher", "--max-results", "24", "--result-port", str(port),
                "--ack-timeout-s", "2", "--discovery-timeout-s", "2", "--socket-timeout-s", "2",
                "--shutdown-grace-s", "0", "--ready-file", str(root / "ready.json"),
            ])
            def native():
                deadline = time.monotonic() + 3
                while not (root / "ready.json").exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("publisher readiness timeout")
                    time.sleep(0.001)
                with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
                    for index in range(24):
                        connection.sendall(formal_result_wire("local-run:%d" % index, index, 262144))
            with mock.patch.object(bridge, "_ros_imports", return_value=(ros, Node, Message, Message, None)), contextlib.redirect_stdout(io.StringIO()):
                threads = [threading.Thread(target=background, args=(function,)) for function in (lambda: bridge.run_result_sink(sink), native)]
                for thread in threads:
                    thread.start()
                try:
                    bridge.run_result_publisher(publisher)
                finally:
                    for thread in threads:
                        thread.join(timeout=12)
                self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual([], errors)
            records = read_trace_jsonl(root / "trace.jsonl")
            self.assertEqual(24, len(records))
            self.assertTrue(any(row["timestamps_ns"]["ros_receive"] < row["timestamps_ns"]["ros_publish_end"] for row in records))
            self.assertEqual(1, maximum_pending[0])
            self.assertEqual(0, nodes["ditl_rsu_result_sink"].failed)
            self.assertEqual({}, nodes["ditl_rsu_result_sink"].pending_results)
            self.assertEqual({}, nodes["ditl_rsu_result_sink"].pending_sidebands)
            self.assertEqual(publisher.ack_topic, json.loads((root / "ready.json").read_text())["ack_topic"])


if __name__ == "__main__":
    unittest.main()
