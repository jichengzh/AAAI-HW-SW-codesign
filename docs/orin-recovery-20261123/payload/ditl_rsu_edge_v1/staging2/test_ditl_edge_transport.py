from __future__ import print_function

import copy
import os
import socket
import sys
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ditl_edge_protocol import (
    ProtocolError,
    ProtocolLimits,
    build_wire_bytes,
    decode_wire_bytes,
    decode_tensor_bundle,
    encode_tensor_bundle,
    recv_envelope,
    send_envelope,
)
from ditl_edge_runtime import (
    ReplayBackend,
    build_round_trip_trace,
    handle_connection,
    request_inference,
)
from ditl_trace_schema import validate_trace_record
import ditl_edge_runtime


def request_header(request_id="req-001"):
    return {
        "protocol_version": "ditl.edge.v1",
        "kind": "request",
        "request_id": request_id,
        "operation": "infer",
        "run_id": "run-001",
        "route_id": "route-146",
        "scenario_id": "scenario-1",
        "source_id": "rsu-01",
        "target_id": "ego-01",
        "frame_id": 42,
        "sequence_id": 7,
        "topic": "/ditl/rsu/features",
        "message_type": "intermediate_feature",
        "compression_ratio": 1.0,
    }


class DitlEdgeProtocolTests(unittest.TestCase):
    def test_service_encodes_response_tensor_payload_once(self):
        backend = ReplayBackend(scale=1.0)
        left, right = socket.socketpair()
        errors = []

        def serve():
            try:
                handle_connection(left, ("local", 0), backend, response_compression="zlib")
            except Exception as error:
                errors.append(error)
            finally:
                left.close()

        with mock.patch.object(
            ditl_edge_runtime,
            "prepare_envelope",
            wraps=ditl_edge_runtime.prepare_envelope,
        ) as prepare:
            thread = threading.Thread(target=serve)
            thread.start()
            send_envelope(
                right,
                request_header("single-response-encode"),
                {"feature": np.zeros((1024,), dtype=np.float32)},
            )
            response, _, _ = recv_envelope(right)
            right.close()
            thread.join(timeout=2)

        self.assertFalse(errors)
        self.assertEqual("response", response["kind"])
        self.assertEqual(1, prepare.call_count)

    def test_trace_describes_request_feature_payload_instead_of_model_outputs(self):
        request = request_header("trace-request-payload")
        request_bundle, _ = encode_tensor_bundle(
            {"spatial_features": np.zeros((2, 64, 8, 8), dtype=np.float32)}
        )
        output_bundle, _ = encode_tensor_bundle(
            {"cls_preds": np.zeros((1, 2, 8, 8), dtype=np.float32)}
        )
        response = {
            "tensor_bundle": output_bundle,
            "durations_ns": {},
            "edge_timestamps_ns": {},
            "model": {},
            "device": {},
        }
        measurement = {
            "request_send_start_ns": 1_000,
            "request_send_end_ns": 1_100,
            "response_receive_ns": 2_000,
            "round_trip_ns": 1_000,
            "request_wire_bytes": 500,
            "response_wire_bytes": 200,
            "request_tensor_bundle": request_bundle,
        }

        trace = build_round_trip_trace(request, response, measurement)

        self.assertEqual(request_bundle["raw_payload_bytes"], trace["message"]["payload_bytes"])
        self.assertEqual([2, 64, 8, 8], trace["message"]["tensor_shape"])
        self.assertEqual("float32", trace["message"]["tensor_dtype"])
        self.assertEqual(
            measurement["request_send_end_ns"], trace["timestamps_ns"]["publish"]
        )

    def test_wire_bytes_round_trip_supports_ros2_byte_messages(self):
        header = request_header("ros-req-001")
        arrays = {"feature": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

        wire = build_wire_bytes(header, arrays, compression="zlib")
        decoded_header, decoded_arrays, wire_bytes = decode_wire_bytes(wire)

        self.assertEqual(len(wire), wire_bytes)
        self.assertEqual("ros-req-001", decoded_header["request_id"])
        np.testing.assert_array_equal(arrays["feature"], decoded_arrays["feature"])

    def test_tensor_bundle_round_trip_is_exact_and_does_not_mutate_inputs(self):
        arrays = {
            "spatial_features": np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4),
            "t_ego": np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
        }
        before = {name: value.copy() for name, value in arrays.items()}

        metadata, payload = encode_tensor_bundle(arrays, compression="zlib")
        decoded = decode_tensor_bundle(metadata, payload)

        self.assertEqual("zlib", metadata["compression"])
        self.assertLess(len(payload), metadata["raw_payload_bytes"])
        for name in arrays:
            np.testing.assert_array_equal(before[name], arrays[name])
            np.testing.assert_array_equal(arrays[name], decoded[name])
            self.assertTrue(decoded[name].flags.writeable)

    def test_tensor_decoder_rejects_hash_mismatch_and_oversized_payload(self):
        arrays = {"feature": np.arange(16, dtype=np.float32)}
        metadata, payload = encode_tensor_bundle(arrays)
        corrupt = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        with self.assertRaisesRegex(ProtocolError, "sha256"):
            decode_tensor_bundle(metadata, corrupt)

        limits = ProtocolLimits(max_payload_bytes=8)
        with self.assertRaisesRegex(ProtocolError, "payload"):
            encode_tensor_bundle(arrays, limits=limits)

    def test_zlib_decoder_streams_with_a_raw_size_bound(self):
        arrays = {"feature": np.zeros((4096,), dtype=np.float32)}
        metadata, payload = encode_tensor_bundle(arrays, compression="zlib")
        with mock.patch(
            "ditl_edge_protocol.zlib.decompress",
            side_effect=AssertionError("unbounded one-shot decompression"),
        ):
            decoded = decode_tensor_bundle(metadata, payload)
        np.testing.assert_array_equal(arrays["feature"], decoded["feature"])

        limits = ProtocolLimits(
            max_payload_bytes=len(payload),
            max_raw_payload_bytes=1024,
        )
        with self.assertRaisesRegex(ProtocolError, "raw payload"):
            decode_tensor_bundle(metadata, payload, limits=limits)

    def test_socket_envelope_handles_partial_reads_and_rejects_invalid_identity(self):
        left, right = socket.socketpair()
        try:
            header = request_header()
            arrays = {"feature": np.arange(32, dtype=np.float16)}
            sender = threading.Thread(
                target=send_envelope,
                args=(left, header, arrays),
                kwargs={"compression": "none", "chunk_bytes": 3},
            )
            sender.start()
            received_header, received_arrays, wire_bytes = recv_envelope(right)
            sender.join(timeout=2)
            self.assertFalse(sender.is_alive())
            self.assertEqual("req-001", received_header["request_id"])
            self.assertGreater(wire_bytes, arrays["feature"].nbytes)
            np.testing.assert_array_equal(arrays["feature"], received_arrays["feature"])
        finally:
            left.close()
            right.close()

        invalid = request_header(request_id="../unsafe")
        left, right = socket.socketpair()
        try:
            with self.assertRaisesRegex(ProtocolError, "request_id"):
                send_envelope(left, invalid, {})
        finally:
            left.close()
            right.close()

    def test_replay_backend_request_response_and_trace_are_contract_valid(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        backend = ReplayBackend(scale=2.0, model_id="replay-rsu-v1")
        errors = []

        def serve_once():
            try:
                connection, address = server.accept()
                with connection:
                    handle_connection(connection, address, backend)
            except Exception as error:  # pragma: no cover - surfaced below
                errors.append(error)
            finally:
                server.close()

        thread = threading.Thread(target=serve_once)
        thread.start()
        inputs = {"spatial_features": np.ones((1, 2, 3), dtype=np.float32)}
        response, outputs, measurement = request_inference(
            "127.0.0.1", port, request_header(), inputs, timeout_s=3.0
        )
        thread.join(timeout=3)

        self.assertFalse(errors)
        self.assertFalse(thread.is_alive())
        self.assertEqual("replay-rsu-v1", response["model"]["model_id"])
        np.testing.assert_array_equal(inputs["spatial_features"] * 2.0,
                                      outputs["spatial_features"])
        self.assertGreater(measurement["round_trip_ns"], 0)
        trace = build_round_trip_trace(
            request_header(), response, measurement, observer_host_id="server-01"
        )
        normalized = validate_trace_record(trace)
        self.assertEqual("tcp_round_trip_probe", normalized["network"]["transport"])
        self.assertEqual(
            measurement["round_trip_ns"], normalized["derived_ns"]["age_at_fusion"]
        )

    def test_service_rejects_unknown_operation_without_running_backend(self):
        backend = ReplayBackend(scale=2.0)
        left, right = socket.socketpair()
        errors = []

        def serve():
            try:
                handle_connection(left, ("local", 0), backend)
            except Exception as error:
                errors.append(error)
            finally:
                left.close()

        thread = threading.Thread(target=serve)
        thread.start()
        header = request_header()
        header["operation"] = "load_arbitrary_engine"
        send_envelope(right, header, {})
        response, arrays, _ = recv_envelope(right)
        right.close()
        thread.join(timeout=2)

        self.assertFalse(errors)
        self.assertEqual("error", response["kind"])
        self.assertEqual("unsupported_operation", response["error_code"])
        self.assertEqual({}, arrays)
        self.assertEqual(0, backend.inference_count)


if __name__ == "__main__":
    unittest.main()
