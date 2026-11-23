from __future__ import print_function

import copy
import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from ditl_trace_schema import (
    DITL_LATENCY_CSV_COLUMNS,
    TraceValidationError,
    read_trace_jsonl,
    validate_trace_record,
    validate_trace_sequence,
    write_latency_csv,
    write_trace_jsonl,
)


def valid_record(sequence_id=7, outcome="consumed"):
    timestamps = {
        "source": 1_000_000_000,
        "inference_start": 1_000_100_000,
        "inference_end": 1_001_100_000,
        "compress_start": 1_001_200_000,
        "compress_end": 1_001_400_000,
        "d2h_start": 1_001_500_000,
        "d2h_end": 1_001_700_000,
        "serialize_start": 1_001_800_000,
        "serialize_end": 1_001_900_000,
        "publish": 1_002_000_000,
        "receive": 1_005_000_000,
        "deserialize_start": 1_005_100_000,
        "deserialize_end": 1_005_300_000,
        "h2d_start": 1_005_400_000,
        "h2d_end": 1_005_600_000,
        "decompress_start": 1_005_700_000,
        "decompress_end": 1_006_000_000,
        "fusion_consume": 1_007_000_000,
    }
    if outcome == "dropped":
        for key in list(timestamps):
            if key in {
                "receive",
                "deserialize_start",
                "deserialize_end",
                "h2d_start",
                "h2d_end",
                "decompress_start",
                "decompress_end",
                "fusion_consume",
            }:
                timestamps[key] = None
    elif outcome in {"delivered", "expired"}:
        timestamps["fusion_consume"] = None

    return {
        "schema_version": "ditl.trace.v1",
        "trace_id": "trace-001",
        "run_id": "run-001",
        "route_id": "route-146",
        "scenario_id": "scenario-1",
        "message": {
            "source_id": "rsu-01",
            "target_id": "ego-01",
            "frame_id": 42,
            "sequence_id": sequence_id,
            "topic": "/ditl/rsu/features",
            "message_type": "intermediate_feature",
            "payload_bytes": 840000,
            "tensor_shape": [1, 128, 96, 288],
            "tensor_dtype": "float16",
            "compression_ratio": 32.0,
        },
        "clock": {
            "sender": {
                "host_id": "orin-agx-01",
                "domain": "ptp",
                "grandmaster_id": "gm-001",
                "sync_state": "locked",
                "offset_ns": 100,
                "uncertainty_ns": 200,
            },
            "receiver": {
                "host_id": "server-01",
                "domain": "ptp",
                "grandmaster_id": "gm-001",
                "sync_state": "locked",
                "offset_ns": -150,
                "uncertainty_ns": 250,
            },
        },
        "status": {
            "outcome": outcome,
            "duplicate": False,
            "out_of_order": False,
            "reason": None,
        },
        "timestamps_ns": timestamps,
        "network": {
            "transport": "tcp",
            "qos": "best_effort",
            "sender_count": sequence_id + 1,
            "receiver_count": sequence_id + (0 if outcome == "dropped" else 1),
        },
        "durations_ns": {
            "gpu_inference": 1_000_000,
            "gpu_compress": 200_000,
        },
    }


def valid_v2_record(sequence_id=0, same_host=True):
    sender_host = "orin-agx-01"
    receiver_host = sender_host if same_host else "server-01"
    sender_clock = {
        "host_id": sender_host,
        "domain": "monotonic_raw" if same_host else "ptp",
        "boot_id": "boot-orin" if same_host else None,
        "sync_state": "locked",
        "offset_ns": 0,
        "uncertainty_ns": 0,
    }
    receiver_clock = {
        "host_id": receiver_host,
        "domain": "monotonic_raw" if same_host else "ptp",
        "boot_id": "boot-orin" if same_host else None,
        "sync_state": "locked",
        "offset_ns": 0,
        "uncertainty_ns": 0,
    }
    if not same_host:
        sender_clock["grandmaster_id"] = "gm-001"
        receiver_clock["grandmaster_id"] = "gm-001"
        sender_clock.pop("boot_id")
        receiver_clock.pop("boot_id")
    names = (
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
        "ros_message_start",
        "ros_message_end",
        "ros_publish_start",
        "ros_publish_end",
        "ros_receive",
        "frame_parse_start",
        "frame_parse_end",
        "decompress_start",
        "decompress_end",
        "verify_start",
        "verify_end",
        "reconstruct_start",
        "reconstruct_end",
        "fusion_ready",
    )
    timestamps = {
        name: 1_000_000_000 + index * 1_000_000
        for index, name in enumerate(names)
    }
    durations = {
        "source_read": 1_000_000,
        "preprocess": 1_000_000,
        "h2d": 1_000_000,
        "gpu_inference": 1_000_000,
        "d2h": 1_000_000,
        "postprocess": 1_000_000,
        "pack": 1_000_000,
        "hash": 1_000_000,
        "compress": 1_000_000,
        "protocol_serialize": 1_000_000,
        "ros_message": 1_000_000,
        "ros_publish_call": 1_000_000,
        "frame_parse": 1_000_000,
        "decompress": 1_000_000,
        "verify": 1_000_000,
        "reconstruct": 1_000_000,
    }
    return {
        "schema_version": "ditl.trace.v2",
        "trace_id": "v2-%d" % sequence_id,
        "run_id": "rsu-local-run",
        "route_id": "tensor-replay",
        "scenario_id": "edge-rsu",
        "measurement_scope": "rsu_to_fusion",
        "headline_eligible": True,
        "headline_ineligible_reason": None,
        "source": {
            "type": "local_feature_replay",
            "payload_semantics": "dense_head_debug",
            "postprocess": "identity",
        },
        "message": {
            "source_id": "rsu-orin-01",
            "target_id": "ego-server-01",
            "frame_id": sequence_id,
            "sequence_id": sequence_id,
            "topic": "/ditl/rsu/perception_result",
            "message_type": "dense_perception_heads",
            "payload_bytes": 2_621_440,
            "tensor_shape": [1, 14, 128, 256],
            "tensor_dtype": "float32",
            "compression_ratio": 1.0,
        },
        "clock": {"sender": sender_clock, "receiver": receiver_clock},
        "clock_comparability": {
            "comparable": True,
            "reason": "same_host_monotonic" if same_host else "ptp_locked",
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
        },
        "codec": {
            "name": "none",
            "raw_payload_bytes": 2_621_440,
            "wire_payload_bytes": 2_621_440,
        },
        "durations_ns": durations,
        "testbed_overhead_ns": {"control": 0, "startup": 0},
        "model": {"backend": "tensorrt", "model_id": "rsu-edge"},
        "device": {"hostname": "orin-agx-01"},
    }


class DitlTraceSchemaTests(unittest.TestCase):
    def test_same_host_v2_derives_one_way_latency_without_rtt_field(self):
        normalized = validate_trace_record(valid_v2_record())

        self.assertIn("rsu_device_pipeline", normalized["derived_ns"])
        self.assertIn("rsu_to_fusion", normalized["derived_ns"])
        self.assertNotIn("round_trip", normalized["derived_ns"])
        self.assertEqual(
            normalized["timestamps_ns"]["fusion_ready"]
            - normalized["timestamps_ns"]["rsu_data_available"],
            normalized["derived_ns"]["rsu_to_fusion"],
        )

    def test_cross_host_v2_without_comparable_ptp_is_diagnostic_only(self):
        record = valid_v2_record(same_host=False)
        record["clock"]["receiver"]["sync_state"] = "unlocked"
        record["clock_comparability"] = {
            "comparable": False,
            "reason": "clock_not_comparable",
        }
        record["headline_eligible"] = False
        record["headline_ineligible_reason"] = "clock_not_comparable"

        normalized = validate_trace_record(record)

        self.assertIn("rsu_device_pipeline", normalized["derived_ns"])
        self.assertIn("receiver_decode", normalized["derived_ns"])
        self.assertNotIn("rsu_to_fusion", normalized["derived_ns"])
        self.assertNotIn("ros_and_network", normalized["derived_ns"])

    def test_v2_headline_claim_requires_comparable_clock(self):
        record = valid_v2_record(same_host=False)
        record["clock"]["receiver"]["sync_state"] = "unlocked"
        record["clock_comparability"]["comparable"] = False
        record["clock_comparability"]["reason"] = "clock_not_comparable"
        with self.assertRaisesRegex(TraceValidationError, "headline_eligible"):
            validate_trace_record(record)

    def test_v2_sender_and_receiver_timeline_order_is_enforced(self):
        record = valid_v2_record()
        record["timestamps_ns"]["inference_end"] = (
            record["timestamps_ns"]["inference_start"] - 1
        )
        with self.assertRaisesRegex(TraceValidationError, "inference_start"):
            validate_trace_record(record)

    def test_valid_cross_host_ptp_record_is_normalized_without_mutating_input(self):
        source = valid_record()
        before = copy.deepcopy(source)

        normalized = validate_trace_record(source, max_clock_error_ns=1_000)

        self.assertEqual(before, source)
        self.assertIsNot(source, normalized)
        self.assertEqual(3_000_000, normalized["derived_ns"]["transport"])
        self.assertEqual(7_000_000, normalized["derived_ns"]["age_at_fusion"])

    def test_missing_required_identity_field_is_rejected(self):
        record = valid_record()
        del record["message"]["source_id"]

        with self.assertRaisesRegex(TraceValidationError, "message.source_id"):
            validate_trace_record(record)

    def test_cross_host_non_ptp_timestamps_are_rejected(self):
        record = valid_record()
        record["clock"]["sender"].update(
            {"domain": "monotonic_raw", "boot_id": "boot-a"}
        )
        record["clock"]["receiver"].update(
            {"domain": "monotonic_raw", "boot_id": "boot-b"}
        )

        with self.assertRaisesRegex(TraceValidationError, "cross-host.*PTP"):
            validate_trace_record(record)

    def test_ptp_gate_rejects_mismatched_grandmaster_and_large_offset(self):
        record = valid_record()
        record["clock"]["receiver"]["grandmaster_id"] = "gm-002"
        with self.assertRaisesRegex(TraceValidationError, "grandmaster"):
            validate_trace_record(record)

        record = valid_record()
        record["clock"]["receiver"]["offset_ns"] = 2_000
        with self.assertRaisesRegex(TraceValidationError, "clock error"):
            validate_trace_record(record, max_clock_error_ns=1_000)

    def test_outcome_controls_receive_and_consume_timestamp_requirements(self):
        dropped = validate_trace_record(valid_record(outcome="dropped"))
        self.assertIsNone(dropped["timestamps_ns"]["receive"])

        broken_drop = valid_record(outcome="dropped")
        broken_drop["timestamps_ns"]["receive"] = 1_005_000_000
        with self.assertRaisesRegex(TraceValidationError, "dropped.*receive"):
            validate_trace_record(broken_drop)

        broken_consumed = valid_record(outcome="consumed")
        broken_consumed["timestamps_ns"]["fusion_consume"] = None
        with self.assertRaisesRegex(TraceValidationError, "consumed.*fusion_consume"):
            validate_trace_record(broken_consumed)

    def test_causal_and_stage_timestamp_order_is_enforced(self):
        record = valid_record()
        record["timestamps_ns"]["receive"] = 1_001_000_000
        with self.assertRaisesRegex(TraceValidationError, "publish.*receive"):
            validate_trace_record(record)

        record = valid_record()
        record["timestamps_ns"]["compress_end"] = 1_001_100_000
        with self.assertRaisesRegex(TraceValidationError, "compress_start"):
            validate_trace_record(record)

    def test_sequence_requires_explicit_duplicate_and_out_of_order_flags(self):
        first = valid_record(sequence_id=10)
        duplicate = valid_record(sequence_id=10)
        duplicate["trace_id"] = "trace-duplicate"
        duplicate["status"]["duplicate"] = True
        late = valid_record(sequence_id=9)
        late["trace_id"] = "trace-late"
        late["status"]["out_of_order"] = True

        normalized = validate_trace_sequence([first, duplicate, late])
        self.assertEqual([10, 10, 9], [r["message"]["sequence_id"] for r in normalized])

        unmarked_duplicate = copy.deepcopy(duplicate)
        unmarked_duplicate["status"]["duplicate"] = False
        with self.assertRaisesRegex(TraceValidationError, "duplicate"):
            validate_trace_sequence([first, unmarked_duplicate])

        unmarked_late = copy.deepcopy(late)
        unmarked_late["status"]["out_of_order"] = False
        with self.assertRaisesRegex(TraceValidationError, "out_of_order"):
            validate_trace_sequence([first, unmarked_late])

    def test_jsonl_round_trip_and_legacy_latency_csv_use_validated_records(self):
        consumed = valid_record(sequence_id=0)
        dropped = valid_record(sequence_id=1, outcome="dropped")
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "trace.jsonl")
            csv_path = os.path.join(tmpdir, "latency.csv")

            write_trace_jsonl(jsonl_path, [consumed, dropped])
            loaded = read_trace_jsonl(jsonl_path)
            exported = write_latency_csv(csv_path, loaded)

            self.assertEqual(2, len(loaded))
            self.assertEqual(1, exported)
            with open(csv_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(DITL_LATENCY_CSV_COLUMNS), list(rows[0].keys()))
            self.assertEqual("0", rows[0]["call_idx"])
            self.assertEqual("7.000000", rows[0]["tau_ms"])
            self.assertEqual("ditl_orin_transport", rows[0]["latency_ms_source"])

    def test_v2_latency_csv_exports_only_headline_eligible_records(self):
        eligible = valid_v2_record(sequence_id=0)
        diagnostic = valid_v2_record(sequence_id=1)
        diagnostic["measurement_scope"] = "testbed_probe"
        diagnostic["headline_eligible"] = False
        diagnostic["headline_ineligible_reason"] = "testbed_probe"
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "latency.csv")

            exported = write_latency_csv(csv_path, [eligible, diagnostic])

            self.assertEqual(1, exported)
            with open(csv_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual("ditl_rsu_to_fusion_v2", rows[0]["latency_ms_source"])
            self.assertEqual("34.000000", rows[0]["tau_ms"])

    def test_jsonl_reader_reports_line_number_for_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "broken.jsonl")
            with open(path, "w") as handle:
                handle.write(json.dumps(valid_record()) + "\n")
                handle.write("{}\n")

            with self.assertRaisesRegex(TraceValidationError, "line 2"):
                read_trace_jsonl(path)


if __name__ == "__main__":
    unittest.main()
