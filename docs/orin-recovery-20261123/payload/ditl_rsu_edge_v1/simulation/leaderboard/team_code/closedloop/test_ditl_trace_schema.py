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


class DitlTraceSchemaTests(unittest.TestCase):
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
