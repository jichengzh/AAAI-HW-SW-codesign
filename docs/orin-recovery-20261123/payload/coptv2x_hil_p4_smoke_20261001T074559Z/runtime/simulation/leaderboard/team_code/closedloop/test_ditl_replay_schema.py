from __future__ import print_function

import copy
import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

from ditl_replay_schema import (
    NORMALIZER_NAME,
    NORMALIZER_VERSION,
    REPLAY_SAMPLE_SCHEMA_VERSION,
    STAGE_NAMES,
    NormalizationResult,
    ReplaySample,
    ReplaySchemaError,
    normalize_raw_trace,
    parse_replay_sample,
    replay_sample_to_dict,
)
from ditl_trace_schema import validate_trace_sequence
from test_ditl_trace_schema import valid_record, valid_v2_record


def raw_record(sequence_id=0, warmup=False):
    record = valid_v2_record(sequence_id=sequence_id)
    record["warmup"] = warmup
    record["network"]["remote_input_tensor_bytes"] = 0
    record["model"]["engine_sha256"] = "a" * 64
    return record


def replay_record():
    return {
        "schema_version": "ditl.replay-sample.v1",
        "sample_id": "sample-001",
        "ordinal": 3,
        "source_trace_id": "trace-001",
        "source_sequence_id": 23,
        "base_delay_ns": 34_000_000,
        "stage_ns": {
            "acquisition_preprocess": 3_000_000,
            "gpu_inference": 1_000_000,
            "packing_postprocess": 6_000_000,
            "transport_receive": 24_000_000,
        },
        "raw_payload_bytes": 2_621_440,
        "wire_payload_bytes": 2_621_440,
        "capture_condition_id": "raw_samehost_idle",
    }


class DitlReplaySchemaTests(unittest.TestCase):
    def test_public_constants_define_the_replay_contract(self):
        self.assertEqual("ditl.replay-sample.v1", REPLAY_SAMPLE_SCHEMA_VERSION)
        self.assertEqual("raw-rsu-to-fusion", NORMALIZER_NAME)
        self.assertEqual("1.0.0", NORMALIZER_VERSION)
        self.assertEqual(
            (
                "acquisition_preprocess",
                "gpu_inference",
                "packing_postprocess",
                "transport_receive",
            ),
            STAGE_NAMES,
        )

    def test_parse_round_trip_copies_and_freezes_stage_mapping(self):
        source = replay_record()
        original = copy.deepcopy(source)

        sample = parse_replay_sample(source)
        source["stage_ns"]["gpu_inference"] = 99

        self.assertEqual(1_000_000, sample.stage_ns["gpu_inference"])
        with self.assertRaises(TypeError):
            sample.stage_ns["gpu_inference"] = 99
        self.assertEqual(original, replay_sample_to_dict(sample))

    def test_direct_replay_sample_copies_and_freezes_stage_mapping(self):
        stages = replay_record()["stage_ns"]
        sample = ReplaySample(
            sample_id="sample-001",
            ordinal=3,
            source_trace_id="trace-001",
            source_sequence_id=23,
            base_delay_ns=34_000_000,
            stage_ns=stages,
            raw_payload_bytes=2_621_440,
            wire_payload_bytes=2_621_440,
            capture_condition_id="raw_samehost_idle",
        )

        stages["gpu_inference"] = 99

        self.assertEqual(1_000_000, sample.stage_ns["gpu_inference"])
        with self.assertRaises(TypeError):
            sample.stage_ns["gpu_inference"] = 99

    def test_direct_normalization_result_copies_and_freezes_identity(self):
        identity = {"run_id": "run-001"}
        result = NormalizationResult(
            samples=(),
            total_count=0,
            warmup_count=0,
            rejected_count=0,
            accepted_count=0,
            identity=identity,
        )

        identity["run_id"] = "changed"

        self.assertEqual("run-001", result.identity["run_id"])
        with self.assertRaises(TypeError):
            result.identity["run_id"] = "changed"

    def test_direct_normalization_result_normalizes_samples_to_tuple(self):
        sample = parse_replay_sample(replay_record())

        result = NormalizationResult(
            samples=[sample],
            total_count=1,
            warmup_count=0,
            rejected_count=0,
            accepted_count=1,
            identity={"run_id": "run-001"},
        )

        self.assertEqual((sample,), result.samples)
        self.assertIsInstance(result.samples, tuple)

    def test_parse_rejects_unknown_and_missing_fields(self):
        cases = []
        unknown = replay_record()
        unknown["unexpected"] = True
        cases.append(("unknown", unknown, "unexpected"))
        missing = replay_record()
        del missing["sample_id"]
        cases.append(("missing", missing, "sample_id"))

        for name, value, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ReplaySchemaError, message):
                    parse_replay_sample(value)

    def test_parse_rejects_boolean_negative_and_non_finite_numbers(self):
        cases = []
        for field in (
            "ordinal",
            "source_sequence_id",
            "base_delay_ns",
            "raw_payload_bytes",
            "wire_payload_bytes",
        ):
            boolean = replay_record()
            boolean[field] = True
            cases.append(("boolean_%s" % field, boolean, field))

            negative = replay_record()
            negative[field] = -1
            cases.append(("negative_%s" % field, negative, field))

            non_finite = replay_record()
            non_finite[field] = math.inf
            cases.append(("non_finite_%s" % field, non_finite, field))

        boolean_stage = replay_record()
        boolean_stage["stage_ns"]["gpu_inference"] = False
        cases.append(("boolean_stage", boolean_stage, "gpu_inference"))
        negative_stage = replay_record()
        negative_stage["stage_ns"]["gpu_inference"] = -1
        cases.append(("negative_stage", negative_stage, "gpu_inference"))
        non_finite_stage = replay_record()
        non_finite_stage["stage_ns"]["gpu_inference"] = math.nan
        cases.append(("non_finite_stage", non_finite_stage, "gpu_inference"))

        for name, value, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ReplaySchemaError, message):
                    parse_replay_sample(value)

    def test_parse_rejects_invalid_stage_keys_and_stage_sum_mismatch(self):
        cases = []
        missing_stage = replay_record()
        del missing_stage["stage_ns"]["transport_receive"]
        cases.append(("missing_stage", missing_stage, "transport_receive"))
        extra_stage = replay_record()
        extra_stage["stage_ns"]["input_validate"] = 0
        cases.append(("extra_stage", extra_stage, "input_validate"))
        bad_sum = replay_record()
        bad_sum["stage_ns"]["transport_receive"] -= 1
        cases.append(("stage_sum", bad_sum, "base_delay_ns"))

        for name, value, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ReplaySchemaError, message):
                    parse_replay_sample(value)

    def test_normalize_raw_trace_groups_stages_without_mutating_input(self):
        records = [raw_record(sequence_id=7)]
        original = copy.deepcopy(records)

        result = normalize_raw_trace(records, "raw_samehost_idle")
        sample = result.samples[0]

        self.assertEqual(original, records)
        self.assertEqual(7, sample.source_sequence_id)
        self.assertEqual(sample.base_delay_ns, sum(sample.stage_ns.values()))
        self.assertEqual(set(STAGE_NAMES), set(sample.stage_ns))
        self.assertEqual(3_000_000, sample.stage_ns["acquisition_preprocess"])
        self.assertEqual(1_000_000, sample.stage_ns["gpu_inference"])
        self.assertEqual(6_000_000, sample.stage_ns["packing_postprocess"])
        self.assertEqual(24_000_000, sample.stage_ns["transport_receive"])
        self.assertNotIn("input_validate", sample.stage_ns)
        self.assertEqual(34_000_000, sample.base_delay_ns)
        self.assertEqual(2_621_440, sample.raw_payload_bytes)
        self.assertEqual(2_621_440, sample.wire_payload_bytes)

    def test_normalize_raw_trace_produces_stable_ids_without_frame_or_time_inputs(self):
        first = raw_record(sequence_id=7)
        changed_capture_fields = copy.deepcopy(first)
        changed_capture_fields["message"]["frame_id"] = 999
        changed_capture_fields["timestamps_ns"] = {
            name: value + 500_000_000
            for name, value in changed_capture_fields["timestamps_ns"].items()
        }

        first_sample = normalize_raw_trace(
            [first], "raw_samehost_idle"
        ).samples[0]
        repeated_sample = normalize_raw_trace(
            [copy.deepcopy(first)], "raw_samehost_idle"
        ).samples[0]
        changed_sample = normalize_raw_trace(
            [changed_capture_fields], "raw_samehost_idle"
        ).samples[0]

        self.assertEqual(first_sample.sample_id, repeated_sample.sample_id)
        self.assertEqual(first_sample.sample_id, changed_sample.sample_id)
        self.assertEqual(
            replay_sample_to_dict(first_sample), replay_sample_to_dict(changed_sample)
        )
        different_condition = normalize_raw_trace([first], "loaded").samples[0]
        self.assertNotEqual(first_sample.sample_id, different_condition.sample_id)

    def test_normalize_raw_trace_excludes_warmup_and_preserves_measured_order(self):
        records = [raw_record(index, warmup=index < 20) for index in range(120)]

        result = normalize_raw_trace(records, "raw_samehost_idle")

        self.assertEqual(120, result.total_count)
        self.assertEqual(20, result.warmup_count)
        self.assertEqual(20, result.rejected_count)
        self.assertEqual(100, result.accepted_count)
        self.assertEqual(tuple(range(20, 120)), tuple(
            sample.source_sequence_id for sample in result.samples
        ))
        self.assertEqual(tuple(range(100)), tuple(
            sample.ordinal for sample in result.samples
        ))

    def test_normalize_raw_trace_copies_and_freezes_result_mappings(self):
        result = normalize_raw_trace([raw_record()], "raw_samehost_idle")

        self.assertEqual("rsu-orin-01", result.identity["message.source_id"])
        with self.assertRaises(TypeError):
            result.identity["run_id"] = "changed"
        with self.assertRaises(TypeError):
            result.samples[0].stage_ns["gpu_inference"] = 0

    def test_normalize_raw_trace_rejects_mixed_profile_identity(self):
        first = raw_record(sequence_id=1)
        second = raw_record(sequence_id=2)
        second["model"]["engine_sha256"] = "1" * 64
        with self.assertRaisesRegex(ReplaySchemaError, "engine_sha256"):
            normalize_raw_trace([first, second], "raw_samehost_idle")

    def test_normalize_raw_trace_rejects_mixed_message_source_id(self):
        first = raw_record(sequence_id=1)
        second = raw_record(sequence_id=2)
        second["message"]["source_id"] = "rsu-orin-02"

        with self.assertRaisesRegex(ReplaySchemaError, "message.source_id"):
            normalize_raw_trace([first, second], "raw_samehost_idle")

    def test_normalize_raw_trace_rejects_every_eligibility_violation(self):
        cases = []

        cases.append(("empty", [], "empty"))
        v1 = valid_record(sequence_id=0)
        cases.append(("v1_schema", [v1], "schema_version"))

        wrong_scope = raw_record()
        wrong_scope["measurement_scope"] = "testbed_probe"
        wrong_scope["headline_eligible"] = False
        wrong_scope["headline_ineligible_reason"] = "testbed_probe"
        cases.append(("measurement_scope", [wrong_scope], "measurement_scope"))

        ineligible = raw_record()
        ineligible["headline_eligible"] = False
        ineligible["headline_ineligible_reason"] = "diagnostic"
        cases.append(("headline_eligible", [ineligible], "headline_eligible"))

        wrong_outcome = raw_record()
        wrong_outcome["status"]["outcome"] = "delivered"
        cases.append(("outcome", [wrong_outcome], "status.outcome"))

        remote_input = raw_record()
        remote_input["network"]["remote_input_tensor_bytes"] = 4
        cases.append(
            ("remote_input", [remote_input], "remote_input_tensor_bytes")
        )

        compressed = raw_record()
        compressed["codec"]["name"] = "zlib"
        cases.append(("codec", [compressed], "codec.name"))

        incomparable = raw_record()
        incomparable["clock"]["receiver"]["boot_id"] = "different-boot"
        incomparable["clock_comparability"] = {
            "comparable": False,
            "reason": "same_host_boot_mismatch",
        }
        incomparable["headline_eligible"] = False
        incomparable["headline_ineligible_reason"] = "clock_not_comparable"
        cases.append(
            ("incomparable_clocks", [incomparable], "clock_comparability")
        )

        duplicate_first = raw_record(sequence_id=4)
        duplicate_second = raw_record(sequence_id=4)
        duplicate_second["trace_id"] = "v2-duplicate"
        duplicate_second["status"]["duplicate"] = True
        cases.append(
            (
                "duplicate_sequence",
                [duplicate_first, duplicate_second],
                "sequence_id",
            )
        )

        ordered_first = raw_record(sequence_id=4)
        decreasing = raw_record(sequence_id=3)
        decreasing["status"]["out_of_order"] = True
        cases.append(
            ("decreasing_sequence", [ordered_first, decreasing], "sequence_id")
        )

        for name, records, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ReplaySchemaError, message):
                    normalize_raw_trace(records, "raw_samehost_idle")

    def test_normalize_raw_trace_rejects_missing_and_negative_derived_total(self):
        validated = validate_trace_sequence([raw_record()])[0]
        cases = []
        missing = copy.deepcopy(validated)
        del missing["derived_ns"]["rsu_to_fusion"]
        cases.append(("missing", missing))
        negative = copy.deepcopy(validated)
        negative["derived_ns"]["rsu_to_fusion"] = -1
        cases.append(("negative", negative))

        for name, normalized in cases:
            with self.subTest(name=name):
                with mock.patch(
                    "ditl_replay_schema.validate_trace_sequence",
                    return_value=[normalized],
                ):
                    with self.assertRaisesRegex(
                        ReplaySchemaError, "derived_ns.rsu_to_fusion"
                    ):
                        normalize_raw_trace([raw_record()], "raw_samehost_idle")

    def test_normalize_raw_trace_rejects_negative_transport_residual(self):
        record = raw_record()
        record["durations_ns"]["source_read"] = 20_000_000
        record["durations_ns"]["preprocess"] = 20_000_000

        with self.assertRaisesRegex(ReplaySchemaError, "transport_receive"):
            normalize_raw_trace([record], "raw_samehost_idle")

    def test_normalize_raw_trace_rejects_mixed_identity_dimensions(self):
        mutations = (
            ("run_id", lambda record: record.__setitem__("run_id", "other-run")),
            (
                "model_id",
                lambda record: record["model"].__setitem__("model_id", "other-model"),
            ),
            (
                "engine_sha256",
                lambda record: record["model"].__setitem__(
                    "engine_sha256", "b" * 64
                ),
            ),
            (
                "source.type",
                lambda record: record["source"].__setitem__(
                    "type", "other-source"
                ),
            ),
            (
                "payload_semantics",
                lambda record: record["source"].__setitem__(
                    "payload_semantics", "other-payload"
                ),
            ),
        )

        for name, mutate in mutations:
            first = raw_record(sequence_id=1)
            second = raw_record(sequence_id=2)
            mutate(second)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ReplaySchemaError, name):
                    normalize_raw_trace([first, second], "raw_samehost_idle")

    def test_warmup_records_are_validated_for_schema_and_identity(self):
        warmup = raw_record(sequence_id=0, warmup=True)
        measured = raw_record(sequence_id=1)
        warmup["model"]["model_id"] = "other-model"

        with self.assertRaisesRegex(ReplaySchemaError, "model_id"):
            normalize_raw_trace([warmup, measured], "raw_samehost_idle")


if __name__ == "__main__":
    unittest.main()
