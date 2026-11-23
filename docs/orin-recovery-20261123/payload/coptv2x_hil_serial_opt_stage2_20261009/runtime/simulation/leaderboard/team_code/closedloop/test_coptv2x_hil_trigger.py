from __future__ import print_function

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np


CLOSEDLOOP = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
EDGE_PUBLISHER = REPO_ROOT / "scripts" / "ditl_rsu_edge_publisher.py"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_trigger import (
    HIL_TRIGGER_SCHEMA_VERSION,
    HilTrigger,
    HilTriggerAdmission,
    HilTriggerError,
    parse_hil_trigger,
)
import ditl_rsu_source as source_module
from ditl_rsu_source import IndexedLocalRsuSource


LOCAL_MANIFEST_SCHEMA_VERSION = "coptv2x.local-input-manifest.v1"
RUN_ID = "run-20260930-001"
ROUTE_ID = "Town05-route-17"
SCENARIO_ID = "scenario-3"
ARM_ID = "joint_shcosearch"
INDEXED_SOURCE_LIMITS = {
    "max_samples": 100,
    "max_total_tensor_bytes": 1024 * 1024,
}


def load_edge_publisher_module():
    spec = importlib.util.spec_from_file_location(
        "ditl_rsu_edge_publisher", str(EDGE_PUBLISHER)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_payload(sequence_no=0, sample_id="sample-000", source_frame=0):
    return {
        "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "route_id": ROUTE_ID,
        "scenario_id": SCENARIO_ID,
        "sample_id": sample_id,
        "sequence_no": sequence_no,
        "source_frame": source_frame,
        "source_sim_time_ns": source_frame * 50_000_000,
        "arm_id": ARM_ID,
    }


@dataclass(frozen=True)
class FakeResult:
    header: object


class FakePipeline(object):
    def __init__(self):
        self.frames = []

    def process(self, frame):
        self.frames.append(frame)
        return FakeResult(
            header=MappingProxyType(
                {
                    "kind": "result",
                    "run_id": RUN_ID,
                    "route_id": ROUTE_ID,
                    "scenario_id": SCENARIO_ID,
                    "sequence_id": frame.sequence_id,
                }
            )
        )


class LocalInputTree(object):
    def __init__(self, root, sample_count=2):
        self.root = Path(root)
        self.contract = {
            "spatial_features": {"shape": (2, 3), "dtype": "float32"}
        }
        samples = []
        for index in range(sample_count):
            path = self.root / ("sample-%03d.npy" % index)
            array = np.arange(6, dtype=np.float32).reshape(2, 3) + index * 10
            np.save(str(path), array)
            samples.append(
                {
                    "sample_id": "sample-%03d" % index,
                    "source_frame": index,
                    "inputs": {
                        "spatial_features": {
                            "path": path.name,
                            "sha256": sha256(path),
                            "shape": [2, 3],
                            "dtype": "float32",
                        }
                    },
                }
            )
        self.manifest = {
            "schema_version": LOCAL_MANIFEST_SCHEMA_VERSION,
            "route_id": ROUTE_ID,
            "scenario_id": SCENARIO_ID,
            "batch_order": ["ego", "rsu"],
            "samples": samples,
        }

    def build_source(self):
        return IndexedLocalRsuSource(
            data_root=self.root,
            manifest=self.manifest,
            input_contract=self.contract,
            source_id="rsu-orin-01",
            **INDEXED_SOURCE_LIMITS
        )


class HilTriggerContractTests(unittest.TestCase):
    def test_package_and_top_level_imports_share_the_same_trigger_class(self):
        leaderboard_root = str(CLOSEDLOOP.parents[1])
        if leaderboard_root not in sys.path:
            sys.path.insert(0, leaderboard_root)
        package_module = importlib.import_module(
            "team_code.closedloop.coptv2x_hil_trigger"
        )
        self.assertIs(package_module.HilTrigger, HilTrigger)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tree = LocalInputTree(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def admission(self):
        return HilTriggerAdmission(
            run_id=RUN_ID,
            route_id=ROUTE_ID,
            scenario_id=SCENARIO_ID,
            arm_id=ARM_ID,
        )

    def processor(self, source=None, pipeline=None):
        publisher = load_edge_publisher_module()
        pipeline = pipeline or FakePipeline()
        return (
            publisher.HilTriggerProcessor(
                admission=self.admission(),
                source=source or self.tree.build_source(),
                pipeline=pipeline,
            ),
            pipeline,
        )

    def test_result_wire_serialize_timer_covers_structurally_final_header(self):
        publisher = load_edge_publisher_module()
        captured_headers = []

        def serialize(header, tensors):
            captured_headers.append(copy.deepcopy(header))
            self.assertEqual({"res0"}, set(tensors))
            return b"header-%d" % len(captured_headers)

        result = SimpleNamespace(
            header={"kind": "result"},
            tensor_bundle={"res0": np.zeros((1,), dtype=np.float32)},
            timestamps_ns={},
            durations_ns={},
            payload=b"payload",
        )
        with mock.patch.object(
            publisher, "monotonic_raw_ns", side_effect=(100, 125)
        ), mock.patch.object(
            publisher, "prepare_envelope_header", side_effect=serialize
        ):
            wire_header, payload = publisher._prepare_result_frame(result, False)

        self.assertEqual(b"header-2", wire_header)
        self.assertEqual(b"payload", payload)
        self.assertEqual(2, len(captured_headers))
        first_timestamps = captured_headers[0]["edge_timestamps_ns"]
        self.assertEqual(100, first_timestamps["protocol_serialize_start"])
        self.assertEqual(100, first_timestamps["protocol_serialize_end"])
        self.assertEqual(100, first_timestamps["local_handoff_start"])
        self.assertEqual(0, captured_headers[0]["durations_ns"]["protocol_serialize"])
        final_timestamps = captured_headers[1]["edge_timestamps_ns"]
        self.assertEqual(125, final_timestamps["protocol_serialize_end"])
        self.assertEqual(25, captured_headers[1]["durations_ns"]["protocol_serialize"])
        # The returned final header is prepared after the local handoff timer
        # starts, so the bridge's local_handoff interval includes this call.
        self.assertEqual(125, final_timestamps["local_handoff_start"])

    def test_parser_accepts_only_the_exact_lightweight_identity_contract(self):
        payload = valid_payload()

        trigger = parse_hil_trigger(payload)

        self.assertIsInstance(trigger, HilTrigger)
        self.assertEqual(payload, trigger.to_metadata())
        self.assertEqual(0, trigger.sequence_no)
        self.assertEqual(ARM_ID, trigger.arm_id)

        forbidden_values = (
            b"tensor",
            bytearray(b"tensor"),
            memoryview(b"tensor"),
            np.ones((2, 3), dtype=np.float32),
            "A" * 4096,
        )
        for name, value in zip(
            ("tensor", "blob", "buffer", "array", "tensor_b64"),
            forbidden_values,
        ):
            with self.subTest(field=name):
                rejected = dict(payload)
                rejected[name] = value
                with self.assertRaisesRegex(HilTriggerError, "field|payload|unknown"):
                    parse_hil_trigger(rejected)

        for field, value in (
            ("run_id", b"run"),
            ("sample_id", "/tmp/input.npy"),
            ("sequence_no", True),
            ("source_frame", -1),
            ("source_sim_time_ns", -1),
        ):
            with self.subTest(field=field, value=value):
                rejected = dict(payload)
                rejected[field] = value
                with self.assertRaises(HilTriggerError):
                    parse_hil_trigger(rejected)

    def test_parser_rejects_missing_unknown_oversized_and_wrong_typed_fields(self):
        payload = valid_payload()
        invalid = []
        missing = dict(payload)
        missing.pop("sample_id")
        invalid.append(missing)
        unknown = dict(payload, extra="not-allowed")
        invalid.append(unknown)
        invalid.append(dict(payload, schema_version="coptv2x.hil-trigger.v0"))
        invalid.append(dict(payload, route_id="r" * 129))
        invalid.append(dict(payload, sequence_no=1.0))
        invalid.append(dict(payload, source_frame=2 ** 63))
        invalid.append(dict(payload, source_sim_time_ns=2 ** 64))
        invalid.append(dict(payload, arm_id="unknown-arm"))

        for index, rejected in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises(HilTriggerError):
                    parse_hil_trigger(rejected)

    def test_admission_rejects_duplicate_reverse_stale_and_cross_identity(self):
        admission = self.admission()
        first = admission.admit(valid_payload(sequence_no=8))
        self.assertEqual(8, first.sequence_no)

        cases = (
            dict(valid_payload(sequence_no=8)),
            dict(valid_payload(sequence_no=7)),
            dict(valid_payload(sequence_no=9), run_id="old-run"),
            dict(valid_payload(sequence_no=9), route_id="Town05-route-18"),
            dict(valid_payload(sequence_no=9), scenario_id="scenario-4"),
            dict(valid_payload(sequence_no=9), arm_id="compression_only"),
        )
        for rejected in cases:
            with self.subTest(rejected=rejected):
                with self.assertRaises(HilTriggerError):
                    admission.admit(rejected)

        self.assertEqual(9, admission.admit(valid_payload(sequence_no=9)).sequence_no)

    def test_route_manifest_is_preloaded_and_looked_up_by_exact_composite_key(self):
        source = self.tree.build_source()
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)

        # Route-start preload is observable: later disk mutation cannot alter the batch.
        (self.tree.root / "sample-000.npy").write_bytes(b"corrupted after preload")
        frame = source.frame_for_trigger(parse_hil_trigger(valid_payload()))

        np.testing.assert_array_equal(expected, frame.arrays["spatial_features"])
        self.assertFalse(frame.arrays["spatial_features"].flags.writeable)
        self.assertEqual(0, frame.frame_id)
        self.assertEqual(0, frame.sequence_id)
        self.assertEqual(valid_payload(), frame.source_metadata["trigger_identity"])
        self.assertEqual(0, frame.source_metadata["remote_input_tensor_bytes"])

        missing_cases = (
            dict(valid_payload(), sample_id="sample-missing"),
            dict(valid_payload(), source_frame=1),
            dict(valid_payload(), route_id="Town05-route-18"),
            dict(valid_payload(), scenario_id="scenario-4"),
        )
        for rejected in missing_cases:
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(ValueError, "sample|route|scenario|key"):
                    source.frame_for_trigger(parse_hil_trigger(rejected))

    def test_route_manifest_rejects_hash_shape_dtype_and_path_mismatch(self):
        cases = []
        wrong_hash = copy.deepcopy(self.tree.manifest)
        wrong_hash["samples"][0]["inputs"]["spatial_features"]["sha256"] = "0" * 64
        cases.append((wrong_hash, "hash"))
        wrong_shape = copy.deepcopy(self.tree.manifest)
        wrong_shape["samples"][0]["inputs"]["spatial_features"]["shape"] = [3, 2]
        cases.append((wrong_shape, "shape"))
        wrong_dtype = copy.deepcopy(self.tree.manifest)
        wrong_dtype["samples"][0]["inputs"]["spatial_features"]["dtype"] = "float16"
        cases.append((wrong_dtype, "dtype"))
        escaped = copy.deepcopy(self.tree.manifest)
        escaped["samples"][0]["inputs"]["spatial_features"]["path"] = "../outside.npy"
        cases.append((escaped, "root"))

        for manifest, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    IndexedLocalRsuSource(
                        self.tree.root,
                        manifest,
                        self.tree.contract,
                        **INDEXED_SOURCE_LIMITS
                    )

    def test_memory_mapped_preload_keeps_exact_tensor_backing_off_heap(self):
        source = IndexedLocalRsuSource(
            self.tree.root,
            self.tree.manifest,
            self.tree.contract,
            memory_map=True,
            **INDEXED_SOURCE_LIMITS
        )
        key = (ROUTE_ID, SCENARIO_ID, "sample-000", 0)
        self.assertIsInstance(
            source._samples[key]["spatial_features"], np.memmap
        )
        frame = source.frame_for_trigger(parse_hil_trigger(valid_payload()))
        self.assertFalse(frame.arrays["spatial_features"].flags.writeable)
        self.assertFalse(
            isinstance(frame.arrays["spatial_features"], np.memmap)
        )

    def test_lazy_memory_map_defers_loading_and_verifies_when_selected(self):
        with mock.patch.object(
            source_module, "_load_verified_indexed_npy_mmap",
            wraps=source_module._load_verified_indexed_npy_mmap,
        ) as load_input:
            source = IndexedLocalRsuSource(
                self.tree.root,
                self.tree.manifest,
                self.tree.contract,
                memory_map=True,
                lazy_load=True,
                **INDEXED_SOURCE_LIMITS
            )
            self.assertEqual(0, load_input.call_count)

            frame = source.frame_for_trigger(parse_hil_trigger(valid_payload()))

            self.assertEqual(1, load_input.call_count)
            np.testing.assert_array_equal(
                self.tree_sample(0), frame.arrays["spatial_features"]
            )

    def tree_sample(self, index):
        return np.arange(6, dtype=np.float32).reshape(2, 3) + index * 10

    def test_indexed_manifest_rejects_npz_before_member_materialization(self):
        path = self.tree.root / "compressed.npz"
        np.savez_compressed(
            str(path),
            spatial_features=np.arange(6, dtype=np.float32).reshape(2, 3),
        )
        manifest = copy.deepcopy(self.tree.manifest)
        entry = manifest["samples"][0]["inputs"]["spatial_features"]
        entry.update(
            path=path.name,
            member="spatial_features",
            sha256=sha256(path),
        )

        with self.assertRaisesRegex(ValueError, "NPY"):
            IndexedLocalRsuSource(
                self.tree.root,
                manifest,
                self.tree.contract,
                **INDEXED_SOURCE_LIMITS
            )

    def test_preload_hash_and_array_come_from_the_same_file_snapshot(self):
        path = self.tree.root / "sample-000.npy"
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected_hash = sha256(path)
        replacement = self.tree.root / "replacement.npy"
        np.save(str(replacement), expected + 1000)
        real_sha256 = source_module.hashlib.sha256

        def replace_path_after_hash(payload):
            digest = real_sha256(payload)
            path.write_bytes(replacement.read_bytes())
            return digest

        with mock.patch.object(
            source_module.hashlib,
            "sha256",
            side_effect=replace_path_after_hash,
        ):
            array = source_module._load_verified_indexed_npy(
                path,
                expected_hash,
                maximum_file_bytes=1024 * 1024,
            )

        np.testing.assert_array_equal(expected, array)
        np.testing.assert_array_equal(
            expected + 1000,
            np.load(str(path), allow_pickle=False),
        )

    def test_route_preload_rejects_aggregate_budgets_before_loading_any_tensor(self):
        with mock.patch.object(
            source_module,
            "_load_verified_indexed_npy",
            side_effect=AssertionError("budget gate must precede materialization"),
        ):
            with self.assertRaisesRegex(ValueError, "max_samples"):
                IndexedLocalRsuSource(
                    self.tree.root,
                    self.tree.manifest,
                    self.tree.contract,
                    max_samples=1,
                    max_total_tensor_bytes=48,
                )
            with self.assertRaisesRegex(ValueError, "max_total_tensor_bytes"):
                IndexedLocalRsuSource(
                    self.tree.root,
                    self.tree.manifest,
                    self.tree.contract,
                    max_samples=2,
                    max_total_tensor_bytes=47,
                )

    def test_route_preload_requires_strict_positive_startup_budgets(self):
        for field, value in (
            ("max_samples", 0),
            ("max_samples", True),
            ("max_samples", 1.0),
            ("max_total_tensor_bytes", 0),
            ("max_total_tensor_bytes", True),
            ("max_total_tensor_bytes", 48.0),
        ):
            kwargs = {"max_samples": 2, "max_total_tensor_bytes": 48}
            kwargs[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    IndexedLocalRsuSource(
                        self.tree.root,
                        self.tree.manifest,
                        self.tree.contract,
                        **kwargs
                    )

    def test_indexed_preload_rejects_symlink_swap_after_path_resolution(self):
        original = self.tree.root / "sample-000.npy"
        outside = self.tree.root.parent / "outside-identical.npy"
        outside.write_bytes(original.read_bytes())
        self.addCleanup(lambda: outside.exists() and outside.unlink())
        real_loader = source_module._load_verified_indexed_npy

        def swap_before_open(path, expected_hash, maximum_file_bytes):
            if path.name == "sample-000.npy":
                path.unlink()
                path.symlink_to(outside)
            return real_loader(path, expected_hash, maximum_file_bytes)

        with mock.patch.object(
            source_module,
            "_load_verified_indexed_npy",
            side_effect=swap_before_open,
        ):
            with self.assertRaisesRegex(ValueError, "symlink|safe regular"):
                self.tree.build_source()

    def test_route_manifest_rejects_unknown_duplicate_and_missing_samples(self):
        duplicate = copy.deepcopy(self.tree.manifest)
        duplicate["samples"].append(copy.deepcopy(duplicate["samples"][0]))
        unknown = copy.deepcopy(self.tree.manifest)
        unknown["samples"][0]["inputs"]["spatial_features"]["frame_index"] = 0
        empty = copy.deepcopy(self.tree.manifest)
        empty["samples"] = []

        for manifest in (duplicate, unknown, empty):
            with self.subTest(manifest=manifest):
                with self.assertRaises(ValueError):
                    IndexedLocalRsuSource(
                        self.tree.root,
                        manifest,
                        self.tree.contract,
                        **INDEXED_SOURCE_LIMITS
                    )

    def test_route_manifest_rejects_malformed_input_contract_cleanly(self):
        malformed_contract = {
            "spatial_features": {"dtype": "float32"}
        }

        with self.assertRaisesRegex(ValueError, "input contract"):
            IndexedLocalRsuSource(
                self.tree.root,
                self.tree.manifest,
                malformed_contract,
                **INDEXED_SOURCE_LIMITS
            )

    def test_indexed_source_rejects_a_trigger_that_bypassed_strict_parser(self):
        source = self.tree.build_source()
        parsed = parse_hil_trigger(valid_payload())

        class ForgedTrigger(object):
            def __init__(self):
                for field, value in parsed.to_metadata().items():
                    setattr(self, field, value)

            def to_metadata(self):
                return parsed.to_metadata()

        with self.assertRaisesRegex(ValueError, "validated HilTrigger"):
            source.frame_for_trigger(ForgedTrigger())

    def test_publisher_runs_at_most_once_and_echoes_every_trigger_field(self):
        processor, pipeline = self.processor()
        payload = valid_payload()

        result = processor.process_trigger(payload)

        self.assertEqual(1, len(pipeline.frames))
        for field, value in payload.items():
            self.assertEqual(value, result.header[field])
        self.assertEqual(0, result.header["remote_input_tensor_bytes"])
        self.assertEqual(payload, result.header["trigger_identity"])

        with self.assertRaisesRegex(HilTriggerError, "sequence"):
            processor.process_trigger(payload)
        self.assertEqual(1, len(pipeline.frames))

    def test_publisher_rejects_pipeline_identity_drift(self):
        class WrongRoutePipeline(FakePipeline):
            def process(self, frame):
                result = super(WrongRoutePipeline, self).process(frame)
                header = dict(result.header)
                header["route_id"] = "Town05-route-99"
                return FakeResult(header=MappingProxyType(header))

        processor, pipeline = self.processor(pipeline=WrongRoutePipeline())

        with self.assertRaisesRegex(ValueError, "route_id"):
            processor.process_trigger(valid_payload())
        self.assertEqual(1, len(pipeline.frames))

    def test_publisher_rejects_invalid_remote_input_tensor_byte_evidence(self):
        for invalid in (1, -1, False, "0", None):
            class RemoteInputPipeline(FakePipeline):
                def process(self, frame):
                    result = super(RemoteInputPipeline, self).process(frame)
                    header = dict(result.header)
                    header["remote_input_tensor_bytes"] = invalid
                    return FakeResult(header=MappingProxyType(header))

            processor, pipeline = self.processor(pipeline=RemoteInputPipeline())
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "remote_input_tensor_bytes"
                ):
                    processor.process_trigger(valid_payload())
                self.assertEqual(1, len(pipeline.frames))

        class ZeroRemoteInputPipeline(FakePipeline):
            def process(self, frame):
                result = super(ZeroRemoteInputPipeline, self).process(frame)
                header = dict(result.header)
                header["remote_input_tensor_bytes"] = 0
                return FakeResult(header=MappingProxyType(header))

        processor, _ = self.processor(pipeline=ZeroRemoteInputPipeline())
        result = processor.process_trigger(valid_payload())
        self.assertIs(type(result.header["remote_input_tensor_bytes"]), int)
        self.assertEqual(0, result.header["remote_input_tensor_bytes"])

    def test_publisher_rejects_every_existing_conflicting_trigger_identity(self):
        payload = valid_payload()
        conflicting_values = {
            "schema_version": "coptv2x.hil-trigger.v0",
            "sample_id": "sample-001",
            "sequence_no": 99,
            "source_frame": 99,
            "source_sim_time_ns": 99,
            "arm_id": "compression_only",
            "sequence_id": 99,
            "frame_id": 99,
        }

        for field, conflicting in conflicting_values.items():
            class ConflictingPipeline(FakePipeline):
                def process(self, frame):
                    result = super(ConflictingPipeline, self).process(frame)
                    header = dict(result.header)
                    header[field] = conflicting
                    return FakeResult(header=MappingProxyType(header))

            processor, pipeline = self.processor(pipeline=ConflictingPipeline())
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    processor.process_trigger(payload)
                self.assertEqual(1, len(pipeline.frames))

        class NestedConflictPipeline(FakePipeline):
            def process(self, frame):
                result = super(NestedConflictPipeline, self).process(frame)
                header = dict(result.header)
                header["trigger_identity"] = dict(payload, arm_id="compression_only")
                return FakeResult(header=MappingProxyType(header))

        processor, _ = self.processor(pipeline=NestedConflictPipeline())
        with self.assertRaisesRegex(ValueError, "trigger_identity"):
            processor.process_trigger(payload)

    def test_missing_sample_does_not_advance_sequence_without_an_inference_task(self):
        processor, pipeline = self.processor()
        missing = valid_payload(
            sequence_no=10,
            sample_id="sample-missing",
            source_frame=10,
        )

        with self.assertRaisesRegex(ValueError, "sample"):
            processor.process_trigger(missing)
        result = processor.process_trigger(valid_payload(sequence_no=0))

        self.assertEqual(0, result.header["sequence_no"])
        self.assertEqual(1, len(pipeline.frames))

    def test_duplicate_is_rejected_before_materializing_the_local_batch_again(self):
        class CountingSource(object):
            def __init__(self, source):
                self.source = source
                self.frame_calls = 0

            def frame_for_trigger(self, trigger):
                self.frame_calls += 1
                return self.source.frame_for_trigger(trigger)

        source = CountingSource(self.tree.build_source())
        processor, pipeline = self.processor(source=source)
        payload = valid_payload()

        processor.process_trigger(payload)
        with self.assertRaisesRegex(HilTriggerError, "sequence"):
            processor.process_trigger(payload)

        self.assertEqual(1, source.frame_calls)
        self.assertEqual(1, len(pipeline.frames))

    def test_concurrent_same_and_adjacent_sequences_never_duplicate_a_task(self):
        def run_concurrently(processor, payloads):
            barrier = threading.Barrier(len(payloads))

            def invoke(payload):
                barrier.wait()
                try:
                    return ("accepted", processor.process_trigger(payload).header)
                except HilTriggerError:
                    return ("rejected", None)

            with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
                return list(executor.map(invoke, payloads))

        same_processor, same_pipeline = self.processor()
        same_results = run_concurrently(
            same_processor, (valid_payload(), valid_payload())
        )
        self.assertEqual(1, sum(result[0] == "accepted" for result in same_results))
        self.assertEqual(1, len(same_pipeline.frames))

        adjacent_tree = LocalInputTree(self.temporary.name, sample_count=2)
        adjacent_processor, adjacent_pipeline = self.processor(
            source=adjacent_tree.build_source()
        )
        adjacent_results = run_concurrently(
            adjacent_processor,
            (
                valid_payload(sequence_no=0, sample_id="sample-000", source_frame=0),
                valid_payload(sequence_no=1, sample_id="sample-001", source_frame=1),
            ),
        )
        accepted = [result for result in adjacent_results if result[0] == "accepted"]
        submitted_sequences = [frame.sequence_id for frame in adjacent_pipeline.frames]
        self.assertGreaterEqual(len(accepted), 1)
        self.assertEqual(len(submitted_sequences), len(set(submitted_sequences)))
        self.assertTrue(set(submitted_sequences).issubset({0, 1}))

    def test_gate_p1_has_100_deterministic_trigger_decisions(self):
        tree = LocalInputTree(self.temporary.name, sample_count=20)
        fixtures = []
        expected = []
        for index in range(20):
            current = valid_payload(
                sequence_no=index,
                sample_id="sample-%03d" % index,
                source_frame=index,
            )
            fixtures.extend(
                (
                    current,
                    dict(current),
                    dict(current, sequence_no=max(0, index - 1)),
                    dict(current, sequence_no=index + 100, run_id="old-run"),
                    dict(
                        current,
                        sequence_no=index + 100,
                        route_id="Town05-route-18",
                    ),
                )
            )
            expected.extend((True, False, False, False, False))

        self.assertEqual(100, len(fixtures))
        self.assertEqual(20, sum(expected))
        observed_runs = []
        for _ in range(2):
            processor, pipeline = self.processor(source=tree.build_source())
            observed = []
            for fixture in fixtures:
                try:
                    result = processor.process_trigger(fixture)
                except (HilTriggerError, ValueError):
                    observed.append(False)
                else:
                    observed.append(True)
                    self.assertEqual(0, result.header["remote_input_tensor_bytes"])
            self.assertEqual(20, len(pipeline.frames))
            observed_runs.append(observed)

        self.assertEqual(expected, observed_runs[0])
        self.assertEqual(expected, observed_runs[1])


if __name__ == "__main__":
    unittest.main()
