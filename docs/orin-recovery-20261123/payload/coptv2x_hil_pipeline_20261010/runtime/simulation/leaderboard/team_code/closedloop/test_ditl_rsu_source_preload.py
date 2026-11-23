"""Bounded initialization snapshots for offline, static replay only."""

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

CLOSEDLOOP = Path(__file__).resolve().parent
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger
import ditl_rsu_source as source_module
from ditl_rsu_source import IndexedLocalRsuSource
from coptv2x_hil_artifacts import TensorContract, load_hil_artifact_registry
from coptv2x_hil_backend import CoptV2xHilBackend, HilBackendError
import coptv2x_hil_backend as backend_module


class StaticReplayPreloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = {"features": {"shape": (2, 3), "dtype": "float32"}}
        self.manifest = {
            "schema_version": "coptv2x.local-input-manifest.v1",
            "route_id": "route-1",
            "scenario_id": "scenario-1",
            "batch_order": ["ego", "rsu"],
            "samples": [],
        }
        for index in range(2):
            path = self.root / ("sample-%d.npy" % index)
            value = np.arange(6, dtype=np.float32).reshape(2, 3) + index * 10
            np.save(path, np.asfortranarray(value))
            self.manifest["samples"].append({
                "sample_id": "sample-%d" % index,
                "source_frame": index + 17,
                "inputs": {"features": {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "shape": [2, 3],
                    "dtype": "float32",
                }},
            })

    def source(self, **kwargs):
        options = {
            "max_samples": 2,
            "max_total_tensor_bytes": 4096,
            "preload_verified": True,
            "max_preloaded_tensor_bytes": 48,
        }
        options.update(kwargs)
        return IndexedLocalRsuSource(
            self.root, self.manifest, self.contract, **options
        )

    def trigger(self, index=0, **changes):
        payload = {
            "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
            "run_id": "run-1",
            "route_id": "route-1",
            "scenario_id": "scenario-1",
            "sample_id": "sample-%d" % index,
            "source_frame": index + 17,
            "sequence_no": index + 31,
            "source_sim_time_ns": (index + 17) * 100_000_000,
            "arm_id": "original_default",
        }
        payload.update(changes)
        return parse_hil_trigger(payload)

    def test_owned_readonly_snapshots_survive_original_file_changes(self):
        source = self.source(memory_map=True, lazy_load=True)
        first = source.frame_for_trigger(self.trigger())
        np.save(self.root / "sample-0.npy", np.full((2, 3), 999, dtype=np.float32))
        (self.root / "sample-1.npy").unlink()
        with mock.patch.object(
            source_module.np, "array",
            side_effect=AssertionError("no per-frame copy"),
        ), mock.patch.object(
            source_module.np, "isfinite",
            side_effect=AssertionError("no per-frame verification"),
        ):
            second = source.frame_for_trigger(self.trigger(sequence_no=44))
            other = source.frame_for_trigger(self.trigger(index=1))
        np.testing.assert_array_equal([[0, 1, 2], [3, 4, 5]], second.arrays["features"])
        np.testing.assert_array_equal([[10, 11, 12], [13, 14, 15]], other.arrays["features"])
        snapshot = second.arrays["features"]
        self.assertIs(first.arrays["features"], snapshot)
        # Bytes-backed snapshots are owned by the source, not mutable NumPy storage.
        self.assertFalse(snapshot.flags.owndata)
        self.assertTrue(snapshot.flags.c_contiguous)
        self.assertFalse(snapshot.flags.writeable)
        self.assertNotIsInstance(snapshot, np.memmap)
        with self.assertRaises(ValueError):
            snapshot[0, 0] = -1
        with self.assertRaises(TypeError):
            second.arrays["other"] = snapshot

    def test_snapshot_and_every_numpy_base_cannot_be_thawed(self):
        source = self.source()
        snapshot = source.frame_for_trigger(self.trigger()).arrays["features"]
        owner = snapshot
        while isinstance(owner, np.ndarray):
            with self.assertRaises(ValueError):
                owner.setflags(write=True)
            owner = owner.base
        self.assertIsInstance(owner, bytes)

    def test_exact_ownership_rejects_views_copies_other_sources_and_legacy_arrays(self):
        source = self.source()
        snapshot = source.frame_for_trigger(self.trigger()).arrays["features"]
        self.assertTrue(source.owns_verified_array("features", snapshot))
        self.assertFalse(source.owns_verified_array("wrong", snapshot))
        self.assertFalse(source.owns_verified_array("features", None))
        for value in (snapshot.view(), snapshot.copy(), np.zeros((2, 3), np.float32)):
            with self.subTest(value=value):
                self.assertFalse(source.owns_verified_array("features", value))
        other = self.source()
        self.assertFalse(other.owns_verified_array("features", snapshot))
        legacy = self.source(preload_verified=False)
        legacy_value = legacy.frame_for_trigger(self.trigger()).arrays["features"]
        self.assertFalse(legacy.owns_verified_array("features", legacy_value))

    def backend(self, source=None, trust=True):
        original = load_hil_artifact_registry()["original_default"]
        spec = replace(
            original, input_contract=(TensorContract("features", (2, 3), "float32"),)
        )
        with mock.patch.object(
            backend_module, "load_hil_artifact_registry",
            return_value={"original_default": spec},
        ):
            return CoptV2xHilBackend(
                "original_default", self.root, trust_readonly_inputs=trust,
                verified_input_source=source,
            )

    def test_backend_omits_finite_scan_only_for_exact_verified_immutable_snapshot(self):
        source = self.source()
        snapshot = source.frame_for_trigger(self.trigger()).arrays["features"]
        backend = self.backend(source)
        with mock.patch.object(
            backend_module, "_all_finite",
            side_effect=AssertionError("verified immutable data must not be rescanned"),
        ):
            validated = backend.validate_inputs({"features": snapshot})
        self.assertIs(snapshot, validated["features"])
        # Turning off the readonly fast path must still make a defensive copy.
        copied = self.backend(source, trust=False).validate_inputs({"features": snapshot})
        self.assertIsNot(snapshot, copied["features"])

    def test_backend_rejects_unproven_readonly_nan_and_inf_even_with_owner(self):
        source = self.source()
        backend = self.backend(source)
        for bad in (np.nan, np.inf):
            value = np.full((2, 3), bad, dtype=np.float32)
            value.setflags(write=False)
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(HilBackendError, "finite"):
                    backend.validate_inputs({"features": value})
        snapshot = source.frame_for_trigger(self.trigger()).arrays["features"]
        other = self.source().frame_for_trigger(self.trigger()).arrays["features"]
        for value in (snapshot.view(), snapshot.copy(), other):
            with mock.patch.object(backend_module, "_all_finite", return_value=False):
                with self.assertRaisesRegex(HilBackendError, "finite"):
                    backend.validate_inputs({"features": value})

    def test_backend_still_checks_keys_shape_and_dtype_for_proven_snapshot(self):
        source = self.source()
        backend = self.backend(source)
        snapshot = source.frame_for_trigger(self.trigger()).arrays["features"]
        with self.assertRaisesRegex(HilBackendError, "names"):
            backend.validate_inputs({"wrong": snapshot})
        for changed in (
            {"shape": (3, 2), "dtype": "float32"},
            {"shape": (2, 3), "dtype": "float64"},
        ):
            backend.input_contract = {"features": changed}
            with self.assertRaisesRegex(HilBackendError, "shape/dtype"):
                backend.validate_inputs({"features": snapshot})

    def test_backend_owner_option_rejects_non_source_and_non_preloaded_source(self):
        class Impostor(object):
            preload_verified = True

            def owns_verified_array(self, name, value):
                return True

        for source in (Impostor(), self.source(preload_verified=False)):
            with self.subTest(source=source):
                with self.assertRaisesRegex(HilBackendError, "verified_input_source"):
                    self.backend(source)

    def test_all_trigger_fields_remain_exact_and_wrong_composite_keys_rejected(self):
        source = self.source()
        for index in range(2):
            trigger = self.trigger(index=index)
            frame = source.frame_for_trigger(trigger)
            self.assertEqual(
                trigger.to_metadata(), dict(frame.source_metadata["trigger_identity"])
            )
            self.assertEqual(index + 17, frame.frame_id)
            self.assertEqual(index + 31, frame.sequence_id)
        for changes in (
            {"route_id": "wrong"}, {"scenario_id": "wrong"},
            {"sample_id": "wrong"}, {"source_frame": 999},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "composite key"):
                    source.frame_for_trigger(self.trigger(**changes))
        with self.assertRaisesRegex(ValueError, "validated HilTrigger"):
            source.frame_for_trigger(self.trigger().to_metadata())

    def test_cache_limit_rejects_before_materializing_any_array(self):
        with mock.patch.object(
            source_module, "_load_verified_indexed_npy",
            side_effect=AssertionError("bound must precede loading"),
        ):
            with self.assertRaisesRegex(ValueError, "max_preloaded_tensor_bytes.*48.*47"):
                self.source(max_preloaded_tensor_bytes=47)
        self.assertEqual(48, self.source().preloaded_tensor_bytes)

    def test_preload_options_require_strict_boolean_and_finite_positive_byte_limit(self):
        for options, label in (
            ({"preload_verified": 1}, "preload_verified"),
            ({"max_preloaded_tensor_bytes": 0}, "max_preloaded_tensor_bytes"),
            ({"max_preloaded_tensor_bytes": True}, "max_preloaded_tensor_bytes"),
            ({"max_preloaded_tensor_bytes": float("inf")}, "max_preloaded_tensor_bytes"),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, label):
                    self.source(**options)

    def test_initialization_verifies_hash_shape_dtype_and_finite_values(self):
        path = self.root / "sample-0.npy"
        original = path.read_bytes()
        original_manifest = copy.deepcopy(self.manifest)
        for array, update_hash, message in (
            (np.full((2, 3), 2, dtype=np.float32), False, "hash"),
            (np.zeros((3, 2), dtype=np.float32), True, "shape"),
            (np.zeros((2, 3), dtype=np.float64), True, "dtype"),
            (np.full((2, 3), np.nan, dtype=np.float32), True, "non-finite"),
            (np.full((2, 3), np.inf, dtype=np.float32), True, "non-finite"),
        ):
            with self.subTest(message=message, array=array):
                self.manifest = copy.deepcopy(original_manifest)
                np.save(path, array)
                if update_hash:
                    entry = self.manifest["samples"][0]["inputs"]["features"]
                    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, message):
                    self.source()
        path.write_bytes(original)

    def test_preload_initialization_time_is_separate_from_per_frame_lookup(self):
        with mock.patch.object(
            source_module, "monotonic_raw_ns",
            side_effect=[1_000_000, 6_000_000],
        ):
            source = self.source()
        self.assertEqual(5.0, source.initialization_ms)
        with mock.patch.object(
            source_module, "monotonic_raw_ns", side_effect=[100, 200, 225]
        ):
            frame = source.frame_for_trigger(self.trigger())
        metadata = frame.source_metadata
        self.assertEqual("static_replay_preloaded", metadata["source_type"])
        self.assertEqual(5.0, metadata["initialization_ms"])
        self.assertEqual(48, metadata["preloaded_tensor_bytes"])
        self.assertEqual(25, metadata["durations_ns"]["source_read"])
        self.assertEqual(0, metadata["remote_input_tensor_bytes"])
        self.assertEqual(("ego", "rsu"), metadata["batch_order"])

    def test_default_source_keeps_existing_copying_and_metadata_behavior(self):
        source = IndexedLocalRsuSource(
            self.root, self.manifest, self.contract,
            max_samples=2, max_total_tensor_bytes=4096,
        )
        first = source.frame_for_trigger(self.trigger())
        second = source.frame_for_trigger(self.trigger())
        self.assertIsNot(first.arrays["features"], second.arrays["features"])
        self.assertEqual("local_indexed_feature", first.source_metadata["source_type"])
        self.assertNotIn("initialization_ms", first.source_metadata)
        lazy = self.source(preload_verified=False, lazy_load=True)
        np.save(self.root / "sample-0.npy", np.zeros((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hash"):
            lazy.frame_for_trigger(self.trigger())


if __name__ == "__main__":
    unittest.main()
