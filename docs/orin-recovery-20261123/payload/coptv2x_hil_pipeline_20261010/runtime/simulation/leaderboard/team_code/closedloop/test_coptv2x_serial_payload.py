"""Serial payload optimizations must preserve frozen FP32 RSU values."""

import hashlib
from dataclasses import replace
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import coptv2x_hil_backend as backend_module
import ditl_rsu_pipeline as pipeline_module
from coptv2x_hil_backend import (
    CoptV2xHilBackend, HilBackendError, hash_feature_payload,
    validate_feature_payload_contract,
)
from ditl_edge_protocol import decode_tensor_bundle, encode_tensor_bundle
from ditl_rsu_pipeline import CoptV2xHilPipeline
from ditl_rsu_source import LocalRsuFrame
from test_coptv2x_hil_backend import ArtifactTree, FakeRuntime, _trigger


RSU_SEMANTICS = "codriving_rsu_multiscale_features.v1"


class SentinelRuntime(FakeRuntime):
    def infer_validated(self, inputs, **unused):
        result = super().infer_validated(inputs, **unused)
        for index, value in enumerate(result["outputs"].values()):
            value[0] = -10 - index
            value[1] = 20 + index + self.inference_count
        self.last_outputs = result["outputs"]
        return result


class FrozenSentinelRuntime(SentinelRuntime):
    def infer_validated(self, inputs, **unused):
        result = super().infer_validated(inputs, **unused)
        result["outputs"] = {
            name: np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            for name, value in result["outputs"].items()
        }
        self.last_outputs = result["outputs"]
        return result


class SerialPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tree = ArtifactTree()
        self.addCleanup(self.tree.close)

    def backend(self, runtime_class=SentinelRuntime, **kwargs):
        return CoptV2xHilBackend(
            "joint_shcosearch", self.tree.artifact_root, self.tree.config_dir,
            runtime_factory=lambda spec, paths, inputs, outputs, root:
                runtime_class(spec, inputs, outputs),
            **kwargs,
        )

    def spatial(self):
        return np.zeros((2, 64, 256, 512), dtype=np.float32)

    def infer(self, backend):
        return backend.infer(self.spatial(), _trigger("joint_shcosearch"), ("ego", "rsu"))

    def test_rsu_projection_preserves_frozen_engine_and_exact_rsu_snapshot(self):
        backend = self.backend(rsu_only=True)
        first = self.infer(backend)
        self.assertEqual(RSU_SEMANTICS, first.payload_semantics)
        self.assertEqual(("rsu",), backend.batch_order)
        for index, name in enumerate(("res0", "res1", "res2")):
            expected_shape = (1,) + backend.output_contract[name]["shape"][1:]
            self.assertEqual(expected_shape, first.outputs[name].shape)
            self.assertEqual(2, backend.load().output_contract[name]["shape"][0])
            self.assertTrue(np.all(first.outputs[name] == 21 + index))
            self.assertFalse(first.outputs[name].flags.writeable)
        second = self.infer(backend)
        self.assertTrue(np.all(first.outputs["res0"] == 21))
        self.assertTrue(np.all(second.outputs["res0"] == 22))
        self.assertEqual(hash_feature_payload(first.outputs), first.payload_sha256)
        self.assertNotEqual(hash_feature_payload(backend.load().last_outputs), first.payload_sha256)

    def test_immutable_runtime_snapshot_is_reused_and_not_changed_by_next_inference(self):
        backend = self.backend(runtime_class=FrozenSentinelRuntime, rsu_only=True)
        result = self.infer(backend)
        snapshot = backend.load().last_outputs["res0"]
        self.assertTrue(np.shares_memory(snapshot, result.outputs["res0"]))
        self.infer(backend)
        self.assertTrue(np.all(result.outputs["res0"] == 21))
        with self.assertRaises(ValueError):
            result.outputs["res0"].setflags(write=True)

    def test_readonly_view_of_mutable_runtime_owner_is_defensively_snapshotted(self):
        backend = self.backend(rsu_only=True)
        runtime = backend.load()
        original_infer = runtime.infer_validated
        def with_readonly_view(inputs):
            result = original_infer(inputs)
            result["outputs"] = {
                name: value.view() for name, value in result["outputs"].items()
            }
            for value in result["outputs"].values():
                value.setflags(write=False)
            return result
        runtime.infer_validated = with_readonly_view
        result = self.infer(backend)
        runtime.last_outputs["res0"][1] = -1
        self.assertTrue(np.all(result.outputs["res0"] == 21))

    def test_payload_validator_binds_semantics_to_shape_without_changing_registry(self):
        backend = self.backend(rsu_only=True)
        result = self.infer(backend)
        identity = backend_module._plain_artifact_identity(backend.spec)
        spec = validate_feature_payload_contract(
            identity, result.outputs, self.tree.config_dir, payload_semantics=RSU_SEMANTICS,
        )
        self.assertEqual(2, spec.output_contract[0].shape[0])
        with self.assertRaises(HilBackendError):
            validate_feature_payload_contract(identity, result.outputs, self.tree.config_dir)
        with self.assertRaises(HilBackendError):
            validate_feature_payload_contract(
                identity, backend.load().last_outputs, self.tree.config_dir,
                payload_semantics=RSU_SEMANTICS,
            )
        with self.assertRaises(HilBackendError):
            validate_feature_payload_contract(
                identity, result.outputs, self.tree.config_dir, payload_semantics="unknown",
            )

    def test_trusted_input_reuses_only_owned_readonly_contiguous_finite_arrays(self):
        backend = self.backend(trust_readonly_inputs=True)
        spatial = self.spatial()
        spatial.setflags(write=False)
        self.assertIs(spatial, backend.validate_inputs({"spatial_features": spatial})["spatial_features"])
        writable = self.spatial()
        self.assertFalse(np.shares_memory(
            writable, backend.validate_inputs({"spatial_features": writable})["spatial_features"],
        ))
        view = writable.view()
        view.setflags(write=False)
        self.assertFalse(np.shares_memory(
            view, backend.validate_inputs({"spatial_features": view})["spatial_features"],
        ))
        invalid = self.spatial()
        invalid.flat[0] = np.nan
        invalid.setflags(write=False)
        with self.assertRaisesRegex(HilBackendError, "finite"):
            backend.validate_inputs({"spatial_features": invalid})
        for invalid in (
            np.zeros((1, 64, 256, 512), dtype=np.float32),
            np.zeros((2, 64, 256, 512), dtype=np.float16),
        ):
            with self.assertRaises(HilBackendError):
                backend.validate_inputs({"spatial_features": invalid})

    def test_local_processing_uses_data_available_fallback_and_preserves_legacy_batch_header(self):
        backend = self.backend()
        trigger = _trigger("joint_shcosearch")
        frame = LocalRsuFrame(
            frame_id=42, sequence_id=7, data_available_ns=5, arrays={"spatial_features": self.spatial()},
            source_metadata={"trigger_identity": trigger.to_metadata(), "batch_order": ["ego", "rsu"]},
        )
        with mock.patch.object(pipeline_module, "monotonic_raw_ns", side_effect=(20, 30, 40, 50, 60, 70, 120)), \
                mock.patch.object(pipeline_module, "collect_device_snapshot", return_value={}):
            result = CoptV2xHilPipeline(backend, run_id="run-task3").process(frame)
        self.assertEqual(5, result.timestamps_ns["local_processing_start"])
        self.assertEqual(115, result.durations_ns["local_processing"])
        self.assertEqual(["ego", "rsu"], result.header["batch_order"])
        self.assertNotIn("source_batch_order", result.header)

    def test_pipeline_rsu_header_and_elapsed_local_boundary_include_metadata_overhead(self):
        backend = self.backend(rsu_only=True)
        trigger = _trigger("joint_shcosearch")
        frame = LocalRsuFrame(
            frame_id=42, sequence_id=7, data_available_ns=5, arrays={"spatial_features": self.spatial()},
            source_metadata={
                "trigger_identity": trigger.to_metadata(), "batch_order": ["ego", "rsu"],
                "timestamps_ns": {"source_read_start": 10, "source_read_end": 15},
                "durations_ns": {"overlapping_not_additive": 999999},
            },
        )
        clock = iter((20, 30, 40, 50, 60, 70, 120))
        def metadata():
            # Metadata collection is local work; it must precede the end sample.
            return {"metadata_collected": True}
        with mock.patch.object(pipeline_module, "monotonic_raw_ns", side_effect=lambda: next(clock)), \
                mock.patch.object(pipeline_module, "collect_device_snapshot", side_effect=metadata):
            result = CoptV2xHilPipeline(backend, run_id="run-task3").process(frame)
        self.assertEqual(RSU_SEMANTICS, result.header["payload_semantics"])
        self.assertEqual(["rsu"], result.header["batch_order"])
        self.assertEqual(["ego", "rsu"], result.header["source_batch_order"])
        self.assertEqual(10, result.timestamps_ns["local_processing_start"])
        self.assertEqual(120, result.timestamps_ns["local_processing_end"])
        self.assertEqual(110, result.durations_ns["local_processing"])
        self.assertEqual(110, result.header["durations_ns"]["local_processing"])
        decoded = decode_tensor_bundle(dict(result.tensor_bundle), result.payload)
        self.assertEqual(result.header["payload_sha256"], hash_feature_payload(decoded))

    def test_pipeline_distinguishes_feature_snapshot_hash_from_wire_hash(self):
        backend = self.backend(rsu_only=True)
        original = backend.infer_hil_validated
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, durations_ns=dict(result.durations_ns, pack=123, hash=456))
        frame = LocalRsuFrame(
            frame_id=42, sequence_id=7, data_available_ns=5,
            arrays={"spatial_features": self.spatial()},
            source_metadata={"trigger_identity": _trigger("joint_shcosearch").to_metadata(),
                             "batch_order": ["ego", "rsu"]})
        with mock.patch.object(backend, "infer_hil_validated", side_effect=observed):
            result = CoptV2xHilPipeline(backend, run_id="run-task3").process(frame)
        self.assertEqual(123, result.durations_ns["feature_pack"])
        self.assertEqual(456, result.durations_ns["feature_hash"])
        self.assertEqual(result.tensor_bundle["codec_stages_ns"]["hash_ns"],
                         result.durations_ns["hash"])


class ImmutableDecoderTests(unittest.TestCase):
    def test_none_payload_is_single_immutable_owner_and_survives_next_decode(self):
        arrays = {"a": np.array([1, 2], dtype=np.float32), "b": np.array([3], dtype=np.float32)}
        metadata, payload = encode_tensor_bundle(arrays, compression="none")
        first = decode_tensor_bundle(metadata, payload, readonly=True)
        self.assertFalse(first["a"].flags.writeable)
        self.assertFalse(first["a"].flags.owndata)
        owner = first["a"]
        while isinstance(owner, np.ndarray):
            owner = owner.base
        self.assertIs(payload, owner)
        next_metadata, next_payload = encode_tensor_bundle({"a": np.array([9], dtype=np.float32)})
        decode_tensor_bundle(next_metadata, next_payload, readonly=True)
        np.testing.assert_array_equal([1, 2], first["a"])
        with self.assertRaises(ValueError):
            first["a"][0] = 7
        self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])

    def test_bytearray_caller_cannot_change_decoded_snapshot(self):
        metadata, payload = encode_tensor_bundle({"a": np.array([1, 2], dtype=np.float32)})
        caller = bytearray(payload)
        decoded = decode_tensor_bundle(metadata, caller, readonly=True)
        caller[:] = b"\x00" * len(caller)
        np.testing.assert_array_equal([1, 2], decoded["a"])
        self.assertFalse(decoded["a"].flags.writeable)


if __name__ == "__main__":
    unittest.main()
