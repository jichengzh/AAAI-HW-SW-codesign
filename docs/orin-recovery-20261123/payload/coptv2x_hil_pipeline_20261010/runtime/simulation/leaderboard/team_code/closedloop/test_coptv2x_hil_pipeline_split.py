"""Split inference must hand off owned values, provenance, and honest timing."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import sys
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import coptv2x_hil_backend as hil
import ditl_rsu_pipeline as pipeline_module
from ditl_edge_protocol import decode_tensor_bundle
from ditl_rsu_pipeline import CoptV2xHilPipeline, RsuEdgePipeline
from ditl_rsu_source import LocalRsuFrame
from test_coptv2x_hil_backend import ArtifactTree, _trigger


class ReusedRuntime:
    """A host-buffer-reusing runtime, including mutable nested metadata."""
    def __init__(self, spec, inputs, outputs, projected=False, readonly=False):
        self.input_contract = inputs
        self.output_contract = outputs
        self.projected = projected
        self.readonly = readonly
        self.outputs = {
            name: np.zeros((1,) + item["shape"][1:] if projected else item["shape"],
                           dtype=item["dtype"])
            for name, item in outputs.items()
        }
        self.model = {"backend": "fixture", "nested": {"calls": 0}}
        self.calls = 0
        if projected:
            self.infer_validated_rsu = self.infer_validated

    def infer_validated(self, inputs):
        self.calls += 1
        for index, value in enumerate(self.outputs.values()):
            value[...] = self.calls * 10 + index
            if not self.projected:
                value[0] = -self.calls * 10 - index
        self.model["nested"]["calls"] = self.calls
        outputs = {name: value.view() for name, value in self.outputs.items()}
        if self.readonly:
            for value in outputs.values():
                value.setflags(write=False)
        return {"outputs": outputs, "model": self.model,
                "durations_ns": {"h2d": 3, "gpu_inference": 5, "d2h": 7},
                "edge_timestamps_ns": {"h2d_start": 30, "h2d_end": 33,
                                       "inference_start": 34, "inference_end": 39,
                                       "d2h_start": 40, "d2h_end": 47}}


class PipelineSplitTests(unittest.TestCase):
    def setUp(self):
        self.tree = ArtifactTree()
        self.addCleanup(self.tree.close)

    def backend(self, rsu_only=True, projected=False, readonly=False, dedup=False):
        backend = hil.CoptV2xHilBackend(
            "joint_shcosearch", self.tree.artifact_root, self.tree.config_dir,
            runtime_factory=lambda spec, paths, inputs, outputs, root:
                ReusedRuntime(spec, inputs, outputs, projected, readonly),
            rsu_only=rsu_only, checksum_dedup=dedup,
        )
        # Small synthetic bindings exercise the production split without GPU memory.
        backend.input_contract = {"spatial_features": {"shape": (2, 1), "dtype": "float32"}}
        backend.output_contract = {
            name: {"shape": (2, 1, 1, 2), "dtype": "float32"}
            for name in ("res0", "res1", "res2")
        }
        return backend

    def frame(self, sequence=7):
        trigger = _trigger("joint_shcosearch", sequence)
        return LocalRsuFrame(
            frame_id=42, sequence_id=sequence, data_available_ns=5,
            arrays={"spatial_features": np.zeros((2, 1), dtype=np.float32)},
            source_metadata={"trigger_identity": trigger.to_metadata(),
                             "batch_order": ["ego", "rsu"],
                             "timestamps_ns": {"source_read_start": 10, "source_read_end": 15},
                             "durations_ns": {"source_read": 5}},
        )

    def raw(self, backend, sequence=7):
        return backend.infer_hil_raw_validated(
            self.frame(sequence).arrays, _trigger("joint_shcosearch", sequence),
            ("ego", "rsu"), preprocess_start=20, preprocess_end=25,
            local_source_timestamps_ns={"source_read_start": 10, "source_read_end": 15},
            local_source_durations_ns={"source_read": 5},
        )

    def test_serial_and_split_preserve_values_payload_and_identity(self):
        for rsu_only, dedup in ((False, False), (True, False), (True, True)):
            with self.subTest(rsu_only=rsu_only, dedup=dedup):
                serial = CoptV2xHilPipeline(self.backend(rsu_only=rsu_only, dedup=dedup),
                                          run_id="run-task3", checksum_dedup=dedup)
                split = CoptV2xHilPipeline(self.backend(rsu_only=rsu_only, dedup=dedup),
                                         run_id="run-task3", checksum_dedup=dedup)
                expected = serial.process(self.frame())
                inferred = split.infer_frame(self.frame())
                actual = split.encode_inferred_frame(inferred)
                self.assertEqual(expected.payload, actual.payload)
                for name in ("trigger_identity", "artifact_identity", "payload_sha256",
                             "payload_semantics", "batch_order", "request_id", "frame_id"):
                    self.assertEqual(expected.header[name], actual.header[name])
                decoded = decode_tensor_bundle(dict(actual.tensor_bundle), actual.payload)
                for index, name in enumerate(("res0", "res1", "res2")):
                    np.testing.assert_array_equal(expected.outputs[name], decoded[name])
                    self.assertTrue(np.all(actual.outputs[name][-1] == 10 + index))
                for name, value in {"source_read_start": 10, "source_read_end": 15,
                                    "inference_start": 34, "d2h_end": 47}.items():
                    self.assertEqual(value, actual.timestamps_ns[name])
                self.assertEqual(5, actual.durations_ns["inference"])

    def test_raw_snapshot_survives_reused_mutable_and_readonly_runtime_buffers(self):
        for rsu_only, projected, readonly in ((False, False, False), (True, False, True),
                                             (True, True, False)):
            with self.subTest(rsu_only=rsu_only, projected=projected, readonly=readonly):
                backend = self.backend(rsu_only=rsu_only, projected=projected, readonly=readonly)
                first = self.raw(backend)
                second = self.raw(backend, 8)
                for name in first.outputs:
                    self.assertFalse(np.shares_memory(first.outputs[name], backend.load().outputs[name]))
                    with self.assertRaises(ValueError):
                        first.outputs[name].setflags(write=True)
                result = backend.finish_hil_inference(first)
                self.assertEqual(1, result.model["nested"]["calls"])
                self.assertTrue(np.all(result.outputs["res0"][-1] == 10))
                self.assertTrue(np.all(backend.finish_hil_inference(second).outputs["res0"][-1] == 20))
                self.assertEqual(2, backend.inference_count)
                with self.assertRaises(TypeError):
                    first.model["nested"]["calls"] = 100

    def test_finite_scan_and_canonical_hash_are_deferred_until_finish(self):
        backend = self.backend()
        with mock.patch.object(hil, "_all_finite", side_effect=AssertionError("early finite scan")), \
                mock.patch.object(hil, "hash_feature_payload", side_effect=AssertionError("early hash")):
            raw = self.raw(backend)
        result = backend.finish_hil_inference(raw)
        self.assertEqual(hil.hash_feature_payload(result.outputs), result.payload_sha256)
        backend.load().outputs["res0"][1] = np.nan
        original = backend.load().infer_validated
        def nonfinite(inputs):
            value = original(inputs)
            value["outputs"]["res0"][1] = np.nan
            return value
        backend.load().infer_validated = nonfinite
        raw = self.raw(backend, 8)
        with self.assertRaisesRegex(hil.HilBackendError, "finite"):
            backend.finish_hil_inference(raw)
        self.assertEqual(1, backend.inference_count)

    def test_duplicate_sequence_rejected_before_finish_and_cannot_be_reencoded(self):
        backend = self.backend()
        pipeline = CoptV2xHilPipeline(backend, run_id="run-task3")
        frame = self.frame()
        inferred = pipeline.infer_frame(frame)
        with self.assertRaisesRegex(ValueError, "sequence"):
            pipeline.infer_frame(frame)
        self.assertEqual(1, backend.load().calls)
        pipeline.encode_inferred_frame(inferred)
        with self.assertRaisesRegex(ValueError, "sequence"):
            pipeline.encode_inferred_frame(inferred)

    def test_finish_of_previous_frame_can_overlap_next_inference_without_state_drift(self):
        backend = self.backend(dedup=True)
        pipeline = CoptV2xHilPipeline(backend, run_id="run-task3", checksum_dedup=True)
        first = pipeline.infer_frame(self.frame())
        entered = threading.Event()
        release = threading.Event()
        original = backend.finish_hil_inference
        def slow_finish(raw):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(raw)
        with ThreadPoolExecutor(max_workers=1) as finish_thread, \
                mock.patch.object(backend, "finish_hil_inference", side_effect=slow_finish):
            pending = finish_thread.submit(pipeline.encode_inferred_frame, first)
            self.assertTrue(entered.wait(5))
            try:
                second = pipeline.infer_frame(self.frame(8))
            finally:
                release.set()
            first_result = pending.result(5)
        second_result = pipeline.encode_inferred_frame(second)
        self.assertEqual((7, 8), (first_result.sequence_id, second_result.sequence_id))
        self.assertTrue(np.all(first_result.outputs["res0"] == 10))
        self.assertTrue(np.all(second_result.outputs["res0"] == 20))
        self.assertEqual(2, backend.inference_count)
        with self.assertRaisesRegex(ValueError, "sequence"):
            pipeline.infer_frame(self.frame(8))

    def test_handoff_captures_source_metadata_before_caller_reuse(self):
        pipeline = CoptV2xHilPipeline(self.backend(), run_id="run-task3")
        frame = self.frame()
        inferred = pipeline.infer_frame(frame)
        frame.source_metadata["trigger_identity"]["sample_id"] = "wrong-frame"
        frame.source_metadata["timestamps_ns"]["source_read_start"] = 999
        encoded = pipeline.encode_inferred_frame(inferred)
        self.assertEqual("sample-000", encoded.header["trigger_identity"]["sample_id"])
        self.assertEqual(10, encoded.timestamps_ns["source_read_start"])

    def test_non_hil_legacy_backend_still_processes_and_splits(self):
        class Backend:
            input_contract = {"spatial_features": {"shape": (2, 1), "dtype": "float32"}}
            def infer(self, inputs):
                return {"outputs": {"debug": inputs["spatial_features"] + 2},
                        "model": {"backend": "plain"}}
        pipeline = RsuEdgePipeline(Backend(), run_id="legacy")
        encoded = pipeline.process(self.frame())
        self.assertEqual("dense_head_debug", encoded.header["payload_semantics"])
        split = pipeline.encode_inferred_frame(pipeline.infer_frame(self.frame(8)))
        self.assertEqual(encoded.payload, split.payload)

    def test_generic_handoff_outputs_cannot_be_thawed(self):
        class Backend:
            input_contract = {"spatial_features": {"shape": (2, 1), "dtype": "float32"}}
            def infer_validated(self, inputs, **unused):
                return {"outputs": {"debug": inputs["spatial_features"] + 2},
                        "model": {"backend": "plain"}}
        pipeline = RsuEdgePipeline(Backend(), run_id="legacy")
        inferred = pipeline.infer_frame(self.frame())
        with self.assertRaises(ValueError):
            inferred.inference["outputs"]["debug"].setflags(write=True)
        result = pipeline.encode_inferred_frame(inferred)
        with self.assertRaises(ValueError):
            result.outputs["debug"].setflags(write=True)

    def test_unprojected_runtime_still_rejects_nonfinite_ego_before_rsu_finish(self):
        backend = self.backend(rsu_only=True)
        original = backend.load().infer_validated
        def nonfinite_ego(inputs):
            inference = original(inputs)
            inference["outputs"]["res0"][0] = np.nan
            return inference
        backend.load().infer_validated = nonfinite_ego
        raw = self.raw(backend)
        with self.assertRaisesRegex(hil.HilBackendError, "finite"):
            backend.finish_hil_inference(raw)

    def test_serial_codec_failure_does_not_consume_legacy_sequence(self):
        pipeline = CoptV2xHilPipeline(self.backend(), run_id="run-task3")
        with mock.patch.object(pipeline_module, "encode_tensor_bundle", side_effect=ValueError("codec failed")):
            with self.assertRaisesRegex(ValueError, "codec failed"):
                pipeline.process(self.frame())
        result = pipeline.process(self.frame())
        self.assertEqual(7, result.sequence_id)

    def test_wrong_producer_and_changed_identity_are_rejected(self):
        backend = self.backend()
        raw = self.raw(backend)
        with self.assertRaisesRegex(hil.HilBackendError, "producer"):
            self.backend().finish_hil_inference(raw)
        for changed in (replace(raw, payload_semantics="wrong"),
                        replace(raw, trigger_identity={"arm_id": "schedule_only"}),
                        replace(raw, artifact_identity={})):
            with self.assertRaisesRegex(hil.HilBackendError, "identity"):
                backend.finish_hil_inference(changed)
        pipeline = CoptV2xHilPipeline(backend, run_id="run-task3")
        inferred = pipeline.infer_frame(self.frame())
        with self.assertRaisesRegex(ValueError, "producer"):
            CoptV2xHilPipeline(backend, run_id="other").encode_inferred_frame(inferred)


if __name__ == "__main__":
    unittest.main()
