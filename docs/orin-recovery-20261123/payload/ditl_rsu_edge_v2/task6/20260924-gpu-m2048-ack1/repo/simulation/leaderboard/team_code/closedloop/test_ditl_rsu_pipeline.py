from __future__ import print_function

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ditl_rsu_pipeline import RsuEdgePipeline
from ditl_rsu_source import LocalRsuFrameSource


class FakeBackend(object):
    def __init__(self):
        self.input_contract = {
            "spatial_features": {"shape": (2, 3), "dtype": "float32"},
            "t_ego": {"shape": (2, 2), "dtype": "float32"},
        }
        self.inference_count = 0

    def infer(self, arrays):
        from ditl_edge_runtime import monotonic_raw_ns

        h2d_start = monotonic_raw_ns()
        h2d_end = monotonic_raw_ns()
        inference_start = monotonic_raw_ns()
        inference_end = monotonic_raw_ns()
        d2h_start = monotonic_raw_ns()
        d2h_end = monotonic_raw_ns()
        self.inference_count += 1
        return {
            "outputs": {
                "cls_preds": np.ascontiguousarray(
                    arrays["spatial_features"] * 2.0
                )
            },
            "model": {"backend": "fake", "model_id": "fake-rsu-v1"},
            "durations_ns": {
                "input_validate": 1,
                "h2d": max(0, h2d_end - h2d_start),
                "gpu_inference": max(0, inference_end - inference_start),
                "d2h": max(0, d2h_end - d2h_start),
            },
            "edge_timestamps_ns": {
                "h2d_start": h2d_start,
                "h2d_end": h2d_end,
                "inference_start": inference_start,
                "inference_end": inference_end,
                "d2h_start": d2h_start,
                "d2h_end": d2h_end,
            },
        }


class DitlRsuSourceAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        spatial = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        t_ego = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        np.save(str(self.root / "spatial.npy"), spatial)
        np.save(str(self.root / "tego.npy"), t_ego)
        self.manifest = {
            "spatial_features": {"path": "spatial.npy", "frame_axis": 0},
            "t_ego": {"path": "tego.npy", "frame_axis": 0},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def build_source(self):
        return LocalRsuFrameSource(
            data_root=self.root,
            manifest=self.manifest,
            input_contract=FakeBackend().input_contract,
            source_id="rsu-orin-01",
        )

    def test_source_sequences_frames_and_returns_independent_read_only_arrays(self):
        source = self.build_source()
        first = source.next_frame()
        second = source.next_frame()

        self.assertEqual((0, 1), (first.sequence_id, second.sequence_id))
        self.assertEqual((0, 1), (first.frame_id, second.frame_id))
        self.assertLessEqual(
            first.data_available_ns,
            first.source_metadata["timestamps_ns"]["source_read_start"],
        )
        self.assertLessEqual(
            first.source_metadata["timestamps_ns"]["source_read_start"],
            first.source_metadata["timestamps_ns"]["source_read_end"],
        )
        self.assertFalse(first.arrays["spatial_features"].flags.writeable)
        self.assertFalse(
            np.shares_memory(
                first.arrays["spatial_features"],
                second.arrays["spatial_features"],
            )
        )
        with self.assertRaises(ValueError):
            first.arrays["spatial_features"][0, 0] = -1.0

    def test_source_can_repeat_one_fixed_local_frame_for_paired_measurement(self):
        manifest = {
            name: dict(entry, frame_index=0)
            for name, entry in self.manifest.items()
        }
        source = LocalRsuFrameSource(
            data_root=self.root,
            manifest=manifest,
            input_contract=FakeBackend().input_contract,
        )

        first = source.next_frame()
        second = source.next_frame()

        self.assertEqual(0, first.frame_id)
        self.assertEqual(0, second.frame_id)
        self.assertEqual(0, second.source_metadata["source_frame_index"])
        for name in first.arrays:
            np.testing.assert_array_equal(first.arrays[name], second.arrays[name])

    def test_source_rejects_manifest_path_escape_and_pickle_array(self):
        with self.assertRaisesRegex(ValueError, "data root"):
            LocalRsuFrameSource(
                data_root=self.root,
                manifest={
                    "spatial_features": {"path": "../x.npy", "frame_axis": 0}
                },
                input_contract={
                    "spatial_features": {"shape": (2, 3), "dtype": "float32"}
                },
            )

        np.save(
            str(self.root / "objects.npy"),
            np.asarray([{"unsafe": True}], dtype=object),
            allow_pickle=True,
        )
        with self.assertRaisesRegex(ValueError, "pickle|object"):
            LocalRsuFrameSource(
                data_root=self.root,
                manifest={
                    "spatial_features": {
                        "path": "objects.npy",
                        "frame_axis": 0,
                    }
                },
                input_contract={
                    "spatial_features": {"shape": (), "dtype": "object"}
                },
            )

    def test_pipeline_uses_local_frame_once_without_opening_a_socket(self):
        source = self.build_source()
        backend = FakeBackend()
        pipeline = RsuEdgePipeline(backend, run_id="local-run")
        frame = source.next_frame()

        with mock.patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("network input is forbidden"),
        ):
            result = pipeline.process(frame)

        self.assertEqual(frame.sequence_id, result.sequence_id)
        self.assertEqual(1, backend.inference_count)
        self.assertEqual("none", result.tensor_bundle["compression"])
        self.assertEqual("dense_head_debug", result.header["payload_semantics"])
        self.assertEqual("identity", result.header["postprocess"])
        self.assertLessEqual(
            result.timestamps_ns["rsu_data_available"],
            result.timestamps_ns["preprocess_start"],
        )
        ordered = [
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
            "encode_start",
            "encode_end",
        ]
        values = [result.timestamps_ns[name] for name in ordered]
        self.assertEqual(values, sorted(values))
        self.assertIn("pack", result.durations_ns)
        self.assertIn("hash", result.durations_ns)
        np.testing.assert_array_equal(
            frame.arrays["spatial_features"] * 2.0,
            result.outputs["cls_preds"],
        )

    def test_pipeline_rejects_duplicate_sequence_without_second_inference(self):
        source = self.build_source()
        backend = FakeBackend()
        pipeline = RsuEdgePipeline(backend, run_id="local-run")
        frame = source.next_frame()
        pipeline.process(frame)

        with self.assertRaisesRegex(ValueError, "sequence"):
            pipeline.process(frame)
        self.assertEqual(1, backend.inference_count)


if __name__ == "__main__":
    unittest.main()
