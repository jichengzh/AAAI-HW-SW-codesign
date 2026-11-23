from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage3_pyramid_calibration_export_v3 as export_v3  # noqa: E402


class Stage3PyramidCalibrationExportV3Tests(unittest.TestCase):
    def test_explicit_checkpoint_loader_preserves_epoch23_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "net_epoch_bestval_at23.pth"
            torch.save(
                {"weight": torch.tensor([[23.0]]), "bias": torch.tensor([23.0])},
                checkpoint,
            )
            model = nn.Linear(1, 1)

            epoch, loaded = export_v3.load_explicit_checkpoint(model, checkpoint)

            self.assertEqual(epoch, 23)
            self.assertEqual(float(loaded.weight.item()), 23.0)

    def test_parse_eval_range_accepts_positive_xy_pair(self) -> None:
        self.assertEqual(export_v3.parse_eval_range("102.4,51.2"), (102.4, 51.2))

    def test_parse_eval_range_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            export_v3.parse_eval_range("0,51.2")

    def test_select_frozen_train_hypes_redirects_validate_dir_to_root_dir(self) -> None:
        raw_hypes = {
            "root_dir": "/data/frozen_train.json",
            "validate_dir": "/data/val.json",
            "test_dir": "/data/test.json",
        }

        selected, split_source = export_v3.select_frozen_train_hypes(raw_hypes)

        self.assertEqual(selected["validate_dir"], "/data/frozen_train.json")
        self.assertEqual(selected["test_dir"], "/data/test.json")
        self.assertEqual(split_source, Path("/data/frozen_train.json"))
        self.assertEqual(raw_hypes["validate_dir"], "/data/val.json")

    def test_validate_spatial_features_requires_four_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape \\[N,C,H,W\\]"):
            export_v3.validate_spatial_features(np.zeros((16, 64, 256), dtype=np.float16))

    def test_validate_scene_capture_requires_agent_axis_matches_record_len(self) -> None:
        scene = np.zeros((3, 64, 256, 512), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "record_len"):
            export_v3.validate_scene_capture(scene, record_len=2, scene_index=0)

    def test_build_summary_and_validator_bind_train_provenance(self) -> None:
        summary = export_v3.build_summary(
            ckpt_dir=Path("/tmp/ckpt"),
            checkpoint_path=Path("/tmp/ckpt/net_epoch24.pth"),
            checkpoint_sha256="a" * 64,
            checkpoint_epoch=24,
            config_path=Path("/tmp/ckpt/config.yaml"),
            config_sha256="b" * 64,
            heal_root=Path("/tmp/HEAL"),
            split_source_file=Path("/data/frozen_train.json"),
            split_source_sha256="c" * 64,
            output_npz=Path("/tmp/out/calib.npz"),
            output_sha256="d" * 64,
            summary_json=Path("/tmp/out/summary.json"),
            num_scenes_requested=16,
            scene_sample_count=16,
            agent_instance_count=26,
            scene_record_lens=[1, 2, 2, 1, 3, 2, 1, 2, 2, 1, 1, 3, 2, 1, 1, 1],
            scene_record_lens_sha256=export_v3.sha256_json([1, 2, 2, 1, 3, 2, 1, 2, 2, 1, 1, 3, 2, 1, 1, 1]),
            spatial_shape=[26, 64, 256, 512],
            eval_range="102.4,102.4",
            gpu_id=0,
            elapsed_secs=2.5,
        )

        validated = export_v3.validate_summary(summary)

        self.assertEqual(validated["schema"], export_v3.SCHEMA)
        self.assertEqual(validated["calibration_split"], "train")
        self.assertEqual(validated["scene_sample_count"], 16)
        self.assertEqual(validated["agent_instance_count"], 26)
        self.assertEqual(validated["spatial_features_shape"], [26, 64, 256, 512])

    def test_validate_summary_rejects_non_train_split(self) -> None:
        summary = {
            "schema": export_v3.SCHEMA,
            "calibration_split": "val",
            "scene_sample_count": 16,
            "agent_instance_count": 16,
            "scene_record_lens": [1] * 16,
            "scene_record_lens_sha256": "a" * 64,
            "spatial_features_shape": [16, 64, 256, 512],
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "split_source_sha256": "d" * 64,
            "output_sha256": "e" * 64,
        }

        with self.assertRaisesRegex(ValueError, "train split"):
            export_v3.validate_summary(summary)

    def test_write_export_artifacts_persists_npz_and_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_npz = root / "calib.npz"
            summary_json = root / "summary.json"
            spatial = np.zeros((4, 64, 8, 8), dtype=np.float16)
            summary = export_v3.write_export_artifacts(
                spatial_features=spatial,
                output_npz=output_npz,
                summary_json=summary_json,
                summary_payload=export_v3.build_summary(
                    ckpt_dir=Path("/tmp/ckpt"),
                    checkpoint_path=Path("/tmp/ckpt/net_epoch24.pth"),
                    checkpoint_sha256="a" * 64,
                    checkpoint_epoch=24,
                    config_path=Path("/tmp/ckpt/config.yaml"),
                    config_sha256="b" * 64,
                    heal_root=Path("/tmp/HEAL"),
                    split_source_file=Path("/data/frozen_train.json"),
                    split_source_sha256="c" * 64,
                    output_npz=output_npz,
                    output_sha256="0" * 64,
                    summary_json=summary_json,
                    num_scenes_requested=3,
                    scene_sample_count=3,
                    agent_instance_count=4,
                    scene_record_lens=[1, 1, 2],
                    scene_record_lens_sha256=export_v3.sha256_json([1, 1, 2]),
                    spatial_shape=[4, 64, 8, 8],
                    eval_range="102.4,102.4",
                    gpu_id=0,
                    elapsed_secs=1.0,
                ),
            )

            with np.load(output_npz, allow_pickle=False) as payload:
                self.assertEqual(payload["spatial_features"].shape, (4, 64, 8, 8))
            loaded = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["output_sha256"], export_v3.sha256_file(output_npz))
            self.assertEqual(summary["output_sha256"], loaded["output_sha256"])


if __name__ == "__main__":
    unittest.main()
