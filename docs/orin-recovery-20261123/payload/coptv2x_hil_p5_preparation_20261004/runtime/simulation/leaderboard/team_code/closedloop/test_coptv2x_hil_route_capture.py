from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CLOSEDLOOP = Path(__file__).resolve().parent
MODULE_PATH = CLOSEDLOOP / "coptv2x_hil_route_capture.py"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

import coptv2x_hil_route_capture as capture_module


class HilRouteCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_capture(self, directory=None):
        return capture_module.RouteInputCapture(
            directory or (self.root / "route-input"),
            route_id="Town05:_1:r2",
            scenario_id="scenario-0",
        )

    def test_captures_exact_backbone_tensor_and_writes_hashed_manifest(self):
        capture = self.make_capture()
        tensor = np.zeros(capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                          dtype=np.float32)

        record = capture.capture_frame(
            {"spatial_features": tensor}, source_frame=17,
            sample_id="carla-frame-17",
        )
        manifest_path = capture.finalize()

        self.assertEqual("carla-frame-17", record["sample_id"])
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(capture_module.ROUTE_CAPTURE_SCHEMA_VERSION,
                         manifest["schema_version"])
        self.assertEqual("integration-input-only", manifest["purpose"])
        self.assertEqual("Town05:_1:r2", manifest["route_id"])
        self.assertEqual("scenario-0", manifest["scenario_id"])
        self.assertEqual([record], manifest["frames"])
        feature_path = Path(record["spatial_features_path"])
        self.assertTrue(feature_path.is_absolute())
        self.assertEqual(hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                         record["spatial_features_sha256"])
        loaded = np.load(feature_path, allow_pickle=False, mmap_mode="r")
        self.assertEqual(capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                         loaded.shape)
        self.assertEqual(np.dtype("float32"), loaded.dtype)

    def test_rejects_missing_or_malformed_tensor(self):
        capture = self.make_capture()
        with self.assertRaisesRegex(ValueError, "spatial_features"):
            capture.capture_frame({}, 1, "carla-frame-1")
        with self.assertRaisesRegex(ValueError, "shape"):
            capture.capture_frame(
                {"spatial_features": np.zeros((2, 2), dtype=np.float32)},
                2, "carla-frame-2",
            )
        class WrongDtypeTensor:
            shape = capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE
            dtype = "float64"

        with self.assertRaisesRegex(ValueError, "dtype"):
            capture.capture_frame(
                {"spatial_features": WrongDtypeTensor()},
                3, "carla-frame-3",
            )
        self.assertEqual((), capture.frames)

    def test_rejects_incorrect_frame_identity_and_duplicates(self):
        capture = self.make_capture()
        tensor = np.zeros(capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                          dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "sample_id"):
            capture.capture_frame({"spatial_features": tensor}, 4,
                                  "carla-frame-5")
        with self.assertRaisesRegex(ValueError, "source_frame"):
            capture.capture_frame({"spatial_features": tensor}, True,
                                  "carla-frame-1")
        capture.capture_frame({"spatial_features": tensor}, 4,
                              "carla-frame-4")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture.capture_frame({"spatial_features": tensor}, 4,
                                  "carla-frame-4")

    def test_does_not_overwrite_existing_frame_or_manifest(self):
        directory = self.root / "existing"
        directory.mkdir()
        existing = directory / "spatial_features_00000007.npy"
        existing.write_bytes(b"preserve me")
        capture = self.make_capture(directory)
        tensor = np.zeros(capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                          dtype=np.float32)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            capture.capture_frame({"spatial_features": tensor}, 7,
                                  "carla-frame-7")
        self.assertEqual(b"preserve me", existing.read_bytes())

        manifest_path = directory / capture_module.ROUTE_CAPTURE_MANIFEST_NAME
        manifest_path.write_text("preserve manifest", encoding="utf-8")
        capture.capture_frame(
            {"spatial_features": np.zeros(
                capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                dtype=np.float32,
            )},
            8, "carla-frame-8",
        )
        with self.assertRaisesRegex(FileExistsError, "manifest"):
            capture.finalize()
        self.assertEqual("preserve manifest",
                         manifest_path.read_text(encoding="utf-8"))

    def test_finalize_requires_at_least_one_frame_and_closes_capture(self):
        capture = self.make_capture()
        with self.assertRaisesRegex(ValueError, "at least one"):
            capture.finalize()
        capture.capture_frame(
            {"spatial_features": np.zeros(
                capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                dtype=np.float32,
            )},
            0, "carla-frame-0",
        )
        capture.finalize()
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            capture.capture_frame(
                {"spatial_features": np.zeros((1,), dtype=np.float32)},
                1, "carla-frame-1",
            )

    def test_backbone_hook_captures_only_the_active_route_frame(self):
        class FakeBackbone:
            def __init__(self):
                self.pre_hooks = []

            def register_forward_pre_hook(self, hook):
                self.pre_hooks.append(hook)
                return type("Handle", (), {"remove": lambda _: self.pre_hooks.remove(hook)})()

            def __call__(self, batch_dict):
                for hook in tuple(self.pre_hooks):
                    hook(self, (batch_dict,))

        class FakeModel:
            def __init__(self):
                self.backbone = FakeBackbone()

        capture = self.make_capture()
        model = FakeModel()
        hook = capture_module.RouteInputCaptureHook(capture, model)
        tensor = np.zeros(capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                          dtype=np.float32)

        hook.begin_frame(31)
        model.backbone({"spatial_features": tensor})
        model.backbone({"spatial_features": tensor})
        record = hook.finish_frame()
        hook.close()
        manifest_path = capture.finalize()

        self.assertEqual(31, record["source_frame"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual([31], [item["source_frame"] for item in manifest["frames"]])
        self.assertEqual([], model.backbone.pre_hooks)

    def test_backbone_hook_skips_single_agent_warmup_and_keeps_first_model_input(self):
        class FakeBackbone:
            def register_forward_pre_hook(self, hook):
                self.hook = hook
                return type("Handle", (), {"remove": lambda _: None})()

            def __call__(self, batch_dict):
                self.hook(self, (batch_dict,))

        class FakeModel:
            backbone = FakeBackbone()

        hook = capture_module.RouteInputCaptureHook(self.make_capture(), FakeModel())
        hook.begin_frame(7)
        singleton = np.zeros((1, 64, 256, 512), dtype=np.float32)
        FakeModel.backbone({"spatial_features": singleton})
        self.assertIsNone(hook.finish_frame())
        hook.begin_frame(8)
        tensor = np.zeros(capture_module.EXPECTED_SPATIAL_FEATURES_SHAPE,
                          dtype=np.float32)
        FakeModel.backbone({"spatial_features": tensor})
        FakeModel.backbone({"spatial_features": singleton})
        record = hook.finish_frame()
        self.assertEqual(8, record["source_frame"])
        self.assertEqual(1, len(hook.capture.frames))
        hook.close()


if __name__ == "__main__":
    unittest.main()
