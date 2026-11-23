"""Frozen per-run contract snapshots avoid repeated YAML parsing, not checks."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import coptv2x_hil_backend as backend_module
from test_coptv2x_hil_backend import ArtifactTree, FakeRuntime, _trigger


class FeatureValidatorCacheTests(unittest.TestCase):
    def test_cached_registry_is_loaded_once_but_checks_every_payload(self):
        tree = ArtifactTree()
        self.addCleanup(tree.close)
        backend = backend_module.CoptV2xHilBackend(
            "joint_shcosearch", tree.artifact_root, tree.config_dir,
            runtime_factory=lambda spec, paths, inputs, outputs, root: FakeRuntime(spec, inputs, outputs),
            rsu_only=True)
        import numpy as np
        result = backend.infer(np.zeros((2, 64, 256, 512), dtype=np.float32),
                               _trigger("joint_shcosearch"), ("ego", "rsu"))
        identity = backend_module._plain_artifact_identity(backend.spec)
        with mock.patch.object(backend_module, "load_hil_artifact_registry",
                               wraps=backend_module.load_hil_artifact_registry) as loader:
            validator = backend_module.create_feature_payload_validator(tree.config_dir)
            for _ in range(2):
                validator(identity, result.outputs, payload_semantics=result.payload_semantics)
            self.assertEqual(1, loader.call_count)
            wrong = dict(identity, widths=[1, 2, 3])
            with self.assertRaises(backend_module.HilBackendError):
                validator(wrong, result.outputs, payload_semantics=result.payload_semantics)
            invalid = dict(result.outputs)
            invalid["res0"] = np.full(result.outputs["res0"].shape, np.nan, dtype=np.float32)
            with self.assertRaises(backend_module.HilBackendError):
                validator(identity, invalid, payload_semantics=result.payload_semantics)
            with self.assertRaises(backend_module.HilBackendError):
                validator(identity, result.outputs)


if __name__ == "__main__":
    unittest.main()
