from __future__ import print_function

from dataclasses import FrozenInstanceError, fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from coptv2x_hil_artifacts import (  # noqa: E402
    HIL_ARM_IDS,
    HilArtifactError,
    HilArtifactSpec,
    get_hil_artifact,
    load_hil_artifact_registry,
    preflight_hil_artifacts,
)


EXPECTED_ARMS = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "joint_shcosearch",
)
EXPECTED_OUTPUT_CONTRACTS = {
    "original_default": (
        ("res0", (2, 64, 128, 256), "float32"),
        ("res1", (2, 128, 64, 128), "float32"),
        ("res2", (2, 256, 32, 64), "float32"),
    ),
    "compression_only": (
        ("res0", (2, 32, 128, 256), "float32"),
        ("res1", (2, 64, 64, 128), "float32"),
        ("res2", (2, 96, 32, 64), "float32"),
    ),
    "schedule_only": (
        ("res0", (2, 64, 128, 256), "float32"),
        ("res1", (2, 128, 64, 128), "float32"),
        ("res2", (2, 256, 32, 64), "float32"),
    ),
    "compress_then_tune": (
        ("res0", (2, 32, 128, 256), "float32"),
        ("res1", (2, 80, 64, 128), "float32"),
        ("res2", (2, 128, 32, 64), "float32"),
    ),
    "joint_shcosearch": (
        ("res0", (2, 16, 128, 256), "float32"),
        ("res1", (2, 32, 64, 128), "float32"),
        ("res2", (2, 64, 32, 64), "float32"),
    ),
}
REPO_ROOT = Path(__file__).resolve().parents[4]
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "coptv2x_hil_preflight.py"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract(widths, dtype):
    return {
        "input_contract": {
            "spatial_features": {
                "shape": [2, 64, 256, 512],
                "dtype": "float32",
            }
        },
        "output_contract": {
            "res0": {
                "shape": [2, widths[0], 128, 256],
                "dtype": dtype,
            },
            "res1": {
                "shape": [2, widths[1], 64, 128],
                "dtype": dtype,
            },
            "res2": {
                "shape": [2, widths[2], 32, 64],
                "dtype": dtype,
            },
        },
    }


class ArtifactTree(object):
    def __init__(self, root):
        self.root = Path(root)
        self.config_dir = self.root / "registry"
        self.artifact_root = self.root / "artifacts"
        self.config_dir.mkdir()
        self.artifact_root.mkdir()
        self.documents = {}
        definitions = (
            (
                "original_default", [64, 128, 256], "fp32", "native_pytorch",
                "net_epoch29.pth", None,
            ),
            (
                "compression_only", [32, 64, 96], "fp16", "tensorrt",
                "net_epoch_bestval_at11.pth", "compression_only",
            ),
            (
                "schedule_only", [64, 128, 256], "fp32", "tensorrt",
                "net_epoch29.pth", "schedule_only",
            ),
            (
                "compress_then_tune", [32, 80, 128], "fp16", "tensorrt",
                "net_epoch_bestval_at5.pth", "compress_then_tune",
            ),
            (
                "joint_shcosearch", [16, 32, 64], "fp16", "tensorrt",
                "net_epoch_bestval_at9.pth", "joint_shcosearch_fp16",
            ),
        )
        for definition in definitions:
            self._write_arm(*definition)

    def _write_payload(self, relative_path, payload):
        path = self.artifact_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return path

    def _write_arm(
        self, arm_id, widths, precision, runtime, checkpoint_name, artifact_arm
    ):
        source_dir = "source/codriving/%03dx%03dx%03d" % tuple(widths)
        config = "%s/config.yaml" % source_dir
        checkpoint = "%s/%s" % (source_dir, checkpoint_name)
        config_path = self._write_payload(config, ("config:" + arm_id).encode("ascii"))
        checkpoint_path = self._write_payload(
            checkpoint, ("checkpoint:" + arm_id).encode("ascii")
        )
        engine = None
        hashes = {
            "config": _sha256(config_path),
            "checkpoint": _sha256(checkpoint_path),
        }
        if artifact_arm is not None:
            engine = "codriving/%s/backbone.engine" % artifact_arm
            engine_path = self._write_payload(
                engine, ("engine:" + artifact_arm).encode("ascii")
            )
            hashes["engine"] = _sha256(engine_path)
        document = {
            "arm_id": arm_id,
            "model": "codriving",
            "widths": widths,
            "precision": precision,
            "runtime": runtime,
            "config": config,
            "checkpoint": checkpoint,
            "engine": engine,
            "artifact_sha256": hashes,
        }
        # TensorRT precision controls internal kernels; the frozen ONNX/native
        # boundary remains float32 for every arm.
        document.update(_contract(widths, "float32"))
        self.documents[arm_id] = document
        self.save(arm_id)

    def save(self, arm_id):
        path = self.config_dir / ("artifact_%s.yaml" % arm_id)
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.documents[arm_id], stream, sort_keys=False)


class HilArtifactRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tree = ArtifactTree(self.temporary.name)

    def test_registry_contains_exactly_five_fixed_arms(self):
        registry = load_hil_artifact_registry(self.tree.config_dir)

        self.assertEqual(EXPECTED_ARMS, HIL_ARM_IDS)
        self.assertEqual(EXPECTED_ARMS, tuple(registry))
        self.assertEqual(
            (
                "arm_id",
                "model",
                "widths",
                "precision",
                "runtime",
                "config",
                "checkpoint",
                "engine",
                "artifact_sha256",
                "input_contract",
                "output_contract",
            ),
            tuple(field.name for field in fields(HilArtifactSpec)),
        )

    def test_each_frozen_yaml_matches_orin_multiscale_output_authority(self):
        registry = load_hil_artifact_registry()

        for arm_id in EXPECTED_ARMS:
            with self.subTest(arm_id=arm_id):
                actual = tuple(
                    (tensor.name, tensor.shape, tensor.dtype)
                    for tensor in registry[arm_id].output_contract
                )
                self.assertEqual(EXPECTED_OUTPUT_CONTRACTS[arm_id], actual)

    def test_specs_and_nested_contracts_are_immutable(self):
        spec = load_hil_artifact_registry(self.tree.config_dir)["compression_only"]

        with self.assertRaises(FrozenInstanceError):
            spec.runtime = "native_pytorch"
        with self.assertRaises(TypeError):
            spec.artifact_sha256["engine"] = "0" * 64
        with self.assertRaises(FrozenInstanceError):
            spec.output_contract[0].dtype = "float32"

    def test_coptv2x_maps_only_to_existing_fp16_joint_engine(self):
        spec = load_hil_artifact_registry(self.tree.config_dir)["joint_shcosearch"]

        self.assertEqual("fp16", spec.precision)
        self.assertEqual("tensorrt", spec.runtime)
        self.assertEqual(
            Path("codriving/joint_shcosearch_fp16/backbone.engine"), spec.engine
        )
        self.assertNotIn("joint_shcosearch/backbone.engine", str(spec.engine))

    def test_unknown_arm_is_rejected(self):
        registry = load_hil_artifact_registry(self.tree.config_dir)

        with self.assertRaisesRegex(HilArtifactError, "unknown HIL arm"):
            get_hil_artifact("candidate_17", registry)

    def test_non_string_unknown_yaml_field_is_rejected_cleanly(self):
        self.tree.documents["original_default"][7] = "invalid YAML field"
        self.tree.documents["original_default"]["extra"] = "invalid YAML field"
        self.tree.save("original_default")

        with self.assertRaisesRegex(HilArtifactError, "fields mismatch"):
            load_hil_artifact_registry(self.tree.config_dir)

    def test_missing_artifact_file_is_rejected(self):
        engine = self.artifact_root("compression_only", "engine")
        engine.unlink()

        with self.assertRaisesRegex(HilArtifactError, "missing engine"):
            preflight_hil_artifacts(self.tree.config_dir, self.tree.artifact_root)

    def test_artifact_sha_mismatch_is_rejected(self):
        checkpoint = self.artifact_root("schedule_only", "checkpoint")
        checkpoint.write_bytes(b"tampered")

        with self.assertRaisesRegex(HilArtifactError, "SHA-256 mismatch"):
            preflight_hil_artifacts(self.tree.config_dir, self.tree.artifact_root)

    def test_early_joint_int8_engine_is_rejected_even_with_matching_sha(self):
        early_engine = "codriving/joint_shcosearch/backbone.engine"
        early_path = self.tree._write_payload(early_engine, b"early-int8-engine")
        document = self.tree.documents["joint_shcosearch"]
        document["engine"] = early_engine
        document["artifact_sha256"]["engine"] = _sha256(early_path)
        self.tree.save("joint_shcosearch")

        with self.assertRaisesRegex(HilArtifactError, "joint_shcosearch_fp16"):
            load_hil_artifact_registry(self.tree.config_dir)

    def test_preflight_reports_names_shapes_dtypes_runtime_and_hashes(self):
        records = preflight_hil_artifacts(
            self.tree.config_dir, self.tree.artifact_root
        )

        self.assertEqual(EXPECTED_ARMS, tuple(record["arm_id"] for record in records))
        self.assertEqual("codriving:original_default", records[0]["artifact_id"])
        self.assertEqual("native_pytorch", records[0]["runtime"])
        self.assertEqual({"config", "checkpoint"}, set(records[0]["artifact_sha256"]))
        self.assertEqual(
            {
                "name": "spatial_features",
                "shape": [2, 64, 256, 512],
                "dtype": "float32",
            },
            records[0]["input_contract"][0],
        )
        self.assertEqual(["res0", "res1", "res2"], [
            tensor["name"] for tensor in records[4]["output_contract"]
        ])
        self.assertEqual("float32", records[4]["output_contract"][0]["dtype"])

    def test_preflight_cli_prints_json_for_all_five_arms(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PREFLIGHT_SCRIPT),
                "--config-dir",
                str(self.tree.config_dir),
                "--artifact-root",
                str(self.tree.artifact_root),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual("ok", payload["status"])
        self.assertEqual(5, payload["artifact_count"])
        self.assertEqual(list(EXPECTED_ARMS), [
            artifact["arm_id"] for artifact in payload["artifacts"]
        ])

    def test_preflight_cli_exits_nonzero_on_inconsistency(self):
        self.artifact_root("compression_only", "engine").write_bytes(b"tampered")

        completed = subprocess.run(
            [
                sys.executable,
                str(PREFLIGHT_SCRIPT),
                "--config-dir",
                str(self.tree.config_dir),
                "--artifact-root",
                str(self.tree.artifact_root),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("SHA-256 mismatch", completed.stderr)

    def artifact_root(self, arm_id, artifact):
        relative_path = self.tree.documents[arm_id][artifact]
        return self.tree.artifact_root / relative_path


if __name__ == "__main__":
    unittest.main()
