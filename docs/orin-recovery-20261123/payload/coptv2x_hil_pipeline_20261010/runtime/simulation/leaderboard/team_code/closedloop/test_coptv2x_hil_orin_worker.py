"""Contract checks for the trigger-driven Orin-local frozen-model worker."""

import json
import contextlib
import io
from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation/leaderboard/team_code/closedloop"))

from scripts import coptv2x_hil_orin_worker as worker


class OrinWorkerContracts(unittest.TestCase):
    def worker_cli(self):
        return ["--data-root", "/inputs", "--manifest", "/inputs/manifest.json",
                "--artifact-root", "/artifacts", "--native-repo-root", "/native",
                "--result-count", "1"]

    def test_wire_compression_defaults_to_none(self):
        args = worker.build_parser().parse_args(self.worker_cli())
        self.assertEqual(args.wire_compression, "none")

    def test_serial_optimization_is_explicit_and_uncompressed(self):
        args = worker.build_parser().parse_args(self.worker_cli() + [
            "--serial-optimized", "--preload-static-inputs",
            "--max-preloaded-tensor-mib", "128"])
        self.assertEqual(worker.backend_options(args), {
            "rsu_only": True, "trust_readonly_inputs": True, "checksum_dedup": True})
        self.assertEqual(worker.source_options(args), {
            "preload_verified": True, "max_preloaded_tensor_bytes": 128 * 1024 * 1024})
        args.wire_compression = "zlib"
        with self.assertRaisesRegex(ValueError, "uncompressed"):
            worker.backend_options(args)

    def test_legacy_namespaces_do_not_enable_fast_paths(self):
        self.assertEqual(worker.backend_options(Namespace()), {})
        self.assertEqual(worker.source_options(Namespace()), {})

    def test_static_preload_requires_positive_bound(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            worker.source_options(Namespace(preload_static_inputs=True,
                                            max_preloaded_tensor_mib=0))

    def test_wire_compression_accepts_all_lossless_options(self):
        for compression in ("none", "zlib", "lz4"):
            with self.subTest(compression=compression):
                args = worker.build_parser().parse_args(
                    self.worker_cli() + ["--wire-compression", compression])
                self.assertEqual(args.wire_compression, compression)

    def test_wire_compression_rejects_unknown_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            worker.build_parser().parse_args(
                self.worker_cli() + ["--wire-compression", "fp16"])
        self.assertEqual(error.exception.code, 2)

    def test_worker_passes_compression_to_real_pipeline_and_preserves_legacy_default(self):
        trigger = worker.parse_worker_trigger_line((json.dumps({
            "schema_version": "coptv2x.hil-trigger.v1",
            "run_id": "smoke-run", "route_id": "Town05-route-2",
            "scenario_id": "scenario1", "sample_id": "carla-frame-1",
            "sequence_no": 0, "source_frame": 1,
            "source_sim_time_ns": 50000000, "arm_id": "joint_shcosearch",
        }) + "\n").encode("utf-8"))
        for configured, expected in ((None, "none"), ("none", "none"),
                                     ("zlib", "zlib"), ("lz4", "lz4")):
            with self.subTest(compression=configured):
                instance = worker.OrinHilWorker.__new__(worker.OrinHilWorker)
                args = Namespace(result_count=1, artifact_root="/artifacts",
                                 config_dir="/configs", native_repo_root="/native", max_payload_mib=64)
                if configured is not None:
                    args.wire_compression = configured
                instance.args = args
                instance.active_key = None
                instance.processed = 0
                instance.connection = object()
                instance.source = SimpleNamespace(frame_for_trigger=lambda value: object())
                result = SimpleNamespace(header={"request_id": "smoke-run:0"})
                # GPU loading, tensor inference and network I/O are external to configuration wiring.
                with patch.object(instance, "_release_arm"), \
                        patch.object(worker, "CoptV2xHilBackend") as backend, \
                        patch.object(worker.CoptV2xHilPipeline, "process", return_value=result), \
                        patch.object(worker, "_prepare_result_frame", return_value=({}, b"")), \
                        patch.object(worker, "send_wire_frame"):
                    backend.return_value.payload_semantics = "codriving_multiscale_features"
                    self.assertEqual(instance.process_trigger(trigger), "smoke-run:0")
                self.assertIsInstance(instance.pipeline, worker.CoptV2xHilPipeline)
                self.assertEqual(instance.pipeline.compression, expected)
                self.assertEqual(instance.processed, 1)

    def test_worker_accepts_only_exact_frozen_trigger_identity(self):
        trigger = {
            "schema_version": "coptv2x.hil-trigger.v1",
            "run_id": "smoke-run", "route_id": "Town05-route-2",
            "scenario_id": "scenario1", "sample_id": "carla-frame-1",
            "sequence_no": 0, "source_frame": 1,
            "source_sim_time_ns": 50000000, "arm_id": "joint_shcosearch",
        }
        self.assertEqual(trigger, worker.parse_worker_trigger_line(
            (json.dumps(trigger) + "\n").encode("utf-8")
        ).to_metadata())
        with self.assertRaisesRegex(ValueError, "trigger"):
            worker.parse_worker_trigger_line(b"{}\n")


if __name__ == "__main__":
    unittest.main()
