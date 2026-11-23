"""Explicit deployment wiring for the approved serial, uncompressed worker."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.coptv2x_hil_episode_services import service_configuration


class SerialServiceContracts(unittest.TestCase):
    def configuration(self, **options):
        args = SimpleNamespace(runtime="/frozen/runtime", container="existing-ros",
                               orin_python="/python", artifact_root="/artifacts",
                               result_port=55022, result_count=20, worker_port=56812,
                               forward_port=56811, ready_timeout_s=180, **options)
        return service_configuration(args, {"token": "new-episode", "arm_id": "compression_only"})

    def test_optimized_worker_flags_are_propagated_without_enabling_codec(self):
        config = self.configuration(serial_optimized=True, preload_static_inputs=True,
                                    max_preloaded_tensor_mib=128)
        worker = next(row for row in config["services"] if row["name"] == "worker")
        self.assertIn("--serial-optimized", worker["argv"])
        self.assertIn("--preload-static-inputs", worker["argv"])
        self.assertEqual(worker["argv"][worker["argv"].index("--max-preloaded-tensor-mib") + 1], "128")
        self.assertEqual(worker["argv"][worker["argv"].index("--wire-compression") + 1], "none")

    def test_serial_service_rejects_codec_and_latest_policy(self):
        for options in ({"wire_compression": "zlib"}, {"trigger_policy": "latest"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.configuration(serial_optimized=True, **options)

    def test_legacy_service_does_not_preload_or_project(self):
        config = self.configuration()
        worker = next(row for row in config["services"] if row["name"] == "worker")
        self.assertNotIn("--serial-optimized", worker["argv"])
        self.assertNotIn("--preload-static-inputs", worker["argv"])
        self.assertNotIn("--startup-arm-id", worker["argv"])
        self.assertNotIn("--runtime-warmup-count", worker["argv"])

    def test_serial_worker_initializes_known_arm_before_ready_and_warms_once(self):
        for count in (None, 0, 2):
            options = {} if count is None else {"runtime_warmup_count": count}
            config = self.configuration(serial_optimized=True, **options)
            argv = next(row["argv"] for row in config["services"] if row["name"] == "worker")
            self.assertEqual(argv[argv.index("--startup-arm-id") + 1], "compression_only")
            self.assertEqual(argv[argv.index("--runtime-warmup-count") + 1], str(1 if count is None else count))

    def test_serial_service_rejects_invalid_warmup_counts(self):
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.configuration(serial_optimized=True, runtime_warmup_count=count)


if __name__ == "__main__":
    unittest.main()
