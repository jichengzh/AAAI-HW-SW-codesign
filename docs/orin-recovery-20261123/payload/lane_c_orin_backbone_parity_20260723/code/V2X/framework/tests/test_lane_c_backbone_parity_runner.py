from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from tools.orin_deploy.lane_c_backbone_parity_runner import (
    builder_flags_for_precision,
    summarize_latency_samples,
    validate_primary_protocol,
    validate_heldout_inputs,
    validate_output_paths,
)


class LaneCBackboneParityRunnerTest(unittest.TestCase):
    def test_builder_flags_match_h800_contract(self) -> None:
        self.assertEqual(builder_flags_for_precision("fp16"), ["fp16"])
        self.assertEqual(
            builder_flags_for_precision("int8"), ["int8", "fp16_fallback"]
        )

    def test_validate_heldout_inputs_requires_batch2_backbone_shape(self) -> None:
        valid = np.zeros((3, 2, 64, 128, 256), dtype=np.float16)
        self.assertEqual(validate_heldout_inputs(valid), [3, 2, 64, 128, 256])
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_heldout_inputs(np.zeros((3, 1, 64, 128, 256), dtype=np.float32))

    def test_latency_summary_uses_h800_order_statistic_contract(self) -> None:
        samples = [float(value) for value in range(1, 101)]
        summary = summarize_latency_samples(samples)
        self.assertEqual(summary["median_ms"], 50.5)
        self.assertEqual(summary["p90_ms"], 90.0)
        self.assertEqual(summary["p99_ms"], 99.0)
        self.assertEqual(summary["mean_ms"], 50.5)

    def test_primary_protocol_is_locked_to_h800_contract(self) -> None:
        validate_primary_protocol(warmup=20, iters=300, repeat=5)
        with self.assertRaisesRegex(ValueError, "warmup=20"):
            validate_primary_protocol(warmup=0, iters=300, repeat=5)

    def test_output_paths_are_confined_to_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            valid = argparse.Namespace(
                command="numerical",
                artifact_root=root,
                output_npz=root / "numerical" / "outputs.npz",
                report_json=root / "numerical" / "report.json",
            )
            validate_output_paths(valid)
            escaped = argparse.Namespace(
                command="numerical",
                artifact_root=root,
                output_npz=root.parent / "escaped.npz",
                report_json=root / "numerical" / "report.json",
            )
            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                validate_output_paths(escaped)

    @mock.patch(
        "tools.orin_deploy.lane_c_backbone_parity_runner.subprocess.run"
    )
    def test_privileged_tegrastats_stop_uses_stop_command(
        self, run: mock.Mock
    ) -> None:
        from tools.orin_deploy.lane_c_backbone_parity_runner import _stop_tegrastats

        process = mock.Mock()
        handle = mock.Mock()
        with mock.patch.dict("os.environ", {"ORIN_SUDO_PW": "temporary"}):
            _stop_tegrastats(process, handle, privileged=True)
        run.assert_called_once_with(
            ["sudo", "-S", "-p", "", "tegrastats", "--stop"],
            input="temporary\n",
            text=True,
            check=True,
            timeout=5,
            env=mock.ANY,
        )
        self.assertNotIn("ORIN_SUDO_PW", run.call_args.kwargs["env"])
        process.wait.assert_called_once_with(timeout=5)
        handle.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
