from __future__ import print_function

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from ditl_stage_summary import summarize_stage_records
from ditl_trace_schema import write_trace_jsonl
from test_ditl_trace_schema import valid_v2_record


REPO_ROOT = Path(__file__).resolve().parents[4]
SUMMARY_CLI = REPO_ROOT / "scripts" / "ditl_summarize_stages.py"


class DitlStageSummaryTests(unittest.TestCase):
    def test_summary_reports_percentiles_payload_and_stage_residual(self):
        cold = valid_v2_record(0)
        warm_a = valid_v2_record(1)
        warm_b = valid_v2_record(2)
        warm_a["durations_ns"]["gpu_inference"] = 40_000_000
        warm_b["durations_ns"]["gpu_inference"] = 60_000_000

        summary = summarize_stage_records([cold, warm_a, warm_b], warmup_count=1)

        self.assertEqual(1, summary["samples"]["warmup"])
        self.assertEqual(2, summary["samples"]["measured"])
        self.assertEqual(50.0, summary["stages"]["gpu_inference"]["p50_ms"])
        self.assertIn("p95_ms", summary["stages"]["gpu_inference"])
        self.assertIn("stage_sum_residual_ms", summary["totals"])
        self.assertEqual(2_621_440, summary["payload"]["raw_bytes"]["p50"])

    def test_summary_excludes_ineligible_cross_host_total(self):
        diagnostic = valid_v2_record(0, same_host=False)
        diagnostic["clock"]["receiver"]["sync_state"] = "unlocked"
        diagnostic["clock_comparability"] = {
            "comparable": False,
            "reason": "clock_not_comparable",
        }
        diagnostic["headline_eligible"] = False
        diagnostic["headline_ineligible_reason"] = "clock_not_comparable"

        summary = summarize_stage_records([diagnostic], warmup_count=0)

        self.assertEqual(0, summary["samples"]["headline_eligible"])
        self.assertNotIn("rsu_to_fusion", summary["totals"])

    def test_summary_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "trace.jsonl"
            target = Path(tmpdir) / "summary.json"
            write_trace_jsonl(source, [valid_v2_record(0), valid_v2_record(1)])
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARY_CLI),
                    "--trace-jsonl",
                    str(source),
                    "--output-json",
                    str(target),
                    "--warmup-count",
                    "1",
                ],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            self.assertEqual(0, result.returncode, result.stdout)
            summary = json.loads(target.read_text())
            self.assertEqual(1, summary["samples"]["measured"])


if __name__ == "__main__":
    unittest.main()
