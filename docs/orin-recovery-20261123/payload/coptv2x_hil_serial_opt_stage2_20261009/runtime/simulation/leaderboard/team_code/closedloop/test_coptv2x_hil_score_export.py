"""Keep valid driving-terminal scores rather than exporting blank metrics."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from coptv2x_hil_result import paired_row


class ScoreExportTests(unittest.TestCase):
    def test_native_carla_scores_export_for_completed_and_driving_terminal(self):
        record = {"scores": {"score_composed": 39.25, "score_route": 80.5}}
        for classification in ("COMPLETED", "DRIVING_TERMINAL"):
            row = paired_row("task", "arm", classification, record)
            self.assertEqual(39.25, row["DS"])
            self.assertEqual(80.5, row["RC"])
            self.assertTrue(row["countable"])

    def test_legacy_metrics_remain_unchanged_but_infrastructure_never_counts(self):
        record = {"DS": 0.5, "RC": 0.8}
        self.assertEqual(0.5, paired_row("task", "arm", "COMPLETED", record)["DS"])
        row = paired_row("task", "arm", "INFRA_INVALID", record)
        self.assertIsNone(row["DS"])
        self.assertFalse(row["countable"])
