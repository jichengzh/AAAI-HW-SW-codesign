# -*- coding: utf-8 -*-
"""Fast contract tests for the action-FIFO experiment worker."""

import os
import importlib.util
import csv
import json
import subprocess
import tempfile
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "action_fifo_multiroute_sc.sh")
RESULT_PARSER = os.path.join(REPO, "scripts", "action_fifo_result.py")


def load_result_parser():
    spec = importlib.util.spec_from_file_location("action_fifo_result", RESULT_PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionFifoHarnessTest(unittest.TestCase):
    def test_check_config_reports_each_frozen_arm(self):
        for delay_ms, delay_frames in ((0, 0), (200, 4), (500, 10), (800, 16)):
            with self.subTest(delay_ms=delay_ms):
                result = subprocess.run(
                    ["bash", SCRIPT, "--check-config", str(delay_ms)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("delay_ms=%d" % delay_ms, result.stdout)
                self.assertIn("delay_frames=%d" % delay_frames, result.stdout)
                self.assertIn("codriving_fifo%d_l1" % delay_ms, result.stdout)

    def test_check_config_rejects_unapproved_delay(self):
        result = subprocess.run(
            ["bash", SCRIPT, "--check-config", "123"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported action FIFO delay", result.stdout)

    def test_worker_keeps_route_level_evaluator_logs(self):
        with open(SCRIPT) as stream:
            script = stream.read()

        self.assertIn("eval_fifo${DELAY}_r${route}_p${PORT}_n${REPEAT}.log", script)
        self.assertIn("> \"$eval_log\" 2>&1", script)

    def test_failed_route_result_is_countable_and_not_infra_retry(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(
                {
                    "_checkpoint": {
                        "global_record": {
                            "status": "Failed",
                            "scores": {"score_route": 6.389, "score_composed": 0.0},
                        }
                    }
                },
                stream,
            )
            stream.flush()

            result = subprocess.run(
                ["bash", SCRIPT, "--check-result", stream.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("route_result_status=Failed", result.stdout)

    def test_missing_route_result_is_infra_retry(self):
        result = subprocess.run(
            ["bash", SCRIPT, "--check-result", "/tmp/does-not-exist-action-fifo.json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("route_result_missing_or_unparseable", result.stdout)

    def test_failed_route_keeps_raw_score_and_exact_collision_fields(self):
        parser = load_result_parser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(
                {
                    "_checkpoint": {
                        "global_record": {
                            "status": "Failed",
                            "scores": {
                                "score_route": 6.389,
                                "score_composed": 4.153,
                            },
                            "infractions": {
                                "collisions_layout": 1.25,
                                "collisions_vehicle": 0.5,
                                "vehicle_blocked": 9.0,
                            },
                        }
                    }
                },
                stream,
            )
            stream.flush()
            result = parser.parse_route_result(stream.name)

        self.assertEqual(result["status"], "Failed")
        self.assertEqual(result["DS"], 4.153)
        self.assertEqual(result["RC"], 6.389)
        self.assertEqual(result["coll_lay"], 1.25)
        self.assertEqual(result["coll_veh"], 0.5)

    def test_result_parser_missing_file_returns_explicit_failed_defaults(self):
        parser = load_result_parser()

        result = parser.parse_route_result("/tmp/missing-action-fifo-result.json")

        self.assertEqual(
            result,
            {
                "status": "FAILED",
                "DS": 0.0,
                "RC": 0.0,
                "coll_lay": 0.0,
                "coll_veh": 0.0,
            },
        )

    def test_result_parser_main_appends_complete_summary_row(self):
        parser = load_result_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "result.json")
            summary_path = os.path.join(tmpdir, "summary.csv")
            with open(raw_path, "w") as stream:
                json.dump(
                    {
                        "_checkpoint": {
                            "global_record": {
                                "status": "Failed",
                                "scores": {
                                    "score_route": 25.0,
                                    "score_composed": 12.5,
                                },
                                "infractions": {
                                    "collisions_layout": ["a", "b"],
                                    "collisions_vehicle": ["c"],
                                },
                            }
                        }
                    },
                    stream,
                )

            parser.main(
                [
                    parser.__file__,
                    raw_path,
                    "317",
                    "2",
                    "200",
                    summary_path,
                ]
            )
            with open(summary_path, newline="") as stream:
                row = next(csv.reader(stream))

        self.assertEqual(row[:4], ["317", "2", "200", "4"])
        self.assertEqual(row[4:9], ["12.5", "25.0", "Failed", "2.0", "1.0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
