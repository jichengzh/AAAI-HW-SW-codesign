"""Contract checks for the trigger-driven Orin-local frozen-model worker."""

import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation/leaderboard/team_code/closedloop"))

from scripts import coptv2x_hil_orin_worker as worker


class OrinWorkerContracts(unittest.TestCase):
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
