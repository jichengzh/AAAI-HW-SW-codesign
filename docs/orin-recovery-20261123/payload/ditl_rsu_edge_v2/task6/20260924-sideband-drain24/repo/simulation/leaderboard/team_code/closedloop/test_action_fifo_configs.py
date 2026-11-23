# -*- coding: utf-8 -*-
"""Contract tests for the four action-FIFO sensitivity configurations."""

import os
import unittest

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(HERE, os.pardir, "agent_config"))


class ActionFifoConfigTest(unittest.TestCase):
    def test_four_arms_isolate_action_delay(self):
        for delay_ms in (0, 200, 500, 800):
            with self.subTest(delay_ms=delay_ms):
                path = os.path.join(
                    CONFIG_DIR,
                    "pnp_config_codriving_fifo%d_l1.yaml" % delay_ms,
                )
                with open(path) as stream:
                    config = yaml.safe_load(stream)
                simulation = config["simulation"]
                self.assertEqual(simulation["action_fifo_delay_ms"], delay_ms)
                self.assertTrue(simulation["action_fifo_audit"])
                self.assertEqual(simulation["latency_inject_ms"], 0)
                self.assertEqual(simulation["tau_ego_ms"], 0)
                self.assertTrue(simulation["l1_control"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
