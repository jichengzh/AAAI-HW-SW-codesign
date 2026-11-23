import os
import unittest
from glob import glob

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(HERE, os.pardir, "agent_config"))

ARMS = {
    "baseline": (0, 0),
    "rsu200": (200, 0),
    "rsu500": (500, 0),
    "rsu800": (800, 0),
    "ego200": (0, 200),
    "ego500": (0, 500),
    "ego800": (0, 800),
}
JOINT_ARMS = {
    "joint_rsu100_ego200": (100, 200),
    "joint_rsu100_ego300": (100, 300),
    "joint_rsu200_ego300": (200, 300),
    "joint_rsu100_ego400": (100, 400),
    "joint_rsu200_ego400": (200, 400),
    "joint_rsu300_ego400": (300, 400),
    "joint_rsu100_ego500": (100, 500),
    "joint_rsu200_ego500": (200, 500),
    "joint_rsu300_ego500": (300, 500),
    "joint_rsu400_ego500": (400, 500),
}
JOINT_C2_ARMS = {
    "jointc2_rsu100_ego200": (100, 200),
    "jointc2_rsu100_ego300": (100, 300),
    "jointc2_rsu200_ego300": (200, 300),
    "jointc2_rsu100_ego400": (100, 400),
    "jointc2_rsu200_ego400": (200, 400),
    "jointc2_rsu300_ego400": (300, 400),
    "jointc2_rsu100_ego500": (100, 500),
    "jointc2_rsu200_ego500": (200, 500),
    "jointc2_rsu300_ego500": (300, 500),
    "jointc2_rsu400_ego500": (400, 500),
}
EXPECTED_BASENAMES = {
    "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm
    for arm in tuple(ARMS) + tuple(JOINT_ARMS) + tuple(JOINT_C2_ARMS)
}
AUTHORITY_PATH = os.path.join(
    CONFIG_DIR, "pnp_config_codriving_lat_d0_l1.yaml"
)


class DualFifoConfigFilesTest(unittest.TestCase):
    def test_exactly_twenty_seven_canonical_files_exist(self):
        actual = {
            os.path.basename(path)
            for path in glob(
                os.path.join(
                    CONFIG_DIR, "pnp_config_codriving_dualfifo_*_l1.yaml"
                )
            )
        }
        self.assertEqual(actual, EXPECTED_BASENAMES)

    def test_joint_arms_use_strict_triangular_delays_and_fixed_control(self):
        with open(AUTHORITY_PATH) as stream:
            authority = yaml.safe_load(stream)
        for arm, expected in JOINT_ARMS.items():
            with self.subTest(arm=arm):
                path = os.path.join(CONFIG_DIR, "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm)
                with open(path) as stream:
                    config = yaml.safe_load(stream)
                simulation = config["simulation"]
                self.assertEqual((simulation["rsu_fifo_delay_ms"],
                                  simulation["ego_output_fifo_delay_ms"]), expected)
                self.assertEqual((config["control"]["max_speed"],
                                  config["control"]["target_point_distance"]), (5, 10))
                self.assertEqual(simulation["ego_num"], 1)
                self.assertEqual(simulation["skip_frames"], 4)
                self.assertEqual(simulation["tau_ego_ms"], 0)
                self.assertTrue(simulation["latency_audit"])
                for section in ("perception", "planning", "control", "RSU"):
                    self.assertEqual(config[section], authority[section])

    def test_joint_c2_arms_keep_the_joint_contract_at_exact_age_cadence(self):
        with open(AUTHORITY_PATH) as stream:
            authority = yaml.safe_load(stream)
        for arm, expected in JOINT_C2_ARMS.items():
            with self.subTest(arm=arm):
                path = os.path.join(CONFIG_DIR, "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm)
                with open(path) as stream:
                    config = yaml.safe_load(stream)
                simulation = config["simulation"]
                self.assertEqual((simulation["rsu_fifo_delay_ms"],
                                  simulation["ego_output_fifo_delay_ms"]), expected)
                self.assertEqual(
                    {key: simulation[key] for key in ("ego_num", "skip_frames", "l1_control",
                                                       "l1_debug", "tau_ego_ms", "latency_audit")},
                    {"ego_num": 1, "skip_frames": 2, "l1_control": True, "l1_debug": False,
                     "tau_ego_ms": 0, "latency_audit": True},
                )
                self.assertEqual((config["control"]["max_speed"],
                                  config["control"]["target_point_distance"]), (5, 10))
                for section in ("perception", "planning", "control", "RSU"):
                    self.assertEqual(config[section], authority[section])

    def test_seven_arms_are_single_variable_and_audited(self):
        for arm, expected in ARMS.items():
            with self.subTest(arm=arm):
                path = os.path.join(
                    CONFIG_DIR,
                    "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm,
                )
                with open(path) as stream:
                    config = yaml.safe_load(stream)
                sim = config["simulation"]
                self.assertEqual(
                    (
                        sim["rsu_fifo_delay_ms"],
                        sim["ego_output_fifo_delay_ms"],
                    ),
                    expected,
                )
                self.assertEqual(sim["ego_num"], 1)
                self.assertEqual(sim["skip_frames"], 4)
                self.assertEqual(sim["tau_ego_ms"], 0)
                self.assertTrue(sim["l1_control"])
                self.assertFalse(sim["l1_debug"])
                self.assertTrue(sim["latency_audit"])
                self.assertNotIn("latency_inject_ms", sim)
                self.assertNotIn("action_fifo_delay_ms", sim)
                self.assertNotIn("action_fifo_audit", sim)

    def test_all_arms_match_authority_and_keep_identical_model_contract(self):
        with open(AUTHORITY_PATH) as stream:
            authority = yaml.safe_load(stream)
        values = []
        for arm in ARMS:
            path = os.path.join(
                CONFIG_DIR,
                "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm,
            )
            with open(path) as stream:
                config = yaml.safe_load(stream)
            for section in ("perception", "planning", "control", "RSU"):
                self.assertEqual(config[section], authority[section])
            self.assertEqual(
                {
                    key: config["simulation"][key]
                    for key in ("ego_num", "skip_frames", "l1_control", "l1_debug")
                },
                {
                    "ego_num": 1,
                    "skip_frames": 4,
                    "l1_control": True,
                    "l1_debug": False,
                },
            )
            values.append(
                (
                    config["perception"],
                    config["planning"],
                    config["control"],
                    config["RSU"],
                    config["simulation"]["ego_num"],
                    config["simulation"]["skip_frames"],
                )
            )
        self.assertTrue(all(value == values[0] for value in values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
