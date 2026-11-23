from __future__ import print_function

from dataclasses import replace
import importlib.util
import os
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
MODULE_PATH = os.path.join(REPO_ROOT, "scripts", "ditl_task6_manifest.py")
SPEC = importlib.util.spec_from_file_location("ditl_task6_manifest", MODULE_PATH)
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


EXPECTED_IDS = (
    "clean",
    "comm_drop_005", "comm_drop_010", "comm_drop_020",
    "spatial_dx_05", "spatial_dx_10", "spatial_dx_20",
    "compute_gpu_low", "compute_gpu_medium", "compute_gpu_high",
)


class Task6ManifestTest(unittest.TestCase):
    def test_build_has_exact_order_and_common_route_contract(self):
        arms = MANIFEST.build_task6_arms()

        self.assertIsInstance(arms, tuple)
        self.assertEqual(EXPECTED_IDS, tuple(arm.arm_id for arm in arms))
        for arm in arms:
            self.assertEqual(146, arm.route_id)
            self.assertEqual("_1", arm.scenario_suffix)
            self.assertEqual(0, arm.repeat)
            self.assertEqual(2000, arm.carla_seed)
            self.assertEqual(2000, arm.traffic_seed)

    def test_exact_single_factor_values(self):
        arms = MANIFEST.build_task6_arms()
        by_id = {arm.arm_id: arm for arm in arms}

        self.assertEqual(
            (0.05, 0.10, 0.20),
            tuple(by_id[name].drop_rate for name in EXPECTED_IDS[1:4]),
        )
        self.assertEqual(
            (0.5, 1.0, 2.0),
            tuple(by_id[name].spatial_dx_m for name in EXPECTED_IDS[4:7]),
        )
        self.assertEqual(
            ("raw_samehost_gpu_low", "raw_samehost_gpu_medium", "raw_samehost_gpu_high"),
            tuple(by_id[name].trace_profile_id for name in EXPECTED_IDS[7:]),
        )
        for arm in arms:
            if arm.family == "communication":
                self.assertEqual((0.0, 0.0), (arm.spatial_dy_m, arm.spatial_dyaw_deg))
                self.assertEqual("raw_samehost_idle", arm.trace_profile_id)
            elif arm.family == "spatial":
                self.assertEqual(0.0, arm.drop_rate)
                self.assertEqual((0.0, 0.0), (arm.spatial_dy_m, arm.spatial_dyaw_deg))
                self.assertEqual("raw_samehost_idle", arm.trace_profile_id)
            elif arm.family == "compute":
                self.assertEqual(0.0, arm.drop_rate)
                self.assertEqual((0.0, 0.0, 0.0), (arm.spatial_dx_m, arm.spatial_dy_m, arm.spatial_dyaw_deg))

    def test_validation_returns_tuple_copy_and_rejects_shape_errors(self):
        arms = MANIFEST.build_task6_arms()
        validated = MANIFEST.validate_task6_arms(arms)
        self.assertIsInstance(validated, tuple)
        self.assertEqual(arms, validated)
        self.assertIsNot(arms, validated)

        def replace_arm(index, **changes):
            candidate = list(arms)
            candidate[index] = replace(candidate[index], **changes)
            return tuple(candidate)

        cases = (
            ("duplicate", replace_arm(1, arm_id=arms[0].arm_id), "duplicate arm ID"),
            ("missing", replace_arm(9, arm_id="missing_arm"), "canonical"),
            ("extra", replace_arm(9, arm_id="extra_arm"), "canonical"),
            ("wrong route", replace_arm(1, route_id=145), "route_id"),
            ("non-finite", replace_arm(1, drop_rate=float("nan")), "finite"),
            ("unsupported rate", replace_arm(1, drop_rate=0.15), "drop_rate"),
            ("unsupported offset", replace_arm(4, spatial_dx_m=3.0), "spatial_dx_m"),
            ("unsupported profile", replace_arm(7, trace_profile_id="raw_samehost_idle"), "trace_profile_id"),
            ("two axes", replace_arm(1, spatial_dx_m=0.5), "spatial_dx_m"),
        )
        for label, candidate, expected_message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, expected_message):
                    MANIFEST.validate_task6_arms(candidate)

    def test_manifest_is_deeply_fresh_and_does_not_mutate_arms(self):
        arms = MANIFEST.build_task6_arms()
        manifest = MANIFEST.manifest_to_dict(arms)
        self.assertEqual("ditl.task6-manifest.v1", manifest["schema"])
        self.assertEqual(
            {"route_id": 146, "scenario_suffix": "_1", "repeat": 0,
             "carla_seed": 2000, "traffic_seed": 2000},
            manifest["route"],
        )
        self.assertEqual(EXPECTED_IDS, tuple(item["arm_id"] for item in manifest["arms"]))
        self.assertEqual(arms[1].drop_rate, manifest["arms"][1]["drop_rate"])

        manifest["route"]["route_id"] = 999
        manifest["arms"][1]["drop_rate"] = 0.99
        second = MANIFEST.manifest_to_dict(arms)
        self.assertEqual(146, second["route"]["route_id"])
        self.assertEqual(0.05, second["arms"][1]["drop_rate"])
        self.assertEqual(146, arms[0].route_id)
        self.assertEqual(0.05, arms[1].drop_rate)


if __name__ == "__main__":
    unittest.main()
