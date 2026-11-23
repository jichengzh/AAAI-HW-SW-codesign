"""CPU-only tests for immutable RSU spatial offset injection."""

from collections import OrderedDict
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from opencood.utils.ditl_spatial_offset import (
    SPATIAL_AUDIT_SCHEMA_VERSION,
    SpatialAuditWriter,
    SpatialOffsetConfig,
    apply_rsu_spatial_offset,
    build_spatial_audit_record,
    resolve_spatial_offset,
)
from opencood.utils.transformation_utils import (
    get_pairwise_transformation_asymmetric,
    x1_to_x2,
)


def fake_base(ego_yaw=90.0):
    lidar_leaf = np.arange(8, dtype=np.float32).reshape(2, 4)
    measurements = {"lidar_pose_x": 12.0, "lidar_pose_y": 4.0}
    car = {
        "ego": True,
        "params": {
            "lidar_pose": [1.0, 2.0, 0.0, 0.0, ego_yaw, 0.0],
            "map_pose": [1.0, 2.0, 0.0, 0.0, ego_yaw, 0.0],
            "lidar_pose_clean": [1.0, 2.0, 0.0, 0.0, ego_yaw, 0.0],
        },
        "lidar_np": lidar_leaf,
    }
    rsu = {
        "ego": False,
        "params": {
            "lidar_pose": [10.0, 20.0, 3.0, 4.0, 30.0, 6.0],
            "map_pose": [11.0, 21.0, 3.0, 4.0, 30.0, 6.0],
            "lidar_pose_clean": [10.0, 20.0, 3.0, 4.0, 30.0, 6.0],
            "true_ego_pos": [10.0, 20.0, 3.0, 4.0, 30.0, 6.0],
            "predicted_ego_pos": [9.0, 19.0, 3.0, 4.0, 29.0, 6.0],
        },
        "measurements": measurements,
        "actors_data": {"17": {"loc": [12.0, 4.0, 0.0]}},
        "lidar_np": lidar_leaf,
        "unrelated": {"keep": "same"},
    }
    return OrderedDict((("car_0", car), ("rsu_0", rsu)))


class SpatialOffsetConfigTest(unittest.TestCase):
    def test_disabled_when_field_is_absent(self):
        self.assertIsNone(resolve_spatial_offset({"skip_frames": 4}))

    def test_exact_mapping_is_normalized_to_frozen_floats(self):
        config = resolve_spatial_offset({
            "ditl_spatial_offset": {"dx_m": 1, "dy_m": -0.5,
                                    "dyaw_deg": 2},
        })
        self.assertEqual(config, SpatialOffsetConfig(1.0, -0.5, 2.0))
        with self.assertRaises(FrozenInstanceError):
            config.dx_m = 3.0

    def test_rejects_invalid_simulation_and_schema(self):
        invalid = (
            None,
            {"ditl_spatial_offset": None},
            {"ditl_spatial_offset": []},
            {"ditl_spatial_offset": {"dx_m": 1, "dy_m": 0}},
            {"ditl_spatial_offset": {
                "dx_m": 1, "dy_m": 0, "dyaw_deg": 0, "extra": 1,
            }},
        )
        for simulation in invalid:
            with self.subTest(simulation=simulation):
                with self.assertRaises((TypeError, ValueError)):
                    resolve_spatial_offset(simulation)

    def test_rejects_boolean_non_number_and_non_finite_values(self):
        bad_values = (True, "1", None, float("nan"), float("inf"),
                      -float("inf"))
        for field in ("dx_m", "dy_m", "dyaw_deg"):
            for value in bad_values:
                values = {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 0.0}
                values[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        resolve_spatial_offset({"ditl_spatial_offset": values})


class SpatialOffsetTransformTest(unittest.TestCase):
    def test_rotates_ego_frame_translation_and_composes_yaw_immutably(self):
        original = fake_base(ego_yaw=90.0)
        transformed, audit = apply_rsu_spatial_offset(
            original, SpatialOffsetConfig(1.0, 2.0, 5.0)
        )

        self.assertIsInstance(transformed, OrderedDict)
        self.assertIsNot(transformed, original)
        self.assertIs(transformed["car_0"], original["car_0"])
        self.assertIsNot(transformed["rsu_0"], original["rsu_0"])
        self.assertIsNot(transformed["rsu_0"]["params"],
                         original["rsu_0"]["params"])
        self.assertIsNot(transformed["rsu_0"]["params"]["lidar_pose"],
                         original["rsu_0"]["params"]["lidar_pose"])
        self.assertIs(transformed["rsu_0"]["lidar_np"],
                      original["rsu_0"]["lidar_np"])
        self.assertIs(transformed["rsu_0"]["measurements"],
                      original["rsu_0"]["measurements"])
        self.assertIs(transformed["rsu_0"]["actors_data"],
                      original["rsu_0"]["actors_data"])

        # At ego yaw 90 degrees: world delta = (-dy, dx).
        self.assertEqual(original["rsu_0"]["params"]["lidar_pose"],
                         [10.0, 20.0, 3.0, 4.0, 30.0, 6.0])
        self.assertEqual(transformed["rsu_0"]["params"]["lidar_pose"],
                         [8.0, 21.0, 3.0, 4.0, 35.0, 6.0])
        self.assertEqual(transformed["rsu_0"]["params"]["map_pose"],
                         [9.0, 22.0, 3.0, 4.0, 35.0, 6.0])
        for key in ("lidar_pose_clean", "true_ego_pos",
                    "predicted_ego_pos"):
            self.assertIs(transformed["rsu_0"]["params"][key],
                          original["rsu_0"]["params"][key])
        self.assertIs(transformed["rsu_0"]["unrelated"],
                      original["rsu_0"]["unrelated"])

        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["cav_id"], "rsu_0")
        self.assertAlmostEqual(audit[0]["world_dx_m"], -2.0)
        self.assertAlmostEqual(audit[0]["world_dy_m"], 1.0)
        self.assertEqual(audit[0]["lidar_pose_before"],
                         (10.0, 20.0, 3.0, 4.0, 30.0, 6.0))
        self.assertEqual(audit[0]["lidar_pose_after"],
                         (8.0, 21.0, 3.0, 4.0, 35.0, 6.0))

    def test_longitudinal_levels_change_pairwise_but_not_clean_transform(self):
        original = fake_base(ego_yaw=0.0)
        baseline_pairwise = get_pairwise_transformation_asymmetric(
            original, max_cav=2, proj_first=False
        )
        clean_before = x1_to_x2(
            original["rsu_0"]["params"]["lidar_pose_clean"],
            original["car_0"]["params"]["lidar_pose_clean"],
        )
        previous_x = baseline_pairwise[1, 0, 0, 3]
        for dx_m in (0.0, 0.5, 1.0, 2.0):
            transformed, _ = apply_rsu_spatial_offset(
                original, SpatialOffsetConfig(dx_m, 0.0, 0.0)
            )
            pairwise = get_pairwise_transformation_asymmetric(
                transformed, max_cav=2, proj_first=False
            )
            self.assertAlmostEqual(
                pairwise[1, 0, 0, 3] - baseline_pairwise[1, 0, 0, 3],
                dx_m,
            )
            self.assertGreaterEqual(pairwise[1, 0, 0, 3], previous_x)
            previous_x = pairwise[1, 0, 0, 3]
            clean_after = x1_to_x2(
                transformed["rsu_0"]["params"]["lidar_pose_clean"],
                transformed["car_0"]["params"]["lidar_pose_clean"],
            )
            np.testing.assert_array_equal(clean_after, clean_before)

    def test_zero_offset_still_structurally_copies_rsu(self):
        original = fake_base()
        transformed, audit = apply_rsu_spatial_offset(
            original, SpatialOffsetConfig(0.0, 0.0, 0.0)
        )
        self.assertIsNot(transformed["rsu_0"], original["rsu_0"])
        self.assertEqual(transformed["rsu_0"]["params"]["map_pose"],
                         original["rsu_0"]["params"]["map_pose"])
        self.assertEqual(audit[0]["world_dx_m"], 0.0)
        self.assertEqual(audit[0]["world_dy_m"], 0.0)

    def test_rejects_missing_or_malformed_poses_and_config(self):
        cases = []
        no_ego = fake_base()
        no_ego["car_0"]["ego"] = False
        cases.append(no_ego)
        missing_ego_map = fake_base()
        del missing_ego_map["car_0"]["params"]["map_pose"]
        cases.append(missing_ego_map)
        malformed_ego_map = fake_base()
        malformed_ego_map["car_0"]["params"]["map_pose"] = [0.0] * 5
        cases.append(malformed_ego_map)
        malformed_rsu = fake_base()
        malformed_rsu["rsu_0"]["params"]["lidar_pose"] = [0.0] * 5
        cases.append(malformed_rsu)
        nonfinite_rsu = fake_base()
        nonfinite_rsu["rsu_0"]["params"]["map_pose"][0] = math.nan
        cases.append(nonfinite_rsu)
        for base_data in cases:
            with self.subTest(base_data=base_data):
                with self.assertRaises((TypeError, ValueError)):
                    apply_rsu_spatial_offset(
                        base_data, SpatialOffsetConfig(1.0, 0.0, 0.0)
                    )
        with self.assertRaises(TypeError):
            apply_rsu_spatial_offset(fake_base(), object())


class SpatialAuditWriterTest(unittest.TestCase):
    def test_writes_one_canonical_record_and_refuses_existing_path(self):
        _, values = apply_rsu_spatial_offset(
            fake_base(), SpatialOffsetConfig(1.0, 0.0, 0.0)
        )
        record = build_spatial_audit_record("route-146", 8,
                                            SpatialOffsetConfig(1, 0, 0),
                                            values)
        self.assertEqual(record["schema"], SPATIAL_AUDIT_SCHEMA_VERSION)
        self.assertEqual(record["applied_rsu_count"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ditl_spatial_audit.jsonl"
            writer = SpatialAuditWriter(path)
            writer.append(record)
            writer.close()
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), record)
            with self.assertRaises(IOError):
                SpatialAuditWriter(path)

    def test_zero_rsu_record_uses_null_world_delta(self):
        record = build_spatial_audit_record(
            "route-146", 0, SpatialOffsetConfig(0, 0, 0), ()
        )
        self.assertEqual(record["applied_rsu_count"], 0)
        self.assertIsNone(record["world_delta_m"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
