from __future__ import print_function

import copy
import hashlib
import json
import subprocess
import sys
import unittest

from scripts.action_fifo_full_manifest import SUBSETS
from scripts.ditl_packet_loss_manifest import (
    build_batches,
    build_partition,
    build_stage1_manifest,
    canonical_json_bytes,
    validate_stage1_manifest,
)
from simulation.leaderboard.team_code.closedloop.ditl_impairment_policy import (
    DeterministicImpairmentPolicy,
    MessageIdentity,
)
from simulation.leaderboard.team_code.closedloop.ditl_replay_config import (
    BurstConfig,
    ImpairmentConfig,
)


EXPECTED_STAGE1 = {
    "_1": (2, 5, 10, 18, 31, 147, 310, 326, 327),
    "_2": (2, 3, 7, 19, 136, 138, 300, 320, 322),
    "_3": (8, 301, 305, 311, 313, 314, 320, 324, 331),
    "_4": (1, 8, 9, 15, 16, 28, 311, 321, 323),
    "_5": (6, 7, 28, 31, 142, 146, 157, 310, 315),
}
EXPECTED_STAGE2 = {
    "_1": (3, 8, 13, 135, 146, 161, 300),
    "_2": (12, 22, 24, 25, 27, 161, 321, 328, 329),
    "_3": (7, 16, 31, 300, 310, 316, 329, 330),
    "_4": (12, 14, 140, 145, 160, 312, 317, 328, 331),
    "_5": (1, 8, 18, 20, 145, 306, 318),
}


class PacketLossManifestTests(unittest.TestCase):
    def test_partition_matches_frozen_stage_lists_and_digest(self):
        partition = build_partition()
        records = partition["records"]
        stage1 = {
            scenario: tuple(row["route"] for row in records
                            if row["scenario"] == scenario and row["stage"] == 1)
            for scenario in EXPECTED_STAGE1
        }
        stage2 = {
            scenario: tuple(row["route"] for row in records
                            if row["scenario"] == scenario and row["stage"] == 2)
            for scenario in EXPECTED_STAGE2
        }

        self.assertEqual(EXPECTED_STAGE1, stage1)
        self.assertEqual(EXPECTED_STAGE2, stage2)
        self.assertEqual({"stage1": 45, "stage2": 40, "total": 85},
                         partition["counts"])
        self.assertEqual(
            "49cfad0e0962bb62a4f195b7a4c26b00c3337a66aa363665707ffe58ce1d4105",
            partition["authority_sha256"],
        )
        self.assertEqual("sha256-per-scenario-top9-v1", partition["algorithm"])
        self.assertEqual(
            "92283c7b5e65143142da767bc620fa8041254f46e228be953ad56bc105cd8dfc",
            hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
        )
        self.assertEqual(SUBSETS, tuple(SUBSETS))

    def test_same_route_in_different_scenarios_remains_distinct(self):
        records = build_partition()["records"]
        route_eight = {
            (row["scenario"], row["route"])
            for row in records
            if row["scenario"] in ("_1", "_4", "_5") and row["route"] == 8
        }
        self.assertEqual({("_1", 8), ("_4", 8), ("_5", 8)}, route_eight)

    def test_manifest_has_exact_arms_tasks_batches_and_task_keys(self):
        manifest = build_stage1_manifest(build_partition())
        self.assertEqual("ditl.packet-loss-stage1-manifest.v1", manifest["schema"])
        self.assertEqual("hmac-sha256-53bit-v1", manifest["rng_version"])
        self.assertEqual(
            (("clean", 0.0), ("drop_010", 0.1),
             ("drop_020", 0.2), ("drop_030", 0.3)),
            tuple((arm["id"], arm["drop_rate"]) for arm in manifest["arms"]),
        )
        tasks = manifest["tasks"]
        self.assertEqual(180, len(tasks))
        self.assertEqual(180, len({task["key"] for task in tasks}))
        for arm in ("clean", "drop_010", "drop_020", "drop_030"):
            arm_tasks = [task for task in tasks if task["arm"] == arm]
            self.assertEqual(45, len(arm_tasks))
            for scenario in EXPECTED_STAGE1:
                self.assertEqual(
                    9,
                    len([task for task in arm_tasks
                         if task["scenario"] == scenario]),
                )
        first = tasks[0]
        self.assertEqual(
            "stage1:clean:_1:r2:cseed2000:tseed2000:replay2000:rep0",
            first["key"],
        )
        self.assertEqual((2000, 2000, 2000, 0, 2),
                         (first["carla_seed"], first["traffic_seed"],
                          first["replay_seed"], first["repeat"],
                          first["skip_frames"]))
        batches = manifest["batches"]
        self.assertEqual(20, len(batches))
        self.assertTrue(all(len(batch["task_keys"]) == 9 for batch in batches))
        self.assertEqual(tuple(batches), build_batches(manifest))

    def test_only_arm_and_drop_rate_change_within_each_cell(self):
        tasks = build_stage1_manifest(build_partition())["tasks"]
        cells = {}
        for task in tasks:
            cells.setdefault((task["scenario"], task["route"]), []).append(task)
        self.assertEqual(45, len(cells))
        for variants in cells.values():
            self.assertEqual(
                ["clean", "drop_010", "drop_020", "drop_030"],
                [task["arm"] for task in variants],
            )
            normalized = []
            for task in variants:
                row = dict(task)
                del row["key"]
                del row["arm"]
                del row["drop_rate"]
                normalized.append(row)
            self.assertEqual([normalized[0]] * 4, normalized)

    def test_loss_thresholds_are_nested_and_cross_process_stable(self):
        def dropped(rate):
            config = ImpairmentConfig(
                extra_delay_ns=0,
                jitter_ns=0,
                drop_rate=rate,
                burst=BurstConfig(False, 0.0, 1.0),
                reorder_rate=0.0,
                reorder_extra_ns=0,
                duplicate_rate=0.0,
            )
            policy = DeterministicImpairmentPolicy(2000, config)
            return {
                index
                for index in range(2000)
                if policy.decide(MessageIdentity(
                    "_1:r2", "rsu-0", "message-%d" % index, index
                ), 0).dropped
            }

        drop_010 = dropped(0.1)
        drop_020 = dropped(0.2)
        drop_030 = dropped(0.3)
        self.assertLess(drop_010, drop_020)
        self.assertLess(drop_020, drop_030)

        script = """
import json
from simulation.leaderboard.team_code.closedloop.ditl_impairment_policy import DeterministicImpairmentPolicy, MessageIdentity
from simulation.leaderboard.team_code.closedloop.ditl_replay_config import BurstConfig, ImpairmentConfig
c = ImpairmentConfig(0, 0, 0.3, BurstConfig(False, 0.0, 1.0), 0.0, 0, 0.0)
p = DeterministicImpairmentPolicy(2000, c)
print(json.dumps([i for i in range(2000) if p.decide(MessageIdentity('_1:r2', 'rsu-0', 'message-%d' % i, i), 0).dropped]))
"""
        output = subprocess.check_output([sys.executable, "-c", script])
        self.assertEqual(drop_030, set(json.loads(output.decode("ascii"))))

    def test_build_partition_rejects_invalid_authority_and_boolean_seed(self):
        with self.assertRaises(TypeError):
            build_partition(seed=True)
        with self.assertRaises(ValueError):
            build_partition(())
        duplicated_route = list(SUBSETS)
        first = list(duplicated_route[0])
        first[3] = tuple(first[3]) + (first[3][0],)
        duplicated_route[0] = tuple(first)
        with self.assertRaises(ValueError):
            build_partition(tuple(duplicated_route))
        duplicated_scenario = tuple(SUBSETS) + (SUBSETS[0],)
        with self.assertRaises(ValueError):
            build_partition(duplicated_scenario)
        changed_authority = list(SUBSETS)
        changed = list(changed_authority[0])
        changed[3] = tuple(changed[3][:-1]) + (999,)
        changed_authority[0] = tuple(changed)
        with self.assertRaises(ValueError):
            build_partition(tuple(changed_authority))

    def test_build_manifest_rejects_noncanonical_partition_and_boolean_seeds(self):
        partition = build_partition()
        injected = copy.deepcopy(partition)
        stage2 = next(row for row in injected["records"] if row["stage"] == 2)
        stage2["stage"] = 1
        with self.assertRaises(ValueError):
            build_stage1_manifest(injected)
        for field in ("carla_seed", "traffic_seed", "replay_seed", "repeat"):
            kwargs = {field: True}
            with self.subTest(field=field), self.assertRaises(TypeError):
                build_stage1_manifest(partition, **kwargs)

    def test_validation_rejects_extra_fields_duplicates_stage2_and_bad_rates(self):
        manifest = build_stage1_manifest(build_partition())
        mutations = []
        extra = copy.deepcopy(manifest)
        extra["unexpected"] = 1
        mutations.append(extra)
        duplicate = copy.deepcopy(manifest)
        duplicate["tasks"][1]["key"] = duplicate["tasks"][0]["key"]
        mutations.append(duplicate)
        stage2 = copy.deepcopy(manifest)
        stage2["tasks"][0]["route"] = 3
        mutations.append(stage2)
        infinite = copy.deepcopy(manifest)
        infinite["arms"][1]["drop_rate"] = float("inf")
        mutations.append(infinite)
        task_extra = copy.deepcopy(manifest)
        task_extra["tasks"][0]["unexpected"] = 1
        mutations.append(task_extra)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_stage1_manifest(mutation)

    def test_validation_and_batches_return_new_objects_in_canonical_order(self):
        manifest = build_stage1_manifest(build_partition())
        validated = validate_stage1_manifest(manifest)
        self.assertEqual(manifest, validated)
        self.assertIsNot(manifest, validated)
        validated["tasks"][0]["route"] = 999
        self.assertEqual(2, manifest["tasks"][0]["route"])

        requested = (
            manifest["tasks"][46]["key"],
            manifest["tasks"][45]["key"],
            manifest["tasks"][90]["key"],
        )
        batches = build_batches(manifest, requested)
        self.assertEqual(2, len(batches))
        self.assertEqual(
            [manifest["tasks"][45]["key"], manifest["tasks"][46]["key"]],
            batches[0]["task_keys"],
        )
        self.assertEqual([manifest["tasks"][90]["key"]],
                         batches[1]["task_keys"])
        with self.assertRaises(ValueError):
            build_batches(manifest, (requested[0], requested[0]))
        with self.assertRaises(ValueError):
            build_batches(manifest, ("unknown",))


if __name__ == "__main__":
    unittest.main()
