# -*- coding: utf-8 -*-
"""Contracts for the official Town05 action-FIFO full sweep."""

from __future__ import print_function

import copy
import glob
import json
import os
import subprocess
import tempfile
import unittest

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(REPO, "scripts", "action_fifo_full_manifest.py")
SWEEP = os.path.join(REPO, "scripts", "action_fifo_full_sweep.sh")
EVAL_ONE = os.path.join(REPO, "scripts", "action_fifo_eval_one.sh")
CONFIG_DIR = os.path.join(
    REPO, "simulation", "leaderboard", "team_code", "agent_config"
)


def run_manifest(*args):
    command = ["python", MANIFEST]
    command.extend(args or ["--format", "json"])
    return subprocess.run(
        command,
        cwd=REPO,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def result_fixture(global_status, route_status=None, exception_status=None):
    """Return the one-route checkpoint shape emitted by leaderboard."""
    record_status = route_status or global_status
    global_meta = {}
    if exception_status is not None:
        global_meta = {
            "exceptions": [["RouteScenario_8", 0, exception_status]]
        }
    return {
        "_checkpoint": {
            "global_record": {
                "index": -1,
                "meta": global_meta,
                "route_id": -1,
                "scores": {
                    "score_composed": 50.0,
                    "score_penalty": 0.5,
                    "score_route": 100.0,
                },
                "status": global_status,
            },
            "progress": [1, 1],
            "records": [
                {
                    "index": 0,
                    "meta": {"duration_game": 72.5, "duration_system": 825.9},
                    "route_id": "RouteScenario_8",
                    "scores": {
                        "score_composed": 50.0,
                        "score_penalty": 0.5,
                        "score_route": 100.0,
                    },
                    "status": record_status,
                }
            ],
        },
        "eligible": True,
        "entry_status": (
            "Finished" if global_status == "Completed"
            else "Finished with agent errors"
        ),
    }


def check_result_fixture(payload):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
        json.dump(payload, stream)
        stream.flush()
        return run_manifest("--check-result", stream.name)


class ActionFifoFullManifestTest(unittest.TestCase):
    def test_smoke_manifest_selects_three_profiles_at_zero_and_800ms(self):
        result = run_manifest("--format", "json", "--smoke")
        self.assertEqual(result.returncode, 0, result.stdout)
        tasks = json.loads(result.stdout)

        self.assertEqual(len(tasks), 6)
        self.assertEqual(len(set(task["key"] for task in tasks)), 6)
        self.assertEqual(
            {
                (task["subset"], task["scenario"], task["route"], task["profile"])
                for task in tasks
            },
            {
                ("1s", "_1", 2, "8_10"),
                ("2s", "_2", 2, "5_10"),
                ("5s", "_5", 1, "5_50"),
            },
        )
        self.assertEqual(
            {delay: sum(task["delay_ms"] == delay for task in tasks)
             for delay in (0, 800)},
            {0: 3, 800: 3},
        )
        self.assertTrue(all(task["experiment"] == "smoke" for task in tasks))

    def test_smoke_namespace_cannot_collide_with_full_sweep(self):
        full = run_manifest("--format", "json", "--seed", "2000", "--repeat", "0")
        smoke = run_manifest(
            "--format", "json", "--seed", "2000", "--repeat", "0", "--smoke"
        )
        for result in (full, smoke):
            self.assertEqual(result.returncode, 0, result.stdout)
        full_tasks = json.loads(full.stdout)
        smoke_tasks = json.loads(smoke.stdout)

        full_paths = {task["result_json"] for task in full_tasks}
        smoke_paths = {task["result_json"] for task in smoke_tasks}
        self.assertFalse(full_paths.intersection(smoke_paths))
        for task in smoke_tasks:
            for field in ("key", "tag", "result_json", "eval_log", "carla_log"):
                self.assertIn("smoke", task[field])

    def test_manifest_has_85_authority_cells_and_340_unique_tasks(self):
        result = run_manifest("--format", "json")
        self.assertEqual(result.returncode, 0, result.stdout)
        tasks = json.loads(result.stdout)

        self.assertEqual(len(tasks), 340)
        self.assertEqual(len(set(task["key"] for task in tasks)), 340)
        self.assertEqual(
            {delay: sum(task["delay_ms"] == delay for task in tasks)
             for delay in (0, 200, 500, 800)},
            {0: 85, 200: 85, 500: 85, 800: 85},
        )
        self.assertEqual(
            {subset: sum(task["subset"] == subset for task in tasks) // 4
             for subset in ("1s", "2s", "3s", "4s", "5s")},
            {"1s": 16, "2s": 18, "3s": 17, "4s": 18, "5s": 16},
        )

    def test_manifest_preserves_batch_authority_profile_mapping(self):
        result = run_manifest("--format", "json")
        self.assertEqual(result.returncode, 0, result.stdout)
        tasks = json.loads(result.stdout)

        by_subset = {}
        for task in tasks:
            by_subset.setdefault(task["subset"], set()).add(
                (task["scenario"], task["profile"])
            )
        self.assertEqual(
            by_subset,
            {
                "1s": {("_1", "8_10")},
                "2s": {("_2", "5_10")},
                "3s": {("_3", "5_10")},
                "4s": {("_4", "8_10")},
                "5s": {("_5", "5_50")},
            },
        )

    def test_manifest_route_lists_match_batch_authority_exactly(self):
        result = run_manifest("--format", "json")
        self.assertEqual(result.returncode, 0, result.stdout)
        tasks = json.loads(result.stdout)
        routes = {}
        for task in tasks:
            if task["delay_ms"] == 0:
                routes.setdefault(task["subset"], []).append(task["route"])

        self.assertEqual(
            routes,
            {
                "1s": [2, 3, 5, 8, 10, 13, 18, 31, 135, 146, 147, 161, 300, 310, 326, 327],
                "2s": [2, 3, 7, 12, 19, 22, 24, 25, 27, 136, 138, 161, 300, 320, 321, 322, 328, 329],
                "3s": [7, 8, 16, 31, 300, 301, 305, 310, 311, 313, 314, 316, 320, 324, 329, 330, 331],
                "4s": [1, 8, 9, 12, 14, 15, 16, 28, 140, 145, 160, 311, 312, 317, 321, 323, 328, 331],
                "5s": [1, 6, 7, 8, 18, 20, 28, 31, 142, 145, 146, 157, 306, 310, 315, 318],
            },
        )

    def test_duplicate_route_across_subsets_has_distinct_complete_paths(self):
        result = run_manifest(
            "--format", "json", "--seed", "2301", "--repeat", "7"
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        tasks = json.loads(result.stdout)
        matches = [
            task for task in tasks
            if task["route"] == 8 and task["delay_ms"] == 500
        ]

        self.assertGreater(len(matches), 1)
        for field in ("key", "tag", "result_json", "eval_log", "carla_log"):
            values = [task[field] for task in matches]
            self.assertEqual(len(values), len(set(values)), field)
            for task, value in zip(matches, values):
                self.assertIn("s%s" % task["subset"], value)
                self.assertIn("sc%s" % task["scenario"].lstrip("_"), value)
                self.assertIn("r8", value)
                self.assertIn("d500", value)
                self.assertIn("seed2301", value)
                self.assertIn("rep7", value)
                self.assertNotIn("_n2", value)

    def test_two_worker_shards_are_deterministic_disjoint_and_complete(self):
        complete = run_manifest("--format", "json")
        shard0 = run_manifest(
            "--format", "json", "--shard-index", "0", "--shard-count", "2"
        )
        shard1 = run_manifest(
            "--format", "json", "--shard-index", "1", "--shard-count", "2"
        )
        for result in (complete, shard0, shard1):
            self.assertEqual(result.returncode, 0, result.stdout)

        all_keys = [task["key"] for task in json.loads(complete.stdout)]
        keys0 = [task["key"] for task in json.loads(shard0.stdout)]
        keys1 = [task["key"] for task in json.loads(shard1.stdout)]
        self.assertEqual(keys0, all_keys[0::2])
        self.assertEqual(keys1, all_keys[1::2])
        self.assertEqual(len(keys0), 170)
        self.assertEqual(len(keys1), 170)
        self.assertFalse(set(keys0).intersection(keys1))
        self.assertEqual(set(keys0).union(keys1), set(all_keys))

    def test_distinct_carla_and_traffic_seeds_are_preserved_in_task_identity(self):
        result = run_manifest(
            "--format", "json", "--carla-seed", "2301",
            "--traffic-seed", "2302",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        task = json.loads(result.stdout)[0]

        self.assertEqual(task["carla_seed"], 2301)
        self.assertEqual(task["traffic_seed"], 2302)
        for field in ("key", "tag", "result_json", "eval_log", "carla_log"):
            self.assertIn("cseed2301_tseed2302", task[field])

    def test_completed_result_is_valid(self):
        completed = check_result_fixture(result_fixture("Completed"))

        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("result_valid status=Completed", completed.stdout)

    def test_blocked_driving_failure_is_valid_but_missing_result_is_infra(self):
        blocked_status = "Failed - Agent got blocked"
        failed = check_result_fixture(
            result_fixture("Failed", blocked_status, blocked_status)
        )
        missing = run_manifest(
            "--check-result", "/tmp/missing-action-fifo-full-result.json"
        )

        self.assertEqual(failed.returncode, 0, failed.stdout)
        self.assertIn("result_valid status=Failed", failed.stdout)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("result_infrastructure_missing", missing.stdout)

    def test_timeout_and_route_deviation_driving_failures_are_valid(self):
        for driving_status in (
            "Failed - Agent timed out",
            "Failed - Agent deviated from the route",
        ):
            with self.subTest(status=driving_status):
                result = check_result_fixture(
                    result_fixture("Failed", driving_status, driving_status)
                )

                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("result_valid status=Failed", result.stdout)

    def test_cuda_agent_crash_is_infrastructure_invalid(self):
        crash_status = "Failed - Agent crashed"
        payload = result_fixture("Failed", crash_status, crash_status)
        payload["_checkpoint"]["records"][0]["meta"]["error"] = (
            "RuntimeError: CUDA unknown error"
        )

        result = check_result_fixture(payload)

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("result_infrastructure_missing", result.stdout)

    def test_simulation_crash_is_infrastructure_invalid(self):
        crash_status = "Failed - Simulation crashed"
        result = check_result_fixture(
            result_fixture("Failed", crash_status, crash_status)
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("result_infrastructure_missing", result.stdout)

    def test_failed_checkpoint_without_single_route_evidence_is_invalid(self):
        result = check_result_fixture(
            {"_checkpoint": {"global_record": {"status": "Failed"}}}
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("result_infrastructure_missing", result.stdout)

    def test_malformed_failure_metadata_is_infrastructure_invalid(self):
        payload = result_fixture("Failed", "Failed - Agent got blocked")
        payload["_checkpoint"]["global_record"]["meta"] = None

        result = check_result_fixture(payload)

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("result_infrastructure_missing", result.stdout)
        self.assertNotIn("Traceback", result.stdout)

    def test_started_checkpoint_is_incomplete_infrastructure_result(self):
        result = check_result_fixture(result_fixture("Started"))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("result_infrastructure_missing", result.stdout)


class ActionFifoFullSweepShellTest(unittest.TestCase):
    def test_smoke_dry_run_assigns_three_tasks_each_to_gpus_4_and_7(self):
        result = subprocess.run(
            [
                "bash", SWEEP, "--dry-run", "--smoke", "--gpus", "4,7",
                "--seed", "2000", "--repeat", "0",
            ],
            cwd=REPO,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "PROFILE_SMOKE_DRY_RUN base_cells=3 delays=2 tasks=6",
            result.stdout,
        )
        self.assertIn("SHARD worker=0 gpu=4 port=44400 tasks=3", result.stdout)
        self.assertIn("SHARD worker=1 gpu=7 port=44700 tasks=3", result.stdout)
        self.assertIn("DRY_RUN_ONLY carla_started=0 evaluator_started=0", result.stdout)
        self.assertNotIn("Traceback", result.stdout)

    def test_eval_helper_forwards_distinct_seeds_and_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = os.path.join(tmpdir, "bin")
            os.mkdir(fake_bin)
            fake_python = os.path.join(fake_bin, "python")
            capture = os.path.join(tmpdir, "args.txt")
            with open(fake_python, "w") as stream:
                stream.write(
                    "#!/bin/sh\n"
                    "printf '%s\\n' \"$@\" > \"$ACTION_FIFO_CAPTURE\"\n"
                )
            os.chmod(fake_python, 0o755)
            environment = dict(os.environ)
            environment["PATH"] = fake_bin + os.pathsep + environment["PATH"]
            environment["ACTION_FIFO_CAPTURE"] = capture
            result = subprocess.run(
                [
                    "bash", EVAL_ONE, "317", "44400", "full-tag", "9",
                    "codriving_8_10_fifo500_l1", "_4",
                    "v2x_final/town05_short_collab/unique-key", "2301", "2302",
                ],
                cwd=tmpdir,
                env=environment,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            with open(capture) as stream:
                arguments = stream.read().splitlines()

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--carlaProviderSeed=2301", arguments)
        self.assertIn("--trafficManagerSeed=2302", arguments)
        self.assertIn(
            "--scenario_parameter=simulation/leaderboard/leaderboard/scenarios/"
            "scenario_parameter_4.yaml",
            arguments,
        )
        self.assertIn(
            "--agent-config=simulation/leaderboard/team_code/agent_config/"
            "pnp_config_codriving_8_10_fifo500_l1.yaml",
            arguments,
        )
        self.assertIn("--skip_existed=0", arguments)

    def test_dry_run_assigns_170_tasks_each_to_gpus_4_and_7(self):
        result = subprocess.run(
            [
                "bash", SWEEP, "--dry-run", "--gpus", "4,7",
                "--seed", "2000", "--repeat", "0",
            ],
            cwd=REPO,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "FULL_SWEEP_DRY_RUN base_cells=85 delays=4 tasks=340 "
            "seed=2000 repeat=0",
            result.stdout,
        )
        self.assertIn("SHARD worker=0 gpu=4 port=44400 tasks=170", result.stdout)
        self.assertIn("SHARD worker=1 gpu=7 port=44700 tasks=170", result.stdout)
        self.assertIn("DRY_RUN_ONLY carla_started=0 evaluator_started=0", result.stdout)
        self.assertNotIn("Traceback", result.stdout)

    def test_invalid_gpu_list_fails_before_execution(self):
        result = subprocess.run(
            ["bash", SWEEP, "--dry-run", "--gpus", "4,4"],
            cwd=REPO,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU list must contain unique numeric IDs", result.stdout)

    def test_dry_run_accepts_distinct_carla_and_traffic_seeds(self):
        result = subprocess.run(
            [
                "bash", SWEEP, "--dry-run", "--gpus", "4,7",
                "--carla-seed", "2301", "--traffic-seed", "2302",
            ],
            cwd=REPO,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("carla_seed=2301 traffic_seed=2302", result.stdout)
        self.assertIn("cseed2301_tseed2302", result.stdout)

    def test_result_gate_is_nonzero_when_infrastructure_results_are_missing(self):
        result = subprocess.run(
            [
                "bash", SWEEP, "--check-results", "--gpus", "4,7",
                "--seed", "987654321", "--repeat", "99",
            ],
            cwd=REPO,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "FULL_SWEEP_RESULT_GATE total=340 valid=0 missing=340",
            result.stdout,
        )


class ActionFifoFullProfileConfigTest(unittest.TestCase):
    def test_profile_configs_preserve_authority_and_only_add_fifo_contract(self):
        for profile in ("8_10", "5_10", "5_50"):
            base_path = os.path.join(
                CONFIG_DIR, "pnp_config_codriving_%s.yaml" % profile
            )
            with open(base_path) as stream:
                base = yaml.safe_load(stream)
            for delay_ms in (0, 200, 500, 800):
                with self.subTest(profile=profile, delay_ms=delay_ms):
                    fifo_path = os.path.join(
                        CONFIG_DIR,
                        "pnp_config_codriving_%s_fifo%d_l1.yaml"
                        % (profile, delay_ms),
                    )
                    self.assertTrue(os.path.isfile(fifo_path), fifo_path)
                    with open(fifo_path) as stream:
                        actual = yaml.safe_load(stream)
                    expected = copy.deepcopy(base)
                    expected["simulation"].update(
                        {
                            "latency_inject_ms": 0,
                            "l1_control": True,
                            "tau_ego_ms": 0,
                            "l1_debug": False,
                            "action_fifo_delay_ms": delay_ms,
                            "action_fifo_audit": True,
                        }
                    )
                    self.assertEqual(actual, expected)

    def test_fifo_profile_configs_point_to_materialized_perception_artifacts(self):
        for profile in ("8_10", "5_10", "5_50"):
            for delay_ms in (0, 200, 500, 800):
                with self.subTest(profile=profile, delay_ms=delay_ms):
                    fifo_path = os.path.join(
                        CONFIG_DIR,
                        "pnp_config_codriving_%s_fifo%d_l1.yaml"
                        % (profile, delay_ms),
                    )
                    with open(fifo_path) as stream:
                        actual = yaml.safe_load(stream)
                    model_dir = os.path.join(
                        REPO, actual["perception"]["perception_model_dir"]
                    )

                    self.assertTrue(
                        os.path.isfile(os.path.join(model_dir, "config.yaml")),
                        model_dir,
                    )
                    self.assertTrue(
                        glob.glob(os.path.join(model_dir, "net_epoch*.pth")),
                        model_dir,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
