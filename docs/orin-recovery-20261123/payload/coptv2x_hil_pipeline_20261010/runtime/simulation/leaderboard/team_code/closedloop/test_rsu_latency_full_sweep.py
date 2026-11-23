# -*- coding: utf-8 -*-
"""Contracts for the full-scene RSU-latency validation sweep."""

from __future__ import print_function

import csv
import contextlib
import io
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(REPO, "scripts", "rsu_latency_full_manifest.py")
COORDINATOR = os.path.join(REPO, "scripts", "rsu_latency_full_sweep.py")
CONFIG_DIR = os.path.join(
    REPO, "simulation", "leaderboard", "team_code", "agent_config"
)
ROUTE_DIR = os.path.join(
    REPO, "simulation", "leaderboard", "data", "evaluation_routes"
)
SCENARIO_DIR = os.path.join(
    REPO, "simulation", "leaderboard", "leaderboard", "scenarios"
)


def run_manifest(*args):
    command = [sys.executable, MANIFEST]
    command.extend(args or ["--format", "json"])
    return subprocess.run(
        command,
        cwd=REPO,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


class RsuLatencyManifestTest(unittest.TestCase):
    def manifest(self, *args):
        result = run_manifest("--format", "json", *args)
        self.assertEqual(result.returncode, 0, result.stdout)
        return json.loads(result.stdout)

    def test_manifest_has_85_cells_and_340_rsu_only_tasks(self):
        tasks = self.manifest("--seed", "2000", "--repeat", "0")
        cells = {(task["scenario"], task["route"]) for task in tasks}
        self.assertEqual(len(cells), 85)
        self.assertEqual(len(tasks), 340)
        self.assertEqual(len({task["key"] for task in tasks}), 340)
        self.assertEqual(
            {delay: sum(task["rsu_delay_ms"] == delay for task in tasks)
             for delay in (0, 200, 500, 800)},
            {0: 85, 200: 85, 500: 85, 800: 85},
        )
        self.assertTrue(all(task["ego_delay_ms"] == 0 for task in tasks))

    def test_manifest_preserves_seed_repeat_and_dual_fifo_agents(self):
        tasks = self.manifest(
            "--carla-seed", "2001", "--traffic-seed", "2002", "--repeat", "0"
        )
        agents = {
            0: "codriving_dualfifo_baseline_l1",
            200: "codriving_dualfifo_rsu200_l1",
            500: "codriving_dualfifo_rsu500_l1",
            800: "codriving_dualfifo_rsu800_l1",
        }
        for task in tasks:
            self.assertEqual(task["carla_seed"], 2001)
            self.assertEqual(task["traffic_seed"], 2002)
            self.assertEqual(task["repeat"], 0)
            self.assertEqual(task["agent"], agents[task["rsu_delay_ms"]])
            self.assertIn("cseed2001_tseed2002", task["key"])
            self.assertIn("_rep0", task["key"])

    def test_all_route_scenario_and_config_inputs_exist(self):
        for task in self.manifest():
            route = os.path.join(
                ROUTE_DIR, "town05_short_r%d.xml" % task["route"]
            )
            scenario = os.path.join(
                SCENARIO_DIR,
                "scenario_parameter%s.yaml" % task["scenario"],
            )
            config = os.path.join(
                CONFIG_DIR, "pnp_config_%s.yaml" % task["agent"]
            )
            self.assertTrue(os.path.isfile(route), route)
            self.assertTrue(os.path.isfile(scenario), scenario)
            self.assertTrue(os.path.isfile(config), config)

    def test_one_two_four_and_six_shards_cover_the_same_tasks(self):
        expected = {task["key"] for task in self.manifest()}
        for count in (1, 2, 4, 6):
            shards = [
                self.manifest(
                    "--shard-index", str(index), "--shard-count", str(count)
                )
                for index in range(count)
            ]
            keys = [task["key"] for shard in shards for task in shard]
            self.assertEqual(len(keys), 340)
            self.assertEqual(len(set(keys)), 340)
            self.assertEqual(set(keys), expected)

    def test_invalid_seed_repeat_and_partial_shard_are_rejected(self):
        for args in (
            ("--seed", "-1"),
            ("--repeat", "-1"),
            ("--shard-count", "6"),
            ("--shard-index", "6", "--shard-count", "6"),
        ):
            with self.subTest(args=args):
                result = run_manifest(*args)
                self.assertNotEqual(result.returncode, 0, result.stdout)


class RsuLatencyCoordinatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("rsu_latency_full_sweep", COORDINATOR)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def tasks(self):
        spec = importlib.util.spec_from_file_location("rsu_latency_full_manifest", MANIFEST)
        module = importlib.util.module_from_spec(spec)
        scripts = os.path.join(REPO, "scripts")
        sys.path.insert(0, scripts)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(scripts)
        return module.build_manifest()

    def manifest_module(self):
        spec = importlib.util.spec_from_file_location(
            "rsu_latency_full_manifest_direct", MANIFEST
        )
        module = importlib.util.module_from_spec(spec)
        scripts = os.path.join(REPO, "scripts")
        sys.path.insert(0, scripts)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(scripts)
        return module

    def test_manifest_direct_cli_covers_tsv_sharding_and_validation(self):
        module = self.manifest_module()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(module.main([
                "--format", "tsv", "--seed", "2000",
                "--shard-index", "1", "--shard-count", "6",
            ]), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 57)
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(module.main(["--seed", "-1"]), 2)
        self.assertIn("non-negative", error.getvalue())
        with self.assertRaises(ValueError):
            module.select_shard(module.build_manifest(), shard_index=0)

    def test_gpu_pool_is_a_count_and_rejects_invalid_values(self):
        for valid in ("1", "2", "4", "6", "8"):
            self.assertEqual(self.module.parse_gpu_pool(valid), int(valid))
        for invalid in (None, "", "0", "-1", "1,2", "gpu", "9"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.module.parse_gpu_pool(invalid)

    def test_idle_gpu_selection_excludes_busy_cards_and_fails_closed(self):
        rows = "0, 0, 2\n1, 7, 2\n2, 0, 51\n3, 5, 50\n7, 0, 2\n"
        self.assertEqual(self.module.parse_idle_gpus(rows), [0, 3, 7])
        self.assertEqual(self.module.select_idle_gpus(rows, 2), [0, 3])
        with self.assertRaisesRegex(RuntimeError, "need 4 idle GPUs"):
            self.module.select_idle_gpus(rows, 4)

    def test_twenty_batches_cover_all_340_tasks_once(self):
        batches = self.module.build_batches(self.tasks())
        self.assertEqual(len(batches), 20)
        keys = [key for batch in batches for key in batch["task_keys"]]
        self.assertEqual(len(keys), 340)
        self.assertEqual(len(set(keys)), 340)
        route_sets = {}
        for batch in batches:
            routes = tuple(batch["routes"])
            route_sets.setdefault(batch["scenario"], routes)
            self.assertEqual(route_sets[batch["scenario"]], routes)
        self.assertEqual(sum(len(routes) for routes in route_sets.values()), 85)
        self.assertEqual(
            {(batch["scenario"], batch["rsu_delay_ms"]) for batch in batches},
            {("_%d" % scenario, delay)
             for scenario in range(1, 6) for delay in (0, 200, 500, 800)},
        )

    def test_preview_with_one_two_four_or_six_workers_has_same_task_set(self):
        batches = self.module.build_batches(self.tasks())
        expected = {key for batch in batches for key in batch["task_keys"]}
        for count in (1, 2, 4, 6):
            preview = self.module.build_preview(batches, count, 45000, "rsu-full-test")
            self.assertEqual(len(preview), 20)
            self.assertEqual(
                {key for item in preview for key in item["task_keys"]}, expected
            )
            self.assertEqual(
                {item["port"] for item in preview},
                {45000 + index * 200 for index in range(20)},
            )
            route_ports = [port for item in preview for port in item["route_ports"]]
            self.assertEqual(len(route_ports), 340)
            self.assertEqual(len(set(route_ports)), 340)
            self.assertEqual(preview[0]["route_ports"][:2], [45000, 45010])
            self.assertEqual(preview[1]["route_ports"][:2], [45200, 45210])
            self.assertLessEqual(max(route_ports) + 5, 65535)

    def test_preview_rejects_a_port_range_that_cannot_fit_all_episodes(self):
        batches = self.module.build_batches(self.tasks())
        with self.assertRaisesRegex(ValueError, "base port range"):
            self.module.build_preview(batches, 6, 62000, "rsu-full-test")

    def test_runner_command_carries_only_rsu_scenario_seeds_and_cohort(self):
        batch = self.module.build_batches(self.tasks())[7]
        command = self.module.runner_command(batch, 3, 45020, "rsu-full-test")
        self.assertEqual(command[0:2], ["bash", "scripts/dual_fifo_multiroute_sc.sh"])
        self.assertIn("--scenario", command)
        self.assertIn(batch["scenario"], command)
        self.assertIn("--carla-seed", command)
        self.assertIn("2000", command)
        self.assertIn("--cohort", command)
        self.assertIn("rsu-full-test", command)
        self.assertEqual(command[-4], str(batch["rsu_delay_ms"]))
        self.assertEqual(command[-3], "0")
        self.assertEqual(command[-1], "0")

    def test_cohort_creation_never_overwrites_an_existing_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.module.create_cohort(Path(tmp), "fixed-cohort")
            (first / "events.jsonl").write_text("old evidence\n")
            with self.assertRaises(FileExistsError):
                self.module.create_cohort(Path(tmp), "fixed-cohort")
            self.assertEqual((first / "events.jsonl").read_text(), "old evidence\n")

    def test_dry_run_writes_a_complete_nonexecuting_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, COORDINATOR, "--dry-run", "--seed", "2000",
                 "--output-root", tmp, "--cohort", "dry-contract"],
                cwd=REPO,
                env=dict(os.environ, GPU_POOL="6"),
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(payload["planned_tasks"], 340)
            self.assertEqual(payload["planned_batches"], 20)
            cohort = Path(payload["cohort_dir"])
            self.assertTrue((cohort / "manifest.json").is_file())
            self.assertTrue((cohort / "batches.json").is_file())
            self.assertFalse((cohort / "events.jsonl").exists())

    def result_fixture(self, root, countable=True, duplicate=False):
        root = Path(root)
        run_dir = root / "runner"
        run_dir.mkdir()
        tasks = []
        rows = []
        for route, status, ds, timeout in (
            (0, "Completed", 100.0, 0.0),
            (1, "Failed", 39.0, 1.0),
        ):
            task = {
                "key": "task-%d" % route, "scenario": "_1", "route": route,
                "rsu_delay_ms": 200, "ego_delay_ms": 0, "carla_seed": 2000,
                "traffic_seed": 2000, "repeat": 0,
                "agent": "codriving_dualfifo_rsu200_l1",
            }
            tasks.append(task)
            ego = run_dir / ("ego-%d.json" % route)
            ego.write_text(json.dumps({"_checkpoint": {"global_record": {
                "status": status,
                "scores": {"score_composed": ds, "score_route": 100.0},
                "infractions": {
                    "collisions_layout": 0.0,
                    "collisions_pedestrian": 0.0,
                    "collisions_vehicle": 1.0 if route == 0 else 0.0,
                    "vehicle_blocked": timeout,
                    "route_timeout": timeout,
                },
            }}}))
            rsu_audit = run_dir / ("rsu-%d.csv" % route)
            ego_audit = run_dir / ("ego-audit-%d.csv" % route)
            rsu_audit.write_text("ok\n")
            ego_audit.write_text("ok\n")
            rows.append({
                "route": str(route), "repeat": "0",
                "rsu_fifo_delay_ms": "200", "rsu_delay_frames": "4",
                "ego_output_fifo_delay_ms": "0", "ego_delay_frames": "0",
                "DS": str(ds), "RC": "100.0", "status": status,
                "countable": str(countable), "coll_lay": "0.0",
                "coll_veh": "1.0" if route == 0 else "0.0",
                "ego_json": str(ego), "rsu_audit_csv": str(rsu_audit),
                "ego_output_audit_csv": str(ego_audit),
                "agent_config": "/repo/pnp_config_codriving_dualfifo_rsu200_l1.yaml",
                "sim_rate_hz": "20", "git_commit": "abc", "git_dirty": "false",
            })
        if duplicate:
            rows.append(dict(rows[0]))
        batch = {
            "key": "sc1_d200_cseed2000_tseed2000_rep0", "scenario": "_1",
            "rsu_delay_ms": 200, "ego_delay_ms": 0, "carla_seed": 2000,
            "traffic_seed": 2000, "repeat": 0, "routes": [0, 1],
            "task_keys": [task["key"] for task in tasks],
        }
        (root / "manifest.json").write_text(json.dumps(tasks))
        (root / "batches.json").write_text(json.dumps([batch]))
        (root / "metadata.json").write_text(json.dumps({
            "git_commit": "abc", "cohort": "fixture-cohort",
        }))
        with (run_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        runner_log = root / "runner.log"
        runner_log.write_text(
            "[worker] GPU=3 port=45000 cohort=fixture-cohort arm=rsu200 "
            "scenario=_1 carla_seed=2000 traffic_seed=2000 routes={0 1} "
            "repeat=0 run=%s\n" % run_dir
        )
        (root / "events.jsonl").write_text(json.dumps({
            "event": "batch_finished", "batch_key": batch["key"],
            "return_code": 0, "run_dir": str(run_dir), "log": str(runner_log),
        }) + "\n")
        return root

    def test_result_check_keeps_failed_driving_and_zeroes_timeout_ds(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.module.summarize_cohort(self.result_fixture(tmp))
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["countable"], 2)
        arm = summary["by_delay_ms"]["200"]
        self.assertEqual(arm["Completed"], 1)
        self.assertEqual(arm["Failed"], 1)
        self.assertEqual(arm["route_timeout"], 1)
        self.assertEqual(arm["vehicle_blocked"], 1)
        self.assertEqual(arm["collision_episodes"], 1)
        self.assertEqual(arm["mean_DS"], 50.0)
        self.assertEqual(arm["mean_RC"], 100.0)

    def test_result_check_reports_duplicate_and_audit_gate_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.module.summarize_cohort(
                self.result_fixture(tmp, countable=False, duplicate=True)
            )
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["infra_invalid"], 3)
        self.assertTrue(any("duplicate" in issue for issue in summary["issues"]))

    def test_result_check_rejects_actual_runner_scenario_or_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort = self.result_fixture(tmp)
            log = cohort / "runner.log"
            log.write_text(log.read_text().replace(
                "scenario=_1 carla_seed=2000",
                "scenario=_2 carla_seed=2001",
            ))
            summary = self.module.summarize_cohort(cohort)
        self.assertFalse(summary["complete"])
        self.assertTrue(any("runner identity mismatch" in issue
                            for issue in summary["issues"]))

    def test_executor_runs_batches_on_owned_worker_slots_and_records_events(self):
        batches = self.module.build_batches(self.tasks())[:3]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp)
            (cohort / "logs").mkdir()
            command = [sys.executable, "-c", "print('run=/tmp/fake-run')"]
            with mock.patch.object(self.module, "runner_command", return_value=command):
                executor = self.module.SweepExecutor(
                    batches, [1, 3], 45000, cohort, "executor-test"
                )
                self.assertEqual(executor.run(), 0)
            events = [json.loads(line) for line in
                      (cohort / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(executor.completed), 3)
        self.assertFalse(executor.failures)
        self.assertEqual(sum(event["event"] == "batch_started" for event in events), 3)
        self.assertEqual(sum(event["event"] == "batch_finished" for event in events), 3)
        self.assertTrue({event["gpu"] for event in events}.issubset({1, 3}))

    def test_executor_dispatches_remaining_batches_after_runner_failure(self):
        batches = self.module.build_batches(self.tasks())[:2]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp)
            (cohort / "logs").mkdir()
            command = [sys.executable, "-c", "raise SystemExit(7)"]
            with mock.patch.object(self.module, "runner_command", return_value=command):
                executor = self.module.SweepExecutor(
                    batches, [1], 45000, cohort, "executor-test"
                )
                self.assertEqual(executor.run(), 1)
            events = [json.loads(line) for line in
                      (cohort / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(executor.failures), 2)
        self.assertEqual(sum(event["event"] == "batch_started" for event in events), 2)

    def test_executor_dispatches_next_batch_when_runner_cannot_start(self):
        batches = self.module.build_batches(self.tasks())[:2]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp)
            (cohort / "logs").mkdir()
            actual_popen = subprocess.Popen
            attempts = 0

            def start(command, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("temporary process limit")
                return actual_popen(command, **kwargs)

            with mock.patch.object(self.module, "runner_command",
                                   return_value=[sys.executable, "-c", "print('ok')"]), \
                    mock.patch.object(self.module.subprocess, "Popen", side_effect=start):
                executor = self.module.SweepExecutor(
                    batches, [1], 45000, cohort, "executor-test"
                )
                self.assertEqual(executor.run(), 1)
            events = [json.loads(line) for line in
                      (cohort / "events.jsonl").read_text().splitlines()]
        self.assertEqual(attempts, 2)
        self.assertEqual(sum(event["event"] == "batch_finished" for event in events), 2)

    def test_recovery_accepts_valid_rows_from_failed_source_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort = self.result_fixture(tmp)
            event_path = cohort / "events.jsonl"
            event = json.loads(event_path.read_text())
            event["return_code"] = 1
            event_path.write_text(json.dumps(event) + "\n")
            accepted, invalid = self.module.collect_verified_results(cohort)
        self.assertEqual(set(accepted), {"task-0", "task-1"})
        self.assertEqual(invalid, {})

    def test_recovery_keeps_countable_driving_timeout_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort = self.result_fixture(tmp)
            summary_path = cohort / "runner" / "summary.csv"
            with summary_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["status"] = "Failed - Agent timed out"
            with summary_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            ego = Path(rows[1]["ego_json"])
            payload = json.loads(ego.read_text())
            payload["_checkpoint"]["global_record"]["status"] = rows[1]["status"]
            ego.write_text(json.dumps(payload))
            accepted, invalid = self.module.collect_verified_results(cohort)
        self.assertIn("task-1", accepted)
        self.assertNotIn("task-1", invalid)
        self.assertEqual(accepted["task-1"]["DS"], 0.0)

    def test_recovery_retries_conflicting_duplicate_route_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort = self.result_fixture(tmp, duplicate=True)
            accepted, invalid = self.module.collect_verified_results(cohort)
            tasks = json.loads((cohort / "manifest.json").read_text())
        self.assertNotIn("task-0", accepted)
        self.assertIn("task-0", invalid)
        retry = self.module.plan_pending_batches(tasks, accepted)
        self.assertIn("task-0", retry[0]["task_keys"])

    def test_recovery_plan_excludes_reused_tasks(self):
        tasks = self.tasks()
        reused = {task["key"] for task in tasks[:97]}
        batches = self.module.plan_pending_batches(tasks, reused)
        self.assertEqual(sum(len(batch["task_keys"]) for batch in batches), 243)
        self.assertTrue(all(key not in reused for batch in batches
                            for key in batch["task_keys"]))

    def test_recovery_summary_zeroes_only_twice_invalid_ds(self):
        tasks = self.tasks()[:2]
        first = {tasks[0]["key"]: {"DS": 60.0, "RC": 90.0,
                                  "status": "Failed", "route_timeout": False,
                                  "vehicle_blocked": False, "collision": False}}
        summary = self.module.summarize_recovery_tasks(tasks, {}, first, {})
        arm = summary["by_delay_ms"]["0"]
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["countable"], 1)
        self.assertEqual(summary["zeroed_after_retry"], 1)
        self.assertEqual(summary["zeroed_task_keys"], [tasks[1]["key"]])
        self.assertEqual(arm["mean_DS"], 30.0)
        self.assertEqual(arm["mean_RC"], 90.0)
        self.assertEqual(arm["first_pass_failure_rate"], 0.5)
        self.assertEqual(arm["final_failure_rate"], 0.5)

    def test_recovery_run_schedules_only_unmeasured_then_only_failed_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "source").mkdir()
            source = self.result_fixture(Path(tmp) / "source")
            tasks = json.loads((source / "manifest.json").read_text())
            reused_row = {"DS": 60.0, "RC": 90.0, "status": "Failed",
                          "route_timeout": False, "vehicle_blocked": False,
                          "collision": False}
            with mock.patch.object(self.module, "build_manifest", return_value=tasks), \
                    mock.patch.object(self.module, "query_idle_gpus", return_value=[0, 1]), \
                    mock.patch.object(self.module, "git_provenance", return_value={
                        "git_commit": "abc", "git_dirty": False,
                    }), \
                    mock.patch.object(self.module, "collect_verified_results",
                        side_effect=[({tasks[0]["key"]: reused_row}, {}),
                                     ({}, {tasks[1]["key"]: "first failure"})]), \
                    mock.patch.object(self.module, "summarize_recovery_cohort",
                        return_value={"complete": True, "reused": 1,
                                      "countable": 1, "zeroed_after_retry": 1}), \
                    mock.patch.object(self.module.SweepExecutor, "run", return_value=1), \
                    mock.patch.dict(os.environ, {"GPU_POOL": "2"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.module.main([
                    "--run", "--reuse-cohort", str(source),
                    "--expected-reused", "1", "--output-root", tmp,
                    "--cohort", "recovery-test",
                ])
            cohort = Path(tmp) / "recovery-test"
            first = json.loads((cohort / "batches.json").read_text())
            retry = json.loads((cohort / "retry_batches.json").read_text())
            self.assertEqual([key for batch in first for key in batch["task_keys"]],
                             [tasks[1]["key"]])
            self.assertEqual([key for batch in retry for key in batch["task_keys"]],
                             [tasks[1]["key"]])
            self.assertEqual(json.loads((cohort / "metadata.json").read_text())
                             ["reuse_cohort"], str(source.resolve()))

    def test_termination_targets_only_active_runner_process_groups(self):
        executor = self.module.SweepExecutor([], [1], 45000, "/tmp", "termination-test")
        process = mock.Mock(pid=12345)
        process.poll.return_value = None
        executor.active[0] = process
        with mock.patch.object(self.module.os, "getpgid", return_value=54321), \
                mock.patch.object(self.module.os, "killpg") as killpg:
            executor.terminate_active()
        killpg.assert_called_once_with(54321, self.module.signal.SIGTERM)
        self.assertTrue(executor.stopping.is_set())

    def test_main_run_selects_gpus_before_creating_and_records_them(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(self.module, "query_idle_gpus", return_value=[2, 4]), \
                mock.patch.object(self.module, "git_provenance", return_value={
                    "git_commit": "abc", "git_dirty": False,
                }), \
                mock.patch.object(self.module.SweepExecutor, "run", return_value=0), \
                mock.patch.dict(os.environ, {"GPU_POOL": "2"}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.module.main([
                "--run", "--output-root", tmp, "--cohort", "main-run-test"
            ])
            metadata = json.loads(
                (Path(tmp) / "main-run-test" / "metadata.json").read_text()
            )
        self.assertEqual(result, 0)
        self.assertEqual(metadata["selected_gpus"], [2, 4])
        self.assertEqual(len(metadata["code_sha256"]), 5)
        self.assertTrue(all(len(value) == 64
                            for value in metadata["code_sha256"].values()))
        self.assertEqual(json.loads(output.getvalue())["selected_gpus"], [2, 4])

    def test_check_results_cli_returns_success_only_for_complete_cohort(self):
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            cohort = self.result_fixture(tmp)
            self.assertEqual(self.module.main(["--check-results", str(cohort)]), 0)
            self.assertTrue(json.loads(output.getvalue())["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
