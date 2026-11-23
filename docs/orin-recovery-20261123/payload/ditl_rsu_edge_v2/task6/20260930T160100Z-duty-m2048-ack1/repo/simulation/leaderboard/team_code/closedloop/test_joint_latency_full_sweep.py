"""Contracts for the ten-arm joint-latency full sweep."""

import json
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
import joint_latency_full_manifest as manifest
import joint_latency_full_sweep as sweep


class JointManifestTest(unittest.TestCase):
    def test_exact_joint_arm_cross_product_and_850_unique_tasks(self):
        tasks = manifest.build_manifest()
        self.assertEqual(len(tasks), 850)
        self.assertEqual(len({task["key"] for task in tasks}), 850)
        self.assertEqual(len({(task["scenario"], task["route"]) for task in tasks}), 85)
        self.assertEqual(len({(task["rsu_delay_ms"], task["ego_delay_ms"]) for task in tasks}), 10)
        self.assertEqual({(task["rsu_delay_ms"], task["ego_delay_ms"]) for task in tasks}, {
            (100, 200), (100, 300), (200, 300), (100, 400), (200, 400),
            (300, 400), (100, 500), (200, 500), (300, 500), (400, 500),
        })
        self.assertTrue(all(task["profile"] == "5_10" for task in tasks))
        self.assertTrue(all(task["agent"] == "codriving_dualfifo_joint_rsu%d_ego%d_l1" %
                            (task["rsu_delay_ms"], task["ego_delay_ms"]) for task in tasks))

    def test_seed_repeat_identity_and_invalid_arguments(self):
        tasks = manifest.build_manifest(2001, 2002, 3)
        self.assertTrue(all("cseed2001_tseed2002_rep3" in task["key"] for task in tasks))
        for value in (-1,):
            with self.assertRaises(ValueError):
                manifest.build_manifest(value, 2000, 0)

    def test_cadence_two_creates_a_disjoint_exact_age_namespace(self):
        joint = manifest.build_manifest()
        exact_age = manifest.build_manifest(cadence=2)
        self.assertEqual(len(exact_age), 850)
        self.assertEqual(len({task["key"] for task in exact_age}), 850)
        self.assertFalse({task["key"] for task in joint} & {task["key"] for task in exact_age})
        self.assertTrue(all(task["key"].startswith("jointc2_") for task in exact_age))
        self.assertTrue(all("jointc2" in task["tag"] for task in exact_age))
        self.assertTrue(all(task["agent"] == "codriving_dualfifo_jointc2_rsu%d_ego%d_l1" %
                            (task["rsu_delay_ms"], task["ego_delay_ms"]) for task in exact_age))


class JointSweepTest(unittest.TestCase):
    def test_strict_effective_matrix_has_four_arms_and_340_tasks(self):
        tasks = sweep.select_tasks_by_arms(
            manifest.build_manifest(), sweep.STRICT_EFFECTIVE_ARMS)
        self.assertEqual({(task["rsu_delay_ms"], task["ego_delay_ms"])
                          for task in tasks}, {
            (200, 300), (200, 400), (200, 500), (400, 500),
        })
        self.assertEqual(len(tasks), 340)
        self.assertEqual(len({task["key"] for task in tasks}), 340)

    def test_101_reused_tasks_leave_exactly_239_routes_to_schedule(self):
        tasks = sweep.select_tasks_by_arms(
            manifest.build_manifest(), sweep.STRICT_EFFECTIVE_ARMS)
        batches = sweep.build_batches(tasks)
        reused = {task["key"]: {} for task in tasks[:101]}
        pending = sweep.plan_retry_batches(batches, reused)
        self.assertEqual(sum(len(batch["task_keys"]) for batch in pending), 239)
        self.assertEqual({key for batch in pending for key in batch["task_keys"]},
                         {task["key"] for task in tasks[101:]})

    def test_reuse_loader_filters_to_selected_tasks_and_merges_rounds(self):
        tasks = manifest.build_manifest()
        selected = sweep.select_tasks_by_arms(tasks, sweep.STRICT_EFFECTIVE_ARMS)
        dropped = next(task for task in tasks if task not in selected)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "batches.json").write_text("[]")
            (source / "retry_batches.json").write_text("[]")
            first = {
                selected[0]["key"]: {"DS": 1.0},
                dropped["key"]: {"DS": 2.0},
            }
            retry = {selected[1]["key"]: {"DS": 3.0}}
            with mock.patch.object(sweep, "collect_verified_results",
                                   side_effect=((first, {}), (retry, {}))):
                reused, report = sweep.load_reused_results(source, selected)
        self.assertEqual(set(reused), {selected[0]["key"], selected[1]["key"]})
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["source_first_accepted"], 2)
        self.assertEqual(report["source_retry_accepted"], 1)

    def test_reused_result_is_not_reported_as_a_new_first_pass(self):
        task = sweep.select_tasks_by_arms(
            manifest.build_manifest(), sweep.STRICT_EFFECTIVE_ARMS)[0]
        result = {"DS": 80.0, "RC": 100.0, "status": "Completed"}
        summary, outcomes = sweep.summarize_tasks([task], {}, {},
                                                  {task["key"]: result})
        self.assertEqual(outcomes[0]["outcome"], "reused")
        self.assertEqual(summary["reused"], 1)
        self.assertEqual(summary["countable"], 1)

    def test_reuse_only_cohort_is_complete_without_synthetic_runner_events(self):
        task = sweep.select_tasks_by_arms(
            manifest.build_manifest(), sweep.STRICT_EFFECTIVE_ARMS)[0]
        result = {"DS": 80.0, "RC": 100.0, "status": "Completed"}
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            (cohort / "manifest.json").write_text(json.dumps([task]))
            (cohort / "batches.json").write_text("[]")
            (cohort / "retry_batches.json").write_text("[]")
            (cohort / "metadata.json").write_text(json.dumps({
                "cohort": "reuse-only", "git_commit": "fixture-commit"}))
            (cohort / "reused_results.json").write_text(
                json.dumps({task["key"]: result}))
            summary, outcomes, _, _ = sweep.summarize_cohort(cohort)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["reused"], 1)
        self.assertEqual(outcomes[0]["outcome"], "reused")

    def test_runner_environment_owns_run_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runner_runs"
            environment = sweep.runner_environment(root)
            self.assertEqual(environment["DUAL_FIFO_RUN_ROOT"], str(root))

    def test_executor_records_failure_and_continues_next_batch(self):
        tasks = manifest.build_manifest()
        batches = sweep.build_batches([tasks[0], tasks[16]])
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.return_code = 1 if command[4] == "_1" else 0
                self.pid = 12345
                self.stdout = kwargs["stdout"]
                self.asserted_env = kwargs["env"]["DUAL_FIFO_RUN_ROOT"]
            def wait(self):
                self.stdout.write("[worker] run=/tmp/fake-run\n")
                return self.return_code
            def poll(self):
                return self.return_code
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            (cohort / "logs").mkdir()
            with mock.patch.object(sweep, "wait_for_gpu_memory_release",
                                   return_value=True), \
                    mock.patch.object(sweep.subprocess, "Popen",
                                      side_effect=FakeProcess) as launch:
                executor = sweep.SweepExecutor(batches, [4], 44000, cohort, "fixture", "first")
                executor.run()
            self.assertEqual(launch.call_count, 2)
            self.assertTrue(all(call[1]["env"]["DUAL_FIFO_RUN_ROOT"] ==
                                str(cohort / "runner_runs") for call in launch.call_args_list))
            events = [json.loads(line) for line in (cohort / "events.jsonl").read_text().splitlines()]
            finished = [event for event in events if event["event"] == "batch_finished"]
            self.assertEqual([event["return_code"] for event in finished], [1, 0])
            self.assertEqual(finished[0]["port"], 44000)
            self.assertEqual(finished[1]["port"], 44200)

    def test_executor_waits_for_gpu_memory_release_before_next_batch(self):
        tasks = manifest.build_manifest(cadence=2)
        batches = sweep.build_batches([tasks[0], tasks[16]])
        state = {"queries": 0, "launches": 0}
        test_case = self

        def gpu_state(_gpu):
            state["queries"] += 1
            return (0, 4096) if state["queries"] == 2 else (0, 1)

        class FakeProcess:
            def __init__(self, command, **kwargs):
                state["launches"] += 1
                if state["launches"] == 2:
                    test_case.assertGreaterEqual(
                        state["queries"], 3,
                        "second batch launched before owned GPU memory was released")
                self.pid = 12345
                self.stdout = kwargs["stdout"]
            def wait(self):
                self.stdout.write("[worker] run=/tmp/fake-run\n")
                return 0
            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            (cohort / "logs").mkdir()
            executor = sweep.SweepExecutor(batches, [0], 44000, cohort,
                                           "fixture", "first")
            with mock.patch.object(sweep, "query_gpu_state", side_effect=gpu_state,
                                   create=True), \
                    mock.patch.object(sweep, "GPU_HANDOFF_POLL_SECONDS", 0,
                                      create=True), \
                    mock.patch.object(sweep.subprocess, "Popen", side_effect=FakeProcess):
                executor.worker(0, 0)
            events = [json.loads(line) for line in
                      (cohort / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(event["event"] == "batch_finished"
                                 for event in events), 2)

    def test_50_batches_cover_tasks_once_and_isolate_route_ports(self):
        tasks = manifest.build_manifest()
        batches = sweep.build_batches(tasks)
        self.assertEqual(len(batches), 50)
        self.assertEqual(len({key for batch in batches for key in batch["task_keys"]}), 850)
        preview = sweep.build_preview(batches, 2, 44000, "joint-test")
        self.assertEqual(len({port for batch in preview for port in batch["route_ports"]}), 850)
        self.assertEqual({batch["slot"] for batch in preview}, {0, 1})
        self.assertEqual(preview[0]["command"][2], "--joint")
        self.assertEqual(preview[0]["command"][-4:-2], ["100", "200"])
        with self.assertRaises(ValueError):
            sweep.build_preview(batches, 2, 56000, "joint-test")

    def test_gpu_pool_is_a_worker_count_from_one_through_eight(self):
        for valid in ("1", "2", "4", "6", "8"):
            self.assertEqual(sweep.parse_gpu_pool(valid), int(valid))
        for invalid in (None, "", "0", "9", "1,2,3,4"):
            with self.assertRaises(ValueError):
                sweep.parse_gpu_pool(invalid)
        rows = "0, 0, 5\n1, 6, 0\n2, 0, 51\n3, 5, 50\n4, 0, 0\n5, 0, 1\n"
        self.assertEqual(sweep.select_idle_gpus(rows, 2), [0, 3])
        with self.assertRaises(RuntimeError):
            sweep.select_idle_gpus(rows, 6)

    def test_dry_run_persists_manifest_and_cohort_without_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts/joint_latency_full_sweep.py"),
                 "--dry-run", "--output-root", temp, "--cohort", "contract"],
                cwd=str(REPO), env=dict(os.environ, GPU_POOL="4"),
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual((payload["planned_tasks"], payload["planned_batches"]), (850, 50))
            cohort = Path(payload["cohort_dir"])
            self.assertEqual(len(json.loads((cohort / "manifest.json").read_text())), 850)
            self.assertEqual(len(json.loads((cohort / "batches.json").read_text())), 50)
            self.assertFalse((cohort / "events.jsonl").exists())
            self.assertFalse((cohort / "task_outcomes.json").exists())

    def test_gpu_admission_failure_does_not_create_a_formal_cohort(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.dict(os.environ, {"GPU_POOL": "2"}), \
                mock.patch.object(sweep, "query_idle_gpus",
                                  side_effect=RuntimeError("insufficient idle GPUs")):
            self.assertEqual(sweep.main([
                "--run", "--output-root", temp, "--cohort", "admission-failed",
            ]), 2)
            self.assertFalse((Path(temp) / "admission-failed").exists())

    def test_exact_age_dry_run_uses_independent_cadence_two_cohort_and_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts/joint_latency_full_sweep.py"),
                 "--exact-age-grid", "--dry-run", "--output-root", temp,
                 "--cohort", "exact-age-contract"],
                cwd=str(REPO), env=dict(os.environ, GPU_POOL="2"),
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual((payload["planned_tasks"], payload["planned_batches"]), (850, 50))
            cohort = Path(payload["cohort_dir"])
            tasks = json.loads((cohort / "manifest.json").read_text())
            batches = json.loads((cohort / "batches.json").read_text())
            metadata = json.loads((cohort / "metadata.json").read_text())
            self.assertTrue(all(task["key"].startswith("jointc2_") for task in tasks))
            self.assertEqual({item["slot"] for item in batches}, {0, 1})
            self.assertTrue(all(item["command"][2] == "--joint-c2" for item in batches))
            self.assertEqual(metadata["cadence"], 2)
            self.assertEqual(metadata["worker_count"], 2)
            self.assertTrue(metadata["independent_exact_age_cohort"])
            self.assertFalse(metadata["reuse_cohort"])

    def test_exact_age_grid_resumes_only_from_the_same_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            tasks = manifest.build_manifest(cadence=2)
            (source / "manifest.json").write_text(json.dumps(tasks))
            (source / "batches.json").write_text("[]")
            (source / "retry_batches.json").write_text("[]")
            (source / "metadata.json").write_text(json.dumps({
                "cohort": "source", "git_commit": "fixture-commit", "cadence": 2,
                "exact_age_grid": True,
            }))
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts/joint_latency_full_sweep.py"),
                 "--exact-age-grid", "--reuse-cohort", str(source),
                 "--dry-run", "--output-root", temp, "--cohort", "resumed"],
                cwd=str(REPO), env=dict(os.environ, GPU_POOL="6"),
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            metadata = json.loads((Path(payload["cohort_dir"]) / "metadata.json").read_text())
            self.assertEqual(metadata["worker_count"], 6)
            self.assertEqual(metadata["reuse_cohort"], str(source))
            self.assertFalse(metadata["independent_exact_age_cohort"])
            self.assertEqual(metadata["cohort_origin"], "exact_age_resume")

    def test_unresolved_tasks_are_zeroed_and_failure_rate_is_reported(self):
        tasks = manifest.build_manifest()[:2]
        first = {tasks[0]["key"]: {"task_key": tasks[0]["key"], "DS": 80.0,
                                   "RC": 100.0, "status": "Completed"}}
        summary, outcomes = sweep.summarize_tasks(tasks, first, {})
        self.assertEqual(summary["countable"], 1)
        self.assertEqual(summary["zeroed_after_retry"], 1)
        self.assertEqual(summary["infra_failure_rate"], 0.5)
        self.assertEqual(summary["mean_DS"], 40.0)
        self.assertEqual(outcomes[1]["DS"], 0.0)
        self.assertEqual(outcomes[1]["outcome"], "infra_failure_zeroed")

    def test_retry_selection_preserves_batch_index_and_only_failed_batches(self):
        batches = sweep.build_batches(manifest.build_manifest())
        retry = sweep.plan_retry_batches(batches, {batches[0]["task_keys"][0]: {}})
        self.assertEqual(len(retry), 50)
        self.assertEqual(retry[0]["index"], 0)
        complete = {key: {} for batch in batches[:1] for key in batch["task_keys"]}
        retry = sweep.plan_retry_batches(batches, complete)
        self.assertEqual(len(retry), 49)
        self.assertNotIn(batches[0]["key"], {batch["key"] for batch in retry})

    def test_exact_age_evidence_uses_jointc2_identity_and_summary_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            task = manifest.build_manifest(cadence=2)[0]
            batch = sweep.build_batches([task])[0]
            run = cohort / "run"
            run.mkdir()
            (cohort / "logs").mkdir()
            checkpoint = run / "ego.json"
            checkpoint.write_text(json.dumps({"_checkpoint": {"global_record": {
                "status": "Completed", "infractions": {},
            }}}))
            audit_rsu, audit_ego = run / "rsu.csv", run / "ego.csv"
            audit_rsu.write_text("evidence\n")
            audit_ego.write_text("evidence\n")
            fields = ["route", "repeat", "rsu_fifo_delay_ms", "ego_output_fifo_delay_ms",
                      "agent_config", "git_commit", "ego_json", "rsu_audit_csv",
                      "ego_output_audit_csv", "countable", "status", "DS", "RC"]
            row = dict(route=task["route"], repeat=task["repeat"],
                       rsu_fifo_delay_ms=task["rsu_delay_ms"],
                       ego_output_fifo_delay_ms=task["ego_delay_ms"],
                       agent_config="pnp_config_%s.yaml" % task["agent"],
                       git_commit="fixture-commit", ego_json=str(checkpoint),
                       rsu_audit_csv=str(audit_rsu), ego_output_audit_csv=str(audit_ego),
                       countable="True", status="Completed", DS="80", RC="100")
            with (run / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            (cohort / "manifest.json").write_text(json.dumps([task]))
            (cohort / "metadata.json").write_text(json.dumps({
                "cohort": "fixture", "git_commit": "fixture-commit", "cadence": 2,
            }))
            log = cohort / "logs" / "first.log"
            def write_identity(arm):
                log.write_text(
                    "[worker] GPU=4 port=44000 cohort=fixture arm=%s scenario=_1 "
                    "carla_seed=2000 traffic_seed=2000 routes={%d} repeat=0 run=%s\n"
                    % (arm, task["route"], run))
            (cohort / "events.jsonl").write_text(json.dumps({
                "event": "batch_finished", "round": "first", "batch_key": batch["key"],
                "return_code": 0, "run_dir": str(run), "log": str(log)}) + "\n")
            arm = "jointc2_rsu%d_ego%d" % (task["rsu_delay_ms"], task["ego_delay_ms"])
            write_identity(arm)
            accepted, invalid = sweep.collect_verified_results(cohort, [batch], "first")
            self.assertEqual(invalid, {})
            self.assertIn(task["key"], accepted)
            summary, _ = sweep.summarize_tasks([task], accepted, {})
            self.assertIn(arm, summary["by_arm"])
            write_identity("joint_rsu%d_ego%d" %
                           (task["rsu_delay_ms"], task["ego_delay_ms"]))
            accepted, invalid = sweep.collect_verified_results(cohort, [batch], "first")
            self.assertEqual(accepted, {})
            self.assertIn(task["key"], invalid)

    def test_exact_age_resume_accepts_verified_rows_from_an_interrupted_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            tasks = manifest.build_manifest(cadence=2)[:2]
            batch = sweep.build_batches(tasks)[0]
            run = cohort / "run"
            run.mkdir()
            (cohort / "logs").mkdir()
            checkpoint = run / "ego.json"
            checkpoint.write_text(json.dumps({"_checkpoint": {"global_record": {
                "status": "Completed", "infractions": {},
            }}}))
            audit_rsu, audit_ego = run / "rsu.csv", run / "ego.csv"
            audit_rsu.write_text("evidence\n")
            audit_ego.write_text("evidence\n")
            fields = ["route", "repeat", "rsu_fifo_delay_ms", "ego_output_fifo_delay_ms",
                      "agent_config", "git_commit", "ego_json", "rsu_audit_csv",
                      "ego_output_audit_csv", "countable", "status", "DS", "RC"]
            task = tasks[0]
            row = dict(route=task["route"], repeat=task["repeat"],
                       rsu_fifo_delay_ms=task["rsu_delay_ms"],
                       ego_output_fifo_delay_ms=task["ego_delay_ms"],
                       agent_config="pnp_config_%s.yaml" % task["agent"],
                       git_commit="fixture-commit", ego_json=str(checkpoint),
                       rsu_audit_csv=str(audit_rsu), ego_output_audit_csv=str(audit_ego),
                       countable="True", status="Completed", DS="80", RC="100")
            with (run / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            (cohort / "manifest.json").write_text(json.dumps(tasks))
            (cohort / "batches.json").write_text(json.dumps([batch]))
            (cohort / "retry_batches.json").write_text("[]")
            (cohort / "metadata.json").write_text(json.dumps({
                "cohort": "fixture", "git_commit": "fixture-commit", "cadence": 2,
                "exact_age_grid": True,
            }))
            log = cohort / "logs" / "first.log"
            log.write_text(
                "[worker] GPU=6 port=10004 cohort=fixture arm=jointc2_rsu100_ego200 "
                "scenario=_1 carla_seed=2000 traffic_seed=2000 routes={%s} "
                "repeat=0 run=%s\n" % (" ".join(map(str, batch["routes"])), run))
            (cohort / "events.jsonl").write_text(json.dumps({
                "event": "batch_finished", "round": "first", "batch_key": batch["key"],
                "return_code": 143, "run_dir": str(run), "log": str(log)}) + "\n")
            reused, report = sweep.load_reused_results(
                cohort, tasks, allow_partial=True, require_manifest_match=True)
            self.assertEqual(set(reused), {tasks[0]["key"]})
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["source_first_invalid"], 1)

    def test_exact_age_resume_preserves_results_reused_by_the_source_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            task = manifest.build_manifest(cadence=2)[0]
            result = {"task_key": task["key"], "status": "Completed",
                      "DS": 75.0, "RC": 100.0, "route_timeout": False,
                      "vehicle_blocked": False, "collision": False}
            (cohort / "manifest.json").write_text(json.dumps([task]))
            (cohort / "batches.json").write_text("[]")
            (cohort / "retry_batches.json").write_text("[]")
            (cohort / "metadata.json").write_text(json.dumps({
                "cohort": "fixture", "git_commit": "fixture-commit", "cadence": 2,
                "exact_age_grid": True,
            }))
            (cohort / "reused_results.json").write_text(json.dumps({task["key"]: result}))
            reused, report = sweep.load_reused_results(
                cohort, [task], allow_partial=True, require_manifest_match=True)
            self.assertEqual(reused, {task["key"]: result})
            self.assertEqual(report["source_inherited_accepted"], 1)

    def test_evidence_gate_accepts_terminal_driving_failure_but_rejects_mismatched_delay(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            task = manifest.build_manifest()[0]
            batch = sweep.build_batches([task])[0]
            run = cohort / "run"
            run.mkdir()
            (cohort / "logs").mkdir()
            checkpoint = run / "ego.json"
            checkpoint.write_text(json.dumps({"_checkpoint": {"global_record": {
                "status": "Failed - Agent got blocked", "infractions": {
                    "route_timeout": 0, "vehicle_blocked": 1,
                },
            }}}))
            audit_rsu, audit_ego = run / "rsu.csv", run / "ego.csv"
            audit_rsu.write_text("evidence\n")
            audit_ego.write_text("evidence\n")
            fields = ["route", "repeat", "rsu_fifo_delay_ms", "ego_output_fifo_delay_ms",
                      "agent_config", "git_commit", "ego_json", "rsu_audit_csv",
                      "ego_output_audit_csv", "countable", "status", "DS", "RC"]
            row = dict(route=task["route"], repeat=task["repeat"],
                       rsu_fifo_delay_ms=task["rsu_delay_ms"],
                       ego_output_fifo_delay_ms=task["ego_delay_ms"],
                       agent_config="pnp_config_%s.yaml" % task["agent"],
                       git_commit="fixture-commit", ego_json=str(checkpoint),
                       rsu_audit_csv=str(audit_rsu), ego_output_audit_csv=str(audit_ego),
                       countable="True", status="Failed - Agent got blocked",
                       DS="31.5", RC="67")
            def write_row(data):
                with (run / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(data)
            write_row(row)
            (cohort / "manifest.json").write_text(json.dumps([task]))
            (cohort / "metadata.json").write_text(json.dumps({
                "cohort": "fixture", "git_commit": "fixture-commit"}))
            log = cohort / "logs" / "first.log"
            log.write_text("[worker] GPU=4 port=44000 cohort=fixture arm=joint_rsu100_ego200 "
                           "scenario=_1 carla_seed=2000 traffic_seed=2000 routes={%d} "
                           "repeat=0 run=%s\n" % (task["route"], run))
            (cohort / "events.jsonl").write_text(json.dumps({
                "event": "batch_finished", "round": "first", "batch_key": batch["key"],
                "return_code": 1, "run_dir": str(run), "log": str(log)}) + "\n")
            accepted, invalid = sweep.collect_verified_results(cohort, [batch], "first")
            self.assertEqual(invalid, {})
            self.assertEqual(accepted[task["key"]]["DS"], 31.5)
            (cohort / "batches.json").write_text(json.dumps([batch]))
            (cohort / "retry_batches.json").write_text("[]")
            summary, outcomes, _, _ = sweep.summarize_cohort(cohort)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["countable"], 1)
            self.assertIn("joint_rsu%d_ego%d" %
                          (task["rsu_delay_ms"], task["ego_delay_ms"]), summary["by_arm"])
            self.assertEqual(outcomes[0]["outcome"], "first_pass")
            with (run / "summary.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writerow(dict(row, route=999))
            accepted, invalid = sweep.collect_verified_results(cohort, [batch], "first")
            self.assertEqual(accepted, {})
            self.assertIn(task["key"], invalid)
            write_row(row)
            with (run / "summary.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writerow(row)
            accepted, invalid = sweep.collect_verified_results(cohort, [batch], "first")
            self.assertEqual(accepted, {})
            self.assertIn(task["key"], invalid)
            write_row(row)
            row["ego_output_fifo_delay_ms"] = "300"
            write_row(row)
            accepted, invalid = sweep.collect_verified_results(cohort, [batch], "first")
            self.assertEqual(accepted, {})
            self.assertIn(task["key"], invalid)


if __name__ == "__main__":
    unittest.main()
