"""Contracts for the recoverable five-arm CoptV2X live HIL cohort."""

from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
import io


REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import coptv2x_hil_manifest as manifest
import coptv2x_hil_result as result
import coptv2x_hil_runner as runner


class ManifestContracts(unittest.TestCase):
    def test_exact_85_route_scenario_cells_cross_five_fixed_arms(self):
        payload = manifest.build_manifest()
        tasks = payload["tasks"]
        self.assertEqual(85, len(payload["route_scenarios"]))
        self.assertEqual(425, len(tasks))
        self.assertEqual(425, len({task["task_id"] for task in tasks}))
        self.assertEqual(85, len({task["route_scenario_id"] for task in tasks}))
        self.assertEqual(
            manifest.HIL_ARMS,
            tuple(payload["arms"]))
        self.assertTrue(all(task["seed"] == 2000 for task in tasks))
        self.assertTrue(all(task["repeat"] == 0 for task in tasks))
        self.assertTrue(all(task["town"] == "Town05" for task in tasks))
        self.assertTrue(all(task["dt_ms"] == 50 and task["cadence"] == 2
                            and task["profile"] == "5_10" for task in tasks))
        self.assertEqual("joint_shcosearch_fp16", payload["arms"][-1]["artifact_id"])
        self.assertEqual(5, len(payload["inputs"]["runtime_config_sha256"]))
        self.assertEqual(64, len(payload["route_scenarios_sha256"]))
        self.assertEqual(64, len(payload["input_manifest_sha256"]))

    def test_pilot_selection_filters_and_pairs_only_frozen_cells(self):
        payload = manifest.build_manifest()
        chosen = payload["route_scenarios"][:2]
        pilot = manifest.select_pilot(payload, chosen)
        self.assertEqual(10, len(pilot["tasks"]))
        self.assertEqual(set(chosen), {task["route_scenario_id"] for task in pilot["tasks"]})
        for cell in chosen:
            self.assertEqual(set(manifest.ARM_IDS), {
                task["arm_id"] for task in pilot["tasks"]
                if task["route_scenario_id"] == cell
            })
        with self.assertRaises(ValueError):
            manifest.select_pilot(payload, ["not-an-authorized-cell"])
        with self.assertRaises(ValueError):
            manifest.select_pilot(payload, [])
        with self.assertRaises(ValueError):
            manifest.select_pilot(payload, [chosen[0], chosen[0]])

    def test_manifest_rejects_protocol_drift_and_cli_persists_fixed_list(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            manifest.build_manifest(seed=2001)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "manifest.json"
            self.assertEqual(0, manifest.main(["--output", str(output)]))
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(85, len(saved["route_scenario_records"]))
            self.assertEqual(425, len(saved["tasks"]))

    def test_five_runtime_configs_exist_without_touching_artifact_configs(self):
        payload = manifest.build_manifest()
        for arm in payload["arms"]:
            config = REPO / arm["runtime_config"]
            self.assertTrue(config.is_file(), str(config))
            self.assertIn("artifact_%s.yaml" % arm["arm_id"],
                          [path.name for path in config.parent.glob("artifact_*.yaml")])


class RunnerContracts(unittest.TestCase):
    def setUp(self):
        python_egg = "carla-0.9.10-py%d.%d-linux-x86_64.egg" % sys.version_info[:2]
        self.carla_egg_glob = mock.patch.object(
            runner.glob,
            "glob",
            return_value=[python_egg, "carla-0.9.10-py2.7-linux-x86_64.egg"],
        )
        self.carla_egg_glob.start()
        self.addCleanup(self.carla_egg_glob.stop)

    def test_atomic_state_resumes_finished_tasks_and_rejects_duplicate_completion(self):
        tasks = manifest.build_manifest()["tasks"][:2]
        with tempfile.TemporaryDirectory() as temp:
            state = runner.RunState(Path(temp) / "state.json", [item["task_id"] for item in tasks])
            state.record(tasks[0]["task_id"], {"status": "Completed"})
            state2 = runner.RunState(Path(temp) / "state.json", [item["task_id"] for item in tasks])
            self.assertEqual([tasks[1]["task_id"]], state2.pending_ids())
            with self.assertRaises(runner.DuplicateTaskResultError):
                state2.record(tasks[0]["task_id"], {"status": "Completed"})
            self.assertFalse(list(Path(temp).glob("*.tmp")))

    def test_same_state_file_refuses_a_different_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            runner.RunState(path, ["task-a"])
            with self.assertRaises(runner.ManifestMismatchError):
                runner.RunState(path, ["task-b"])

    def test_same_task_ids_with_changed_manifest_hash_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            runner.RunState(path, ["task-a"], manifest_sha256="a" * 64)
            with self.assertRaises(runner.ManifestMismatchError):
                runner.RunState(path, ["task-a"], manifest_sha256="b" * 64)

    def test_state_rejects_duplicate_ids_corruption_and_nonterminal_result(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unique"):
                runner.RunState(Path(temp) / "duplicate.json", ["x", "x"])
            path = Path(temp) / "bad.json"
            path.write_text("not-json")
            with self.assertRaisesRegex(ValueError, "unreadable"):
                runner.RunState(path, ["x"])
            path.write_text(json.dumps({"manifest_task_ids_sha256": runner._task_ids_digest(["x"]),
                                        "task_ids": ["x"], "attempts": [], "results": {}}))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                runner.RunState(path, ["x"])
            state = runner.RunState(Path(temp) / "state.json", ["x"])
            with self.assertRaisesRegex(ValueError, "terminal"):
                state.record("x", {"classification": "INFRA_RETRY"})
            with self.assertRaises(KeyError):
                state.record_attempt("other", {})

    def test_failure_classifier_separates_driver_terminal_and_infrastructure_retry(self):
        for status in ("collision", "blocked", "route_timeout", "route_not_completed"):
            self.assertEqual("DRIVING_TERMINAL", runner.classify_episode(status, 0))
        for status in ("ros_disconnected", "orin_engine_crash", "identity_mismatch",
                       "result_corrupt", "carla_crash", "agent_crash"):
            self.assertEqual("INFRA_RETRY", runner.classify_episode(status, 1))
        self.assertEqual("COMPLETED", runner.classify_episode("completed", 0))

    def test_any_cwd_dry_run_resolves_explicit_config_and_output_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "cohort"
            env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
            command = [sys.executable, str(SCRIPTS / "coptv2x_hil_runner.py"),
                       "--dry-run", "--output-dir", str(output)]
            completed = subprocess.run(command, cwd=temp, env=env, text=True,
                                       capture_output=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(425, report["planned_tasks"])
            self.assertTrue((output / "manifest.json").is_file())

    def test_run_isolation_has_per_task_workdirs_and_no_gpu_pin(self):
        task = manifest.build_manifest()["tasks"][0]
        first = runner.build_episode_command(task, Path("/runs/a"))
        second = runner.build_episode_command(task, Path("/runs/b"))
        self.assertNotEqual(first, second)
        self.assertTrue(any(argument.startswith(str(Path("/runs/a").resolve()))
                            for argument in first))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", " ".join(first))
        self.assertIn("--agent-config", first)

    def test_config_and_environment_are_explicit_per_arm_and_ros_domain(self):
        task = manifest.build_manifest()["tasks"][0]
        with tempfile.TemporaryDirectory() as temp:
            config_path = runner.prepare_task_config(task, temp, "cohort", 71)
            config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
            artifact = runner.load_hil_artifact_registry()[task["arm_id"]]
            expected_perception_dir = Path(temp).resolve() / "runtime-perception" / task["arm_id"]
            self.assertEqual(
                str(expected_perception_dir),
                config["perception"]["perception_model_dir"],
            )
            perception_hypes = __import__("yaml").load(
                (expected_perception_dir / "config.yaml").read_text(encoding="utf-8"),
                Loader=__import__("yaml").Loader,
            )
            self.assertEqual("v2xverse", perception_hypes["fusion"]["dataset"])
            artifact_checkpoint = (runner.DEFAULT_ARTIFACT_ROOT / artifact.checkpoint).resolve()
            self.assertEqual(artifact_checkpoint,
                             (expected_perception_dir / artifact.checkpoint.name).resolve())
            self.assertEqual(2, config["simulation"]["skip_frames"])
            hil = config["simulation"]["coptv2x_hil"]
            self.assertEqual("reverse_ssh_ros_proxy", hil["transport"])
            self.assertEqual(56610, hil["proxy_listen_port"])
            self.assertEqual(task["scenario_id"], hil["scenario_id"])
            self.assertIn("fusion_audit.jsonl", hil["audit_jsonl"])
            env = runner.runner_environment(temp, 71, 46000)
            self.assertEqual("71", env["ROS_DOMAIN_ID"])
            self.assertEqual("46005", env["TM_PORT"])
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            python_paths = env["PYTHONPATH"].split(os.pathsep)
            self.assertIn(str(runner.REPO_ROOT), python_paths)
            self.assertIn(
                str(runner.REPO_ROOT / "external_paths/carla_root/PythonAPI/carla"),
                python_paths,
            )
            expected_egg = "carla-0.9.10-py%d.%d-linux-x86_64.egg" % sys.version_info[:2]
            self.assertTrue(any(path.endswith(expected_egg) for path in python_paths))
            self.assertFalse(any("py2.7" in path for path in python_paths))

    def test_runner_environment_preserves_explicit_cuda_device_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
                with mock.patch.object(
                    runner.glob, "glob", return_value=[
                        "carla-0.9.10-py%d.%d-test.egg" % sys.version_info[:2]
                    ]
                ):
                    env = runner.runner_environment(temp, 71, 46000)

        self.assertEqual("2", env["CUDA_VISIBLE_DEVICES"])

    def test_runner_environment_sets_route_authority_for_agent_audits(self):
        route = REPO / "simulation/leaderboard/data/evaluation_routes/town05_short_r2.xml"
        with tempfile.TemporaryDirectory() as temp:
            env = runner.runner_environment(temp, 71, 56600, route_path=str(route))
        self.assertEqual(str(route.resolve()), env["ROUTES"])

    def test_task_config_selects_each_frozen_arm_checkpoint_directory(self):
        yaml = __import__("yaml")
        for task in manifest.build_manifest()["tasks"]:
            with tempfile.TemporaryDirectory() as temp:
                path = runner.prepare_task_config(task, temp, "cohort", 71)
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                artifact = runner.load_hil_artifact_registry()[task["arm_id"]]
                expected = str(Path(temp).resolve() / "runtime-perception" / task["arm_id"])
                self.assertEqual(expected, config["perception"]["perception_model_dir"])
                runtime_hypes = yaml.load((Path(expected) / "config.yaml").read_text(), Loader=yaml.Loader)
                frozen_hypes = yaml.safe_load(
                    (runner.DEFAULT_ARTIFACT_ROOT / artifact.config).read_text()
                )
                expected_model = dict(frozen_hypes["model"])
                expected_model_args = dict(expected_model["args"])
                for key in ("multi_class",):
                    if key in runtime_hypes["model"]["args"]:
                        expected_model_args[key] = runtime_hypes["model"]["args"][key]
                self.assertNotIn("shrink_header", frozen_hypes["model"]["args"])
                self.assertNotIn("shrink_header", runtime_hypes["model"]["args"])
                expected_model["args"] = expected_model_args
                self.assertEqual(expected_model, runtime_hypes["model"])
                self.assertEqual("v2xverse", runtime_hypes["fusion"]["dataset"])
                self.assertEqual(
                    str(runner.DEFAULT_PLANNING_FEATURE_ADAPTER_CHECKPOINT),
                    config["planning"]["feature_adapter_checkpoint"],
                )
                self.assertEqual(
                    str(runner.DEFAULT_SIMULATION_PERCEPTION_CONFIG),
                    config["planning"]["feature_adapter_hypes"],
                )
                self.assertEqual([128, 128, 128],
                                 config["planning"]["feature_adapter_source_widths"])
                expected_retained = {
                    "original_default": [128, 128, 128],
                    "schedule_only": [128, 128, 128],
                    "compression_only": [64, 64, 64],
                    "compress_then_tune": [80, 80, 80],
                    "joint_shcosearch": [32, 32, 32],
                }
                self.assertEqual(expected_retained[task["arm_id"]],
                                 config["planning"]["feature_adapter_retained_widths"])
                self.assertEqual(
                    runtime_hypes["model"]["args"]["multi_class"],
                    runner.yaml.load(
                        runner.DEFAULT_SIMULATION_PERCEPTION_CONFIG.read_text(),
                        Loader=runner.yaml.Loader,
                    )["model"]["args"]["multi_class"],
                )

    def test_checkpoint_parser_reports_completed_driving_corrupt_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            path.write_text(json.dumps({"global_record": {"status": "Completed"}}))
            self.assertEqual("completed", runner._read_checkpoint(path)[0])
            path.write_text(json.dumps({"global_record": {"status": "Failed"},
                                        "records": [{"status": "Collision"}]}))
            self.assertEqual("Collision", runner._read_checkpoint(path)[0])
            path.write_text("{")
            self.assertEqual("result_corrupt", runner._read_checkpoint(path)[0])
            path.write_text("{}")
            self.assertEqual("missing_result", runner._read_checkpoint(path)[0])

    def test_checkpoint_parser_falls_back_to_per_ego_route_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "_checkpoint": {
                    "global_record": {}, "progress": [0, 1], "records": [],
                },
            }))
            ego_path = path.parent / "ego_vehicle_0" / path.name
            ego_path.parent.mkdir()
            ego_path.write_text(json.dumps({
                "_checkpoint": {
                    "global_record": {"status": "Failed"},
                    "records": [{
                        "status": "Failed - Agent timed out",
                        "infractions": {"route_timeout": ["Route timeout."]},
                    }],
                },
            }))
            self.assertEqual(
                "route_timeout", runner._read_checkpoint(path)[0]
            )

    def test_timeout_attempts_remain_infrastructure_not_zero_score(self):
        full = manifest.build_manifest()
        selected = manifest.select_pilot(full, [full["route_scenarios"][0]])
        args = SimpleNamespace(run=True, host="127.0.0.1", port=46000,
                               ros_domain_id=71, episode_timeout=1, max_infra_attempts=1,
                               local_input_manifest="validated-before-test")
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(runner, "validate_local_input_manifest", return_value={}), \
                mock.patch.object(runner, "validate_carla_environment", return_value={"reachable": True}), \
                mock.patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 1)):
            output = Path(temp) / "infra"
            summary = runner.run_cohort(selected, output, args)
            self.assertEqual(5, summary["terminal_tasks"])
            state = json.loads((output / "state.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["classification"] == "INFRA_INVALID"
                                for item in state["results"].values()))
            self.assertTrue(all(item["carla_result"] is None for item in state["results"].values()))

    def test_infrastructure_retry_retries_same_tasks_in_one_run(self):
        full = manifest.build_manifest()
        selected = manifest.select_pilot(full, [full["route_scenarios"][0]])
        args = SimpleNamespace(run=True, host="127.0.0.1", port=46000,
                               ros_domain_id=71, episode_timeout=1, max_infra_attempts=2,
                               local_input_manifest="validated")
        calls = {"count": 0}

        def fail_once_then_complete(command, cwd, env, stdout, stderr, timeout):
            calls["count"] += 1
            if calls["count"] % 2:
                raise subprocess.TimeoutExpired("fake", 1)
            checkpoint = Path(env["CHECKPOINT_ENDPOINT"])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"global_record": {"status": "Completed"},
                                              "records": [{"status": "Completed"}]}))
            config = __import__("yaml").safe_load(Path(command[command.index("--agent-config") + 1]).read_text())
            hil = config["simulation"]["coptv2x_hil"]
            identity = {"run_id": hil["run_id"], "route_id": hil["route_id"],
                        "scenario_id": hil["scenario_id"], "arm_id": hil["arm_id"],
                        "sequence_no": 1, "source_frame": 1}
            audit = {"batch_order": ["ego", "rsu"], "trigger_identity": identity,
                     "selected_result_trigger_identity": identity,
                     "artifact_sha256": next(item["artifact_sha256"] for item in selected["tasks"]
                                              if item["arm_id"] == hil["arm_id"]),
                     "rsu_aoi_frames": 0, "remote_input_tensor_bytes": 0,
                     "selected_result_sequence_no": 1}
            audit_path = Path(hil["audit_jsonl"])
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit) + "\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(runner, "validate_local_input_manifest", return_value={
                    "path": "input.json", "sha256": "c" * 64,
                    "version": "coptv2x.route-input-host-index.v1",
                    "route_scenarios_sha256": selected["route_scenarios_sha256"],
                    "verified_feature_files": 1}), \
                mock.patch.object(runner, "validate_carla_environment", return_value={"reachable": True}), \
                mock.patch.object(runner.subprocess, "run", side_effect=fail_once_then_complete):
            output = Path(temp) / "retry"
            summary = runner.run_cohort(selected, output, args)
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(5, summary["terminal_tasks"])
            self.assertEqual(10, calls["count"])
            self.assertTrue(all(len(attempts) == 2 for attempts in state["attempts"].values()))
            self.assertTrue(all(item["classification"] == "COMPLETED" for item in state["results"].values()))

    def test_live_run_requires_exact_deployed_local_input_mapping(self):
        payload = manifest.build_manifest()
        with self.assertRaisesRegex(RuntimeError, "requires --host-input-index"):
            runner.validate_local_input_manifest(payload, None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            path.write_text(json.dumps({"schema": "coptv2x.route-input-host-index.v1",
                                        "status": "NOT_DEPLOYED"}),
                            encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "HOST_INDEX_VALID"):
                runner.validate_local_input_manifest(payload, path)

    def test_local_input_manifest_validates_all_frozen_cells_and_frame_index(self):
        payload = manifest.build_manifest()
        with tempfile.TemporaryDirectory() as temp:
            feature = Path(temp) / "frame.npy"
            import numpy as np
            with feature.open("wb") as stream:
                np.save(stream, np.zeros((2, 64, 256, 512), dtype=np.float32), allow_pickle=False)
            feature_hash = result.file_sha256(feature)
            entries = {cell: [{"source_frame": 1, "sample_id": "carla-frame-1",
                               "spatial_features_path": str(feature),
                               "spatial_features_sha256": feature_hash,
                               "shape": [2, 64, 256, 512], "dtype": "float32"}]
                       for cell in payload["route_scenarios"]}
            input_path = Path(temp) / "input.json"
            input_path.write_text(json.dumps({"schema": "coptv2x.route-input-host-index.v1",
                                              "status": "HOST_INDEX_VALID",
                                              "route_scenarios_sha256": payload["route_scenarios_sha256"],
                                              "route_scenarios": entries}), encoding="utf-8")
            accepted = runner.validate_local_input_manifest(payload, input_path)
            self.assertEqual(64, len(accepted["sha256"]))
            data = json.loads(input_path.read_text())
            first_cell = payload["route_scenarios"][0]
            data["route_scenarios"][first_cell][0]["sample_id"] = "arbitrary-static-sample"
            input_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sample_id"):
                runner.validate_local_input_manifest(payload, input_path)
            data["route_scenarios"][first_cell][0]["sample_id"] = "carla-frame-1"
            data["route_scenarios"].pop(payload["route_scenarios"][0])
            input_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "all 85"):
                runner.validate_local_input_manifest(payload, input_path)

    def test_local_input_manifest_rejects_garbage_tensor(self):
        payload = manifest.build_manifest()
        with tempfile.TemporaryDirectory() as temp:
            feature = Path(temp) / "garbage.npy"
            feature.write_bytes(b"sample")
            feature_hash = result.file_sha256(feature)
            entries = {cell: [{"source_frame": 1, "sample_id": "carla-frame-1",
                               "spatial_features_path": str(feature),
                               "spatial_features_sha256": feature_hash,
                               "shape": [2, 64, 256, 512], "dtype": "float32"}]
                       for cell in payload["route_scenarios"]}
            path = Path(temp) / "input.json"
            path.write_text(json.dumps({"schema": "coptv2x.route-input-host-index.v1",
                                        "status": "HOST_INDEX_VALID",
                                        "route_scenarios_sha256": payload["route_scenarios_sha256"],
                                        "route_scenarios": entries}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tensor|SHA-256"):
                runner.validate_local_input_manifest(payload, path)

    def test_mocked_live_cohort_writes_carla_hil_and_paired_evidence(self):
        full = manifest.build_manifest()
        selected = manifest.select_pilot(full, [full["route_scenarios"][0]])
        args = SimpleNamespace(run=True, host="127.0.0.1", port=46000,
                               ros_domain_id=71, episode_timeout=2, max_infra_attempts=1,
                               local_input_manifest="validated-before-test")

        def fake_run(command, cwd, env, stdout, stderr, timeout):
            self.assertTrue(Path(env["RESULT_ROOT"]).is_dir())
            self.assertTrue(Path(env["SAVE_PATH"]).is_dir())
            checkpoint = Path(env["CHECKPOINT_ENDPOINT"])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"_checkpoint": {
                "global_record": {"status": "Completed", "DS": 1.0, "RC": 1.0},
                "records": [{"status": "Completed"}]}}), encoding="utf-8")
            config_path = Path(command[command.index("--agent-config") + 1])
            config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
            hil = config["simulation"]["coptv2x_hil"]
            identity = {"run_id": hil["run_id"], "route_id": hil["route_id"],
                        "scenario_id": hil["scenario_id"], "arm_id": hil["arm_id"],
                        "sequence_no": 1, "source_frame": 10}
            audit = {"batch_order": ["ego", "rsu"], "trigger_identity": identity,
                     "selected_result_trigger_identity": identity,
                     "artifact_sha256": next(item["artifact_sha256"] for item in selected["tasks"]
                                              if item["arm_id"] == hil["arm_id"]),
                     "rsu_aoi_frames": 0, "remote_input_tensor_bytes": 0,
                     "selected_result_sequence_no": 1}
            audit_path = Path(hil["audit_jsonl"])
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(runner, "validate_local_input_manifest", return_value={
                    "path": "local-index.json", "sha256": "a" * 64,
                    "version": "coptv2x.route-input-host-index.v1",
                    "route_scenarios_sha256": selected["route_scenarios_sha256"],
                    "verified_feature_files": 1}), \
                mock.patch.object(runner, "validate_carla_environment", return_value={"reachable": True}), \
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
            summary = runner.run_cohort(selected, Path(temp) / "paired", args)
            self.assertEqual(5, summary["terminal_tasks"])
            state = json.loads((Path(temp) / "paired/state.json").read_text(encoding="utf-8"))
            self.assertEqual(5, len(state["results"]))
            self.assertEqual("a" * 64, state["host_input_index"]["sha256"])
            persisted_manifest = json.loads((Path(temp) / "paired/manifest.json").read_text())
            self.assertEqual("a" * 64, persisted_manifest["host_input_index"]["sha256"])
            self.assertTrue(all(item["classification"] == "COMPLETED"
                                for item in state["results"].values()))
            self.assertEqual(5, len((Path(temp) / "paired/paired_results.csv").read_text().splitlines()) - 1)

    def test_live_run_refuses_unreachable_carla_before_creating_cohort(self):
        payload = manifest.select_pilot(manifest.build_manifest(), [manifest.build_manifest()["route_scenarios"][0]])
        args = SimpleNamespace(run=True, host="127.0.0.1", port=46000,
                               local_input_manifest="validated", ros_domain_id=71)
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(runner, "validate_local_input_manifest", return_value={}), \
                mock.patch.object(runner, "validate_carla_environment", return_value={"reachable": False}):
            with self.assertRaisesRegex(RuntimeError, "CARLA server is not reachable"):
                runner.run_cohort(payload, Path(temp) / "no-run", args)
            self.assertFalse((Path(temp) / "no-run").exists())

    def test_dry_run_cohort_persists_noncountable_tasks_without_zero_scores(self):
        full = manifest.build_manifest()
        selected = manifest.select_pilot(full, [full["route_scenarios"][0]])
        args = SimpleNamespace(run=False, host="127.0.0.1", port=46000,
                               ros_domain_id=71, max_infra_attempts=1)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dry"
            summary = runner.run_cohort(selected, output, args)
            self.assertEqual(5, summary["terminal_tasks"])
            rows = (output / "paired_results.csv").read_text(encoding="utf-8")
            self.assertEqual(5, rows.count("INFRA_INVALID"))
            self.assertNotIn(",0,0,", rows)

    def test_cli_main_dry_run_reports_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = runner.main(["--dry-run", "--output-dir", str(Path(temp) / "cohort")])
            self.assertEqual(0, code)
            self.assertEqual(425, json.loads(stdout.getvalue())["planned_tasks"])


class ResultAuditContracts(unittest.TestCase):
    def test_result_requires_batch_order_trigger_artifact_hash_and_aoi(self):
        audit = result.validate_episode_audit({
            "batch_order": ["ego", "rsu"],
            "trigger_identity": {"run_id": "run", "route_id": "r2",
                                 "scenario_id": "_1", "arm_id": "original_default",
                                 "sequence_no": 1, "source_frame": 10},
            "result_identity": {"run_id": "run", "route_id": "r2",
                                "scenario_id": "_1", "arm_id": "original_default",
                                "sequence_no": 1, "source_frame": 10},
            "artifact_sha256": {"engine": "a" * 64},
            "aoi_frames": 2,
            "remote_input_tensor_bytes": 0,
        })
        self.assertEqual(2, audit["aoi_frames"])
        for key in ("batch_order", "trigger_identity", "artifact_sha256", "aoi_frames"):
            damaged = dict(audit)
            damaged.pop(key)
            with self.assertRaises(result.HilAuditError):
                result.validate_episode_audit(damaged)

    def test_fusion_audit_allows_older_result_but_rejects_cross_run_identity(self):
        audit = {
            "batch_order": ["ego", "rsu"],
            "trigger_identity": {"run_id": "run", "route_id": "r2",
                                 "scenario_id": "_1", "arm_id": "original_default",
                                 "sequence_no": 8, "source_frame": 80},
            "result_identity": {"run_id": "run", "route_id": "r2",
                                "scenario_id": "_1", "arm_id": "original_default",
                                "sequence_no": 7, "source_frame": 78},
            "artifact_sha256": {"checkpoint": "b" * 64},
            "aoi_frames": 2, "remote_input_tensor_bytes": 0,
        }
        self.assertEqual(7, result.validate_episode_audit(audit)["result_identity"]["sequence_no"])
        audit["result_identity"] = dict(audit["result_identity"], run_id="wrong-run")
        with self.assertRaises(result.HilAuditError):
            result.validate_episode_audit(audit)

    def test_infrastructure_failure_is_never_synthesized_as_zero_driving_score(self):
        outcome = result.paired_row("task-x", "arm-x", "INFRA_RETRY", None)
        self.assertIsNone(outcome["DS"])
        self.assertIsNone(outcome["RC"])
        self.assertEqual("INFRA_INVALID", outcome["status"])

    def test_rejects_bad_aoi_remote_bytes_artifact_and_future_result(self):
        identity = {"run_id": "r", "route_id": "r1", "scenario_id": "s1",
                    "arm_id": "a", "sequence_no": 1, "source_frame": 2}
        base = {"batch_order": ["ego", "rsu"], "trigger_identity": identity,
                "result_identity": identity, "artifact_sha256": {"model": "c" * 64},
                "aoi_frames": 0, "remote_input_tensor_bytes": 0}
        for change in ({"batch_order": ["rsu", "ego"]}, {"aoi_frames": -1},
                       {"remote_input_tensor_bytes": 4},
                       {"artifact_sha256": {"model": "not-a-hash"}}):
            with self.subTest(change=change), self.assertRaises(result.HilAuditError):
                result.validate_episode_audit(dict(base, **change))
        future = dict(identity, sequence_no=2)
        with self.assertRaisesRegex(result.HilAuditError, "newer"):
            result.validate_episode_audit(dict(base, result_identity=future))

    def test_result_classifier_and_audit_rejects_nonobjects_and_bad_identity(self):
        self.assertEqual("COMPLETED", result.classify_driving_result("complete"))
        self.assertEqual("DRIVING_TERMINAL", result.classify_driving_result("route-timeout"))
        self.assertEqual("INFRA_RETRY", result.classify_driving_result("ros-disconnected"))
        with self.assertRaises(result.HilAuditError):
            result.validate_episode_audit(None)
        identity = {"run_id": "r", "route_id": "r1", "scenario_id": "s1",
                    "arm_id": "a", "sequence_no": 1, "source_frame": 2}
        invalid = {"batch_order": ["ego", "rsu"], "trigger_identity": identity,
                   "result_identity": dict(identity, route_id="wrong"),
                   "artifact_sha256": {"a": "a" * 64}, "aoi_frames": 0,
                   "remote_input_tensor_bytes": 0}
        with self.assertRaisesRegex(result.HilAuditError, "route_id"):
            result.validate_episode_audit(invalid)

    def test_atomic_json_failure_cleans_up_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TypeError):
                result.atomic_json(Path(temp) / "bad.json", {"bad": object()})
            self.assertEqual([], list(Path(temp).glob("*.tmp")))

    def test_atomic_json_and_paired_csv_round_trip(self):
        row = result.paired_row("task", "arm", "COMPLETED", {"DS": 0.5, "RC": 0.8})
        with tempfile.TemporaryDirectory() as temp:
            json_path, csv_path = Path(temp) / "state.json", Path(temp) / "paired.csv"
            result.atomic_json(json_path, {"row": row})
            result.write_paired_csv(csv_path, [row])
            self.assertEqual(row, json.loads(json_path.read_text()) ["row"])
            self.assertIn("0.5", csv_path.read_text())
            self.assertEqual([], list(Path(temp).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
