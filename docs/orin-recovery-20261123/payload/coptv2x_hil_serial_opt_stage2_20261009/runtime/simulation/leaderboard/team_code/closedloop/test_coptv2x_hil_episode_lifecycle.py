"""Real child-process checks for isolated, resumable episode service hooks."""

import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import coptv2x_hil_runner as runner


class EpisodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = self.root / "events.jsonl"
        self.task = {"task_id": "cell-original", "arm_id": "original_default",
                     "route_xml": "README.md", "artifact_sha256": {"engine": "a" * 64}}
        self.manifest = {"tasks": [self.task]}
        self.args = SimpleNamespace(run=True, host="127.0.0.1", port=56800,
                                    ros_domain_id=71, episode_timeout=10,
                                    max_infra_attempts=1, local_input_manifest=None,
                                    episode_hook_timeout=2,
                                    episode_start_command=self.hook("start"),
                                    episode_stop_command=self.hook("stop"))
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text('''import json, os
from pathlib import Path
events = Path(os.environ["EVENTS"])
with events.open("a") as stream:
    stream.write(json.dumps({"event":"evaluate", "attempt":os.environ.get("COPT_HIL_ATTEMPT")}) + "\\n")
run_dir = Path(os.environ["SAVE_PATH"]).parent
checkpoint = run_dir / "carla/results.json"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text(json.dumps({"_checkpoint":{"global_record":{"status":"Completed", "DS":60,"RC":100}}}))
identity = {"run_id":"cohort","route_id":"Town05-route-2","scenario_id":"_1","arm_id":"original_default","sequence_no":1,"source_frame":1}
row = {"batch_order":["ego","rsu"], "trigger_identity":identity, "selected_result_trigger_identity":identity,
       "selected_result_sequence_no":1, "artifact_sha256":{"engine":"a"*64}, "rsu_aoi_frames":0,"remote_input_tensor_bytes":0}
audit = run_dir / "evidence/fusion_audit.jsonl"
audit.parent.mkdir(parents=True, exist_ok=True)
audit.write_text(json.dumps(row) + "\\n")
''', encoding="utf-8")
        self.patches = [
            mock.patch.object(runner, "validate_local_input_manifest", return_value={"sha256":"b" * 64}),
            mock.patch.object(runner, "validate_carla_environment", return_value={"reachable":True}),
            mock.patch.object(runner, "prepare_task_config", side_effect=lambda task, run_dir, *a, **kw: Path(run_dir) / "agent.yaml"),
            mock.patch.object(runner, "build_episode_command", side_effect=lambda *a, **kw: [sys.executable, str(self.evaluator), "--agent-config", "ignored"]),
            mock.patch.object(runner, "runner_environment", side_effect=self.environment),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def environment(self, run_dir, *args, **kwargs):
        return dict(os.environ, EVENTS=str(self.events),
                    RESULT_ROOT=str(Path(run_dir) / "results"),
                    SAVE_PATH=str(Path(run_dir) / "images"))

    def hook(self, event, exit_code=0, first_attempt_failure=False, delay=0):
        path = self.root / (event + ".py")
        path.write_text('''import json, os, sys, time
from pathlib import Path
with Path(os.environ["EVENTS"]).open("a") as stream:
    stream.write(json.dumps(dict(event=%r, attempt=os.environ.get("COPT_HIL_ATTEMPT"),
                                run_dir=os.environ.get("COPT_HIL_RUN_DIR"),
                                arm=os.environ.get("COPT_HIL_ARM_ID"),
                                task=os.environ.get("COPT_HIL_TASK_ID"),
                                cohort=os.environ.get("COPT_HIL_COHORT_ID"))) + "\\n")
time.sleep(%r)
print("hook-output")
sys.exit(9 if %r and os.environ.get("COPT_HIL_ATTEMPT") == "1" else %r)
''' % (event, delay, first_attempt_failure, exit_code), encoding="utf-8")
        return " ".join(shlex.quote(value) for value in [sys.executable, str(path)])

    def run_once(self):
        return runner.run_cohort(self.manifest, self.root / "cohort", self.args)

    def recorded_events(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def state(self):
        return json.loads((self.root / "cohort/state.json").read_text())

    def test_hooks_surround_evaluator_and_receive_attempt_identity(self):
        self.run_once()
        events = self.recorded_events()
        self.assertEqual(["start", "evaluate", "stop"], [event["event"] for event in events])
        for event in (events[0], events[2]):
            self.assertEqual("1", event["attempt"])
            self.assertEqual("original_default", event["arm"])
            self.assertEqual("cell-original", event["task"])
            self.assertEqual("cohort", event["cohort"])
            self.assertTrue(event["run_dir"].endswith("attempt-01"))
        record = self.state()["results"]["cell-original"]
        self.assertEqual("COMPLETED", record["classification"])
        self.assertEqual(0, record["episode_start"]["return_code"])
        self.assertIn("hook-output", Path(record["episode_stop"]["log_path"]).read_text())

    def test_failed_start_skips_evaluator_but_still_stops_services(self):
        self.args.episode_start_command = self.hook("start", exit_code=8)
        self.run_once()
        self.assertEqual(["start", "stop"], [event["event"] for event in self.recorded_events()])
        record = self.state()["results"]["cell-original"]
        self.assertEqual("INFRA_INVALID", record["classification"])
        self.assertEqual("episode_start_failed", record["status"])

    def test_failed_stop_invalidates_an_otherwise_valid_episode(self):
        self.args.episode_stop_command = self.hook("stop", exit_code=7)
        self.run_once()
        record = self.state()["results"]["cell-original"]
        self.assertEqual("INFRA_INVALID", record["classification"])
        self.assertEqual("episode_stop_failed", record["status"])

    def test_retry_has_a_fresh_attempt_and_resume_never_restarts_services(self):
        self.args.max_infra_attempts = 2
        self.args.episode_start_command = self.hook("start", first_attempt_failure=True)
        self.run_once()
        events = self.recorded_events()
        self.assertEqual(["start", "stop", "start", "evaluate", "stop"], [event["event"] for event in events])
        self.assertEqual(["1", "1", "2", "2", "2"], [event["attempt"] for event in events])
        self.assertEqual("COMPLETED", self.state()["results"]["cell-original"]["classification"])
        before = self.events.read_bytes()
        report = self.run_once()
        self.assertEqual(0, report["pending_tasks"])
        self.assertEqual(before, self.events.read_bytes())

    def test_start_timeout_still_runs_cleanup_and_preserves_timeout_evidence(self):
        self.args.episode_hook_timeout = 1
        self.args.episode_start_command = self.hook("start", delay=3)
        self.run_once()
        record = self.state()["results"]["cell-original"]
        self.assertEqual("INFRA_INVALID", record["classification"])
        self.assertEqual(124, record["episode_start"]["return_code"])
        self.assertEqual(["start", "stop"], [event["event"] for event in self.recorded_events()])

    def test_missing_hook_executable_is_recorded_and_cleanup_still_runs(self):
        self.args.episode_start_command = str(self.root / "missing-hook")
        self.run_once()
        record = self.state()["results"]["cell-original"]
        self.assertEqual("INFRA_INVALID", record["classification"])
        self.assertEqual(127, record["episode_start"]["return_code"])
        self.assertIn("No such file", Path(record["episode_start"]["log_path"]).read_text())
        self.assertEqual(["stop"], [event["event"] for event in self.recorded_events()])

    def test_unmatched_hook_quote_is_recorded_instead_of_abandoning_cleanup(self):
        self.args.episode_start_command = "'unterminated"
        self.run_once()
        record = self.state()["results"]["cell-original"]
        self.assertEqual("episode_start_failed", record["status"])
        self.assertEqual(127, record["episode_start"]["return_code"])
        self.assertEqual(["stop"], [event["event"] for event in self.recorded_events()])

    def test_disabled_hooks_preserve_existing_episode_execution(self):
        self.args.episode_start_command = None
        self.args.episode_stop_command = None
        self.run_once()
        self.assertEqual(["evaluate"], [event["event"] for event in self.recorded_events()])
        self.assertEqual("COMPLETED", self.state()["results"]["cell-original"]["classification"])

    def test_driving_terminal_without_fusion_evidence_is_not_accepted(self):
        self.evaluator.write_text('''import json, os
from pathlib import Path
checkpoint = Path(os.environ["SAVE_PATH"]).parent / "carla/results.json"
checkpoint.write_text(json.dumps({"_checkpoint":{"global_record":{"status":"Failed"},"records":[{"status":"Failed", "infractions":{"route_timeout":["timeout"]}}]}}))
''', encoding="utf-8")
        self.run_once()
        record = self.state()["results"]["cell-original"]
        self.assertEqual("INFRA_INVALID", record["classification"])
        self.assertEqual("result_corrupt", record["status"])


if __name__ == "__main__":
    unittest.main()
