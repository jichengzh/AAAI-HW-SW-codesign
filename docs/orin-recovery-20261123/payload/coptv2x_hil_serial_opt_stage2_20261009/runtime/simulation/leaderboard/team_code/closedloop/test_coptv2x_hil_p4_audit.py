"""Independent P4 bundle checks: complete frozen inputs, five arms, no performance claim."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import coptv2x_hil_manifest as manifest_module
import coptv2x_hil_runner as runner
import coptv2x_hil_p4_audit as audit


class P4AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cohort = self.root / "cohort"
        self.cohort.mkdir()
        self.cell = "Town05:_1:r2"
        self.host = self.root / "host.json"
        self.remote = self.root / "orin.json"
        frames = [{"source_frame": frame, "sample_id": "carla-frame-%d" % frame,
                   "spatial_features_sha256": str(frame) * 64,
                   "spatial_features_path": "/frozen/frame%d.npy" % frame,
                   "shape": [2, 64, 256, 512], "dtype": "float32"} for frame in (1, 2)]
        payload = manifest_module.select_pilot(manifest_module.build_manifest(), [self.cell])
        self.save(self.host, {"schema":"coptv2x.route-input-host-index.v1", "status":"HOST_INDEX_VALID",
                              "route_scenarios_sha256":payload["route_scenarios_sha256"],
                              "route_scenarios":{self.cell:frames}})
        self.host_sha = runner.file_sha256(self.host)
        self.save(self.remote, {"schema_version":"coptv2x.local-input-manifest.v1",
                                "route_id":"Town05-route-2", "scenario_id":"scenario1",
                                "batch_order":["ego", "rsu"], "samples":[
                                    {"source_frame":frame["source_frame"], "sample_id":frame["sample_id"],
                                     "inputs":{"spatial_features":{"path":"frame.npy",
                                               "shape":frame["shape"], "dtype":frame["dtype"],
                                               "sha256":frame["spatial_features_sha256"]}}}
                                    for frame in frames]})
        payload["host_input_index"] = {"path":str(self.host), "sha256":self.host_sha,
                                       "route_scenarios_sha256":payload["route_scenarios_sha256"],
                                       "verified_feature_files":2}
        self.save(self.cohort / "manifest.json", payload)
        results = {}
        attempts = {}
        for task in payload["tasks"]:
            run_dir = self.cohort / "runs" / task["task_id"] / "attempt-01"
            identity = {"run_id":"cohort", "route_id":"Town05-route-2", "scenario_id":"scenario1",
                        "arm_id":task["arm_id"], "sequence_no":1, "source_frame":1,
                        "schema_version":"coptv2x.hil-trigger.v1", "sample_id":"carla-frame-1",
                        "source_sim_time_ns":50000000}
            hil = {"batch_order":["ego", "rsu"], "trigger_identity":identity,
                   "result_identity":identity, "artifact_sha256":task["artifact_sha256"],
                   "aoi_frames":0, "remote_input_tensor_bytes":0}
            row = {"batch_order":hil["batch_order"], "trigger_identity":identity,
                   "selected_result_trigger_identity":identity, "artifact_sha256":task["artifact_sha256"],
                   "rsu_aoi_frames":0,"remote_input_tensor_bytes":0,"selected_result_sequence_no":1}
            fusion = run_dir / "evidence/fusion_audit.jsonl"
            fusion.parent.mkdir(parents=True)
            fusion.write_text(json.dumps(row) + "\n")
            carla = {"status":"Completed", "scores":{"score_composed":60,"score_route":100}}
            checkpoint = run_dir / "carla/results.json"
            self.save(checkpoint, {"_checkpoint":{"global_record":carla,"records":[carla]}})
            result = {"classification":"COMPLETED", "status":"completed", "return_code":0,
                      "run_dir":str(run_dir), "arm_id":task["arm_id"], "attempt":1,
                      "carla_result_path":str(checkpoint), "fusion_audit_path":str(fusion),
                      "carla_result":carla, "hil_audit":hil, "artifact_sha256":task["artifact_sha256"]}
            results[task["task_id"]] = result
            attempts[task["task_id"]] = [dict(result)]
        self.state_path = self.cohort / "state.json"
        self.save(self.state_path, {"schema":"coptv2x.hil-run-state.v1", "results":results,
                                    "attempts":attempts, "task_ids":[task["task_id"] for task in payload["tasks"]],
                                    "manifest_task_ids_sha256":runner._task_ids_digest([task["task_id"] for task in payload["tasks"]]),
                                    "manifest_sha256":runner._manifest_sha256(payload),
                                    "host_input_index":payload["host_input_index"]})

    @staticmethod
    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True))

    def summary(self):
        return audit.audit_p4_bundle(self.cohort, self.host, self.host_sha, self.remote,
                                     expected_input_frames=2)

    def edit(self, path, change):
        payload = json.loads(path.read_text())
        change(payload)
        self.save(path, payload)

    def test_valid_bundle_is_explicitly_smoke_only_and_keeps_live_files_unchanged(self):
        before = {path:str(hashlib.sha256(path.read_bytes()).hexdigest())
                  for path in self.cohort.rglob("*") if path.is_file()}
        summary = self.summary()
        self.assertEqual("P4_PASS", summary["status"])
        self.assertTrue(summary["integration_smoke_only"])
        self.assertFalse(summary["performance_conclusion_allowed"])
        self.assertEqual(5, len(summary["episodes"]))
        self.assertEqual(2, summary["input_authority"]["verified_frame_identities"])
        self.assertEqual(before, {path:str(hashlib.sha256(path.read_bytes()).hexdigest())
                                  for path in self.cohort.rglob("*") if path.is_file()})

    def test_nonempty_but_truncated_authority_does_not_pass(self):
        with self.assertRaisesRegex(ValueError, "frame count"):
            audit.audit_p4_bundle(self.cohort, self.host, self.host_sha, self.remote,
                                   expected_input_frames=3)

    def test_host_index_must_match_pinned_historical_bytes(self):
        self.edit(self.host, lambda d:d.update(purpose="different-authority"))
        with self.assertRaisesRegex(ValueError, "host input index SHA"):
            self.summary()

    def test_orin_same_count_but_changed_frame_hash_is_rejected(self):
        self.edit(self.remote, lambda d:d["samples"][0]["inputs"]["spatial_features"].update(sha256="f"*64))
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            self.summary()

    def test_duplicate_remote_identity_cannot_mask_a_missing_frame(self):
        self.edit(self.remote, lambda d:d["samples"].__setitem__(1, d["samples"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.summary()

    def test_infrastructure_terminal_is_not_p4_completion(self):
        self.edit(self.state_path, lambda d:next(iter(d["results"].values())).update(classification="INFRA_INVALID"))
        with self.assertRaisesRegex(ValueError, "driving terminal"):
            self.summary()

    def test_state_cannot_belong_to_another_manifest(self):
        self.edit(self.state_path, lambda d:d.update(manifest_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.summary()

    def test_valid_stored_audit_without_real_fusion_evidence_is_rejected(self):
        record = next(iter(json.loads(self.state_path.read_text())["results"].values()))
        Path(record["fusion_audit_path"]).write_text("")
        with self.assertRaisesRegex(ValueError, "selected fusion"):
            self.summary()

    def test_selected_fusion_cannot_use_another_route_identity(self):
        record = next(iter(json.loads(self.state_path.read_text())["results"].values()))
        path = Path(record["fusion_audit_path"])
        row = json.loads(path.read_text())
        row["trigger_identity"]["route_id"] = "Town05-route-999"
        row["selected_result_trigger_identity"]["route_id"] = "Town05-route-999"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "task identity"):
            self.summary()

    def test_raw_agent_crash_cannot_be_hidden_by_stored_completed_label(self):
        record = next(iter(json.loads(self.state_path.read_text())["results"].values()))
        self.save(Path(record["carla_result_path"]), {"_checkpoint":{"global_record":{"status":"Failed"},
                  "records":[{"status":"Failed - Agent crashed", "infractions":{}}]}})
        with self.assertRaisesRegex(ValueError, "CARLA"):
            self.summary()

    def test_real_route_timeout_with_valid_fusion_is_an_allowed_smoke_terminal(self):
        state = json.loads(self.state_path.read_text())
        record = next(iter(state["results"].values()))
        carla = {"status":"Failed", "scores":{"score_composed":30,"score_route":50}}
        self.save(Path(record["carla_result_path"]), {"_checkpoint":{"global_record":carla,
                  "records":[{"status":"Failed", "infractions":{"route_timeout":["timeout"]}}]}})
        record.update(classification="DRIVING_TERMINAL", status="route_timeout", carla_result=carla)
        self.save(self.state_path, state)
        summary = self.summary()
        self.assertEqual("P4_PASS", summary["status"])
        episode = next(item for item in summary["episodes"] if item["arm_id"] == record["arm_id"])
        self.assertEqual("DRIVING_TERMINAL", episode["classification"])
        self.assertFalse(episode["performance_conclusion_allowed"])

    def test_evidence_hashes_the_per_ego_checkpoint_that_supplied_the_result(self):
        record = next(iter(json.loads(self.state_path.read_text())["results"].values()))
        primary = Path(record["carla_result_path"])
        actual = primary.parent / "ego_vehicle_0" / primary.name
        self.save(actual, json.loads(primary.read_text()))
        self.save(primary, {"_checkpoint":{"global_record":{},"records":[]}})
        summary = self.summary()
        episode = next(item for item in summary["episodes"] if item["arm_id"] == record["arm_id"])
        self.assertEqual(str(actual), episode["carla_result_path"])
        self.assertEqual(runner.file_sha256(actual), episode["carla_result_sha256"])

    def test_malformed_primary_records_follow_the_runner_fallback_rule(self):
        state = json.loads(self.state_path.read_text())
        record = next(iter(state["results"].values()))
        primary = Path(record["carla_result_path"])
        actual = primary.parent / "ego_vehicle_0" / primary.name
        carla = {"status":"Failed", "scores":{"score_composed":30,"score_route":50}}
        self.save(primary, {"_checkpoint":{"global_record":carla,"records":[None]}})
        self.save(actual, {"_checkpoint":{"global_record":carla,
                  "records":[{"status":"Failed", "infractions":{"route_timeout":["timeout"]}}]}})
        record.update(classification="DRIVING_TERMINAL", status="route_timeout", carla_result=carla)
        self.save(self.state_path, state)
        summary = self.summary()
        episode = next(item for item in summary["episodes"] if item["arm_id"] == record["arm_id"])
        self.assertEqual(str(actual), episode["carla_result_path"])

    def test_stop_hook_failure_cannot_be_hidden_by_completed_label(self):
        self.edit(self.state_path, lambda d:next(iter(d["results"].values())).update(episode_stop={"return_code":7}))
        with self.assertRaisesRegex(ValueError, "hook failed"):
            self.summary()

    def test_selected_result_must_belong_to_complete_frozen_input_identity_set(self):
        record = next(iter(json.loads(self.state_path.read_text())["results"].values()))
        path = Path(record["fusion_audit_path"])
        row = json.loads(path.read_text())
        for key in ("trigger_identity", "selected_result_trigger_identity"):
            row[key].update(source_frame=999, sample_id="carla-frame-999")
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "outside exact frozen"):
            self.summary()

    def test_cli_failure_writes_noncomplete_scope_instead_of_stale_pass(self):
        output = self.root / "summary.json"
        self.save(output, {"status":"P4_PASS"})
        code = audit.main(["--cohort-dir",str(self.cohort), "--host-input-index",str(self.host),
                           "--expected-input-index-sha256", "0"*64,
                           "--orin-input-manifest",str(self.remote), "--expected-input-frames","2",
                           "--output",str(output)])
        self.assertEqual(2, code)
        summary = json.loads(output.read_text())
        self.assertEqual("P4_NOT_COMPLETE", summary["status"])
        self.assertFalse(summary["performance_conclusion_allowed"])


if __name__ == "__main__":
    unittest.main()
