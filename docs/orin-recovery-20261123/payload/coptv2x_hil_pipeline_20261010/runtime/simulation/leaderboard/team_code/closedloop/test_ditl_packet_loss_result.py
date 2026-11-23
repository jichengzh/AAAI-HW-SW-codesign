#!/usr/bin/env python
"""Tests for strict packet-loss stage-one result evidence and summaries."""

from __future__ import print_function

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.ditl_packet_loss_manifest import build_partition, build_stage1_manifest
from scripts.ditl_packet_loss_result import (
    classify_task_evidence,
    finalize_task_results,
    plan_retry_task_keys,
)
import scripts.ditl_packet_loss_result as packet_result


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_audit_row(task):
    return {
        "schema_version": "ditl.event-audit.v1",
        "run_id": "packet-loss-stage1-test",
        "route_id": "town05_short_r%d" % task["route"],
        "stream_id": "rsu-1",
        "consume_frame": 2,
        "consume_sim_time_ns": 200000000,
        "submitted_message_id": "message-1",
        "submitted_source_frame": 1,
        "trace_sample_id": "sample-1",
        "cycle_index": 0,
        "used_message_id": "message-1",
        "used_source_frame": 1,
        "source_sim_time_ns": 100000000,
        "ready_time_ns": 150000000,
        "base_delay_ns": 50000000,
        "effective_delay_ns": 50000000,
        "aoi_ns": 100000000,
        "zoh_hold_ns": 0,
        "is_zoh": False,
        "is_startup_ego_only": False,
        "is_expired": False,
        "arrival_deadline_miss": False,
        "fusion_deadline_miss": False,
        "drop_reason": None,
        "reorder_status": "none",
        "duplicate_status": "none",
        "policy_seed": task["replay_seed"],
        "policy_config_hash": "b" * 64,
        "rng_version": "ditl.blake2b-u53.v1",
        "queue_depth": 0,
        "accepted_count": 1,
        "dropped_count": 0,
        "expired_count": 0,
    }


class PacketLossResultTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cohort = Path(self.temporary.name).resolve()
        self.manifest = build_stage1_manifest(build_partition())

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _route_result(status="Completed"):
        return {"_checkpoint": {"global_record": {"status": status}}}

    def _checkpoint_tree(self, label="route"):
        checkpoint = self.cohort / "work" / label / "results.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(_json_bytes(self._route_result()))
        child = checkpoint.parent / "ego_vehicle_0" / checkpoint.name
        child.parent.mkdir(exist_ok=True)
        return checkpoint, child

    def test_ego_result_resolver_selects_only_direct_terminal_child(self):
        checkpoint, child = self._checkpoint_tree()
        child.write_bytes(_json_bytes(self._route_result("Failed")))

        selected = packet_result.resolve_ego_result(checkpoint)

        self.assertEqual(child, selected)
        self.assertNotEqual(checkpoint, selected)

    def test_ego_result_resolver_rejects_invalid_child_without_outer_fallback(self):
        fixtures = {
            "missing": None,
            "malformed": b"{not-json\n",
            "statusless": _json_bytes({"_checkpoint": {"global_record": {}}}),
            "nonterminal": _json_bytes(self._route_result("Running")),
        }
        for label, content in fixtures.items():
            with self.subTest(label=label):
                checkpoint, child = self._checkpoint_tree(label)
                if content is not None:
                    child.write_bytes(content)
                with self.assertRaises(ValueError):
                    packet_result.resolve_ego_result(checkpoint)
                if child.exists():
                    child.unlink()

        checkpoint, child = self._checkpoint_tree("symlink")
        target = self.cohort / "terminal-result.json"
        target.write_bytes(_json_bytes(self._route_result()))
        child.symlink_to(target)
        with self.assertRaises(ValueError):
            packet_result.resolve_ego_result(checkpoint)

    def test_ego_result_resolver_rejects_symlinked_ego_directory(self):
        checkpoint, child = self._checkpoint_tree("symlinked-ego-directory")
        child.parent.rmdir()
        external = self.cohort / "external-ego"
        external.mkdir()
        (external / checkpoint.name).write_bytes(
            _json_bytes(self._route_result()))
        child.parent.symlink_to(external, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "symlink"):
            packet_result.resolve_ego_result(checkpoint)

    def test_resolve_ego_result_cli_is_fail_closed_for_shell_command_substitution(self):
        checkpoint, child = self._checkpoint_tree()
        child.write_bytes(_json_bytes(self._route_result()))
        command = [sys.executable, "-m", "scripts.ditl_packet_loss_result",
                   "--resolve-ego-result", str(checkpoint)]

        valid = subprocess.run(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(0, valid.returncode, valid.stderr)
        self.assertEqual(str(child), valid.stdout.strip())

        child.write_bytes(_json_bytes({"_checkpoint": {"global_record": {}}}))
        invalid = subprocess.run(command, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
        self.assertNotEqual(0, invalid.returncode)
        self.assertEqual("", invalid.stdout)

    def _attempt(self, task, name, attempt=1, status="Completed", ds=80.0,
                 rc=90.0, returncode=0, omit=(), corrupt_hash=None,
                 collision=False, pedestrian_collision=False):
        root = self.cohort / "attempts" / name
        root.mkdir(parents=True)
        config_sha = "a" * 64
        command = ["eval-one", "--task-key", task["key"]]
        result = {
            "_checkpoint": {
                "global_record": {
                    "status": status,
                    "scores": {"score_composed": ds, "score_route": rc},
                    "infractions": {
                        "route_timeout": 1 if status == "Failed - Agent timed out" else 0,
                        "collisions_layout": ["x"] if collision else [],
                        "collisions_vehicle": [],
                        "collisions_pedestrian": (
                            ["pedestrian"] if pedestrian_collision else []),
                    },
                    "meta": {},
                },
                "records": [],
                "meta": {},
            }
        }
        audit = [_canonical_audit_row(task)]
        state = {
            "schema": "ditl.packet-loss-stage1-task-status.v1",
            "task_key": task["key"], "arm": task["arm"],
            "scenario": task["scenario"], "route": task["route"],
            "carla_seed": task["carla_seed"],
            "traffic_seed": task["traffic_seed"],
            "replay_seed": task["replay_seed"], "repeat": task["repeat"],
            "config_sha256": config_sha, "attempt": attempt,
            "worker": "worker6", "gpu": 6,
            "ports": [47000, 47001, 47002, 47005],
            "state": "completed" if returncode == 0 else "failed",
            "returncode": returncode, "command_argv": command,
        }
        identity = {
            "schema": "ditl.packet-loss-stage1-command-identity.v1",
            "task_key": task["key"], "attempt": attempt,
            "worker": "worker6", "gpu": 6,
            "ports": [47000, 47001, 47002, 47005],
            "command_argv": command,
        }
        values = {"result.json": result, "event_audit.jsonl": audit,
                  "status.json": state, "command_identity.json": identity}
        for filename, value in values.items():
            if filename in omit:
                continue
            path = root / filename
            if filename.endswith(".jsonl"):
                path.write_bytes(b"".join(_json_bytes(row) for row in value))
            else:
                path.write_bytes(_json_bytes(value))
        artifacts = {}
        for filename in values:
            if filename not in omit:
                digest = _sha(root / filename)
                if corrupt_hash == filename:
                    digest = "0" * 64
                artifacts[filename] = {"path": filename, "sha256": digest}
        evidence = {
            "schema": "ditl.packet-loss-stage1-task-evidence.v1",
            "cohort_path": str(self.cohort),
            "task_identity": copy.deepcopy(task),
            "config_sha256": config_sha,
            "policy_config_hash": "b" * 64,
            "attempt": attempt, "worker": "worker6", "gpu": 6,
            "ports": [47000, 47001, 47002, 47005],
            "command_argv": command, "artifacts": artifacts,
        }
        (root / "evidence.json").write_bytes(_json_bytes(evidence))
        return root

    def test_completed_and_terminal_driving_failure_are_countable(self):
        completed_task = self.manifest["tasks"][0]
        timeout_task = self.manifest["tasks"][1]
        collision_task = self.manifest["tasks"][2]
        completed = classify_task_evidence(
            completed_task, self._attempt(completed_task, "completed"))
        timeout = classify_task_evidence(
            timeout_task, self._attempt(
                timeout_task, "timeout", status="Failed - Agent timed out",
                ds=71.0, rc=51.0))
        collision = classify_task_evidence(
            collision_task, self._attempt(
                collision_task, "collision", status="Failed", collision=True))
        self.assertEqual("countable", completed["classification"])
        self.assertEqual("countable", timeout["classification"])
        self.assertEqual(0.0, timeout["DS"])
        self.assertEqual("countable", collision["classification"])
        first = {row["task_key"]: row for row in (completed, timeout, collision)}
        self.assertFalse(set(first).intersection(plan_retry_task_keys(
            self.manifest, first)))

    def test_canonical_audit_without_task_key_uses_attempt_identity_contract(self):
        task = self.manifest["tasks"][0]
        attempt = self._attempt(task, "canonical-audit")
        audit = json.loads((attempt / "event_audit.jsonl").read_text())
        self.assertNotIn("task_key", audit)
        row = classify_task_evidence(task, attempt)
        self.assertEqual("countable", row["classification"])
        self.assertEqual(1, row["event_metrics"]["accepted_count"])

    def test_attempt_identity_rejects_cross_task_audit_substitution(self):
        first = self.manifest["tasks"][0]
        second = next(task for task in self.manifest["tasks"]
                      if task["route"] == first["route"] and
                      task["scenario"] != first["scenario"])
        attempt = self._attempt(first, "cross-task")
        row = classify_task_evidence(second, attempt)
        self.assertEqual("infrastructure_failure", row["classification"])
        self.assertIn("task identity mismatch", row["failure_reason"])

    def test_infrastructure_failures_retry_once_then_zero(self):
        tasks = self.manifest["tasks"][:4]
        failures = (
            self._attempt(tasks[0], "crash", returncode=7),
            self._attempt(tasks[1], "missing", omit=("event_audit.jsonl",)),
            self._attempt(tasks[2], "hash", corrupt_hash="result.json"),
            self._attempt(tasks[3], "identity"),
        )
        identity = json.loads((failures[3] / "status.json").read_text())
        identity["gpu"] = 7
        (failures[3] / "status.json").write_bytes(_json_bytes(identity))
        evidence = json.loads((failures[3] / "evidence.json").read_text())
        evidence["artifacts"]["status.json"]["sha256"] = _sha(
            failures[3] / "status.json")
        (failures[3] / "evidence.json").write_bytes(_json_bytes(evidence))
        first = {
            task["key"]: classify_task_evidence(task, path)
            for task, path in zip(tasks, failures)
        }
        self.assertEqual(tuple(task["key"] for task in tasks),
                         plan_retry_task_keys(self.manifest, first))
        retry = {}
        for index, task in enumerate(tasks):
            path = (self.cohort / "attempts" / "attempt-2" / "missing"
                    if index == len(tasks) - 1 else
                    self._attempt(task, "retry-%d" % index, attempt=2,
                                  returncode=9))
            retry[task["key"]] = classify_task_evidence(task, path)
        finalized = finalize_task_results(self.manifest, first, retry)
        for task in tasks:
            row = finalized[task["key"]]
            self.assertEqual("zeroed_after_retry", row["final_status"])
            self.assertEqual(0.0, row["DS"])
            self.assertIsNone(row["RC"])
            self.assertIsNone(row["event_metrics"])

    def test_partial_batch_preserves_valid_routes(self):
        valid_task, failed_task = self.manifest["tasks"][:2]
        valid = classify_task_evidence(
            valid_task, self._attempt(valid_task, "valid"))
        failed = classify_task_evidence(
            failed_task, self._attempt(failed_task, "failed", returncode=6))
        first = {valid_task["key"]: valid, failed_task["key"]: failed}
        self.assertEqual((failed_task["key"],),
                         plan_retry_task_keys(self.manifest, first))
        retry = {failed_task["key"]: classify_task_evidence(
            failed_task, self._attempt(failed_task, "retry", attempt=2))}
        finalized = finalize_task_results(self.manifest, first, retry)
        self.assertEqual("first_attempt", finalized[valid_task["key"]]["source"])
        self.assertEqual("retry_attempt", finalized[failed_task["key"]]["source"])

    def test_policy_hash_is_distinct_and_all_collision_types_are_counted(self):
        task = self.manifest["tasks"][0]
        row = classify_task_evidence(
            task, self._attempt(task, "pedestrian", pedestrian_collision=True))
        self.assertEqual("countable", row["classification"])
        self.assertTrue(row["collisions"])
        self.assertEqual("a" * 64, row["provenance"]["config_sha256"])
        self.assertEqual("b" * 64, row["provenance"]["policy_config_hash"])

    def test_reuse_provenance_is_preserved_in_classification(self):
        task = self.manifest["tasks"][0]
        attempt = self._attempt(task, "reused")
        evidence_path = attempt / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["reuse_provenance"] = {
            "schema": "ditl.packet-loss-stage1-reuse-provenance.v1",
            "source_kind": "canonical",
            "source_cohort": "/var/tmp/codriving-ditl/packet-loss-stage1-old-r4",
            "source_attempt_dir": "/var/tmp/codriving-ditl/packet-loss-stage1-old-r4/attempts/attempt-1/task",
            "source_hashes": {"evidence.json": "c" * 64},
        }
        evidence_path.write_bytes(_json_bytes(evidence))

        row = classify_task_evidence(task, attempt)

        self.assertEqual("countable", row["classification"])
        self.assertEqual(evidence["reuse_provenance"],
                         row["provenance"]["reuse_provenance"])

    def test_malformed_reuse_provenance_is_infrastructure_failure(self):
        task = self.manifest["tasks"][0]
        attempt = self._attempt(task, "bad-reuse")
        evidence_path = attempt / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["reuse_provenance"] = {
            "schema": "ditl.packet-loss-stage1-reuse-provenance.v1",
            "source_kind": "unknown",
            "source_cohort": "/var/tmp/codriving-ditl/packet-loss-stage1-old-r4",
            "source_attempt_dir": "/var/tmp/codriving-ditl/packet-loss-stage1-old-r4/attempts/attempt-1/task",
            "source_hashes": {"evidence.json": "z" * 64},
        }
        evidence_path.write_bytes(_json_bytes(evidence))

        row = classify_task_evidence(task, attempt)

        self.assertEqual("infrastructure_failure", row["classification"])
        self.assertIn("reuse provenance", row["failure_reason"])

    def test_finalize_rejects_illegal_attempt_and_cross_cohort_evidence(self):
        valid_task, failed_task = self.manifest["tasks"][:2]
        valid = classify_task_evidence(
            valid_task, self._attempt(valid_task, "guard-valid"))
        failed = classify_task_evidence(
            failed_task, self._attempt(failed_task, "guard-failed", returncode=8))
        first = {valid_task["key"]: valid, failed_task["key"]: failed}
        retry = classify_task_evidence(
            failed_task, self._attempt(failed_task, "guard-retry", attempt=2))
        illegal_third = copy.deepcopy(retry)
        illegal_third["attempt"] = 3
        with self.assertRaisesRegex(ValueError, "retry evidence"):
            finalize_task_results(
                self.manifest, first, {failed_task["key"]: illegal_third})
        with self.assertRaisesRegex(ValueError, "exactly match"):
            finalize_task_results(self.manifest, first, {
                valid_task["key"]: valid, failed_task["key"]: retry})
        other = self.cohort / "other"
        (other / "attempt").mkdir(parents=True)
        external_first = copy.deepcopy(first)
        external_first[failed_task["key"]]["cohort_path"] = str(other)
        external_first[failed_task["key"]]["evidence_dir"] = str(other / "attempt")
        with self.assertRaisesRegex(ValueError, "single cohort"):
            finalize_task_results(self.manifest, external_first, {
                failed_task["key"]: retry})

    def test_finalize_rejects_first_attempt_directory_outside_cohort(self):
        task = self.manifest["tasks"][0]
        first = classify_task_evidence(
            task, self._attempt(task, "first-outside-guard"))
        outside = self.cohort.parent / (self.cohort.name + "-foreign")
        outside.mkdir()
        first["evidence_dir"] = str(outside)
        with self.assertRaisesRegex(ValueError, "first-pass evidence.*external"):
            finalize_task_results(self.manifest, {task["key"]: first}, {})

    def _task(self, arm, cell_index):
        rows = [task for task in self.manifest["tasks"] if task["arm"] == arm]
        return rows[cell_index]

    def _finalized(self, task, ds=80.0, rc=90.0, completed=True,
                   collision=False, accepted=9, dropped=1, aoi=(),
                   severe=None, zoh=0.0, status="countable", provenance=None):
        samples = list(aoi or (100.0,))
        if severe is None:
            severe = float(sum(value >= 600.0 for value in samples)) / len(samples)
        return {
            "task_key": task["key"], "arm": task["arm"],
            "scenario": task["scenario"], "route": task["route"],
            "final_status": status, "source": "first_attempt",
            "DS": float(ds), "RC": None if rc is None else float(rc),
            "collisions": collision, "route_completed": completed,
            "event_metrics": None if status != "countable" else {
                "accepted_count": accepted, "dropped_count": dropped,
                "actual_drop_rate": float(dropped) / (accepted + dropped),
                "aoi_samples_ms": samples,
                "mean_aoi_ms": sum(samples) / len(samples),
                "severe_stale_rate": severe, "zoh_rate": zoh,
                "cold_start_count": 2, "observation_count": len(samples),
                "stream_count": 2,
            },
            "provenance": provenance or {},
        }

    def test_route_balanced_mechanism_metrics_use_hand_calculated_fixture(self):
        first, second = self._task("drop_010", 0), self._task("drop_010", 1)
        finalized = {
            first["key"]: self._finalized(
                first, accepted=1, dropped=1, aoi=(100.0, 700.0), zoh=0.5),
            second["key"]: self._finalized(
                second, accepted=9, dropped=1, aoi=(200.0,) * 8, zoh=0.25),
        }
        summary = packet_result.summarize_stage1(
            self.manifest, finalized, bootstrap_repetitions=100)
        mechanism = summary["arms"]["drop_010"]["mechanism"]
        self.assertAlmostEqual(0.30, mechanism["actual_drop_rate_macro"]["value"])
        self.assertAlmostEqual(2.0 / 12.0,
                               mechanism["actual_drop_rate_micro"]["value"])
        self.assertAlmostEqual(300.0, mechanism["mean_aoi_ms"]["value"])
        self.assertAlmostEqual(0.25, mechanism["severe_stale_rate"]["value"])
        self.assertAlmostEqual(0.375, mechanism["zoh_rate"]["value"])
        self.assertEqual(2, mechanism["mean_aoi_ms"]["N_valid"])

    def test_equal_route_mixture_quantiles_do_not_pool_events(self):
        first, second = self._task("drop_020", 0), self._task("drop_020", 1)
        finalized = {
            first["key"]: self._finalized(first, aoi=(0.0,) * 100),
            second["key"]: self._finalized(second, aoi=(1000.0,)),
        }
        summary = packet_result.summarize_stage1(
            self.manifest, finalized, bootstrap_repetitions=100)
        mechanism = summary["arms"]["drop_020"]["mechanism"]
        self.assertEqual(1000.0, mechanism["aoi_p95_ms"]["value"])
        self.assertEqual(1000.0, mechanism["aoi_p99_ms"]["value"])

    def _paired_fixture(self):
        finalized = {}
        clean_ds = (80.0, 70.0, 60.0)
        arm_ds = {
            "drop_010": (78.0, 68.0, 58.0),
            "drop_020": (75.0, 65.0, 55.0),
            "drop_030": (70.0, 60.0, 50.0),
        }
        for index, value in enumerate(clean_ds):
            task = self._task("clean", index)
            finalized[task["key"]] = self._finalized(task, ds=value)
        for arm, values in arm_ds.items():
            for index, value in enumerate(values):
                task = self._task(arm, index)
                finalized[task["key"]] = self._finalized(task, ds=value)
        return finalized

    def test_paired_ds_and_bootstrap_are_deterministic(self):
        finalized = self._paired_fixture()
        first = packet_result.summarize_stage1(self.manifest, finalized)
        second = packet_result.summarize_stage1(self.manifest, finalized)
        self.assertEqual(first["paired"], second["paired"])
        self.assertEqual(-2.0, first["paired"]["drop_010"]["primary"]["mean_delta_DS"])
        self.assertEqual(-5.0, first["paired"]["drop_020"]["primary"]["mean_delta_DS"])
        self.assertEqual(-10.0, first["paired"]["drop_030"]["primary"]["mean_delta_DS"])
        self.assertEqual(2000, first["bootstrap"]["seed"])
        self.assertEqual(10000, first["bootstrap"]["repetitions"])

    def test_paired_ds_uses_full_seed_and_repeat_unit_identity(self):
        def task(arm, seed, repeat, key):
            return {
                "key": key, "arm": arm, "scenario": "_1", "route": 2,
                "carla_seed": seed, "traffic_seed": seed,
                "replay_seed": seed, "repeat": repeat,
            }

        clean_a = task("clean", 2000, 0, "clean-a")
        clean_b = task("clean", 2001, 1, "clean-b")
        drop_a = task("drop_010", 2000, 0, "drop-a")
        drop_b = task("drop_010", 2001, 1, "drop-b")
        manifest = {"tasks": [clean_a, clean_b, drop_a, drop_b]}
        finalized = {
            "clean-a": {"DS": 100.0, "final_status": "countable"},
            "clean-b": {"DS": 200.0, "final_status": "countable"},
            "drop-a": {"DS": 90.0, "final_status": "countable"},
            "drop-b": {"DS": 150.0, "final_status": "countable"},
        }
        paired = packet_result._paired_summaries(
            manifest, finalized, seed=2000, repetitions=100)
        primary = paired["drop_010"]["primary"]
        self.assertEqual(2, primary["N_valid"])
        self.assertEqual([-10.0, -50.0], primary["unit_delta_DS"])
        self.assertEqual(-30.0, primary["mean_delta_DS"])

    def test_driving_summary_reports_arm_and_paired_views(self):
        finalized = self._paired_fixture()
        task = self._task("drop_010", 0)
        finalized[task["key"]]["collisions"] = True
        finalized[task["key"]]["route_completed"] = False
        summary = packet_result.summarize_stage1(
            self.manifest, finalized, bootstrap_repetitions=100)
        driving = summary["arms"]["drop_010"]["driving"]
        self.assertEqual(68.0, driving["DS_mean"]["value"])
        self.assertEqual(68.0, driving["DS_median"]["value"])
        self.assertEqual(90.0, driving["RC_mean"]["value"])
        self.assertAlmostEqual(2.0 / 3.0, driving["route_completion_rate"]["value"])
        self.assertAlmostEqual(1.0 / 3.0, driving["collision_rate"]["value"])
        for metric in driving.values():
            self.assertEqual(3, metric["N_valid"])
        self.assertEqual(3, summary["paired"]["drop_010"]["primary"]["N_valid"])

    def test_zeroed_results_report_primary_and_sensitivity_views(self):
        finalized = self._paired_fixture()
        clean = self._task("clean", 1)
        drop = self._task("drop_010", 2)
        finalized[clean["key"]] = self._finalized(
            clean, ds=0.0, rc=None, status="zeroed_after_retry")
        finalized[drop["key"]] = self._finalized(
            drop, ds=0.0, rc=None, status="zeroed_after_retry")
        summary = packet_result.summarize_stage1(
            self.manifest, finalized, bootstrap_repetitions=100)
        views = summary["paired"]["drop_010"]
        self.assertEqual(3, views["primary"]["N_valid"])
        self.assertEqual(1, views["sensitivity"]["N_valid"])
        self.assertEqual(-2.0, views["sensitivity"]["mean_delta_DS"])
        driving = summary["arms"]["drop_010"]["driving"]
        self.assertEqual(3, driving["DS_mean"]["N_valid"])
        self.assertEqual(2, driving["RC_mean"]["N_valid"])

    def test_shared_gpu_performance_fields_are_excluded(self):
        task = self._task("clean", 0)
        finalized = {task["key"]: self._finalized(task, provenance={
            "execution_latency_ms": 33.0, "throughput_fps": 12.0,
            "energy_joules": 8.0, "gpu_utilization_percent": 91,
            "worker": "worker6",
        })}
        summary = packet_result.summarize_stage1(
            self.manifest, finalized, bootstrap_repetitions=100)
        scientific = {
            "route_metrics": summary["route_metrics"],
            "arms": summary["arms"], "paired": summary["paired"],
        }
        encoded = json.dumps(scientific, sort_keys=True)
        for field in ("execution_latency_ms", "throughput_fps",
                      "energy_joules", "gpu_utilization_percent"):
            self.assertNotIn(field, encoded)
            self.assertIn(field, summary["provenance"]["task_provenance"][task["key"]])

    def test_generate_artifacts_is_complete_atomic_and_byte_stable(self):
        (self.cohort / "manifest.json").write_bytes(_json_bytes(self.manifest))
        for task in self.manifest["tasks"]:
            safe = task["key"].replace(":", "__").replace("/", "_")
            self._attempt(task, "attempt-1/" + safe)
        first = packet_result.generate_artifacts(self.cohort)
        names = (
            "stage1_integrity.json", "stage1_task_results.json",
            "stage1_route_metrics.csv", "stage1_arm_summary.csv",
            "stage1_summary.json",
        )
        before = {name: (self.cohort / name).read_bytes() for name in names}
        second = packet_result.generate_artifacts(self.cohort)
        after = {name: (self.cohort / name).read_bytes() for name in names}
        self.assertEqual(before, after)
        self.assertEqual(first, second)
        self.assertFalse(list(self.cohort.glob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
