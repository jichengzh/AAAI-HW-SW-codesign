#!/usr/bin/env python
"""Tests for validated packet-loss evidence reuse."""

from __future__ import print_function

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.ditl_packet_loss_manifest import build_partition, build_stage1_manifest
from scripts.ditl_packet_loss_result import classify_task_evidence
from scripts.ditl_packet_loss_reuse import (build_reuse_plan,
    inspect_reuse_candidate, materialize_reuse_plan, validate_reuse_source)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(key):
    return key.replace(":", "__").replace("/", "_")


class PacketLossReuseTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.manifest = build_stage1_manifest(build_partition())
        self.task = self.manifest["tasks"][0]
        self.config = b"simulation:\n  skip_frames: 2\n"
        self.config_sha = hashlib.sha256(self.config).hexdigest()
        self.expected = {self.task["key"]: self.config_sha}

    def tearDown(self):
        self.temporary.cleanup()

    def _cohort(self, stamp, suffix):
        cohort = self.sources / ("packet-loss-stage1-%s-seed2000-%s" %
                                 (stamp, suffix))
        cohort.mkdir()
        (cohort / "manifest.json").write_bytes(_bytes(self.manifest))
        return cohort

    def _attempt(self, cohort, legacy=False, ds=80.0):
        task = self.task
        attempt = cohort / "attempts" / "attempt-1" / _safe(task["key"])
        attempt.mkdir(parents=True)
        config_path = cohort / "configs" / (_safe(task["key"]) + ".yaml")
        config_path.parent.mkdir()
        config_path.write_bytes(self.config)
        provenance_path = cohort / "provenance.json"
        provenance_path.write_bytes(_bytes({
            "config_sha256": {task["key"]: self.config_sha},
        }))
        command = ["eval-one", "--task", task["key"]]
        result = {"_checkpoint": {"global_record": {
            "status": "Completed", "scores": {
                "score_composed": ds, "score_route": 90.0},
            "infractions": {}, "meta": {}}, "records": [], "meta": {}}}
        outer = {"_checkpoint": {"global_record": {}, "records": [],
                                  "progress": [0, 1], "meta": {}}}
        audit = {
            "schema_version": "ditl.event-audit.v1", "run_id": "reuse-test",
            "route_id": "town05_short_r%d" % task["route"],
            "stream_id": "rsu-1", "consume_frame": 2,
            "consume_sim_time_ns": 200000000,
            "submitted_message_id": "m1", "submitted_source_frame": 1,
            "trace_sample_id": "s1", "cycle_index": 0,
            "used_message_id": "m1", "used_source_frame": 1,
            "source_sim_time_ns": 100000000, "ready_time_ns": 150000000,
            "base_delay_ns": 50000000, "effective_delay_ns": 50000000,
            "aoi_ns": 100000000, "zoh_hold_ns": 0, "is_zoh": False,
            "is_startup_ego_only": False, "is_expired": False,
            "arrival_deadline_miss": False, "fusion_deadline_miss": False,
            "drop_reason": None, "reorder_status": "none",
            "duplicate_status": "none", "policy_seed": task["replay_seed"],
            "policy_config_hash": "b" * 64,
            "rng_version": "ditl.blake2b-u53.v1", "queue_depth": 0,
            "accepted_count": 1, "dropped_count": 0, "expired_count": 0,
        }
        shared = {"task_key": task["key"], "attempt": 1,
                  "worker": "worker6", "gpu": 6,
                  "ports": [47000, 47001, 47002, 47005],
                  "command_argv": command}
        status = dict(shared, schema="ditl.packet-loss-stage1-task-status.v1",
                      arm=task["arm"], scenario=task["scenario"],
                      route=task["route"], carla_seed=task["carla_seed"],
                      traffic_seed=task["traffic_seed"],
                      replay_seed=task["replay_seed"], repeat=task["repeat"],
                      config_sha256=self.config_sha, state="completed",
                      returncode=0)
        identity = dict(shared,
                        schema="ditl.packet-loss-stage1-command-identity.v1")
        values = {"result.json": outer if legacy else result,
                  "event_audit.jsonl": audit, "status.json": status,
                  "command_identity.json": identity}
        for name, value in values.items():
            payload = _bytes(value)
            (attempt / name).write_bytes(payload)
        if legacy:
            direct = (attempt / "work" / "v2x_final" /
                      "town05_short_collab" /
                      ("r%d_repeat0" % task["route"]) /
                      "ego_vehicle_0" / "results.json")
            direct.parent.mkdir(parents=True)
            direct.write_bytes(_bytes(result))
        artifacts = {name: {"path": name, "sha256": _sha(attempt / name)}
                     for name in values}
        evidence = {
            "schema": "ditl.packet-loss-stage1-task-evidence.v1",
            "cohort_path": str(cohort), "task_identity": copy.deepcopy(task),
            "config_sha256": self.config_sha,
            "policy_config_hash": "b" * 64, "attempt": 1,
            "worker": "worker6", "gpu": 6,
            "ports": list(shared["ports"]), "command_argv": command,
            "artifacts": artifacts,
        }
        (attempt / "evidence.json").write_bytes(_bytes(evidence))
        return attempt

    def test_canonical_source_uses_existing_classifier_and_hash_contract(self):
        cohort = self._cohort("20260928T065846Z", "r2")
        attempt = self._attempt(cohort)

        source = validate_reuse_source(
            self.manifest, cohort, self.expected, [self.sources])
        candidate = inspect_reuse_candidate(
            self.task, attempt, "canonical", self.config_sha)

        self.assertEqual(1, source["accepted_count"])
        self.assertEqual("accepted", candidate["status"])
        self.assertEqual("canonical", candidate["source_kind"])
        self.assertEqual(_sha(attempt / "evidence.json"),
                         candidate["source_hashes"]["evidence.json"])

    def test_legacy_recovers_only_direct_result_after_all_checks(self):
        cohort = self._cohort("20260926T120101Z", "r1")
        attempt = self._attempt(cohort, legacy=True, ds=61.5)

        candidate = inspect_reuse_candidate(
            self.task, attempt, "legacy_r1", self.config_sha)
        self.assertEqual("accepted", candidate["status"])
        self.assertEqual(61.5, candidate["DS"])
        self.assertIn("legacy_result.json", candidate["source_hashes"])

        evidence = json.loads((attempt / "evidence.json").read_text())
        evidence["artifacts"]["status.json"]["sha256"] = "0" * 64
        (attempt / "evidence.json").write_bytes(_bytes(evidence))
        rejected = inspect_reuse_candidate(
            self.task, attempt, "legacy_r1", self.config_sha)
        self.assertEqual("rejected", rejected["status"])
        self.assertIn("hash", rejected["rejection_reason"])

    def test_source_provenance_config_hash_must_match_target(self):
        cohort = self._cohort("20260928T065846Z", "r2")
        self._attempt(cohort)
        provenance = {
            "config_sha256": {self.task["key"]: "0" * 64},
        }
        (cohort / "provenance.json").write_bytes(_bytes(provenance))

        source = validate_reuse_source(
            self.manifest, cohort, self.expected, [self.sources])
        self.assertEqual(0, source["accepted_count"])
        self.assertEqual(1, source["rejected_count"])
        self.assertIn("provenance config hash",
                      source["candidates"][0]["rejection_reason"])

    def test_legacy_rejects_ambiguous_ego_result(self):
        cohort = self._cohort("20260926T120101Z", "r1")
        attempt = self._attempt(cohort, legacy=True)
        original = (attempt / "work" / "v2x_final" /
                    "town05_short_collab" /
                    ("r%d_repeat0" % self.task["route"]) /
                    "ego_vehicle_0" / "results.json")
        duplicate = original.parent.parent / "ego_vehicle_1" / "results.json"
        duplicate.parent.mkdir()
        duplicate.write_bytes(original.read_bytes())

        rejected = inspect_reuse_candidate(
            self.task, attempt, "legacy_r1", self.config_sha)
        self.assertEqual("rejected", rejected["status"])
        self.assertIn("exactly one", rejected["rejection_reason"])

    def test_plan_selection_is_order_independent_and_canonical_first(self):
        legacy = self._cohort("20260929T120101Z", "r1")
        old = self._cohort("20260928T065846Z", "r2")
        new = self._cohort("20260928T123554Z", "r4")
        self._attempt(legacy, legacy=True, ds=99.0)
        self._attempt(old, ds=70.0)
        new_attempt = self._attempt(new, ds=75.0)

        first = build_reuse_plan(
            self.manifest, [legacy, old, new], self.expected, [self.sources])
        second = build_reuse_plan(
            self.manifest, [new, legacy, old], self.expected, [self.sources])

        self.assertEqual(first, second)
        selected = first["selected"][self.task["key"]]
        self.assertEqual(str(new_attempt), selected["source_attempt_dir"])
        self.assertEqual(1, first["selected_count"])
        self.assertEqual(len(self.manifest["tasks"]) - 1,
                         len(first["missing_task_keys"]))
        self.assertEqual({"canonical": 1}, first["source_counts"])
        self.assertEqual(2, len(first["rejections"]))
        json.dumps(first, sort_keys=True, allow_nan=False)

    def test_materialize_makes_independent_read_only_countable_evidence(self):
        source = self._cohort("20260928T123554Z", "r4")
        source_attempt = self._attempt(source, ds=77.0)
        plan = build_reuse_plan(
            self.manifest, [source], self.expected, [self.sources])
        target = self.root / "target" / "packet-loss-stage1-20260930T000000Z-fill"
        target.mkdir(parents=True)

        report = materialize_reuse_plan(self.manifest, plan, target)

        destination = (target / "attempts" / "attempt-1" /
                       _safe(self.task["key"]))
        classified = classify_task_evidence(self.task, destination)
        evidence = json.loads((destination / "evidence.json").read_text())
        self.assertEqual("countable", classified["classification"])
        self.assertEqual([self.task["key"]], report["selected_task_keys"])
        self.assertEqual(str(target), evidence["cohort_path"])
        self.assertEqual(str(source_attempt),
                         evidence["reuse_provenance"]["source_attempt_dir"])
        for name in ("evidence.json", "result.json", "event_audit.jsonl",
                     "status.json", "command_identity.json"):
            path = destination / name
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(0, path.stat().st_mode & 0o222)
            self.assertNotEqual(os.stat(str(path)).st_ino,
                                os.stat(str(source_attempt / name)).st_ino)

    def test_materialize_revalidates_source_config_after_planning(self):
        source = self._cohort("20260928T123554Z", "r4")
        self._attempt(source)
        plan = build_reuse_plan(
            self.manifest, [source], self.expected, [self.sources])
        config = source / "configs" / (_safe(self.task["key"]) + ".yaml")
        config.write_bytes(b"changed-after-planning\n")
        target = self.root / "target-after-change"
        target.mkdir()

        with self.assertRaisesRegex(ValueError, "revalidation"):
            materialize_reuse_plan(self.manifest, plan, target)

    def test_source_must_be_authorized_direct_child_without_symlink(self):
        nested = self.sources / "nested"
        nested.mkdir()
        cohort = nested / "packet-loss-stage1-20260928T065846Z-r2"
        cohort.mkdir()
        with self.assertRaisesRegex(ValueError, "direct child"):
            validate_reuse_source(
                self.manifest, cohort, self.expected, [self.sources])

        real = self._cohort("20260928T075846Z", "r2")
        link = self.sources / "packet-loss-stage1-20260928T085846Z-r2"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate_reuse_source(
                self.manifest, link, self.expected, [self.sources])


if __name__ == "__main__":
    unittest.main()
