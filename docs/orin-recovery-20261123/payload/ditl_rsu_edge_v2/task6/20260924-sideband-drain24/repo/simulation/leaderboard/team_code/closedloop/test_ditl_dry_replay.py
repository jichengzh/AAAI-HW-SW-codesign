from __future__ import print_function

import argparse
import hashlib
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


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "ditl_dry_replay.py"
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
sys.path.insert(0, str(CLOSEDLOOP))

from ditl_replay_schema import ReplaySample  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location("ditl_dry_replay", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DryReplayCliContractTest(unittest.TestCase):
    def test_help_is_cpu_only_and_exposes_exact_required_surface(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        for option in (
                "--library-root", "--catalog", "--bundle", "--profile",
                "--frames", "--tick-ms", "--route-id", "--stream-id",
                "--seed", "--exhaustion", "--ttl-ms", "--deadline-ms",
                "--policy-json", "--audit-out", "--summary-out"):
            self.assertIn(option, result.stdout)
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("import carla", "import rclpy", "import torch",
                          "import tensorrt", "import pycuda"):
            self.assertNotIn(forbidden, source.lower())

    def test_bundle_and_profile_are_exactly_one_and_numbers_are_positive(self):
        replay = _load_script()
        parser = replay.build_parser()
        common = [
            "--library-root", "/tmp/library", "--catalog", "/tmp/library/catalog.json",
            "--frames", "1", "--tick-ms", "50", "--route-id", "route",
            "--stream-id", "rsu", "--seed", "1", "--exhaustion", "cycle",
            "--policy-json", "/tmp/policy.json", "--audit-out", "/tmp/audit.jsonl",
            "--summary-out", "/tmp/summary.json",
        ]
        for selection in ([], ["--bundle", "/tmp/x", "--profile", "p"]):
            with self.subTest(selection=selection), self.assertRaises(SystemExit):
                parser.parse_args(common + selection)
        with self.assertRaises(SystemExit):
            parser.parse_args(common + ["--profile", "p", "--frames", "0"])

    def test_canonical_percentiles_and_summary_recompute_from_audit(self):
        replay = _load_script()
        records = [
            {"base_delay_ns": 10, "aoi_ns": None, "queue_depth": 1,
             "cycle_index": 0, "accepted_count": 0, "dropped_count": 0,
             "expired_count": 0, "effective_delay_ns": 12,
             "drop_reason": None, "reorder_status": "scheduled",
             "duplicate_status": "none", "arrival_deadline_miss": False,
             "fusion_deadline_miss": False, "is_expired": False},
            {"base_delay_ns": 20, "aoi_ns": 25, "queue_depth": 2,
             "cycle_index": 1, "accepted_count": 1, "dropped_count": 0,
             "expired_count": 0, "effective_delay_ns": None,
             "drop_reason": "bernoulli", "reorder_status": "not_applicable",
             "duplicate_status": "not_applicable",
             "arrival_deadline_miss": False,
             "fusion_deadline_miss": False, "is_expired": False},
            {"base_delay_ns": 30, "aoi_ns": 40, "queue_depth": 1,
             "cycle_index": 1, "accepted_count": 2, "dropped_count": 1,
             "expired_count": 1, "effective_delay_ns": 36,
             "drop_reason": None, "reorder_status": "delivered",
             "duplicate_status": "discarded",
             "arrival_deadline_miss": True,
             "fusion_deadline_miss": True, "is_expired": True},
        ]
        metrics = replay.recompute_audit_metrics(records)
        self.assertEqual(2, metrics["max_queue_depth"])
        self.assertEqual(2, metrics["cycles"])
        self.assertEqual({"p50": 20, "p95": 30, "p99": 30},
                         metrics["delay_percentiles_ns"])
        self.assertEqual({"p50": 25, "p95": 40, "p99": 40},
                         metrics["aoi_percentiles_ns"])
        self.assertEqual({"p50": 12, "p95": 36, "p99": 36},
                         metrics["effective_delay_percentiles_ns"])
        self.assertEqual({"bernoulli": 1}, metrics["drop_reason_counts"])
        self.assertEqual(
            {"delivered": 1, "not_applicable": 1, "scheduled": 1},
            metrics["reorder_status_counts"],
        )
        self.assertEqual(
            {"discarded": 1, "none": 1, "not_applicable": 1},
            metrics["duplicate_status_counts"],
        )
        self.assertEqual(1, metrics["arrival_deadline_miss_count"])
        self.assertEqual(1, metrics["fusion_deadline_miss_count"])
        self.assertEqual(1, metrics["expired_visible_count"])
        self.assertEqual(2, metrics["accepted_count"])
        self.assertEqual(1, metrics["dropped_count"])
        self.assertEqual(1, metrics["expired_count"])

    def test_policy_loader_rejects_noncanonical_and_unknown_json(self):
        replay = _load_script()
        canonical = {
            "extra_delay_ms": 0, "jitter_ms": 0, "drop_rate": 0,
            "burst": {"enabled": False, "enter_rate": 0, "exit_rate": 1},
            "reorder_rate": 0, "reorder_extra_ms": 0, "duplicate_rate": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(canonical, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "canonical"):
                replay.load_canonical_policy(path)
            canonical["unknown"] = 1
            path.write_text(json.dumps(canonical, sort_keys=True,
                                       separators=(",", ":")) + "\n")
            with self.assertRaisesRegex(ValueError, "unknown"):
                replay.load_canonical_policy(path)

    def test_thousand_frame_dry_runs_are_byte_identical_and_recomputable(self):
        replay = _load_script()
        samples = tuple(
            ReplaySample(
                sample_id="sample-%d" % index, ordinal=index,
                source_trace_id="trace-%d" % index,
                source_sequence_id=index, base_delay_ns=delay,
                stage_ns={
                    "acquisition_preprocess": delay,
                    "gpu_inference": 0,
                    "packing_postprocess": 0,
                    "transport_receive": 0,
                },
                raw_payload_bytes=1, wire_payload_bytes=1,
                capture_condition_id="condition",
            )
            for index, delay in enumerate((50 * 1000 * 1000,
                                           125 * 1000 * 1000))
        )
        policy = {
            "extra_delay_ms": 0, "jitter_ms": 0, "drop_rate": 0,
            "burst": {"enabled": False, "enter_rate": 0, "exit_rate": 1},
            "reorder_rate": 0, "reorder_extra_ms": 0, "duplicate_rate": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            bundle_path = root / "v1" / "bundles" / ("a" * 64)
            bundle_path.mkdir(parents=True)
            policy_path = Path(tmp) / "policy.json"
            policy_path.write_text(json.dumps(
                policy, sort_keys=True, separators=(",", ":")) + "\n")
            bundle = SimpleNamespace(
                feature_id="a" * 64, profile_id="synthetic",
                bundle_path=bundle_path, samples=samples,
            )
            summaries = []
            audits = []
            for label in ("a", "b"):
                output = Path(tmp) / label
                output.mkdir()
                args = argparse.Namespace(
                    library_root=str(root), catalog=str(root / "catalog.json"),
                    bundle=str(bundle_path), profile=None, frames=1000,
                    tick_ms=50, route_id="dry-route", stream_id="rsu-1",
                    seed=20260923, exhaustion="cycle",
                    ttl_ms=None, deadline_ms=None,
                    policy_json=str(policy_path),
                    audit_out=str(output / "audit.jsonl"),
                    summary_out=str(output / "summary.json"),
                )
                harness = replay.build_replay_engine(bundle, args)
                summaries.append(replay.simulate_integer_ticks(harness, args))
                audits.append(Path(args.audit_out).read_bytes())
            self.assertEqual(audits[0], audits[1])
            self.assertEqual(hashlib.sha256(audits[0]).hexdigest(),
                             summaries[0]["audit_sha256"])
            self.assertEqual(summaries[0], summaries[1])
            self.assertEqual(500, summaries[0]["cycles"])
            self.assertLessEqual(summaries[0]["max_queue_depth"],
                                 summaries[0]["max_queue_events"])
            records = [json.loads(line) for line in
                       audits[0].decode("ascii").splitlines()]
            independently = replay.recompute_audit_metrics(records)
            self.assertEqual(independently, {
                key: summaries[0][key] for key in independently
            })

    def test_cli_ttl_expires_delayed_messages_in_executable_replay(self):
        replay = _load_script()
        sample = ReplaySample(
            sample_id="sample-0", ordinal=0, source_trace_id="trace-0",
            source_sequence_id=0, base_delay_ns=85 * 1000 * 1000,
            stage_ns={
                "acquisition_preprocess": 85 * 1000 * 1000,
                "gpu_inference": 0,
                "packing_postprocess": 0,
                "transport_receive": 0,
            },
            raw_payload_bytes=1, wire_payload_bytes=1,
            capture_condition_id="condition",
        )
        policy = {
            "extra_delay_ms": 0, "jitter_ms": 0, "drop_rate": 0,
            "burst": {"enabled": False, "enter_rate": 0, "exit_rate": 1},
            "reorder_rate": 0, "reorder_extra_ms": 0, "duplicate_rate": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            bundle_path = root / "v1" / "bundles" / ("a" * 64)
            bundle_path.mkdir(parents=True)
            policy_path = Path(tmp) / "policy.json"
            policy_path.write_text(json.dumps(
                policy, sort_keys=True, separators=(",", ":")) + "\n")
            bundle = SimpleNamespace(
                feature_id="a" * 64, profile_id="synthetic",
                bundle_path=bundle_path, samples=(sample,),
            )
            output = Path(tmp) / "output"
            output.mkdir()
            args = argparse.Namespace(
                library_root=str(root), catalog=str(root / "catalog.json"),
                bundle=str(bundle_path), profile=None, frames=10,
                tick_ms=50, route_id="dry-route", stream_id="rsu-1",
                seed=20260923, exhaustion="cycle",
                ttl_ms=40, deadline_ms=75,
                policy_json=str(policy_path),
                audit_out=str(output / "audit.jsonl"),
                summary_out=str(output / "summary.json"),
            )
            harness = replay.build_replay_engine(bundle, args)
            summary = replay.simulate_integer_ticks(harness, args)
            self.assertEqual("ditl.dry-replay-summary.v2",
                             summary["schema_version"])
            self.assertGreater(summary["expired_count"], 0)
            self.assertEqual(40 * 1000 * 1000, summary["ttl_ns"])
            self.assertEqual(75 * 1000 * 1000, summary["deadline_ns"])

    def test_executable_replay_reports_policy_drops(self):
        replay = _load_script()
        sample = ReplaySample(
            sample_id="sample-0", ordinal=0, source_trace_id="trace-0",
            source_sequence_id=0, base_delay_ns=85 * 1000 * 1000,
            stage_ns={
                "acquisition_preprocess": 85 * 1000 * 1000,
                "gpu_inference": 0,
                "packing_postprocess": 0,
                "transport_receive": 0,
            },
            raw_payload_bytes=1, wire_payload_bytes=1,
            capture_condition_id="condition",
        )
        policy = {
            "extra_delay_ms": 0, "jitter_ms": 0, "drop_rate": 1,
            "burst": {"enabled": False, "enter_rate": 0, "exit_rate": 1},
            "reorder_rate": 0, "reorder_extra_ms": 0, "duplicate_rate": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            bundle_path = root / "v1" / "bundles" / ("a" * 64)
            bundle_path.mkdir(parents=True)
            policy_path = Path(tmp) / "policy.json"
            policy_path.write_text(json.dumps(
                policy, sort_keys=True, separators=(",", ":")) + "\n")
            bundle = SimpleNamespace(
                feature_id="a" * 64, profile_id="synthetic",
                bundle_path=bundle_path, samples=(sample,),
            )
            output = Path(tmp) / "output"
            output.mkdir()
            args = argparse.Namespace(
                library_root=str(root), catalog=str(root / "catalog.json"),
                bundle=str(bundle_path), profile=None, frames=10,
                tick_ms=50, route_id="dry-route", stream_id="rsu-1",
                seed=20260923, exhaustion="cycle",
                ttl_ms=None, deadline_ms=None,
                policy_json=str(policy_path),
                audit_out=str(output / "audit.jsonl"),
                summary_out=str(output / "summary.json"),
            )
            harness = replay.build_replay_engine(bundle, args)
            summary = replay.simulate_integer_ticks(harness, args)
            self.assertEqual(10, summary["dropped_count"])
            self.assertEqual(0, summary["accepted_count"])
            self.assertEqual({"bernoulli": 10},
                             summary["drop_reason_counts"])
            self.assertEqual({"not_applicable": 10},
                             summary["reorder_status_counts"])
            self.assertEqual({"not_applicable": 10},
                             summary["duplicate_status_counts"])

    def test_audit_creation_race_fails_without_appending_or_summary(self):
        replay = _load_script()
        real_writer = replay.DitlAuditWriter
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.jsonl"
            summary = root / "summary.json"
            args = argparse.Namespace(
                frames=1, tick_ms=50, route_id="route", stream_id="rsu",
                seed=1, exhaustion="cycle", audit_out=str(audit),
                summary_out=str(summary),
            )
            harness = SimpleNamespace(policy=SimpleNamespace(
                config_hash="a" * 64))
            foreign = b'{"foreign":true}\n'

            def race_writer(path, **kwargs):
                Path(path).write_bytes(foreign)
                return real_writer(path, **kwargs)

            with mock.patch.object(replay, "DitlAuditWriter",
                                   side_effect=race_writer):
                with self.assertRaisesRegex(IOError, str(audit)):
                    replay.simulate_integer_ticks(harness, args)
            self.assertEqual(foreign, audit.read_bytes())
            self.assertFalse(summary.exists())

    def test_direct_simulation_revalidates_frame_and_tick_constraints(self):
        replay = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            base = dict(
                route_id="route", stream_id="rsu", seed=1,
                exhaustion="cycle", audit_out=str(Path(tmp) / "audit.jsonl"),
                summary_out=str(Path(tmp) / "summary.json"),
            )
            for frames, tick_ms, message in (
                    (0, 50, "frames"),
                    (replay.MAX_FRAMES + 1, 50, "frames"),
                    (1, 0, "tick-ms"),
                    (True, 50, "frames")):
                with self.subTest(frames=frames, tick_ms=tick_ms):
                    args = argparse.Namespace(
                        frames=frames, tick_ms=tick_ms, **base
                    )
                    with self.assertRaisesRegex((TypeError, ValueError),
                                                message):
                        replay.simulate_integer_ticks(object(), args)


if __name__ == "__main__":
    unittest.main()
