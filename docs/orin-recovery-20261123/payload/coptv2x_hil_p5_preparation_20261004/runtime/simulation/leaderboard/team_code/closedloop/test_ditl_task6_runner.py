"""CPU-only contracts for the isolated Task 6 closed-loop runner."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "ditl_task6_runner.py"
BASELINE = ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_dualfifo_baseline_l1.yaml"
ROUTE = ROOT / "simulation/leaderboard/data/evaluation_routes/town05_short_r146.xml"
PROFILES = (
    "raw_samehost_idle",
    "raw_samehost_gpu_low",
    "raw_samehost_gpu_medium",
    "raw_samehost_gpu_high",
)


def load_runner():
    spec = importlib.util.spec_from_file_location("ditl_task6_runner_under_test", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Task6RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()
        root = Path("/var/tmp/codriving-ditl")
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="task6-runner-test-", dir=str(root))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.library = self.base / "library"
        (self.library / "v1/bundles").mkdir(parents=True)
        (self.library / "catalog.json").write_text("{}")
        self.report = self.base / "orin_capture_report.json"
        self.bundle_rows = {}
        self.verified = {}
        for index, profile in enumerate(PROFILES):
            feature_id = ("%x" % (index + 1)) * 64
            bundle = self.library / "v1/bundles" / feature_id
            bundle.mkdir()
            (bundle / "manifest.json").write_text(json.dumps({"profile_id": profile}))
            manifest_sha = digest(bundle / "manifest.json")
            self.bundle_rows[profile] = {
                "feature_id": feature_id,
                "bundle_path": str(bundle),
                "manifest_sha256": manifest_sha,
            }
            self.verified[str(bundle)] = types.SimpleNamespace(
                feature_id=feature_id,
                profile_id=profile,
                bundle_path=bundle,
                manifest={"accepted_count": 100},
            )
        self.report.write_text(json.dumps({
            "status": "verified", "profiles": [], "profile_bundles": self.bundle_rows,
        }))

    def verifier(self, bundle_path, library_root, catalog_path=None):
        self.assertEqual(Path(library_root), self.library)
        self.assertEqual(Path(catalog_path), self.library / "catalog.json")
        return self.verified[str(bundle_path)]

    def mapped(self):
        return self.runner.resolve_bundle_mapping(
            self.report, self.library / "catalog.json", verifier=self.verifier
        )

    def cohort_path(self, label="cohort"):
        return self.base.parent / ("task6-%s-%s" % (self.base.name, label))

    @staticmethod
    def write_route_result(result_root, status="Completed", route_id=146,
                           repeat=0):
        path = (Path(result_root) / "v2x_final/town05_short_collab" /
                ("r%d_repeat%d" % (route_id, repeat)) /
                "ego_vehicle_0/results.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        route_status = status
        global_status = "Completed" if status == "Completed" else "Failed"
        route = {
            "route_id": "RouteScenario_%d" % route_id,
            "status": route_status,
            "scores": {"score_composed": 12.5, "score_route": 25.0},
            "infractions": {"collisions_layout": [], "collisions_vehicle": [],
                            "route_timeout": []},
            "meta": {},
        }
        global_record = copy.deepcopy(route)
        global_record.update({"route_id": -1, "status": global_status,
                              "infractions": {"collisions_layout": 0.0,
                                              "collisions_vehicle": 0.0,
                                              "route_timeout": 0.0},
                              "meta": {"exceptions": []}})
        if status != "Completed":
            global_record["meta"]["exceptions"] = [
                ["RouteScenario_%d" % route_id, 0, status]]
        path.write_text(json.dumps({"_checkpoint": {
            "global_record": global_record, "records": [route],
        }}))
        return path

    def runtime_fixture(self):
        runtime = self.base / "runtime"
        trusted = self.base / "trusted"
        (runtime / "simulation/leaderboard/team_code/agent_config").mkdir(parents=True)
        (runtime / "codriving/hypes_yaml/codriving").mkdir(parents=True)
        (runtime / "opencood/utils").mkdir(parents=True)
        (trusted / "external_paths/carla_root").mkdir(parents=True)
        (trusted / "external_paths/data_root").mkdir(parents=True)
        (trusted / "checkpoints/codriving/perception").mkdir(parents=True)
        (trusted / "checkpoints/codriving/planner").mkdir(parents=True)
        (trusted / "opencood/utils").mkdir(parents=True)
        (runtime / "external_paths").symlink_to(trusted / "external_paths", target_is_directory=True)
        (runtime / "checkpoints").symlink_to(trusted / "checkpoints", target_is_directory=True)
        launcher = trusted / "external_paths/carla_root/CarlaUE4.sh"
        launcher.write_text("#!/bin/sh\n")
        launcher.chmod(0o755)
        (trusted / "checkpoints/codriving/perception/config.yaml").write_text("model: {}\n")
        perception_checkpoint = trusted / "checkpoints/codriving/perception/net_epoch_bestval_at16.pth"
        perception_checkpoint.write_bytes(b"perception")
        planner_checkpoint = trusted / "checkpoints/codriving/planner/codriving_planner.ckpt"
        planner_checkpoint.write_bytes(b"planner")
        planner_config = runtime / "codriving/hypes_yaml/codriving/end2end_codriving.yaml"
        planner_config.write_text("model: {}\n")
        native_extension = trusted / "opencood/utils/box_overlaps.cpython-37m-x86_64-linux-gnu.so"
        native_extension.write_bytes(b"python-3.7-native-extension")
        (runtime / "opencood/utils/box_overlaps.cpython-37m-x86_64-linux-gnu.so").symlink_to(
            native_extension)
        baseline = runtime / "simulation/leaderboard/team_code/agent_config/baseline.yaml"
        baseline.write_text(yaml.safe_dump({
            "simulation": {},
            "perception": {"perception_model_dir": "checkpoints/codriving/perception"},
            "planning": {
                "planner_config": "codriving/hypes_yaml/codriving/end2end_codriving.yaml",
                "planner_model_checkpoint": "checkpoints/codriving/planner/codriving_planner.ckpt",
            },
        }))
        return {
            "runtime": runtime, "trusted": trusted, "baseline": baseline,
            "launcher": launcher, "data_root": trusted / "external_paths/data_root",
            "perception_checkpoint": perception_checkpoint,
            "planner_checkpoint": planner_checkpoint, "native_extension": native_extension,
        }

    @staticmethod
    def idle_snapshot():
        return {
            "protected": {"head": "a" * 40, "status": "",
                          "supervisor": {"pid": 2088364, "active": True, "start_token": "1"}},
            "source": {"head": "a" * 40, "status": ""},
            "process_tree": [], "listening_ports": [],
            "gpus": {
                "6": {"utilization_percent": 0, "memory_used_mib": 0, "compute_pids": []},
                "7": {"utilization_percent": 0, "memory_used_mib": 0, "compute_pids": []},
            },
        }

    def test_exact_mapping_is_verified_without_profile_catalog_search(self):
        mapped = self.mapped()
        self.assertEqual(set(mapped), set(PROFILES))
        for profile in PROFILES:
            self.assertEqual(mapped[profile]["feature_id"], self.bundle_rows[profile]["feature_id"])
            self.assertEqual(mapped[profile]["bundle_path"], self.bundle_rows[profile]["bundle_path"])
        for field, replacement, message in (
            ("feature_id", "f" * 64, "feature_id"),
            ("bundle_path", str(self.library / "v1/bundles" / ("f" * 64)), "bundle_path"),
            ("manifest_sha256", "f" * 64, "manifest"),
        ):
            bad = json.loads(self.report.read_text())
            bad["profile_bundles"][PROFILES[0]][field] = replacement
            self.report.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, message):
                self.runner.resolve_bundle_mapping(
                    self.report, self.library / "catalog.json", verifier=self.verifier
                )
            self.report.write_text(json.dumps({
                "status": "verified", "profiles": [], "profile_bundles": self.bundle_rows,
            }))
        bad = json.loads(self.report.read_text())
        del bad["profile_bundles"][PROFILES[-1]]
        self.report.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "exact four"):
            self.runner.resolve_bundle_mapping(
                self.report, self.library / "catalog.json", verifier=self.verifier
            )

    def test_mapping_rejects_malformed_and_underfilled_verified_records(self):
        original = {"status": "verified", "profiles": [], "profile_bundles": self.bundle_rows}
        failures = []
        failures.append((dict(original, status="candidate"), "status"))
        extra = copy.deepcopy(original)
        extra["profile_bundles"][PROFILES[0]]["extra"] = True
        failures.append((extra, "mapping"))
        invalid_id = copy.deepcopy(original)
        invalid_id["profile_bundles"][PROFILES[0]]["feature_id"] = "bad"
        failures.append((invalid_id, "feature_id"))
        invalid_hash = copy.deepcopy(original)
        invalid_hash["profile_bundles"][PROFILES[0]]["manifest_sha256"] = "bad"
        failures.append((invalid_hash, "manifest_sha256"))
        for report, message in failures:
            self.report.write_text(json.dumps(report))
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.runner.resolve_bundle_mapping(
                    self.report, self.library / "catalog.json", verifier=self.verifier
                )
        self.report.write_text(json.dumps(original))
        verified = copy.deepcopy(self.verified)
        first = self.bundle_rows[PROFILES[0]]["bundle_path"]
        verified[first] = types.SimpleNamespace(
            feature_id=self.bundle_rows[PROFILES[0]]["feature_id"], profile_id=PROFILES[0],
            bundle_path=Path(first), manifest={"accepted_count": 99})
        with self.assertRaisesRegex(ValueError, "fewer than 100"):
            self.runner.resolve_bundle_mapping(
                self.report, self.library / "catalog.json",
                verifier=lambda path, *args, **kwargs: verified[str(path)])
        wrong = copy.deepcopy(self.verified)
        wrong[first] = types.SimpleNamespace(
            feature_id="f" * 64, profile_id=PROFILES[0], bundle_path=Path(first),
            manifest={"accepted_count": 100})
        with self.assertRaisesRegex(ValueError, "verified feature"):
            self.runner.resolve_bundle_mapping(
                self.report, self.library / "catalog.json",
                verifier=lambda path, *args, **kwargs: wrong[str(path)])

    def test_catalog_derivation_and_regular_file_guards(self):
        report = json.loads(self.report.read_text())
        self.assertEqual(self.runner._derive_catalog(report), self.library / "catalog.json")
        bad = copy.deepcopy(report)
        bad["profile_bundles"][PROFILES[0]]["bundle_path"] = "/tmp/other/v1/bundles/" + "a" * 64
        with self.assertRaisesRegex(ValueError, "one library"):
            self.runner._derive_catalog(bad)
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.runner._require_regular(Path("relative"), "input")
        with self.assertRaisesRegex(ValueError, "regular"):
            self.runner._require_regular(self.base / "missing", "input")

    def test_configs_replace_legacy_latency_and_change_one_factor(self):
        cohort = self.cohort_path("configs")
        source = {"head": "a" * 40, "status": ""}
        manifest = self.runner.generate_cohort(
            cohort, BASELINE, self.mapped(), self.library / "catalog.json", ROUTE,
            source_commit="a" * 40, render_gpu=6, infer_gpu=7,
            world_port=46000, protected_baseline=source,
        )
        self.addCleanup(lambda: __import__("shutil").rmtree(str(cohort), ignore_errors=True))
        self.assertEqual(len(manifest["arms"]), 10)
        configs = [yaml.safe_load(Path(row["config_path"]).read_text()) for row in manifest["arms"]]
        for section in ("control", "perception", "planning"):
            self.assertTrue(all(config[section] == configs[0][section] for config in configs))
        for row, config in zip(manifest["arms"], configs):
            simulation = config["simulation"]
            for legacy in ("rsu_fifo_delay_ms", "comm_latency", "latency_inject_ms", "latency_trace"):
                self.assertNotIn(legacy, simulation)
            replay = simulation["ditl_replay"]
            self.assertEqual(replay["bundle"], row["bundle_path"])
            self.assertEqual(replay["seed"], 2000)
            self.assertEqual(replay["exhaustion"], "cycle")
            self.assertIsNone(replay["ttl_ms"])
            self.assertIsNone(replay["deadline_ms"])
            self.assertGreater(replay["max_queue_events"], 0)
            impairment = replay["impairment"]
            expected_drop = row["contract"]["drop_rate"]
            self.assertEqual(impairment["drop_rate"], expected_drop)
            self.assertEqual({key: impairment[key] for key in (
                "extra_delay_ms", "jitter_ms", "reorder_rate", "reorder_extra_ms", "duplicate_rate"
            )}, {"extra_delay_ms": 0, "jitter_ms": 0, "reorder_rate": 0,
                 "reorder_extra_ms": 0, "duplicate_rate": 0})
            self.assertEqual(impairment["burst"], {"enabled": False, "enter_rate": 0, "exit_rate": 0})
            if row["contract"]["family"] == "spatial":
                self.assertEqual(simulation["ditl_spatial_offset"], {
                    "dx_m": row["contract"]["spatial_dx_m"], "dy_m": 0.0, "dyaw_deg": 0.0,
                })
            else:
                self.assertNotIn("ditl_spatial_offset", simulation)

    def test_manifest_contains_complete_provenance_and_ten_unique_outputs(self):
        cohort = self.cohort_path("manifest")
        manifest = self.runner.generate_cohort(
            cohort, BASELINE, self.mapped(), self.library / "catalog.json", ROUTE,
            source_commit="b" * 40, render_gpu=6, infer_gpu=7, world_port=46000,
            protected_baseline={"head": "c" * 40, "status": " M result"},
        )
        self.addCleanup(lambda: __import__("shutil").rmtree(str(cohort), ignore_errors=True))
        disk = json.loads((cohort / "manifest.json").read_text())
        self.assertEqual(manifest, disk)
        self.assertEqual(disk["source_commit"], "b" * 40)
        self.assertEqual(disk["route"]["sha256"], digest(ROUTE))
        self.assertEqual(len({row["config_sha256"] for row in disk["arms"]}), 10)
        self.assertEqual(len({row["result_root"] for row in disk["arms"]}), 10)
        for row in disk["arms"]:
            self.assertTrue(Path(row["result_root"]).is_dir())
            self.assertEqual(Path(row["result_root"]).parent, cohort / "results")
            self.assertEqual(row["route_file_sha256"], digest(ROUTE))
            self.assertEqual(row["seeds"], {"carla": 2000, "traffic": 2000, "replay": 2000})
            self.assertEqual(row["gpus"], {"render": 6, "inference": 7})
            self.assertEqual(row["ports"], {"world": 46000, "traffic": 46005})
            self.assertEqual(row["bundle_feature_id"], Path(row["bundle_path"]).name)
            self.assertIn("perception_model_dir", row["model_paths"])
            self.assertIn("planner_model_checkpoint", row["model_paths"])
            self.assertEqual(row["command_argv"][0], str(ROOT / "scripts/ditl_task6_eval_one.sh"))

    def test_cohort_must_be_new_absolute_confined_and_without_symlink_components(self):
        relative = Path("task6-relative")
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.runner.validate_new_cohort_path(relative)
        with self.assertRaisesRegex(ValueError, "task6"):
            self.runner.validate_new_cohort_path(Path("/tmp/not-authorized"))
        existing = self.cohort_path("existing")
        existing.mkdir()
        self.addCleanup(lambda: existing.rmdir() if existing.exists() else None)
        with self.assertRaises(FileExistsError):
            self.runner.validate_new_cohort_path(existing)
        outside = self.base / "outside"
        outside.mkdir()
        link = self.base.parent / ("task6-%s-link" % self.base.name)
        link.symlink_to(outside, target_is_directory=True)
        self.addCleanup(lambda: link.unlink() if link.is_symlink() else None)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.runner.validate_new_cohort_path(link / "child")

    def test_resource_guards_reject_busy_equal_wrong_and_occupied(self):
        idle = {
            "gpus": {
                "6": {"utilization_percent": 0, "memory_used_mib": 0, "compute_pids": []},
                "7": {"utilization_percent": 0, "memory_used_mib": 0, "compute_pids": []},
            },
            "listening_ports": [],
        }
        self.runner.validate_resource_snapshot(idle, 6, 7, 46000, 46005)
        cases = [
            (idle, 6, 6, 46000, 46005, "different"),
            (idle, 6, 7, 40000, 40005, "reserved"),
            (dict(idle, listening_ports=[46005]), 6, 7, 46000, 46005, "occupied"),
        ]
        busy = copy.deepcopy(idle)
        busy["gpus"]["7"]["compute_pids"] = [123]
        cases.append((busy, 6, 7, 46000, 46005, "busy"))
        high = copy.deepcopy(idle)
        high["gpus"]["6"]["utilization_percent"] = 6
        cases.append((high, 6, 7, 46000, 46005, "busy"))
        for snapshot, render, infer, world, traffic, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.runner.validate_resource_snapshot(snapshot, render, infer, world, traffic)
        for invalid, message in ((None, "missing"), ({}, "listening"),
                                 ({"listening_ports": [], "gpus": None}, "GPU snapshot")):
            with self.assertRaisesRegex(ValueError, message):
                self.runner.validate_resource_snapshot(invalid, 6, 7, 46000, 46005)

    def test_runtime_assets_fail_closed_before_cohort_creation(self):
        fixture = self.runtime_fixture()
        allowed = (fixture["runtime"], fixture["trusted"])
        self.assertTrue(hasattr(self.runner, "validate_runtime_assets"),
                        "runtime asset preflight is missing")
        validated = self.runner.validate_runtime_assets(
            fixture["baseline"], root=fixture["runtime"], allowed_roots=allowed)
        self.assertEqual(validated["carla_launcher"], fixture["launcher"].resolve())
        self.assertEqual(validated["perception_checkpoints"],
                         [fixture["perception_checkpoint"].resolve()])
        self.assertIn("native_extension", validated,
                      "Python 3.7 native extension gate is missing")
        self.assertEqual(validated["native_extension"], fixture["native_extension"].resolve())

        missing_cases = (
            ("launcher", "CARLA launcher"),
            ("data_root", "data root"),
            ("perception_checkpoint", "perception checkpoint"),
            ("planner_checkpoint", "planner checkpoint"),
            ("native_extension", "box_overlaps Python 3.7 extension"),
        )
        for key, message in missing_cases:
            path = fixture[key]
            wrong_abi = None
            if path.is_dir():
                path.rmdir()
            else:
                content, mode = path.read_bytes(), path.stat().st_mode
                path.unlink()
                if key == "native_extension":
                    wrong_abi = path.with_name(
                        "box_overlaps.cpython-38-x86_64-linux-gnu.so")
                    wrong_abi.write_bytes(b"wrong-python-abi")
            with self.subTest(asset=key), self.assertRaisesRegex(ValueError, message):
                self.runner.validate_runtime_assets(
                    fixture["baseline"], root=fixture["runtime"], allowed_roots=allowed)
            if wrong_abi is not None:
                wrong_abi.unlink()
            if key == "data_root":
                path.mkdir()
            else:
                path.write_bytes(content)
                path.chmod(mode)

        original_config = yaml.safe_load(fixture["baseline"].read_text())
        config = copy.deepcopy(original_config)
        config["planning"]["planner_model_checkpoint"] = str(self.base / "outside.ckpt")
        (self.base / "outside.ckpt").write_bytes(b"outside")
        fixture["baseline"].write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "relative"):
            self.runner.validate_runtime_assets(
                fixture["baseline"], root=fixture["runtime"], allowed_roots=allowed)

        escape = fixture["trusted"] / "checkpoints/escape.ckpt"
        escape.symlink_to(self.base / "outside.ckpt")
        config = copy.deepcopy(original_config)
        config["planning"]["planner_model_checkpoint"] = "checkpoints/escape.ckpt"
        fixture["baseline"].write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "outside the allowed"):
            self.runner.validate_runtime_assets(
                fixture["baseline"], root=fixture["runtime"], allowed_roots=allowed)

        cohort = self.cohort_path("missing-runtime")
        args = types.SimpleNamespace(
            cohort=cohort, baseline=fixture["baseline"], capture_report=self.report,
            catalog=self.library / "catalog.json", route_file=ROUTE,
            render_gpu=6, infer_gpu=7, world_port=46000, dry_run=True,
            isolation_snapshot=None,
        )
        with mock.patch.object(self.runner, "validate_runtime_assets",
                               side_effect=ValueError("CARLA launcher is missing")), \
             mock.patch.object(self.runner, "resolve_bundle_mapping") as resolve:
            with self.assertRaisesRegex(ValueError, "CARLA launcher"):
                self.runner.execute(args)
        self.assertFalse(cohort.exists())
        resolve.assert_not_called()

    def test_live_bind_guard_checks_ipv4_and_ipv6_and_closes_every_probe(self):
        probes = []
        class Probe:
            def __init__(self, family, kind):
                self.family, self.bound, self.closed = family, None, False
                probes.append(self)
            def setsockopt(self, *args):
                self.option = args
            def bind(self, address):
                self.bound = address
            def close(self):
                self.closed = True
        self.runner.bind_guard_ports(46000, 46005, socket_factory=Probe)
        self.assertEqual([probe.bound for probe in probes], [
            ("0.0.0.0", 46000), ("::", 46000),
            ("0.0.0.0", 46001), ("::", 46001),
            ("0.0.0.0", 46002), ("::", 46002),
            ("0.0.0.0", 46005), ("::", 46005)])
        self.assertTrue(all(probe.closed for probe in probes))

    def test_port_release_wait_retries_the_exact_ports_until_bindable(self):
        now = [0.0]
        sleeps = []
        attempts = []

        def guard(world, traffic):
            attempts.append((world, traffic))
            if len(attempts) < 3:
                raise RuntimeError("still releasing")

        def pause(delay):
            sleeps.append(delay)
            now[0] += delay

        self.runner.wait_for_ports_bindable(
            46000, 46005, timeout_s=1.0, poll_s=0.2,
            bind_guard=guard, listening_ports=lambda: [],
            monotonic=lambda: now[0], pause=pause)

        self.assertEqual(attempts, [(46000, 46005)] * 3)
        self.assertEqual(sleeps, [0.2, 0.2])

    def test_default_port_release_wait_covers_tcp_time_wait_window(self):
        now = [0.0]
        attempts = []

        def guard(world, traffic):
            attempts.append((world, traffic))
            if now[0] < 61.0:
                raise RuntimeError("no listener; kernel address residue remains")

        def pause(delay):
            now[0] += delay

        self.runner.wait_for_ports_bindable(
            46000, 46005, bind_guard=guard, listening_ports=lambda: [],
            monotonic=lambda: now[0], pause=pause)

        self.assertEqual(now[0], 61.0)
        self.assertEqual(attempts, [(46000, 46005)] * 245)

    def test_port_release_wait_fails_closed_on_persistent_foreign_listener(self):
        now = [0.0]
        attempts = []

        def occupied(world, traffic):
            attempts.append((world, traffic))
            raise RuntimeError("Address already in use by foreign listener")

        def pause(delay):
            now[0] += delay

        with mock.patch.object(self.runner.os, "killpg") as signal_group:
            with self.assertRaisesRegex(RuntimeError, "ports 46000,46001,46002,46005.*foreign"):
                self.runner.wait_for_ports_bindable(
                    46000, 46005, timeout_s=0.5, poll_s=0.2,
                    bind_guard=occupied, listening_ports=lambda: [46001],
                    monotonic=lambda: now[0], pause=pause)
        signal_group.assert_not_called()
        self.assertEqual(attempts, [(46000, 46005)])

    def test_port_release_wait_times_out_on_unowned_kernel_residue(self):
        now = [0.0]
        attempts = []

        def unavailable(world, traffic):
            attempts.append((world, traffic))
            raise RuntimeError("Address already in use")

        def pause(delay):
            now[0] += delay

        with self.assertRaisesRegex(RuntimeError, "did not become bindable"):
            self.runner.wait_for_ports_bindable(
                46000, 46005, timeout_s=0.5, poll_s=0.2,
                bind_guard=unavailable, listening_ports=lambda: [],
                monotonic=lambda: now[0], pause=pause)
        self.assertEqual(attempts, [(46000, 46005)] * 4)

    def test_adjacent_carla_ports_are_rejected_from_snapshot(self):
        for occupied in (46000, 46001, 46002, 46005):
            snapshot = self.idle_snapshot()
            snapshot["listening_ports"] = [occupied]
            with self.subTest(port=occupied), self.assertRaisesRegex(ValueError, "occupied"):
                self.runner.validate_resource_snapshot(snapshot, 6, 7, 46000, 46005)

    def test_arm_result_requires_countable_terminal_exact_route_identity(self):
        result_root = self.base / "arm-result"
        row = {"arm_id": "clean", "result_root": str(result_root),
               "contract": {"route_id": 146, "repeat": 0}}

        accepted = self.write_route_result(
            result_root, status="Failed - Agent got blocked")
        self.assertEqual(self.runner.validate_arm_result(row), accepted)

        for invalid_status in ("Failed - Agent crashed", "Started"):
            accepted.unlink()
            self.write_route_result(result_root, status=invalid_status)
            with self.subTest(status=invalid_status), self.assertRaisesRegex(
                    RuntimeError, "non-countable"):
                self.runner.validate_arm_result(row)

        accepted.unlink()
        wrong = self.write_route_result(result_root, route_id=147)
        with self.assertRaisesRegex(RuntimeError, "canonical result"):
            self.runner.validate_arm_result(row)
        wrong.unlink()
        self.write_route_result(result_root, route_id=146, repeat=1)
        with self.assertRaisesRegex(RuntimeError, "canonical result"):
            self.runner.validate_arm_result(row)

    def test_arm_result_rejects_stale_output_before_launch(self):
        result_root = self.base / "stale-result"
        result_root.mkdir()
        row = {"arm_id": "clean", "result_root": str(result_root),
               "contract": {"route_id": 146, "repeat": 0}}
        self.runner.require_empty_arm_result_root(row)
        self.write_route_result(result_root)
        with self.assertRaisesRegex(RuntimeError, "pre-existing result output"):
            self.runner.require_empty_arm_result_root(row)

    def test_arm_result_rejects_malformed_checkpoint_metadata(self):
        result_root = self.base / "malformed-result"
        row = {"arm_id": "clean", "result_root": str(result_root),
               "contract": {"route_id": 146, "repeat": 0}}
        path = self.write_route_result(result_root)
        document = json.loads(path.read_text())
        document["_checkpoint"]["global_record"]["meta"] = None
        path.write_text(json.dumps(document))

        with self.assertRaisesRegex(RuntimeError, "metadata"):
            self.runner.validate_arm_result(row)

    def test_dry_run_lists_exactly_ten_sequential_commands_without_launch(self):
        cohort = self.cohort_path("dry")
        args = types.SimpleNamespace(
            cohort=cohort, baseline=BASELINE, capture_report=self.report,
            catalog=self.library / "catalog.json", route_file=ROUTE,
            render_gpu=6, infer_gpu=7, world_port=46000, dry_run=True,
        )
        output = []
        source = {"head": "a" * 40, "status": "", "supervisor": {"pid": 2088364, "active": True}}
        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()), \
             mock.patch.object(self.runner, "capture_isolation_snapshot", return_value={
                 "protected": source, "source": {"head": "a" * 40, "status": ""}, "gpus": {
                     "6": {"utilization_percent": 0, "memory_used_mib": 0, "compute_pids": []},
                     "7": {"utilization_percent": 0, "memory_used_mib": 0, "compute_pids": []},
                 }, "listening_ports": [], "process_tree": []}), \
             mock.patch.object(self.runner.subprocess, "Popen") as launch:
            self.assertEqual(self.runner.execute(args, emit=output.append), 0)
        self.addCleanup(lambda: __import__("shutil").rmtree(str(cohort), ignore_errors=True))
        launch.assert_not_called()
        commands = [json.loads(line) for line in output]
        self.assertEqual(len(commands), 10)
        for index, command in enumerate(commands):
            self.assertEqual(command["sequence"], index)
            argv = command["argv"]
            self.assertIn("146", argv)
            self.assertIn("_1", argv)
            self.assertIn("2000", argv)
            self.assertIn("46000", argv)
            self.assertIn("46005", argv)
            self.assertIn("6", argv)
            self.assertIn("7", argv)
            result = Path(argv[argv.index("--result-root") + 1])
            self.assertEqual(result.parent, cohort / "results")
            process_dir = Path(argv[argv.index("--process-dir") + 1])
            log_dir = Path(argv[argv.index("--log-dir") + 1])
            self.assertEqual(process_dir.parent, cohort / "processes")
            self.assertEqual(log_dir.parent, cohort / "logs")

    def test_dry_run_accepts_complete_cpu_fixture_snapshot_only(self):
        fixture = self.base / "snapshot.json"
        fixture.write_text(json.dumps(self.idle_snapshot()))
        self.assertEqual(self.runner._fixture_isolation_snapshot(fixture), self.idle_snapshot())
        bad = self.base / "bad-snapshot.json"
        bad.write_text("{}")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.runner._fixture_isolation_snapshot(bad)
        args = types.SimpleNamespace(dry_run=False, isolation_snapshot=fixture)
        args.render_gpu, args.infer_gpu, args.world_port = 6, 7, 46000
        args.capture_report, args.catalog = self.report, self.library / "catalog.json"
        args.cohort, args.baseline, args.route_file = self.cohort_path("fixture-live"), BASELINE, ROUTE
        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()):
            with self.assertRaisesRegex(ValueError, "only with --dry-run"):
                self.runner.execute(args)

    def test_live_execution_is_sequential_and_failed_arm_is_terminal(self):
        class FakeProcess:
            next_pid = 1000
            def __init__(self, returncode):
                self.pid = FakeProcess.next_pid
                FakeProcess.next_pid += 1
                self.returncode = returncode
            def wait(self):
                return self.returncode
            def poll(self):
                return self.returncode

        def args(label):
            return types.SimpleNamespace(
                cohort=self.cohort_path(label), baseline=BASELINE, capture_report=self.report,
                catalog=self.library / "catalog.json", route_file=ROUTE,
                render_gpu=6, infer_gpu=7, world_port=46000, dry_run=False,
                isolation_snapshot=None,
            )

        successful = args("live-success")
        processes = []
        def launch(argv, **unused_kwargs):
            result_root = Path(argv[argv.index("--result-root") + 1])
            self.write_route_result(result_root)
            process = FakeProcess(0)
            processes.append(process)
            return process
        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()), \
             mock.patch.object(self.runner, "capture_isolation_snapshot", side_effect=lambda *x: copy.deepcopy(self.idle_snapshot())), \
             mock.patch.object(self.runner.subprocess, "Popen", side_effect=launch), \
             mock.patch.object(self.runner, "_owned_process_record", return_value={}), \
             mock.patch.object(self.runner, "bind_guard_ports"), \
             mock.patch.object(self.runner, "terminate_recorded_groups") as terminate:
            self.assertEqual(self.runner.execute(successful, emit=lambda x: None), 0)
        self.addCleanup(lambda: __import__("shutil").rmtree(str(successful.cohort), ignore_errors=True))
        self.assertEqual(len(processes), 10)
        self.assertEqual(terminate.call_count, 10)
        statuses = sorted((successful.cohort / "status").glob("*.json"))
        self.assertEqual(len(statuses), 10)
        self.assertTrue(all(json.loads(path.read_text())["status"] == "completed" for path in statuses))

        failed = args("live-failure")
        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()), \
             mock.patch.object(self.runner, "capture_isolation_snapshot", side_effect=lambda *x: copy.deepcopy(self.idle_snapshot())), \
             mock.patch.object(self.runner.subprocess, "Popen", return_value=FakeProcess(9)), \
             mock.patch.object(self.runner, "_owned_process_record", return_value={}), \
             mock.patch.object(self.runner, "bind_guard_ports"), \
             mock.patch.object(self.runner, "terminate_recorded_groups"):
            with self.assertRaisesRegex(RuntimeError, "will not be replaced"):
                self.runner.execute(failed, emit=lambda x: None)
        self.addCleanup(lambda: __import__("shutil").rmtree(str(failed.cohort), ignore_errors=True))
        statuses = list((failed.cohort / "status").glob("*.json"))
        self.assertEqual(len(statuses), 1)
        self.assertEqual(json.loads(statuses[0].read_text())["status"], "failed")

        interrupted = args("live-interrupted")
        class InterruptedProcess:
            pid = 3000
            def wait(self):
                raise RuntimeError("interrupted by signal 15")
            def poll(self):
                return None
        terminated = []
        owner = {"pid": 3000, "pgid": 3000, "sid": 3000, "start_token": "x"}
        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()), \
             mock.patch.object(self.runner, "capture_isolation_snapshot", side_effect=lambda *x: copy.deepcopy(self.idle_snapshot())), \
             mock.patch.object(self.runner.subprocess, "Popen", return_value=InterruptedProcess()), \
             mock.patch.object(self.runner, "_owned_process_record", return_value=owner), \
             mock.patch.object(self.runner, "bind_guard_ports"), \
             mock.patch.object(self.runner, "terminate_recorded_groups", side_effect=lambda records: terminated.extend(records)):
            with self.assertRaisesRegex(RuntimeError, "will not be replaced"):
                self.runner.execute(interrupted, emit=lambda x: None)
        self.addCleanup(lambda: __import__("shutil").rmtree(str(interrupted.cohort), ignore_errors=True))
        status = json.loads(next((interrupted.cohort / "status").glob("*.json")).read_text())
        self.assertEqual(status["status"], "failed")
        self.assertIn("signal 15", status["execution_error"])
        self.assertEqual(terminated, [owner])

        postflight = args("live-postflight-failure")
        snapshots = [self.idle_snapshot(), self.idle_snapshot(), self.idle_snapshot()]
        snapshots[-1]["protected"]["status"] = "changed"
        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()), \
             mock.patch.object(self.runner, "capture_isolation_snapshot", side_effect=snapshots), \
             mock.patch.object(self.runner.subprocess, "Popen", return_value=FakeProcess(0)), \
             mock.patch.object(self.runner, "_owned_process_record", return_value=owner), \
             mock.patch.object(self.runner, "bind_guard_ports"), \
             mock.patch.object(self.runner, "terminate_recorded_groups"):
            with self.assertRaisesRegex(RuntimeError, "will not be replaced"):
                self.runner.execute(postflight, emit=lambda x: None)
        self.addCleanup(lambda: __import__("shutil").rmtree(str(postflight.cohort), ignore_errors=True))
        status = json.loads(next((postflight.cohort / "status").glob("*.json")).read_text())
        self.assertEqual(status["status"], "failed")
        self.assertIn("protected", status["postflight_error"])

    def test_zero_exit_agent_crash_fails_arm_before_next_launch(self):
        args = types.SimpleNamespace(
            cohort=self.cohort_path("agent-crash"), baseline=BASELINE,
            capture_report=self.report, catalog=self.library / "catalog.json",
            route_file=ROUTE, render_gpu=6, infer_gpu=7, world_port=46000,
            dry_run=False, isolation_snapshot=None,
        )
        test = self

        class AgentCrashProcess:
            pid = 4000
            returncode = 0

            def wait(self):
                test.write_route_result(
                    args.cohort / "results/clean", status="Failed - Agent crashed")
                return self.returncode

            def poll(self):
                return self.returncode

        with mock.patch.object(self.runner, "validate_runtime_assets"), \
             mock.patch.object(self.runner, "resolve_bundle_mapping", return_value=self.mapped()), \
             mock.patch.object(self.runner, "capture_isolation_snapshot",
                               side_effect=lambda *x: copy.deepcopy(self.idle_snapshot())), \
             mock.patch.object(self.runner.subprocess, "Popen",
                               return_value=AgentCrashProcess()) as launch, \
             mock.patch.object(self.runner, "_owned_process_record", return_value={}), \
             mock.patch.object(self.runner, "bind_guard_ports"), \
             mock.patch.object(self.runner, "wait_for_ports_bindable"), \
             mock.patch.object(self.runner, "terminate_recorded_groups"):
            with self.assertRaisesRegex(RuntimeError, "will not be replaced"):
                self.runner.execute(args, emit=lambda x: None)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            str(args.cohort), ignore_errors=True))

        self.assertEqual(launch.call_count, 1)
        status = json.loads((args.cohort / "status/clean.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["returncode"], 0)
        self.assertIn("non-countable", status["result_validation_error"])

    def test_snapshot_collectors_parse_commands_and_process_identity(self):
        outputs = ["3, 4\n", "123\n456\n"]
        with mock.patch.object(self.runner.subprocess, "check_output", side_effect=outputs):
            self.assertEqual(self.runner._gpu_snapshot(6), {
                "utilization_percent": 3, "memory_used_mib": 4, "compute_pids": [123, 456],
            })
        with mock.patch.object(self.runner, "_git_snapshot", side_effect=[{"head": "p", "status": ""}, {"head": "s", "status": ""}]), \
             mock.patch.object(self.runner, "_supervisor_snapshot", return_value={"pid": 2088364, "active": False, "start_token": None}), \
             mock.patch.object(self.runner, "_listening_ports", return_value=[80]), \
             mock.patch.object(self.runner, "_gpu_snapshot", side_effect=lambda gpu: {"gpu": gpu}), \
             mock.patch.object(self.runner.subprocess, "check_output", return_value="process\n"):
            snapshot = self.runner.capture_isolation_snapshot(6, 7)
        self.assertEqual(snapshot["listening_ports"], [80])
        self.assertEqual(snapshot["gpus"], {"6": {"gpu": 6}, "7": {"gpu": 7}})
        with mock.patch("pathlib.Path.read_text", return_value="100 (worker) S 50 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 999 0"):
            process = self.runner.inspect_process(100)
        self.assertEqual((process["ppid"], process["pgid"], process["sid"], process["start_token"]),
                         (50, 100, 100, "999"))
        with mock.patch.object(self.runner.subprocess, "check_output", side_effect=["head\n", " M file\n"]):
            self.assertEqual(self.runner._git_snapshot(self.base), {"head": "head", "status": " M file\n"})
        with mock.patch.object(self.runner, "inspect_process", return_value={"state": "S", "start_token": "9"}):
            self.assertTrue(self.runner._supervisor_snapshot(5)["active"])
        with mock.patch.object(self.runner, "inspect_process", side_effect=OSError("gone")):
            self.assertFalse(self.runner._supervisor_snapshot(5)["active"])
        tcp = "header\n  0: 00000000:B3B0 00000000:0000 0A rest\n"
        with mock.patch("pathlib.Path.read_text", return_value=tcp):
            self.assertEqual(self.runner._listening_ports(), [46000])

    def test_identity_files_and_owned_group_are_strict(self):
        identity_dir = self.base / "ids"
        identity_dir.mkdir()
        record = {"pid": 10, "pgid": 10, "sid": 10, "start_token": "x"}
        (identity_dir / "one.identity.json").write_text(json.dumps(record))
        self.assertEqual(self.runner._read_identity_records(identity_dir), [record])
        supervisor = {"pid": 20, "ppid": 12, "pgrp": 20, "session": 20, "start_token": "y"}
        (identity_dir / "two.identity.json").write_text(json.dumps(supervisor))
        normalized = self.runner._read_identity_records(identity_dir)[1]
        self.assertEqual((normalized["pgid"], normalized["sid"], normalized["owner_pid"]),
                         (20, 20, 12))
        process = types.SimpleNamespace(pid=10)
        with mock.patch.object(self.runner, "inspect_process", return_value=record):
            self.assertEqual(self.runner._owned_process_record(process), record)
        with mock.patch.object(self.runner, "inspect_process", return_value=dict(record, pgid=11)):
            with self.assertRaisesRegex(RuntimeError, "group leader"):
                self.runner._owned_process_record(process)
        bad = identity_dir / "bad.identity.json"
        bad.symlink_to(identity_dir / "one.identity.json")
        with self.assertRaisesRegex(RuntimeError, "identity file"):
            self.runner._read_identity_records(identity_dir)

    def test_arg_parser_exposes_fixture_dry_run_contract(self):
        values = self.runner.parse_args([
            "--cohort", str(self.cohort_path("parse")), "--render-gpu", "6",
            "--infer-gpu", "7", "--world-port", "46000", "--dry-run",
            "--isolation-snapshot", str(self.base / "snapshot.json"),
        ])
        self.assertTrue(values.dry_run)
        self.assertEqual((values.render_gpu, values.infer_gpu, values.world_port), (6, 7, 46000))

    def test_cli_rejects_invalid_resources_without_creating_cohort(self):
        cohort = self.cohort_path("cli-reject")
        with mock.patch.object(self.runner, "parse_args", return_value=types.SimpleNamespace(
                cohort=cohort, baseline=BASELINE, capture_report=self.report,
                catalog=self.library / "catalog.json", route_file=ROUTE,
                render_gpu=6, infer_gpu=6, world_port=46000, dry_run=True,
                isolation_snapshot=None)):
            self.assertEqual(self.runner.main([]), 3)
        self.assertFalse(cohort.exists())

    def test_protected_state_change_is_rejected(self):
        baseline = {"head": "a" * 40, "status": "", "supervisor": {"pid": 2088364, "start": "1"}}
        self.runner.assert_protected_unchanged(baseline, copy.deepcopy(baseline))
        for field in ("head", "status", "supervisor"):
            changed = copy.deepcopy(baseline)
            changed[field] = "changed"
            with self.assertRaisesRegex(RuntimeError, "protected"):
                self.runner.assert_protected_unchanged(baseline, changed)

    def test_cleanup_signals_only_live_identity_matching_recorded_groups(self):
        records = [
            {"pid": 100, "pgid": 100, "sid": 100, "start_token": "live", "owner_pid": 50},
            {"pid": 200, "pgid": 200, "sid": 200, "start_token": "stale", "owner_pid": 50},
        ]
        states = {
            100: {"pid": 100, "pgid": 100, "sid": 100, "start_token": "live"},
            200: {"pid": 200, "pgid": 200, "sid": 200, "start_token": "reused"},
        }
        calls = []
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.runner.terminate_recorded_groups(
                records, inspect=lambda pid: states[pid], signal_group=lambda pgid, sig: calls.append((pgid, sig)),
                pause=lambda _: None,
            )
        self.assertEqual(calls, [], "a foreign/stale record must abort before any signal")
        calls = []
        self.runner.terminate_recorded_groups(
            records[:1], inspect=lambda pid: states[pid],
            signal_group=lambda pgid, sig: calls.append((pgid, sig)), pause=lambda _: None,
        )
        self.assertEqual([call[0] for call in calls], [100, 100])

    def test_status_creation_is_exclusive_and_failure_stops_sequence(self):
        status = self.base / "status.json"
        self.runner.write_status_once(status, {"status": "failed", "returncode": 7})
        with self.assertRaises(FileExistsError):
            self.runner.write_status_once(status, {"status": "completed", "returncode": 0})
        self.assertEqual(json.loads(status.read_text())["status"], "failed")

    def test_signal_guard_defers_until_launch_identity_boundary(self):
        handlers = {}
        def install(sig, handler):
            old = handlers.get(sig, self.runner.signal.SIG_DFL)
            handlers[sig] = handler
            return old
        guard = self.runner.SignalCleanupGuard()
        with mock.patch.object(self.runner.signal, "signal", side_effect=install):
            with self.assertRaisesRegex(RuntimeError, "signal"):
                with guard:
                    guard.begin_launch()
                    handlers[self.runner.signal.SIGTERM](self.runner.signal.SIGTERM, None)
                    self.assertEqual(guard.pending, self.runner.signal.SIGTERM)
                    guard.finish_launch()
        self.assertEqual(handlers[self.runner.signal.SIGTERM], self.runner.signal.SIG_DFL)

    def test_eval_script_has_explicit_paths_and_no_broad_kill(self):
        script = (ROOT / "scripts/ditl_task6_eval_one.sh").read_text()
        self.assertNotIn("pkill", script)
        self.assertIn("--config", script)
        self.assertIn("--result-root", script)
        self.assertIn("setsid", script)
        self.assertIn("identity", script)
        self.assertIn("begin_owned_launch", script)
        self.assertIn("stop_pending_child", script)
        self.assertIn("--process-start-token", script)
        self.assertIn("--stop-child", script)
        self.assertEqual(script.count("if claim_owned_identity"), 2,
                         "CARLA and evaluator claims must both retain pending ownership")
        self.assertEqual(script.count("exit \"$claim_status\""), 2,
                         "both claim failures must preserve the original status")

    def test_wrapper_interrupt_before_identity_publication_leaves_no_owned_child(self):
        base = self.base / "ownership-race"
        (base / "scripts").mkdir(parents=True)
        (base / "external_paths/carla_root").mkdir(parents=True)
        delayed_supervisor = base / "scripts/dual_fifo_result.py"
        delayed_supervisor.write_text(
            "import os, sys, time\n"
            "real = %r\n"
            "if len(sys.argv) > 1 and sys.argv[1] == '--supervise': time.sleep(2)\n"
            "os.execv(sys.executable, [sys.executable, real] + sys.argv[1:])\n"
            % str(ROOT / "scripts/dual_fifo_result.py"))
        target_record = base / "target.pid"
        fake_carla = base / "external_paths/carla_root/CarlaUE4.sh"
        fake_carla.write_text(
            "#!/bin/bash\n"
            "start=$(awk '{print $22}' /proc/$$/stat)\n"
            "printf '%s %s\\n' \"$$\" \"$start\" > \"$DITL_TASK6_TEST_TARGET_PID_FILE\"\n"
            "exec sleep 30\n")
        fake_carla.chmod(0o700)
        config = base / "config.yaml"; config.write_text("simulation: {}\n")
        result = base / "result"; result.mkdir()
        processes = base / "processes"; processes.mkdir()
        logs = base / "logs"; logs.mkdir()
        command = [
            str(ROOT / "scripts/ditl_task6_eval_one.sh"),
            "--config", str(config), "--result-root", str(result),
            "--route", "146", "--scenario", "_1", "--repeat", "0",
            "--carla-seed", "2000", "--traffic-seed", "2000",
            "--render-gpu", "6", "--infer-gpu", "7",
            "--world-port", "46000", "--traffic-port", "46005",
            "--process-dir", str(processes), "--log-dir", str(logs),
        ]
        environment = dict(os.environ, DITL_TASK6_TEST_TARGET_PID_FILE=str(target_record))
        wrapper = subprocess.Popen(command, cwd=str(base), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        def stop_wrapper():
            if wrapper.poll() is None:
                wrapper.terminate()
                try:
                    wrapper.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    wrapper.kill()
                    wrapper.wait(timeout=3)
        self.addCleanup(stop_wrapper)
        child_pid = child_start = None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and child_pid is None:
            try:
                children = (Path("/proc") / str(wrapper.pid) / "task" /
                            str(wrapper.pid) / "children").read_text().split()
            except OSError:
                children = []
            for candidate in children:
                try:
                    current = self.runner.inspect_process(int(candidate))
                except (OSError, ValueError, RuntimeError):
                    continue
                if current["ppid"] == wrapper.pid and current["pgid"] == current["pid"] == current["sid"]:
                    child_pid, child_start = candidate, current["start_token"]
                    break
            time.sleep(.01)
        self.assertIsNotNone(child_pid, "did not observe pending owned supervisor")
        self.assertFalse((processes / "carla.identity.json").exists(),
                         "test must interrupt before supervisor identity publication")
        os.kill(wrapper.pid, signal.SIGTERM)
        _, stderr = wrapper.communicate(timeout=10)
        self.assertEqual(wrapper.returncode, 143, stderr)

        def still_same_process(pid, start):
            try:
                current = self.runner.inspect_process(int(pid))
                return current["state"] != "Z" and current["start_token"] == start
            except (OSError, ValueError, RuntimeError):
                return False

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and still_same_process(child_pid, child_start):
            time.sleep(.02)
        survivors = []
        if still_same_process(child_pid, child_start):
            survivors.append((child_pid, child_start))
        if target_record.exists():
            target_pid, target_start = target_record.read_text().split()
            if still_same_process(target_pid, target_start):
                survivors.append((target_pid, target_start))
        try:
            self.assertEqual(survivors, [], "pending supervisor/target survived wrapper interruption")
        finally:
            if still_same_process(child_pid, child_start):
                os.killpg(int(child_pid), signal.SIGKILL)

    def test_wrapper_claim_failure_stops_unpublished_owned_child(self):
        base = self.base / "claim-failure"
        (base / "scripts").mkdir(parents=True)
        (base / "external_paths/carla_root").mkdir(parents=True)
        supervisor_record = base / "supervisor.pid"
        delayed_supervisor = base / "scripts/dual_fifo_result.py"
        delayed_supervisor.write_text(
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "real = %r\n"
            "if len(sys.argv) > 1 and sys.argv[1] == '--supervise':\n"
            "    raw = (Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(') ', 1)[1].split()\n"
            "    Path(os.environ['DITL_TASK6_TEST_SUPERVISOR_PID_FILE']).write_text(str(os.getpid()) + ' ' + raw[19])\n"
            "    time.sleep(2)\n"
            "if len(sys.argv) > 1 and sys.argv[1] == '--await-owned': sys.argv[-1] = '.05'\n"
            "os.execv(sys.executable, [sys.executable, real] + sys.argv[1:])\n"
            % str(ROOT / "scripts/dual_fifo_result.py"))
        target_record = base / "target.pid"
        fake_carla = base / "external_paths/carla_root/CarlaUE4.sh"
        fake_carla.write_text(
            "#!/bin/bash\n"
            "start=$(awk '{print $22}' /proc/$$/stat)\n"
            "printf '%s %s\\n' \"$$\" \"$start\" > \"$DITL_TASK6_TEST_TARGET_PID_FILE\"\n"
            "exec sleep 30\n")
        fake_carla.chmod(0o700)
        config = base / "config.yaml"; config.write_text("simulation: {}\n")
        result = base / "result"; result.mkdir()
        processes = base / "processes"; processes.mkdir()
        logs = base / "logs"; logs.mkdir()
        command = [
            str(ROOT / "scripts/ditl_task6_eval_one.sh"),
            "--config", str(config), "--result-root", str(result),
            "--route", "146", "--scenario", "_1", "--repeat", "0",
            "--carla-seed", "2000", "--traffic-seed", "2000",
            "--render-gpu", "6", "--infer-gpu", "7",
            "--world-port", "46000", "--traffic-port", "46005",
            "--process-dir", str(processes), "--log-dir", str(logs),
        ]
        environment = dict(os.environ,
            DITL_TASK6_TEST_SUPERVISOR_PID_FILE=str(supervisor_record),
            DITL_TASK6_TEST_TARGET_PID_FILE=str(target_record))
        completed = subprocess.run(command, cwd=str(base), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True, timeout=10)
        child_pid, child_start = supervisor_record.read_text().split()

        def still_same_process(pid, start):
            try:
                current = self.runner.inspect_process(int(pid))
                return current["state"] != "Z" and current["start_token"] == start
            except (OSError, ValueError, RuntimeError):
                return False

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and still_same_process(child_pid, child_start):
            time.sleep(.02)
        survivors = []
        if still_same_process(child_pid, child_start):
            survivors.append((child_pid, child_start))
        if target_record.exists():
            target_pid, target_start = target_record.read_text().split()
            if still_same_process(target_pid, target_start):
                survivors.append((target_pid, target_start))
        try:
            self.assertEqual(completed.returncode, 3, completed.stderr)
            self.assertIn("timed out waiting for owned identity", completed.stderr)
            self.assertFalse((processes / "carla.identity.json").exists())
            self.assertEqual(survivors, [], "claim failure orphaned supervisor/target")
        finally:
            if still_same_process(child_pid, child_start):
                os.killpg(int(child_pid), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
