"""CPU-only contracts for strict Task 6 result aggregation."""
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[4]


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


result = load_script("ditl_task6_result")
task6 = load_script("ditl_task6_manifest")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + "\n")


class CanonicalCohort(object):
    def __init__(self, root, ds_values=None, mechanism_values=None):
        self.root = Path(root)
        self.root.mkdir()
        for name in ("configs", "results", "status", "library"):
            (self.root / name).mkdir()
        self.ds_values = ds_values or {
            "clean": 80.0,
            "comm_drop_005": 79.6, "comm_drop_010": 78.0, "comm_drop_020": 72.0,
            "spatial_dx_05": 79.5, "spatial_dx_10": 76.0, "spatial_dx_20": 68.0,
            "compute_gpu_low": 79.0, "compute_gpu_medium": 75.0, "compute_gpu_high": 64.0,
        }
        defaults = {
            "communication": ((1, 19), (2, 18), (4, 16)),
            "compute_aoi_ns": (120000000, 180000000, 260000000),
            "gpu_ns": {"raw_samehost_idle": 50000000,
                       "raw_samehost_gpu_low": 70000000,
                       "raw_samehost_gpu_medium": 100000000,
                       "raw_samehost_gpu_high": 160000000},
        }
        self.mechanism = mechanism_values or defaults
        self.bundles = self._bundles()
        self.manifest = self._cohort()

    def _bundles(self):
        bundles = {}
        for profile, gpu_ns in self.mechanism["gpu_ns"].items():
            feature_id = hashlib.sha256(profile.encode("ascii")).hexdigest()
            bundle = self.root / "library" / feature_id
            (bundle / "raw").mkdir(parents=True)
            (bundle / "provenance").mkdir()
            trace = []
            for index, value in enumerate((gpu_ns // 2, gpu_ns, gpu_ns + 1000000)):
                trace.append({"warmup": index == 0, "status": {"outcome": "consumed"},
                              "durations_ns": {"gpu_inference": value}})
            (bundle / "raw" / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in trace))
            write_json(bundle / "provenance" / "command_manifest.txt",
                       {"source_commit": "b" * 40})
            write_json(bundle / "manifest.json",
                       {"feature_id": feature_id, "profile_id": profile,
                        "accepted_count": 2,
                        "source_trace_sha256": digest(bundle / "raw" / "trace.jsonl")})
            bundles[profile] = (bundle, digest(bundle / "manifest.json"))
        return bundles

    def _cohort(self):
        rows = []
        expected_arms = task6.build_task6_arms()
        family_index = {"communication": 0, "spatial": 0, "compute": 0}
        for sequence, arm in enumerate(expected_arms):
            bundle, bundle_sha = self.bundles[arm.trace_profile_id]
            config_path = self.root / "configs" / ("%02d_%s.yaml" % (sequence, arm.arm_id))
            impairment = {
                "extra_delay_ms": 0, "jitter_ms": 0, "drop_rate": arm.drop_rate,
                "burst": {"enabled": False, "enter_rate": 0.0, "exit_rate": 1.0},
                "reorder_rate": 0.0, "reorder_extra_ms": 0, "duplicate_rate": 0.0,
            }
            replay = {"bundle": str(bundle), "seed": 2000, "impairment": impairment}
            simulation = {"ditl_replay": replay}
            if arm.family == "spatial":
                simulation["ditl_spatial_offset"] = {
                    "dx_m": arm.spatial_dx_m, "dy_m": 0.0, "dyaw_deg": 0.0}
            config_path.write_text(yaml.safe_dump({"simulation": simulation}))
            result_root = self.root / "results" / arm.arm_id
            episode = (result_root / "image" / "v2x_final" / "town05_short_collab" /
                       "r146_repeat0" / ("town05_short_r146_%02d" % sequence))
            (episode / "meta").mkdir(parents=True)
            result_path = (result_root / "v2x_final" / "town05_short_collab" /
                           "r146_repeat0" / "ego_vehicle_0" / "results.json")
            result_path.parent.mkdir(parents=True)
            self._route_result(result_path, self.ds_values[arm.arm_id])
            accepted, dropped = 20, 0
            aoi = 100000000
            if arm.family == "communication":
                index = family_index[arm.family]
                dropped, accepted = self.mechanism["communication"][index]
                aoi = (140000000, 190000000, 280000000)[index]
            elif arm.family == "compute":
                index = family_index[arm.family]
                aoi = self.mechanism["compute_aoi_ns"][index]
            self._event_audit(episode / "meta" / "ditl_event_audit.jsonl",
                              config_path, episode.name, accepted, dropped, aoi)
            if arm.family == "spatial":
                self._spatial_audit(episode / "meta" / "ditl_spatial_audit.jsonl",
                                    arm.spatial_dx_m)
            write_json(self.root / "status" / (arm.arm_id + ".json"), {
                "schema": "ditl.task6-arm-status.v1", "arm_id": arm.arm_id,
                "status": "completed", "returncode": 0, "cleanup_error": None,
                "execution_error": None, "postflight_error": None})
            rows.append({
                "sequence": sequence, "arm_id": arm.arm_id,
                "contract": dict(vars(arm)), "config_path": str(config_path),
                "config_sha256": digest(config_path),
                "bundle_feature_id": bundle.name, "bundle_path": str(bundle),
                "bundle_manifest_sha256": bundle_sha,
                "route_file": "/frozen/town05_short_r146.xml",
                "route_file_sha256": "c" * 64,
                "result_root": str(result_root),
            })
            if arm.family in family_index:
                family_index[arm.family] += 1
        manifest = {
            "schema": "ditl.task6-cohort.v1", "source_commit": "a" * 40,
            "route": {"id": 146, "scenario_suffix": "_1", "repeat": 0,
                      "path": "/frozen/town05_short_r146.xml", "sha256": "c" * 64},
            "arms": rows,
        }
        write_json(self.root / "manifest.json", manifest)
        return manifest

    @staticmethod
    def _route_result(path, ds):
        record = {
            "status": "Completed", "scores": {"score_composed": ds, "score_route": 95.0},
            "infractions": {"collisions_layout": 0.0, "collisions_vehicle": 0.0,
                            "route_timeout": 0.0}, "meta": {}, "route_id": "RouteScenario_146",
        }
        global_record = copy.deepcopy(record)
        global_record["route_id"] = -1
        write_json(path, {"_checkpoint": {"global_record": global_record,
                                          "records": [copy.deepcopy(record)]}})

    @staticmethod
    def _policy_hash(config_path):
        impairment = yaml.safe_load(Path(config_path).read_text())["simulation"]["ditl_replay"]["impairment"]
        canonical = {
            "extra_delay_ns": int(impairment["extra_delay_ms"] * 1000000),
            "jitter_ns": int(impairment["jitter_ms"] * 1000000),
            "drop_rate": float(impairment["drop_rate"]),
            "burst": impairment["burst"], "reorder_rate": float(impairment["reorder_rate"]),
            "reorder_extra_ns": int(impairment["reorder_extra_ms"] * 1000000),
            "duplicate_rate": float(impairment["duplicate_rate"]),
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def _event_audit(self, path, config, run_id, accepted, dropped, aoi):
        records = []
        outcomes = [False] * accepted + [True] * dropped
        for index, is_dropped in enumerate(outcomes):
            dropped_so_far = sum(outcomes[:index + 1])
            records.append({
                "schema_version": "ditl.event-audit.v1", "run_id": run_id,
                "route_id": "town05_short_r146", "stream_id": "rsu-1",
                "policy_config_hash": self._policy_hash(config),
                "accepted_count": index + 1 - dropped_so_far,
                "dropped_count": dropped_so_far, "expired_count": 0,
                "aoi_ns": None if is_dropped else aoi + index * 1000000,
                "drop_reason": "bernoulli" if is_dropped else None,
            })
        path.write_text("".join(json.dumps(row) + "\n" for row in records))

    @staticmethod
    def _spatial_audit(path, dx):
        before = [10.0, 20.0, 1.0, 0.0, 30.0, 0.0]
        after = [10.0 + dx, 20.0, 1.0, 0.0, 30.0, 0.0]
        row = {
            "schema": "ditl.spatial-audit.v1", "route": "town05_short_r146", "frame": 10,
            "applied_rsu_count": 1,
            "configured_ego_frame_offset": {"dx_m": dx, "dy_m": 0.0, "dyaw_deg": 0.0},
            "world_delta_m": {"dx_m": dx, "dy_m": 0.0},
            "rsus": [{"cav_id": "rsu_1", "dx_m": dx, "dy_m": 0.0, "dyaw_deg": 0.0,
                      "world_dx_m": dx, "world_dy_m": 0.0,
                      "lidar_pose_before": before, "lidar_pose_after": after,
                      "map_pose_before": before, "map_pose_after": after}],
        }
        path.write_text(json.dumps(row) + "\n")


class Task6ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cohort = CanonicalCohort(Path(self.temp.name) / "task6-fixture")

    def parse(self):
        return result.parse_cohort(self.cohort.root)

    def test_aggregates_final_counters_nearest_rank_p95_and_rd(self):
        report = self.parse()
        self.assertEqual([row["arm_id"] for row in report["rows"]], list(task6.ARM_IDS))
        comm = report["rows"][1]
        self.assertAlmostEqual(comm["actual_drop_rate"], 1.0 / 20.0)
        self.assertEqual(comm["drop_reason_counts"], {"bernoulli": 1})
        self.assertAlmostEqual(report["rows"][3]["actual_drop_rate"], 4.0 / 20.0,
                               msg="use the final cumulative counter, never its repeated sum")
        # 19 accepted samples, nearest-rank p95 is ordered item ceil(.95*19)=19.
        self.assertEqual(comm["aoi_p95_ms"], 158.0)
        self.assertEqual(report["rows"][0]["gpu_inference_p95_ms"], 51.0)
        self.assertAlmostEqual(comm["RD"], 1.0 - 79.6 / 80.0)
        self.assertEqual(comm["RC"], 95.0)
        self.assertEqual(comm["status"], "Completed")
        self.assertEqual(comm["collision_markers"], {"layout": 0.0, "vehicle": 0.0})
        self.assertEqual(comm["configured_severity_value"], 0.05)
        self.assertEqual(comm["actual_severity_value"], comm["actual_drop_rate"])
        for field in ("config_sha256", "event_audit_sha256", "result_sha256",
                      "status_sha256", "bundle_manifest_sha256", "bundle_trace_sha256"):
            self.assertRegex(comm[field], r"^[0-9a-f]{64}$")

    def test_event_counters_allow_delivery_lag_and_simultaneous_drop(self):
        audit = next((self.cohort.root / "results" / "comm_drop_005").glob(
            "**/ditl_event_audit.jsonl"))
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        records[0]["accepted_count"] = 0
        records[0]["dropped_count"] = 0
        records[0]["aoi_ns"] = None
        records[1]["accepted_count"] = 1
        records[1]["dropped_count"] = 1
        records[1]["drop_reason"] = "bernoulli"
        for index in range(2, len(records)):
            records[index]["accepted_count"] = index - 1
            records[index]["dropped_count"] = 1
            records[index]["drop_reason"] = None
        audit.write_text("".join(json.dumps(row) + "\n" for row in records))
        report = self.parse()
        comm = report["rows"][1]
        self.assertEqual(comm["accepted_count"], 18)
        self.assertEqual(comm["dropped_count"], 1)
        self.assertAlmostEqual(comm["actual_drop_rate"], 1.0 / 19.0)
        self.assertEqual(comm["drop_reason_counts"], {"bernoulli": 1})

    def test_spatial_actual_offset_comes_from_audit_and_preserves_clean_components(self):
        report = self.parse()
        rows = [row for row in report["rows"] if row["family"] == "spatial"]
        self.assertEqual([row["actual_input_offset_m"] for row in rows], [0.5, 1.0, 2.0])
        self.assertEqual(report["families"]["spatial"]["mechanism_axis_label"], "input offset (m)")
        self.assertNotIn("target", report["families"]["spatial"]["mechanism_axis_label"])

    def test_classifies_sensitive_insensitive_and_invalid_in_that_order(self):
        report = self.parse()
        self.assertEqual(report["families"]["communication"]["classification"], "sensitive")
        flat = dict(self.cohort.ds_values)
        flat.update({arm: 79.6 for arm in task6.ARM_IDS if arm != "clean"})
        insensitive = CanonicalCohort(Path(self.temp.name) / "flat", ds_values=flat)
        parsed = result.parse_cohort(insensitive.root)
        self.assertTrue(all(value["classification"] == "insensitive_at_current_route_and_severity"
                            for value in parsed["families"].values()))
        broken_values = copy.deepcopy(self.cohort.mechanism)
        broken_values["communication"] = ((1, 19), (4, 16), (2, 18))
        broken = CanonicalCohort(Path(self.temp.name) / "broken", mechanism_values=broken_values)
        invalid = result.parse_cohort(broken.root)
        self.assertEqual(invalid["families"]["communication"]["classification"],
                         "mechanism_invalid")

    def test_mechanism_trend_includes_shared_clean_baseline(self):
        broken_values = copy.deepcopy(self.cohort.mechanism)
        broken_values["gpu_ns"] = {
            "raw_samehost_idle": 90000000,
            "raw_samehost_gpu_low": 70000000,
            "raw_samehost_gpu_medium": 100000000,
            "raw_samehost_gpu_high": 160000000,
        }
        broken = CanonicalCohort(Path(self.temp.name) / "broken-compute-clean",
                                 mechanism_values=broken_values)
        parsed = result.parse_cohort(broken.root)
        self.assertEqual(parsed["families"]["compute"]["classification"],
                         "mechanism_invalid")

    def test_rejects_missing_or_duplicate_clean_and_duplicate_arm_result(self):
        manifest_path = self.cohort.root / "manifest.json"
        for mutation in ("missing", "duplicate"):
            original = copy.deepcopy(self.cohort.manifest)
            if mutation == "missing":
                original["arms"] = original["arms"][1:]
            else:
                original["arms"][1] = copy.deepcopy(original["arms"][0])
            write_json(manifest_path, original)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.parse()
        write_json(manifest_path, self.cohort.manifest)
        result_path = next((self.cohort.root / "results" / "clean").glob("**/results.json"))
        duplicate = result_path.parents[1] / "ego_vehicle_1" / "results.json"
        duplicate.parent.mkdir()
        shutil.copyfile(str(result_path), str(duplicate))
        with self.assertRaisesRegex(ValueError, "result"):
            self.parse()

    def test_rejects_nonpositive_clean_ds_and_nonterminal_result(self):
        clean_result = next((self.cohort.root / "results" / "clean").glob("**/results.json"))
        parsed = json.loads(clean_result.read_text())
        parsed["_checkpoint"]["global_record"]["scores"]["score_composed"] = 0.0
        write_json(clean_result, parsed)
        with self.assertRaisesRegex(ValueError, "clean DS"):
            self.parse()
        parsed["_checkpoint"]["global_record"]["scores"]["score_composed"] = 80.0
        parsed["_checkpoint"]["global_record"]["status"] = "Started"
        write_json(clean_result, parsed)
        with self.assertRaisesRegex(ValueError, "terminal|countable"):
            self.parse()

    def test_rejects_global_record_route_identity_mismatch(self):
        clean_result = next((self.cohort.root / "results" / "clean").glob(
            "**/ego_vehicle_0/results.json"))
        document = json.loads(clean_result.read_text())
        self.assertEqual(document["_checkpoint"]["records"][0]["route_id"],
                         "RouteScenario_146")
        self.assertEqual(document["_checkpoint"]["global_record"]["route_id"], -1)
        document["_checkpoint"]["global_record"]["route_id"] = "RouteScenario_999"
        write_json(clean_result, document)

        with self.assertRaisesRegex(ValueError, "route identity mismatch"):
            self.parse()

    def test_rejects_missing_or_conflicting_consumed_route_identities(self):
        for issue in ("missing_global", "missing_record", "later_record", "exception"):
            fixture = CanonicalCohort(Path(self.temp.name) / ("route-" + issue))
            clean_result = next((fixture.root / "results" / "clean").glob(
                "**/ego_vehicle_0/results.json"))
            document = json.loads(clean_result.read_text())
            checkpoint = document["_checkpoint"]
            if issue == "missing_global":
                del checkpoint["global_record"]["route_id"]
            elif issue == "missing_record":
                del checkpoint["records"][0]["route_id"]
            elif issue == "later_record":
                later = copy.deepcopy(checkpoint["records"][0])
                later["route_id"] = "RouteScenario_999"
                checkpoint["records"].append(later)
            else:
                checkpoint["global_record"]["meta"]["exceptions"] = [
                    ["RouteScenario_999", 0, "Failed"]]
            write_json(clean_result, document)

            with self.subTest(issue=issue), self.assertRaisesRegex(
                    ValueError, "route identity"):
                result.parse_cohort(fixture.root)

    def test_rejects_truncated_jsonl_and_missing_final_counters(self):
        audit = next((self.cohort.root / "results" / "clean").glob("**/ditl_event_audit.jsonl"))
        original = audit.read_text()
        audit.write_text(original.rstrip("\n"))
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.parse()
        audit.write_text(original)
        rows = [json.loads(line) for line in original.splitlines()]
        del rows[-1]["accepted_count"]
        audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "accepted_count|counter"):
            self.parse()

    def test_rejects_route_arm_config_hash_and_bundle_identity_mismatches(self):
        mutations = []
        mutations.append(("route", lambda: self.cohort.manifest["route"].__setitem__("id", 145)))
        mutations.append(("arm", lambda: json.loads((self.cohort.root / "status/clean.json").read_text()).update()))
        for kind in ("route", "config", "bundle", "arm"):
            fixture = CanonicalCohort(Path(self.temp.name) / ("bad-" + kind))
            if kind == "route":
                audit = next((fixture.root / "results/clean").glob("**/ditl_event_audit.jsonl"))
                records = [json.loads(line) for line in audit.read_text().splitlines()]
                records[0]["route_id"] = "town05_short_r999"
                audit.write_text("".join(json.dumps(row) + "\n" for row in records))
            elif kind == "config":
                config = Path(fixture.manifest["arms"][0]["config_path"])
                config.write_text(config.read_text() + "# changed\n")
            elif kind == "bundle":
                fixture.manifest["arms"][0]["bundle_feature_id"] = "f" * 64
                write_json(fixture.root / "manifest.json", fixture.manifest)
            else:
                status = fixture.root / "status/clean.json"
                value = json.loads(status.read_text()); value["arm_id"] = "comm_drop_005"
                write_json(status, value)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, kind):
                result.parse_cohort(fixture.root)

    def test_rejects_nonfinite_metrics_and_duplicate_json_keys(self):
        audit = next((self.cohort.root / "results/clean").glob("**/ditl_event_audit.jsonl"))
        audit.write_text(audit.read_text().replace("100000000", "NaN", 1))
        with self.assertRaisesRegex(ValueError, "finite|constant"):
            self.parse()
        fixture = CanonicalCohort(Path(self.temp.name) / "duplicate-key")
        audit = next((fixture.root / "results/clean").glob("**/ditl_event_audit.jsonl"))
        first, rest = audit.read_text().split("\n", 1)
        first = first[:-1] + ',"accepted_count":99}'
        audit.write_text(first + "\n" + rest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            result.parse_cohort(fixture.root)

    def test_parse_is_pure_until_all_ten_arms_validate(self):
        bad = self.cohort.root / "status" / "compute_gpu_high.json"
        value = json.loads(bad.read_text()); value["status"] = "failed"; write_json(bad, value)
        with self.assertRaises(ValueError):
            result.generate_artifacts(self.cohort.root, self.cohort.root)
        for name in result.OUTPUT_NAMES:
            self.assertFalse((self.cohort.root / name).exists())

    def test_writes_canonical_outputs_fixed_order_and_dual_axis_plots(self):
        report = result.generate_artifacts(self.cohort.root, self.cohort.root)
        self.assertEqual([row["arm_id"] for row in report["rows"]], list(task6.ARM_IDS))
        for name in result.OUTPUT_NAMES:
            self.assertTrue((self.cohort.root / name).is_file(), name)
        with (self.cohort.root / "task6_results.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["arm_id"] for row in rows], list(task6.ARM_IDS))
        acceptance = json.loads((self.cohort.root / "task6_acceptance.json").read_text())
        self.assertTrue(acceptance["all_results_valid"])
        self.assertEqual(acceptance["validated_arm_count"], 10)
        self.assertEqual(acceptance["spatial_metric_scope"], "input_offset_only")
        for name in ("communication_trend.png", "spatial_trend.png", "compute_trend.png"):
            self.assertEqual((self.cohort.root / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
