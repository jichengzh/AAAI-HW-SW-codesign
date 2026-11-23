"""Read-only seven-arm verification, using real parser and temporary artifacts."""

import contextlib
import copy
import csv
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from .test_dual_fifo_harness import ARMS, REPO, contract_fixture
from scripts import dual_fifo_result as gate


VERIFIER = REPO / "scripts/verify_dual_fifo_evidence.py"
SIM_FIELDS = ("source_sim_time_s", "consume_sim_time_s")


def strip_sim_times(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    fields = [key for key in rows[0] if key not in SIM_FIELDS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class EvidenceVerifierTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence.json"
        self.rows = []
        for rsu, ego, arm, rf, ef in ARMS:
            path, config, meta = contract_fixture(self.root / arm, rsu, ego)
            strip_sim_times(meta / "latency_align_audit.csv")
            self.rows.append({"arm": arm, "summary": {
                "route": "0", "repeat": "0", "git_commit": "d" * 40,
                "git_dirty": "false", "countable": "True", "sim_rate_hz": "20.0",
                "rsu_fifo_delay_ms": str(rsu), "ego_output_fifo_delay_ms": str(ego),
                "rsu_delay_frames": str(rf), "ego_delay_frames": str(ef),
                "DS": "0.0", "RC": "15.0", "status": "Failed",
                "coll_lay": "2.0", "coll_veh": "0.5", "agent_config": str(config),
                "ego_json": str(path), "rsu_audit_csv": str(meta / "latency_align_audit.csv"),
                "ego_output_audit_csv": str(meta / "ego_output_fifo_audit.csv"),
            }})

    def verifier(self):
        self.assertTrue(VERIFIER.is_file(), "read-only evidence verifier is missing")
        spec = importlib.util.spec_from_file_location("verify_dual_fifo_evidence", str(VERIFIER))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def invoke(self, rows):
        self.evidence.write_text(json.dumps(rows))
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = self.verifier().main([str(self.evidence)])
        return code, output.getvalue(), errors.getvalue()

    def assert_controlled_rejection(self, rows):
        try:
            code, output, error = self.invoke(rows)
        except Exception as error:
            self.fail("CLI leaked %s: %s" % (type(error).__name__, error))
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("evidence verification failed:", error)
        self.assertNotIn("Traceback", error)

    def test_rejects_symlink_audit_pair_from_different_meta_directories(self):
        summary = self.rows[0]["summary"]
        for key, run in (("rsu_audit_csv", "run_A"), ("ego_output_audit_csv", "run_B")):
            path = Path(summary[key])
            target = path.parents[2] / run / "meta" / path.name
            target.parent.mkdir(parents=True)
            path.rename(target)
            path.symlink_to(target)
        self.assert_controlled_rejection(self.rows)

    def test_rejects_route_json_symlink_with_wrong_canonical_tail(self):
        path = Path(self.rows[0]["summary"]["ego_json"])
        target = self.root / "different_route/results.json"
        target.parent.mkdir()
        path.rename(target)
        path.symlink_to(target)
        self.assert_controlled_rejection(self.rows)

    def test_rejects_route_json_symlink_to_different_experiment_root(self):
        path = Path(self.rows[0]["summary"]["ego_json"])
        target, _, _ = contract_fixture(self.root / "different_experiment")
        path.unlink()
        path.symlink_to(target)
        self.assert_controlled_rejection(self.rows)

    def test_rejects_audit_symlink_with_wrong_canonical_filename(self):
        path = Path(self.rows[0]["summary"]["rsu_audit_csv"])
        target = path.with_name("wrong.csv")
        path.rename(target)
        path.symlink_to(target)
        self.assert_controlled_rejection(self.rows)

    def test_whole_tree_alias_uses_canonical_paths_for_validation_and_reads(self):
        alias = self.root / "tree_alias"
        alias.symlink_to(self.root, target_is_directory=True)
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for key in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv"):
                row["summary"][key] = str(alias / Path(row["summary"][key]).relative_to(self.root))
        with mock.patch.object(gate, "_read_audit", wraps=gate._read_audit) as audit_read, \
                mock.patch.object(gate, "parse_route_result", wraps=gate.parse_route_result) as route_read:
            code, output, error = self.invoke(rows)
        self.assertEqual(code, 0, error)
        self.assertEqual(len(json.loads(output)), 7)
        for call in audit_read.call_args_list + route_read.call_args_list:
            self.assertEqual(str(call[0][0]), str(Path(call[0][0]).resolve()))

    def test_rejects_boolean_metrics_even_when_matching_zero_or_one(self):
        for key in ("DS", "RC", "coll_lay", "coll_veh"):
            for number in (0, 1):
                rows = copy.deepcopy(self.rows)
                path = Path(rows[0]["summary"]["ego_json"])
                payload = json.loads(path.read_text())
                record = payload["_checkpoint"]["global_record"]
                section = "scores" if key in ("DS", "RC") else "infractions"
                field = {"DS": "score_composed", "RC": "score_route",
                         "coll_lay": "collisions_layout", "coll_veh": "collisions_vehicle"}[key]
                record[section][field] = number
                path.write_text(json.dumps(payload))
                for metric, value in gate.parse_route_result(str(path)).items():
                    if metric in ("DS", "RC", "coll_lay", "coll_veh"):
                        rows[0]["summary"][metric] = str(value)
                with self.subTest(key=key, value=number):
                    self.assertEqual(self.invoke(rows)[0], 0, "numeric strings must remain accepted")
                    rows[0]["summary"][key] = bool(number)
                    self.assert_controlled_rejection(rows)

    def test_malformed_yaml_shapes_return_controlled_failure(self):
        original = Path(self.rows[0]["summary"]["agent_config"])
        config = self.root / original.name
        rows = copy.deepcopy(self.rows)
        rows[0]["summary"]["agent_config"] = str(config)
        for content in ("simulation: []", "simulation: null", "simulation: false", "[]", "null"):
            with self.subTest(content=content):
                config.write_text(content)
                self.assert_controlled_rejection(rows)

    def test_symlink_loop_returns_controlled_failure(self):
        for key in ("ego_json", "rsu_audit_csv"):
            with self.subTest(key=key):
                path = Path(self.rows[0]["summary"][key])
                content = path.read_bytes()
                path.unlink()
                path.symlink_to(path)
                try:
                    self.assert_controlled_rejection(self.rows)
                finally:
                    path.unlink()
                    path.write_bytes(content)

    def test_rejects_nonstandard_constants_and_nonfinite_extra_fields(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            rows = copy.deepcopy(self.rows)
            rows[0]["summary"]["extra"] = {"nested": [value]}
            with self.subTest(value=value):
                self.assert_controlled_rejection(rows)
        self.evidence.write_text(json.dumps(self.rows).replace('"DS": "0.0"', '"extra": 1e999, "DS": "0.0"', 1))
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            try:
                code = self.verifier().main([str(self.evidence)])
            except Exception as caught:
                self.fail("CLI leaked %s" % caught)
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("evidence verification failed:", error.getvalue())

    def test_output_serialization_failure_is_controlled_before_stdout(self):
        self.evidence.write_text(json.dumps(self.rows))
        module = self.verifier()
        for failure in (ValueError("invalid output"), TypeError("not serializable")):
            original_dumps = json.dumps

            def serialize(value, **kwargs):
                if isinstance(value, list):
                    raise failure
                return original_dumps(value, **kwargs)

            output, error = io.StringIO(), io.StringIO()
            with mock.patch.object(module.json, "dumps", side_effect=serialize), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                try:
                    code = module.main([str(self.evidence)])
                except Exception as caught:
                    self.fail("CLI leaked %s" % caught)
            self.assertEqual(code, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("evidence verification failed:", error.getvalue())

    def test_accepts_exact_seven_legacy_arms_without_writes_or_glob(self):
        self.evidence.write_text(json.dumps(self.rows))
        paths = [self.evidence] + [Path(row["summary"][key]) for row in self.rows
                                   for key in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv")]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        output = io.StringIO()
        with mock.patch.object(Path, "glob", side_effect=AssertionError("must not glob")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(self.verifier().main([str(self.evidence)]), 0)
        records = json.loads(output.getvalue())
        self.assertEqual([row["arm"] for row in records], [arm[2] for arm in ARMS])
        for row in records:
            self.assertEqual(row["rsu_schema"], "legacy_15")
            self.assertEqual(row["ego_schema"], "v2_20")
            self.assertEqual(row["missing_sim_time_columns"], list(SIM_FIELDS))
            self.assertEqual(row["sim_time_validation"], "unavailable_in_legacy_schema")
            self.assertEqual(row["assertions"], "PASS")
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_rejects_missing_duplicate_unknown_and_extra_arms(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1],
                     self.rows[:-1] + self.rows[:1],
                     [dict(self.rows[0], arm="unknown")] + self.rows[1:]):
            with self.subTest(arms=[row["arm"] for row in rows]):
                code, _, error = self.invoke(rows)
                self.assertNotEqual(code, 0)
                self.assertIn("arm", error)

    def test_rejects_provenance_delay_config_countability_and_metric_mismatches(self):
        changes = (("git_commit", "a" * 40), ("git_commit", ""),
                   ("git_dirty", "true"), ("git_dirty", "False"), ("git_dirty", 0),
                   ("route", "1"), ("repeat", "1"), ("route", ""),
                   ("rsu_fifo_delay_ms", "500"), ("ego_output_fifo_delay_ms", "200"),
                   ("rsu_delay_frames", "10"), ("ego_delay_frames", "4"),
                   ("sim_rate_hz", "5"), ("countable", "False"), ("countable", 1),
                   ("agent_config", self.rows[2]["summary"]["agent_config"]),
                   ("ego_json", "relative/results.json"),
                   ("rsu_audit_csv", self.rows[2]["summary"]["rsu_audit_csv"]),
                   ("ego_output_audit_csv", "relative.csv"),
                   ("DS", "999"), ("RC", "nan"), ("status", "Completed"))
        for key, value in changes:
            rows = copy.deepcopy(self.rows)
            rows[1]["summary"][key] = value
            with self.subTest(key=key, value=value):
                code, _, error = self.invoke(rows)
                self.assertNotEqual(code, 0)
                self.assertTrue(error)

    def test_malformed_or_missing_evidence_fails_with_diagnostic(self):
        for payload in ({}, [None] * 7, [dict(row, summary=None) for row in self.rows]):
            self.assertNotEqual(self.invoke(payload)[0], 0)
        self.evidence.write_text("{bad")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertNotEqual(self.verifier().main([str(self.evidence)]), 0)
            self.assertNotEqual(self.verifier().main([str(self.root / "missing.json")]), 0)

    def test_cli_failure_returns_nonzero(self):
        self.verifier()
        self.evidence.write_text("[]")
        result = subprocess.run([sys.executable, str(VERIFIER), str(self.evidence)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"arm", result.stderr)

    def test_current_schema_is_validated_and_labeled(self):
        from .test_dual_fifo_harness import write_fixture_audits
        for (rsu, ego, _, _, _), row in zip(ARMS, self.rows):
            write_fixture_audits(Path(row["summary"]["rsu_audit_csv"]).parent, rsu, ego)
        code, output, error = self.invoke(self.rows)
        self.assertEqual(code, 0, error)
        for row in json.loads(output):
            self.assertEqual(row["rsu_schema"], "current_17")
            self.assertEqual(row["missing_sim_time_columns"], [])
            self.assertEqual(row["sim_time_validation"], "PASS")


class ExplicitLegacyGateTest(unittest.TestCase):
    def setUp(self):
        parameters = inspect.signature(gate.validate_audits).parameters
        self.assertIn("allow_legacy_rsu_schema", parameters)
        self.assertIn("audit_paths", parameters)

    def test_legacy_opt_in_rejects_all_non_boolean_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config, meta = contract_fixture(tmp)
            strip_sim_times(meta / "latency_align_audit.csv")
            args = (str(path), "0", "0", 0, 0, str(config))
            for value in ("false", "true", 0, 1, None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    gate.validate_audits(*args, allow_legacy_rsu_schema=value)

    def test_default_rejects_legacy_but_opt_in_validates_existing_contracts(self):
        for rsu, ego, arm, _, _ in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                path, config, meta = contract_fixture(tmp, rsu, ego)
                audit = meta / "latency_align_audit.csv"
                strip_sim_times(audit)
                args = (str(path), "0", "0", rsu, ego, str(config))
                with self.assertRaisesRegex(ValueError, "headers"):
                    gate.validate_audits(*args)
                self.assertIn(str(audit), gate.validate_audits(*args, allow_legacy_rsu_schema=True))
                audit.write_text(audit.read_text().replace("injected_from_E7", "invalid_source"))
                with self.assertRaisesRegex(ValueError, "latency_ms_source"):
                    gate.validate_audits(*args, allow_legacy_rsu_schema=True)

    def test_explicit_audit_paths_do_not_glob_and_must_match_route_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config, meta = contract_fixture(tmp)
            paths = (str(meta / "latency_align_audit.csv"), str(meta / "ego_output_fifo_audit.csv"))
            args = (str(path), "0", "0", 0, 0, str(config))
            self.assertEqual(gate.validate_audits(*args), paths)
            with mock.patch.object(Path, "glob", side_effect=AssertionError("must not glob")):
                self.assertEqual(gate.validate_audits(*args, audit_paths=paths), paths)
                for bad in ((paths[0], "relative.csv"), (paths[0], paths[0]),
                            (paths[0].replace("image/", "other/"), paths[1]),
                            (paths[0].replace("ego0/", "other/"), paths[1]), (), "bad"):
                    with self.subTest(paths=bad), self.assertRaises(ValueError):
                        gate.validate_audits(*args, audit_paths=bad)

    def test_legacy_opt_in_does_not_ignore_partial_or_invalid_sim_time_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config, meta = contract_fixture(tmp)
            args = (str(path), "0", "0", 0, 0, str(config))
            audit = meta / "latency_align_audit.csv"
            with audit.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[-1]["source_sim_time_s"] = "99"
            for fields in (list(rows[0]), [key for key in rows[0] if key != "source_sim_time_s"]):
                with audit.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    gate.validate_audits(*args, allow_legacy_rsu_schema=True)


if __name__ == "__main__":
    unittest.main()
