"""Behavioral dual-FIFO contracts; external CARLA/GPU boundaries are faked."""

import csv
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import inspect
import signal
import socket
import time
from unittest import mock
import unittest


REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts/dual_fifo_multiroute_sc.sh"
RESULT_PARSER = REPO / "scripts/dual_fifo_result.py"
CONFIG_DIR = Path("simulation/leaderboard/team_code/agent_config")
FIELDS = (
    "route,repeat,rsu_fifo_delay_ms,rsu_delay_frames,"
    "ego_output_fifo_delay_ms,ego_delay_frames,DS,RC,status,countable,"
    "coll_lay,coll_veh,ego_json,rsu_audit_csv,ego_output_audit_csv,"
    "agent_config,sim_rate_hz,git_commit,git_dirty"
).split(",")
ARMS = ((0, 0, "baseline", 0, 0), (200, 0, "rsu200", 4, 0),
        (500, 0, "rsu500", 10, 0), (800, 0, "rsu800", 16, 0),
        (0, 200, "ego200", 0, 4), (0, 500, "ego500", 0, 10),
        (0, 800, "ego800", 0, 16))
JOINT_ARMS = ((100, 200), (100, 300), (200, 300),
              (100, 400), (200, 400), (300, 400),
              (100, 500), (200, 500), (300, 500), (400, 500))


def run_command(args, cwd=None, extra_env=None):
    return subprocess.run(
        [str(arg) for arg in args], cwd=str(cwd or "/tmp"),
        env=dict(dict(os.environ, DUAL_FIFO_PYTHON=sys.executable), **(extra_env or {})),
        universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=30,
    )


def result_fixture(status="Failed", ds=0.0):
    return {"_checkpoint": {"global_record": {
        "status": status, "scores": {"score_route": 15.0, "score_composed": ds},
        "infractions": {"collisions_layout": ["a", "b"], "collisions_vehicle": 0.5},
    }}}


def invalid_score_fixtures():
    record = result_fixture()["_checkpoint"]["global_record"]
    without_scores = {key: value for key, value in record.items() if key != "scores"}
    yield "missing_scores", {"_checkpoint": {"global_record": without_scores}}
    for scores in (None, [], "", False):
        yield "scores_%r" % scores, {"_checkpoint": {"global_record": dict(record, scores=scores)}}
    for field in ("score_composed", "score_route"):
        other_scores = {key: value for key, value in record["scores"].items() if key != field}
        yield "missing_" + field, {"_checkpoint": {"global_record": dict(record, scores=other_scores)}}
        for bad in (None, [], {}, "", "0", False, True, float("nan"), float("inf")):
            scores = dict(record["scores"], **{field: bad})
            yield "%s_%r" % (field, bad), {"_checkpoint": {"global_record": dict(record, scores=scores)}}


def write_fixture_audits(meta, rsu_ms, ego_ms, fault="", skip_frames=4):
    """Independent, hand-computed fixed-delay fixture on a 24-tick route."""
    import csv
    from pathlib import Path
    meta = Path(meta)
    meta.mkdir(parents=True, exist_ok=True)
    rsu, ego, previous = [], [], None
    calls = [f for f in range(24) if f <= skip_frames or f % skip_frames == 0]
    for f in calls:
        ready = [s for s in calls if s <= f and s + rsu_ms // 50 <= f]
        source = max(ready) if ready else None
        used_ready = None if source is None else source + rsu_ms // 50
        age = None if source is None else f - source
        rsu.append(dict(consume_frame=f, submitted_rsu_source_frame=f,
                        submitted_rsu_ready_frame=f + rsu_ms // 50,
                        used_rsu_source_frame=source, used_rsu_ready_frame=used_ready,
                        source_sim_time_s=None if source is None else source / 20.0,
                        consume_sim_time_s=f / 20.0,
                        configured_delay_ms=None if source is None else rsu_ms,
                        delay_frames=None if source is None else rsu_ms // 50,
                        information_age_frames=age, is_zoh_held=source == previous,
                        is_ego_only_startup=source is None, step=f,
                        delta=None if source is None else rsu_ms // 50,
                        used_rsu_frame_id=source, latency_ms_source="" if source is None else "injected_from_E7",
                        zoh_age_frames=None if source is None else f - used_ready))
        previous = source
    for f in range(24):
        source = f - ego_ms // 50 if f >= ego_ms // 50 else None
        ego.append(dict(apply_frame=f, generated_source_frame=f,
                        generated_ready_frame=f + ego_ms // 50, applied_source_frame=source,
                        applied_ready_frame=None if source is None else f, configured_delay_ms=ego_ms,
                        delay_frames=ego_ms // 50, control_age_frames=None if source is None else ego_ms // 50,
                        is_startup_neutral=source is None, generated_throttle=f / 100., generated_steer=-f / 100.,
                        generated_brake=0., applied_throttle=0. if source is None else source / 100.,
                        applied_steer=0. if source is None else -source / 100., applied_brake=0., step=f,
                        source_step=source, delay_ms=ego_ms, age_frames=None if source is None else ego_ms // 50,
                        is_startup_hold=source is None))
    mutations = {"rsu_delay": (rsu, -1, "configured_delay_ms", 800),
                 "rsu_ready": (rsu, 0, "submitted_rsu_ready_frame", 999),
                 "rsu_used": (rsu, -1, "used_rsu_source_frame", 999),
                 "rsu_age": (rsu, -1, "information_age_frames", 999),
                 "rsu_zoh": (rsu, -1, "is_zoh_held", True),
                 "rsu_startup": (rsu, 0, "is_ego_only_startup", False),
                 "ego_ready": (ego, -1, "generated_ready_frame", 999),
                 "ego_source": (ego, -1, "applied_source_frame", 999),
                 "ego_age": (ego, -1, "control_age_frames", 999),
                 "ego_startup": (ego, 0, "is_startup_neutral", False),
                 "ego_neutral": (ego, 0, "applied_brake", 1.),
                 "ego_control": (ego, -1, "applied_throttle", .99)}
    if fault in mutations:
        rows, index, key, value = mutations[fault]
        rows[index][key] = value
    if fault == "ego_gap":
        del ego[5]
    if fault == "rsu_gap":
        del rsu[-1]
    for name, rows in (("latency_align_audit.csv", rsu), ("ego_output_fifo_audit.csv", ego)):
        path = meta / name
        if fault == "missing_rsu" and rows is rsu or fault == "missing_ego" and rows is ego:
            continue
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if rows is rsu and fault in ("empty", "header", "truncated", "malformed", "ambiguous"):
            content = path.read_text()
            if fault == "ambiguous":
                other = meta.parent.parent / "other_run/meta"
                other.mkdir(parents=True)
                (other / name).write_text(content)
            else:
                path.write_text({"empty": "", "header": content.splitlines()[0] + "\n",
                                 "truncated": content[:-12], "malformed": "bad,header\n1,2\n"}[fault])


def contract_fixture(tmp, rsu=0, ego=0, route="0", repeat="0", joint_c2=False):
    arm = ("jointc2_rsu%d_ego%d" % (rsu, ego) if joint_c2 else
           "rsu%d" % rsu if rsu else "ego%d" % ego if ego else "baseline")
    config = REPO / CONFIG_DIR / ("pnp_config_codriving_dualfifo_%s_l1.yaml" % arm)
    setting = Path("v2x_final/town05_short_collab/r%s_repeat%s" % (route, repeat))
    root = Path(tmp) / "results_driving_trial"
    path, meta = root / setting / "ego_vehicle_0/results.json", root / "image" / setting / "ego0/meta"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result_fixture()))
    write_fixture_audits(meta, rsu, ego, skip_frames=2 if joint_c2 else 4)
    return path, config, meta


class DualFifoHarnessTest(unittest.TestCase):
    def parser(self):
        self.assertTrue(RESULT_PARSER.is_file(), "dual_fifo_result.py is missing")
        spec = importlib.util.spec_from_file_location("dual_fifo_result", str(RESULT_PARSER))
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        return parser

    def test_check_config_accepts_seven_single_variable_arms(self):
        for rsu, ego, arm, rsu_frames, ego_frames in ARMS:
            with self.subTest(arm=arm):
                result = run_command(["bash", SCRIPT, "--check-config", rsu, ego])
                self.assertEqual(result.returncode, 0, result.stdout)
                for token in ("rsu_delay_ms=%s" % rsu, "ego_delay_ms=%s" % ego,
                              "rsu_delay_frames=%s" % rsu_frames,
                              "ego_delay_frames=%s" % ego_frames,
                              "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm):
                    self.assertIn(token, result.stdout)

    def test_joint_check_config_accepts_exact_triangular_arm_set(self):
        for rsu, ego in JOINT_ARMS:
            with self.subTest(rsu=rsu, ego=ego):
                result = run_command(["bash", SCRIPT, "--joint", "--check-config", rsu, ego])
                self.assertEqual(result.returncode, 0, result.stdout)
                for token in ("rsu_delay_ms=%s" % rsu, "ego_delay_ms=%s" % ego,
                              "rsu_delay_frames=%s" % (rsu // 50),
                              "ego_delay_frames=%s" % (ego // 50),
                              "pnp_config_codriving_dualfifo_joint_rsu%s_ego%s_l1.yaml" % (rsu, ego)):
                    self.assertIn(token, result.stdout)

    def test_joint_check_config_rejects_nonexperimental_pairs(self):
        for rsu, ego in ((0, 0), (0, 200), (100, 0), (100, 100),
                         (200, 200), (300, 200), (400, 400), (500, 600),
                         (50, 200), (100, 250), (500, 500), (800, 500)):
            with self.subTest(rsu=rsu, ego=ego):
                result = run_command(["bash", SCRIPT, "--joint", "--check-config", rsu, ego])
                self.assertEqual(result.returncode, 2, result.stdout)

    def test_joint_c2_check_config_accepts_exact_triangular_arm_set(self):
        for rsu, ego in JOINT_ARMS:
            with self.subTest(rsu=rsu, ego=ego):
                result = run_command(["bash", SCRIPT, "--joint-c2", "--check-config", rsu, ego])
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("pnp_config_codriving_dualfifo_jointc2_rsu%s_ego%s_l1.yaml" %
                              (rsu, ego), result.stdout)

    def test_joint_c2_check_config_rejects_invalid_or_combined_modes(self):
        for args in (("--joint-c2", "--check-config", "100", "100"),
                     ("--joint", "--joint-c2", "--check-config", "100", "200")):
            with self.subTest(args=args):
                result = run_command(["bash", SCRIPT, *args])
                self.assertEqual(result.returncode, 2, result.stdout)

    def test_joint_c2_fixture_uses_exact_age_namespace_and_cadence(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, _ = contract_fixture(tmp, 100, 200, joint_c2=True)
            parser.validate_audits(str(path), "0", "0", 100, 200, str(config))

    def test_run_root_override_rejects_empty_and_relative_paths_before_gpu_access(self):
        args = ["bash", SCRIPT, "--joint", "0", "40000", "100", "200", "0"]
        for run_root in ("", "relative/results", "/"):
            with self.subTest(run_root=run_root):
                result = run_command(args, extra_env={"DUAL_FIFO_RUN_ROOT": run_root})
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("DUAL_FIFO_RUN_ROOT", result.stdout)

    def test_summary_gate_requires_audits_but_raw_json_parser_does_not(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, meta = contract_fixture(tmp)
            (meta / "latency_align_audit.csv").unlink()
            self.assertTrue(parser.parse_route_result(str(path))["countable"])
            row = parser.build_summary_row(str(path), "0", "0", "0", "0", str(config), "abc", "false")
            self.assertEqual((row[8], row[9]), ("INFRA_INVALID", False))

    def test_empty_or_wrong_process_ownership_never_sends_a_signal(self):
        parser = self.parser()
        self.assertTrue(hasattr(parser, "stop_owned_process"))
        with mock.patch.object(parser.os, "killpg") as terminate:
            parser.stop_owned_process("", "")
            parser.stop_owned_process(str(os.getpid()), "not-the-start-token")
            terminate.assert_not_called()

    def test_live_group_cannot_claim_an_unrelated_listener(self):
        parser = self.parser()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            try:
                token = parser.process_token(proc.pid)
                self.assertFalse(parser.owned_listener_ready(proc.pid, token, listener.getsockname()[1]))
                parser.stop_owned_process(proc.pid, token)
                self.assertEqual(proc.wait(timeout=3), -signal.SIGKILL)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()

    def test_ipv6_listener_also_blocks_port_preflight(self):
        parser = self.parser()
        with socket.socket(socket.AF_INET6) as listener, contextlib.redirect_stderr(io.StringIO()):
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            listener.bind(("::1", 0))
            listener.listen()
            self.assertEqual(parser.main(["parser", "--ports-free", str(listener.getsockname()[1])]), 3)

    def test_audit_gate_preserves_all_arms_and_driving_outcomes(self):
        parser = self.parser()
        for rsu, ego, arm, _, _ in ARMS:
            with tempfile.TemporaryDirectory() as tmp:
                path, config, _ = contract_fixture(tmp, rsu, ego)
                for status, expected in (("Completed", 4.153), ("Failed", 4.153),
                                         ("Failed - Agent timed out", 0.0)):
                    path.write_text(json.dumps(result_fixture(status, 4.153)))
                    parsed = parser.classify_execution(str(path), route="0", repeat="0", rsu_delay=rsu,
                                                       ego_delay=ego, agent_config=str(config))
                    self.assertTrue(parsed["countable"], (arm, parsed))
                    self.assertEqual(parsed["DS"], expected)

    def test_rsu_sim_time_gate_rejects_missing_invalid_and_wrong_values(self):
        parser = self.parser()
        for field, index, value in (
                ("source_sim_time_s", 0, "0"),
                ("source_sim_time_s", -1, ""),
                ("consume_sim_time_s", 0, ""),
                ("source_sim_time_s", -1, "0.800000002"),
                ("consume_sim_time_s", -1, "1.000000002"),
                ("source_sim_time_s", -1, "nan"),
                ("consume_sim_time_s", -1, "inf")):
            with self.subTest(field=field, index=index, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    path, config, meta = contract_fixture(tmp, 200)
                    audit = meta / "latency_align_audit.csv"
                    with audit.open() as stream:
                        rows = list(csv.DictReader(stream))
                    rows[index] = dict(rows[index], **{field: value})
                    with audit.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    with self.assertRaisesRegex(ValueError, field):
                        parser.validate_audits(str(path), "0", "0", 200, 0, str(config))
        for missing in ("source_sim_time_s", "consume_sim_time_s"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                path, config, meta = contract_fixture(tmp, 200)
                audit = meta / "latency_align_audit.csv"
                with audit.open() as stream:
                    rows = list(csv.DictReader(stream))
                fields = [key for key in rows[0] if key != missing]
                with audit.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "headers"):
                    parser.validate_audits(str(path), "0", "0", 200, 0, str(config))

    def test_rsu_sim_time_gate_accepts_absolute_roundoff_within_one_nanosecond(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, meta = contract_fixture(tmp, 200)
            audit = meta / "latency_align_audit.csv"
            with audit.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[-1] = dict(rows[-1], source_sim_time_s="0.8000000005",
                            consume_sim_time_s="0.9999999995")
            rows[0] = dict(rows[0], consume_sim_time_s="0.000000001")
            with audit.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            parser.validate_audits(str(path), "0", "0", 200, 0, str(config))

    def test_acceptance_requires_matching_route_repeat_config_and_flush(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, _ = contract_fixture(tmp, 200, 0)
            context = dict(route="0", repeat="0", rsu_delay=200, ego_delay=0, agent_config=str(config))
            for wrong in (dict(route="1"), dict(repeat="1"), dict(rsu_delay=800),
                          dict(rsu_delay=200.5), dict(agent_config="missing.yaml")):
                self.assertFalse(parser.classify_execution(str(path), **dict(context, **wrong))["countable"])
            log = Path(tmp) / "eval.log"
            for message in ("latency audit dump failed", "ego output FIFO audit close error"):
                log.write_text(message + "\n")
                self.assertFalse(parser.classify_execution(str(path), str(log), **context)["countable"])

    def test_malformed_config_is_infrastructure_invalid_not_an_exception(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, _ = contract_fixture(tmp)
            bad = Path(tmp) / config.name
            bad.write_text("simulation: [")
            try:
                result = parser.classify_execution(str(path), route="0", repeat="0", rsu_delay=0,
                                                   ego_delay=0, agent_config=str(bad))
            except Exception as error:
                self.fail("malformed config escaped the gate: %s" % error)
            self.assertEqual((result["status"], result["countable"]), ("INFRA_INVALID", False))

    def test_check_config_rejects_joint_arm(self):
        result = run_command(["bash", SCRIPT, "--check-config", 200, 200])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("joint arm is not enabled", result.stdout)

    def test_check_config_rejects_unsupported_and_malformed_inputs(self):
        for args in (("--check-config", "123", "0"),
                     ("--check-config", "0", "-1"),
                     ("--check-config", "200"),
                     ("--check-config", "0", "0", "extra"),
                     ("bad", "40000", "0", "0", "0"),
                     ("0", "65534", "0", "0", "0"),
                     ("0", "40000", "0", "0", "../escape"),
                     ("0", "40000", "0", "0", ""),
                     ("0", "40000", "0", "0", "0", "-1")):
            with self.subTest(args=args):
                self.assertTrue(SCRIPT.is_file(), "dual_fifo_multiroute_sc.sh is missing")
                result = run_command(["bash", SCRIPT] + list(args))
                self.assertEqual(result.returncode, 2, result.stdout)

    def test_failed_or_timeout_ds_zero_is_countable(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for status in ("Failed", "Failed - Agent timed out", "Completed"):
                path.write_text(json.dumps(result_fixture(status)))
                parsed = parser.parse_route_result(str(path))
                self.assertTrue(parsed["countable"])
                self.assertEqual(parsed["status"], status)
                self.assertEqual(parsed["DS"], 0.0)

    def test_failed_route_preserves_raw_scores_and_collision_fields(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(result_fixture(ds=4.153)))
            parsed = parser.parse_route_result(str(path))
        self.assertEqual(parsed["DS"], 4.153)
        self.assertEqual(parsed["RC"], 15.0)
        self.assertEqual(parsed["coll_lay"], 2.0)
        self.assertEqual(parsed["coll_veh"], 0.5)

    def test_checkpoint_evidence_names_do_not_masquerade_as_carla_timeout(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            payload = result_fixture(ds=39.0)
            infractions = payload["_checkpoint"]["global_record"]["infractions"]
            infractions["collisions_vehicle"] = ["carlamotors.carlacola"]
            infractions["route_timeout"] = 0
            path.write_text(json.dumps(payload))
            parsed = parser.parse_route_result(str(path))
        self.assertTrue(parsed["countable"], parsed)
        self.assertEqual((parsed["status"], parsed["DS"]), ("Failed", 39.0))

    def test_checkpoint_explicit_infrastructure_error_is_invalid(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            payload = result_fixture("Completed", ds=39.0)
            payload["_checkpoint"]["global_record"]["meta"] = {"error": "CARLA connection failed"}
            path.write_text(json.dumps(payload))
            parsed = parser.parse_route_result(str(path))
        self.assertFalse(parsed["countable"], parsed)

    def test_checkpoint_exception_fields_do_not_join_path_and_failed_status(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            payload = result_fixture(ds=39.0)
            payload["_checkpoint"]["global_record"]["meta"] = {
                "exceptions": [["carlamotors.carlacola", 0, "Failed"]]}
            path.write_text(json.dumps(payload))
            parsed = parser.parse_route_result(str(path))
        self.assertTrue(parsed["countable"], parsed)

    def test_rejects_nonterminal_unknown_and_infrastructure_global_statuses(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for status in ("Started", "INFRA_INVALID", "Mystery", "Crashed",
                           "Failed - Run scenario crashed", "Failed - unexpected reason", True):
                with self.subTest(status=status):
                    path.write_text(json.dumps(result_fixture(status)))
                    parsed = parser.parse_route_result(str(path))
                    self.assertFalse(parsed["countable"], parsed)
                    self.assertEqual(parsed["status"], "INFRA_INVALID")

    def test_rejects_invalid_statuses_hidden_by_global_failed(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for status in ("Started", "Mystery", "Failed - Run scenario crashed"):
                for source in ("records", "exceptions"):
                    with self.subTest(status=status, source=source):
                        payload = result_fixture()
                        if source == "records":
                            payload["_checkpoint"]["records"] = [{"status": status}]
                        else:
                            payload["_checkpoint"]["global_record"]["meta"] = {
                                "exceptions": [["route_0", 0, status]]}
                        path.write_text(json.dumps(payload))
                        parsed = parser.parse_route_result(str(path))
                        self.assertFalse(parsed["countable"], parsed)

    def test_required_scores_reject_missing_and_malformed_values(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for label, payload in invalid_score_fixtures():
                with self.subTest(case=label):
                    path.write_text(json.dumps(payload))
                    parsed = parser.parse_route_result(str(path))
                    self.assertFalse(parsed["countable"], parsed)
                    self.assertEqual(parsed["status"], "INFRA_INVALID")
                    self.assertTrue(parsed["error"])

    def test_supported_terminal_driving_statuses_keep_numeric_zero(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for status in ("Completed", "Failed", "Failed - Agent timed out",
                           "Failed - Agent got blocked", "Failed - Agent deviated from the route"):
                for zero in (0, 0.0):
                    with self.subTest(status=status, zero=zero):
                        record = {"status": status,
                                  "scores": {"score_route": zero, "score_composed": zero}}
                        path.write_text(json.dumps({"_checkpoint": {"global_record": record}}))
                        parsed = parser.parse_route_result(str(path))
                        self.assertTrue(parsed["countable"], parsed)
                        self.assertEqual(parsed["status"], status)
                        for key in ("DS", "RC", "coll_lay", "coll_veh"):
                            self.assertEqual(parsed[key], 0.0)

    def test_explicit_driving_timeout_normalizes_ds_but_preserves_rc(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for fixture in (result_fixture("Failed - Agent timed out", 4.153),
                            result_fixture(ds=4.153)):
                if fixture["_checkpoint"]["global_record"]["status"] == "Failed":
                    fixture["_checkpoint"]["global_record"]["infractions"]["route_timeout"] = 1.0
                path.write_text(json.dumps(fixture))
                parsed = parser.parse_route_result(str(path))
                self.assertTrue(parsed["countable"])
                self.assertEqual(parsed["DS"], 0.0)
                self.assertEqual(parsed["RC"], 15.0)

    def test_missing_corrupt_and_malformed_json_are_infrastructure_invalid(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for payload in (None, "{bad", "null", "{}", '{"_checkpoint": []}',
                            '{"_checkpoint": {"global_record": {}}}',
                            json.dumps(result_fixture(ds="bad")),
                            json.dumps(result_fixture(ds=float("nan")))):
                if payload is not None:
                    path.write_text(payload)
                parsed = parser.parse_route_result(str(path))
                self.assertEqual(parsed["status"], "INFRA_INVALID")
                self.assertFalse(parsed["countable"])
                self.assertTrue(parsed["error"])

    def test_explicit_agent_and_simulation_crashes_are_infrastructure_invalid(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for status in ("Failed - Agent crashed", "Failed - Simulation crashed"):
                fixture = result_fixture()
                fixture["_checkpoint"]["global_record"]["meta"] = {
                    "exceptions": [["route_0", 0, status]]}
                path.write_text(json.dumps(fixture))
                self.assertFalse(parser.parse_route_result(str(path))["countable"])

    def test_summary_finds_only_matching_route_audits_and_records_provenance(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, meta = contract_fixture(tmp, 0, 500, "317", "2")
            path.write_text(json.dumps(result_fixture(ds=4.153)))
            rsu_audit = meta / "latency_align_audit.csv"
            ego_audit = meta / "ego_output_fifo_audit.csv"
            row = parser.build_summary_row(str(path), "317", "2", "0", "500",
                                           str(config), "abc123", "true")
            self.assertTrue(row[9])
            self.assertEqual(len(row), len(FIELDS))
            data = dict(zip(FIELDS, row))
            self.assertEqual(data["rsu_delay_frames"], 0)
            self.assertEqual(data["ego_delay_frames"], 10)
            self.assertEqual(data["rsu_audit_csv"], str(rsu_audit))
            self.assertEqual(data["ego_output_audit_csv"], str(ego_audit))
            self.assertEqual(data["sim_rate_hz"], 20.0)
            self.assertEqual(data["agent_config"], str(config))
            self.assertEqual(data["git_commit"], "abc123")
            self.assertEqual(data["git_dirty"], "true")
            rsu_audit.unlink()
            row = parser.build_summary_row(str(path), "317", "2", "0", "500",
                                           str(config), "abc123", "false")
            self.assertEqual(row[13], "")

    def test_parser_cli_appends_header_and_rows(self):
        self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, _ = contract_fixture(tmp, 200, 0, "317", "2")
            summary = Path(tmp) / "summary.csv"
            args = [sys.executable, RESULT_PARSER, path, "317", "2", "200", "0",
                    summary, config, "abc123", "false"]
            for _ in range(2):
                result = run_command(args)
                self.assertEqual(result.returncode, 0, result.stdout)
            with summary.open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], FIELDS)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1][:6], ["317", "2", "200", "4", "0", "0"])
            self.assertEqual(rows[1][8:10], ["Failed", "True"])

    def test_execution_gate_explains_infrastructure_failures(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            path, config, _ = contract_fixture(tmp)
            log = Path(tmp) / "eval.log"
            for kwargs in ({"eval_log": str(log)}, {"process_exit": 9},
                           {"infra_error": "carla_start_failed"}):
                parsed = parser.classify_execution(str(path), **kwargs)
                self.assertFalse(parsed["countable"])
                self.assertTrue(parsed["error"])
            log.write_text("RuntimeError: CUDA out of memory\n")
            self.assertFalse(parser.classify_execution(str(path), str(log))["countable"])
            log.write_text("route finished\n")
            self.assertTrue(parser.classify_execution(str(path), str(log), route="0", repeat="0",
                                                      rsu_delay=0, ego_delay=0, agent_config=str(config))["countable"])

    def test_main_applies_execution_evidence_to_summary_and_check_exit(self):
        parser = self.parser()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path, config, _ = contract_fixture(tmp, 0, 200)
            summary = Path(tmp) / "summary.csv"
            self.assertEqual(parser.main(["parser", "--check-result", str(path), "--route", "0",
                                          "--repeat", "0", "--rsu-delay", "0", "--ego-delay", "200",
                                          "--agent-config", str(config)]), 0)
            self.assertEqual(parser.main(["parser", "--check-result", str(path),
                                          "--process-exit", "9"]), 1)
            self.assertEqual(parser.main(["parser", str(path), "0", "0", "0", "200",
                                          str(summary), str(config), "abc", "true",
                                          "--infra-error", "carla_start_failed"]), 0)
            with summary.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["countable"], "False")
            self.assertEqual(row["status"], "INFRA_INVALID")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.main(["parser", "not-enough-arguments"])


# The real legacy evaluation shell runs a tiny evaluator fixture. Only its
# external GPU/server/process-control boundaries are replaced in PATH.
FAKE_EVALUATOR = r'''
import json
import os
from pathlib import Path
import sys
import time
with open(os.environ["FIXTURE_PROCESSES"], "a") as stream:
    stream.write("%s,%s\n" % (os.getpid(), os.getpgrp()))
Path(os.environ["FIXTURE_ARGV"]).write_text(os.environ["FIXTURE_EVAL_ARGV"])

counter = Path(os.environ["FIXTURE_COUNTER"])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if os.environ.get("FIXTURE_EVAL_WAIT"):
    time.sleep(30)
mode = os.environ.get("FIXTURE_MODE", "failed")
path = Path(os.environ["CHECKPOINT_ENDPOINT"]).parent / "ego_vehicle_0/results.json"
path.parent.mkdir(parents=True, exist_ok=True)
record = {"status": "Failed", "scores": {"score_route": 15, "score_composed": 0}, "infractions": {}}
payload = json.loads(os.environ["FIXTURE_PAYLOAD"]) if "FIXTURE_PAYLOAD" in os.environ else {"_checkpoint": {"global_record": record}}
if mode != "missing" and (mode != "retry" or count > 1):
    path.write_text(json.dumps(payload))
if mode == "corrupt":
    path.write_text("{not-json")
if mode == "no_global":
    path.write_text(json.dumps({"_checkpoint": {"records": []}}))
meta = Path(os.environ["SAVE_PATH"]) / "ego0_timestamp/meta"
import yaml
from fixture_support import write_fixture_audits
simulation = yaml.safe_load(Path(os.environ["TEAM_CONFIG"]).read_text())["simulation"]
write_fixture_audits(meta, simulation["rsu_fifo_delay_ms"], simulation["ego_output_fifo_delay_ms"],
                     os.environ.get("FIXTURE_AUDIT_FAULT", ""), simulation["skip_frames"])
if os.environ.get("FIXTURE_AUDIT_FAULT") == "flush":
    print("ego output FIFO audit flush failed: final close error")
if mode == "cuda":
    print("RuntimeError: CUDA out of memory")
if mode == "process":
    sys.exit(9)
print("fixture evaluator completed normally")
'''


class DualFifoRunnerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "dual_fifo_multiroute_sc.sh is missing")
        self.assertTrue(RESULT_PARSER.is_file(), "dual_fifo_result.py is missing")
        self.temp = tempfile.TemporaryDirectory(prefix="dual-fifo-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        for name in (SCRIPT.name, RESULT_PARSER.name, "action_fifo_eval_one.sh"):
            shutil.copy2(str(REPO / "scripts" / name), str(self.repo / "scripts" / name))
        # Capture the real runner's nine positional shell arguments, then run
        # the seed/scenario-aware evaluator shell. The fake evaluator persists them.
        eval_shell = self.repo / "scripts/action_fifo_eval_one.sh"
        eval_shell.write_text(
            '#!/bin/bash\nexport FIXTURE_EVAL_ARGV="$(python -c '
            "'import json,sys; print(json.dumps(sys.argv[1:]))' \"$@\")\"\n" +
            eval_shell.read_text())
        (self.repo / CONFIG_DIR).mkdir(parents=True)
        for path in (REPO / CONFIG_DIR).glob("pnp_config_codriving_dualfifo_*_l1.yaml"):
            shutil.copy2(str(path), str(self.repo / CONFIG_DIR / path.name))
        evaluator = self.repo / "simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text(FAKE_EVALUATOR)
        (evaluator.parent / "fixture_support.py").write_text(inspect.getsource(write_fixture_audits))
        carla = self.repo / "external_paths/carla_root/CarlaUE4.sh"
        carla.parent.mkdir(parents=True)
        carla.write_text("#!%s\n" % sys.executable + '''import os, socket, sys, time
with open(os.environ["FIXTURE_PROCESSES"], "a") as stream:
    stream.write("%s,%s\\n" % (os.getpid(), os.getpgrp()))
if os.environ.get("FIXTURE_NC_EXIT") == "1" or os.environ.get("FIXTURE_EARLY_EXIT"):
    sys.exit(7)
port = int(next(arg.split("=")[1] for arg in sys.argv if arg.startswith("--world-port=")))
sock = socket.socket()
sock.bind(("127.0.0.1", port))
sock.listen()
time.sleep(60)
''')
        carla.chmod(0o755)
        fakebin = self.repo / "fakebin"
        fakebin.mkdir()
        programs = {"nvidia-smi": '#!/bin/bash\necho "${FIXTURE_GPU:-0, 0}"\n',
                    "nc": '#!/bin/bash\nexit "${FIXTURE_NC_EXIT:-0}"\n',
                    "sleep": "#!/bin/bash\nexit 0\n",
                    "pkill": '#!/bin/bash\necho forbidden >> "$FIXTURE_KILLS"\nexit 0\n'}
        for name, source in programs.items():
            path = fakebin / name
            path.write_text(source)
            path.chmod(0o755)
        self.counter = self.repo / "eval_calls"
        self.processes = self.repo / "processes"
        self.kills = self.repo / "kills"
        self.env = {"PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
                    "FIXTURE_COUNTER": str(self.counter), "FIXTURE_PROCESSES": str(self.processes),
                    "FIXTURE_ARGV": str(self.repo / "evaluator_argv.json"),
                    "FIXTURE_KILLS": str(self.kills)}
        self.addCleanup(self.cleanup_fixture_children)
        for command in (["git", "init", "-q"], ["git", "add", "scripts", "simulation"],
                        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                         "commit", "-q", "-m", "fixture"]):
            result = run_command(command, self.repo)
            self.assertEqual(result.returncode, 0, result.stdout)

    def cleanup_fixture_children(self):
        if self.processes.exists():
            for line in self.processes.read_text().splitlines():
                try:
                    pid = int(line.split(",")[0])
                    if os.readlink("/proc/%s/cwd" % pid).startswith(str(self.repo) + "/"):
                        os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, FileNotFoundError):
                    pass

    def assert_owned_children_gone(self):
        self.assertFalse(self.kills.exists(), "global pattern kill is forbidden")
        self.assertTrue(self.processes.exists())
        for line in self.processes.read_text().splitlines():
            pid, group = map(int, line.split(","))
            self.assertNotEqual(group, os.getpgrp(), "child was not isolated into an owned group")
            stat = Path("/proc/%s/stat" % pid)
            self.assertTrue(not stat.exists() or stat.read_text().split(") ")[1].startswith("Z"))

    def test_token_publication_precedes_supervised_command(self):
        status, identity = self.repo / "status", self.repo / "identity.json"
        command = "import pathlib,sys; sys.exit(0 if pathlib.Path(sys.argv[1]).is_file() else 91)"
        proc = subprocess.Popen([sys.executable, str(RESULT_PARSER), "--supervise", str(status),
                                 str(identity), sys.executable, "-c", command, str(identity)],
                                start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 3
            while not status.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(status.exists(), "supervisor did not publish command status")
            self.assertEqual(status.read_text(), "0")
            payload = json.loads(identity.read_text())
            self.assertEqual(set(payload), {"pid", "ppid", "pgrp", "session", "start_token"})
            self.assertEqual((payload["pid"], payload["pgrp"], payload["session"]), (proc.pid,) * 3)
            self.assertEqual(payload["ppid"], os.getpid())
            self.assertFalse(Path(str(identity) + ".tmp").exists())
            token = Path("/proc/%s/stat" % proc.pid).read_text().rsplit(") ", 1)[1].split()[19]
            result = run_command([sys.executable, RESULT_PARSER, "--await-owned", identity,
                                  proc.pid, token, os.getpid(), ".2"])
            self.assertEqual((result.returncode, result.stdout), (0, token + "\n"))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate(timeout=3)

    def install_delayed_session_fixture(self, delay=30):
        """Freeze the fork before setsid: the old one-shot token cannot succeed."""
        ready = self.repo / "prepublication_ready"
        wrapper = self.repo / "fakebin/setsid"
        wrapper.write_text("#!%s\n" % sys.executable + '''import os, pathlib, sys, time
pathlib.Path(os.environ["FIXTURE_READY"]).write_text(str(os.getpid()))
with open(os.environ["FIXTURE_READY"] + ".launches", "a") as stream:
    stream.write(str(os.getpid()) + "\\n")
time.sleep(%r)
os.execv(%r, [%r] + sys.argv[1:])
''' % (delay, shutil.which("setsid"), shutil.which("setsid")))
        wrapper.chmod(0o755)
        return ready

    def assert_prepublication_signal(self, signum, expected):
        ready = self.install_delayed_session_fixture()
        env = dict(os.environ, **dict(self.env, DUAL_FIFO_PYTHON=sys.executable, FIXTURE_READY=str(ready)))
        proc = subprocess.Popen(["bash", str(self.repo / "scripts" / SCRIPT.name),
                                 "0", "41000", "200", "0", "0"], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        child = None
        try:
            deadline = time.monotonic() + 3
            while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists(), "fixture never reached pre-publication window")
            child = int(ready.read_text())
            started = time.monotonic()
            proc.send_signal(signum)
            try:
                output, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.fail("runner exceeded 5s during pre-publication signal %s" % signum)
            self.assertEqual(proc.returncode, expected, output)
            self.assertLess(time.monotonic() - started, 5)
            self.assertFalse(self.processes.exists(), "real command started before publication")
            self.assertFalse(self.kills.exists())
            stat = Path("/proc/%s/stat" % child)
            self.assertTrue(not stat.exists() or stat.read_text().rsplit(") ", 1)[1].startswith("Z"))
        finally:
            if child is not None:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=3)

    def test_prepublication_interrupt_is_bounded_and_starts_no_command(self):
        self.assert_prepublication_signal(signal.SIGINT, 130)

    def test_prepublication_term_is_bounded_and_starts_no_command(self):
        self.assert_prepublication_signal(signal.SIGTERM, 143)

    def test_delayed_session_establishment_still_starts_and_cleans_normally(self):
        ready = self.install_delayed_session_fixture(.2)
        result = self.run_fixture(extra_env={"FIXTURE_READY": str(ready)})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.counter.read_text(), "1")
        self.assert_owned_children_gone()
        for pid in Path(str(ready) + ".launches").read_text().splitlines():
            self.assertFalse(Path("/proc/%s" % pid).exists(), "supervisor was not reaped")

    def test_unpublished_identity_times_out_and_retries_without_starting_commands(self):
        ready = self.install_delayed_session_fixture()
        started = time.monotonic()
        result = self.run_fixture(extra_env={"FIXTURE_READY": str(ready)})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertLess(time.monotonic() - started, 15)
        self.assertFalse(self.processes.exists())
        self.assertFalse(self.counter.exists())
        self.assertFalse(self.kills.exists())
        launches = Path(str(ready) + ".launches").read_text().splitlines()
        self.assertEqual(len(launches), 2)
        for pid in launches:
            self.assertFalse(Path("/proc/%s" % pid).exists(), "unclaimed child was not reaped")
        self.assertEqual(self.summary()[0]["countable"], "False")

    def test_corrupt_identity_file_fails_without_unrelated_signal(self):
        parser = DualFifoHarnessTest().parser()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            token = Path("/proc/%s/stat" % proc.pid).read_text().rsplit(") ", 1)[1].split()[19]
            identity = self.repo / "identity.json"
            valid = dict(pid=proc.pid, ppid=os.getpid(), pgrp=proc.pid, session=proc.pid, start_token=token)
            self.assertTrue(hasattr(parser, "await_owned_identity"), "bounded identity handshake is missing")
            for field, value in (("pid", proc.pid + 1), ("ppid", os.getpid() + 1),
                                 ("pgrp", proc.pid + 1), ("session", proc.pid + 1), ("start_token", "wrong")):
                with self.subTest(field=field), mock.patch.object(parser.os, "killpg") as killpg:
                    identity.write_text(json.dumps(dict(valid, **{field: value})))
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(parser.main(["parser", "--await-owned", str(identity), str(proc.pid),
                                                      token, str(os.getpid()), ".1"]), 3)
                    killpg.assert_not_called()
                    self.assertIsNone(proc.poll(), "unrelated process was signalled")
        finally:
            proc.kill()
            proc.wait(timeout=3)

    def test_stop_child_requires_start_time_and_owner_before_session_establishment(self):
        parser = DualFifoHarnessTest().parser()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(hasattr(parser, "process_start_token"), "pre-session ownership is missing")
            token = parser.process_start_token(proc.pid, os.getpid())
            self.assertEqual(os.getpgid(proc.pid), os.getpgrp())
            parser.stop_direct_child(proc.pid, "wrong", os.getpid())
            parser.stop_direct_child(proc.pid, token, os.getpid() + 1)
            self.assertIsNone(proc.poll())
            parser.stop_direct_child(proc.pid, token, os.getpid())
            self.assertLess(proc.wait(timeout=3), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)

    def test_missing_identity_handshake_has_monotonic_deadline(self):
        parser = DualFifoHarnessTest().parser()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(hasattr(parser, "await_owned_identity"), "bounded identity handshake is missing")
            token = Path("/proc/%s/stat" % proc.pid).read_text().rsplit(") ", 1)[1].split()[19]
            started = time.monotonic()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(parser.main(["parser", "--await-owned", str(self.repo / "absent"),
                                              str(proc.pid), token, str(os.getpid()), ".05"]), 3)
            self.assertLess(time.monotonic() - started, .5)
            self.assertIsNone(proc.poll())
        finally:
            proc.kill()
            proc.wait(timeout=3)

    def test_live_unverified_slot_refuses_wait_and_does_not_signal(self):
        # Execute the real shell function; the wait spy makes any unsafe wait visible.
        source = SCRIPT.read_text().split("stop_slot() {", 1)[1].split("\nteardown()", 1)[0]
        wait_marker = self.repo / "unsafe_wait"
        shell = "stop_slot() {" + source + '\nwait() { touch "$WAIT_MARKER"; }\nstop_slot "$1" "" ""'
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            started = time.monotonic()
            result = run_command(["bash", "-c", shell, "fixture", proc.pid],
                                 extra_env={"PY_BIN": sys.executable, "PARSER": str(RESULT_PARSER),
                                            "WAIT_MARKER": str(wait_marker)})
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("refusing wait", result.stdout)
            self.assertLess(time.monotonic() - started, 6)
            self.assertFalse(wait_marker.exists())
            self.assertIsNone(proc.poll(), "unverified process was signalled")
        finally:
            proc.kill()
            proc.wait(timeout=3)

    def test_stop_child_escalates_only_the_verified_term_ignoring_child(self):
        parser = DualFifoHarnessTest().parser()
        ready = self.repo / "term_ignored"
        command = ("import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                   "pathlib.Path(sys.argv[1]).touch(); time.sleep(30)")
        proc = subprocess.Popen([sys.executable, "-c", command, str(ready)])
        try:
            deadline = time.monotonic() + 3
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            token = parser.process_start_token(proc.pid, os.getpid())
            started = time.monotonic()
            parser.stop_direct_child(proc.pid, token, os.getpid())
            self.assertEqual(proc.wait(timeout=3), -signal.SIGKILL)
            self.assertLess(time.monotonic() - started, 1)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)

    def test_publication_failure_starts_no_command(self):
        marker = self.repo / "command_started"
        result = run_command([sys.executable, RESULT_PARSER, "--supervise", self.repo / "status",
                              self.repo / "identity", sys.executable, "-c",
                              "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()", marker])
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertFalse(marker.exists())

    def test_published_but_unclaimed_supervisor_cleans_its_command_on_term(self):
        """Do not let killing only the supervisor orphan its already-started child."""
        wrapper = self.repo / "delayed-python"
        ready = self.repo / "published"
        wrapper.write_text("#!%s\n" % sys.executable + '''import importlib.util, os, pathlib, sys, time
if "--await-owned" in sys.argv:
    time.sleep(.1)
    sys.exit(3)
if "--supervise" in sys.argv:
    spec = importlib.util.spec_from_file_location("parser", sys.argv[1])
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    original = getattr(parser, "publish_owned_identity", None)
    def publish(path):
        identity = original(path)
        pathlib.Path(os.environ["FIXTURE_READY"]).write_text(str(os.getpid()))
        return identity
    parser.publish_owned_identity = publish
    sys.exit(parser.main(sys.argv[1:]))
os.execv(%r, [%r] + sys.argv[1:])
''' % (sys.executable, sys.executable))
        wrapper.chmod(0o755)
        env = dict(os.environ, **dict(self.env, DUAL_FIFO_PYTHON=str(wrapper), FIXTURE_READY=str(ready)))
        proc = subprocess.Popen(["bash", str(self.repo / "scripts" / SCRIPT.name),
                                 "0", "41000", "200", "0", "0"], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            deadline = time.monotonic() + 3
            while not self.processes.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists(), "identity was never published")
            self.assertTrue(self.processes.exists(), "supervised command did not start")
            proc.send_signal(signal.SIGTERM)
            output, _ = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 143, output)
            self.assert_owned_children_gone()
            supervisor = int(ready.read_text())
            self.assertFalse(Path("/proc/%s" % supervisor).exists(), "supervisor was not reaped")
        finally:
            if ready.exists():
                try:
                    os.killpg(int(ready.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=3)

    def test_occupied_required_ports_abort_without_terminating_existing_processes(self):
        for offset in (0, 1, 2, 5):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 40000 + offset))
                listener.listen()
                result = self.run_fixture()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(self.processes.exists(), result.stdout)
                self.assertFalse(self.kills.exists())

    def test_owned_success_and_retry_cleanup_avoids_global_patterns(self):
        for mode in ("failed", "retry"):
            result = self.run_fixture(mode)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assert_owned_children_gone()

    def test_early_server_exit_is_not_readiness(self):
        result = self.run_fixture(extra_env={"FIXTURE_EARLY_EXIT": "1"})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.counter.exists())
        self.assert_owned_children_gone()

    def test_interrupt_and_term_cleanup_owned_groups(self):
        for signum, expected in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            self.counter.unlink() if self.counter.exists() else None
            env = dict(os.environ, **dict(self.env, DUAL_FIFO_PYTHON=sys.executable, FIXTURE_EVAL_WAIT="1"))
            proc = subprocess.Popen(["bash", str(self.repo / "scripts" / SCRIPT.name),
                                     "0", "40000", "200", "0", "0"], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.addCleanup(lambda proc=proc: proc.kill() if proc.poll() is None else None)
            deadline = time.monotonic() + 10
            while not self.counter.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(self.counter.exists())
            proc.send_signal(signum)
            output, _ = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, expected, output)
            self.assert_owned_children_gone()

    def run_fixture(self, mode="failed", extra_env=None, rsu=200, ego=0,
                    runner_options=(), routes="0"):
        env = dict(self.env, FIXTURE_MODE=mode, **(extra_env or {}))
        return run_command(["bash", self.repo / "scripts" / SCRIPT.name,
                            *runner_options, "0", "40000", rsu, ego, routes, "2"],
                           extra_env=env)

    def summary(self):
        files = list((self.repo / "results/dual_fifo_pilot").rglob("summary*.csv"))
        self.assertEqual(len(files), 1, files)
        with files[0].open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, FIELDS)
            return list(reader)

    def test_failed_driving_run_is_not_retried_and_is_fully_scoped(self):
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.counter.read_text(), "1")
        rows = self.summary()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["countable"], "True")
        self.assertEqual(row["status"], "Failed")
        self.assertEqual(row["DS"], "0.0")
        self.assertEqual(row["git_dirty"], "false")
        self.assertEqual(row["git_commit"], run_command(["git", "rev-parse", "HEAD"], self.repo).stdout.strip())
        self.assertEqual(row["agent_config"], str(self.repo / CONFIG_DIR / "pnp_config_codriving_dualfifo_rsu200_l1.yaml"))
        argv_path = self.repo / "evaluator_argv.json"
        self.assertTrue(argv_path.is_file(), "fake evaluator must capture all nine shell argv")
        self.assertEqual(json.loads(argv_path.read_text()), [
            "0", "40000", "dualfifo_pilot_rsu200_sc1_r0_cseed2000_tseed2000_n2_a1",
            "2", "codriving_dualfifo_rsu200_l1", "_1",
            "v2x_final/town05_short_collab/r0_repeat2", "2000", "2000",
        ])
        owned = (self.repo / "results/dual_fifo_pilot").resolve()
        for field in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv"):
            artifact = Path(row[field]).resolve()
            self.assertTrue(artifact.is_file(), row)
            self.assertIn(owned, artifact.parents)
        self.assertEqual([p.name for p in (self.repo / "results").iterdir()], ["dual_fifo_pilot"])

    def test_joint_run_root_override_contains_summary_and_artifacts(self):
        run_root = Path(self.temp.name) / "joint-runs"
        result = self.run_fixture(rsu=100, ego=200, runner_options=("--joint",),
                                  extra_env={"DUAL_FIFO_RUN_ROOT": str(run_root)})
        self.assertEqual(result.returncode, 0, result.stdout)
        summaries = list(run_root.glob("*/summary.csv"))
        self.assertEqual(len(summaries), 1, result.stdout)
        with summaries[0].open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["rsu_fifo_delay_ms"],
                          rows[0]["ego_output_fifo_delay_ms"]), ("100", "200"))
        self.assertIn("joint_rsu100_ego200", str(summaries[0]))
        for field in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv"):
            artifact = Path(rows[0][field]).resolve()
            self.assertTrue(artifact.is_file(), rows[0])
            self.assertIn(run_root.resolve(), artifact.parents)
        self.assertFalse((self.repo / "results/dual_fifo_pilot").exists())

    def test_joint_c2_fake_runner_records_namespace_and_artifacts(self):
        run_root = Path(self.temp.name) / "jointc2-runs"
        result = self.run_fixture(rsu=100, ego=200, runner_options=("--joint-c2",),
                                  extra_env={"DUAL_FIFO_RUN_ROOT": str(run_root)})
        self.assertEqual(result.returncode, 0, result.stdout)
        summaries = list(run_root.glob("*/summary.csv"))
        self.assertEqual(len(summaries), 1, result.stdout)
        with summaries[0].open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertIn("jointc2_rsu100_ego200", str(summaries[0]))
        self.assertEqual(Path(rows[0]["agent_config"]).name,
                         "pnp_config_codriving_dualfifo_jointc2_rsu100_ego200_l1.yaml")
        self.assertEqual(json.loads((self.repo / "evaluator_argv.json").read_text())[4],
                         "codriving_dualfifo_jointc2_rsu100_ego200_l1")
        for field in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv"):
            self.assertTrue(Path(rows[0][field]).is_file(), rows[0])

    def test_consecutive_routes_use_distinct_carla_and_traffic_ports(self):
        result = self.run_fixture(routes="0 1")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([row["route"] for row in self.summary()], ["0", "1"])
        self.assertEqual(json.loads((self.repo / "evaluator_argv.json").read_text())[:2],
                         ["1", "40010"])
        self.assertIn("[route] r0 port=40000 tm_port=40005", result.stdout)
        self.assertIn("[route] r1 port=40010 tm_port=40015", result.stdout)

    def test_occupied_later_route_port_aborts_before_any_episode(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 40010))
            listener.listen()
            result = self.run_fixture(routes="0 1")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.counter.exists(), result.stdout)
        self.assertFalse(self.processes.exists(), result.stdout)
        self.assertFalse((self.repo / "results").exists(), result.stdout)

    def test_joint_occupied_later_route_port_allows_first_then_rejects_current(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 40010))
            listener.listen()
            result = self.run_fixture(routes="0 1", rsu=100, ego=200,
                                      runner_options=("--joint",))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(self.counter.read_text(), "1", result.stdout)
        rows = self.summary()
        self.assertEqual([(row["route"], row["countable"]) for row in rows],
                         [("0", "True"), ("1", "False")])
        self.assertIn("[route] r1 port=40010", result.stdout)
        self.assertIn("Address already in use", result.stdout)

    def test_scenario_seeds_and_cohort_are_forwarded_without_ambient_defaults(self):
        result = self.run_fixture(runner_options=(
            "--scenario", "_3", "--carla-seed", "2001",
            "--traffic-seed", "2002", "--cohort", "rsu-full-v1",
        ))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(json.loads((self.repo / "evaluator_argv.json").read_text()), [
            "0", "40000",
            "dualfifo_rsu-full-v1_rsu200_sc3_r0_cseed2001_tseed2002_n2_a1",
            "2", "codriving_dualfifo_rsu200_l1", "_3",
            "v2x_final/town05_short_collab/r0_repeat2", "2001", "2002",
        ])

    def test_invalid_scenario_seed_and_cohort_are_rejected_before_gpu_access(self):
        invalid_options = (
            ("--scenario", "_6"),
            ("--carla-seed", "-1"),
            ("--traffic-seed", "seed"),
            ("--cohort", "bad/name"),
        )
        for options in invalid_options:
            with self.subTest(options=options):
                result = self.run_fixture(runner_options=options)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertFalse(self.counter.exists())

    def test_missing_result_is_retried_and_second_attempt_is_counted(self):
        result = self.run_fixture("retry")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.counter.read_text(), "2")
        self.assertEqual(self.summary()[0]["countable"], "True")
        failures = list((self.repo / "results/dual_fifo_pilot").rglob("failures*.log"))
        self.assertIn("INFRA_INVALID", failures[0].read_text())
        self.assertIn("No such file", failures[0].read_text())

    def test_max_attempts_one_defers_retry_and_records_first_failure(self):
        result = self.run_fixture("retry", runner_options=("--max-attempts", "1"))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(self.counter.read_text(), "1")
        self.assertEqual((self.summary()[0]["status"], self.summary()[0]["countable"]),
                         ("INFRA_INVALID", "False"))
        failures = list((self.repo / "results/dual_fifo_pilot").rglob("failures*.log"))
        self.assertEqual(len(failures[0].read_text().splitlines()), 1)

    def test_max_attempts_rejects_invalid_values_before_gpu_access(self):
        for value in ("0", "3", "no", ""):
            with self.subTest(value=value):
                result = self.run_fixture(runner_options=("--max-attempts", value))
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertFalse(self.counter.exists())

    def test_rejects_duplicate_or_multiline_route_lists_before_evaluation(self):
        for routes in ("0 0", "0\n1"):
            result = run_command(["bash", self.repo / "scripts" / SCRIPT.name,
                                  "0", "40000", "200", "0", routes], extra_env=self.env)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertFalse(self.counter.exists())

    def test_multiple_routes_use_default_repeat_and_record_dirty_provenance(self):
        config = self.repo / CONFIG_DIR / "pnp_config_codriving_dualfifo_rsu200_l1.yaml"
        config.write_text(config.read_text() + "\n# tracked fixture edit\n")
        result = run_command(["bash", self.repo / "scripts" / SCRIPT.name,
                              "0", "40000", "200", "0", "0 1"], extra_env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout)
        rows = self.summary()
        self.assertEqual([row["route"] for row in rows], ["0", "1"])
        self.assertEqual([row["repeat"] for row in rows], ["0", "0"])
        self.assertEqual([row["git_dirty"] for row in rows], ["true", "true"])

    def test_relative_python_override_survives_private_workspace_chdir(self):
        (self.repo / "selected-python").symlink_to(sys.executable)
        result = run_command(["bash", self.repo / "scripts" / SCRIPT.name,
                              "0", "40000", "200", "0", "0"], cwd=self.repo,
                             extra_env=dict(self.env, DUAL_FIFO_PYTHON="./selected-python"))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.counter.read_text(), "1")
        self.assertEqual(self.summary()[0]["countable"], "True")

    def test_infrastructure_failure_never_counts_even_with_global_record(self):
        for mode in ("cuda", "process", "missing", "corrupt", "no_global"):
            with self.subTest(mode=mode):
                result = self.run_fixture(mode)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.counter.read_text(), "10")
        files = list((self.repo / "results/dual_fifo_pilot").rglob("summary*.csv"))
        self.assertEqual(len(files), 5)
        for path in files:
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "INFRA_INVALID")
            self.assertEqual(row["countable"], "False")

    def assert_json_retried_without_process_or_log_failure(self, payload):
        output = self.repo / "results/dual_fifo_pilot"
        previous_summaries = set(output.rglob("summary.csv"))
        previous_calls = int(self.counter.read_text()) if self.counter.exists() else 0
        result = self.run_fixture(extra_env={"FIXTURE_PAYLOAD": json.dumps(payload)})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(int(self.counter.read_text()) - previous_calls, 2)
        created = set(output.rglob("summary.csv")) - previous_summaries
        self.assertEqual(len(created), 1)
        summary = created.pop()
        with summary.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "INFRA_INVALID")
        self.assertEqual(rows[0]["countable"], "False")
        logs = list(summary.parent.glob("eval_*.log"))
        self.assertEqual(len(logs), 2)
        for log in logs:
            self.assertEqual(log.read_text(), "fixture evaluator completed normally\n")

    def test_invalid_json_status_retries_despite_zero_exit_and_benign_log(self):
        for status in ("Started", "INFRA_INVALID", "Failed - Run scenario crashed"):
            with self.subTest(status=status):
                self.assert_json_retried_without_process_or_log_failure(result_fixture(status))
        payload = result_fixture()
        payload["_checkpoint"]["global_record"]["meta"] = {
            "exceptions": [["route_0", 0, "Failed - Run scenario crashed"]]}
        self.assert_json_retried_without_process_or_log_failure(payload)

    def test_malformed_scores_retry_twice_then_write_one_noncountable_row(self):
        for label, payload in invalid_score_fixtures():
            with self.subTest(case=label):
                self.assert_json_retried_without_process_or_log_failure(payload)

    def test_invalid_audits_are_retried_with_zero_evaluator_exit(self):
        faults = ("missing_rsu", "missing_ego", "empty", "header", "truncated", "malformed",
                  "ambiguous", "rsu_delay", "rsu_ready", "rsu_used", "rsu_age", "rsu_zoh",
                  "rsu_startup", "ego_ready", "ego_source", "ego_age", "ego_control", "ego_gap", "rsu_gap", "flush",
                  "ego_startup", "ego_neutral")
        output = self.repo / "results/dual_fifo_pilot"
        for fault in faults:
            with self.subTest(fault=fault):
                before = set(output.rglob("summary.csv"))
                calls = int(self.counter.read_text()) if self.counter.exists() else 0
                positive_ego = fault in ("ego_startup", "ego_neutral")
                result = self.run_fixture(extra_env={"FIXTURE_AUDIT_FAULT": fault},
                                          rsu=0 if positive_ego else 200, ego=200 if positive_ego else 0)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertEqual(int(self.counter.read_text()) - calls, 2)
                summary = (set(output.rglob("summary.csv")) - before).pop()
                with summary.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual((rows[0]["status"], rows[0]["countable"]), ("INFRA_INVALID", "False"))
                for log in summary.parent.glob("eval_*.log"):
                    self.assertIn("fixture evaluator completed normally", log.read_text())

    def test_carla_start_failure_retries_twice_without_evaluation(self):
        result = self.run_fixture(extra_env={"FIXTURE_NC_EXIT": "1"})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.counter.exists())
        self.assertEqual(self.summary()[0]["countable"], "False")
        failures = list((self.repo / "results/dual_fifo_pilot").rglob("failures*.log"))
        self.assertEqual(failures[0].read_text().count("carla_start_failed"), 2)

    def test_busy_or_unreadable_gpu_aborts_before_evaluation(self):
        for gpu in ("6, 0", "0, 51", "not available"):
            result = self.run_fixture(extra_env={"FIXTURE_GPU": gpu})
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(self.counter.exists())
            self.assertIn("[ABORT]", result.stdout)

    def test_joint_mode_allows_residual_utilization_when_memory_is_idle(self):
        result = self.run_fixture(
            rsu=200, ego=300, runner_options=("--joint",),
            extra_env={"FIXTURE_GPU": "20, 1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.counter.read_text(), "1")

    def test_check_config_validates_yaml_not_just_arm_name(self):
        import yaml
        config = self.repo / CONFIG_DIR / "pnp_config_codriving_dualfifo_rsu200_l1.yaml"
        original = yaml.safe_load(config.read_text())
        for key, value in (("rsu_fifo_delay_ms", 0), ("ego_output_fifo_delay_ms", 200),
                           ("tau_ego_ms", 1), ("latency_audit", False), ("ego_num", 2)):
            changed = dict(original, simulation=dict(original["simulation"], **{key: value}))
            config.write_text(yaml.safe_dump(changed))
            result = run_command(["bash", self.repo / "scripts" / SCRIPT.name,
                                  "--check-config", "200", "0"])
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(key, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
