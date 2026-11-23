from __future__ import print_function

import argparse
import copy
import errno
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "ditl_packet_loss_runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("packet_loss_runner_test", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PacketLossRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()
        self.bundle = {
            "library_root": "/var/tmp/library",
            "catalog_path": "/var/tmp/library/catalog.json",
            "bundle_path": "/var/tmp/library/v1/bundles/" + "a" * 64,
            "feature_id": "a" * 64,
            "manifest_sha256": "b" * 64,
        }

    def _candidate(self, suffix="test"):
        return Path("/var/tmp/codriving-ditl") / (
            "packet-loss-stage1-20260926T080000Z-%s-%d" % (suffix, os.getpid()))

    def _snapshot(self, listening=None):
        def gpu(index, util, memory, pid):
            return {"index": index, "uuid": "GPU-%d" % pid,
                    "utilization_percent": util, "memory_used_mib": memory,
                    "compute_pids": [pid],
                    "compute_processes": [{"pid": pid, "process_name": "train",
                                            "used_memory_mib": memory}]}
        return {"schema": "ditl.packet-loss-stage1-resource-snapshot.v1",
                "gpus": {"6": gpu(6, 99, 48000, 116),
                         "7": gpu(7, 88, 47000, 227)},
                "listening_ports": list(listening or []),
                "protected": {"head": "a" * 40, "status": "",
                              "supervisor": {"pid": 2088364, "active": True,
                                             "start_token": "123"}}}

    def test_build_task_config_freezes_profile_cadence_bundle_and_single_factor(self):
        for profile, path in self.runner.PROFILE_CONFIGS.items():
            with self.subTest(profile=profile):
                source = yaml.safe_load(path.read_text())
                frozen = copy.deepcopy(source)
                config = self.runner.build_task_config(path, self.bundle, 0.2)
                self.assertEqual(frozen, source)
                simulation = config["simulation"]
                self.assertEqual(2, simulation["skip_frames"])
                self.assertEqual(0, simulation["action_fifo_delay_ms"])
                self.assertEqual(0, simulation.get("rsu_fifo_delay_ms", 0))
                self.assertNotIn("rsu_fifo_delay_ms", simulation)
                self.assertNotIn("latency_inject_ms", simulation)
                self.assertNotIn("ditl_spatial_offset", simulation)
                replay = simulation["ditl_replay"]
                self.assertEqual("cycle", replay["exhaustion"])
                self.assertEqual(2000, replay["seed"])
                self.assertIsNone(replay["ttl_ms"])
                self.assertIsNone(replay["deadline_ms"])
                self.assertEqual(self.bundle["bundle_path"], replay["bundle"])
                impairment = replay["impairment"]
                self.assertEqual(0.2, impairment["drop_rate"])
                for field in ("extra_delay_ms", "jitter_ms", "reorder_rate",
                              "duplicate_rate"):
                    self.assertEqual(0, impairment[field])
                self.assertFalse(impairment["burst"]["enabled"])
                from simulation.leaderboard.team_code.closedloop.ditl_replay_config import resolve_ditl_replay_config
                self.assertIsNotNone(resolve_ditl_replay_config(simulation))

    def test_new_cohort_is_exclusive_and_confined(self):
        candidate = self._candidate()
        self.assertEqual(candidate, self.runner.validate_new_cohort_path(candidate))
        for bad in (Path("relative"), Path("/var/tmp/codriving-ditl/task6-old")):
            with self.assertRaises(ValueError):
                self.runner.validate_new_cohort_path(bad)
        candidate.mkdir()
        try:
            with self.assertRaises(FileExistsError):
                self.runner.validate_new_cohort_path(candidate)
        finally:
            candidate.rmdir()
        target, link = self._candidate("target"), self._candidate("link")
        target.mkdir()
        try:
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.runner.validate_new_cohort_path(link)
        finally:
            link.unlink(); target.rmdir()

    def test_parser_collects_reuse_cohorts(self):
        args = self.runner.parse_args([
            "--dry-run", "--cohort-id",
            "packet-loss-stage1-20260928T120000Z-reuse",
            "--reuse-cohort", "/var/tmp/codriving-ditl/old-a",
            "--reuse-cohort", "/data/jichengzhi_v2x/ditl/old-b"])
        self.assertEqual([
            Path("/var/tmp/codriving-ditl/old-a"),
            Path("/data/jichengzhi_v2x/ditl/old-b")], args.reuse_cohort)

    def test_new_cohort_accepts_only_configured_output_roots(self):
        name = "packet-loss-stage1-20260928T120000Z-data-%d" % os.getpid()
        data_candidate = Path("/data/jichengzhi_v2x/ditl") / name
        self.assertEqual(data_candidate,
                         self.runner.validate_new_cohort_path(data_candidate))
        with self.assertRaises(ValueError):
            self.runner.validate_new_cohort_path(Path("/data") / name)
        with self.assertRaises(ValueError):
            self.runner.validate_new_cohort_path(Path("/tmp") / name)

    def test_parser_exposes_explicit_output_root(self):
        args = self.runner.parse_args([
            "--dry-run", "--cohort-id",
            "packet-loss-stage1-20260928T120000Z-data",
            "--output-root", "/data/jichengzhi_v2x/ditl"])
        self.assertEqual(Path("/data/jichengzhi_v2x/ditl"), args.output_root)

    def test_dry_run_uses_explicit_data_output_root(self):
        cohort_id = "packet-loss-stage1-20260928T120000Z-data"
        args = self.runner.parse_args([
            "--dry-run", "--cohort-id", cohort_id,
            "--output-root", "/data/jichengzhi_v2x/ditl",
            "--capture-report", "/tmp/capture.json"])
        emitted = []
        with mock.patch.object(self.runner, "resolve_idle_bundle",
                               return_value=self.bundle), \
             mock.patch.object(self.runner, "validate_frozen_inputs"), \
             mock.patch.object(self.runner, "capture_resource_snapshot",
                               return_value=self._snapshot()), \
             mock.patch.object(self.runner, "validate_resource_snapshot"), \
             mock.patch.object(self.runner, "wait_for_ports_bindable"):
            self.assertEqual(0, self.runner.execute(args, emitted.append))
        payload = json.loads(emitted[-1])
        self.assertEqual("/data/jichengzhi_v2x/ditl/" + cohort_id,
                         payload["cohort"])

    def test_dry_run_checks_without_creating_cohort_or_processes(self):
        candidate = self._candidate("dry")
        args = self.runner.parse_args(["--dry-run", "--cohort-id", candidate.name,
                                      "--capture-report", "/tmp/capture.json"])
        emitted = []
        with mock.patch.object(self.runner, "resolve_idle_bundle", return_value=self.bundle), \
             mock.patch.object(self.runner, "validate_frozen_inputs"), \
             mock.patch.object(self.runner, "capture_resource_snapshot",
                               return_value=self._snapshot()), \
             mock.patch.object(self.runner, "validate_resource_snapshot"), \
             mock.patch.object(self.runner, "wait_for_ports_bindable") as wait_ports, \
             mock.patch.object(self.runner.subprocess, "Popen") as popen:
            self.assertEqual(0, self.runner.execute(args, emitted.append))
        self.assertFalse(candidate.exists()); popen.assert_not_called()
        payload = json.loads(emitted[-1])
        self.assertEqual((45, 4, 180, 20),
                         (payload["cells"], payload["arms"], payload["tasks"],
                          payload["first_pass_batches"]))
        expected_first = [62000, 62001, 62002, 62005,
                          62100, 62101, 62102, 62105]
        expected_retry = [63000, 63001, 63002, 63005,
                          63100, 63101, 63102, 63105]
        self.assertEqual(expected_first + expected_retry, payload["all_ports"])
        self.assertEqual(self._snapshot(), payload["resource_preflight"])
        self.assertEqual(20, len(payload["first_pass_commands"]))
        self.assertEqual(expected_retry, payload["retry_ports"])
        port_labels = ("--world-port", "--streaming-port",
                       "--secondary-port", "--traffic-port")
        for index, command in enumerate(payload["first_pass_commands"]):
            actual = [int(command[command.index(label) + 1]) for label in port_labels]
            offset = 0 if index % 2 == 0 else 4
            self.assertEqual(expected_first[offset:offset + 4], actual)
        self.assertIn("argv", payload)
        wait_ports.assert_called_once_with(self.runner.ALL_PORTS)

    def test_reuse_dry_run_reports_source_cohort_counts(self):
        candidate = self._candidate("reuse-dry")
        args = self.runner.parse_args([
            "--dry-run", "--cohort-id", candidate.name,
            "--capture-report", "/tmp/capture.json",
            "--reuse-cohort", "/data/jichengzhi_v2x/ditl/packet-loss-stage1-20260928T123554Z-r4",
            "--reuse-cohort", "/var/tmp/codriving-ditl/packet-loss-stage1-20260928T065846Z-r2",
            "--reuse-cohort", "/var/tmp/codriving-ditl/packet-loss-stage1-20260926T120101Z-r1",
        ])
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        missing = [task["key"] for task in manifest["tasks"][:2]]
        selected = {
            manifest["tasks"][2]["key"]: {
                "source_cohort": "/data/jichengzhi_v2x/ditl/packet-loss-stage1-20260928T123554Z-r4",
                "source_kind": "canonical",
            },
            manifest["tasks"][3]["key"]: {
                "source_cohort": "/var/tmp/codriving-ditl/packet-loss-stage1-20260928T065846Z-r2",
                "source_kind": "canonical",
            },
            manifest["tasks"][4]["key"]: {
                "source_cohort": "/var/tmp/codriving-ditl/packet-loss-stage1-20260926T120101Z-r1",
                "source_kind": "legacy_r1",
            },
        }
        plan = {
            "selected": selected,
            "selected_count": len(selected),
            "missing_task_keys": missing,
            "source_counts": {"canonical": 2, "legacy_r1": 1},
        }
        emitted = []
        with mock.patch.object(self.runner, "resolve_idle_bundle", return_value=self.bundle), \
             mock.patch("scripts.ditl_packet_loss_reuse.build_reuse_plan",
                        return_value=plan), \
             mock.patch.object(self.runner, "validate_frozen_inputs"), \
             mock.patch.object(self.runner, "capture_resource_snapshot",
                               return_value=self._snapshot()), \
             mock.patch.object(self.runner, "validate_resource_snapshot"), \
             mock.patch.object(self.runner, "wait_for_ports_bindable"):
            self.assertEqual(0, self.runner.execute(args, emitted.append))

        payload = json.loads(emitted[-1])
        self.assertEqual({
            "packet-loss-stage1-20260928T123554Z-r4": 1,
            "packet-loss-stage1-20260928T065846Z-r2": 1,
            "packet-loss-stage1-20260926T120101Z-r1": 1,
        }, payload["reuse_cohort_counts"])
        self.assertEqual(2, payload["scheduled_tasks"])
        self.assertEqual(1, payload["first_pass_batches"])

    def test_busy_gpus_are_recorded_and_do_not_block_launch(self):
        snapshot = self._snapshot()
        self.assertIs(snapshot, self.runner.validate_resource_snapshot(
            snapshot, (6, 7), self.runner.ALL_PORTS))
        metadata = self.runner.build_metadata({}, snapshot, "c" * 64)
        self.assertEqual(snapshot, metadata["resource_preflight"])

    def test_missing_gpu_or_wrong_assignment_is_rejected(self):
        self.assertEqual(2088364, self.runner.PROTECTED_SUPERVISOR_PID)
        self.assertEqual(frozenset((2088364,)), self.runner.PROTECTED_PIDS)
        snapshot = self._snapshot(); del snapshot["gpus"]["7"]
        with self.assertRaises(ValueError):
            self.runner.validate_resource_snapshot(snapshot, (6, 7), ())
        stopped = self._snapshot()
        stopped["protected"]["supervisor"] = {
            "pid": 2088364, "active": False, "start_token": None}
        self.assertIs(stopped, self.runner.validate_resource_snapshot(
            stopped, (6, 7), ()))
        inconsistent = self._snapshot()
        inconsistent["protected"]["supervisor"] = {
            "pid": 2088364, "active": False, "start_token": "stale"}
        with self.assertRaises(ValueError):
            self.runner.validate_resource_snapshot(inconsistent, (6, 7), ())
        bad = (("first", "worker6", 7, self.runner.FIRST_ASSIGNMENTS[0]["ports"]),
               ("first", "worker0", 0, (62000, 62001, 62002, 62005)),
               ("retry", "worker6", 6, self.runner.RETRY_ASSIGNMENTS[1]["ports"]))
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.runner.validate_launch_coordinates(*row)

    def test_gpu_snapshot_query_failure_fails_closed(self):
        with mock.patch.object(self.runner.subprocess, "check_output",
                               side_effect=OSError("nvidia-smi failed")):
            with self.assertRaises(OSError):
                self.runner.capture_resource_snapshot((6, 7), ())
        invalid = self._snapshot(); invalid["gpus"]["6"].pop("compute_processes")
        with self.assertRaises(ValueError):
            self.runner.validate_resource_snapshot(invalid, (6, 7), ())

    def test_occupied_ports_fail_closed_for_the_selected_round(self):
        for occupied in (62000, 62105, 63000, 63105):
            with self.subTest(port=occupied), self.assertRaises(ValueError):
                self.runner.validate_resource_snapshot(
                    self._snapshot([occupied]), (6, 7), (occupied,))
        class RefusingSocket(object):
            def setsockopt(self, *_args): pass
            def bind(self, _address): raise OSError("occupied")
            def close(self): pass
        with self.assertRaises(RuntimeError):
            self.runner.bind_guard_ports((62000,), socket_factory=lambda *_: RefusingSocket())

    def test_port_release_wait_retries_eaddrinuse_until_bindable(self):
        now = [0.0]
        sleeps = []
        calls = [0]

        class Probe(object):
            def setsockopt(self, *_args): pass
            def bind(self, _address): pass
            def close(self): pass

        def socket_factory(*_args):
            calls[0] += 1
            if calls[0] <= 2:
                raise OSError(errno.EADDRINUSE, "still releasing")
            return Probe()

        def pause(delay):
            sleeps.append(delay)
            now[0] += delay

        self.runner.wait_for_ports_bindable(
            (62000,), timeout_s=1.0, poll_s=0.25,
            socket_factory=socket_factory, monotonic=lambda: now[0],
            pause=pause)
        self.assertEqual([0.25, 0.25], sleeps)
        self.assertEqual(4, calls[0])

    def test_port_release_timeout_is_clear_and_prevents_popen(self):
        now = [0.0]

        def occupied(*_args):
            raise OSError(errno.EADDRINUSE, "persistently occupied")

        def pause(delay):
            now[0] += delay

        with self.assertRaisesRegex(
                RuntimeError, "ports 62000.*did not become bindable within 0.5s"):
            self.runner.wait_for_ports_bindable(
                (62000,), timeout_s=0.5, poll_s=0.25,
                socket_factory=occupied, monotonic=lambda: now[0], pause=pause)

        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, _owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])
            with mock.patch.object(self.runner, "capture_resource_snapshot",
                                   return_value=self._snapshot()), \
                 mock.patch.object(self.runner, "validate_resource_snapshot"), \
                 mock.patch.object(self.runner, "verify_cohort_frozen"), \
                 mock.patch.object(self.runner, "wait_for_ports_bindable",
                                   side_effect=RuntimeError(
                                       "packet-loss ports did not become bindable")), \
                 mock.patch.object(self.runner.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "did not become bindable"):
                    launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
            popen.assert_not_called()

    def test_port_wait_exits_when_peer_cancels_during_poll_sleep(self):
        now = [0.0]
        attempts = [0]

        def occupied(*_args):
            attempts[0] += 1
            raise OSError(errno.EADDRINUSE, "still occupied")

        def peer_cancels(delay):
            now[0] += delay
            self.runner._STOP.set()

        self.runner._STOP.clear()
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.runner.wait_for_ports_bindable(
                    (62000,), timeout_s=1.0, poll_s=0.25,
                    socket_factory=occupied, monotonic=lambda: now[0],
                    pause=peer_cancels)
            self.assertEqual(1, attempts[0])
            self.assertEqual(0.25, now[0])
        finally:
            self.runner._STOP.clear()

    def test_launch_rechecks_peer_stop_after_wait_before_popen(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, _owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])

            def peer_stops_after_wait(_ports):
                self.runner._STOP.set()

            self.runner._STOP.clear()
            try:
                with mock.patch.object(self.runner, "capture_resource_snapshot",
                                       return_value=self._snapshot()), \
                     mock.patch.object(self.runner, "validate_resource_snapshot"), \
                     mock.patch.object(self.runner, "verify_cohort_frozen"), \
                     mock.patch.object(self.runner, "wait_for_ports_bindable",
                                       side_effect=peer_stops_after_wait), \
                     mock.patch.object(self.runner.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(RuntimeError, "cancelled"):
                        launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
                popen.assert_not_called()
            finally:
                self.runner._STOP.clear()

    def test_peer_stop_during_launch_port_wait_never_reaches_popen(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        real_wait = self.runner.wait_for_ports_bindable
        now = [0.0]
        attempts = [0]

        def occupied(*_args):
            attempts[0] += 1
            raise OSError(errno.EADDRINUSE, "still occupied")

        def peer_cancels(delay):
            now[0] += delay
            self.runner._STOP.set()

        def wait_with_peer_cancel(ports):
            return real_wait(
                ports, timeout_s=1.0, poll_s=0.25,
                socket_factory=occupied, monotonic=lambda: now[0],
                pause=peer_cancels)

        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, _owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])
            self.runner._STOP.clear()
            try:
                with mock.patch.object(self.runner, "capture_resource_snapshot",
                                       return_value=self._snapshot()), \
                     mock.patch.object(self.runner, "validate_resource_snapshot"), \
                     mock.patch.object(self.runner, "verify_cohort_frozen"), \
                     mock.patch.object(self.runner, "wait_for_ports_bindable",
                                       side_effect=wait_with_peer_cancel), \
                     mock.patch.object(self.runner.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(RuntimeError, "cancelled"):
                        launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
                popen.assert_not_called()
                self.assertEqual(1, attempts[0])
            finally:
                self.runner._STOP.clear()

    def test_initial_and_retry_resource_gates_use_same_bounded_wait(self):
        self.assertEqual(120.0, self.runner.PORT_RELEASE_TIMEOUT_S)
        retry_ports = tuple(port for assignment in self.runner.RETRY_ASSIGNMENTS
                            for port in assignment["ports"])
        snapshot = self._snapshot(listening=[retry_ports[0]])
        with mock.patch.object(self.runner, "capture_resource_snapshot",
                               return_value=snapshot), \
             mock.patch.object(self.runner, "validate_resource_snapshot") as validate, \
             mock.patch.object(self.runner, "wait_for_ports_bindable") as wait_ports:
            self.assertIs(snapshot, self.runner.validate_round_resources(
                self.runner.RETRY_ASSIGNMENTS))
        validate.assert_called_once_with(snapshot, (6, 7), ())
        wait_ports.assert_called_once_with(retry_ports)

    def test_first_and_retry_ports_are_disjoint_and_exact(self):
        first = tuple(p for row in self.runner.FIRST_ASSIGNMENTS for p in row["ports"])
        retry = tuple(p for row in self.runner.RETRY_ASSIGNMENTS for p in row["ports"])
        self.assertEqual((62000, 62001, 62002, 62005,
                          62100, 62101, 62102, 62105), first)
        self.assertEqual((63000, 63001, 63002, 63005,
                          63100, 63101, 63102, 63105), retry)
        self.assertTrue(all(port > 60999 for port in first + retry))
        self.assertFalse(set(first).intersection(retry))
        self.assertEqual(first + retry, self.runner.ALL_PORTS)

    def test_batch_command_binds_carla_and_inference_to_worker_gpu(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        cohort = Path("/var/tmp/codriving-ditl/packet-loss-stage1-20260926T080000Z-cmd")
        command = self.runner.build_batch_command(
            batch, self.runner.FIRST_ASSIGNMENTS[0], cohort, 1,
            task_map={task["key"]: task for task in manifest["tasks"]})
        self.assertEqual(str(self.runner.EVAL_BATCH), command[0])
        self.assertIn("worker6", command); self.assertIn("6", command)
        self.assertEqual(9, command.count("--task"))
        for key in batch["task_keys"]: self.assertIn(key, command)
        coordinates = {
            label: command[command.index(label) + 1]
            for label in ("--world-port", "--streaming-port",
                          "--secondary-port", "--traffic-port")
        }
        self.assertEqual({"--world-port": "62000", "--streaming-port": "62001",
                          "--secondary-port": "62002", "--traffic-port": "62005"},
                         coordinates)

    def test_batch_shell_allowlist_accepts_only_new_fixed_coordinates(self):
        source = self.runner.EVAL_BATCH.read_text(encoding="utf-8")
        expected = (
            "1:worker6:6:62000:62001:62002:62005",
            "1:worker7:7:62100:62101:62102:62105",
            "2:worker6:6:63000:63001:63002:63005",
            "2:worker7:7:63100:63101:63102:63105",
        )
        obsolete = (
            "1:worker6:6:47000:47001:47002:47005",
            "1:worker7:7:47100:47101:47102:47105",
            "2:worker6:6:48000:48001:48002:48005",
            "2:worker7:7:48100:48101:48102:48105",
        )
        for coordinates in expected:
            self.assertIn(coordinates, source)
        for coordinates in obsolete:
            self.assertNotIn(coordinates, source)

    def test_batch_shell_accepts_both_authorized_cohort_roots(self):
        source = self.runner.EVAL_BATCH.read_text(encoding="utf-8")
        self.assertIn(
            "/var/tmp/codriving-ditl/packet-loss-stage1-*|"
            "/data/jichengzhi_v2x/ditl/packet-loss-stage1-*",
            source)

    def test_two_workers_consume_shared_queue_without_gpu_idle_wait(self):
        batches = [{"id": "b%d" % i, "task_keys": ["k%d" % i]} for i in range(4)]
        active, workers, synchronized, lock = set(), set(), set(), threading.Lock()
        gate = threading.Barrier(2)
        def launch(batch, assignment, attempt):
            with lock:
                self.assertNotIn(assignment["id"], active)
                active.add(assignment["id"]); workers.add(assignment["id"])
                first_for_worker = assignment["id"] not in synchronized
                synchronized.add(assignment["id"])
            if first_for_worker: gate.wait(timeout=2)
            with lock: active.remove(assignment["id"])
            return {"batch_id": batch["id"], "returncode": 0,
                    "worker": assignment["id"], "attempt": attempt}
        with mock.patch.object(self.runner.time, "sleep") as sleep:
            rows = self.runner.run_round(batches, self.runner.FIRST_ASSIGNMENTS, 1, launch)
        self.assertEqual(4, len(rows)); self.assertEqual({"worker6", "worker7"}, workers)
        sleep.assert_not_called()

    def test_worker_gate_failure_stops_shared_queue_before_more_batches_launch(self):
        batches = [{"id": "b%d" % index, "task_keys": ["k%d" % index]}
                   for index in range(6)]
        entered = threading.Barrier(2)
        release_peer = threading.Event()
        launched = []
        lock = threading.Lock()
        def launch(batch, assignment, _attempt):
            with lock:
                launched.append((batch["id"], assignment["id"]))
            entered.wait(timeout=2)
            if assignment["id"] == "worker6":
                raise RuntimeError("resource gate failed")
            self.assertTrue(release_peer.wait(timeout=2))
            return {"batch_id": batch["id"]}
        cancel_calls = []
        def cancel_owned():
            cancel_calls.append(True)
            release_peer.set()
        launch.cancel_owned = cancel_owned
        self.runner._STOP.clear()
        try:
            with self.assertRaisesRegex(RuntimeError, "resource gate failed"):
                self.runner.run_round(batches, self.runner.FIRST_ASSIGNMENTS, 1, launch)
            self.assertEqual(2, len(launched))
            self.assertEqual([True], cancel_calls)
            self.assertTrue(self.runner._STOP.is_set())
        finally:
            self.runner._STOP.clear()

    def test_peer_cancellation_does_not_mask_primary_launch_failure(self):
        batches = [{"id": "b0", "task_keys": ["k0"]},
                   {"id": "b1", "task_keys": ["k1"]}]
        entered = threading.Barrier(2)

        def launch(_batch, assignment, _attempt):
            entered.wait(timeout=2)
            if assignment["id"] == "worker6":
                raise RuntimeError("primary resource failure")
            raise self.runner.RoundCancelled("peer cancelled")

        launch.stop_and_cancel = self.runner._STOP.set
        self.runner._STOP.clear()
        try:
            with mock.patch.object(
                    self.runner, "as_completed",
                    side_effect=lambda futures: reversed(futures)):
                with self.assertRaisesRegex(RuntimeError, "primary resource failure"):
                    self.runner.run_round(
                        batches, self.runner.FIRST_ASSIGNMENTS, 1, launch)
        finally:
            self.runner._STOP.clear()

    def test_retry_contains_only_infrastructure_failures_and_uses_retry_ports(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        keys = [task["key"] for task in manifest["tasks"]]
        events = []
        first = {key: {"task_key": key, "classification": "countable"} for key in keys}
        first[keys[1]] = {"task_key": keys[1], "classification": "infrastructure_failure"}
        def run_round(batches, assignments, attempt, _launch):
            events.append((attempt, [p for row in assignments for p in row["ports"]],
                           [key for batch in batches for key in batch["task_keys"]]))
            return []
        with mock.patch.object(self.runner, "run_round", side_effect=run_round), \
             mock.patch.object(self.runner, "validate_round_resources"), \
             mock.patch.object(self.runner, "classify_attempt",
                               side_effect=[first, {keys[1]: dict(first[keys[1]], attempt=2)}]), \
             mock.patch.object(self.runner, "finalize_task_results", return_value=first), \
             mock.patch.object(self.runner, "generate_artifacts", return_value={}):
            self.runner.execute_two_rounds(manifest, Path("/tmp/cohort"), lambda *_: {})
        self.assertEqual((1, 2), (events[0][0], events[1][0]))
        self.assertEqual([keys[1]], events[1][2])
        self.assertEqual([p for row in self.runner.RETRY_ASSIGNMENTS for p in row["ports"]],
                         events[1][1])

    def test_completion_first_pass_runs_only_missing_tasks(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        keys = [task["key"] for task in manifest["tasks"]]
        missing = [keys[1], keys[7]]
        first = {key: {"task_key": key, "classification": "countable"}
                 for key in keys}
        events = []

        def run_round(batches, assignments, attempt, _launch):
            events.append((attempt, [key for batch in batches
                                     for key in batch["task_keys"]]))
            return []

        with mock.patch.object(self.runner, "run_round", side_effect=run_round), \
             mock.patch.object(self.runner, "classify_attempt", return_value=first), \
             mock.patch.object(self.runner, "finalize_task_results", return_value=first), \
             mock.patch.object(self.runner, "generate_artifacts", return_value={}):
            self.runner.execute_two_rounds(
                manifest, Path("/tmp/cohort"), lambda *_: {},
                first_task_keys=missing)
        self.assertEqual([(1, missing)], events)

    def test_two_round_completion_rejects_incomplete_finalized_set(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        keys = [task["key"] for task in manifest["tasks"]]
        first = {key: {"task_key": key, "classification": "countable"}
                 for key in keys}
        with mock.patch.object(self.runner, "run_round"), \
             mock.patch.object(self.runner, "classify_attempt", return_value=first), \
             mock.patch.object(self.runner, "finalize_task_results", return_value={}), \
             mock.patch.object(self.runner, "generate_artifacts") as artifacts:
            with self.assertRaisesRegex(RuntimeError, "finalized task set"):
                self.runner.execute_two_rounds(
                    manifest, Path("/tmp/cohort"), lambda *_: {})
        artifacts.assert_not_called()

    def test_signal_cleanup_matches_pid_start_token_parent_and_cohort(self):
        cohort = Path("/tmp/cohort").resolve()
        base = {"pid": 9001, "ppid": os.getpid(), "pgid": 9001, "sid": 9001,
                "start_token": "abc", "cohort": str(cohort)}
        current = {k: base[k] for k in ("pid", "ppid", "pgid", "sid", "start_token")}
        signaled = []
        self.runner.terminate_owned_processes(
            [base], cohort, inspect=lambda _pid: current,
            signal_group=lambda pgid, sig: signaled.append((pgid, sig)), pause=lambda _: None)
        self.assertEqual(2, len(signaled))
        for change in ({"start_token": "reused"}, {"ppid": 1}, {"cohort": "/tmp/other"},
                       {"pid": 2088364, "pgid": 2088364, "sid": 2088364}):
            record = dict(base, **change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.runner.terminate_owned_processes(
                    [record], cohort, inspect=lambda _pid: current,
                    signal_group=lambda *_: None, pause=lambda _: None)

    def test_stale_record_does_not_prevent_later_owned_process_cleanup(self):
        cohort = Path("/tmp/cohort").resolve()
        stale = {"pid": 9001, "ppid": os.getpid(), "pgid": 9001, "sid": 9001,
                 "start_token": "old", "cohort": str(cohort)}
        valid = {"pid": 9002, "ppid": os.getpid(), "pgid": 9002, "sid": 9002,
                 "start_token": "valid", "cohort": str(cohort)}
        current = {
            9001: dict(stale, start_token="reused"),
            9002: dict(valid),
        }
        signaled = []
        with self.assertRaisesRegex(RuntimeError, "stale or foreign"):
            self.runner.terminate_owned_processes(
                [stale, valid], cohort, inspect=lambda pid: current[pid],
                signal_group=lambda pgid, sig: signaled.append((pgid, sig)),
                pause=lambda _: None)
        self.assertEqual([(9002, signal.SIGTERM), (9002, signal.SIGKILL)], signaled)
        self.assertNotIn(9001, [pgid for pgid, _sig in signaled])

    def test_file_process_and_gpu_snapshot_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"; source.write_bytes(b"payload")
            self.assertEqual(hashlib.sha256(b"payload").hexdigest(),
                             self.runner.sha_file(source))
            created = root / "created.json"
            self.runner._canonical_write_new(created, {"b": 2, "a": 1})
            self.assertEqual('{"a":1,"b":2}\n', created.read_text())
            replaced = root / "replace.json"
            self.runner._atomic_json(replaced, {"state": "ok"})
            self.assertEqual("ok", json.loads(replaced.read_text())["state"])
        current = self.runner.inspect_process(os.getpid())
        self.assertEqual(os.getpid(), current["pid"])
        self.assertTrue(self.runner._supervisor_snapshot(os.getpid())["active"])
        self.assertFalse(self.runner._supervisor_snapshot(99999999)["active"])
        calls = ["6, GPU-six, 97, 48000\n", "111, train.py, 47000\n"]
        with mock.patch.object(self.runner.subprocess, "check_output", side_effect=calls):
            row = self.runner._gpu_snapshot(6)
        self.assertEqual((6, [111], 97),
                         (row["index"], row["compute_pids"], row["utilization_percent"]))
        with mock.patch.object(self.runner.subprocess, "check_output",
                               return_value="7, GPU-seven, 1, 5\n"):
            with self.assertRaises(RuntimeError): self.runner._gpu_snapshot(6)

    def test_capture_snapshot_records_protected_source_ports_and_busy_processes(self):
        snapshots = iter(({"head": "protected", "status": ""},
                          {"head": "source", "status": ""}))
        with mock.patch.object(self.runner, "_git_snapshot", side_effect=lambda _p: next(snapshots)), \
             mock.patch.object(self.runner, "_supervisor_snapshot",
                               return_value={"pid": 2088364, "active": True,
                                             "start_token": "token"}), \
             mock.patch.object(self.runner, "_gpu_snapshot",
                               side_effect=lambda gpu: self._snapshot()["gpus"][str(gpu)]), \
             mock.patch.object(self.runner, "_listening_ports", return_value=[1234]):
            snapshot = self.runner.capture_resource_snapshot((6, 7), (62000,))
        self.assertEqual("protected", snapshot["protected"]["head"])
        self.assertEqual("source", snapshot["source"]["head"])
        self.assertEqual([1234], snapshot["listening_ports"])

    def test_idle_bundle_and_capture_report_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"; report.write_text("{}\n")
            with self.assertRaises(ValueError):
                self.runner.resolve_idle_bundle(report)
        bundle = self.runner.resolve_idle_bundle(self.runner.DEFAULT_CAPTURE_REPORT)
        self.assertEqual(self.runner.EXPECTED_IDLE_FEATURE_ID, bundle["feature_id"])
        self.assertEqual(self.runner.EXPECTED_IDLE_MANIFEST_SHA256,
                         bundle["manifest_sha256"])

    def test_validate_frozen_inputs_hashes_routes_scenarios_and_rejects_dirty_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "scripts").mkdir()
            (root / "scripts/ditl_packet_loss_manifest.py").write_text("# manifest\n")
            (root / "scripts/ditl_packet_loss_result.py").write_text("# result\n")
            (root / "scripts/ditl_packet_loss_reuse.py").write_text("# reuse\n")
            eval_batch = root / "scripts/eval.sh"; eval_batch.write_text("#!/bin/sh\n")
            profile = root / "profile.yaml"
            profile.write_text("simulation: {}\nperception:\n  perception_model_dir: models/perception\nplanning:\n  planner_model_checkpoint: models/planner.ckpt\n")
            (root / "models/perception").mkdir(parents=True)
            (root / "models/perception/model.bin").write_bytes(b"model")
            (root / "models/planner.ckpt").write_bytes(b"planner")
            capture = root / "capture.json"; capture.write_text("{}\n")
            catalog = root / "catalog.json"; catalog.write_text("{}\n")
            bundle = root / "bundle"; bundle.mkdir(); (bundle / "manifest.json").write_text("{}\n")
            routes = root / "simulation/leaderboard/data/evaluation_routes"; routes.mkdir(parents=True)
            scenarios = root / "simulation/leaderboard/leaderboard/scenarios"; scenarios.mkdir(parents=True)
            (routes / "town05_short_r2.xml").write_text("<routes/>\n")
            (scenarios / "scenario_parameter_1.yaml").write_text("{}\n")
            manifest = {"tasks": [{"route": 2, "scenario": "_1"}]}
            bundle_map = dict(self.bundle, catalog_path=str(catalog), bundle_path=str(bundle))
            with mock.patch.object(self.runner, "ROOT", root), \
                 mock.patch.object(self.runner, "EVAL_BATCH", eval_batch), \
                 mock.patch.object(self.runner, "PROFILE_CONFIGS", {"8_10": profile}), \
                 mock.patch.object(self.runner, "_git_snapshot",
                                   return_value={"head": "abc", "status": ""}):
                hashes = self.runner.validate_frozen_inputs(manifest, capture, bundle_map)
            self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),
                             hashes[str(capture)])
            with mock.patch.object(self.runner, "ROOT", root), \
                 mock.patch.object(self.runner, "EVAL_BATCH", eval_batch), \
                 mock.patch.object(self.runner, "PROFILE_CONFIGS", {"8_10": profile}), \
                 mock.patch.object(self.runner, "_git_snapshot",
                                   return_value={"head": "abc", "status": " M dirty"}):
                with self.assertRaises(RuntimeError):
                    self.runner.validate_frozen_inputs(manifest, capture, bundle_map)

    def test_create_cohort_freezes_documents_and_configs(self):
        cohort = self._candidate("create")
        partition = self.runner.build_partition()
        manifest = self.runner.build_stage1_manifest(partition)
        args = argparse.Namespace(capture_report=Path("/tmp/capture"), argv=["--run"])
        try:
            self.runner._create_cohort(cohort, partition, manifest, self.bundle,
                                       self._snapshot(), {"input": "f" * 64}, args)
            self.assertEqual(180, len(list((cohort / "configs").glob("*.yaml"))))
            metadata = json.loads((cohort / "metadata.json").read_text())
            self.assertEqual(180, metadata["task_count"])
            self.assertEqual(20, metadata["batch_count"])
            self.assertEqual(0, (cohort / "manifest.json").stat().st_mode & 0o222)
            commands = json.loads((cohort / "commands.json").read_text())
            self.assertEqual(20, len(commands["first_pass"]))
            self.assertTrue((cohort / "events.jsonl").is_file())
        finally:
            if cohort.exists():
                for path in cohort.rglob("*"):
                    if path.is_file(): path.chmod(0o600)
                shutil.rmtree(str(cohort))

    def test_launch_factory_rechecks_gate_and_records_owned_batch(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            baseline = self._snapshot()["protected"]
            launch, owned = self.runner._launch_factory(cohort, manifest, baseline)
            process = mock.Mock(pid=9001); process.wait.return_value = 0
            identity = {"pid": 9001, "ppid": os.getpid(), "pgid": 9001,
                        "sid": 9001, "state": "S", "start_token": "abc"}
            with mock.patch.object(self.runner, "capture_resource_snapshot",
                                   return_value=self._snapshot()), \
                 mock.patch.object(self.runner, "validate_resource_snapshot"), \
                 mock.patch.object(self.runner, "verify_cohort_frozen"), \
                 mock.patch.object(self.runner, "wait_for_ports_bindable"), \
                 mock.patch.object(self.runner.subprocess, "Popen", return_value=process), \
                 mock.patch.object(self.runner, "inspect_process", return_value=identity):
                row = launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
            self.assertEqual(0, row["returncode"])
            self.assertEqual([], owned)
            self.assertTrue(next((cohort / "processes").glob("*.json")).is_file())
            events = [json.loads(line) for line in (cohort / "events.jsonl").read_text().splitlines()]
            self.assertEqual(["batch_started", "batch_finished"],
                             [row["event"] for row in events])

    def test_consecutive_batches_wait_for_port_release_before_each_popen(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batches = self.runner.build_batches(manifest)[:2]
        events = []

        class Process(object):
            def __init__(self, pid): self.pid = pid
            def wait(self):
                events.append("process-wait-%d" % self.pid)
                return 0

        processes = iter((Process(9001), Process(9002)))

        def popen(*_args, **_kwargs):
            process = next(processes)
            events.append("popen-%d" % process.pid)
            return process

        def identity(pid):
            return {"pid": pid, "ppid": os.getpid(), "pgid": pid, "sid": pid,
                    "state": "S", "start_token": "token-%d" % pid}

        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])
            with mock.patch.object(self.runner, "capture_resource_snapshot",
                                   return_value=self._snapshot()), \
                 mock.patch.object(self.runner, "validate_resource_snapshot"), \
                 mock.patch.object(self.runner, "verify_cohort_frozen"), \
                 mock.patch.object(self.runner, "wait_for_ports_bindable",
                                   side_effect=lambda _ports: events.append("port-wait")), \
                 mock.patch.object(self.runner.subprocess, "Popen", side_effect=popen), \
                 mock.patch.object(self.runner, "inspect_process", side_effect=identity):
                for batch in batches:
                    launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
            self.assertEqual([], owned)
        self.assertEqual([
            "port-wait", "popen-9001", "process-wait-9001",
            "port-wait", "popen-9002", "process-wait-9002",
        ], events)

    def test_stop_set_while_writing_snapshot_prevents_popen(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, _owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])

            def stop_during_snapshot(*_args, **_kwargs):
                self.runner._STOP.set()

            self.runner._STOP.clear()
            try:
                with mock.patch.object(self.runner, "capture_resource_snapshot",
                                       return_value=self._snapshot()), \
                     mock.patch.object(self.runner, "validate_resource_snapshot"), \
                     mock.patch.object(self.runner, "verify_cohort_frozen"), \
                     mock.patch.object(self.runner, "wait_for_ports_bindable"), \
                     mock.patch.object(self.runner, "_atomic_json",
                                       side_effect=stop_during_snapshot), \
                     mock.patch.object(self.runner.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(RuntimeError, "cancelled"):
                        launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
                popen.assert_not_called()
            finally:
                self.runner._STOP.clear()

    def test_stop_set_while_building_command_prevents_popen(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, _owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])

            def stop_during_command(*_args, **_kwargs):
                self.runner._STOP.set()
                return ["eval-batch"]

            self.runner._STOP.clear()
            try:
                with mock.patch.object(self.runner, "capture_resource_snapshot",
                                       return_value=self._snapshot()), \
                     mock.patch.object(self.runner, "validate_resource_snapshot"), \
                     mock.patch.object(self.runner, "verify_cohort_frozen"), \
                     mock.patch.object(self.runner, "wait_for_ports_bindable"), \
                     mock.patch.object(self.runner, "build_batch_command",
                                       side_effect=stop_during_command), \
                     mock.patch.object(self.runner.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(RuntimeError, "cancelled"):
                        launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
                popen.assert_not_called()
            finally:
                self.runner._STOP.clear()

    def test_cancel_gate_observes_and_cleans_process_started_inside_gate(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        batch = self.runner.build_batches(manifest)[0]
        popen_entered = threading.Event()
        release_popen = threading.Event()
        cancel_started = threading.Event()
        cleanup_called = threading.Event()
        process = mock.Mock(pid=9001)
        cleanup_records = []
        thread_errors = []

        def wait_for_cleanup():
            if not cleanup_called.wait(timeout=2):
                raise RuntimeError("cleanup did not run")
            return 0

        process.wait.side_effect = wait_for_cleanup

        def popen(*_args, **_kwargs):
            popen_entered.set()
            if not release_popen.wait(timeout=2):
                raise RuntimeError("test did not release Popen")
            return process

        def identity(pid):
            return {"pid": pid, "ppid": os.getpid(), "pgid": pid, "sid": pid,
                    "state": "S", "start_token": "token-%d" % pid}

        def cleanup(records, _cohort):
            cleanup_records.append(list(records))
            cleanup_called.set()

        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "processes").mkdir(); (cohort / "snapshots").mkdir()
            (cohort / "events.jsonl").touch()
            launch, owned = self.runner._launch_factory(
                cohort, manifest, self._snapshot()["protected"])

            def run_launch():
                try:
                    launch(batch, self.runner.FIRST_ASSIGNMENTS[0], 1)
                except BaseException as error:
                    thread_errors.append(error)

            def cancel():
                cancel_started.set()
                try:
                    launch.cancel_owned()
                except BaseException as error:
                    thread_errors.append(error)

            self.runner._STOP.clear()
            try:
                with mock.patch.object(self.runner, "capture_resource_snapshot",
                                       return_value=self._snapshot()), \
                     mock.patch.object(self.runner, "validate_resource_snapshot"), \
                     mock.patch.object(self.runner, "verify_cohort_frozen"), \
                     mock.patch.object(self.runner, "wait_for_ports_bindable"), \
                     mock.patch.object(self.runner.subprocess, "Popen", side_effect=popen), \
                     mock.patch.object(self.runner, "inspect_process", side_effect=identity), \
                     mock.patch.object(self.runner, "terminate_owned_processes",
                                       side_effect=cleanup):
                    launch_thread = threading.Thread(target=run_launch)
                    cancel_thread = threading.Thread(target=cancel)
                    launch_thread.start()
                    self.assertTrue(popen_entered.wait(timeout=2))
                    cancel_thread.start()
                    self.assertTrue(cancel_started.wait(timeout=2))
                    release_popen.set()
                    launch_thread.join(timeout=2)
                    cancel_thread.join(timeout=2)
                    self.assertFalse(launch_thread.is_alive())
                    self.assertFalse(cancel_thread.is_alive())
                self.assertEqual([], thread_errors)
                self.assertTrue(cleanup_called.is_set())
                self.assertEqual(9001, cleanup_records[0][0]["pid"])
                self.assertTrue(self.runner._STOP.is_set())
                self.assertEqual([], owned)
            finally:
                release_popen.set()
                self.runner._STOP.clear()

    def test_batch_gate_rejects_frozen_input_or_manifest_drift(self):
        partition = self.runner.build_partition()
        manifest = self.runner.build_stage1_manifest(partition)
        batch = self.runner.build_batches(manifest)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp); (cohort / "configs").mkdir()
            manifest_bytes = self.runner.canonical_json_bytes(manifest)
            (cohort / "manifest.json").write_bytes(manifest_bytes)
            (cohort / "partition.json").write_bytes(
                self.runner.canonical_json_bytes(partition))
            frozen = cohort / "source.txt"; frozen.write_text("frozen")
            configs = {}
            for key in batch["task_keys"]:
                path = self.runner._config_path(cohort, key)
                path.write_text("simulation: {}\n")
                configs[key] = self.runner.sha_file(path)
            (cohort / "metadata.json").write_text(json.dumps({
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
            (cohort / "provenance.json").write_text(json.dumps({
                "frozen_input_sha256": {str(frozen): self.runner.sha_file(frozen)},
                "config_sha256": configs}))
            self.runner.verify_cohort_frozen(cohort, batch)
            frozen.write_text("changed")
            with self.assertRaises(RuntimeError):
                self.runner.verify_cohort_frozen(cohort, batch)

    def test_run_execute_path_publishes_final_summary_without_real_process(self):
        candidate = self._candidate("run")
        args = self.runner.parse_args(["--run", "--cohort-id", candidate.name,
                                      "--capture-report", "/tmp/capture.json"])
        result = {"finalized": {"a": {}}, "retry_attempt": {"b": {}},
                  "artifacts": {"summary": {}}}
        emitted = []
        with mock.patch.object(self.runner, "resolve_idle_bundle", return_value=self.bundle), \
             mock.patch.object(self.runner, "validate_frozen_inputs", return_value={}), \
             mock.patch.object(self.runner, "capture_resource_snapshot",
                               return_value=self._snapshot()), \
             mock.patch.object(self.runner, "validate_resource_snapshot"), \
             mock.patch.object(self.runner, "wait_for_ports_bindable"), \
             mock.patch.object(self.runner, "_create_cohort"), \
             mock.patch.object(self.runner, "_launch_factory", return_value=(lambda *_: {}, [])), \
             mock.patch.object(self.runner, "execute_two_rounds", return_value=result), \
             mock.patch.object(self.runner.signal, "signal", return_value=signal.SIG_DFL):
            self.assertEqual(0, self.runner.execute(args, emitted.append))
        payload = json.loads(emitted[-1])
        self.assertEqual((1, 1), (payload["finalized"], payload["retry_count"]))

    def test_signal_interrupt_aborts_without_retry_artifacts_or_finished_event(self):
        candidate = self._candidate("interrupt")
        args = self.runner.parse_args(["--run", "--cohort-id", candidate.name,
                                      "--capture-report", "/tmp/capture.json"])
        emitted = []
        handlers = {}
        artifacts = mock.Mock()
        launch = lambda *_args: {}
        launch.stop_and_cancel = mock.Mock()
        def install(sig, handler):
            previous = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = handler
            return previous
        def interrupt_during_first_round(_manifest, _cohort, _launch):
            handlers[signal.SIGINT](signal.SIGINT, None)
            artifacts()
            return {"finalized": {}, "retry_attempt": {}, "artifacts": {}}
        with mock.patch.object(self.runner, "resolve_idle_bundle", return_value=self.bundle), \
             mock.patch.object(self.runner, "validate_frozen_inputs", return_value={}), \
             mock.patch.object(self.runner, "capture_resource_snapshot",
                               return_value=self._snapshot()), \
             mock.patch.object(self.runner, "validate_resource_snapshot"), \
             mock.patch.object(self.runner, "wait_for_ports_bindable"), \
             mock.patch.object(self.runner, "_create_cohort"), \
             mock.patch.object(self.runner, "_launch_factory",
                               return_value=(launch, [])), \
             mock.patch.object(self.runner, "_atomic_json"), \
             mock.patch.object(self.runner, "execute_two_rounds",
                               side_effect=interrupt_during_first_round), \
             mock.patch.object(self.runner.signal, "signal", side_effect=install):
            self.assertEqual(130, self.runner.execute(args, emitted.append))
        artifacts.assert_not_called()
        launch.stop_and_cancel.assert_called_once_with()
        self.assertEqual([], emitted)

    def test_classify_attempt_selects_only_requested_task_keys(self):
        manifest = self.runner.build_stage1_manifest(self.runner.build_partition())
        selected = manifest["tasks"][3]["key"]
        with mock.patch.object(self.runner, "classify_task_evidence",
                               return_value={"classification": "countable"}) as classify:
            rows = self.runner.classify_attempt(manifest, Path("/tmp/c"), 2, [selected])
        self.assertEqual([selected], list(rows)); self.assertEqual(1, classify.call_count)


if __name__ == "__main__":
    unittest.main()
