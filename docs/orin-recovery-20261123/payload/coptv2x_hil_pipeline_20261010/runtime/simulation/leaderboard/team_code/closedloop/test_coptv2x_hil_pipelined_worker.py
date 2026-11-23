"""Local socket integration for admission-only replies and exact FIFO identities."""

import contextlib
from concurrent.futures import TimeoutError
import ast
import base64
import io
import json
from pathlib import Path
import select
import shlex
import socket
import sys
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation/leaderboard/team_code/closedloop"))

from scripts import coptv2x_hil_orin_worker as worker
from scripts import coptv2x_hil_ros_proxy as proxy
from scripts import coptv2x_hil_episode_services as episode_services
from scripts.coptv2x_hil_episode_services import service_configuration
from ditl_edge_protocol import decode_wire_frame, recv_wire_frame
import test_coptv2x_hil_orin_model_lifecycle as lifecycle


class PipelineWorkerTests(unittest.TestCase):
    runtime_factory = lifecycle.ResidentModelLifecycleTests.runtime_factory
    write_manifest = lifecycle.ResidentModelLifecycleTests.write_manifest
    trigger = lifecycle.ResidentModelLifecycleTests.trigger

    def setUp(self):
        lifecycle.ResidentModelLifecycleTests.setUp(self)

    def enable_pipeline(self):
        self.instance.close()
        self.sender, self.receiver = socket.socketpair()
        self.receiver.settimeout(2)
        self.addCleanup(self.sender.close)
        self.addCleanup(self.receiver.close)
        self.args.pipeline_depth = 2
        self.args.startup_arm_id = "joint_shcosearch"
        with contextlib.redirect_stderr(io.StringIO()), \
                patch.object(worker, "_connect_local_result_handoff", return_value=self.sender):
            self.instance = worker.OrinHilWorker(self.args)
        self.addCleanup(self.instance.close)

    def test_explicit_cli_and_invalid_mode_requirements(self):
        args = worker.build_parser().parse_args([
            "--data-root", "/inputs", "--manifest", "/manifest", "--artifact-root", "/model",
            "--native-repo-root", "/native", "--result-count", "2"])
        self.assertEqual(0, getattr(args, "pipeline_depth", None))
        for options in ({"pipeline_depth": 1}, {"pipeline_depth": True},
                        {"pipeline_depth": 2},
                        {"pipeline_depth": 2, "serial_optimized": True},
                        {"pipeline_depth": 2, "serial_optimized": True,
                         "startup_arm_id": "joint_shcosearch", "wire_compression": "zlib"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                worker.pipeline_options(SimpleNamespace(**options))

    def test_accepted_is_admission_not_finished_result_and_ordered_results_are_real(self):
        self.assertTrue(hasattr(self.instance, "submit_trigger"), "pipeline worker API is missing")
        self.enable_pipeline()
        entered, release = threading.Event(), threading.Event()
        original = worker.CoptV2xHilPipeline.encode_inferred_frame

        def finish(pipeline, inferred):
            entered.set()
            self.assertTrue(release.wait(2))
            return original(pipeline, inferred)

        local, client = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(client.close)
        client.settimeout(2)
        errors = []

        def serve():
            try:
                worker.serve_trigger_request(self.instance, local)
            except Exception as error:
                errors.append(error)

        with patch.object(worker.CoptV2xHilPipeline, "encode_inferred_frame", finish):
            server = threading.Thread(target=serve)
            server.start()
            try:
                trigger = self.trigger()
                client.sendall((json.dumps(trigger.to_metadata()) + "\n").encode())
                self.assertEqual(b"ACCEPTED run-1:0\n", worker._read_bounded_line(client, 2048))
                self.assertTrue(entered.wait(2))
                self.assertEqual(1, self.instance.submitted)
                self.assertEqual(0, self.instance.processed)
                self.assertFalse(select.select([self.receiver], [], [], 0)[0])
                request_id, second = self.instance.submit_trigger(self.trigger(1, 2))
                self.assertEqual("run-1:1", request_id)
                self.assertFalse(second.done())
            finally:
                release.set()
                server.join(2)
        self.assertFalse(server.is_alive())
        self.assertEqual([], errors)
        self.instance.drain(timeout=2)
        for sequence in (0, 1):
            header_bytes, payload, _ = recv_wire_frame(self.receiver)
            header, arrays = decode_wire_frame(header_bytes, payload)
            self.assertEqual("run-1:%d" % sequence, header["request_id"])
            self.assertEqual(sequence + 1, header["trigger_identity"]["source_frame"])
            self.assertEqual({"res0", "res1", "res2"}, set(arrays))
            for array in arrays.values():
                self.assertEqual((sequence + 1,), tuple(set(array.flat)))
        self.assertEqual(2, self.instance.processed)
        self.assertEqual("run-1:1", second.result(2).header["request_id"])

    def test_fixed_arm_episode_and_submission_limit_are_enforced(self):
        self.assertTrue(hasattr(self.instance, "submit_trigger"), "pipeline worker API is missing")
        self.enable_pipeline()
        self.args.result_count = 2
        with self.assertRaisesRegex(ValueError, "startup arm"):
            self.instance.submit_trigger(self.trigger(arm_id="schedule_only"))
        _, first = self.instance.submit_trigger(self.trigger())
        with self.assertRaisesRegex(ValueError, "episode"):
            self.instance.submit_trigger(self.trigger(1, 2, run_id="run-2"))
        with self.assertRaisesRegex(ValueError, "strictly"):
            self.instance.submit_trigger(self.trigger())
        _, second = self.instance.submit_trigger(self.trigger(1, 2))
        with self.assertRaisesRegex(ValueError, "result_count"):
            self.instance.submit_trigger(self.trigger(2, 2))
        first.result(2)
        second.result(2)
        with self.assertRaisesRegex(ValueError, "pipeline"):
            self.instance.process_trigger(self.trigger(3, 2))

    def test_credit_timeout_cancels_reservation_without_reading_or_losing_trigger(self):
        self.enable_pipeline()
        entered, release = threading.Event(), threading.Event()
        original = worker.CoptV2xHilPipeline.encode_inferred_frame

        def finish(pipeline, inferred):
            entered.set()
            self.assertTrue(release.wait(2))
            return original(pipeline, inferred)

        with patch.object(worker.CoptV2xHilPipeline, "encode_inferred_frame", finish):
            _, first = self.instance.submit_trigger(self.trigger())
            self.assertTrue(entered.wait(2))
            _, second = self.instance.submit_trigger(self.trigger(1, 2))
            try:
                with self.assertRaises(TimeoutError):
                    self.instance.submit_trigger(self.trigger(2, 2), timeout=0)
                self.assertEqual(2, self.instance.submitted)
                self.assertNotIn(2, self.instance.admission._reserved_sequence_nos)
            finally:
                release.set()
            first.result(2)
            second.result(2)
        _, third = self.instance.submit_trigger(self.trigger(2, 2))
        self.assertEqual("run-1:2", third.result(2).header["request_id"])

    def test_closed_worker_rejects_new_admission_without_reloading_backend(self):
        self.enable_pipeline()
        self.instance.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.instance.submit_trigger(self.trigger())
        self.assertIsNone(self.instance.backend)


    def test_async_source_error_is_reported_and_close_cleans_up(self):
        self.assertTrue(hasattr(self.instance, "submit_trigger"), "pipeline worker API is missing")
        self.enable_pipeline()
        with patch.object(self.instance.source, "frame_for_trigger", side_effect=ValueError("source corrupt")):
            _, future = self.instance.submit_trigger(self.trigger())
            with self.assertRaisesRegex(ValueError, "source corrupt"):
                future.result(2)
        with self.assertRaisesRegex(ValueError, "source corrupt"):
            self.instance.raise_if_failed()
        with self.assertRaisesRegex(ValueError, "source corrupt"):
            self.instance.close()
        self.assertIsNone(self.instance.connection)
        self.assertIsNone(self.instance.backend)

    def test_async_failure_interrupts_blocked_sender_before_resource_teardown(self):
        self.enable_pipeline()
        self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        delivering, closed = threading.Event(), threading.Event()
        original_send = worker.send_wire_frame
        close_errors = []

        def blocked_send(connection, header, payload):
            delivering.set()
            original_send(connection, header, b"x" * (4 * 1024 * 1024))

        def close():
            try:
                self.instance.close()
            except Exception as error:
                close_errors.append(error)
            finally:
                closed.set()

        with patch.object(worker, "send_wire_frame", blocked_send):
            _, first = self.instance.submit_trigger(self.trigger())
            self.assertTrue(delivering.wait(2))
            with patch.object(self.instance.source, "frame_for_trigger", side_effect=ValueError("source corrupt")):
                _, second = self.instance.submit_trigger(self.trigger(1, 2))
                with self.assertRaisesRegex(ValueError, "source corrupt"):
                    second.result(2)
            closer = threading.Thread(target=close)
            closer.start()
            try:
                self.assertTrue(closed.wait(0.5), "failure left sender blocked during close")
            finally:
                self.receiver.close()
                closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual("source corrupt", str(close_errors[0]))
        with self.assertRaisesRegex(ValueError, "source corrupt"):
            first.result(2)

    def test_legacy_socket_reply_stays_done(self):
        self.assertTrue(hasattr(worker, "serve_trigger_request"), "socket handler API is missing")
        local, client = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(client.close)
        client.settimeout(2)
        client.sendall((json.dumps(self.trigger().to_metadata()) + "\n").encode())
        worker.serve_trigger_request(self.instance, local)
        self.assertEqual(b"DONE run-1:0\n", worker._read_bounded_line(client, 2048))
        self.assertEqual(1, self.instance.processed)

    def test_main_real_listener_accepts_two_requests_and_drains_before_complete(self):
        self.enable_pipeline()
        self.args.result_count = 2
        ready = threading.Event()
        completed, errors = [], []
        reservation = socket.socket()
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()
        ready_path = self.tree.root / "worker.ready"
        argv = ["--data-root", self.args.data_root, "--manifest", self.args.manifest,
                "--artifact-root", self.args.artifact_root, "--native-repo-root", self.args.native_repo_root,
                "--result-count", "2", "--worker-port", str(port), "--ready-file", str(ready_path),
                "--serial-optimized", "--startup-arm-id", "joint_shcosearch", "--pipeline-depth", "2"]
        write_text = Path.write_text

        def ready_write(path, *args, **kwargs):
            value = write_text(path, *args, **kwargs)
            if path == ready_path:
                ready.set()
            return value

        def main():
            try:
                completed.append(worker.main(argv))
            except Exception as error:
                errors.append(error)

        with patch.object(worker, "OrinHilWorker", return_value=self.instance), \
                patch.object(Path, "write_text", ready_write), contextlib.redirect_stdout(io.StringIO()):
            server = threading.Thread(target=main)
            server.start()
            self.assertTrue(ready.wait(2))
            for sequence in (0, 1):
                with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                    client.sendall((json.dumps(self.trigger(sequence, sequence + 1).to_metadata()) + "\n").encode())
                    self.assertEqual(("ACCEPTED run-1:%d\n" % sequence).encode(),
                                     worker._read_bounded_line(client, 2048))
            for sequence in (0, 1):
                header, _payload, _ = recv_wire_frame(self.receiver)
                self.assertEqual(sequence, json.loads(header)["trigger_identity"]["sequence_no"])
            server.join(3)
        self.assertFalse(server.is_alive())
        self.assertEqual([], errors)
        self.assertEqual([0], completed)
        self.assertEqual(2, self.instance.processed)
        self.assertIsNone(self.instance.connection)
        self.assertIsNone(self.instance.backend)


class PipelineProxyServiceTests(unittest.TestCase):
    def test_service_rejects_invalid_depth_before_generating_remote_configuration(self):
        for depth in (1, -1, True, 2.0):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "depth"):
                service_configuration(SimpleNamespace(pipeline_depth=depth), {})

    def test_service_main_parses_path_flags_and_packages_actual_configuration_without_network(self):
        commands = []
        with tempfile.TemporaryDirectory() as run_dir:
            environment = {"COPT_HIL_RUN_DIR": run_dir, "COPT_HIL_TASK_ID": "task-1",
                           "COPT_HIL_ARM_ID": "compression_only", "COPT_HIL_COHORT_ID": "cohort-1",
                           "COPT_HIL_ATTEMPT": "1"}
            argv = ["--mode", "start", "--remote-root", "/new/remote", "--runtime", "/new/runtime",
                    "--native-repo-root", "/old/native/V2Xverse", "--container-runtime", "/tmp/new-runtime",
                    "--host-state-root", "/old/runtimestate/new-pipeline",
                    "--container-state-root", "/state/new-pipeline", "--serial-optimized", "--pipeline-depth", "2"]
            # Only the process/network execution boundary is replaced. Parsing,
            # config generation, shell quoting, payload and evidence paths run.
            with patch.dict("os.environ", environment), \
                    patch.object(episode_services, "checked_command",
                                 side_effect=lambda command, **kwargs: commands.append(command)):
                self.assertEqual(0, episode_services.main(argv))
            self.assertEqual(3, len(commands))
            writer = shlex.split(commands[0][-1])[-1]
            decode = next(node for node in ast.walk(ast.parse(writer))
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                          and node.func.attr == "b64decode")
            config = json.loads(base64.b64decode(ast.literal_eval(decode.args[0])))
            self.assertTrue(config["state_dir"].startswith("/old/runtimestate/new-pipeline/"))
            worker_command = next(row["argv"] for row in config["services"] if row["name"] == "worker")
            self.assertEqual("/old/native/V2Xverse",
                             worker_command[worker_command.index("--native-repo-root") + 1])
            self.assertEqual("2", worker_command[worker_command.index("--pipeline-depth") + 1])
            for row in config["services"]:
                if row["container"] is not None:
                    self.assertEqual("/tmp/new-runtime/scripts/coptv2x_hil_orin_services.py",
                                     row["container_script"])
                    self.assertTrue(row["container_state_dir"].startswith("/state/new-pipeline/"))
            self.assertEqual("rsync", commands[-1][0])
            self.assertEqual(str(Path(run_dir) / "evidence/orin") + "/", commands[-1][-1])
            self.assertTrue((Path(run_dir) / "evidence/orin").is_dir())

    def test_service_explicit_runtime_native_and_state_paths_target_new_tree(self):
        options = dict(runtime="/new/runtime", container="existing-ros", orin_python="/python",
                       artifact_root="/artifacts", result_port=55022, result_count=20, worker_port=56812,
                       forward_port=56811, ready_timeout_s=180, serial_optimized=True, pipeline_depth=2,
                       native_repo_root="/old/native/V2Xverse", container_runtime="/tmp/new-runtime",
                       host_state_root="/old/runtimestate/new-pipeline",
                       container_state_root="/state/new-pipeline")
        config = service_configuration(SimpleNamespace(**options),
                                       {"token": "episode", "arm_id": "compression_only"})
        self.assertEqual("/old/runtimestate/new-pipeline/episode", config["state_dir"])
        argv = next(row["argv"] for row in config["services"] if row["name"] == "worker")
        self.assertEqual("/old/native/V2Xverse", argv[argv.index("--native-repo-root") + 1])
        self.assertEqual("/old/runtimestate/new-pipeline/episode/worker.ready",
                         argv[argv.index("--ready-file") + 1])
        for row in config["services"]:
            if row["container"] is None:
                continue
            self.assertEqual("/tmp/new-runtime/scripts/coptv2x_hil_orin_services.py",
                             row["container_script"])
            shell = shlex.split(row["argv"][-1])
            command = json.loads(shell[shell.index("--command-json") + 1])
            self.assertTrue(command[1].startswith("/tmp/new-runtime/scripts/"))
            self.assertIn("PYTHONPATH=/tmp/new-runtime:/tmp/new-runtime/simulation/leaderboard/team_code/closedloop",
                          row["argv"])
            if row["name"] == "publisher":
                self.assertEqual("/state/new-pipeline/episode/publisher.ready",
                                 command[command.index("--ready-file") + 1])

    def test_real_tcp_dispatcher_validates_accepted_identity_and_keeps_legacy_done(self):
        for depth, reply, failed in ((2, b"ACCEPTED run-1:0\n", False),
                                     (2, b"ACCEPTED wrong:0\n", True),
                                     (2, b"DONE run-1:0\n", True),
                                     (0, b"DONE legacy-id\n", False)):
            with self.subTest(depth=depth, reply=reply):
                args = proxy.build_parser().parse_args([])
                args.worker_pipeline_depth = depth
                args.worker_timeout_s = 2
                listener = socket.socket()
                self.addCleanup(listener.close)
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(2)
                args.worker_port = listener.getsockname()[1]
                bridge = proxy.CoptV2xHilRosProxy(args, Mock(), lambda name: Mock(),
                                                SimpleNamespace, SimpleNamespace, None)
                payload = {"schema_version": "coptv2x.hil-trigger.v1", "run_id": "run-1",
                           "route_id": "route-1", "scenario_id": "scenario-1",
                           "sample_id": "sample-1", "sequence_no": 0, "source_frame": 1,
                           "source_sim_time_ns": 50000000, "arm_id": "joint_shcosearch"}
                bridge.triggers_to_worker.put(payload)
                dispatcher = threading.Thread(target=bridge._worker_dispatcher)
                dispatcher.start()
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    self.assertEqual(payload, json.loads(proxy._read_line(connection, 16384)))
                    connection.sendall(reply)
                bridge.triggers_to_worker.join()
                bridge.stop_event.set()
                dispatcher.join(2)
                self.assertFalse(dispatcher.is_alive())
                self.assertEqual(failed, bridge.failed is not None)

    def test_pipeline_proxy_rejects_latest_and_service_forwards_explicit_depth(self):
        args = proxy.build_parser().parse_args([])
        self.assertEqual(0, getattr(args, "worker_pipeline_depth", None))
        args.worker_pipeline_depth, args.trigger_policy = 2, "latest"
        with self.assertRaisesRegex(ValueError, "FIFO"):
            proxy.CoptV2xHilRosProxy(args, Mock(), lambda name: Mock(),
                                   SimpleNamespace, SimpleNamespace, None)
        options = dict(runtime="/frozen/runtime", container="existing-ros", orin_python="/python",
                       artifact_root="/artifacts", result_port=55022, result_count=20, worker_port=56812,
                       forward_port=56811, ready_timeout_s=180, pipeline_depth=2)
        context = {"token": "new-episode", "arm_id": "compression_only"}
        with self.assertRaises(ValueError):
            service_configuration(SimpleNamespace(**options), context)
        config = service_configuration(SimpleNamespace(**options, serial_optimized=True), context)
        argv = next(row["argv"] for row in config["services"] if row["name"] == "worker")
        self.assertEqual("2", argv[argv.index("--pipeline-depth") + 1])
        docker = next(row["argv"] for row in config["services"] if row["name"] == "proxy")
        shell = shlex.split(docker[-1])
        command = json.loads(shell[shell.index("--command-json") + 1])
        self.assertEqual("2", command[command.index("--worker-pipeline-depth") + 1])
