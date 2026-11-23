"""Pure protocol checks for the ROS 2 / reverse-SSH HIL proxy."""

import json
import io
import queue
from contextlib import redirect_stdout
from pathlib import Path
import sys
import unittest
import socket
import threading
from unittest import mock
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation/leaderboard/team_code/closedloop"))

from ditl_edge_protocol import FRAME_PREFIX, MAGIC
from scripts import coptv2x_hil_ros_proxy as proxy


def trigger_payload(sequence=0, **changes):
    return {
        "schema_version": "coptv2x.hil-trigger.v1",
        "run_id": "smoke-run", "route_id": "Town05-route-2",
        "scenario_id": "scenario1", "sample_id": "frame-%s" % sequence,
        "sequence_no": sequence, "source_frame": sequence + 10,
        "source_sim_time_ns": sequence * 50000000,
        "arm_id": "joint_shcosearch", **changes,
    }


def make_bridge(policy="latest", legacy=False):
    args = proxy.build_parser().parse_args([])
    args.queue_capacity = 8
    args.worker_timeout_s = 2.0
    if legacy:
        if hasattr(args, "trigger_policy"):
            del args.trigger_policy
    else:
        args.trigger_policy = policy
    node = mock.Mock()
    bridge = proxy.CoptV2xHilRosProxy(
        args, mock.Mock(), lambda _name: node, SimpleNamespace, SimpleNamespace, None,
    )
    return bridge


class TriggerSchedulingTests(unittest.TestCase):
    def test_host_stream_preserves_every_result_and_ack_with_latest_triggers(self):
        bridge = make_bridge()
        connection, host = socket.socketpair()
        self.addCleanup(connection.close)
        self.addCleanup(host.close)
        connection.settimeout(3)
        host.settimeout(3)
        accepted = threading.Event()
        original_accept = bridge._accept_host_trigger

        def accept(payload):
            original_accept(payload)
            if payload["sequence_no"] == 4:
                accepted.set()

        def connect():
            with bridge.send_lock:
                bridge.socket = connection
            return connection

        def receive(size):
            data = b""
            while len(data) < size:
                chunk = host.recv(size - len(data))
                if not chunk:
                    raise EOFError("test host stream closed")
                data += chunk
            return data

        with mock.patch.object(bridge, "_connect_host_proxy", side_effect=connect), \
                mock.patch.object(bridge, "_accept_host_trigger", side_effect=accept):
            bridge.start()
            try:
                for sequence in (0, 1, 4):
                    host.sendall(("TRIGGER " + json.dumps(trigger_payload(sequence)) + "\n").encode())
                self.assertTrue(accepted.wait(2))
                self.assertEqual(4, bridge.triggers_to_publish.get_nowait()["sequence_no"])
                bridge.triggers_to_publish.task_done()
                for request_id in ("request-001", "request-002"):
                    header = json.dumps({"kind": "result", "status": "ok", "request_id": request_id}).encode()
                    wire = FRAME_PREFIX.pack(MAGIC, len(header), 0) + header
                    bridge._on_result(SimpleNamespace(data=wire))
                    size = proxy.LENGTH_PREFIX.unpack(receive(4))[0]
                    self.assertEqual(wire, receive(size))
                    host.sendall(("ACK " + request_id + "\n").encode())
                    self.assertEqual(request_id, bridge.acks_to_ros.get(timeout=2))
                    bridge.acks_to_ros.task_done()
            finally:
                with redirect_stdout(io.StringIO()):
                    bridge.close()
        self.assertIsNone(bridge.failed)
        self.assertEqual(0, bridge.results_to_host.unfinished_tasks)
        self.assertEqual(0, bridge.acks_to_ros.unfinished_tasks)

    def test_host_reader_reconnect_keeps_latest_sequence_floor(self):
        bridge = make_bridge()
        pairs = [socket.socketpair(), socket.socketpair()]
        for pair in pairs:
            for connection in pair:
                self.addCleanup(connection.close)
                connection.settimeout(3)
        accepted = threading.Event()
        reconnected = threading.Event()
        original_accept = bridge._accept_host_trigger
        connections = iter([pairs[0][0], pairs[1][0]])

        def accept(payload):
            original_accept(payload)
            accepted.set()

        def connect():
            connection = next(connections)
            with bridge.send_lock:
                bridge.socket = connection
            if connection is pairs[1][0]:
                reconnected.set()
            return connection

        with mock.patch.object(bridge, "_connect_host_proxy", side_effect=connect), \
                mock.patch.object(bridge, "_accept_host_trigger", side_effect=accept):
            reader = threading.Thread(target=bridge._socket_reader)
            bridge.threads = [reader]
            reader.start()
            try:
                line = ("TRIGGER " + json.dumps(trigger_payload(4)) + "\n").encode()
                pairs[0][1].sendall(line)
                self.assertTrue(accepted.wait(2))
                pairs[0][1].close()
                self.assertTrue(reconnected.wait(2))
                pairs[1][1].sendall(line)
                self.assertTrue(bridge.stop_event.wait(2))
            finally:
                with redirect_stdout(io.StringIO()):
                    bridge.close()
        self.assertIsInstance(bridge.failed, ValueError)
        self.assertEqual(1, bridge.trigger_summary()["host_to_ros"]["received"])
        self.assertEqual(1, bridge.trigger_summary()["host_to_ros"]["rejected"])
        self.assertEqual(4, bridge.triggers_to_publish.get_nowait()["sequence_no"])
        bridge.triggers_to_publish.task_done()

    def test_fifo_backpressure_does_not_block_dispatch_accounting(self):
        bridge = make_bridge(policy="fifo")
        bridge.triggers_to_publish = queue.Queue(maxsize=1)
        bridge._accept_host_trigger(trigger_payload())
        entered = threading.Event()
        errors = []
        channel = bridge.triggers_to_publish
        original_put = channel.put

        def monitored_put(*args, **kwargs):
            entered.set()
            return original_put(*args, **kwargs)

        def produce():
            try:
                bridge._accept_host_trigger(trigger_payload(1))
            except Exception as error:
                errors.append(error)

        with mock.patch.object(channel, "put", side_effect=monitored_put):
            producer = threading.Thread(target=produce)
            producer.start()
            self.assertTrue(entered.wait(1))
            dispatcher = threading.Thread(
                target=bridge._record_trigger_dispatch,
                args=("host_to_ros", trigger_payload()),
            )
            dispatcher.start()
            dispatcher.join(0.2)
            try:
                self.assertFalse(dispatcher.is_alive(), "FIFO put holds the diagnostic lock")
            finally:
                channel.get_nowait()
                channel.task_done()
                producer.join(2)
                dispatcher.join(2)
        self.assertEqual([], errors)
        self.assertEqual(1, channel.get_nowait()["sequence_no"])
        channel.task_done()

    def test_spin_publishes_newest_trigger_and_every_ack(self):
        bridge = make_bridge()
        trigger_publisher, ack_publisher = mock.Mock(), mock.Mock()
        bridge.trigger_publisher = trigger_publisher
        bridge.ack_publisher = ack_publisher
        for sequence in (0, 4):
            bridge._accept_host_trigger(trigger_payload(sequence))
        for request_id in ("request-001", "request-002"):
            bridge.acks_to_ros.put(request_id)
        bridge.rclpy.ok.side_effect = (True, False)
        bridge.spin()
        published = [json.loads(call[0][0].data)["sequence_no"]
                     for call in trigger_publisher.publish.call_args_list]
        self.assertEqual([4], published)
        self.assertEqual(["request-001", "request-002"],
                         [call[0][0].data for call in ack_publisher.publish.call_args_list])
        self.assertEqual(0, bridge.triggers_to_publish.unfinished_tasks)
        self.assertEqual(0, bridge.acks_to_ros.unfinished_tasks)
        self.assertEqual(1, bridge.trigger_summary()["host_to_ros"]["dispatched"])
        self.assertEqual(14, bridge.trigger_summary()["host_to_ros"]["last_dispatched_source_frame"])

    def test_publish_failure_still_balances_queue_accounting(self):
        bridge = make_bridge()
        bridge._accept_host_trigger(trigger_payload())
        bridge.trigger_publisher.publish.side_effect = RuntimeError("publish failed")
        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            bridge.spin()
        self.assertEqual(0, bridge.triggers_to_publish.unfinished_tasks)
        self.assertEqual(0, bridge.trigger_summary()["host_to_ros"]["dispatched"])

    def test_parser_and_legacy_namespace_preserve_fifo_order(self):
        self.assertEqual("fifo", getattr(proxy.build_parser().parse_args([]), "trigger_policy", None))
        bridge = make_bridge(legacy=True)
        for sequence in (0, 1, 4):
            bridge._accept_host_trigger(trigger_payload(sequence))
            bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(sequence))))
        for channel in (bridge.triggers_to_publish, bridge.triggers_to_worker):
            self.assertEqual([0, 1, 4], [channel.get_nowait()["sequence_no"] for _ in range(3)])
            for _ in range(3):
                channel.task_done()
            self.assertEqual(0, channel.unfinished_tasks)

    def test_latest_keeps_only_newest_pending_at_both_stages(self):
        bridge = make_bridge()
        for sequence in (0, 1, 4):
            bridge._accept_host_trigger(trigger_payload(sequence))
            bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(sequence))))
        for channel in (bridge.triggers_to_publish, bridge.triggers_to_worker):
            self.assertEqual(1, channel.maxsize)
            self.assertEqual(1, channel.qsize())
            self.assertEqual(1, channel.unfinished_tasks)
            self.assertEqual(4, channel.get_nowait()["sequence_no"])
            channel.task_done()
            channel.join()
        summary = bridge.trigger_summary()
        for stage in ("host_to_ros", "ros_to_worker"):
            self.assertEqual(3, summary[stage]["received"])
            self.assertEqual(2, summary[stage]["merged"])
            self.assertEqual(14, summary[stage]["last_received_source_frame"])
            self.assertEqual(11, summary[stage]["last_merged_source_frame"])

    def test_latest_rejects_each_wrong_identity_before_replacement(self):
        for field, wrong in (("run_id", "wrong-run"), ("route_id", "wrong-route"),
                             ("scenario_id", "wrong-scenario"), ("arm_id", "original_default")):
            for stage in ("host", "worker"):
                with self.subTest(field=field, stage=stage):
                    bridge = make_bridge()
                    bridge._accept_host_trigger(trigger_payload())
                    if stage == "worker":
                        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload())))
                        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(1, **{field: wrong}))))
                        self.assertIsInstance(bridge.failed, ValueError)
                        channel = bridge.triggers_to_worker
                    else:
                        with self.assertRaises(ValueError):
                            bridge._accept_host_trigger(trigger_payload(1, **{field: wrong}))
                        channel = bridge.triggers_to_publish
                    self.assertEqual(0, channel.get_nowait()["sequence_no"])
                    channel.task_done()

    def test_latest_sequence_floor_survives_socket_detach(self):
        bridge = make_bridge()
        bridge._accept_host_trigger(trigger_payload(4))
        bridge._detach_socket(None)
        for sequence in (4, 3):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                bridge._accept_host_trigger(trigger_payload(sequence))
        bridge._accept_host_trigger(trigger_payload(7))
        self.assertEqual(7, bridge.triggers_to_publish.get_nowait()["sequence_no"])
        bridge.triggers_to_publish.task_done()

    def test_latest_ros_sequence_validation_is_independent_and_allows_gaps(self):
        bridge = make_bridge()
        bridge._accept_host_trigger(trigger_payload(7))
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(0))))
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(4))))
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(4))))
        self.assertIsInstance(bridge.failed, ValueError)
        self.assertEqual(4, bridge.triggers_to_worker.get_nowait()["sequence_no"])
        bridge.triggers_to_worker.task_done()
        self.assertEqual(2, bridge.trigger_summary()["ros_to_worker"]["received"])

    def test_latest_worker_requires_host_bound_episode(self):
        bridge = make_bridge()
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload())))
        self.assertIsInstance(bridge.failed, ValueError)
        self.assertTrue(bridge.triggers_to_worker.empty())

    def test_latest_does_not_replace_an_inflight_worker_request(self):
        bridge = make_bridge()
        bridge._accept_host_trigger(trigger_payload())
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload())))
        entered = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        sent = []
        errors = []
        pairs = [socket.socketpair(), socket.socketpair()]
        for pair in pairs:
            for connection in pair:
                self.addCleanup(connection.close)
                connection.settimeout(3)

        def serve():
            try:
                sent.append(json.loads(proxy._read_line(pairs[0][1], 16384))["sequence_no"])
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test release timed out")
                pairs[0][1].sendall(b"DONE first\n")
                sent.append(json.loads(proxy._read_line(pairs[1][1], 16384))["sequence_no"])
                pairs[1][1].sendall(b"DONE latest\n")
                bridge.stop_event.set()
            except Exception as error:
                errors.append(error)
                bridge.stop_event.set()
            finally:
                completed.set()

        server = threading.Thread(target=serve)
        worker = threading.Thread(target=bridge._worker_dispatcher)
        with mock.patch.object(proxy.socket, "create_connection", side_effect=[pairs[0][0], pairs[1][0]]):
            server.start()
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                for sequence in (1, 4):
                    bridge._accept_host_trigger(trigger_payload(sequence))
                    bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(sequence))))
                self.assertEqual(2, bridge.triggers_to_worker.unfinished_tasks)
            finally:
                release.set()
                completed.wait(3)
                bridge.stop_event.set()
                server.join(3)
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual([], errors)
        self.assertIsNone(bridge.failed)
        self.assertEqual([0, 4], sent)
        self.assertEqual(0, bridge.triggers_to_worker.unfinished_tasks)
        self.assertEqual(2, bridge.trigger_summary()["ros_to_worker"]["dispatched"])

    def test_latest_concurrent_replacement_keeps_accounting_balanced(self):
        bridge = make_bridge()
        completed = threading.Event()
        delivered = []
        errors = []

        def produce():
            try:
                for sequence in range(400):
                    bridge._accept_host_trigger(trigger_payload(sequence))
            except Exception as error:
                errors.append(error)
            finally:
                completed.set()

        def consume():
            while not completed.is_set() or not bridge.triggers_to_publish.empty():
                try:
                    value = bridge.triggers_to_publish.get(timeout=0.01)
                except queue.Empty:
                    continue
                delivered.append(value["sequence_no"])
                bridge.triggers_to_publish.task_done()

        threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(sorted(set(delivered)), delivered)
        self.assertEqual(399, delivered[-1])
        self.assertEqual(0, bridge.triggers_to_publish.unfinished_tasks)
        stats = bridge.trigger_summary()["host_to_ros"]
        self.assertEqual(400, stats["received"])
        self.assertEqual(400, stats["merged"] + len(delivered))

    def test_latest_does_not_coalesce_results_or_acks(self):
        bridge = make_bridge()
        for request_id in ("request-001", "request-002"):
            header = json.dumps({"kind": "result", "status": "ok", "request_id": request_id}).encode()
            wire = FRAME_PREFIX.pack(MAGIC, len(header), 0) + header
            bridge._on_result(SimpleNamespace(data=wire))
            bridge.acks_to_ros.put_nowait(request_id)
        self.assertIsNone(bridge.failed)
        self.assertEqual(2, bridge.results_to_host.qsize())
        self.assertEqual(2, bridge.acks_to_ros.qsize())

    def test_close_prints_parseable_summary_even_when_cleanup_fails(self):
        bridge = make_bridge()
        bridge._accept_host_trigger(trigger_payload())
        bridge._accept_host_trigger(trigger_payload(4))
        bridge._fail(ValueError("invalid trigger"))
        bridge.node.destroy_node.side_effect = RuntimeError("cleanup failed")
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            bridge.close()
        summary = json.loads(output.getvalue())
        self.assertEqual("coptv2x_hil_ros_proxy_summary", summary["event"])
        self.assertTrue(summary["failed"])
        self.assertEqual("latest", summary["trigger_policy"])
        self.assertEqual(2, summary["host_to_ros"]["received"])
        self.assertEqual(1, summary["host_to_ros"]["merged"])


class ReverseProxyProtocolTests(unittest.TestCase):
    def test_result_validation_fails_closed_without_admitting_bad_frames(self):
        bridge = make_bridge()
        for wire in (b"short", FRAME_PREFIX.pack(b"BAD!", 0, 0),
                     FRAME_PREFIX.pack(MAGIC, 1, 0) + b"{",
                     FRAME_PREFIX.pack(MAGIC, 2, 0) + b"{}"):
            with self.subTest(wire=wire):
                bridge._on_result(SimpleNamespace(data=wire))
                self.assertIsInstance(bridge.failed, ValueError)
                self.assertTrue(bridge.results_to_host.empty())

    def test_latest_malformed_trigger_never_replaces_pending_valid_trigger(self):
        bridge = make_bridge()
        bridge._accept_host_trigger(trigger_payload())
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload())))
        bridge._on_trigger(SimpleNamespace(data=json.dumps(trigger_payload(4, sequence_no="4"))))
        self.assertIsInstance(bridge.failed, ValueError)
        self.assertEqual(0, bridge.triggers_to_worker.get_nowait()["sequence_no"])
        bridge.triggers_to_worker.task_done()

    def test_detaching_socket_cannot_close_a_newer_connection(self):
        bridge = proxy.CoptV2xHilRosProxy.__new__(proxy.CoptV2xHilRosProxy)
        bridge.send_lock = threading.Lock()
        old, old_peer = socket.socketpair()
        self.addCleanup(old.close)
        current, current_peer = socket.socketpair()
        self.addCleanup(old_peer.close)
        self.addCleanup(current_peer.close)
        self.addCleanup(current.close)
        bridge.socket = current
        bridge._detach_socket(old)
        self.assertIs(bridge.socket, current)
        current.sendall(b"alive")
        self.assertEqual(b"alive", current_peer.recv(5))
        self.assertEqual(-1, old.fileno())

    def test_close_releases_inflight_ack_and_detaches_connection(self):
        bridge = proxy.CoptV2xHilRosProxy.__new__(proxy.CoptV2xHilRosProxy)
        bridge.send_lock = threading.Lock()
        bridge.ack_lock = threading.Lock()
        bridge.stop_event = threading.Event()
        event = threading.Event()
        bridge.awaiting_ack = ("request-001", event)
        connection, peer = socket.socketpair()
        self.addCleanup(peer.close)
        bridge.socket = connection
        bridge.threads = []
        bridge.node = mock.Mock()
        bridge.close()
        self.assertTrue(event.is_set())
        self.assertIsNone(bridge.socket)
        self.assertEqual(-1, connection.fileno())
        self.assertEqual(b"", peer.recv(1))
        bridge.node.destroy_node.assert_called_once_with()

    def test_episode_scoped_tcp_close_is_reconnectable(self):
        self.assertTrue(proxy.is_reconnectable_disconnect(EOFError("closed")))
        self.assertTrue(proxy.is_reconnectable_disconnect(ConnectionResetError("reset")))
        self.assertFalse(proxy.is_reconnectable_disconnect(ValueError("bad trigger")))

    def test_result_wire_is_length_prefixed_and_keeps_request_identity(self):
        header = json.dumps({
            "kind": "result", "status": "ok", "request_id": "request-001",
        }, sort_keys=True).encode("utf-8")
        wire = FRAME_PREFIX.pack(MAGIC, len(header), 0) + header
        framed = proxy.frame_result_for_host(wire)
        self.assertEqual(len(wire), proxy.LENGTH_PREFIX.unpack(framed[:4])[0])
        self.assertEqual(wire, framed[4:])
        self.assertEqual("request-001", proxy.request_id_from_result_wire(wire))

    def test_host_trigger_line_is_parsed_as_json_without_identity_rewrite(self):
        payload = {
            "schema_version": "coptv2x.hil-trigger.v1",
            "run_id": "smoke-run", "route_id": "Town05-route-2",
            "scenario_id": "scenario1", "sample_id": "carla-frame-1",
            "sequence_no": 0, "source_frame": 1,
            "source_sim_time_ns": 50000000, "arm_id": "joint_shcosearch",
        }
        self.assertEqual(payload, proxy.parse_host_message(
            ("TRIGGER " + json.dumps(payload) + "\n").encode("utf-8")
        ))

    def test_host_ack_must_match_the_result_being_forwarded(self):
        self.assertEqual("request-001", proxy.validate_ack_line(
            b"ACK request-001\n", "request-001"
        ))
        with self.assertRaisesRegex(ValueError, "does not match"):
            proxy.validate_ack_line(b"ACK request-002\n", "request-001")


if __name__ == "__main__":
    unittest.main()
