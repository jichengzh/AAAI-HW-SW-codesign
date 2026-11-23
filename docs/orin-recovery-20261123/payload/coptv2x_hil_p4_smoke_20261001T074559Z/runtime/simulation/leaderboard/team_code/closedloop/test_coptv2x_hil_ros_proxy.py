"""Pure protocol checks for the ROS 2 / reverse-SSH HIL proxy."""

import json
from pathlib import Path
import sys
import unittest
import socket
import threading
from unittest import mock

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation/leaderboard/team_code/closedloop"))

from ditl_edge_protocol import FRAME_PREFIX, MAGIC
from scripts import coptv2x_hil_ros_proxy as proxy


class ReverseProxyProtocolTests(unittest.TestCase):
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
