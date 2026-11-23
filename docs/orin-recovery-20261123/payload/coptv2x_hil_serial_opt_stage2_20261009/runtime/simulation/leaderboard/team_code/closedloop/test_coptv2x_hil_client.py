from __future__ import print_function

import copy
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from types import MappingProxyType
from unittest import mock

import numpy as np


sys.path.insert(0, os.path.dirname(__file__))

from coptv2x_hil_artifacts import DEFAULT_CONFIG_DIR, load_hil_artifact_registry
from coptv2x_hil_backend import (
    COPTV2X_PAYLOAD_SEMANTICS,
    HilFeatureResult,
    hash_feature_payload,
)
from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger
from coptv2x_hil_client import (
    AsyncHilClient,
    HilResultEnvelope,
    JsonlAuditWriter,
    Ros2HilTransport,
    create_ros2_hil_client,
    run_hil_prefusion_forward,
)
from coptv2x_hil_proxy_transport import ReverseSshRosProxyTransport
from coptv2x_hil_result_buffer import CoptV2xHilResultBuffer


ARM_ID = "joint_shcosearch"


def _artifact_identity():
    spec = load_hil_artifact_registry(DEFAULT_CONFIG_DIR)[ARM_ID]
    return {
        "arm_id": spec.arm_id,
        "model": spec.model,
        "widths": list(spec.widths),
        "precision": spec.precision,
        "runtime": spec.runtime,
        "config": str(spec.config),
        "checkpoint": str(spec.checkpoint),
        "engine": None if spec.engine is None else str(spec.engine),
        "artifact_sha256": dict(spec.artifact_sha256),
    }


def _trigger(sequence_no, source_frame=None, run_id="run-live", arm_id=ARM_ID):
    if source_frame is None:
        source_frame = sequence_no
    return parse_hil_trigger(
        {
            "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
            "run_id": run_id,
            "route_id": "Town05-route-01",
            "scenario_id": "scenario-01",
            "sample_id": "sample-%05d" % sequence_no,
            "sequence_no": sequence_no,
            "source_frame": source_frame,
            "source_sim_time_ns": source_frame * 50000000,
            "arm_id": arm_id,
        }
    )


def _small_result(trigger, marker=0.0, remote_input_tensor_bytes=0):
    outputs = {
        "res0": np.full((2, 1, 2, 2), marker, dtype=np.float32),
        "res1": np.full((2, 2, 1, 1), marker, dtype=np.float32),
        "res2": np.full((2, 4, 1, 1), marker, dtype=np.float32),
    }
    return HilFeatureResult(
        outputs=MappingProxyType(outputs),
        artifact_identity=MappingProxyType(_artifact_identity()),
        trigger_identity=MappingProxyType(trigger.to_metadata()),
        payload_semantics=COPTV2X_PAYLOAD_SEMANTICS,
        payload_sha256=hash_feature_payload(outputs),
        remote_input_tensor_bytes=remote_input_tensor_bytes,
        timestamps_ns=MappingProxyType({}),
        durations_ns=MappingProxyType({}),
        model=MappingProxyType({}),
    )


def _envelope(trigger, marker=0.0, **kwargs):
    return HilResultEnvelope(
        feature_result=_small_result(trigger, marker=marker, **kwargs),
        batch_order=("ego", "rsu"),
    )


def _small_contract_validator(artifact_identity, outputs):
    if artifact_identity != _artifact_identity():
        raise ValueError("artifact mismatch")
    expected = {
        "res0": (2, 1, 2, 2),
        "res1": (2, 2, 1, 1),
        "res2": (2, 4, 1, 1),
    }
    if set(outputs) != set(expected):
        raise ValueError("output names")
    for name, shape in expected.items():
        value = np.asarray(outputs[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError("output contract")
        if not np.isfinite(value).all():
            raise ValueError("non-finite")


class FakeTransport(object):
    def __init__(self, block_publish=False):
        self.block_publish = block_publish
        self.publish_entered = threading.Event()
        self.publish_release = threading.Event()
        self.closed = threading.Event()
        self.handler = None
        self.published = []

    def start(self, handler):
        self.handler = handler

    def publish_trigger(self, payload):
        self.publish_entered.set()
        if self.block_publish:
            self.publish_release.wait(2.0)
        self.published.append(copy.deepcopy(payload))

    def deliver(self, result):
        self.handler(result)

    def close(self):
        self.closed.set()
        self.publish_release.set()


class FakeClock(object):
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class HilResultBufferTests(unittest.TestCase):
    def buffer(self, capacity=4):
        return CoptV2xHilResultBuffer(
            run_id="run-live",
            route_id="Town05-route-01",
            scenario_id="scenario-01",
            arm_id=ARM_ID,
            capacity=capacity,
            contract_validator=_small_contract_validator,
        )

    def register(self, buffer, trigger):
        buffer.register_trigger(trigger, published_ns=trigger.source_sim_time_ns)

    def test_startup_empty_then_new_then_zoh_never_regresses_on_late_result(self):
        buffer = self.buffer()
        for sequence in range(4):
            self.register(buffer, _trigger(sequence))

        empty = buffer.select_for_fusion(0, 0, _trigger(0), selection_ns=1)
        self.assertIsNone(empty.result)
        self.assertEqual("empty", empty.audit["selection_reason"])

        self.assertTrue(buffer.accept(_envelope(_trigger(2), marker=2), 120).accepted)
        newest = buffer.select_for_fusion(2, 100000000, _trigger(2), selection_ns=121)
        self.assertEqual(2, newest.audit["selected_result_sequence_no"])
        self.assertEqual("new", newest.audit["selection_reason"])

        self.assertTrue(buffer.accept(_envelope(_trigger(1), marker=1), 130).accepted)
        held = buffer.select_for_fusion(3, 150000000, _trigger(3), selection_ns=131)
        self.assertEqual(2, held.audit["selected_result_sequence_no"])
        self.assertEqual("zoh", held.audit["selection_reason"])
        self.assertEqual(1, held.audit["rsu_aoi_frames"])
        self.assertEqual(50.0, held.audit["rsu_aoi_ms"])

    def test_rejects_duplicate_cross_run_arm_future_identity_conflict_and_nonzero_input(self):
        cases = []
        buffer = self.buffer()
        valid = _trigger(1)
        self.register(buffer, valid)
        cases.append(("duplicate", _envelope(valid), _envelope(valid)))

        cross_run = _trigger(1, run_id="other-run")
        cases.append(("run_id", None, _envelope(cross_run)))
        cross_arm = _trigger(1, arm_id="compression_only")
        cases.append(("arm_id", None, _envelope(cross_arm)))
        cases.append(("future", None, _envelope(_trigger(2))))

        conflict_result = _small_result(valid)
        conflict_identity = dict(conflict_result.trigger_identity)
        conflict_identity["sample_id"] = "conflicting-sample"
        conflict_result = replace(
            conflict_result,
            trigger_identity=MappingProxyType(conflict_identity),
        )
        cases.append(
            (
                "identity",
                None,
                HilResultEnvelope(conflict_result, ("ego", "rsu")),
            )
        )
        cases.append(("remote_input", None, _envelope(valid, remote_input_tensor_bytes=4)))

        for label, first, rejected in cases:
            with self.subTest(label=label):
                local = buffer if label == "duplicate" else self.buffer()
                if label != "duplicate":
                    self.register(local, valid)
                if first is not None:
                    self.assertTrue(local.accept(first, 10).accepted)
                decision = local.accept(rejected, 11)
                self.assertFalse(decision.accepted)
                self.assertIn(label, decision.reason)

    def test_rejects_payload_hash_contract_batch_order_and_freezes_thread_boundary_copy(self):
        trigger = _trigger(0)
        mutations = []

        invalid_hash = _small_result(trigger)
        mutations.append(
            ("payload_hash", HilResultEnvelope(replace(invalid_hash, payload_sha256="0" * 64), ("ego", "rsu")))
        )
        invalid_contract = _small_result(trigger)
        broken = dict(invalid_contract.outputs)
        broken["res0"] = np.zeros((2, 1, 3, 3), dtype=np.float32)
        mutations.append(
            ("payload_contract", HilResultEnvelope(replace(invalid_contract, outputs=broken, payload_sha256=hash_feature_payload(broken)), ("ego", "rsu")))
        )
        mutations.append(("batch_order", HilResultEnvelope(_small_result(trigger), ("rsu", "ego"))))

        for label, envelope in mutations:
            with self.subTest(label=label):
                buffer = self.buffer()
                self.register(buffer, trigger)
                decision = buffer.accept(envelope, 5)
                self.assertFalse(decision.accepted)
                self.assertIn(label, decision.reason)

        buffer = self.buffer()
        self.register(buffer, trigger)
        original = _envelope(trigger, marker=7)
        self.assertTrue(buffer.accept(original, 5).accepted)
        original.feature_result.outputs["res0"].fill(99)
        selected = buffer.select_for_fusion(0, 0, trigger, 6)
        self.assertEqual(7.0, float(selected.result.outputs["res0"][0, 0, 0, 0]))
        self.assertFalse(selected.result.outputs["res0"].flags.writeable)

    def test_capacity_evicts_lowest_sequence_and_rejects_retired_rearrival(self):
        buffer = self.buffer(capacity=2)
        for sequence in range(3):
            trigger = _trigger(sequence)
            self.register(buffer, trigger)
            self.assertTrue(buffer.accept(_envelope(trigger, marker=sequence), sequence).accepted)
        self.assertEqual(2, buffer.size)
        self.assertEqual("evict_lowest_sequence", buffer.drop_policy)
        rejected = buffer.accept(_envelope(_trigger(0)), 4)
        self.assertFalse(rejected.accepted)
        self.assertIn("retired", rejected.reason)
        selected = buffer.select_for_fusion(2, 100000000, _trigger(2), 5)
        self.assertEqual(2, selected.audit["selected_result_sequence_no"])

    def test_full_cache_drops_late_incoming_lowest_sequence(self):
        buffer = self.buffer(capacity=2)
        for sequence in range(3):
            self.register(buffer, _trigger(sequence))
        self.assertTrue(buffer.accept(_envelope(_trigger(1)), 1).accepted)
        self.assertTrue(buffer.accept(_envelope(_trigger(2)), 2).accepted)

        decision = buffer.accept(_envelope(_trigger(0)), 3)

        self.assertFalse(decision.accepted)
        self.assertEqual("buffer_overflow_evict_incoming", decision.reason)
        self.assertEqual(0, decision.evicted_sequence_no)
        selected = buffer.select_for_fusion(2, 100000000, _trigger(2), 4)
        self.assertEqual(2, selected.audit["selected_result_sequence_no"])

    def test_delayed_result_survives_a_full_route_trigger_history(self):
        buffer = self.buffer(capacity=2)
        triggers = [_trigger(sequence) for sequence in range(40)]
        for trigger in triggers:
            self.register(buffer, trigger)

        delayed_result = buffer.accept(_envelope(triggers[0]), arrival_ns=100)

        self.assertTrue(delayed_result.accepted, delayed_result.reason)
        selected = buffer.select_for_fusion(39, 1950000000, triggers[-1], 101)
        self.assertEqual(0, selected.audit["selected_result_sequence_no"])

    def test_expensive_result_validation_does_not_hold_trigger_registration_lock(self):
        validation_entered = threading.Event()
        validation_release = threading.Event()

        def blocking_validator(artifact_identity, outputs):
            del artifact_identity, outputs
            validation_entered.set()
            validation_release.wait(1.0)

        buffer = CoptV2xHilResultBuffer(
            "run-live", "Town05-route-01", "scenario-01", ARM_ID,
            capacity=2, contract_validator=blocking_validator,
        )
        self.register(buffer, _trigger(0))
        accepting = threading.Thread(
            target=buffer.accept, args=(_envelope(_trigger(0)), 1)
        )
        accepting.start()
        self.assertTrue(validation_entered.wait(0.5))

        registered = threading.Event()

        def register_next():
            self.register(buffer, _trigger(1))
            registered.set()

        registering = threading.Thread(target=register_next)
        registering.start()
        try:
            self.assertTrue(
                registered.wait(0.1),
                "CARLA trigger registration waited on result hash/copy",
            )
        finally:
            validation_release.set()
            accepting.join(1.0)
            registering.join(1.0)

    def test_audit_carries_full_trigger_identities_and_local_rejection(self):
        buffer = self.buffer()
        trigger = _trigger(0)
        self.register(buffer, trigger)
        buffer.record_rejection("outbound_overflow", trigger.sequence_no)

        rejected = buffer.select_for_fusion(0, 0, trigger, 1)
        self.assertEqual(trigger.to_metadata(), dict(rejected.audit["trigger_identity"]))
        self.assertIsNone(rejected.audit["selected_result_trigger_identity"])
        self.assertEqual("rejected", rejected.audit["selection_reason"])
        self.assertEqual("outbound_overflow", rejected.audit["rejection_reason"])

        self.assertTrue(buffer.accept(_envelope(trigger), 2).accepted)
        selected = buffer.select_for_fusion(0, 0, trigger, 3)
        self.assertEqual(
            trigger.to_metadata(),
            dict(selected.audit["selected_result_trigger_identity"]),
        )

    def test_real_registry_contract_is_enforced_by_default(self):
        trigger = _trigger(0)
        spec = load_hil_artifact_registry(DEFAULT_CONFIG_DIR)[ARM_ID]
        outputs = {
            tensor.name: np.zeros(tensor.shape, dtype=np.dtype(tensor.dtype))
            for tensor in spec.output_contract
        }
        result = HilFeatureResult(
            outputs=outputs,
            artifact_identity=_artifact_identity(),
            trigger_identity=trigger.to_metadata(),
            payload_semantics=COPTV2X_PAYLOAD_SEMANTICS,
            payload_sha256=hash_feature_payload(outputs),
            remote_input_tensor_bytes=0,
            timestamps_ns={},
            durations_ns={},
            model={},
        )
        buffer = CoptV2xHilResultBuffer(
            "run-live", "Town05-route-01", "scenario-01", ARM_ID, capacity=1
        )
        self.register(buffer, trigger)
        self.assertTrue(
            buffer.accept(HilResultEnvelope(result, ("ego", "rsu")), 1).accepted
        )

    def test_real_registry_identity_freeze_preserves_exact_contract_values(self):
        trigger = _trigger(1)
        identity = _artifact_identity()
        outputs = {
            tensor.name: np.zeros(tensor.shape, dtype=np.dtype(tensor.dtype))
            for tensor in load_hil_artifact_registry(DEFAULT_CONFIG_DIR)[ARM_ID].output_contract
        }
        result = HilFeatureResult(
            outputs=outputs,
            artifact_identity=identity,
            trigger_identity=trigger.to_metadata(),
            payload_semantics=COPTV2X_PAYLOAD_SEMANTICS,
            payload_sha256=hash_feature_payload(outputs),
            remote_input_tensor_bytes=0,
            timestamps_ns={},
            durations_ns={},
            model={},
        )
        buffer = CoptV2xHilResultBuffer(
            "run-live", "Town05-route-01", "scenario-01", ARM_ID, capacity=2
        )
        self.register(buffer, trigger)
        self.assertTrue(
            buffer.accept(HilResultEnvelope(result, ("ego", "rsu")), 1).accepted
        )

        changed = _trigger(2)
        changed_identity = dict(identity)
        changed_identity["widths"] = [16, 32, 65]
        changed_result = replace(
            result,
            artifact_identity=MappingProxyType(changed_identity),
            trigger_identity=MappingProxyType(changed.to_metadata()),
        )
        self.register(buffer, changed)
        rejected = buffer.accept(
            HilResultEnvelope(changed_result, ("ego", "rsu")), 2
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual("payload_contract_invalid", rejected.reason)


class ReverseSshRosProxyTests(unittest.TestCase):
    def test_result_envelope_class_is_shared_across_import_paths(self):
        import coptv2x_hil_result_buffer as top_level
        package_module = sys.modules["team_code.closedloop.coptv2x_hil_result_buffer"]

        self.assertIs(top_level.HilResultEnvelope, package_module.HilResultEnvelope)

    def test_proxy_transport_exposes_result_decode_failure(self):
        import socket
        import struct

        received = []
        arrived = threading.Event()
        transport = ReverseSshRosProxyTransport(listen_port=0, connect_timeout_s=2.0)
        transport.start(lambda envelope: (received.append(envelope), arrived.set()))
        proxy = socket.create_connection(("127.0.0.1", transport.bound_port), timeout=2.0)
        try:
            proxy.sendall(struct.pack("!I", 4) + b"nope")
            self.assertTrue(arrived.wait(2.0))
            self.assertEqual([None], received)
            self.assertIsInstance(transport.last_error, ValueError)
            self.assertIn("wire", str(transport.last_error))
        finally:
            proxy.close()
            transport.close()

    def test_proxy_transport_result_stream_outlives_connect_timeout(self):
        import socket

        transport = ReverseSshRosProxyTransport(
            listen_port=0, connect_timeout_s=0.05, socket_timeout_s=1.0
        )
        transport.start(lambda _envelope: None)
        proxy = socket.create_connection(("127.0.0.1", transport.bound_port), timeout=1.0)
        proxy.settimeout(1.0)
        try:
            time.sleep(0.1)
            transport.publish_trigger(_trigger(0).to_metadata())
            message = b""
            while not message.endswith(b"\n"):
                message += proxy.recv(1)
            self.assertTrue(message.startswith(b"TRIGGER "))
            self.assertIsNone(transport.last_error)
        finally:
            proxy.close()
            transport.close()

    def test_proxy_transport_sends_trigger_and_acknowledges_framed_result(self):
        import socket
        import struct

        received = []
        transport = ReverseSshRosProxyTransport(listen_port=0, connect_timeout_s=2.0)
        transport.start(received.append)
        proxy = socket.create_connection(("127.0.0.1", transport.bound_port), timeout=2.0)
        proxy.settimeout(2.0)
        trigger = _trigger(0)
        expected_result = HilResultEnvelope(_small_result(trigger), ("ego", "rsu"))
        try:
            with mock.patch(
                "coptv2x_hil_proxy_transport._decode_result_wire",
                return_value=(expected_result, "request-1"),
            ):
                transport.publish_trigger(trigger.to_metadata())
                line = b""
                while not line.endswith(b"\n"):
                    line += proxy.recv(1)
                message = json.loads(line.decode("utf-8").split(" ", 1)[1])
                self.assertEqual(trigger.to_metadata(), message)
                proxy.sendall(struct.pack("!I", 2) + b"ok")
                acknowledgement = b""
                while not acknowledgement.endswith(b"\n"):
                    acknowledgement += proxy.recv(1)
                self.assertEqual(b"ACK request-1\n", acknowledgement)
                self.assertEqual([expected_result], received)
        finally:
            proxy.close()
            transport.close()


class AsyncHilClientTests(unittest.TestCase):
    def client(self, transport, clock, capacity=8):
        buffer = CoptV2xHilResultBuffer(
            "run-live",
            "Town05-route-01",
            "scenario-01",
            ARM_ID,
            capacity=capacity,
            contract_validator=_small_contract_validator,
        )
        return AsyncHilClient(
            transport,
            buffer,
            outbound_capacity=capacity,
            inbound_capacity=capacity,
            clock_ns=clock,
        )

    def test_proxy_eof_invalidates_cached_result_and_next_trigger_immediately(self):
        import socket
        import struct

        transport = ReverseSshRosProxyTransport(listen_port=0, connect_timeout_s=1)
        client = self.client(transport, FakeClock())
        client.start()
        proxy = socket.create_connection(("127.0.0.1", transport.bound_port), timeout=1)
        proxy.settimeout(1)
        trigger = _trigger(0)
        try:
            self.assertTrue(client.publish_trigger(trigger).accepted)
            trigger_line = b""
            while not trigger_line.endswith(b"\n"):
                trigger_line += proxy.recv(1)
            with mock.patch("coptv2x_hil_proxy_transport._decode_result_wire",
                            return_value=(_envelope(trigger), "result-before-eof")):
                proxy.sendall(struct.pack("!I", 2) + b"ok")
                acknowledgement = b""
                while not acknowledgement.endswith(b"\n"):
                    acknowledgement += proxy.recv(1)
            self.assertTrue(client.wait_until_idle(1))
            self.assertEqual(0, client.select_for_fusion(0, 0, trigger).audit["selected_result_sequence_no"])
            proxy.close()
            deadline = time.monotonic() + 1
            while transport.last_error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsInstance(transport.last_error, EOFError)
            with self.assertRaisesRegex(RuntimeError, "EOFError: .*closed"):
                client.select_for_fusion(1, 50000000, trigger)
            with self.assertRaisesRegex(RuntimeError, "EOFError: .*closed"):
                client.publish_trigger(_trigger(1))
        finally:
            proxy.close()
            client.shutdown()

    def test_explicit_transport_fatal_preserves_type_message_without_blocking(self):
        transport = FakeTransport()
        client = self.client(transport, FakeClock())
        client.start()
        try:
            transport.last_error = ValueError("engine result wire is corrupt")
            with self.assertRaisesRegex(RuntimeError, "ValueError: engine result wire is corrupt"):
                client.publish_trigger(_trigger(0))
            self.assertEqual(0, client.stats["published_triggers"])
        finally:
            client.shutdown()
        self.assertFalse(client.publish_trigger(_trigger(1)).accepted)

    def test_main_thread_returns_while_transport_publish_is_blocked(self):
        transport = FakeTransport(block_publish=True)
        client = self.client(transport, FakeClock())
        client.start()
        returned = threading.Event()

        def call():
            decision = client.publish_trigger(_trigger(0))
            self.assertTrue(decision.accepted)
            returned.set()

        caller = threading.Thread(target=call)
        caller.start()
        self.assertTrue(returned.wait(0.1), "CARLA caller waited on transport")
        self.assertTrue(transport.publish_entered.wait(0.5))
        self.assertFalse(transport.publish_release.is_set())
        transport.publish_release.set()
        caller.join(0.5)
        client.shutdown()

    def test_20hz_harness_150ms_result_delay_adds_three_aoi_frames_and_keeps_ego_current(self):
        clock = FakeClock()
        transport = FakeTransport()
        client = self.client(transport, clock)
        client.start()
        audits = []
        for frame in range(4):
            clock.now = frame * 50000000
            trigger = _trigger(frame)
            self.assertTrue(client.publish_trigger(trigger).accepted)
            if frame == 3:
                transport.deliver(_envelope(_trigger(0), marker=10))
                self.assertTrue(client.wait_until_idle(0.5))
            selection = client.select_for_fusion(frame, clock.now, trigger)
            audits.append(selection.audit)

        self.assertEqual("new", audits[3]["selection_reason"])
        self.assertGreaterEqual(audits[3]["rsu_aoi_frames"], 3)
        self.assertGreaterEqual(audits[3]["rsu_aoi_ms"], 150.0)
        self.assertEqual(3, audits[3]["trigger_source_frame"])
        client.shutdown()

    def test_no_return_out_of_order_duplicate_overflow_and_shutdown_are_bounded(self):
        clock = FakeClock()
        transport = FakeTransport()
        client = self.client(transport, clock, capacity=2)
        client.start()
        for sequence in range(3):
            self.assertTrue(client.publish_trigger(_trigger(sequence)).accepted)
            self.assertTrue(client.wait_until_idle(0.5))
        for result in (
            _envelope(_trigger(2), marker=2),
            _envelope(_trigger(1), marker=1),
            _envelope(_trigger(2), marker=2),
        ):
            transport.deliver(result)
            self.assertTrue(client.wait_until_idle(0.5))
        selected = client.select_for_fusion(3, 150000000, _trigger(2))
        self.assertEqual(2, selected.audit["selected_result_sequence_no"])
        self.assertGreaterEqual(client.stats["rejected_results"], 1)
        client.shutdown()
        client.shutdown()
        self.assertTrue(transport.closed.is_set())
        self.assertFalse(client.is_running)
        self.assertFalse(client.publish_trigger(_trigger(4)).accepted)

    def test_transport_publish_failure_is_in_next_fusion_audit(self):
        class FailingTransport(FakeTransport):
            def publish_trigger(self, payload):
                self.publish_entered.set()
                raise OSError("injected transport failure")

        clock = FakeClock()
        transport = FailingTransport()
        client = self.client(transport, clock)
        client.start()
        trigger = _trigger(0)
        self.assertTrue(client.publish_trigger(trigger).accepted)
        self.assertTrue(client.wait_until_idle(0.5))
        selection = client.select_for_fusion(0, 0, trigger)
        self.assertEqual("rejected", selection.audit["selection_reason"])
        self.assertEqual("trigger_publish_failed", selection.audit["rejection_reason"])
        self.assertEqual(1, client.stats["publish_failures"])
        client.shutdown()


class ClientBoundaryTests(unittest.TestCase):
    def test_audit_writer_is_lazy_serializes_nested_immutable_data_and_closes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "nested", "audit.jsonl")
            writer = JsonlAuditWriter(path)
            self.assertFalse(os.path.exists(path))
            writer.append(MappingProxyType({"roles": ("ego", "rsu")}))
            writer.close()
            with open(path, "r", encoding="utf-8") as stream:
                self.assertEqual({"roles": ["ego", "rsu"]}, json.loads(stream.read()))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                writer.append({"late": True})

    def test_factory_rejects_invalid_config_and_builds_frozen_arm_client(self):
        with self.assertRaises(TypeError):
            create_ros2_hil_client(None)
        with self.assertRaisesRegex(ValueError, "missing"):
            create_ros2_hil_client({})
        with self.assertRaisesRegex(ValueError, "not frozen"):
            create_ros2_hil_client({
                "run_id": "run-live",
                "route_id": "route",
                "scenario_id": "scenario",
                "arm_id": "search",
                "audit_jsonl": "/tmp/unused.jsonl",
            })
        with tempfile.TemporaryDirectory() as temporary:
            client = create_ros2_hil_client({
                "run_id": "run-live",
                "route_id": "route",
                "scenario_id": "scenario",
                "arm_id": ARM_ID,
                "audit_jsonl": os.path.join(temporary, "audit.jsonl"),
                "buffer_capacity": 2,
                "trigger_capacity": 24,
                "outbound_capacity": 3,
                "inbound_capacity": 4,
                "max_payload_mib": 1,
            })
            self.assertEqual(2, client.result_buffer.capacity)
            self.assertEqual(24, client.result_buffer.trigger_capacity)
            self.assertEqual(3, client._outbound.maxsize)
            self.assertEqual(4, client._inbound.maxsize)
            client.audit_writer.close()

    def test_factory_selects_reverse_ssh_ros_proxy_without_local_rclpy(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = create_ros2_hil_client({
                "run_id": "run-live",
                "route_id": "route",
                "scenario_id": "scenario",
                "arm_id": ARM_ID,
                "audit_jsonl": os.path.join(temporary, "audit.jsonl"),
                "transport": "reverse_ssh_ros_proxy",
                "proxy_listen_port": 0,
            })
            self.assertIsInstance(client.transport, ReverseSshRosProxyTransport)
            self.assertEqual(0, client.transport.listen_port)
            client.audit_writer.close()

    def test_serial_factory_enables_verified_checksum_reuse_only_for_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("async", "serial_fixed_comm"):
                client = create_ros2_hil_client({
                    "run_id": "run-live", "route_id": "route", "scenario_id": "scenario",
                    "arm_id": ARM_ID, "audit_jsonl": os.path.join(temporary, mode + ".jsonl"),
                    "transport": "reverse_ssh_ros_proxy", "proxy_listen_port": 0,
                    "timing_mode": mode,
                })
                self.assertEqual(mode == "serial_fixed_comm", client.result_buffer.checksum_dedup)
                client.audit_writer.close()

    def test_fake_ros_transport_trigger_result_ack_connectivity(self):
        state = {"ok": False, "init": 0, "shutdown": 0}

        class Message(object):
            def __init__(self):
                self.data = []

        class Publisher(object):
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        class Node(object):
            def __init__(self, name):
                self.name = name
                self.destroyed = False

            def create_publisher(self, message_type, topic, qos):
                del message_type, topic, qos
                return Publisher()

            def create_subscription(self, message_type, topic, callback, qos):
                del message_type, topic, qos
                return callback

            def destroy_node(self):
                self.destroyed = True

        class Executor(object):
            def __init__(self):
                self.node = None
                self.closed = False

            def add_node(self, node):
                self.node = node

            def spin(self):
                return None

            def shutdown(self, timeout_sec):
                self.closed = timeout_sec == 1.0

        class QoSProfile(object):
            def __init__(self, **kwargs):
                self.options = kwargs

        rclpy = types.ModuleType("rclpy")
        rclpy.ok = lambda: state["ok"]

        def initialize(args=None):
            del args
            state["init"] += 1
            state["ok"] = True

        def shutdown():
            state["shutdown"] += 1
            state["ok"] = False

        rclpy.init = initialize
        rclpy.shutdown = shutdown
        executors = types.ModuleType("rclpy.executors")
        executors.SingleThreadedExecutor = Executor
        nodes = types.ModuleType("rclpy.node")
        nodes.Node = Node
        qos = types.ModuleType("rclpy.qos")
        qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile")
        qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
        qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="reliable")
        qos.QoSProfile = QoSProfile
        std_msgs = types.ModuleType("std_msgs")
        std_msgs_msg = types.ModuleType("std_msgs.msg")
        std_msgs_msg.String = Message
        std_msgs_msg.UInt8MultiArray = Message
        modules = {
            "rclpy": rclpy,
            "rclpy.executors": executors,
            "rclpy.node": nodes,
            "rclpy.qos": qos,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
        }

        trigger = _trigger(7)
        result = _small_result(trigger, marker=7)
        header = dict(trigger.to_metadata())
        header.update({
            "kind": "result",
            "status": "ok",
            "request_id": "request-7",
            "trigger_identity": trigger.to_metadata(),
            "batch_order": ["ego", "rsu"],
            "artifact_identity": dict(result.artifact_identity),
            "payload_semantics": result.payload_semantics,
            "payload_sha256": result.payload_sha256,
            "remote_input_tensor_bytes": 0,
        })
        delivered = []
        with mock.patch.dict(sys.modules, modules), mock.patch(
            "ditl_edge_protocol.decode_wire_bytes",
            return_value=(header, dict(result.outputs), {}, {}),
        ):
            transport = Ros2HilTransport(max_payload_mib=1)
            transport.start(delivered.append)
            transport.publish_trigger(trigger.to_metadata())
            trigger_message = transport._node.trigger_publisher.messages[-1]
            self.assertEqual(trigger.to_metadata(), json.loads(trigger_message.data))

            wire_message = Message()
            wire_message.data = [1, 2, 3]
            transport._node.on_result(wire_message)
            self.assertEqual(1, len(delivered))
            self.assertEqual(("ego", "rsu"), delivered[0].batch_order)
            self.assertEqual(
                "request-7", transport._node.ack_publisher.messages[-1].data
            )
            header["status"] = "error"
            transport._node.on_result(wire_message)
            header["status"] = "ok"
            self.assertEqual(2, len(delivered))
            self.assertIsNone(delivered[-1])
            self.assertEqual(1, len(transport._node.ack_publisher.messages))
            node = transport._node
            executor = transport._executor
            transport.close()

        self.assertEqual(1, state["init"])
        self.assertEqual(1, state["shutdown"])
        self.assertTrue(node.destroyed)
        self.assertTrue(executor.closed)


class PrefusionInjectionTests(unittest.TestCase):
    @staticmethod
    def _stub_optional_model_dependencies():
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: args[0] if args else None
        sys.modules.setdefault("icecream", module)

    def test_codriving_precomputed_channel_is_stateless_and_legacy_still_computes(self):
        self._stub_optional_model_dependencies()
        try:
            import torch
            from opencood.models.fuse_modules.codriving_attn import (
                _select_multiscale_features,
            )
        except ImportError as error:
            if "torch" in str(error):
                self.skipTest("torch unavailable")
            raise

        class Backbone(object):
            def __init__(self):
                self.calls = 0

            def resnet(self, value):
                self.calls += 1
                return (value + 1, value + 2, value + 3)

        backbone = Backbone()
        spatial = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        precomputed = (spatial + 10, spatial + 20, spatial + 30)
        selected = _select_multiscale_features(
            backbone, spatial, 3, precomputed
        )
        self.assertEqual(0, backbone.calls)
        self.assertIs(precomputed[0], selected[0])

        with self.assertRaisesRegex(ValueError, "level"):
            _select_multiscale_features(backbone, spatial, 3, precomputed[:2])
        self.assertEqual(0, backbone.calls)

        for invalid in (
            (object(), precomputed[1], precomputed[2]),
            (spatial[0], precomputed[1], precomputed[2]),
            (torch.zeros((1, 1, 2, 2)), precomputed[1], precomputed[2]),
            (torch.zeros((2, 1, 2, 2), dtype=torch.float64), precomputed[1], precomputed[2]),
        ):
            with self.subTest(invalid=type(invalid[0]).__name__):
                with self.assertRaisesRegex(ValueError, "level 0"):
                    _select_multiscale_features(backbone, spatial, 3, invalid)
        with self.assertRaisesRegex(ValueError, "level 0"):
            _select_multiscale_features(backbone, spatial, 3, (None, spatial, spatial))
        self.assertEqual(0, backbone.calls)

        legacy = _select_multiscale_features(backbone, spatial, 3, None)
        self.assertEqual(1, backbone.calls)
        self.assertEqual(1.0, float(legacy[0][0, 0, 0, 0]))

    def test_centerpoint_hil_builder_computes_only_current_ego_once(self):
        self._stub_optional_model_dependencies()
        try:
            import torch
            from opencood.models.center_point_codriving import centerpointcodriving
        except ImportError:
            self.skipTest("torch model dependencies unavailable")

        class Backbone(object):
            def __init__(self):
                self.calls = []

            def resnet(self, value):
                self.calls.append(value.detach().clone())
                return (value + 1, value + 2, value + 3)

        backbone = Backbone()
        owner = type("Owner", (), {"backbone": backbone})()
        spatial = torch.tensor([[[[40.0]]], [[[999.0]]]])
        remote = tuple(
            np.array([[[[value]]]], dtype=np.float32)
            for value in (10.0, 20.0, 30.0)
        )
        features = centerpointcodriving._hil_multiscale_features(
            owner, spatial, torch.tensor([2]), remote
        )
        self.assertEqual(1, len(backbone.calls))
        self.assertEqual([40.0], backbone.calls[0].reshape(-1).tolist())
        self.assertEqual([41.0, 10.0], features[0].reshape(-1).tolist())

        zero_filled = centerpointcodriving._hil_multiscale_features(
            owner, spatial, torch.tensor([2]), None
        )
        self.assertEqual([0.0], zero_filled[0].reshape(-1)[1:].tolist())
        with self.assertRaisesRegex(ValueError, "spatial batch"):
            centerpointcodriving._hil_multiscale_features(
                owner, spatial[:1], torch.tensor([2]), remote
            )
        with self.assertRaisesRegex(ValueError, "record_len"):
            centerpointcodriving._hil_multiscale_features(
                owner, spatial, torch.tensor([1]), remote
            )

        bad_remote_cases = (
            ("levels", remote[:2], "three scales"),
            ("shape", (np.zeros((1, 2, 1, 1), dtype=np.float32),) + remote[1:], "shape mismatch"),
            ("finite", (np.full((1, 1, 1, 1), np.nan, dtype=np.float32),) + remote[1:], "non-finite"),
        )
        for label, bad_remote, message in bad_remote_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    centerpointcodriving._hil_multiscale_features(
                        owner, spatial, torch.tensor([2]), bad_remote
                    )

        class IncompleteBackbone(object):
            def resnet(self, value):
                return (value,)

        with self.assertRaisesRegex(ValueError, "three scales"):
            centerpointcodriving._hil_multiscale_features(
                type("Owner", (), {"backbone": IncompleteBackbone()})(),
                spatial,
                torch.tensor([2]),
                None,
            )

    def test_centerpoint_forward_routes_hil_and_legacy_paths_without_state_leak(self):
        self._stub_optional_model_dependencies()
        try:
            import torch
            from opencood.models.center_point_codriving import centerpointcodriving
        except ImportError:
            self.skipTest("torch model dependencies unavailable")

        class IdentityHead(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.channels = channels
                self.seen = []

            def forward(self, value):
                self.seen.append(value.detach().clone())
                return torch.zeros(
                    (value.shape[0], self.channels, value.shape[2], value.shape[3]),
                    dtype=value.dtype,
                    device=value.device,
                )

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.resnet_calls = []
                self.decode_calls = []
                self.deblocks = []

            def resnet(self, value):
                self.resnet_calls.append(value.detach().clone())
                return tuple(value + offset for offset in (1, 2, 3))

            def decode_multiscale_feature(self, values):
                self.decode_calls.append(values)
                return values[0]

            def forward(self, batch):
                result = dict(batch)
                result["spatial_features_2d"] = batch["spatial_features"]
                return result

        class Fusion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, spatial, confidence, record_len, transforms, backbone,
                        waypoints, precomputed_multiscale=None):
                self.calls.append(precomputed_multiscale)
                return spatial[:1], 0, {"features_before_fusion": []}

        model = centerpointcodriving.__new__(centerpointcodriving)
        torch.nn.Module.__init__(model)
        model.pillar_vfe = torch.nn.Identity()
        model.scatter = torch.nn.Module()
        model.backbone = Backbone()
        model.multi_scale = True
        model.shrink_flag = False
        model.compression = False
        model.dcn = False
        model.early_flag = False
        model.cls_head = IdentityHead(1)
        model.reg_head = IdentityHead(8)
        model.fusion_net = Fusion()
        model.regroup = lambda value, record_len: (value,)
        model.generate_predicted_boxes = lambda cls, box: (
            cls,
            torch.zeros((cls.shape[0], 1, 7), dtype=cls.dtype),
        )
        record_len = torch.tensor([2])
        spatial = torch.tensor([[[[30.0]]], [[[999.0]]]])

        class Scatter(torch.nn.Module):
            def forward(self, batch):
                result = dict(batch)
                result["spatial_features"] = spatial
                return result

        model.scatter = Scatter()
        data = {
            "processed_lidar": {
                "voxel_features": torch.zeros((1, 1)),
                "voxel_coords": torch.zeros((1, 4), dtype=torch.int64),
                "voxel_num_points": torch.ones((1,), dtype=torch.int64),
            },
            "record_len": record_len,
            "pairwise_t_matrix": torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1),
        }
        remote = tuple(
            np.full((1, 1, 1, 1), value, dtype=np.float32)
            for value in (10, 20, 30)
        )
        hil_output = model(data, use_hil=True, hil_remote_multiscale=remote)
        self.assertEqual(1, len(model.backbone.decode_calls))
        self.assertEqual(1, len(model.backbone.resnet_calls))
        self.assertEqual(1, len(model.fusion_net.calls))
        self.assertEqual([31.0, 10.0], model.fusion_net.calls[0][0].reshape(-1).tolist())
        self.assertIn("cls_preds", hil_output)

        legacy_output = model(data)
        self.assertEqual(1, len(model.backbone.resnet_calls))
        self.assertIsNone(model.fusion_net.calls[-1])
        self.assertIn("cls_preds", legacy_output)
        with self.assertRaisesRegex(ValueError, "use_hil=True"):
            model(data, hil_remote_multiscale=remote)

        model.multi_scale = False
        with self.assertRaisesRegex(ValueError, "CoDriving ResNet"):
            model(data, use_hil=True, hil_remote_multiscale=remote)

    def test_codriving_forward_uses_precomputed_and_legacy_features(self):
        self._stub_optional_model_dependencies()
        try:
            import torch
            from opencood.models.fuse_modules.codriving_attn import CoDriving
        except ImportError:
            self.skipTest("torch model dependencies unavailable")

        class IdentityFusion(torch.nn.Module):
            def forward(self, neighbors):
                return neighbors[:1]

        class Backbone(object):
            deblocks = ()

            def __init__(self):
                self.calls = 0

            def resnet(self, value):
                self.calls += 1
                return tuple(value + offset for offset in (1, 2, 3))

        fusion = CoDriving.__new__(CoDriving)
        torch.nn.Module.__init__(fusion)
        fusion.downsample_rate = 1
        fusion.discrete_ratio = 1.0
        fusion.multi_scale = True
        fusion.num_levels = 3
        fusion.communication = False
        fusion.fuse_modules = torch.nn.ModuleList(
            [IdentityFusion(), IdentityFusion(), IdentityFusion()]
        )
        backbone = Backbone()
        x = torch.zeros((2, 1, 2, 2))
        record_len = torch.tensor([2])
        transforms = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
        precomputed = tuple(x + value for value in (10, 20, 30))

        fusion(x, x, record_len, transforms, backbone,
               precomputed_multiscale=precomputed)
        self.assertEqual(0, backbone.calls)
        fusion(x, x, record_len, transforms, backbone)
        self.assertEqual(1, backbone.calls)

    def test_remote_ego_is_ignored_current_local_ego_and_remote_rsu_order_are_preserved(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        calls = {"resnet": [], "forward": 0}

        class IdentityStage(torch.nn.Module):
            def forward(self, value):
                return value

        class Backbone(torch.nn.Module):
            def __init__(self):
                super(Backbone, self).__init__()
                self.resnet = IdentityStage()

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.backbone = Backbone()
                original = self.backbone.resnet

                class Counting(torch.nn.Module):
                    def forward(_, value):
                        calls["resnet"].append(value.detach().clone())
                        return (original(value) + 1, original(value) + 2, original(value) + 3)

                self.backbone.resnet = Counting()

            def forward(
                self,
                cav_content,
                use_hil=False,
                hil_remote_multiscale=None,
            ):
                calls["forward"] += 1
                local = self.backbone.resnet(cav_content["spatial_features"][:1])
                if not use_hil:
                    features = self.backbone.resnet(cav_content["spatial_features"])
                else:
                    features = tuple(
                        torch.cat(
                            (
                                local_feature,
                                torch.as_tensor(remote, dtype=local_feature.dtype),
                            ),
                            dim=0,
                        )
                        for local_feature, remote in zip(
                            local, hil_remote_multiscale
                        )
                    )
                return {"fusion_inputs": features, "fused_feature": features[0]}

        model = Model()
        current = torch.tensor([[[[30.0]]], [[[999.0]]]])
        remote = {
            "res0": np.array([[[[100.0]]], [[[10.0]]]], dtype=np.float32),
            "res1": np.array([[[[200.0]]], [[[20.0]]]], dtype=np.float32),
            "res2": np.array([[[[300.0]]], [[[30.0]]]], dtype=np.float32),
        }
        output = run_hil_prefusion_forward(
            model, {"spatial_features": current}, remote
        )

        self.assertEqual(1, len(calls["resnet"]))
        self.assertEqual((1, 1, 1, 1), tuple(calls["resnet"][0].shape))
        self.assertEqual(30.0, float(calls["resnet"][0][0, 0, 0, 0]))
        self.assertEqual(1, calls["forward"])
        for level, expected in enumerate(((31.0, 10.0), (32.0, 20.0), (33.0, 30.0))):
            values = output["fusion_inputs"][level][:, 0, 0, 0].tolist()
            self.assertEqual(list(expected), values)
        self.assertNotIn(100.0, output["fusion_inputs"][0])

    def test_hil_failure_and_following_legacy_forward_do_not_leak_state(self):
        class StatelessModel(object):
            def __init__(self):
                self.calls = []

            def __call__(self, content, **kwargs):
                self.calls.append(dict(kwargs))
                if kwargs.get("use_hil") and content.get("fail"):
                    raise RuntimeError("injected failure")
                return "legacy" if not kwargs else "hil"

        model = StatelessModel()
        remote = {
            "res0": np.zeros((2, 1, 1, 1), dtype=np.float32),
            "res1": np.zeros((2, 1, 1, 1), dtype=np.float32),
            "res2": np.zeros((2, 1, 1, 1), dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            run_hil_prefusion_forward(model, {"fail": True}, remote)
        self.assertEqual("legacy", model({"fail": False}))
        self.assertEqual({}, model.calls[-1])


if __name__ == "__main__":
    unittest.main()
