"""Receiver optimizations reuse only immutable wire storage, never caller memory."""
import os
import socket
import struct
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from coptv2x_hil_client import AsyncHilClient, Ros2HilTransport, _remote_rsu_only, create_ros2_hil_client
from coptv2x_hil_backend import hash_feature_payload
from coptv2x_hil_proxy_transport import ReverseSshRosProxyTransport, _decode_result_wire
from coptv2x_hil_result_buffer import CoptV2xHilResultBuffer, HilResultEnvelope
from ditl_edge_protocol import (FRAME_PREFIX, MAGIC, decode_tensor_bundle,
                                encode_tensor_bundle, prepare_envelope_header)
from test_coptv2x_hil_client import ARM_ID, _small_result, _trigger
from test_coptv2x_hil_serial_clock import RSU_SEMANTICS, buffer, result


def rsu_contract(identity, outputs, *, payload_semantics):
    if payload_semantics != RSU_SEMANTICS:
        raise ValueError("semantics mismatch")
    expected = {"res0": (1, 1, 2, 2), "res1": (1, 2, 1, 1), "res2": (1, 4, 1, 1)}
    if set(outputs) != set(expected):
        raise ValueError("schema mismatch")
    for name, shape in expected.items():
        if outputs[name].shape != shape or outputs[name].dtype != np.float32:
            raise ValueError("shape mismatch")
        if not np.isfinite(outputs[name]).all():
            raise ValueError("finite mismatch")


def decoded_outputs(envelope):
    metadata, payload = encode_tensor_bundle(dict(envelope.feature_result.outputs))
    return decode_tensor_bundle(metadata, payload, readonly=True)


def marked_result(trigger, marker):
    envelope = result(trigger, semantics=RSU_SEMANTICS)
    outputs = {name: np.full_like(tensor, marker)
               for name, tensor in envelope.feature_result.outputs.items()}
    return HilResultEnvelope(replace(envelope.feature_result, outputs=outputs,
                                    payload_sha256=hash_feature_payload(outputs)), ("rsu",))


def result_wire(envelope):
    value = envelope.feature_result
    header = dict(value.trigger_identity)
    header.update({
        "protocol_version": "ditl.edge.v1", "kind": "result", "status": "ok",
        "request_id": "receiver-memory-test",
        "trigger_identity": dict(value.trigger_identity),
        "batch_order": list(envelope.batch_order),
        "artifact_identity": dict(value.artifact_identity),
        "payload_semantics": value.payload_semantics,
        "payload_sha256": value.payload_sha256,
        "remote_input_tensor_bytes": 0,
        "edge_timestamps_ns": dict(value.timestamps_ns),
        "durations_ns": dict(value.durations_ns),
    })
    bundle, payload = encode_tensor_bundle(dict(value.outputs))
    header["frame_id"] = value.trigger_identity["source_frame"]
    header["sequence_id"] = value.trigger_identity["sequence_no"]
    header_bytes = prepare_envelope_header(header, bundle)
    return FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload)) + header_bytes + payload


class ReceiverMemoryTests(unittest.TestCase):
    def admit(self, outputs, trigger=None):
        trigger = trigger or _trigger(0)
        envelope = result(trigger, semantics=RSU_SEMANTICS)
        value = replace(envelope.feature_result, outputs=outputs)
        store = buffer(rsu_contract)
        store.register_trigger(trigger, 0)
        self.assertTrue(store.accept(HilResultEnvelope(value, ("rsu",)), 0).accepted)
        return store.get_exact_result(trigger)

    def test_real_decoder_storage_is_shared_through_admission_and_prefusion(self):
        outputs = decoded_outputs(result(_trigger(0), semantics=RSU_SEMANTICS))
        admitted = self.admit(outputs)
        remote = _remote_rsu_only(admitted.outputs, payload_semantics=RSU_SEMANTICS)
        for name, forwarded in zip(("res0", "res1", "res2"), remote):
            self.assertTrue(np.shares_memory(outputs[name], admitted.outputs[name]))
            self.assertTrue(np.shares_memory(outputs[name], forwarded))
            self.assertFalse(forwarded.flags.writeable)
            with self.assertRaises(ValueError):
                forwarded.setflags(write=True)
        following = decoded_outputs(marked_result(_trigger(1), 9))
        self.assertTrue(np.all(following["res0"] == 9))
        self.assertTrue(np.all(admitted.outputs["res0"] == 0))

    def test_readonly_views_with_mutable_owners_are_snapshotted(self):
        for owner_kind in ("ndarray", "bytearray", "readonly_memoryview"):
            with self.subTest(owner_kind=owner_kind):
                sources = {}
                owners = []
                template = result(_trigger(0), semantics=RSU_SEMANTICS).feature_result.outputs
                for name, tensor in template.items():
                    if owner_kind == "ndarray":
                        owner = np.array(tensor, copy=True)
                        view = owner.view()
                        view.setflags(write=False)
                    else:
                        owner = bytearray(tensor.tobytes())
                        backing = memoryview(owner).toreadonly() if owner_kind == "readonly_memoryview" else owner
                        view = np.frombuffer(backing, dtype=np.float32).reshape(tensor.shape)
                        view.setflags(write=False)
                    sources[name] = view
                    owners.append(owner)
                admitted = self.admit(sources)
                forwarded = _remote_rsu_only(sources, payload_semantics=RSU_SEMANTICS)
                for name, item in zip(("res0", "res1", "res2"), forwarded):
                    self.assertFalse(np.shares_memory(sources[name], admitted.outputs[name]))
                    self.assertFalse(np.shares_memory(sources[name], item))
                for owner in owners:
                    if isinstance(owner, np.ndarray):
                        owner.fill(9)
                    else:
                        owner[:] = b"\xff" * len(owner)
                self.assertTrue(all(np.all(tensor == 0) for tensor in admitted.outputs.values()))
                self.assertTrue(all(np.all(tensor == 0) for tensor in forwarded))

    def test_readonly_memoryview_terminal_bytes_is_reused_through_prefusion(self):
        template = result(_trigger(0), semantics=RSU_SEMANTICS).feature_result.outputs
        outputs = {
            name: np.frombuffer(memoryview(tensor.tobytes()), dtype=np.float32).reshape(tensor.shape)
            for name, tensor in template.items()
        }
        admitted = self.admit(outputs)
        remote = _remote_rsu_only(admitted.outputs, payload_semantics=RSU_SEMANTICS)
        for name, forwarded in zip(("res0", "res1", "res2"), remote):
            self.assertTrue(np.shares_memory(outputs[name], admitted.outputs[name]))
            self.assertTrue(np.shares_memory(outputs[name], forwarded))
            self.assertFalse(forwarded.flags.writeable)
            with self.assertRaises(ValueError):
                forwarded.setflags(write=True)

    def test_legacy_batch2_still_copies_even_with_immutable_decoder_storage(self):
        from test_coptv2x_hil_client import _small_contract_validator
        trigger = _trigger(0)
        template = _small_result(trigger)
        outputs = decoded_outputs(HilResultEnvelope(template, ("ego", "rsu")))
        store = CoptV2xHilResultBuffer(
            "run-live", "Town05-route-01", "scenario-01", ARM_ID, 4,
            contract_validator=_small_contract_validator,
        )
        store.register_trigger(trigger, 0)
        self.assertTrue(store.accept(HilResultEnvelope(replace(template, outputs=outputs), ("ego", "rsu")), 0).accepted)
        admitted = store.get_exact_result(trigger)
        forwarded = _remote_rsu_only(admitted.outputs)
        for name, tensor in zip(("res0", "res1", "res2"), forwarded):
            self.assertFalse(np.shares_memory(outputs[name], admitted.outputs[name]))
            self.assertFalse(np.shares_memory(admitted.outputs[name], tensor))

    def test_actual_proxy_decoder_optin_pins_readonly_storage_legacy_remains_writable(self):
        wire = result_wire(result(_trigger(0), semantics=RSU_SEMANTICS))
        envelope, _ = _decode_result_wire(wire, 1024 * 1024, readonly_results=True)
        admitted = self.admit(envelope.feature_result.outputs)
        self.assertFalse(envelope.feature_result.outputs["res0"].flags.writeable)
        self.assertTrue(np.shares_memory(envelope.feature_result.outputs["res0"], admitted.outputs["res0"]))
        legacy, _ = _decode_result_wire(wire, 1024 * 1024)
        self.assertTrue(legacy.feature_result.outputs["res0"].flags.writeable)
        corrupted = wire[:-1] + bytes([wire[-1] ^ 1])
        with self.assertRaises(ValueError):
            _decode_result_wire(corrupted, 1024 * 1024, readonly_results=True)

    def test_actual_proxy_transport_passes_immutable_storage_to_async_admission(self):
        transport = ReverseSshRosProxyTransport(listen_port=0, readonly_results=True)
        client = AsyncHilClient(transport, buffer(rsu_contract), result_timeout_s=1.)
        client.start()
        connection = socket.create_connection(("127.0.0.1", transport.bound_port), timeout=1.)
        stream = connection.makefile("rb")
        try:
            for sequence, marker in ((0, 0), (1, 9)):
                trigger = _trigger(sequence)
                self.assertTrue(client.publish_trigger(trigger).accepted)
                self.assertTrue(stream.readline().startswith(b"TRIGGER "))
                wire = result_wire(marked_result(trigger, marker))
                connection.sendall(struct.pack("!I", len(wire)) + wire)
                admitted = client.wait_for_result(trigger)
                self.assertEqual(b"ACK receiver-memory-test\n", stream.readline())
                forwarded = _remote_rsu_only(admitted.outputs, payload_semantics=RSU_SEMANTICS)
                self.assertTrue(np.shares_memory(admitted.outputs["res0"], forwarded[0]))
                self.assertFalse(forwarded[0].flags.writeable)
                self.assertTrue(np.all(forwarded[0] == marker))
                if sequence == 0:
                    first = admitted
            self.assertTrue(np.all(first.outputs["res0"] == 0))
        finally:
            client.shutdown()
            stream.close()
            connection.close()

    def test_factory_activates_readonly_decode_only_for_serial_mode_both_transports(self):
        with tempfile.TemporaryDirectory() as directory:
            for transport in ("ros2", "reverse_ssh_ros_proxy"):
                for mode, readonly in (("async", False), ("serial_fixed_comm", True)):
                    config = {"run_id": "run-live", "route_id": "Town05-route-01",
                              "scenario_id": "scenario-01", "arm_id": ARM_ID,
                              "audit_jsonl": directory + "/audit.jsonl", "transport": transport,
                              "timing_mode": mode}
                    client = create_ros2_hil_client(config)
                    self.assertEqual(readonly, client.transport.readonly_results)

    def test_immutable_reuse_keeps_feature_hash_schema_and_finite_validation(self):
        trigger = _trigger(0)
        envelope = result(trigger, semantics=RSU_SEMANTICS)
        original = envelope.feature_result
        for change, expected in (("hash", "payload_hash_mismatch"),
                                 ("schema", "payload_contract_invalid"),
                                 ("finite", "payload_contract_invalid")):
            with self.subTest(change=change):
                outputs = dict(original.outputs)
                if change == "schema":
                    del outputs["res2"]
                elif change == "finite":
                    outputs["res0"] = np.full((1, 1, 2, 2), np.nan, np.float32)
                bundle, payload = encode_tensor_bundle(outputs)
                outputs = decode_tensor_bundle(bundle, payload, readonly=True)
                candidate = replace(original, outputs=outputs,
                                    payload_sha256="bad" if change == "hash" else original.payload_sha256)
                store = buffer(rsu_contract)
                store.register_trigger(trigger, 0)
                decision = store.accept(HilResultEnvelope(candidate, ("rsu",)), 0)
                self.assertFalse(decision.accepted)
                self.assertEqual(expected, decision.reason)
                if change != "hash":
                    with self.assertRaises(ValueError):
                        _remote_rsu_only(outputs, payload_semantics=RSU_SEMANTICS)

    def test_real_ros_callback_readonly_decode_pins_snapshot_across_next_message(self):
        class Publisher:
            def __init__(self):
                self.messages = []
            def publish(self, value):
                self.messages.append(value)
        class Node:
            def __init__(self, name):
                pass
            def create_publisher(self, *args):
                return Publisher()
            def create_subscription(self, *args):
                return args[2]
            def destroy_node(self):
                pass
        class Executor:
            def add_node(self, node):
                pass
            def spin(self):
                pass
            def shutdown(self, **kwargs):
                pass
        def module(name, **attributes):
            value = types.ModuleType(name)
            value.__dict__.update(attributes)
            return value
        option = types.SimpleNamespace
        modules = {
            "rclpy": module("rclpy", ok=lambda: True),
            "rclpy.node": module("rclpy.node", Node=Node),
            "rclpy.executors": module("rclpy.executors", SingleThreadedExecutor=Executor),
            "rclpy.qos": module("rclpy.qos", QoSProfile=lambda **kw: option(**kw),
                                DurabilityPolicy=option(VOLATILE=0),
                                HistoryPolicy=option(KEEP_LAST=0),
                                ReliabilityPolicy=option(RELIABLE=0)),
            "std_msgs": module("std_msgs"),
            "std_msgs.msg": module("std_msgs.msg", String=option, UInt8MultiArray=option),
        }
        delivered = []
        with mock.patch.dict(sys.modules, modules):
            transport = Ros2HilTransport(readonly_results=True)
            transport.start(delivered.append)
            try:
                caller = bytearray(result_wire(result(_trigger(0), semantics=RSU_SEMANTICS)))
                transport._node.on_result(option(data=caller))
                first = delivered[0]
                admitted = self.admit(first.feature_result.outputs)
                caller[:] = b"\xff" * len(caller)
                transport._node.on_result(option(data=result_wire(marked_result(_trigger(1), 9))))
                self.assertEqual(2, len(delivered))
                self.assertTrue(np.all(delivered[1].feature_result.outputs["res0"] == 9))
                self.assertEqual(2, len(transport._node.ack_publisher.messages))
                for name, tensor in admitted.outputs.items():
                    self.assertTrue(np.shares_memory(first.feature_result.outputs[name], tensor))
                    self.assertFalse(tensor.flags.writeable)
                    self.assertTrue(np.all(tensor == 0))
            finally:
                transport.close()


if __name__ == "__main__":
    unittest.main()
