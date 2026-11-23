"""Canonical checksum reuse must preserve payload and descriptor integrity."""
import copy
from dataclasses import replace
import hashlib
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ditl_edge_protocol as protocol
import coptv2x_hil_result_buffer as buffer_module
from coptv2x_hil_backend import hash_feature_payload
from coptv2x_hil_result_buffer import CoptV2xHilResultBuffer, HilResultEnvelope
from test_coptv2x_hil_receiver_memory import rsu_contract
from test_coptv2x_hil_serial_clock import RSU_SEMANTICS, result
from test_coptv2x_hil_client import _trigger
import test_coptv2x_serial_payload as payload_fixtures
from test_coptv2x_hil_backend import _trigger as backend_trigger
from ditl_rsu_pipeline import CoptV2xHilPipeline
from ditl_rsu_source import LocalRsuFrame


class IntegrityDedupTests(unittest.TestCase):
    def outputs(self):
        return dict(result(_trigger(0), semantics=RSU_SEMANTICS).feature_result.outputs)

    def optimized(self, outputs=None, readonly=True):
        outputs = self.outputs() if outputs is None else outputs
        metadata, payload = protocol.encode_tensor_bundle(outputs, feature_checksum=True)
        return metadata, payload, protocol.decode_tensor_bundle(metadata, payload, readonly=readonly)

    def admit(self, outputs, digest, dedup=True):
        trigger = _trigger(0)
        template = result(trigger, semantics=RSU_SEMANTICS)
        store = CoptV2xHilResultBuffer(
            trigger.run_id, trigger.route_id, trigger.scenario_id, trigger.arm_id, 4,
            contract_validator=rsu_contract, timing_mode="serial_fixed_comm",
            checksum_dedup=dedup,
        )
        store.register_trigger(trigger, 0)
        decision = store.accept(HilResultEnvelope(replace(
            template.feature_result, outputs=outputs, payload_sha256=digest,
        ), ("rsu",)), 0)
        return store, decision

    def test_verified_decode_eliminates_buffer_rehash_without_changing_digest(self):
        metadata, _, decoded = self.optimized()
        self.assertEqual("codriving_feature_payload.v1", metadata["sha256_scope"])
        # Golden digest uses the pre-existing descriptor JSON and zero FP32 bytes.
        self.assertEqual("9a210d26a2debb734ba6bf5673135eff313bbc02727853b06b387dd52c78f8c6",
                         metadata["sha256"])
        self.assertEqual(metadata["sha256"], hash_feature_payload(self.outputs()))
        self.assertEqual(metadata["sha256"], protocol.verified_feature_payload_sha256(decoded))
        with mock.patch.object(buffer_module, "hash_feature_payload", side_effect=AssertionError("duplicate scan")):
            store, decision = self.admit(decoded, metadata["sha256"])
        self.assertTrue(decision.accepted, decision.reason)
        self.assertIs(decoded, store.get_exact_result(_trigger(0)).outputs)
        with self.assertRaises(TypeError):
            decoded["res0"] = decoded["res1"]
        with self.assertRaises(ValueError):
            decoded["res0"].setflags(write=True)

    def test_corrupted_payload_descriptors_and_scope_are_rejected(self):
        metadata, payload, _ = self.optimized()
        corrupt = payload[:-1] + bytes([payload[-1] ^ 1])
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_tensor_bundle(metadata, corrupt, readonly=True)
        for change in ("shape", "dtype", "name", "scope", "downgrade", "offset"):
            with self.subTest(change=change):
                changed = copy.deepcopy(metadata)
                if change == "shape":
                    changed["arrays"][0]["shape"] = [1, 1, 1, 4]
                elif change == "dtype":
                    changed["arrays"][0]["dtype"] = "int32"
                elif change == "name":
                    changed["arrays"][0]["name"] = "renamed"
                elif change == "offset":
                    changed["arrays"][0]["offset"] = 4
                elif change == "scope":
                    changed["sha256_scope"] = "unknown"
                else:
                    del changed["sha256_scope"]
                with self.assertRaises(protocol.ProtocolError):
                    protocol.decode_tensor_bundle(changed, payload, readonly=True)

    def test_untrusted_flags_cannot_skip_wire_or_buffer_checksum(self):
        outputs = self.outputs()
        metadata, payload = protocol.encode_tensor_bundle(outputs)
        metadata.update(checksum_verified=True, verified_payload_sha256="0" * 64)
        corrupt = payload[:-1] + bytes([payload[-1] ^ 1])
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_tensor_bundle(metadata, corrupt, readonly=True)
        class ForgedDict(dict):
            checksum_verified = True
            payload_sha256 = "0" * 64
        _, decision = self.admit(ForgedDict(outputs), "0" * 64)
        self.assertFalse(decision.accepted)
        self.assertEqual("payload_hash_mismatch", decision.reason)

    def test_writable_copies_views_and_changed_descriptors_do_not_reuse_proof(self):
        metadata, _, decoded = self.optimized()
        for outputs in (dict(decoded), {name: tensor.copy() for name, tensor in decoded.items()},
                        {name: tensor.view() for name, tensor in decoded.items()}):
            self.assertIsNone(protocol.verified_feature_payload_sha256(outputs))
        mutable = {name: tensor.copy() for name, tensor in decoded.items()}
        mutable["res0"].flat[0] = 9
        _, decision = self.admit(mutable, metadata["sha256"])
        self.assertEqual("payload_hash_mismatch", decision.reason)
        decoded["res0"].shape = (1, 1, 1, 4)
        self.assertIsNone(protocol.verified_feature_payload_sha256(decoded))
        _, decision = self.admit(decoded, metadata["sha256"])
        self.assertEqual("payload_contract_invalid", decision.reason)

    def test_writable_decode_and_legacy_buffer_still_hash(self):
        metadata, _, decoded = self.optimized(readonly=False)
        self.assertTrue(decoded["res0"].flags.writeable)
        self.assertIsNone(protocol.verified_feature_payload_sha256(decoded))
        for outputs, dedup in ((decoded, True), (self.optimized()[2], False)):
            with mock.patch.object(buffer_module, "hash_feature_payload", wraps=hash_feature_payload) as scan:
                _, decision = self.admit(outputs, metadata["sha256"], dedup=dedup)
            self.assertTrue(decision.accepted)
            self.assertEqual(1, scan.call_count)

    def test_opt_in_rejects_compression_and_non_serial_buffer(self):
        for compression in ("zlib", "lz4"):
            with self.assertRaises(protocol.ProtocolError):
                protocol.encode_tensor_bundle(self.outputs(), compression, feature_checksum=True)
        trigger = _trigger(0)
        with self.assertRaises(ValueError):
            CoptV2xHilResultBuffer(
                trigger.run_id, trigger.route_id, trigger.scenario_id, trigger.arm_id, 4,
                contract_validator=rsu_contract, checksum_dedup=True,
            )

    def test_valid_checksum_does_not_bypass_finite_or_declared_digest_checks(self):
        outputs = self.outputs()
        outputs["res0"].flat[0] = np.nan
        metadata, _, decoded = self.optimized(outputs)
        _, decision = self.admit(decoded, metadata["sha256"])
        self.assertEqual("payload_contract_invalid", decision.reason)
        metadata, _, decoded = self.optimized()
        _, decision = self.admit(decoded, "0" * 64)
        self.assertEqual("payload_hash_mismatch", decision.reason)

    def test_legacy_none_zlib_lz4_keep_raw_checksum_and_mutability(self):
        for compression in ("none", "zlib", "lz4"):
            with self.subTest(compression=compression):
                try:
                    metadata, payload = protocol.encode_tensor_bundle(self.outputs(), compression)
                except protocol.ProtocolError as error:
                    if compression == "lz4" and "unavailable" in str(error):
                        continue
                    raise
                self.assertNotIn("sha256_scope", metadata)
                decoded = protocol.decode_tensor_bundle(metadata, payload)
                raw = b"".join(memoryview(decoded[name]).cast("B") for name in sorted(decoded))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), metadata["sha256"])
                self.assertTrue(all(value.flags.writeable for value in decoded.values()))
                self.assertIsNone(protocol.verified_feature_payload_sha256(decoded))

    def test_optimized_pipeline_runs_exactly_one_sender_payload_hash(self):
        fixture = payload_fixtures.SerialPayloadTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        backend = fixture.backend(rsu_only=True, checksum_dedup=True)
        backend.load()  # Artifact hashing is not payload hashing.
        frame = LocalRsuFrame(
            frame_id=42, sequence_id=7, data_available_ns=5,
            arrays={"spatial_features": fixture.spatial()},
            source_metadata={"trigger_identity": backend_trigger("joint_shcosearch").to_metadata(),
                             "batch_order": ["ego", "rsu"]},
        )
        with mock.patch.object(protocol.hashlib, "sha256", wraps=hashlib.sha256) as scans:
            encoded = CoptV2xHilPipeline(backend, run_id="run-task3", checksum_dedup=True).process(frame)
        self.assertEqual(1, scans.call_count)
        with mock.patch.object(protocol.hashlib, "sha256", wraps=hashlib.sha256) as scans:
            decoded = protocol.decode_tensor_bundle(dict(encoded.tensor_bundle), encoded.payload, readonly=True)
        self.assertEqual(1, scans.call_count)
        self.assertEqual(encoded.header["payload_sha256"], protocol.verified_feature_payload_sha256(decoded))


if __name__ == "__main__":
    unittest.main()
