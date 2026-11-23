"""The legacy ROS result bridge must validate RSU-only payloads as features."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ditl_edge_protocol import (
    ProtocolError, ProtocolLimits, build_wire_bytes, decode_wire_bytes,
)
from scripts.ditl_ros2_edge_bridge import _finalize_result_wire
import test_coptv2x_hil_backend as backend_tests


RSU_SEMANTICS = "codriving_rsu_multiscale_features.v1"


class RsuResultBridgeTests(unittest.TestCase):
    def wire(self, changes=None, batch_size=1, invalid_value=False):
        outputs = {
            "res0": np.full((batch_size, 16, 128, 256), 11, dtype=np.float32),
            "res1": np.full((batch_size, 32, 64, 128), 22, dtype=np.float32),
            "res2": np.full((batch_size, 64, 32, 64), 33, dtype=np.float32),
        }
        if invalid_value:
            outputs["res0"].flat[0] = np.nan
        legacy_wire, _ = backend_tests.CoptV2xRosBridgeTests()._wire_fixture(outputs=outputs)
        header, _, _ = decode_wire_bytes(legacy_wire)
        header.update({
            "payload_semantics": RSU_SEMANTICS,
            "batch_order": ["rsu"],
            "source_batch_order": ["ego", "rsu"],
        })
        header.update(changes or {})
        return build_wire_bytes(header, outputs)

    def finalize(self, wire):
        return _finalize_result_wire(
            wire,
            {
                "request_id": "run-task3:7", "local_handoff_end": 50,
                "ros_message_start": 23, "ros_message_end": 24,
                "ros_publish_start": 25, "ros_publish_end": 26,
            },
            ros_receive_ns=27,
            receiver_host_id="receiver-host", receiver_boot_id="receiver-boot",
            limits=ProtocolLimits(),
        )

    def test_rsu_result_preserves_feature_values_identity_and_trace_category(self):
        outputs, record = self.finalize(self.wire())
        self.assertEqual(RSU_SEMANTICS, record["source"]["payload_semantics"])
        self.assertEqual("codriving_multiscale_features", record["message"]["message_type"])
        self.assertEqual(["rsu"], record["model"]["batch_order"])
        self.assertEqual(["ego", "rsu"], record["model"]["source_batch_order"])
        self.assertEqual("joint_shcosearch", record["model"]["artifact_identity"]["arm_id"])
        self.assertEqual(7, record["model"]["trigger_identity"]["sequence_no"])
        for name, expected in (("res0", 11), ("res1", 22), ("res2", 33)):
            self.assertEqual(1, outputs[name].shape[0])
            self.assertEqual(np.dtype("float32"), outputs[name].dtype)
            self.assertTrue(np.all(outputs[name] == expected))

    def test_rsu_result_rejects_wrong_payload_or_source_batch_order(self):
        for changes in (
            {"batch_order": ["ego", "rsu"]},
            {"batch_order": ["ego"]},
            {"source_batch_order": None},
            {"source_batch_order": ["rsu", "ego"]},
            {"source_batch_order": ["rsu"]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                self.finalize(self.wire(changes))

    def test_rsu_result_rejects_full_batch_and_self_hashed_nonfinite_values(self):
        with self.assertRaises(ProtocolError):
            self.finalize(self.wire(batch_size=2))
        with self.assertRaises(ProtocolError):
            self.finalize(self.wire(invalid_value=True))

    def test_rsu_result_rejects_hash_and_artifact_trigger_conflicts(self):
        with self.assertRaises(ProtocolError):
            self.finalize(self.wire({"payload_sha256": "a" * 64}))
        with self.assertRaises(ProtocolError):
            self.finalize(self.wire({"arm_id": "compression_only"}))


if __name__ == "__main__":
    unittest.main()
