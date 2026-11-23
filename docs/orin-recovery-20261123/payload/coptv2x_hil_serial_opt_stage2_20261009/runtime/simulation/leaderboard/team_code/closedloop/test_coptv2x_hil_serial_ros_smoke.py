"""Validate real frozen RSU wire payloads without requiring ROS in unit tests."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from coptv2x_hil_artifacts import DEFAULT_CONFIG_DIR
from ditl_edge_protocol import ProtocolError, build_wire_bytes, decode_wire_bytes
import test_coptv2x_hil_backend as backend_tests


class SerialRosSmokeValidationTests(unittest.TestCase):
    def fixture(self, changes=None, batch=1, dtype=np.float32, nonfinite=False):
        outputs = {
            "res0": np.full((batch, 16, 128, 256), 11, dtype=dtype),
            "res1": np.full((batch, 32, 64, 128), 22, dtype=dtype),
            "res2": np.full((batch, 64, 32, 64), 33, dtype=dtype),
        }
        if nonfinite:
            outputs["res0"].flat[0] = np.nan
        wire, _ = backend_tests.CoptV2xRosBridgeTests()._wire_fixture(outputs=outputs)
        header, _, _ = decode_wire_bytes(wire)
        header.update({
            "payload_semantics": "codriving_rsu_multiscale_features.v1",
            "batch_order": ["rsu"], "source_batch_order": ["ego", "rsu"],
        })
        header.update(changes or {})
        return build_wire_bytes(header, outputs), header

    def validate(self, wire, **kwargs):
        from scripts.coptv2x_hil_serial_ros_smoke import validate_rsu_wire

        return validate_rsu_wire(wire, DEFAULT_CONFIG_DIR, **kwargs)

    def test_accepts_exact_rsu_payload_and_reports_identity_not_network_measurements(self):
        wire, header = self.fixture()
        result = self.validate(wire)
        self.assertEqual("run-task3:7", result["request_id"])
        self.assertEqual(3670016, result["payload_bytes"])
        self.assertEqual(header["payload_sha256"], result["expected_payload_sha256"])
        self.assertEqual(header["payload_sha256"], result["actual_payload_sha256"])
        self.assertEqual(header["artifact_identity"]["artifact_sha256"], result["artifact_sha256"])
        self.assertEqual(["rsu"], result["batch_order"])
        self.assertEqual(["ego", "rsu"], result["source_batch_order"])
        self.assertFalse(any("timestamp" in key or "duration" in key for key in result))
        self.assertNotIn("is_ros_smoke", result)  # Pure validation is not ROS evidence.

    def test_rejects_semantics_batch_identity_shape_precision_and_nonfinite_mismatches(self):
        for changes in (
            {"kind": "response"}, {"status": "error"},
            {"payload_semantics": "codriving_multiscale_features"},
            {"batch_order": ["ego", "rsu"]},
            {"source_batch_order": ["rsu", "ego"]},
            {"payload_sha256": "0" * 64},
        ):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                self.validate(self.fixture(changes)[0])
        for kwargs in ({"batch": 2}, {"dtype": np.float16}, {"nonfinite": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProtocolError):
                self.validate(self.fixture(**kwargs)[0])

    def test_receiver_must_match_published_request_and_hash(self):
        wire, header = self.fixture()
        self.validate(wire, expected_request_id="run-task3:7", expected_payload_sha256=header["payload_sha256"])
        with self.assertRaises(ProtocolError):
            self.validate(wire, expected_request_id="other-frame")
        with self.assertRaises(ProtocolError):
            self.validate(wire, expected_payload_sha256="0" * 64)

    def test_uses_cli_config_directory_as_artifact_authority(self):
        from scripts.coptv2x_hil_serial_ros_smoke import validate_rsu_wire

        tree = backend_tests.ArtifactTree()
        self.addCleanup(tree.close)
        wire, _ = self.fixture()
        with self.assertRaises(ProtocolError):
            validate_rsu_wire(wire, tree.config_dir)


if __name__ == "__main__":
    unittest.main()
