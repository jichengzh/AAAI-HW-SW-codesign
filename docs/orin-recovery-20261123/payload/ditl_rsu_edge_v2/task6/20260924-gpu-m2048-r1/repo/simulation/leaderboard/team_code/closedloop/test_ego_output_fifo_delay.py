# -*- coding: utf-8 -*-
"""Regression tests for the canonical ego-output FIFO latency API."""

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from action_fifo_delay import ActionFifoDelay, EgoOutputFifoDelay
from test_action_fifo_delay import control


class EgoOutputFifoDelayTest(unittest.TestCase):
    def test_csv_preserves_v2_prefix_and_legacy_suffix_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ego_output_fifo_audit.csv")
            fifo = EgoOutputFifoDelay(delay_ms=200, audit_path=path)
            fifo.release([control(1)], step=0)
            fifo.close()
            with open(path) as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(rows[0][:15], [
            "apply_frame", "generated_source_frame", "generated_ready_frame",
            "applied_source_frame", "applied_ready_frame", "configured_delay_ms",
            "delay_frames", "control_age_frames", "is_startup_neutral",
            "generated_throttle", "generated_steer", "generated_brake",
            "applied_throttle", "applied_steer", "applied_brake",
        ])
        self.assertEqual(rows[0][15:], [
            "step", "source_step", "delay_ms", "age_frames", "is_startup_hold",
        ])
        self.assertEqual(len(rows[1]), 20)

    def test_legacy_name_is_exact_alias(self):
        self.assertIs(ActionFifoDelay, EgoOutputFifoDelay)

    def test_v2_audit_proves_generated_ready_and_applied_frames(self):
        fifo = EgoOutputFifoDelay(delay_ms=200, sim_rate_hz=20)
        for frame in range(5):
            released = fifo.release([control(frame + 1)], step=frame)
        audit = fifo.last_audit
        self.assertEqual(released[0].throttle, 1.0)
        self.assertEqual(audit["apply_frame"], 4)
        self.assertEqual(audit["generated_source_frame"], 4)
        self.assertEqual(audit["generated_ready_frame"], 8)
        self.assertEqual(audit["applied_source_frame"], 0)
        self.assertEqual(audit["applied_ready_frame"], 4)
        self.assertEqual(audit["control_age_frames"], 4)
        self.assertFalse(audit["is_startup_neutral"])

    def test_startup_audit_marks_neutral_control(self):
        fifo = EgoOutputFifoDelay(delay_ms=200, sim_rate_hz=20)
        released = fifo.release([control(7)], step=0)
        self.assertEqual(released[0].throttle, 0.0)
        self.assertTrue(fifo.last_audit["is_startup_neutral"])
        self.assertIsNone(fifo.last_audit["applied_source_frame"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
