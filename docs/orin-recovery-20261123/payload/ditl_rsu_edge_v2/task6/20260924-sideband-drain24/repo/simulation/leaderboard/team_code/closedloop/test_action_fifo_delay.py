# -*- coding: utf-8 -*-
"""Unit tests for action-side FIFO latency. Pure Python; no CARLA/GPU."""

import os
import sys
import tempfile
import unittest
import csv

sys.path.insert(0, os.path.dirname(__file__))

from action_fifo_delay import ActionFifoDelay


class FakeVehicleControl:
    def __init__(
        self,
        throttle=0.0,
        steer=0.0,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
        gear=0,
    ):
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse
        self.manual_gear_shift = manual_gear_shift
        self.gear = gear


def control(value):
    return FakeVehicleControl(
        throttle=float(value),
        steer=-float(value),
        brake=float(value) / 10.0,
        hand_brake=value % 2 == 0,
        reverse=value % 3 == 0,
        manual_gear_shift=True,
        gear=value,
    )


class ActionFifoDelayTest(unittest.TestCase):
    def test_zero_ms_is_exact_passthrough(self):
        delay = ActionFifoDelay(delay_ms=0, sim_rate_hz=20)
        generated = [control(7)]

        released = delay.release(generated, step=11)

        self.assertIsNot(released, generated)
        self.assertIsNot(released[0], generated[0])
        self.assertEqual(released[0].throttle, generated[0].throttle)
        self.assertEqual(released[0].steer, generated[0].steer)
        self.assertEqual(released[0].brake, generated[0].brake)
        self.assertEqual(released[0].gear, generated[0].gear)
        self.assertEqual(delay.last_audit["source_step"], 11)
        self.assertEqual(delay.last_audit["delay_frames"], 0)
        self.assertFalse(delay.last_audit["is_startup_hold"])

    def test_requested_delays_release_exact_historical_tick(self):
        for delay_ms, delay_frames in ((200, 4), (500, 10), (800, 16)):
            with self.subTest(delay_ms=delay_ms):
                delay = ActionFifoDelay(delay_ms=delay_ms, sim_rate_hz=20)
                outputs = []
                for step in range(delay_frames + 2):
                    outputs.append(delay.release([control(step + 1)], step=step))

                for startup in outputs[:delay_frames]:
                    self.assertEqual(startup[0].throttle, 0.0)
                    self.assertEqual(startup[0].steer, 0.0)
                    self.assertEqual(startup[0].brake, 0.0)
                self.assertEqual(outputs[delay_frames][0].throttle, 1.0)
                self.assertEqual(outputs[delay_frames + 1][0].throttle, 2.0)
                self.assertEqual(delay.last_audit["source_step"], 1)
                self.assertEqual(delay.last_audit["age_frames"], delay_frames)

    def test_buffer_snapshot_is_not_changed_by_later_mutation(self):
        delay = ActionFifoDelay(delay_ms=100, sim_rate_hz=20)
        generated = [control(3)]
        delay.release(generated, step=0)
        generated[0].throttle = 99.0

        delay.release([control(4)], step=1)
        released = delay.release([control(5)], step=2)

        self.assertEqual(released[0].throttle, 3.0)
        self.assertEqual(released[0].gear, 3)
        self.assertTrue(released[0].manual_gear_shift)

    def test_dict_shape_and_none_entries_are_preserved(self):
        delay = ActionFifoDelay(delay_ms=50, sim_rate_hz=20)
        generated = {0: control(6), 1: None}

        startup = delay.release(generated, step=0)
        released = delay.release({0: control(7), 1: None}, step=1)

        self.assertIsInstance(startup, dict)
        self.assertEqual(set(startup), {0, 1})
        self.assertIsNone(startup[1])
        self.assertIsInstance(released, dict)
        self.assertEqual(released[0].throttle, 6.0)
        self.assertIsNone(released[1])

    def test_fractional_tick_uses_ceiling_and_negative_delay_is_rejected(self):
        self.assertEqual(ActionFifoDelay(delay_ms=1, sim_rate_hz=20).delay_frames, 1)
        self.assertEqual(ActionFifoDelay(delay_ms=51, sim_rate_hz=20).delay_frames, 2)
        with self.assertRaises(ValueError):
            ActionFifoDelay(delay_ms=-1, sim_rate_hz=20)

    def test_audit_csv_records_generated_and_applied_controls(self):
        delay = ActionFifoDelay(delay_ms=50, sim_rate_hz=20)
        delay.release([control(2)], step=4)
        delay.release([control(3)], step=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = os.path.join(tmpdir, "action_fifo_audit.csv")
            delay.write_audit_csv(audit_path)
            with open(audit_path, newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["is_startup_hold"], "True")
        self.assertEqual(rows[0]["generated_throttle"], "2.0")
        self.assertEqual(rows[0]["applied_throttle"], "0.0")
        self.assertEqual(rows[1]["source_step"], "4")
        self.assertEqual(rows[1]["age_frames"], "1")
        self.assertEqual(rows[1]["generated_throttle"], "3.0")
        self.assertEqual(rows[1]["applied_throttle"], "2.0")

    def test_configured_audit_path_is_flushed_during_long_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = os.path.join(tmpdir, "meta", "action_fifo_audit.csv")
            delay = ActionFifoDelay(
                delay_ms=50,
                sim_rate_hz=20,
                audit_path=audit_path,
                audit_flush_interval=2,
            )
            delay.release([control(1)], step=0)
            self.assertFalse(os.path.exists(audit_path))
            delay.release([control(2)], step=1)
            self.assertTrue(os.path.exists(audit_path))
            with open(audit_path, newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
