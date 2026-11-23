# -*- coding: utf-8 -*-
"""Regression tests for 800 ms RSU information FIFO semantics."""
import os
import sys
import unittest
import ast
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from latency_frame_align import (
    RsuLatencyAligner,
    apply_delayed_collaboration,
    latency_to_delay_frames,
)


class RsuInformationFifoTest(unittest.TestCase):
    def test_csv_writer_persists_sim_time_and_empty_startup_source(self):
        path = Path(HERE).parent / "pnp_infer_action_e2e.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                      and cls.name == "PnP_infer" for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_dump_latency_audit")
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = {"os": os}
        exec(compile(module, str(path), "exec"), namespace)
        aligner = RsuLatencyAligner()
        aligner.submit(4, "rsu-4", 200, "injected_from_E7")
        _, startup = aligner.consume(4)
        _, ready = aligner.consume(8)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta").mkdir()
            state = SimpleNamespace(_latency_audit_enabled=True, save_path=tmp,
                                    _lat_audit_rows=[startup, ready])
            namespace["_dump_latency_audit"](state)
            with (Path(tmp) / "meta/latency_align_audit.csv").open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertIn("source_sim_time_s", rows[0])
        self.assertEqual(rows[0]["source_sim_time_s"], "")
        self.assertEqual(rows[0]["consume_sim_time_s"], "0.2")
        self.assertEqual(rows[1]["source_sim_time_s"], "0.2")
        self.assertEqual(rows[1]["consume_sim_time_s"], "0.4")

    def test_audit_reports_deterministic_source_and_consume_sim_time(self):
        aligner = RsuLatencyAligner(frame_ms=50.0)
        aligner.submit(4, {"rsu_data": []}, 200, "injected_from_E7")
        _, startup = aligner.consume(4)
        _, ready = aligner.consume(8)
        self.assertIn("source_sim_time_s", startup)
        self.assertIn("consume_sim_time_s", startup)
        self.assertIsNone(startup["source_sim_time_s"])
        self.assertEqual(startup["consume_sim_time_s"], 0.2)
        self.assertEqual(ready["source_sim_time_s"], 0.2)
        self.assertEqual(ready["consume_sim_time_s"], 0.4)

    def test_sim_time_uses_instance_frame_duration_and_preserves_zoh_source(self):
        aligner = RsuLatencyAligner(frame_ms=100.0)
        aligner.submit(4, "rsu-4", 200, "injected_from_E7")
        _, ready = aligner.consume(6)
        _, held = aligner.consume(7)
        self.assertIn("source_sim_time_s", ready)
        self.assertEqual(ready["source_sim_time_s"], 0.4)
        self.assertEqual(ready["consume_sim_time_s"], 0.6)
        self.assertTrue(held["is_zoh_held"])
        self.assertEqual(held["source_sim_time_s"], 0.4)
        self.assertEqual(held["consume_sim_time_s"], 0.7)

    def test_standard_delays_are_not_silently_capped(self):
        self.assertEqual(latency_to_delay_frames(0), 0)
        self.assertEqual(latency_to_delay_frames(200), 4)
        self.assertEqual(latency_to_delay_frames(500), 10)
        self.assertEqual(latency_to_delay_frames(800), 16)

    def test_explicit_limit_raises_instead_of_clipping(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            latency_to_delay_frames(800, max_delay_frames=10)

    def test_startup_returns_ego_only_without_current_collaboration(self):
        current_cars = [{"frame": 7, "role": "ego"}, {"frame": 7, "role": "cav"}]
        current_rsu = [{"frame": 7, "role": "rsu"}]
        cars, rsu = apply_delayed_collaboration(
            current_cars, current_rsu, used_payload=None
        )
        self.assertEqual(cars, [current_cars[0], None])
        self.assertEqual(rsu, [])
        self.assertEqual(current_cars[1]["frame"], 7)
        self.assertEqual(current_rsu[0]["frame"], 7)

    def test_delayed_payload_replaces_every_collaboration_slot(self):
        current = [{"frame": 9}, {"frame": 9}, {"frame": 9}]
        payload = {
            "car_data": [None, {"frame": 3}, None],
            "rsu_data": [{"frame": 3}],
        }
        cars, rsu = apply_delayed_collaboration(current, [], payload)
        self.assertEqual(cars, [current[0], {"frame": 3}, None])
        self.assertEqual(rsu, [{"frame": 3}])

    def test_startup_audit_marks_ego_only_and_has_no_used_source(self):
        payload, audit = RsuLatencyAligner().consume(0)
        self.assertIsNone(payload)
        self.assertTrue(audit["is_ego_only_startup"])
        self.assertIsNone(audit["used_rsu_source_frame"])
        self.assertIsNone(audit["used_rsu_ready_frame"])
        self.assertIsNone(audit["information_age_frames"])

    def test_unbounded_gc_keeps_last_used_without_none_arithmetic(self):
        fifo = RsuLatencyAligner(max_delay_frames=None)
        fifo.submit(0, "rsu-0", 0, "injected")
        fifo.consume(0)
        fifo.submit(1, "rsu-1", 0, "injected")
        fifo.consume(1)
        fifo.gc(1)
        self.assertNotIn(0, fifo._buffer)
        self.assertIn(1, fifo._buffer)

    def test_unbounded_gc_before_consume_keeps_unready_entry(self):
        fifo = RsuLatencyAligner(max_delay_frames=None)
        fifo.submit(0, "rsu-0", 800, "injected")
        fifo.gc(0)
        self.assertIn(0, fifo._buffer)

    def test_v2_audit_records_submission_and_restores_zoh_latency(self):
        fifo = RsuLatencyAligner()
        fifo.submit(4, "rsu-4", 100, "injected")
        payload, audit = fifo.consume(6)

        self.assertEqual(payload, "rsu-4")
        self.assertEqual(audit["submitted_rsu_source_frame"], 4)
        self.assertEqual(audit["submitted_rsu_ready_frame"], 6)
        self.assertEqual(audit["configured_delay_ms"], 100)
        self.assertEqual(audit["delay_frames"], 2)

        held_payload, held_audit = fifo.consume(7)
        self.assertEqual(held_payload, "rsu-4")
        self.assertTrue(held_audit["is_zoh_held"])
        self.assertEqual(held_audit["used_rsu_source_frame"], 4)
        self.assertEqual(held_audit["used_rsu_ready_frame"], 6)
        self.assertEqual(held_audit["configured_delay_ms"], 100)
        self.assertEqual(held_audit["delay_frames"], 2)
        self.assertEqual(held_audit["information_age_frames"], 3)

    def test_v2_audit_proves_source_ready_and_information_age(self):
        fifo = RsuLatencyAligner()
        for frame in range(17):
            fifo.submit(frame, "rsu-%d" % frame, 800, "injected")
            payload, audit = fifo.consume(frame)
        self.assertEqual(payload, "rsu-0")
        self.assertEqual(audit["consume_frame"], 16)
        self.assertEqual(audit["used_rsu_source_frame"], 0)
        self.assertEqual(audit["used_rsu_ready_frame"], 16)
        self.assertEqual(audit["information_age_frames"], 16)
        self.assertFalse(audit["is_ego_only_startup"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
