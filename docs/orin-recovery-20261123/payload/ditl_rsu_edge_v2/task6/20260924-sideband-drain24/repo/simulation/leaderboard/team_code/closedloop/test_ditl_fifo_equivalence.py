from __future__ import print_function

import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ditl_event_scheduler import (  # noqa: E402
    DitlEventScheduler, DitlReplayEngine, ReplaySubmission,
)
from ditl_impairment_policy import DeterministicImpairmentPolicy  # noqa: E402
from ditl_replay_config import BurstConfig, ImpairmentConfig  # noqa: E402
from ditl_replay_schema import ReplaySample  # noqa: E402
from ditl_trace_library import ReplaySampleCursor  # noqa: E402
from latency_frame_align import RsuLatencyAligner  # noqa: E402


TICK_NS = 50 * 1000 * 1000


def _sample(delay_ms):
    delay_ns = delay_ms * 1000 * 1000
    return ReplaySample(
        sample_id="fixed-%d" % delay_ms,
        ordinal=0,
        source_trace_id="fixed-delay",
        source_sequence_id=0,
        base_delay_ns=delay_ns,
        stage_ns={
            "acquisition_preprocess": delay_ns,
            "gpu_inference": 0,
            "packing_postprocess": 0,
            "transport_receive": 0,
        },
        raw_payload_bytes=1,
        wire_payload_bytes=1,
        capture_condition_id="fixed-delay-condition",
    )


def _event_sequence(delay_ms, frame_count):
    impairment = ImpairmentConfig(
        extra_delay_ns=0, jitter_ns=0, drop_rate=0.0,
        burst=BurstConfig(False, 0.0, 1.0), reorder_rate=0.0,
        reorder_extra_ns=0, duplicate_rate=0.0,
    )
    engine = DitlReplayEngine(
        ReplaySampleCursor((_sample(delay_ms),), "cycle"),
        DeterministicImpairmentPolicy(20260923, impairment),
        DitlEventScheduler(frame_count + 1, None),
        ("rsu-1",), None, None,
    )
    used = []
    for frame in range(frame_count):
        engine.submit(ReplaySubmission(
            route_id="route", stream_id="rsu-1", source_id="rsu",
            source_frame=frame, source_sim_time_ns=frame * TICK_NS,
            sequence_id=frame, payload={"frame": frame},
        ))
        used.append(engine.advance(frame, frame * TICK_NS).streams[
            "rsu-1"].source_frame)
    return used


def _legacy_sequence(delay_ms, frame_count):
    aligner = RsuLatencyAligner(frame_ms=50.0, max_delay_frames=None)
    used = []
    for frame in range(frame_count):
        aligner.submit(frame, {"frame": frame}, delay_ms, "fixed")
        _, audit = aligner.consume(frame)
        used.append(audit["used_rsu_frame_id"])
    return used


class FixedFifoEquivalenceTest(unittest.TestCase):
    def test_fixed_delays_match_legacy_fifo_message_selection(self):
        for delay_ms in (0, 100, 200, 500, 800):
            with self.subTest(delay_ms=delay_ms):
                self.assertEqual(
                    _legacy_sequence(delay_ms, 40),
                    _event_sequence(delay_ms, 40),
                )

    def test_startup_and_exact_ceiling_boundaries_are_preserved(self):
        self.assertEqual([None, None, 0, 1], _event_sequence(100, 4))
        self.assertEqual([None, None, 0, 1], _legacy_sequence(100, 4))
        self.assertEqual([0, 1, 2], _event_sequence(0, 3))


if __name__ == "__main__":
    unittest.main()
