from __future__ import print_function

from dataclasses import FrozenInstanceError
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ditl_event_scheduler import (  # noqa: E402
    DeliveryOutcome,
    DitlEventScheduler,
    DitlReplayEngine,
    LogicalMessage,
    ReplayCounters,
    ReplaySubmission,
    ReplayTick,
    StreamSelection,
)
from ditl_impairment_policy import (  # noqa: E402
    DeliveryPlan, DeterministicImpairmentPolicy, PolicyDecision,
)
from ditl_replay_config import BurstConfig, ImpairmentConfig  # noqa: E402
from ditl_replay_schema import ReplaySample  # noqa: E402
from ditl_trace_library import ReplaySampleCursor, SampleSelection  # noqa: E402


def sample(ordinal, delay_ns):
    return ReplaySample(
        sample_id="sample-%d" % ordinal,
        ordinal=ordinal,
        source_trace_id="capture-%d" % (9000 + ordinal),
        source_sequence_id=9000 + ordinal,
        base_delay_ns=delay_ns,
        stage_ns={
            "acquisition_preprocess": delay_ns,
            "gpu_inference": 0,
            "packing_postprocess": 0,
            "transport_receive": 0,
        },
        raw_payload_bytes=128,
        wire_payload_bytes=128,
        capture_condition_id="condition-1",
    )


def policy(**overrides):
    values = {
        "extra_delay_ns": 0,
        "jitter_ns": 0,
        "drop_rate": 0.0,
        "burst": BurstConfig(False, 0.0, 1.0),
        "reorder_rate": 0.0,
        "reorder_extra_ns": 0,
        "duplicate_rate": 0.0,
    }
    values.update(overrides)
    return DeterministicImpairmentPolicy(20260923, ImpairmentConfig(**values))


def make_engine(delays=(10,), exhaustion="cycle", ttl_ns=None,
                deadline_ns=None, streams=("rsu-1",), max_queue_events=32,
                **policy_overrides):
    samples = tuple(sample(index, delay) for index, delay in enumerate(delays))
    cursor = ReplaySampleCursor(samples, exhaustion)
    scheduler = DitlEventScheduler(
        max_queue_events=max_queue_events, ttl_ns=ttl_ns
    )
    return DitlReplayEngine(
        sample_cursor=cursor,
        policy=policy(**policy_overrides),
        scheduler=scheduler,
        stream_ids=streams,
        ttl_ns=ttl_ns,
        deadline_ns=deadline_ns,
    )


def submission(sequence=0, source_ns=0, stream="rsu-1", payload=None,
               route="route-1"):
    if payload is None:
        payload = {"sequence": sequence}
    return ReplaySubmission(
        route_id=route,
        stream_id=stream,
        source_id="source-%s" % stream,
        source_frame=sequence,
        source_sim_time_ns=source_ns,
        sequence_id=sequence,
        payload=payload,
    )


class ReplaySampleCursorTest(unittest.TestCase):
    def test_error_exhaustion_raises_before_any_sample_is_reused(self):
        first, second = sample(0, 11), sample(1, 12)
        cursor = ReplaySampleCursor((first, second), "error")
        self.assertEqual(SampleSelection(first, 0), cursor.next())
        self.assertEqual(SampleSelection(second, 0), cursor.next())
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            cursor.next()
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            cursor.next()

    def test_cycle_preserves_exact_order_and_increments_cycle_on_wrap(self):
        first, second = sample(0, 11), sample(1, 12)
        cursor = ReplaySampleCursor((first, second), "cycle")
        observed = [cursor.next() for _ in range(5)]
        self.assertEqual(
            [
                SampleSelection(first, 0),
                SampleSelection(second, 0),
                SampleSelection(first, 1),
                SampleSelection(second, 1),
                SampleSelection(first, 2),
            ],
            observed,
        )

    def test_constructor_rejects_mutable_empty_or_invalid_inputs(self):
        for samples in ([], (), (object(),)):
            with self.subTest(samples=samples):
                with self.assertRaises((TypeError, ValueError)):
                    ReplaySampleCursor(samples, "error")
        with self.assertRaisesRegex(ValueError, "exhaustion"):
            ReplaySampleCursor((sample(0, 1),), "reuse")


class DitlReplayEngineFreshnessTest(unittest.TestCase):
    def test_feature_delay_drives_ready_time_not_capture_identity(self):
        engine = make_engine(delays=(17,), exhaustion="error")
        outcome = engine.submit(submission(sequence=3, source_ns=100))
        self.assertEqual("sample-0", outcome.message.trace_sample_id)
        self.assertEqual(17, outcome.message.base_delay_ns)
        self.assertEqual(117, outcome.enqueued[0].ready_time_ns)
        self.assertEqual((), engine.advance(3, 116).delivery_outcomes)
        tick = engine.advance(4, 117)
        self.assertEqual(3, tick.streams["rsu-1"].used_sequence_id)
        self.assertEqual(17, tick.streams["rsu-1"].aoi_ns)

    def test_startup_is_ego_only_for_every_configured_stream(self):
        tick = make_engine(streams=("rsu-1", "rsu-2")).advance(0, 0)
        self.assertEqual(("rsu-1", "rsu-2"), tuple(tick.streams))
        for selected in tick.streams.values():
            self.assertIsNone(selected.payload)
            self.assertIsNone(selected.materialize_payload())
            self.assertTrue(selected.is_startup_ego_only)
            self.assertFalse(selected.is_expired)
            self.assertFalse(selected.is_zoh)

    def test_ttl_boundary_is_inclusive_then_expires_one_nanosecond_later(self):
        engine = make_engine(delays=(10,), ttl_ns=100)
        _, accepted_tick = engine.submit_and_advance(
            submission(source_ns=0), consume_frame=1, consume_sim_time_ns=10
        )
        accepted = accepted_tick.streams["rsu-1"]
        self.assertEqual("message-0", accepted.used_message_id)
        self.assertFalse(accepted.is_zoh)

        before = engine.advance(2, 99).streams["rsu-1"]
        self.assertEqual(99, before.aoi_ns)
        self.assertTrue(before.is_zoh)
        boundary = engine.advance(3, 100).streams["rsu-1"]
        self.assertEqual("message-0", boundary.used_message_id)
        self.assertEqual(100, boundary.aoi_ns)

        expired_tick = engine.advance(4, 101)
        expired = expired_tick.streams["rsu-1"]
        self.assertIsNone(expired.used_message_id)
        self.assertTrue(expired.is_expired)
        self.assertFalse(expired.is_startup_ego_only)
        self.assertEqual(ReplayCounters(1, 0, 1), expired_tick.counters)

    def test_arrival_after_ttl_is_recorded_and_never_enters_fusion(self):
        engine = make_engine(delays=(101,), ttl_ns=100)
        engine.submit(submission())
        tick = engine.advance(1, 101)
        selected = tick.streams["rsu-1"]
        self.assertIsNone(selected.payload)
        self.assertTrue(selected.is_expired)
        self.assertFalse(selected.is_startup_ego_only)
        self.assertEqual(["expired"], [item.status for item in tick.delivery_outcomes])
        self.assertEqual(ReplayCounters(0, 0, 1), tick.counters)

    def test_expired_newer_arrival_does_not_hide_valid_held_state(self):
        engine = make_engine(delays=(0, 60), ttl_ns=100)
        engine.submit(submission(sequence=1, source_ns=20))
        first = engine.advance(1, 20).streams["rsu-1"]
        self.assertEqual(1, first.used_sequence_id)
        # Model a per-message expiry supplied by a future scheduler policy.
        # The engine must judge each event's frozen expiry independently.
        engine._scheduler._ttl_ns = 20
        engine.submit(submission(sequence=2, source_ns=50))
        tick = engine.advance(2, 110)
        self.assertEqual(["expired"], [item.status for item in tick.delivery_outcomes])
        held = tick.streams["rsu-1"]
        self.assertEqual(1, held.used_sequence_id)
        self.assertTrue(held.is_zoh)
        self.assertFalse(held.is_expired)
        self.assertEqual(90, held.aoi_ns)

    def test_late_old_event_cannot_roll_back_or_revive_newer_state(self):
        engine = make_engine(delays=(200, 10), ttl_ns=500)
        engine.submit(submission(sequence=1, source_ns=0))
        engine.submit(submission(sequence=2, source_ns=10))
        self.assertEqual(2, engine.advance(2, 20).streams["rsu-1"].used_sequence_id)
        result = engine.advance(3, 200)
        self.assertEqual(2, result.streams["rsu-1"].used_sequence_id)
        self.assertIn(
            "late_superseded",
            [outcome.status for outcome in result.delivery_outcomes],
        )

    def test_old_delivery_cannot_revive_state_after_newer_state_expires(self):
        engine = make_engine(delays=(200, 0), ttl_ns=100)
        engine.submit(submission(sequence=1, source_ns=0))
        engine.submit(submission(sequence=2, source_ns=10))
        self.assertEqual(2, engine.advance(2, 10).streams["rsu-1"].used_sequence_id)
        self.assertTrue(engine.advance(3, 111).streams["rsu-1"].is_expired)
        result = engine.advance(4, 200)
        self.assertIsNone(result.streams["rsu-1"].used_message_id)
        self.assertTrue(result.streams["rsu-1"].is_expired)
        self.assertEqual("expired", result.delivery_outcomes[0].status)

    def test_newest_valid_candidate_wins_independent_of_heap_release_order(self):
        engine = make_engine(delays=(20, 10))
        engine.submit(submission(sequence=1, source_ns=0))
        engine.submit(submission(sequence=2, source_ns=10))
        tick = engine.advance(2, 20)
        self.assertEqual(2, tick.streams["rsu-1"].used_sequence_id)
        statuses = {
            item.event.message.sequence_id: item.status
            for item in tick.delivery_outcomes
        }
        self.assertEqual({1: "late_superseded", 2: "accepted"}, statuses)


class DitlReplayEngineDuplicateAndStreamTest(unittest.TestCase):
    def test_duplicate_never_refreshes_state_ttl_or_aoi(self):
        engine = make_engine(
            delays=(10,), ttl_ns=100, duplicate_rate=1.0,
            reorder_rate=0.5, reorder_extra_ns=20,
        )
        engine.submit(submission(source_ns=0))
        accepted = engine.advance(1, 10)
        self.assertEqual("accepted", accepted.delivery_outcomes[0].status)
        duplicate = engine.advance(2, 30)
        self.assertEqual("duplicate_discarded", duplicate.delivery_outcomes[0].status)
        selected = duplicate.streams["rsu-1"]
        self.assertEqual(30, selected.aoi_ns)
        self.assertTrue(selected.is_zoh)
        self.assertEqual(20, selected.zoh_hold_ns)
        expired = engine.advance(3, 101).streams["rsu-1"]
        self.assertTrue(expired.is_expired)
        self.assertIsNone(expired.used_message_id)

    def test_expired_counter_counts_primary_message_once_across_copy_ticks(self):
        engine = make_engine(delays=(0,), ttl_ns=10)
        message = LogicalMessage(
            logical_message_id="logical-1", stream_id="rsu-1",
            source_id="source-rsu-1", source_frame=1,
            source_sim_time_ns=0, sequence_id=1,
            payload={"sequence": 1}, trace_sample_id="sample-0",
            base_delay_ns=0, cycle_index=0,
        )
        decision = PolicyDecision(
            dropped=False, drop_reason=None, delay_clamped=False,
            loss_uniform=0.0, duplicate_uniform=0.0,
            burst_state="disabled",
            deliveries=(DeliveryPlan(0, 0, 0), DeliveryPlan(1, 0, 20)),
        )
        engine._scheduler.submit(message, decision)

        accepted = engine.advance(1, 0)
        self.assertEqual("accepted", accepted.delivery_outcomes[0].status)
        self.assertEqual(ReplayCounters(1, 0, 0), accepted.counters)
        boundary = engine.advance(2, 10)
        self.assertEqual("logical-1", boundary.streams["rsu-1"].used_message_id)
        expired = engine.advance(3, 11)
        self.assertTrue(expired.streams["rsu-1"].is_expired)
        self.assertEqual(ReplayCounters(1, 0, 1), expired.counters)

        late_copy = engine.advance(4, 20)
        self.assertEqual("expired", late_copy.delivery_outcomes[0].status)
        self.assertEqual(1, late_copy.delivery_outcomes[0].event.copy_index)
        self.assertEqual(ReplayCounters(1, 0, 1), late_copy.counters)

    def test_secondary_copy_accepted_first_still_expires_exactly_once(self):
        engine = make_engine(delays=(0,), ttl_ns=100)
        message = LogicalMessage(
            logical_message_id="message-1", stream_id="rsu-1",
            source_id="source-rsu-1", source_frame=1,
            source_sim_time_ns=0, sequence_id=1,
            payload={"sequence": 1}, trace_sample_id="sample-0",
            base_delay_ns=0, cycle_index=0,
        )
        decision = PolicyDecision(
            dropped=False, drop_reason=None, delay_clamped=False,
            loss_uniform=0.0, duplicate_uniform=0.0,
            burst_state="disabled",
            deliveries=(DeliveryPlan(0, 0, 200), DeliveryPlan(1, 0, 0)),
        )
        engine._scheduler.submit(message, decision)

        accepted = engine.advance(1, 0)
        self.assertEqual(1, accepted.delivery_outcomes[0].event.copy_index)
        self.assertEqual("accepted", accepted.delivery_outcomes[0].status)
        self.assertEqual(ReplayCounters(1, 0, 0), accepted.counters)

        expired_hold = engine.advance(2, 101)
        self.assertTrue(expired_hold.streams["rsu-1"].is_expired)
        self.assertEqual(ReplayCounters(1, 0, 1), expired_hold.counters)

        late_primary = engine.advance(3, 200)
        self.assertEqual(0, late_primary.delivery_outcomes[0].event.copy_index)
        self.assertEqual("expired", late_primary.delivery_outcomes[0].status)
        self.assertEqual(ReplayCounters(1, 0, 1), late_primary.counters)

    def test_stream_state_is_isolated_and_all_streams_are_returned(self):
        engine = make_engine(delays=(10, 20), streams=("rsu-a", "rsu-b"))
        engine.submit(submission(sequence=1, stream="rsu-a"))
        engine.submit(submission(sequence=1, stream="rsu-b"))
        first = engine.advance(1, 10)
        self.assertEqual(1, first.streams["rsu-a"].used_sequence_id)
        self.assertTrue(first.streams["rsu-b"].is_startup_ego_only)
        second = engine.advance(2, 20)
        self.assertTrue(second.streams["rsu-a"].is_zoh)
        self.assertEqual(1, second.streams["rsu-b"].used_sequence_id)

    def test_unknown_submission_stream_is_rejected_before_sample_consumption(self):
        engine = make_engine(delays=(10,), exhaustion="error")
        with self.assertRaisesRegex(ValueError, "configured"):
            engine.submit(submission(stream="rsu-unknown"))
        outcome = engine.submit(submission(stream="rsu-1"))
        self.assertEqual("sample-0", outcome.message.trace_sample_id)

    def test_dropped_messages_increment_only_drop_counter(self):
        engine = make_engine(delays=(10,), drop_rate=1.0)
        outcome, tick = engine.submit_and_advance(submission(), 0, 0)
        self.assertTrue(outcome.policy_decision.dropped)
        self.assertEqual((), tick.delivery_outcomes)
        self.assertTrue(tick.streams["rsu-1"].is_startup_ego_only)
        self.assertEqual(ReplayCounters(0, 1, 0), tick.counters)


class DitlReplayEngineSubmissionAtomicityTest(unittest.TestCase):
    @staticmethod
    def make_atomic_engine(max_queue_events=32):
        return make_engine(
            delays=(10, 20),
            max_queue_events=max_queue_events,
            burst=BurstConfig(True, 0.0, 1.0),
        )

    def assert_outcome_matches_control(self, actual, control):
        self.assertEqual(control.message.trace_sample_id,
                         actual.message.trace_sample_id)
        self.assertEqual(control.message.base_delay_ns,
                         actual.message.base_delay_ns)
        self.assertEqual(control.message.cycle_index,
                         actual.message.cycle_index)
        self.assertEqual(control.policy_decision, actual.policy_decision)

    def test_nonincreasing_sequence_rejections_do_not_advance_inputs(self):
        actual = self.make_atomic_engine()
        control = self.make_atomic_engine()
        actual.submit(submission(sequence=1, source_ns=100))
        control.submit(submission(sequence=1, source_ns=100))

        for rejected_sequence in (1, 0):
            with self.subTest(sequence=rejected_sequence):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    actual.submit(submission(
                        sequence=rejected_sequence, source_ns=100
                    ))

        retried = actual.submit(submission(sequence=2, source_ns=101))
        expected = control.submit(submission(sequence=2, source_ns=101))
        self.assert_outcome_matches_control(retried, expected)

    def test_decreasing_source_time_rejection_is_fully_retryable(self):
        actual = self.make_atomic_engine()
        control = self.make_atomic_engine()
        actual.submit(submission(sequence=1, source_ns=100))
        control.submit(submission(sequence=1, source_ns=100))

        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            actual.submit(submission(sequence=2, source_ns=99))

        retried = actual.submit(submission(sequence=2, source_ns=101))
        expected = control.submit(submission(sequence=2, source_ns=101))
        self.assert_outcome_matches_control(retried, expected)

    def test_queue_overflow_then_drain_and_retry_preserves_all_inputs(self):
        actual = self.make_atomic_engine(max_queue_events=1)
        control = self.make_atomic_engine(max_queue_events=1)
        actual.submit(submission(sequence=1, source_ns=0))
        control.submit(submission(sequence=1, source_ns=0))

        with self.assertRaisesRegex(OverflowError, "capacity"):
            actual.submit(submission(sequence=2, source_ns=1))

        actual.advance(1, 10)
        control.advance(1, 10)
        retried = actual.submit(submission(sequence=2, source_ns=11))
        expected = control.submit(submission(sequence=2, source_ns=11))
        self.assert_outcome_matches_control(retried, expected)


class DitlReplayEngineDeadlineAndImmutabilityTest(unittest.TestCase):
    def test_advance_refreezes_system_generated_nested_mappings(self):
        payload = {"metadata": {"source": "rsu-1", "shape": [2, 4]}}
        engine = make_engine(delays=(0,))

        engine.submit(submission(payload=payload))
        tick = engine.advance(0, 0)

        selected = tick.streams["rsu-1"]
        self.assertEqual("rsu-1", selected.payload["metadata"]["source"])
        self.assertEqual((2, 4), selected.payload["metadata"]["shape"])
        with self.assertRaises(TypeError):
            selected.payload["metadata"]["source"] = "tampered"

        held = engine.advance(1, 1).streams["rsu-1"]
        self.assertTrue(held.is_zoh)
        self.assertEqual("rsu-1", held.payload["metadata"]["source"])
        self.assertEqual((2, 4), held.payload["metadata"]["shape"])

    def test_deadline_equal_is_on_time_then_misses_one_nanosecond_later(self):
        engine = make_engine(delays=(10,), deadline_ns=10)
        engine.submit(submission())
        boundary = engine.advance(1, 10)
        self.assertFalse(boundary.delivery_outcomes[0].arrival_deadline_miss)
        self.assertFalse(boundary.streams["rsu-1"].fusion_deadline_miss)
        later = engine.advance(2, 11)
        self.assertTrue(later.streams["rsu-1"].fusion_deadline_miss)
        self.assertEqual("message-0", later.streams["rsu-1"].used_message_id)

    def test_none_deadline_never_marks_miss(self):
        engine = make_engine(delays=(1000,), deadline_ns=None)
        engine.submit(submission())
        tick = engine.advance(1, 1000)
        self.assertFalse(tick.delivery_outcomes[0].arrival_deadline_miss)
        self.assertFalse(tick.streams["rsu-1"].fusion_deadline_miss)

    def test_models_mappings_and_payload_state_are_immutable(self):
        payload = {"nested": [1, 2], "array": np.arange(2)}
        engine = make_engine(delays=(0,))
        _, tick = engine.submit_and_advance(submission(payload=payload), 0, 0)
        payload["nested"].append(3)
        payload["array"][0] = 99
        selected = tick.streams["rsu-1"]
        self.assertEqual((1, 2), selected.payload["nested"])
        self.assertEqual(0, selected.payload["array"][0])
        self.assertFalse(selected.payload["array"].flags.writeable)
        with self.assertRaises(TypeError):
            tick.streams["new"] = selected
        with self.assertRaises(TypeError):
            selected.payload["nested"] = ()
        for model, field, value in (
                (selected, "is_zoh", True),
                (tick, "queue_depth", 9),
                (tick.counters, "accepted_count", 9),
                (tick.delivery_outcomes[0], "status", "tampered")):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(model, field, value)

    def test_cycle_boundary_keeps_sample_order_and_policy_determinism(self):
        first = make_engine(delays=(7, 13), exhaustion="cycle")
        second = make_engine(delays=(7, 13), exhaustion="cycle")
        first_outcomes = [first.submit(submission(sequence=index)) for index in range(5)]
        second_outcomes = [second.submit(submission(sequence=index)) for index in range(5)]
        self.assertEqual(
            [0, 0, 1, 1, 2],
            [item.message.cycle_index for item in first_outcomes],
        )
        self.assertEqual(
            [7, 13, 7, 13, 7],
            [item.message.base_delay_ns for item in first_outcomes],
        )
        self.assertEqual(
            [item.policy_decision for item in first_outcomes],
            [item.policy_decision for item in second_outcomes],
        )

    def test_public_contract_types_are_frozen(self):
        engine = make_engine(delays=(0,))
        outcome, tick = engine.submit_and_advance(submission(), 0, 0)
        models = (
            SampleSelection(sample(0, 0), 0),
            submission(),
            DeliveryOutcome(outcome.enqueued[0], "accepted", False),
            StreamSelection(
                "rsu-1", None, None, None, None, None, None,
                None, None, False, True, False, False,
            ),
            ReplayCounters(0, 0, 0),
            ReplayTick(0, 0, tick.streams, (), 0, ReplayCounters(0, 0, 0)),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(model, "unexpected", True)

if __name__ == "__main__":
    unittest.main()
