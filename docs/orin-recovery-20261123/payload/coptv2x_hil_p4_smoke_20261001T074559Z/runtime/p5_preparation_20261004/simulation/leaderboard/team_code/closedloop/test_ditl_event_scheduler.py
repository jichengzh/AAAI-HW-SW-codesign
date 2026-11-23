from __future__ import print_function

from collections import UserDict
from dataclasses import FrozenInstanceError
import os
import sys
from types import MappingProxyType
import unittest

import numpy as np
from PIL import Image


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ditl_event_scheduler import (  # noqa: E402
    AdvanceResult,
    DeliveryEvent,
    DitlEventScheduler,
    LogicalMessage,
    SubmissionOutcome,
)
from ditl_impairment_policy import DeliveryPlan, PolicyDecision  # noqa: E402


def message(sequence=1, source_ns=0, stream="rsu-1", payload=None,
            logical_id=None):
    if payload is None:
        payload = {"sequence": sequence}
    return LogicalMessage(
        logical_message_id=(logical_id or "message-%d" % sequence),
        stream_id=stream,
        source_id="source-%s" % stream,
        source_frame=sequence,
        source_sim_time_ns=source_ns,
        sequence_id=sequence,
        payload=payload,
        trace_sample_id="sample-%d" % sequence,
        base_delay_ns=17,
        cycle_index=sequence,
    )


def decision(delay_ns=100, reorder_ns=0, copies=1, dropped=False,
             drop_reason=None):
    plans = () if dropped else tuple(
        DeliveryPlan(index, delay_ns, reorder_ns)
        for index in range(copies)
    )
    return PolicyDecision(
        dropped=dropped,
        drop_reason=drop_reason,
        delay_clamped=False,
        loss_uniform=0.25,
        duplicate_uniform=0.75,
        burst_state="disabled",
        deliveries=plans,
    )


class DitlEventSchedulerTimingTest(unittest.TestCase):
    def test_public_models_are_frozen(self):
        logical = message()
        delivery = DeliveryEvent("delivery", logical, 0, 10, 10, None, 0)
        outcome = SubmissionOutcome(logical, decision(delay_ns=10), (delivery,))
        advanced = AdvanceResult(10, (delivery,), 0)
        for model, name, value in (
                (logical, "sequence_id", 2),
                (delivery, "ready_time_ns", 11),
                (outcome, "enqueued", ()),
                (advanced, "queue_depth", 1)):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(model, name, value)

    def test_ready_boundary_and_equal_time_order_are_exact(self):
        scheduler = DitlEventScheduler(max_queue_events=4, ttl_ns=None)
        scheduler.submit(message(sequence=1, source_ns=0),
                         decision(delay_ns=100))
        scheduler.submit(message(sequence=2, source_ns=0),
                         decision(delay_ns=100))
        self.assertEqual((), scheduler.advance(99).ready)
        result = scheduler.advance(100)
        self.assertEqual(
            [1, 2], [event.message.sequence_id for event in result.ready]
        )
        self.assertEqual(0, result.queue_depth)

    def test_later_message_can_overtake_and_reorder_is_counted_once(self):
        scheduler = DitlEventScheduler(max_queue_events=4, ttl_ns=500)
        first = scheduler.submit(
            message(sequence=1, source_ns=0),
            decision(delay_ns=100, reorder_ns=20),
        )
        second = scheduler.submit(
            message(sequence=2, source_ns=10),
            decision(delay_ns=5, reorder_ns=0),
        )
        self.assertEqual(120, first.enqueued[0].ready_time_ns)
        self.assertEqual(15, second.enqueued[0].ready_time_ns)
        self.assertEqual(500, first.enqueued[0].expiry_time_ns)
        self.assertEqual(510, second.enqueued[0].expiry_time_ns)
        self.assertEqual(
            [2],
            [item.message.sequence_id for item in scheduler.advance(15).ready],
        )
        self.assertEqual(
            [1],
            [item.message.sequence_id for item in scheduler.advance(120).ready],
        )

    def test_equal_keys_use_insertion_order_across_streams(self):
        scheduler = DitlEventScheduler(4, None)
        scheduler.submit(
            message(1, 0, stream="rsu-b", logical_id="message-1"), decision(10)
        )
        scheduler.submit(
            message(1, 0, stream="rsu-a", logical_id="message-1"), decision(10)
        )
        self.assertEqual(
            ["rsu-b", "rsu-a"],
            [item.message.stream_id for item in scheduler.advance(10).ready],
        )

    def test_duplicate_copy_order_uses_copy_index(self):
        scheduler = DitlEventScheduler(4, None)
        outcome = scheduler.submit(message(), decision(10, copies=2))
        self.assertEqual([0, 1], [item.copy_index for item in outcome.enqueued])
        self.assertEqual(
            [0, 1], [item.copy_index for item in scheduler.advance(10).ready]
        )
        self.assertNotEqual(
            outcome.enqueued[0].delivery_id,
            outcome.enqueued[1].delivery_id,
        )
        self.assertEqual(
            ["rsu-1:message-1:1:0", "rsu-1:message-1:1:1"],
            [item.delivery_id for item in outcome.enqueued],
        )


class DitlEventSchedulerBoundsTest(unittest.TestCase):
    def test_sequence_must_increase_per_stream_but_ids_may_repeat_cross_stream(self):
        scheduler = DitlEventScheduler(4, None)
        scheduler.submit(message(2, stream="a", logical_id="message-2"), decision())
        scheduler.submit(message(2, stream="b", logical_id="message-2"), decision())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            scheduler.submit(message(2, stream="a", logical_id="message-2"), decision())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            scheduler.submit(message(1, stream="a", logical_id="message-1"), decision())

    def test_same_stream_logical_id_reuse_fails_atomically(self):
        actual = DitlEventScheduler(4, None)
        control = DitlEventScheduler(4, None)
        for scheduler in (actual, control):
            scheduler.submit(
                message(1, stream="a", logical_id="message-1"), decision(10)
            )

        with self.assertRaisesRegex(ValueError, "logical_message_id"):
            actual.submit(
                message(2, stream="a", logical_id="message-1"), decision(10)
            )

        actual_retry = actual.submit(
            message(2, stream="a", logical_id="message-2"), decision(10)
        )
        control_retry = control.submit(
            message(2, stream="a", logical_id="message-2"), decision(10)
        )
        self.assertEqual(control_retry.enqueued, actual_retry.enqueued)
        self.assertEqual(control.advance(10), actual.advance(10))

    def test_nonconsecutive_logical_id_reuse_is_rejected_by_identity_contract(self):
        scheduler = DitlEventScheduler(4, None)
        scheduler.submit(message(1, stream="a", logical_id="message-1"),
                         decision(10))
        scheduler.submit(message(2, stream="a", logical_id="message-2"),
                         decision(10))

        with self.assertRaisesRegex(ValueError, "logical_message_id"):
            scheduler.submit(message(3, stream="a", logical_id="message-1"),
                             decision(10))

    def test_negative_and_decreasing_times_are_rejected_without_state_change(self):
        scheduler = DitlEventScheduler(4, None)
        with self.assertRaisesRegex(ValueError, "source_sim_time_ns"):
            scheduler.submit(message(1, -1), decision())
        scheduler.submit(message(1, 10), decision())
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            scheduler.submit(message(2, 9), decision())
        scheduler.submit(message(2, 10), decision())
        with self.assertRaisesRegex(ValueError, "consume_sim_time_ns"):
            scheduler.advance(-1)
        scheduler.advance(10)
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            scheduler.advance(9)

    def test_queue_overflow_is_atomic_for_all_copies_and_identity(self):
        scheduler = DitlEventScheduler(2, None)
        scheduler.submit(message(1), decision(100))
        with self.assertRaisesRegex(OverflowError, "capacity"):
            scheduler.submit(message(2), decision(100, copies=2))
        self.assertEqual(1, scheduler.advance(0).queue_depth)
        scheduler.advance(100)
        retried = scheduler.submit(message(2, source_ns=100),
                                   decision(1, copies=2))
        self.assertEqual(2, len(retried.enqueued))

    def test_dropped_outcome_keeps_decision_and_does_not_enqueue(self):
        scheduler = DitlEventScheduler(1, None)
        dropped = decision(dropped=True, drop_reason="bernoulli")
        outcome = scheduler.submit(message(), dropped)
        self.assertEqual(dropped, outcome.policy_decision)
        self.assertEqual("bernoulli", outcome.policy_decision.drop_reason)
        self.assertEqual((), outcome.enqueued)
        self.assertEqual(0, scheduler.advance(1000).queue_depth)

    def test_constructor_and_model_boundaries_are_validated(self):
        for maximum in (0, -1, True, 1.5):
            with self.subTest(maximum=maximum):
                with self.assertRaises((TypeError, ValueError)):
                    DitlEventScheduler(maximum, None)
        for ttl in (-1, True, 1.5):
            with self.subTest(ttl=ttl):
                with self.assertRaises((TypeError, ValueError)):
                    DitlEventScheduler(1, ttl)
        scheduler = DitlEventScheduler(1, None)
        with self.assertRaises(TypeError):
            scheduler.submit(object(), decision())
        with self.assertRaises(TypeError):
            scheduler.submit(message(), object())
        with self.assertRaisesRegex(ValueError, "copy_index"):
            invalid = PolicyDecision(
                False, None, False, 0.0, 0.0, "disabled",
                (DeliveryPlan(-1, 0, 0),),
            )
            scheduler.submit(message(), invalid)

    def test_policy_decision_requires_immutable_exact_tuple_deliveries(self):
        mutable_deliveries = [DeliveryPlan(0, 10, 0)]
        mutable_decision = PolicyDecision(
            False, None, False, 0.0, 0.0, "disabled",
            mutable_deliveries,
        )
        scheduler = DitlEventScheduler(1, None)
        with self.assertRaisesRegex(TypeError, "deliveries must be a tuple"):
            scheduler.submit(message(), mutable_decision)
        mutable_deliveries.clear()

        accepted = scheduler.submit(message(), decision(10))
        self.assertEqual(1, len(accepted.policy_decision.deliveries))

    def test_submission_snapshots_policy_decision_from_caller_mutation(self):
        caller_decision = decision(10)
        caller_plan = caller_decision.deliveries[0]
        scheduler = DitlEventScheduler(1, None)
        outcome = scheduler.submit(message(), caller_decision)

        object.__setattr__(caller_plan, "effective_delay_ns", 999)
        object.__setattr__(caller_decision, "deliveries", ())
        object.__setattr__(caller_decision, "drop_reason", "tampered")
        self.assertEqual(1, len(outcome.policy_decision.deliveries))
        self.assertEqual(
            10, outcome.policy_decision.deliveries[0].effective_delay_ns
        )
        self.assertIsNone(outcome.policy_decision.drop_reason)
        self.assertIsNot(caller_decision, outcome.policy_decision)
        self.assertIsNot(caller_plan, outcome.policy_decision.deliveries[0])


class DitlEventSchedulerSnapshotTest(unittest.TestCase):
    def test_snapshot_accepts_nested_exact_mappingproxy_values(self):
        inner_backing = {"sequence": 1}
        backing = {"inner": MappingProxyType(inner_backing)}
        nested = MappingProxyType(backing)
        caller_payload = {"metadata": nested}
        scheduler = DitlEventScheduler(1, None)

        outcome = scheduler.submit(
            message(payload=caller_payload), decision(delay_ns=10)
        )
        self.assertEqual(1, inner_backing["sequence"])
        inner_backing["sequence"] = 99

        ready_payload = scheduler.advance(10).ready[0].message.payload
        self.assertEqual(
            1, outcome.message.payload["metadata"]["inner"]["sequence"]
        )
        self.assertEqual(
            1, ready_payload["metadata"]["inner"]["sequence"]
        )
        self.assertEqual(99, inner_backing["sequence"])

    def test_submit_copies_nested_containers_arrays_and_images(self):
        lidar = np.zeros((32768, 4), dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        image = Image.new("RGB", (2, 2), color=(1, 2, 3))
        payload = {
            "car_data": [None, {"lidar": lidar}],
            "rsu_data": [{"transformation_matrix": transform,
                          "camera": image}],
            "shape": (2, 2),
        }
        scheduler = DitlEventScheduler(max_queue_events=2, ttl_ns=None)
        outcome = scheduler.submit(
            message(sequence=1, source_ns=0, payload=payload), decision(10)
        )

        payload["rsu_data"].clear()
        lidar[0, 0] = 88
        transform[0, 0] = 99
        image.putpixel((0, 0), (9, 9, 9))

        event = scheduler.advance(10).ready[0]
        stored = event.message.payload
        self.assertEqual(1, len(stored["rsu_data"]))
        self.assertEqual(0.0, stored["car_data"][1]["lidar"][0, 0])
        self.assertEqual(
            1.0, stored["rsu_data"][0]["transformation_matrix"][0, 0]
        )
        self.assertEqual(
            (1, 2, 3), stored["rsu_data"][0]["camera"].getpixel((0, 0))
        )
        self.assertIsInstance(stored["shape"], tuple)
        self.assertIsNot(outcome.message.payload, event.message.payload)

    def test_submit_outcome_and_returned_copies_cannot_mutate_heap_or_each_other(self):
        array = np.array([1, 2], dtype=np.int64)
        image = Image.new("L", (1, 1), color=7)
        scheduler = DitlEventScheduler(2, None)
        outcome = scheduler.submit(
            message(payload={"items": [{"array": array, "image": image}]}),
            decision(10, copies=2),
        )

        outcome.message.payload["items"].clear()
        outcome.enqueued[0].message.payload["items"][0]["array"][0] = 9
        outcome.enqueued[0].message.payload["items"][0]["image"].putpixel(
            (0, 0), 9
        )
        ready = scheduler.advance(10).ready
        self.assertEqual(2, len(ready))
        self.assertEqual(1, ready[0].message.payload["items"][0]["array"][0])
        self.assertEqual(7, ready[0].message.payload["items"][0]["image"].getpixel((0, 0)))

        ready[0].message.payload["items"][0]["array"][0] = 5
        ready[0].message.payload["items"][0]["image"].putpixel((0, 0), 5)
        self.assertEqual(1, ready[1].message.payload["items"][0]["array"][0])
        self.assertEqual(7, ready[1].message.payload["items"][0]["image"].getpixel((0, 0)))

    def test_cycles_and_unsupported_mutable_leaves_are_rejected_atomically(self):
        cyclic = []
        cyclic.append(cyclic)
        proxy_backing = {}
        cyclic_proxy = MappingProxyType(proxy_backing)
        proxy_backing["self"] = cyclic_proxy
        scheduler = DitlEventScheduler(2, None)
        with self.assertRaisesRegex(ValueError, "cycle"):
            scheduler.submit(message(1, payload={"cycle": cyclic}), decision())
        with self.assertRaisesRegex(ValueError, "cycle"):
            scheduler.submit(
                message(1, payload={"cycle": cyclic_proxy}), decision()
            )
        with self.assertRaisesRegex(TypeError, "unsupported payload"):
            scheduler.submit(message(1, payload={"set": set([1])}), decision())
        with self.assertRaisesRegex(TypeError, "unsupported payload"):
            scheduler.submit(
                message(1, payload={"mapping": UserDict({"value": 1})}),
                decision(),
            )
        accepted = scheduler.submit(message(1), decision())
        self.assertEqual(1, len(accepted.enqueued))

    def test_snapshot_does_not_invoke_arbitrary_copy_methods(self):
        class Hostile(object):
            def copy(self):
                raise AssertionError("must not call arbitrary copy")

        scheduler = DitlEventScheduler(1, None)
        with self.assertRaisesRegex(TypeError, "unsupported payload"):
            scheduler.submit(message(payload={"hostile": Hostile()}), decision())

    def test_object_dtype_array_is_rejected_instead_of_shallow_copied(self):
        mutable = [{"value": 1}]
        array = np.array(mutable, dtype=object)
        scheduler = DitlEventScheduler(1, None)
        with self.assertRaisesRegex(TypeError, "object dtype"):
            scheduler.submit(message(payload={"array": array}), decision())

    def test_pillow_subclass_is_rejected_without_invoking_its_methods(self):
        calls = []

        class HostileImage(Image.Image):
            def load(self):
                calls.append("load")
                raise AssertionError("subclass method must not run")

            def copy(self):
                calls.append("copy")
                raise AssertionError("subclass method must not run")

        scheduler = DitlEventScheduler(1, None)
        with self.assertRaisesRegex(TypeError, "Pillow Image subclass"):
            scheduler.submit(
                message(payload={"image": HostileImage()}), decision()
            )
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
