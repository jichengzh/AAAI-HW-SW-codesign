"""Event-driven bounded-credit and stage ownership checks (no GPU required)."""

from concurrent.futures import TimeoutError
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))


class BoundedPipelineTests(unittest.TestCase):
    def pipeline(self, **stages):
        # Missing executor is an explicit feature failure during the RED run.
        try:
            from coptv2x_hil_pipeline_executor import BoundedHilPipeline
        except ImportError:
            self.fail("BoundedHilPipeline has not been implemented")
        defaults = {name: lambda item: item for name in ("prepare", "infer", "finish", "deliver")}
        defaults.update(stages)
        return BoundedHilPipeline(**defaults)

    def test_finish_overlaps_next_infer_and_single_infer_preserves_fifo(self):
        finish_entered, next_infer, release = (threading.Event() for _ in range(3))
        infer_lock = threading.Lock()
        delivered = []

        def infer(item):
            self.assertTrue(infer_lock.acquire(False), "concurrent GPU context use")
            try:
                if item == 1:
                    self.assertTrue(finish_entered.wait(2))
                    next_infer.set()
                return item
            finally:
                infer_lock.release()

        def finish(item):
            if item == 0:
                finish_entered.set()
                self.assertTrue(release.wait(2))
            return item * 10

        pipeline = self.pipeline(infer=infer, finish=finish, deliver=delivered.append)
        try:
            first, second = pipeline.submit(0), pipeline.submit(1)
            self.assertTrue(next_infer.wait(2), "GPU waited for prior CPU finish")
            self.assertFalse(first.done())
            release.set()
            self.assertEqual([0, 10], [first.result(2), second.result(2)])
            pipeline.drain(timeout=2)
            self.assertEqual([0, 10], delivered)
            self.assertEqual(2, pipeline.snapshot()["inflight_peak"])
            self.assertEqual(0, pipeline.snapshot()["inflight"])
            times = first.pipeline_timestamps_ns
            self.assertLessEqual(times["submitted"], times["admitted"])
            self.assertLessEqual(times["deliver_start"], times["deliver_end"])
            self.assertEqual(times["prepare_end"] - times["prepare_start"],
                             first.pipeline_durations_ns["source_read"])
            self.assertEqual(times["finish_end"] - times["prepare_start"],
                             first.pipeline_durations_ns["source_to_ready"])
            with self.assertRaises(TypeError):
                times["admitted"] = 0
        finally:
            release.set()
            pipeline.close(timeout=2)

    def test_slow_delivery_keeps_credit_and_gpu_finish_can_advance(self):
        delivering, second_finished, release = (threading.Event() for _ in range(3))
        prepared, delivered = [], []

        def finish(item):
            if item == 1:
                second_finished.set()
            return item

        def deliver(item):
            if item == 0:
                delivering.set()
                self.assertTrue(release.wait(2))
            delivered.append(item)

        pipeline = self.pipeline(prepare=lambda item: (prepared.append(item), item)[1],
                                 finish=finish, deliver=deliver)
        try:
            futures = [pipeline.submit(item) for item in (0, 1)]
            self.assertTrue(delivering.wait(2))
            self.assertTrue(second_finished.wait(2), "sender blocked CPU finish")
            with self.assertRaises(TimeoutError):
                pipeline.submit(2, timeout=0)
            with self.assertRaises(TimeoutError):
                pipeline.drain(timeout=0)
            self.assertEqual([0, 1], prepared)
            self.assertFalse(futures[0].done(), "ACCEPTED became fake completion")
            release.set()
            self.assertEqual([0, 1], [future.result(2) for future in futures])
            third = pipeline.submit(2)
            self.assertEqual(2, third.result(2))
            self.assertEqual([0, 1, 2], delivered)
            self.assertEqual(3, pipeline.snapshot()["completed"])
        finally:
            release.set()
            pipeline.close(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pipeline.submit(3)

    def test_each_stage_failure_fails_all_futures_and_wakes_waiting_submit(self):
        for failed_stage in ("prepare", "infer", "finish", "deliver"):
            with self.subTest(stage=failed_stage):
                entered, release, submitted = (threading.Event() for _ in range(3))
                failure = ValueError("broken " + failed_stage)

                def fail(item):
                    entered.set()
                    self.assertTrue(release.wait(2))
                    raise failure

                pipeline = self.pipeline(**{failed_stage: fail})
                futures = [pipeline.submit(item) for item in (0, 1)]
                self.assertTrue(entered.wait(2))
                waiting_errors = []

                def submit():
                    submitted.set()
                    try:
                        pipeline.submit(2)
                    except Exception as error:
                        waiting_errors.append(error)

                producer = threading.Thread(target=submit)
                producer.start()
                self.assertTrue(submitted.wait(2))
                release.set()
                producer.join(2)
                self.assertFalse(producer.is_alive())
                for future in futures:
                    with self.assertRaisesRegex(ValueError, "broken"):
                        future.result(2)
                self.assertEqual([failure], waiting_errors)
                with self.assertRaisesRegex(ValueError, "broken"):
                    pipeline.drain(2)
                with self.assertRaisesRegex(ValueError, "broken"):
                    pipeline.close(timeout=2)
                self.assertEqual(0, pipeline.snapshot()["inflight"])
                self.assertEqual(2, pipeline.snapshot()["failed"])
                self.assertTrue(pipeline.snapshot()["errors"])

    def test_close_releases_blocked_admission_but_drains_accepted_work(self):
        entered, release = threading.Event(), threading.Event()

        def infer(item):
            entered.set()
            self.assertTrue(release.wait(2))
            return item

        pipeline = self.pipeline(infer=infer, capacity=1)
        first = pipeline.submit(1)
        self.assertTrue(entered.wait(2))
        self.assertFalse(first.cancel(), "accepted work must never silently disappear")
        errors = []

        def submit():
            try:
                pipeline.submit(2)
            except Exception as error:
                errors.append(error)

        producer = threading.Thread(target=submit)
        producer.start()
        pipeline.close(wait=False)
        producer.join(2)
        self.assertFalse(producer.is_alive())
        self.assertIsInstance(errors[0], RuntimeError)
        release.set()
        pipeline.close(timeout=2)
        self.assertEqual(1, first.result(2))
        self.assertEqual(1, pipeline.snapshot()["submitted"])

    def test_invalid_capacity_and_timeout(self):
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.pipeline(capacity=capacity)
        with self.assertRaisesRegex(ValueError, "callable"):
            self.pipeline(infer=None)
        pipeline = self.pipeline()
        try:
            with self.assertRaises(ValueError):
                pipeline.submit(1, timeout=-1)
        finally:
            pipeline.close(timeout=2)

    def test_close_timeout_keeps_resources_live_until_running_callback_returns(self):
        entered, release = threading.Event(), threading.Event()

        def infer(item):
            entered.set()
            self.assertTrue(release.wait(2))
            return item

        pipeline = self.pipeline(infer=infer)
        future = pipeline.submit(7)
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaises(TimeoutError):
                pipeline.close(timeout=0)
            self.assertEqual(1, pipeline.snapshot()["inflight"])
            self.assertFalse(future.done())
        finally:
            release.set()
            pipeline.close(timeout=2)
        self.assertEqual(7, future.result(2))

    def test_credit_wait_is_separate_from_prepare_service_and_full_timeline_is_frozen(self):
        delivering, release, producer_entered = (threading.Event() for _ in range(3))

        def deliver(item):
            if item == 0:
                delivering.set()
                self.assertTrue(release.wait(2))

        pipeline = self.pipeline(deliver=deliver, capacity=1)
        first = pipeline.submit(0)
        self.assertTrue(delivering.wait(2))
        accepted = []

        def submit():
            producer_entered.set()
            accepted.append(pipeline.submit(1))

        producer = threading.Thread(target=submit)
        producer.start()
        self.assertTrue(producer_entered.wait(2))
        release.set()
        producer.join(2)
        pipeline.close(timeout=2)
        self.assertEqual(0, first.result(2))
        future = accepted[0]
        self.assertEqual(1, future.result(2))
        times = future.pipeline_timestamps_ns
        self.assertEqual({"submitted", "admitted", "prepare_start", "prepare_end", "infer_start",
                          "infer_end", "finish_start", "finish_end", "deliver_start", "deliver_end"}, set(times))
        durations = future.pipeline_durations_ns
        self.assertEqual(times["admitted"] - times["submitted"], durations["credit_wait"])
        self.assertGreater(durations["credit_wait"], 0)
        self.assertEqual(times["prepare_end"] - times["prepare_start"], durations["prepare_service"])
        self.assertEqual(times["finish_end"] - times["submitted"], durations["ingress_to_edge_ready"])
        self.assertEqual("monotonic_raw", pipeline.snapshot()["clock_domain"])
