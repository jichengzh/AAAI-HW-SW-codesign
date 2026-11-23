"""Bounded FIFO prepare → single GPU caller → CPU finish → ordered sender.

Credit covers the complete accepted lifetime, not queue length. Callbacks must
return or raise (network callbacks should use socket timeouts); Python cannot
interrupt an executing callback. Failure rejects admission and fails all accepted
futures promptly; close still joins running callbacks before resources are freed.
"""

from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
import math
import queue
import threading
import time
from types import MappingProxyType

from ditl_edge_runtime import monotonic_raw_ns


_STOP = object()
_STAGES = ("prepare", "infer", "finish", "deliver")


def _deadline(timeout):
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and non-negative")
    return None if timeout is None else time.monotonic() + timeout


def _remaining(deadline):
    return None if deadline is None else max(0, deadline - time.monotonic())


class PipelineFuture(Future):
    """Result is the finished value, only after deliver succeeds.

    The two read-only mappings are frozen before success/failure resolves this
    future. Times use the same local CLOCK_MONOTONIC_RAW as runtime headers.
    Failed jobs expose only stage boundaries actually observed at failure.
    source_read is a prepare_service alias (only callback execution, no ingress
    or credit waiting). source_to_ready includes preparation through CPU finish;
    ingress_to_edge_ready additionally includes admission/credit and queue waits.
    """

    def __init__(self):
        super(PipelineFuture, self).__init__()
        self._timestamps = MappingProxyType({})
        self._durations = MappingProxyType({})
        # Accepted FIFO tasks cannot be silently cancelled by callers.
        self.set_running_or_notify_cancel()

    @property
    def pipeline_timestamps_ns(self):
        return self._timestamps

    @property
    def pipeline_durations_ns(self):
        return self._durations

    def _freeze_timing(self, times):
        self._timestamps = MappingProxyType(dict(times))
        boundaries = {
            "credit_wait": ("submitted", "admitted"),
            "prepare_queue": ("admitted", "prepare_start"),
            "prepare_service": ("prepare_start", "prepare_end"),
            "source_read": ("prepare_start", "prepare_end"),
            "infer_queue": ("prepare_end", "infer_start"),
            "infer": ("infer_start", "infer_end"),
            "finish_queue": ("infer_end", "finish_start"),
            "finish": ("finish_start", "finish_end"),
            "deliver_queue": ("finish_end", "deliver_start"),
            "deliver": ("deliver_start", "deliver_end"),
            "elapsed": ("submitted", "deliver_end"),
            "source_to_ready": ("prepare_start", "finish_end"),
            "ingress_to_edge_ready": ("submitted", "finish_end"),
        }
        self._durations = MappingProxyType({
            name: times[end] - times[start]
            for name, (start, end) in boundaries.items() if start in times and end in times
        })


@dataclass
class _Job:
    value: object
    future: PipelineFuture
    times: dict = field(default_factory=dict)


class BoundedHilPipeline:
    """One thread per stage; unbounded internal queues share bounded credits.

    submit blocks before input preparation when credit is exhausted. Timeout
    admits nothing. drain waits for accepted callbacks to settle and propagates
    the first failure; close rejects new work, drains FIFO and joins threads.
    close(wait=False) initiates that same drain asynchronously. A later close
    can wait; a finite timeout reports a stuck callback without freeing resources.
    """

    def __init__(self, prepare, infer, finish, deliver, capacity=2):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        callbacks = (prepare, infer, finish, deliver)
        if not all(callable(callback) for callback in callbacks):
            raise ValueError("all pipeline stages must be callable")
        self.capacity = capacity
        self._condition = threading.Condition()
        self._queues = [queue.Queue() for _ in _STAGES]
        self._jobs = {}
        self._closed = False
        self._stopping = False
        self._error = None
        self._errors = ()
        self._submitted = self._completed = self._failed = self._peak = 0
        self._threads = []
        for index, (name, callback) in enumerate(zip(_STAGES, callbacks)):
            thread = threading.Thread(target=self._run, args=(index, name, callback),
                                      name="hil-pipeline-" + name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def submit(self, item, timeout=None):
        submitted = monotonic_raw_ns()
        deadline = _deadline(timeout)
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError("pipeline is closed")
                if len(self._jobs) < self.capacity:
                    break
                remaining = _remaining(deadline)
                if remaining == 0:
                    raise TimeoutError("pipeline credit admission timed out")
                self._condition.wait(remaining)
            future = PipelineFuture()
            job = _Job(item, future, {"submitted": submitted, "admitted": monotonic_raw_ns()})
            self._jobs[id(job)] = job
            self._submitted += 1
            self._peak = max(self._peak, len(self._jobs))
            # Admission order and queue insertion are one atomic FIFO operation.
            self._queues[0].put(job)
            return future

    def _publish(self, job, error=None):
        if not job.future.done():
            job.future._freeze_timing(job.times)
            if error is None:
                job.future.set_result(job.value)
            else:
                job.future.set_exception(error)

    def _abort(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
                self._errors = ("%s: %s" % (type(error).__name__, error),)
                self._closed = True
                for job in tuple(self._jobs.values()):
                    self._publish(job, error)
                self._stop_locked()
                self._condition.notify_all()

    def _settle(self, job):
        with self._condition:
            self._jobs.pop(id(job))
            if self._error is None:
                self._completed += 1
            else:
                self._failed += 1
            self._publish(job, self._error)
            self._condition.notify_all()

    def _run(self, index, name, callback):
        channel = self._queues[index]
        while True:
            job = channel.get()
            try:
                if job is _STOP:
                    if index + 1 < len(self._queues):
                        self._queues[index + 1].put(_STOP)
                    return
                with self._condition:
                    failed = self._error is not None
                if failed:
                    self._settle(job)
                    continue
                job.times[name + "_start"] = monotonic_raw_ns()
                try:
                    if name == "deliver":
                        callback(job.value)
                    else:
                        job.value = callback(job.value)
                except BaseException as error:
                    job.times[name + "_end"] = monotonic_raw_ns()
                    self._abort(error)
                    self._settle(job)
                    continue
                job.times[name + "_end"] = monotonic_raw_ns()
                if index + 1 == len(self._queues):
                    self._settle(job)
                else:
                    self._queues[index + 1].put(job)
            finally:
                channel.task_done()
                del job

    def _stop_locked(self):
        if not self._stopping:
            self._stopping = True
            self._queues[0].put(_STOP)

    def drain(self, timeout=None):
        deadline = _deadline(timeout)
        with self._condition:
            while self._jobs:
                remaining = _remaining(deadline)
                if remaining == 0:
                    raise TimeoutError("pipeline drain timed out")
                self._condition.wait(remaining)
            if self._error is not None:
                raise self._error

    def close(self, wait=True, timeout=None):
        deadline = _deadline(timeout)
        with self._condition:
            self._closed = True
            self._stop_locked()
            self._condition.notify_all()
        if wait:
            for thread in self._threads:
                thread.join(_remaining(deadline))
                if thread.is_alive():
                    raise TimeoutError("pipeline close timed out; callback still running")
            self.drain(_remaining(deadline))

    def snapshot(self):
        with self._condition:
            return {"capacity": self.capacity, "submitted": self._submitted,
                    "completed": self._completed, "failed": self._failed,
                    "inflight": len(self._jobs), "inflight_peak": self._peak,
                    "errors": self._errors, "closed": self._closed,
                    "clock_domain": "monotonic_raw"}
