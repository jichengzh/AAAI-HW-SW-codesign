# -*- coding: utf-8 -*-
"""Ego-output fixed latency for CARLA control batches.

Each call snapshots the controls generated at the current simulator tick.  For
non-zero latency, the snapshot is released exactly ``delay_frames`` later;
until then, a neutral control with the same container shape is returned.
"""

from collections import deque
import csv
import math
import os


_CONTROL_FIELDS = (
    "throttle",
    "steer",
    "brake",
    "hand_brake",
    "reverse",
    "manual_gear_shift",
    "gear",
)


def _copy_control(value, neutral=False):
    if value is None:
        return None
    copied = type(value)()
    if not neutral:
        for field in _CONTROL_FIELDS:
            if hasattr(value, field):
                setattr(copied, field, getattr(value, field))
    return copied


def _copy_batch(batch, neutral=False):
    if isinstance(batch, dict):
        return {key: _copy_control(value, neutral) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_copy_control(value, neutral) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_copy_control(value, neutral) for value in batch)
    raise TypeError("control batch must be a dict, list, or tuple")


def _first_control(batch):
    values = batch.values() if isinstance(batch, dict) else batch
    return next((value for value in values if value is not None), None)


def _control_value(control, field):
    if control is None:
        return None
    return getattr(control, field, None)


class EgoOutputFifoDelay(object):
    """Release each ego control-output batch after a fixed frame delay."""

    def __init__(
        self,
        delay_ms,
        sim_rate_hz=20.0,
        audit_path=None,
        audit_flush_interval=100,
    ):
        delay_ms = float(delay_ms)
        sim_rate_hz = float(sim_rate_hz)
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if sim_rate_hz <= 0:
            raise ValueError("sim_rate_hz must be positive")
        if int(audit_flush_interval) <= 0:
            raise ValueError("audit_flush_interval must be positive")
        self.delay_ms = delay_ms
        self.sim_rate_hz = sim_rate_hz
        self.delay_frames = int(math.ceil(delay_ms * sim_rate_hz / 1000.0))
        self._queue = deque()
        self.last_audit = None
        self.audit_rows = []
        self._audit_path = audit_path
        self._audit_flush_interval = int(audit_flush_interval)

    def release(self, generated, step):
        step = int(step)
        snapshot = _copy_batch(generated)
        self._queue.append((step, snapshot))

        if len(self._queue) <= self.delay_frames:
            source_step = None
            released = _copy_batch(generated, neutral=True)
        else:
            source_step, released = self._queue.popleft()

        generated_control = _first_control(generated)
        applied_control = _first_control(released)
        generated_ready_frame = step + self.delay_frames
        applied_ready_frame = (
            None if source_step is None else source_step + self.delay_frames
        )
        control_age_frames = (
            None if source_step is None else step - source_step
        )
        audit = {
            "apply_frame": step,
            "generated_source_frame": step,
            "generated_ready_frame": generated_ready_frame,
            "applied_source_frame": source_step,
            "applied_ready_frame": applied_ready_frame,
            "configured_delay_ms": self.delay_ms,
            "delay_frames": self.delay_frames,
            "control_age_frames": control_age_frames,
            "is_startup_neutral": source_step is None,
            "generated_throttle": _control_value(generated_control, "throttle"),
            "generated_steer": _control_value(generated_control, "steer"),
            "generated_brake": _control_value(generated_control, "brake"),
            "applied_throttle": _control_value(applied_control, "throttle"),
            "applied_steer": _control_value(applied_control, "steer"),
            "applied_brake": _control_value(applied_control, "brake"),
            "step": step,
            "source_step": source_step,
            "delay_ms": self.delay_ms,
            "age_frames": control_age_frames,
            "is_startup_hold": source_step is None,
        }
        self.last_audit = dict(audit)
        self.audit_rows.append(dict(audit))
        if (
            self._audit_path is not None
            and len(self.audit_rows) % self._audit_flush_interval == 0
        ):
            self.write_audit_csv(self._audit_path)
        return released

    def close(self):
        if self._audit_path is not None:
            self.write_audit_csv(self._audit_path)

    def write_audit_csv(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fieldnames = [
            "apply_frame",
            "generated_source_frame",
            "generated_ready_frame",
            "applied_source_frame",
            "applied_ready_frame",
            "configured_delay_ms",
            "delay_frames",
            "control_age_frames",
            "is_startup_neutral",
            "generated_throttle",
            "generated_steer",
            "generated_brake",
            "applied_throttle",
            "applied_steer",
            "applied_brake",
            "step",
            "source_step",
            "delay_ms",
            "age_frames",
            "is_startup_hold",
        ]
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.audit_rows)


ActionFifoDelay = EgoOutputFifoDelay
