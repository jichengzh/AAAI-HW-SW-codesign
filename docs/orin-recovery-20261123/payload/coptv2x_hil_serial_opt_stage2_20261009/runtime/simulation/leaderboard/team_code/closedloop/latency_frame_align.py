# -*- coding: utf-8 -*-
"""
Latency-aware frame alignment for V2Xverse closed-loop simulation.

Implements the design in
  multi_agent/methods/design/sim_closedloop_arch_v1.md §3.2 / §3.3.

Core idea (producer view, see design §3.2):
  - At frame j the agent submits RSU perception result + its measured/injected
    latency L_ms. We convert L to a delay in CARLA frames:
        Delta = ceil(L_ms / FRAME_MS).
    The result becomes "available" from frame (j + Delta).
  - At consume frame k, we pick the newest RSU result whose
    (source_frame + its Delta) <= k. If none newer than the last used one
    is available, we zero-order-hold (reuse the previous RSU result).
  - Ego self-perception is NEVER delayed and NEVER zero-order-held; only the
    RSU / other-CAV branch goes through this aligner (design §3.3-3).

CARLA runs sync @ 20 Hz => fixed_delta = 50 ms / frame (design §3.0). The
agent's wallclock blocking does NOT advance game time, so the ONLY game-time
delay source is Delta (design §3.2 invariant). Submission must therefore be
non-blocking on the agent side (design C-2 acceptance criterion).

This module is pure-python (no torch) and unit-testable in isolation.
"""

import csv
import math
import os


FRAME_MS = 50.0          # CARLA sync 20 Hz => 50 ms / frame (design §3.0)
DEFAULT_MAX_DELAY_FRAMES = None


# ---------------------------------------------------------------------------
# Trace-replay latency source (§3.8 R-3)
# ---------------------------------------------------------------------------

class LatencyTraceReader:
    """Read a pre-collected per-call latency trace CSV, cycling when exhausted.

    CSV format (output of scripts/collect_rsu_latency_trace.py):
        call_idx,tau_ms
        0,<float>
        1,<float>
        ...

    The reader iterates over tau_ms values in call_idx order.  When all
    values have been consumed:
      - ``cycle=True``  (default): wrap to index 0 and increment cycle_count.
      - ``cycle=False`` : hold the last value indefinitely.

    Either way a warning is emitted on the first wrap/hold so the audit log
    can flag it.  ``latency_ms_source`` is always
    ``'trace_replay_codriving_4090'`` (§3.8 口径).
    """

    LATENCY_SOURCE = 'trace_replay_codriving_4090'

    def __init__(self, csv_path, cycle=True):
        self._csv_path = csv_path
        self._cycle = cycle
        self._values = self._load(csv_path)   # list of float tau_ms
        self._pos = 0
        self._cycle_count = 0
        if not self._values:
            raise ValueError(
                "LatencyTraceReader: empty or missing tau_ms in %s" % csv_path)
        print("[LatencyTraceReader] loaded %d entries from %s (cycle=%s)" %
              (len(self._values), csv_path, cycle))

    @staticmethod
    def _load(csv_path):
        values = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                values.append(float(row['tau_ms']))
        return values

    def next(self):
        """Return next tau_ms from the trace.

        Cycles (or holds last) when exhausted.  Emits a log line on first wrap.
        """
        if self._pos >= len(self._values):
            if self._cycle:
                self._cycle_count += 1
                self._pos = 0
                print("[LatencyTraceReader] trace exhausted, cycling (cycle #%d)" %
                      self._cycle_count)
            else:
                # hold last value
                if self._cycle_count == 0:
                    self._cycle_count = 1  # mark as 'held' state
                    print("[LatencyTraceReader] trace exhausted, holding last value")
                return self._values[-1]
        val = self._values[self._pos]
        self._pos += 1
        return val

    @property
    def n_calls(self):
        """Total number of trace entries."""
        return len(self._values)

    @property
    def cycle_count(self):
        """Number of times the trace has wrapped around (cycle=True) or 1 if holding."""
        return self._cycle_count

    @property
    def pos(self):
        """Current read position (0-based index into values list)."""
        return self._pos


def latency_to_delay_frames(latency_ms, frame_ms=FRAME_MS,
                            max_delay_frames=DEFAULT_MAX_DELAY_FRAMES):
    """Convert a measured/injected latency (ms) to an integer frame delay.

    ceil so an unfinished frame is never made available early (safety side,
    design §3.3-2). Delta=0 only when latency_ms<=0 (the ideal/no-delay arm).
    An explicitly configured max_delay_frames rejects overflow instead of
    silently clipping it.
    """
    if latency_ms is None or latency_ms <= 0.0:
        return 0
    delta = int(math.ceil(float(latency_ms) / float(frame_ms)))
    if max_delay_frames is not None and delta > int(max_delay_frames):
        raise ValueError(
            "delay_frames=%d exceeds max_delay_frames=%d" %
            (delta, int(max_delay_frames))
        )
    return delta


class RsuLatencyAligner:
    """Frame-id keyed buffer of RSU perception results with delay + ZOH.

    The aligner is agnostic to *what* the RSU result payload is (it stores it
    opaquely), so it works for raw-data dicts, detection tensors, or fused
    BEV features. Placement (fusion-front, RSU-only) is enforced by the caller
    wiring it on the RSU branch only (design §3.3-3).
    """

    def __init__(self, frame_ms=FRAME_MS, max_delay_frames=DEFAULT_MAX_DELAY_FRAMES):
        self.frame_ms = frame_ms
        self.max_delay_frames = max_delay_frames
        # available_at_frame -> (source_frame, payload). We keep the mapping
        # keyed by source_frame and compute availability on read.
        self._buffer = {}            # source_frame -> dict(avail_frame, payload, latency_ms, source)
        self._last_used = None       # dict(source_frame, payload) of last consumed RSU result
        self._last_submitted = None   # dict(source_frame, ready_frame, delay_frames) of latest submission

    def submit(self, source_frame, payload, latency_ms, latency_source):
        """Producer: register an RSU result computed for `source_frame`.

        latency_ms: measured (in-carla) or injected (from E7) latency.
        latency_source: 'measured_in_carla' | 'injected_from_E7' (audit, design §3.3-3).
        Returns the computed delay (frames) for logging.
        """
        delta = latency_to_delay_frames(latency_ms, self.frame_ms, self.max_delay_frames)
        ready_frame = source_frame + delta
        self._buffer[source_frame] = {
            'source_frame': source_frame,
            'ready_frame': ready_frame,
            'delay_frames': delta,
            # Keep original private aliases for compatibility with existing
            # trace/replay callers that may inspect buffered entries.
            'avail_frame': ready_frame,
            'payload': payload,
            'latency_ms': latency_ms,
            'latency_source': latency_source,
            'delta': delta,
        }
        self._last_submitted = {
            'source_frame': source_frame,
            'ready_frame': ready_frame,
            'delay_frames': delta,
            'latency_ms': latency_ms,
            'latency_source': latency_source,
        }
        return delta

    def consume(self, k):
        """Consumer at frame k: return (payload, audit) of the RSU result to use.

        Picks the newest entry whose avail_frame <= k. If that is no newer than
        what we already used (or nothing is available yet), zero-order-hold the
        previous result. Returns (payload_or_None, audit_dict).

        audit_dict fields (design §3.3-3, 6 fields):
          step, delta, used_rsu_frame_id, is_zoh_held, latency_ms_source, zoh_age_frames
        """
        # candidates available by frame k
        avail = [(sf, e) for sf, e in self._buffer.items() if e['ready_frame'] <= k]

        chosen = None
        if avail:
            # newest available by source_frame
            sf, e = max(avail, key=lambda x: x[0])
            chosen = (sf, e)

        is_zoh = False
        if chosen is None:
            # nothing available yet -> ZOH last used (or None at the very start)
            is_zoh = True
            if self._last_used is None:
                audit = self._mk_audit(
                    k, delta=None, used_frame=None, used_ready_frame=None,
                    latency_ms=None, is_zoh=True, latency_source=None,
                    zoh_age=None, is_ego_only_startup=True,
                )
                return None, audit
            sf = self._last_used['source_frame']
            payload = self._last_used['payload']
            latency_source = self._last_used.get('latency_source')
            delta = self._last_used.get('delta')
            latency_ms = self._last_used.get('latency_ms')
            used_ready_frame = self._last_used.get('ready_frame')
        else:
            sf, e = chosen
            # if the chosen result is not newer than last used, it's effectively a hold
            if self._last_used is not None and sf <= self._last_used['source_frame']:
                is_zoh = True
                sf = self._last_used['source_frame']
                payload = self._last_used['payload']
                latency_source = self._last_used.get('latency_source')
                delta = self._last_used.get('delta')
                latency_ms = self._last_used.get('latency_ms')
                used_ready_frame = self._last_used.get('ready_frame')
            else:
                payload = e['payload']
                latency_source = e['latency_source']
                delta = e['delta']
                latency_ms = e['latency_ms']
                used_ready_frame = e['ready_frame']
                self._last_used = {
                    'source_frame': sf, 'payload': payload,
                    'latency_source': latency_source, 'delta': delta,
                    'latency_ms': latency_ms, 'ready_frame': used_ready_frame,
                }

        # zoh_age = how many frames stale the used RSU result is, relative to when
        # it *became available* (= source_frame + delta). >=0. (design §3.3-3)
        zoh_age = max(0, k - used_ready_frame)
        audit = self._mk_audit(
            k, delta=delta, used_frame=sf, used_ready_frame=used_ready_frame,
            latency_ms=latency_ms, is_zoh=is_zoh,
            latency_source=latency_source, zoh_age=zoh_age,
            is_ego_only_startup=False,
        )
        return payload, audit

    def gc(self, k):
        """Drop buffer entries strictly older than the last-used source frame
        (and older than max_delay window) to bound memory (design §3.3-2)."""
        if self.max_delay_frames is None:
            if self._last_used is None:
                return
            keep_floor = self._last_used['source_frame']
        else:
            keep_floor = k - self.max_delay_frames
        for sf in [s for s in self._buffer if s < keep_floor]:
            # never drop the last-used source frame
            if self._last_used is not None and sf == self._last_used['source_frame']:
                continue
            del self._buffer[sf]

    def _mk_audit(self, step, delta, used_frame, used_ready_frame, latency_ms,
                  is_zoh, latency_source, zoh_age, is_ego_only_startup):
        submitted = self._last_submitted
        return {
            'consume_frame': step,
            'submitted_rsu_source_frame': (
                submitted['source_frame'] if submitted is not None else None),
            'submitted_rsu_ready_frame': (
                submitted['ready_frame'] if submitted is not None else None),
            'used_rsu_source_frame': used_frame,
            'used_rsu_ready_frame': used_ready_frame,
            'source_sim_time_s': (
                None if used_frame is None else used_frame * self.frame_ms / 1000.0),
            'consume_sim_time_s': step * self.frame_ms / 1000.0,
            'configured_delay_ms': latency_ms,
            'delay_frames': delta,
            'information_age_frames': (
                step - used_frame if used_frame is not None else None),
            'is_ego_only_startup': bool(is_ego_only_startup),
            'step': step,
            'delta': delta,
            'used_rsu_frame_id': used_frame,
            'is_zoh_held': bool(is_zoh),
            'latency_ms_source': latency_source,
            'zoh_age_frames': zoh_age,
        }


def apply_delayed_collaboration(current_car_data, current_rsu_data, used_payload):
    """Return current ego data plus delayed collaboration inputs only.

    The returned lists are new list containers, so selecting delayed data never
    mutates the caller's current-frame lists.  During FIFO startup, CAV and RSU
    inputs are deliberately absent rather than leaking current-frame data.
    """
    current_cars = list(current_car_data or [])
    if not current_cars:
        return [], []
    delayed_cars = [current_cars[0]] + [None] * (len(current_cars) - 1)
    if used_payload is None:
        return delayed_cars, []
    payload_cars = list(used_payload.get('car_data') or [])
    for index in range(1, len(delayed_cars)):
        delayed_cars[index] = (
            payload_cars[index] if index < len(payload_cars) else None
        )
    return delayed_cars, list(used_payload.get('rsu_data') or [])
