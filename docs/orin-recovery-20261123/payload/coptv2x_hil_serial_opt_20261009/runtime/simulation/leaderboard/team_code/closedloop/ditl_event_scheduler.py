"""Bounded continuous-time delivery scheduling for DitL replay."""
from __future__ import print_function

from dataclasses import dataclass
import heapq
import math
import numbers
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from .ditl_impairment_policy import (
        DeliveryPlan, DeterministicImpairmentPolicy, MessageIdentity,
        PolicyDecision,
    )
    from .ditl_trace_library import ReplaySampleCursor
except ImportError:
    from ditl_impairment_policy import (
        DeliveryPlan, DeterministicImpairmentPolicy, MessageIdentity,
        PolicyDecision,
    )
    from ditl_trace_library import ReplaySampleCursor


@dataclass(frozen=True)
class LogicalMessage:
    logical_message_id: str
    stream_id: str
    source_id: str
    source_frame: int
    source_sim_time_ns: int
    sequence_id: int
    payload: Mapping[str, object]
    trace_sample_id: str
    base_delay_ns: int
    cycle_index: int


@dataclass(frozen=True)
class DeliveryEvent:
    delivery_id: str
    message: LogicalMessage
    copy_index: int
    effective_delay_ns: int
    ready_time_ns: int
    expiry_time_ns: Optional[int]
    reorder_delay_ns: int


@dataclass(frozen=True)
class SubmissionOutcome:
    message: LogicalMessage
    policy_decision: PolicyDecision
    enqueued: Tuple[DeliveryEvent, ...]


@dataclass(frozen=True)
class AdvanceResult:
    consume_sim_time_ns: int
    ready: Tuple[DeliveryEvent, ...]
    queue_depth: int


@dataclass(frozen=True)
class ReplaySubmission:
    route_id: str
    stream_id: str
    source_id: str
    source_frame: int
    source_sim_time_ns: int
    sequence_id: int
    payload: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True)
class DeliveryOutcome:
    event: DeliveryEvent
    status: str
    arrival_deadline_miss: bool

    def __post_init__(self):
        object.__setattr__(self, "event", _freeze_event(self.event))


@dataclass(frozen=True)
class StreamSelection:
    stream_id: str
    used_message_id: Optional[str]
    used_sequence_id: Optional[int]
    payload: Optional[Mapping[str, object]]
    source_frame: Optional[int]
    source_sim_time_ns: Optional[int]
    ready_time_ns: Optional[int]
    aoi_ns: Optional[int]
    zoh_hold_ns: Optional[int]
    is_zoh: bool
    is_startup_ego_only: bool
    is_expired: bool
    fusion_deadline_miss: bool

    def __post_init__(self):
        if self.payload is not None:
            object.__setattr__(self, "payload", _freeze_payload(self.payload))

    def materialize_payload(self):
        """Return an independent native payload for one consumer call."""
        if self.payload is None:
            return None
        return _thaw_payload(self.payload)


@dataclass(frozen=True)
class ReplayCounters:
    accepted_count: int
    dropped_count: int
    expired_count: int


@dataclass(frozen=True)
class ReplayTick:
    consume_frame: int
    consume_sim_time_ns: int
    streams: Mapping[str, StreamSelection]
    delivery_outcomes: Tuple[DeliveryOutcome, ...]
    queue_depth: int
    counters: ReplayCounters

    def __post_init__(self):
        object.__setattr__(self, "streams", MappingProxyType(dict(self.streams)))
        object.__setattr__(self, "delivery_outcomes", tuple(self.delivery_outcomes))


@dataclass(frozen=True)
class _AcceptedState:
    event: DeliveryEvent
    payload: Mapping[str, object]
    accepted_at_ns: int


def _require_nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be a non-negative integer" % name)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % name)


def _snapshot_payload(value, active=None):
    """Copy the explicitly supported payload graph without user dispatch."""
    if active is None:
        active = set()

    if value is None or isinstance(value, (bool, str, bytes, numbers.Number)):
        return value
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError("unsupported payload ndarray with object dtype")
        return np.ndarray.copy(value)
    if isinstance(value, Image.Image) and type(value) is not Image.Image:
        raise TypeError("unsupported payload Pillow Image subclass")
    if type(value) is Image.Image:
        return Image.Image.copy(value)

    value_type = type(value)
    if value_type not in (dict, MappingProxyType, list, tuple):
        raise TypeError(
            "unsupported payload value type: %s" % value_type.__name__
        )

    identity = id(value)
    if identity in active:
        raise ValueError("payload container cycle is not supported")
    active.add(identity)
    try:
        if value_type in (dict, MappingProxyType):
            copied = {}
            for key, item in value.items():
                copied_key = _snapshot_payload(key, active)
                copied[copied_key] = _snapshot_payload(item, active)
            return copied
        if value_type is list:
            return [_snapshot_payload(item, active) for item in value]
        return tuple(_snapshot_payload(item, active) for item in value)
    finally:
        active.remove(identity)


def _freeze_snapshot(value):
    if type(value) in (dict, MappingProxyType):
        return MappingProxyType({key: _freeze_snapshot(item)
                                 for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze_snapshot(item) for item in value)
    if type(value) is np.ndarray:
        frozen = np.ndarray.copy(value)
        frozen.setflags(write=False)
        return frozen
    if type(value) is Image.Image:
        return Image.Image.copy(value)
    return value


def _freeze_payload(value):
    if type(value) not in (dict, MappingProxyType):
        raise TypeError("payload must be a dictionary")
    return _freeze_snapshot(_snapshot_payload(value))


def _thaw_payload(value):
    if type(value) is MappingProxyType:
        return {key: _thaw_value(item) for key, item in value.items()}
    raise TypeError("payload must be an immutable mapping")


def _thaw_value(value):
    if type(value) is MappingProxyType:
        return {key: _thaw_value(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_thaw_value(item) for item in value)
    if type(value) is np.ndarray:
        return np.ndarray.copy(value)
    if type(value) is Image.Image:
        return Image.Image.copy(value)
    return value


def _freeze_event(event):
    message = event.message
    frozen_message = LogicalMessage(
        message.logical_message_id, message.stream_id, message.source_id,
        message.source_frame, message.source_sim_time_ns, message.sequence_id,
        _freeze_payload(message.payload), message.trace_sample_id,
        message.base_delay_ns, message.cycle_index,
    )
    return DeliveryEvent(
        event.delivery_id, frozen_message, event.copy_index,
        event.effective_delay_ns, event.ready_time_ns, event.expiry_time_ns,
        event.reorder_delay_ns,
    )


def _validate_message(message):
    if not isinstance(message, LogicalMessage):
        raise TypeError("message must be a LogicalMessage")
    for name in (
            "logical_message_id", "stream_id", "source_id",
            "trace_sample_id"):
        value = getattr(message, name)
        if not isinstance(value, str) or not value:
            raise ValueError("%s must be a non-empty string" % name)
    for name in (
            "source_frame", "source_sim_time_ns", "sequence_id",
            "base_delay_ns", "cycle_index"):
        _require_nonnegative_integer(name, getattr(message, name))
    identity_prefix, separator, identity_sequence = (
        message.logical_message_id.rpartition("-")
    )
    if (not separator or not identity_prefix or
            identity_sequence != str(message.sequence_id)):
        raise ValueError(
            "logical_message_id must end with the canonical sequence_id"
        )
    if type(message.payload) is not dict:
        raise TypeError("payload must be a dictionary")


def _canonical_decision(decision):
    if type(decision) is not PolicyDecision:
        raise TypeError("decision must be a PolicyDecision")
    if type(decision.dropped) is not bool:
        raise ValueError("decision.dropped must be a boolean")
    if type(decision.delay_clamped) is not bool:
        raise ValueError("decision.delay_clamped must be a boolean")
    for name in ("loss_uniform", "duplicate_uniform"):
        value = getattr(decision, name)
        if (type(value) is not float or not math.isfinite(value) or
                not 0.0 <= value <= 1.0):
            raise ValueError("decision.%s must be a finite float in [0, 1]" % name)
    if type(decision.burst_state) is not str or decision.burst_state not in (
            "disabled", "good", "bad"):
        raise ValueError("decision.burst_state is invalid")
    if type(decision.deliveries) is not tuple:
        raise TypeError("decision.deliveries must be a tuple")

    if decision.dropped:
        if type(decision.drop_reason) is not str or not decision.drop_reason:
            raise ValueError("a dropped decision requires a drop_reason")
        if decision.deliveries:
            raise ValueError("a dropped decision cannot contain deliveries")
    else:
        if decision.drop_reason is not None:
            raise ValueError("a kept decision cannot contain a drop_reason")
        if not decision.deliveries:
            raise ValueError("a kept decision requires at least one delivery")

    seen_copy_indices = set()
    canonical_deliveries = []
    for plan in decision.deliveries:
        if type(plan) is not DeliveryPlan:
            raise TypeError("deliveries must contain DeliveryPlan values")
        for name in ("copy_index", "effective_delay_ns", "reorder_delay_ns"):
            _require_nonnegative_integer(name, getattr(plan, name))
        if plan.copy_index in seen_copy_indices:
            raise ValueError("copy_index values must be unique")
        seen_copy_indices.add(plan.copy_index)
        canonical_deliveries.append(DeliveryPlan(
            copy_index=plan.copy_index,
            effective_delay_ns=plan.effective_delay_ns,
            reorder_delay_ns=plan.reorder_delay_ns,
        ))
    return PolicyDecision(
        dropped=decision.dropped,
        drop_reason=decision.drop_reason,
        delay_clamped=decision.delay_clamped,
        loss_uniform=decision.loss_uniform,
        duplicate_uniform=decision.duplicate_uniform,
        burst_state=decision.burst_state,
        deliveries=tuple(canonical_deliveries),
    )


def _snapshot_message(message):
    _validate_message(message)
    return LogicalMessage(
        logical_message_id=message.logical_message_id,
        stream_id=message.stream_id,
        source_id=message.source_id,
        source_frame=message.source_frame,
        source_sim_time_ns=message.source_sim_time_ns,
        sequence_id=message.sequence_id,
        payload=_snapshot_payload(message.payload),
        trace_sample_id=message.trace_sample_id,
        base_delay_ns=message.base_delay_ns,
        cycle_index=message.cycle_index,
    )


def _copy_event(event):
    return DeliveryEvent(
        delivery_id=event.delivery_id,
        message=_snapshot_message(event.message),
        copy_index=event.copy_index,
        effective_delay_ns=event.effective_delay_ns,
        ready_time_ns=event.ready_time_ns,
        expiry_time_ns=event.expiry_time_ns,
        reorder_delay_ns=event.reorder_delay_ns,
    )


class DitlEventScheduler(object):
    """Schedule immutable message snapshots in a capacity-bounded min-heap."""

    def __init__(self, max_queue_events: int, ttl_ns: Optional[int]):
        if isinstance(max_queue_events, bool) or not isinstance(
                max_queue_events, int):
            raise TypeError("max_queue_events must be a positive integer")
        if max_queue_events <= 0:
            raise ValueError("max_queue_events must be a positive integer")
        if ttl_ns is not None:
            _require_nonnegative_integer("ttl_ns", ttl_ns)

        self._max_queue_events = max_queue_events
        self._ttl_ns = ttl_ns
        self._heap = []
        self._insertion_ordinal = 0
        self._last_submitted_sequence = {}  # type: Dict[str, int]
        self._last_submitted_source_time_ns = None  # type: Optional[int]
        self._last_consume_time_ns = None  # type: Optional[int]

    @property
    def ttl_ns(self):
        return self._ttl_ns

    def submit(self, message: LogicalMessage,
               decision: PolicyDecision) -> SubmissionOutcome:
        _validate_message(message)
        stored_decision = _canonical_decision(decision)

        previous_sequence = self._last_submitted_sequence.get(
            message.stream_id
        )
        if previous_sequence is not None and message.sequence_id <= previous_sequence:
            raise ValueError(
                "sequence_id must be strictly increasing per stream"
            )
        if (self._last_submitted_source_time_ns is not None and
                message.source_sim_time_ns < self._last_submitted_source_time_ns):
            raise ValueError("source_sim_time_ns must be non-decreasing")

        required = len(stored_decision.deliveries)
        if len(self._heap) + required > self._max_queue_events:
            raise OverflowError("scheduler queue capacity would be exceeded")

        stored_message = _snapshot_message(message)
        pending = []
        next_ordinal = self._insertion_ordinal
        for plan in stored_decision.deliveries:
            ready_time_ns = (
                stored_message.source_sim_time_ns +
                plan.effective_delay_ns +
                plan.reorder_delay_ns
            )
            expiry_time_ns = (
                None if self._ttl_ns is None else
                stored_message.source_sim_time_ns + self._ttl_ns
            )
            event = DeliveryEvent(
                delivery_id="%s:%s:%d:%d" % (
                    stored_message.stream_id,
                    stored_message.logical_message_id,
                    stored_message.sequence_id,
                    plan.copy_index,
                ),
                message=stored_message,
                copy_index=plan.copy_index,
                effective_delay_ns=plan.effective_delay_ns,
                ready_time_ns=ready_time_ns,
                expiry_time_ns=expiry_time_ns,
                reorder_delay_ns=plan.reorder_delay_ns,
            )
            key = (
                ready_time_ns,
                stored_message.sequence_id,
                plan.copy_index,
                next_ordinal,
            )
            pending.append((key, event))
            next_ordinal += 1

        for entry in pending:
            heapq.heappush(self._heap, entry)
        self._insertion_ordinal = next_ordinal
        self._last_submitted_sequence[stored_message.stream_id] = (
            stored_message.sequence_id
        )
        self._last_submitted_source_time_ns = stored_message.source_sim_time_ns

        result_message = _snapshot_message(stored_message)
        result_events = tuple(_copy_event(event) for _, event in pending)
        return SubmissionOutcome(result_message, stored_decision, result_events)

    def advance(self, consume_sim_time_ns: int) -> AdvanceResult:
        _require_nonnegative_integer(
            "consume_sim_time_ns", consume_sim_time_ns
        )
        if (self._last_consume_time_ns is not None and
                consume_sim_time_ns < self._last_consume_time_ns):
            raise ValueError("consume_sim_time_ns must be non-decreasing")

        ready = []
        while self._heap and self._heap[0][0][0] <= consume_sim_time_ns:
            _, event = heapq.heappop(self._heap)
            ready.append(_copy_event(event))
        self._last_consume_time_ns = consume_sim_time_ns
        return AdvanceResult(
            consume_sim_time_ns=consume_sim_time_ns,
            ready=tuple(ready),
            queue_depth=len(self._heap),
        )


def _validate_replay_submission(submission, stream_ids):
    if type(submission) is not ReplaySubmission:
        raise TypeError("submission must be a ReplaySubmission")
    for name in ("route_id", "stream_id", "source_id"):
        value = getattr(submission, name)
        if not isinstance(value, str) or not value:
            raise ValueError("%s must be a non-empty string" % name)
    if submission.stream_id not in stream_ids:
        raise ValueError("stream_id must identify a configured stream")
    for name in ("source_frame", "source_sim_time_ns", "sequence_id"):
        _require_nonnegative_integer(name, getattr(submission, name))


class DitlReplayEngine(object):
    """Select the freshest usable delivery independently for each stream."""

    def __init__(self, sample_cursor: ReplaySampleCursor,
                 policy: DeterministicImpairmentPolicy,
                 scheduler: DitlEventScheduler, stream_ids: Tuple[str, ...],
                 ttl_ns: Optional[int], deadline_ns: Optional[int]):
        if not isinstance(sample_cursor, ReplaySampleCursor):
            raise TypeError("sample_cursor must be a ReplaySampleCursor")
        if not isinstance(policy, DeterministicImpairmentPolicy):
            raise TypeError("policy must be a DeterministicImpairmentPolicy")
        if not isinstance(scheduler, DitlEventScheduler):
            raise TypeError("scheduler must be a DitlEventScheduler")
        if type(stream_ids) is not tuple or not stream_ids:
            raise ValueError("stream_ids must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in stream_ids):
            raise ValueError("stream_ids must contain non-empty strings")
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream_ids must be unique")
        if ttl_ns is not None:
            _require_nonnegative_integer("ttl_ns", ttl_ns)
        if deadline_ns is not None:
            _require_nonnegative_integer("deadline_ns", deadline_ns)
        if scheduler.ttl_ns != ttl_ns:
            raise ValueError("scheduler ttl_ns must match engine ttl_ns")

        self._cursor = sample_cursor
        self._policy = policy
        self._scheduler = scheduler
        self._stream_ids = tuple(stream_ids)
        self._stream_id_set = frozenset(stream_ids)
        self._deadline_ns = deadline_ns
        self._accepted = {}  # type: Dict[str, _AcceptedState]
        self._last_accepted_sequence = {}  # type: Dict[str, int]
        self._last_accepted_identity = {}  # type: Dict[str, str]
        self._expired_visibility = {item: False for item in stream_ids}
        self._last_consume_frame = None  # type: Optional[int]
        self._accepted_count = 0
        self._dropped_count = 0
        self._expired_count = 0

    def submit(self, submission: ReplaySubmission) -> SubmissionOutcome:
        _validate_replay_submission(submission, self._stream_id_set)
        selected = self._cursor._peek()
        logical_message_id = "message-%d" % submission.sequence_id
        identity = MessageIdentity(
            route_id=submission.route_id,
            stream_id=submission.stream_id,
            logical_message_id=logical_message_id,
            sequence_id=submission.sequence_id,
        )
        policy_preview = self._policy._preview_decision(
            identity, selected.sample.base_delay_ns
        )
        message = LogicalMessage(
            logical_message_id=logical_message_id,
            stream_id=submission.stream_id,
            source_id=submission.source_id,
            source_frame=submission.source_frame,
            source_sim_time_ns=submission.source_sim_time_ns,
            sequence_id=submission.sequence_id,
            payload=_thaw_payload(submission.payload),
            trace_sample_id=selected.sample.sample_id,
            base_delay_ns=selected.sample.base_delay_ns,
            cycle_index=selected.cycle_index,
        )
        outcome = self._scheduler.submit(message, policy_preview.decision)
        self._cursor._commit_peek()
        self._policy._commit_preview(policy_preview)
        if outcome.policy_decision.dropped:
            self._dropped_count += 1
        return outcome

    def submit_and_advance(self, submission: ReplaySubmission,
                           consume_frame: int,
                           consume_sim_time_ns: int):
        outcome = self.submit(submission)
        return outcome, self.advance(consume_frame, consume_sim_time_ns)

    def _expire_held_states(self, consume_sim_time_ns):
        expired_messages = set()
        for stream_id, state in tuple(self._accepted.items()):
            expiry = state.event.expiry_time_ns
            if expiry is not None and consume_sim_time_ns > expiry:
                del self._accepted[stream_id]
                self._expired_visibility[stream_id] = True
                # Held state represents one accepted logical message,
                # regardless of which duplicate copy arrived first.
                self._expired_count += 1
                expired_messages.add((
                    stream_id, state.event.message.logical_message_id
                ))
        return expired_messages

    def _arrival_missed(self, event):
        return (
            self._deadline_ns is not None and
            event.ready_time_ns - event.message.source_sim_time_ns >
            self._deadline_ns
        )

    def _classify_ready(self, ready, consume_sim_time_ns, expired_messages):
        statuses = [None] * len(ready)
        candidates = {}  # type: Dict[str, list]
        for index, event in enumerate(ready):
            stream_id = event.message.stream_id
            if stream_id not in self._stream_id_set:
                raise ValueError("scheduler released an unconfigured stream")
            if (event.expiry_time_ns is not None and
                    consume_sim_time_ns > event.expiry_time_ns):
                statuses[index] = "expired"
                identity = (stream_id, event.message.logical_message_id)
                last_sequence = self._last_accepted_sequence.get(stream_id)
                if (event.copy_index == 0 and
                        identity not in expired_messages and
                        (last_sequence is None or
                         event.message.sequence_id > last_sequence)):
                    self._expired_count += 1
                    expired_messages.add(identity)
                if stream_id not in self._accepted:
                    self._expired_visibility[stream_id] = True
                continue
            last_identity = self._last_accepted_identity.get(stream_id)
            if event.message.logical_message_id == last_identity:
                statuses[index] = "duplicate_discarded"
                continue
            last_sequence = self._last_accepted_sequence.get(stream_id)
            if (last_sequence is not None and
                    event.message.sequence_id <= last_sequence):
                statuses[index] = "late_superseded"
                continue
            candidates.setdefault(stream_id, []).append(index)

        fresh_streams = set()
        for stream_id, indices in candidates.items():
            winner = max(indices, key=lambda item: ready[item].message.sequence_id)
            winner_event = ready[winner]
            for index in indices:
                if index == winner:
                    statuses[index] = "accepted"
                elif (ready[index].message.logical_message_id ==
                      winner_event.message.logical_message_id):
                    statuses[index] = "duplicate_discarded"
                else:
                    statuses[index] = "late_superseded"
            self._accepted[stream_id] = _AcceptedState(
                event=winner_event,
                payload=_freeze_payload(winner_event.message.payload),
                accepted_at_ns=consume_sim_time_ns,
            )
            self._last_accepted_sequence[stream_id] = (
                winner_event.message.sequence_id
            )
            self._last_accepted_identity[stream_id] = (
                winner_event.message.logical_message_id
            )
            self._expired_visibility[stream_id] = False
            self._accepted_count += 1
            fresh_streams.add(stream_id)
        return statuses, fresh_streams

    def _stream_selection(self, stream_id, consume_sim_time_ns, fresh):
        state = self._accepted.get(stream_id)
        if state is None:
            expired = self._expired_visibility[stream_id]
            return StreamSelection(
                stream_id, None, None, None, None, None, None, None, None,
                False, not expired, expired, False,
            )
        event = state.event
        source_time = event.message.source_sim_time_ns
        return StreamSelection(
            stream_id=stream_id,
            used_message_id=event.message.logical_message_id,
            used_sequence_id=event.message.sequence_id,
            payload=state.payload,
            source_frame=event.message.source_frame,
            source_sim_time_ns=source_time,
            ready_time_ns=event.ready_time_ns,
            aoi_ns=consume_sim_time_ns - source_time,
            zoh_hold_ns=(0 if fresh else
                         consume_sim_time_ns - state.accepted_at_ns),
            is_zoh=not fresh,
            is_startup_ego_only=False,
            is_expired=False,
            fusion_deadline_miss=(
                self._deadline_ns is not None and
                consume_sim_time_ns - source_time > self._deadline_ns
            ),
        )

    def advance(self, consume_frame: int,
                consume_sim_time_ns: int) -> ReplayTick:
        _require_nonnegative_integer("consume_frame", consume_frame)
        _require_nonnegative_integer(
            "consume_sim_time_ns", consume_sim_time_ns
        )
        if (self._last_consume_frame is not None and
                consume_frame < self._last_consume_frame):
            raise ValueError("consume_frame must be non-decreasing")
        advanced = self._scheduler.advance(consume_sim_time_ns)
        self._last_consume_frame = consume_frame
        expired_messages = self._expire_held_states(consume_sim_time_ns)
        statuses, fresh_streams = self._classify_ready(
            advanced.ready, consume_sim_time_ns, expired_messages
        )
        outcomes = tuple(
            DeliveryOutcome(event, statuses[index], self._arrival_missed(event))
            for index, event in enumerate(advanced.ready)
        )
        streams = {
            stream_id: self._stream_selection(
                stream_id, consume_sim_time_ns, stream_id in fresh_streams
            )
            for stream_id in self._stream_ids
        }
        return ReplayTick(
            consume_frame=consume_frame,
            consume_sim_time_ns=consume_sim_time_ns,
            streams=streams,
            delivery_outcomes=outcomes,
            queue_depth=advanced.queue_depth,
            counters=ReplayCounters(
                self._accepted_count,
                self._dropped_count,
                self._expired_count,
            ),
        )
