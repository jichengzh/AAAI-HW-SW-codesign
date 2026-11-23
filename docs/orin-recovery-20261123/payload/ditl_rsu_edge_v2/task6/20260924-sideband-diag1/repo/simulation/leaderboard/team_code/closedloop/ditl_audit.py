"""Canonical append-only audit records for event-driven DitL replay."""
from __future__ import print_function

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Mapping

try:
    from .ditl_event_scheduler import ReplayTick, SubmissionOutcome
except ImportError:
    from ditl_event_scheduler import ReplayTick, SubmissionOutcome


AUDIT_SCHEMA_VERSION = "ditl.event-audit.v1"

_AUDIT_FIELDS = frozenset((
    "schema_version", "run_id", "route_id", "stream_id",
    "consume_frame", "consume_sim_time_ns", "submitted_message_id",
    "submitted_source_frame", "trace_sample_id", "cycle_index",
    "used_message_id", "used_source_frame", "source_sim_time_ns",
    "ready_time_ns", "base_delay_ns", "effective_delay_ns", "aoi_ns",
    "zoh_hold_ns", "is_zoh", "is_startup_ego_only", "is_expired",
    "arrival_deadline_miss", "fusion_deadline_miss", "drop_reason",
    "reorder_status", "duplicate_status", "policy_seed",
    "policy_config_hash", "rng_version", "queue_depth",
    "accepted_count", "dropped_count", "expired_count",
))
_REQUIRED_STRINGS = (
    "schema_version", "run_id", "route_id", "stream_id",
    "submitted_message_id", "trace_sample_id", "policy_config_hash",
    "rng_version", "reorder_status", "duplicate_status",
)
_NONNEGATIVE_INTEGERS = (
    "consume_frame", "consume_sim_time_ns", "submitted_source_frame",
    "cycle_index", "base_delay_ns", "queue_depth", "accepted_count",
    "dropped_count", "expired_count",
)
_OPTIONAL_NONNEGATIVE_INTEGERS = (
    "used_source_frame", "source_sim_time_ns", "ready_time_ns",
    "effective_delay_ns", "aoi_ns", "zoh_hold_ns",
)
_BOOLEANS = (
    "is_zoh", "is_startup_ego_only", "is_expired",
    "arrival_deadline_miss", "fusion_deadline_miss",
)
_REORDER_STATUSES = frozenset((
    "none", "scheduled", "delivered", "late_superseded",
    "not_applicable",
))
_DUPLICATE_STATUSES = frozenset((
    "none", "scheduled", "discarded", "not_applicable",
))
_HEX_DIGITS = frozenset("0123456789abcdef")


class DitlAuditError(IOError):
    """Raised when a validated audit record cannot be persisted."""


def _require_nonempty_string(name, value):
    if type(value) is not str:
        raise TypeError("%s must be a non-empty string" % name)
    if not value:
        raise ValueError("%s must be a non-empty string" % name)


def _require_integer(name, value, nonnegative=False):
    if type(value) is not int:
        raise TypeError("%s must be an integer" % name)
    if nonnegative and value < 0:
        raise ValueError("%s must be a non-negative integer" % name)


def _validate_hash(value):
    _require_nonempty_string("policy_config_hash", value)
    if len(value) != 64 or any(character not in _HEX_DIGITS
                               for character in value):
        raise ValueError(
            "policy_config_hash must be a lowercase SHA-256 hex digest"
        )


@dataclass(frozen=True)
class AuditContext:
    run_id: str
    route_id: str
    policy_seed: int
    policy_config_hash: str
    rng_version: str

    def __post_init__(self):
        _require_nonempty_string("run_id", self.run_id)
        _require_nonempty_string("route_id", self.route_id)
        _require_integer("policy_seed", self.policy_seed)
        _validate_hash(self.policy_config_hash)
        _require_nonempty_string("rng_version", self.rng_version)


def _validate_audit_values(values):
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    copied = dict(values)
    if set(copied) != _AUDIT_FIELDS:
        missing = sorted(_AUDIT_FIELDS - set(copied))
        extra = sorted(set(copied) - _AUDIT_FIELDS)
        raise ValueError(
            "audit fields must match schema; missing=%r extra=%r" %
            (missing, extra)
        )
    for name in _REQUIRED_STRINGS:
        _require_nonempty_string(name, copied[name])
    if copied["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise ValueError("schema_version is unsupported")
    _validate_hash(copied["policy_config_hash"])
    _require_integer("policy_seed", copied["policy_seed"])
    for name in _NONNEGATIVE_INTEGERS:
        _require_integer(name, copied[name], nonnegative=True)
    for name in _OPTIONAL_NONNEGATIVE_INTEGERS:
        if copied[name] is not None:
            _require_integer(name, copied[name], nonnegative=True)
    for name in _BOOLEANS:
        if type(copied[name]) is not bool:
            raise TypeError("%s must be a boolean" % name)
    for name in ("used_message_id", "drop_reason"):
        if copied[name] is not None:
            _require_nonempty_string(name, copied[name])
    if copied["reorder_status"] not in _REORDER_STATUSES:
        raise ValueError("reorder_status is invalid")
    if copied["duplicate_status"] not in _DUPLICATE_STATUSES:
        raise ValueError("duplicate_status is invalid")
    for name, value in copied.items():
        if type(value) is float and not math.isfinite(value):
            raise ValueError("%s must be finite" % name)
    return MappingProxyType(copied)


@dataclass(frozen=True)
class DitlAuditRecord:
    values: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "values", _validate_audit_values(self.values))


def _stream_outcomes(tick, stream_id):
    """Return outcomes that occurred for this stream on this fusion tick."""
    return tuple(
        outcome for outcome in tick.delivery_outcomes
        if outcome.event.message.stream_id == stream_id
    )


def _reorder_status(submission, outcomes):
    if any(outcome.status == "late_superseded" for outcome in outcomes):
        return "late_superseded"
    reordered_outcomes = tuple(
        outcome for outcome in outcomes
        if outcome.event.reorder_delay_ns > 0
    )
    if reordered_outcomes:
        return "delivered"
    if submission.policy_decision.dropped:
        return "not_applicable"
    if any(event.reorder_delay_ns > 0 for event in submission.enqueued):
        return "scheduled"
    return "none"


def _duplicate_status(submission, outcomes):
    if any(outcome.status == "duplicate_discarded" for outcome in outcomes):
        return "discarded"
    if submission.policy_decision.dropped:
        return "not_applicable"
    if len(submission.enqueued) > 1:
        return "scheduled"
    return "none"


def build_audit_record(tick: ReplayTick, submission: SubmissionOutcome,
                       stream_id: str,
                       context: AuditContext) -> DitlAuditRecord:
    """Build one complete immutable record for a stream's fusion tick."""
    if type(tick) is not ReplayTick:
        raise TypeError("tick must be a ReplayTick")
    if type(submission) is not SubmissionOutcome:
        raise TypeError("submission must be a SubmissionOutcome")
    if type(context) is not AuditContext:
        raise TypeError("context must be an AuditContext")
    _require_nonempty_string("stream_id", stream_id)
    if stream_id not in tick.streams:
        raise ValueError("stream_id is absent from the replay tick")
    if submission.message.stream_id != stream_id:
        raise ValueError("submission stream_id does not match audit stream")

    selected = tick.streams[stream_id]
    outcomes = _stream_outcomes(tick, stream_id)
    effective_delay_ns = (
        None if not submission.enqueued else
        submission.enqueued[0].effective_delay_ns
    )
    values = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": context.run_id,
        "route_id": context.route_id,
        "stream_id": stream_id,
        "consume_frame": tick.consume_frame,
        "consume_sim_time_ns": tick.consume_sim_time_ns,
        "submitted_message_id": submission.message.logical_message_id,
        "submitted_source_frame": submission.message.source_frame,
        "trace_sample_id": submission.message.trace_sample_id,
        "cycle_index": submission.message.cycle_index,
        "used_message_id": selected.used_message_id,
        "used_source_frame": selected.source_frame,
        "source_sim_time_ns": selected.source_sim_time_ns,
        "ready_time_ns": selected.ready_time_ns,
        "base_delay_ns": submission.message.base_delay_ns,
        "effective_delay_ns": effective_delay_ns,
        "aoi_ns": selected.aoi_ns,
        "zoh_hold_ns": selected.zoh_hold_ns,
        "is_zoh": selected.is_zoh,
        "is_startup_ego_only": selected.is_startup_ego_only,
        "is_expired": selected.is_expired,
        "arrival_deadline_miss": any(
            outcome.arrival_deadline_miss for outcome in outcomes
        ),
        "fusion_deadline_miss": selected.fusion_deadline_miss,
        "drop_reason": submission.policy_decision.drop_reason,
        "reorder_status": _reorder_status(submission, outcomes),
        "duplicate_status": _duplicate_status(submission, outcomes),
        "policy_seed": context.policy_seed,
        "policy_config_hash": context.policy_config_hash,
        "rng_version": context.rng_version,
        "queue_depth": tick.queue_depth,
        "accepted_count": tick.counters.accepted_count,
        "dropped_count": tick.counters.dropped_count,
        "expired_count": tick.counters.expired_count,
    }
    return DitlAuditRecord(values=values)


def _canonical_line(record):
    return (json.dumps(
        dict(record.values), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")


class DitlAuditWriter(object):
    """Bounded append-only writer for canonical DitL JSONL records."""

    def __init__(self, path: Path, max_buffer_records: int = 1,
                 exclusive: bool = False):
        if type(max_buffer_records) is not int:
            raise TypeError("max_buffer_records must be a positive integer")
        if max_buffer_records <= 0:
            raise ValueError("max_buffer_records must be a positive integer")
        if type(exclusive) is not bool:
            raise TypeError("exclusive must be a boolean")
        try:
            self.path = Path(path)
        except (TypeError, ValueError) as exc:
            raise DitlAuditError("invalid audit output path: %r: %s" %
                                 (path, exc))
        self._path_text = str(self.path)
        self._max_buffer_records = max_buffer_records
        self._exclusive = exclusive
        self._buffer = ()
        self._fd = -1
        self._closed = False
        self._failure = None
        if self.path.name == "latency_align_audit.csv":
            raise DitlAuditError(
                "refusing to use legacy latency audit path %s" %
                self._path_text
            )
        try:
            self._fd = self._open_secure()
        except DitlAuditError:
            raise
        except (OSError, ValueError) as exc:
            raise DitlAuditError(
                "cannot open audit output %s: %s" %
                (self._path_text, exc)
            )

    def _open_secure(self):
        original_parts = self.path.parts
        if ".." in original_parts:
            raise DitlAuditError(
                "audit output path cannot contain '..': %s" %
                self._path_text
            )
        absolute = Path(os.path.abspath(str(self.path)))
        components = absolute.parts[1:]
        if not components:
            raise DitlAuditError("invalid audit output path %s" %
                                 self._path_text)
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        directory_fd = os.open("/", directory_flags)
        try:
            for component in components[:-1]:
                next_fd = os.open(
                    component, directory_flags, dir_fd=directory_fd
                )
                os.close(directory_fd)
                directory_fd = next_fd
                if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                    raise OSError("path component is not a directory")
            return self._open_final(directory_fd, components[-1])
        finally:
            os.close(directory_fd)

    def _open_final(self, directory_fd, filename):
        try:
            prior = os.stat(filename, dir_fd=directory_fd,
                            follow_symlinks=False)
        except FileNotFoundError:
            prior = None
        if prior is not None and not stat.S_ISREG(prior.st_mode):
            raise OSError("audit output is not a regular file")
        flags = (
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC |
            os.O_NOFOLLOW | os.O_NONBLOCK
        )
        if self._exclusive:
            flags |= os.O_EXCL
        fd = os.open(filename, flags, 0o600, dir_fd=directory_fd)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("audit output is not a regular file")
            current = os.stat(filename, dir_fd=directory_fd,
                              follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (opened.st_dev,
                                                    opened.st_ino):
                raise OSError("audit output changed while opening")
            if opened.st_size:
                self._require_complete_existing_file(
                    directory_fd, filename, opened
                )
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _require_complete_existing_file(directory_fd, filename, opened):
        read_fd = os.open(
            filename, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW |
            os.O_NONBLOCK, dir_fd=directory_fd,
        )
        try:
            readable = os.fstat(read_fd)
            if (readable.st_dev, readable.st_ino) != (opened.st_dev,
                                                      opened.st_ino):
                raise OSError("audit output changed while validating")
            if os.pread(read_fd, 1, opened.st_size - 1) != b"\n":
                raise OSError(
                    "existing audit output has a partial final record"
                )
        finally:
            os.close(read_fd)

    def _remember_failure(self, action, exc):
        if self._failure is None:
            self._failure = DitlAuditError(
                "%s failed for audit output %s: %s" %
                (action, self._path_text, exc)
            )
        return self._failure

    def _require_writable(self):
        if self._failure is not None:
            raise self._failure
        if self._closed or self._fd < 0:
            raise DitlAuditError(
                "audit output is closed: %s" % self._path_text
            )

    def _write_all(self, payload):
        remaining = memoryview(payload)
        while remaining:
            written = os.write(self._fd, remaining)
            if written <= 0:
                raise OSError("write made no progress")
            remaining = remaining[written:]

    def _flush_buffer(self):
        if not self._buffer:
            return
        payload = b"".join(self._buffer)
        try:
            self._write_all(payload)
        except OSError as exc:
            raise self._remember_failure("write", exc)
        self._buffer = ()

    def append(self, record: DitlAuditRecord) -> None:
        self._require_writable()
        if type(record) is not DitlAuditRecord:
            raise TypeError("record must be a DitlAuditRecord")
        try:
            line = _canonical_line(record)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DitlAuditError(
                "serialization failed for audit output %s: %s" %
                (self._path_text, exc)
            )
        self._buffer = self._buffer + (line,)
        if len(self._buffer) >= self._max_buffer_records:
            self._flush_buffer()

    def close(self) -> None:
        if self._closed:
            if self._failure is not None:
                raise self._failure
            return
        try:
            if self._failure is None:
                self._flush_buffer()
                os.fsync(self._fd)
        except DitlAuditError:
            pass
        except OSError as exc:
            self._remember_failure("fsync", exc)
        finally:
            if self._fd >= 0:
                try:
                    os.close(self._fd)
                except OSError as exc:
                    self._remember_failure("close", exc)
                self._fd = -1
            self._closed = True
        if self._failure is not None:
            raise self._failure
