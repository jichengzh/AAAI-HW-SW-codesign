#!/usr/bin/env python3
"""CPU-only deterministic dry replay for one verified DitL feature bundle."""
from __future__ import print_function

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from ditl_audit import AuditContext, DitlAuditWriter, build_audit_record
from ditl_event_scheduler import (
    DitlEventScheduler, DitlReplayEngine, ReplaySubmission,
)
from ditl_impairment_policy import DeterministicImpairmentPolicy, RNG_VERSION
from ditl_replay_config import resolve_ditl_replay_config
from ditl_trace_library import ReplaySampleCursor, verify_feature_bundle


SUMMARY_SCHEMA_VERSION = "ditl.dry-replay-summary.v2"
MAX_POLICY_BYTES = 1024 * 1024
MAX_AUDIT_BYTES = 512 * 1024 * 1024
MAX_FRAMES = 100000
MAX_JSON_DEPTH = 64
MAX_QUEUE_EVENTS = 1024
_POLICY_FIELDS = frozenset((
    "extra_delay_ms", "jitter_ms", "drop_rate", "burst",
    "reorder_rate", "reorder_extra_ms", "duplicate_rate",
))
_BURST_FIELDS = frozenset(("enabled", "enter_rate", "exit_rate"))


@dataclass(frozen=True)
class ReplayHarness:
    engine: DitlReplayEngine
    bundle: object
    policy: DeterministicImpairmentPolicy
    ttl_ns: Optional[int]
    deadline_ns: Optional[int]


def _positive_integer(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a positive integer")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _frame_count(value):
    parsed = _positive_integer(value)
    if parsed > MAX_FRAMES:
        raise argparse.ArgumentTypeError(
            "frames must not exceed %d" % MAX_FRAMES
        )
    return parsed


def _nonnegative_number(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return parsed


def _nonempty(value):
    if not isinstance(value, str) or not value:
        raise argparse.ArgumentTypeError("value must be a non-empty string")
    return value


def add_verified_bundle_arguments(parser):
    parser.add_argument("--library-root", required=True)
    parser.add_argument("--catalog", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--bundle")
    choice.add_argument("--profile", type=_nonempty)


def add_replay_arguments(parser):
    parser.add_argument("--frames", required=True, type=_frame_count)
    parser.add_argument("--tick-ms", required=True, type=_positive_integer)
    parser.add_argument("--route-id", required=True, type=_nonempty)
    parser.add_argument("--stream-id", required=True, type=_nonempty)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exhaustion", required=True,
                        choices=("error", "cycle"))
    parser.add_argument("--ttl-ms", type=_nonnegative_number)
    parser.add_argument("--deadline-ms", type=_nonnegative_number)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--summary-out", required=True)


def build_parser():
    parser = argparse.ArgumentParser(description="CPU-only DitL replay")
    add_verified_bundle_arguments(parser)
    add_replay_arguments(parser)
    return parser


def _absolute_path(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty absolute path" % name)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("%s must be an absolute path without '..'" % name)
    return path


def _read_regular_nofollow(path, maximum_bytes, name):
    path = _absolute_path(str(path), name)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open("/", directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(path.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                     dir_fd=directory_fd)
        try:
            details = os.fstat(fd)
            if not stat.S_ISREG(details.st_mode):
                raise ValueError("%s must be a regular file" % name)
            if details.st_size > maximum_bytes:
                raise ValueError("%s exceeds the byte limit" % name)
            chunks = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > maximum_bytes:
                raise ValueError("%s exceeds the byte limit" % name)
            return data
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)


def _strict_json(data, name):
    def pairs(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("%s contains duplicate fields" % name)
        return result

    def invalid_constant(value):
        raise ValueError("%s contains non-finite number %s" % (name, value))

    try:
        text = data.decode("utf-8")
        depth = 0
        quoted = False
        escaped = False
        for character in text:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "[{":
                depth += 1
                if depth > MAX_JSON_DEPTH:
                    raise ValueError("%s exceeds the JSON depth limit" % name)
            elif character in "]}":
                depth -= 1
        return json.loads(text, object_pairs_hook=pairs,
                          parse_constant=invalid_constant)
    except UnicodeDecodeError as exc:
        raise ValueError("%s must be UTF-8: %s" % (name, exc))


def load_canonical_policy(path):
    data = _read_regular_nofollow(path, MAX_POLICY_BYTES, "policy-json")
    value = _strict_json(data, "policy-json")
    if not isinstance(value, dict):
        raise ValueError("policy-json must contain an object")
    unknown = sorted(set(value) - _POLICY_FIELDS)
    missing = sorted(_POLICY_FIELDS - set(value))
    if unknown:
        raise ValueError("policy-json contains unknown field %s" % unknown[0])
    if missing:
        raise ValueError("policy-json is missing field %s" % missing[0])
    if not isinstance(value["burst"], dict):
        raise ValueError("policy-json burst must be an object")
    burst_unknown = sorted(set(value["burst"]) - _BURST_FIELDS)
    burst_missing = sorted(_BURST_FIELDS - set(value["burst"]))
    if burst_unknown:
        raise ValueError("policy-json burst contains unknown field %s" %
                         burst_unknown[0])
    if burst_missing:
        raise ValueError("policy-json burst is missing field %s" %
                         burst_missing[0])
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    if data != canonical:
        raise ValueError("policy-json must use canonical JSON encoding")
    return value


def _bundle_for_profile(catalog_path, library_root, profile_id):
    data = _read_regular_nofollow(catalog_path, 16 * 1024 * 1024, "catalog")
    catalog = _strict_json(data, "catalog")
    if not isinstance(catalog, dict) or set(catalog) != {
            "library_schema_version", "features"}:
        raise ValueError("catalog has invalid fields")
    features = catalog.get("features")
    if not isinstance(features, dict):
        raise ValueError("catalog features must be an object")
    matches = []
    for feature_id, entry in features.items():
        if isinstance(entry, dict) and entry.get("profile_id") == profile_id:
            matches.append((feature_id, entry))
    if len(matches) != 1:
        raise ValueError("profile must resolve to exactly one registered bundle")
    feature_id, entry = matches[0]
    expected_relative = "v1/bundles/" + feature_id
    if entry.get("relative_path") != expected_relative:
        raise ValueError("catalog profile entry has a noncanonical path")
    return library_root / "v1" / "bundles" / feature_id


def load_verified_bundle_from_args(args):
    root = _absolute_path(args.library_root, "library-root")
    catalog = _absolute_path(args.catalog, "catalog")
    if catalog != root / "catalog.json":
        raise ValueError("catalog must equal library-root/catalog.json")
    if args.bundle is not None:
        bundle_path = _absolute_path(args.bundle, "bundle")
    else:
        bundle_path = _bundle_for_profile(catalog, root, args.profile)
    return verify_feature_bundle(bundle_path, root, catalog)


def build_replay_engine(bundle, args):
    policy_value = load_canonical_policy(
        _absolute_path(args.policy_json, "policy-json")
    )
    config = resolve_ditl_replay_config({"ditl_replay": {
        "library_root": _absolute_path(args.library_root, "library-root"),
        "catalog": _absolute_path(args.catalog, "catalog"),
        "bundle": bundle.bundle_path,
        "exhaustion": args.exhaustion,
        "seed": args.seed,
        "ttl_ms": args.ttl_ms,
        "deadline_ms": args.deadline_ms,
        "max_queue_events": MAX_QUEUE_EVENTS,
        "impairment": policy_value,
    }})
    policy = DeterministicImpairmentPolicy(config.seed, config.impairment)
    engine = DitlReplayEngine(
        ReplaySampleCursor(bundle.samples, config.exhaustion), policy,
        DitlEventScheduler(config.max_queue_events, config.ttl_ns),
        (args.stream_id,), config.ttl_ns, config.deadline_ns,
    )
    return ReplayHarness(
        engine=engine, bundle=bundle, policy=policy,
        ttl_ns=config.ttl_ns, deadline_ns=config.deadline_ns,
    )


def _nearest_rank(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(math.ceil(probability * len(ordered))) - 1)
    return ordered[index]


def _percentiles(values):
    return {
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
    }


def _value_counts(values):
    counts = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def recompute_audit_metrics(records):
    records = list(records)
    delays = [record["base_delay_ns"] for record in records]
    effective_delays = [record["effective_delay_ns"] for record in records
                        if record["effective_delay_ns"] is not None]
    aois = [record["aoi_ns"] for record in records
            if record["aoi_ns"] is not None]
    final = records[-1] if records else {}
    return {
        "accepted_count": final.get("accepted_count", 0),
        "dropped_count": final.get("dropped_count", 0),
        "expired_count": final.get("expired_count", 0),
        "max_queue_depth": max(
            (record["queue_depth"] for record in records), default=0),
        "cycles": max(
            (record["cycle_index"] for record in records), default=-1) + 1,
        "delay_percentiles_ns": _percentiles(delays),
        "effective_delay_percentiles_ns": _percentiles(effective_delays),
        "aoi_percentiles_ns": _percentiles(aois),
        "drop_reason_counts": _value_counts(
            record["drop_reason"] for record in records
        ),
        "reorder_status_counts": _value_counts(
            record["reorder_status"] for record in records
        ),
        "duplicate_status_counts": _value_counts(
            record["duplicate_status"] for record in records
        ),
        "arrival_deadline_miss_count": sum(
            bool(record["arrival_deadline_miss"]) for record in records
        ),
        "fusion_deadline_miss_count": sum(
            bool(record["fusion_deadline_miss"]) for record in records
        ),
        "expired_visible_count": sum(
            bool(record["is_expired"]) for record in records
        ),
    }


def _require_new_output(path, name):
    path = _absolute_path(path, name)
    if not path.parent.is_dir():
        raise ValueError("%s parent directory does not exist" % name)
    try:
        os.lstat(str(path))
    except FileNotFoundError:
        return path
    raise ValueError("%s already exists; refusing to append or replace" % name)


def _write_new_canonical_json(path, value):
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    parent_fd = os.open("/", directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                 os.O_NOFOLLOW)
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        try:
            remaining = memoryview(payload)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise IOError("summary write made no progress")
                remaining = remaining[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _validate_simulation_arguments(args):
    if type(args.frames) is not int:
        raise TypeError("frames must be an integer")
    if args.frames <= 0 or args.frames > MAX_FRAMES:
        raise ValueError("frames must be in [1, %d]" % MAX_FRAMES)
    if type(args.tick_ms) is not int:
        raise TypeError("tick-ms must be an integer")
    if args.tick_ms <= 0:
        raise ValueError("tick-ms must be a positive integer")


def simulate_integer_ticks(harness, args):
    _validate_simulation_arguments(args)
    audit_path = _require_new_output(args.audit_out, "audit-out")
    summary_path = _require_new_output(args.summary_out, "summary-out")
    if audit_path == summary_path:
        raise ValueError("audit-out and summary-out must be different")
    tick_ns = args.tick_ms * 1000 * 1000
    context = AuditContext(
        run_id="dry-replay-v1", route_id=args.route_id,
        policy_seed=args.seed, policy_config_hash=harness.policy.config_hash,
        rng_version=RNG_VERSION,
    )
    records = []
    writer = DitlAuditWriter(
        audit_path, max_buffer_records=64, exclusive=True
    )
    try:
        for frame in range(args.frames):
            source_time_ns = frame * tick_ns
            submission = harness.engine.submit(ReplaySubmission(
                route_id=args.route_id, stream_id=args.stream_id,
                source_id=args.stream_id, source_frame=frame,
                source_sim_time_ns=source_time_ns, sequence_id=frame,
                payload={"source_frame": frame, "stream_id": args.stream_id},
            ))
            tick = harness.engine.advance(frame, source_time_ns)
            record = build_audit_record(
                tick, submission, args.stream_id, context
            )
            writer.append(record)
            records.append(dict(record.values))
    finally:
        writer.close()

    audit_bytes = _read_regular_nofollow(
        audit_path, MAX_AUDIT_BYTES, "audit-out"
    )
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    metrics = recompute_audit_metrics(records)
    summary = dict(metrics, **{
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "feature_id": harness.bundle.feature_id,
        "profile_id": harness.bundle.profile_id,
        "route_id": args.route_id,
        "stream_id": args.stream_id,
        "frames": args.frames,
        "tick_ns": tick_ns,
        "seed": args.seed,
        "exhaustion": args.exhaustion,
        "policy_config_hash": harness.policy.config_hash,
        "rng_version": RNG_VERSION,
        "max_queue_events": MAX_QUEUE_EVENTS,
        "ttl_ns": harness.ttl_ns,
        "deadline_ns": harness.deadline_ns,
        "audit_sha256": audit_sha256,
    })
    _write_new_canonical_json(summary_path, summary)
    return summary


def run_dry_replay(args):
    bundle = load_verified_bundle_from_args(args)
    harness = build_replay_engine(bundle, args)
    return simulate_integer_ticks(harness, args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_dry_replay(args)
    except (IOError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
