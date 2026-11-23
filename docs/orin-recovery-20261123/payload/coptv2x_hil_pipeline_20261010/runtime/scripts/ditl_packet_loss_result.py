#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate and summarize packet-loss stage-one evidence."""

from __future__ import print_function

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

from simulation.leaderboard.team_code.closedloop.ditl_audit import DitlAuditRecord
from scripts.ditl_packet_loss_manifest import build_batches, validate_stage1_manifest
from scripts.dual_fifo_result import TERMINAL_DRIVING_STATUSES, parse_route_result


EVIDENCE_SCHEMA = "ditl.packet-loss-stage1-task-evidence.v1"
STATUS_SCHEMA = "ditl.packet-loss-stage1-task-status.v1"
IDENTITY_SCHEMA = "ditl.packet-loss-stage1-command-identity.v1"
ARTIFACT_NAMES = ("result.json", "event_audit.jsonl", "status.json",
                  "command_identity.json")
IDENTITY_FIELDS = (
    "key", "arm", "drop_rate", "subset", "scenario", "profile", "route",
    "carla_seed", "traffic_seed", "replay_seed", "repeat", "skip_frames",
)
UNIT_IDENTITY_FIELDS = (
    "scenario", "route", "carla_seed", "traffic_seed", "replay_seed",
    "repeat",
)


def _pairs(rows):
    value = {}
    for key, item in rows:
        if key in value:
            raise ValueError("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _constant(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_pairs, parse_constant=_constant)


def resolve_ego_result(checkpoint):
    """Return the direct ego-0 result child when it is terminal and valid."""
    checkpoint = Path(checkpoint)
    ego_directory = checkpoint.parent / "ego_vehicle_0"
    candidate = ego_directory / checkpoint.name
    if ego_directory.is_symlink() or candidate.is_symlink() or \
            not candidate.is_file():
        raise ValueError("direct ego result is missing or symlinked")
    document = _read_json(candidate)
    if not isinstance(document, dict):
        raise ValueError("ego result must be a JSON object")
    checkpoint_data = document.get("_checkpoint")
    if not isinstance(checkpoint_data, dict):
        raise ValueError("ego result checkpoint is missing or malformed")
    global_record = checkpoint_data.get("global_record")
    if not isinstance(global_record, dict):
        raise ValueError("ego result global record is missing or malformed")
    status = global_record.get("status")
    if not isinstance(status, str) or status not in TERMINAL_DRIVING_STATUSES:
        raise ValueError("ego result lacks a terminal status")
    return candidate


def _read_jsonl(path):
    content = Path(path).read_text(encoding="utf-8")
    if not content or not content.endswith("\n"):
        raise ValueError("event audit is empty or truncated")
    values = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line:
            raise ValueError("blank event audit line %d" % number)
        row = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
        if not isinstance(row, dict):
            raise ValueError("event audit line must be an object")
        values.append(row)
    return values


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % label)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % label)
    return value


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return value


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _infer_cohort(attempt_dir):
    resolved = Path(attempt_dir).resolve()
    for parent in resolved.parents:
        if parent.name == "attempts":
            return parent.parent
    return None


def _artifact_path(attempt_dir, descriptor, filename):
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("invalid artifact descriptor for %s" % filename)
    relative = descriptor["path"]
    if not isinstance(relative, str) or Path(relative).is_absolute() or \
            Path(relative).parts != (filename,):
        raise ValueError("artifact path mismatch for %s" % filename)
    path = attempt_dir / relative
    if path.is_symlink() or not path.is_file() or not _inside(path, attempt_dir):
        raise ValueError("artifact is missing or external: %s" % filename)
    digest = descriptor["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or _sha(path) != digest:
        raise ValueError("artifact hash mismatch for %s" % filename)
    return path


def _expected_identity(task):
    return {field: task[field] for field in IDENTITY_FIELDS}


def _validate_execution_identity(task, evidence, status, identity):
    if evidence.get("task_identity") != _expected_identity(task):
        raise ValueError("task identity mismatch")
    attempt = evidence.get("attempt")
    if attempt not in (1, 2):
        raise ValueError("attempt must be one or two")
    config_sha = evidence.get("config_sha256")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise ValueError("invalid config hash")
    policy_sha = evidence.get("policy_config_hash")
    if not isinstance(policy_sha, str) or len(policy_sha) != 64:
        raise ValueError("invalid policy config hash")
    command = evidence.get("command_argv")
    ports = evidence.get("ports")
    if not isinstance(command, list) or not command or not all(
            isinstance(item, str) for item in command):
        raise ValueError("invalid command identity")
    if not isinstance(ports, list) or len(ports) != 4 or not all(
            isinstance(port, int) and not isinstance(port, bool) for port in ports):
        raise ValueError("invalid port identity")
    shared = {
        "task_key": task["key"], "attempt": attempt,
        "worker": evidence.get("worker"), "gpu": evidence.get("gpu"),
        "ports": ports, "command_argv": command,
    }
    if identity.get("schema") != IDENTITY_SCHEMA or any(
            identity.get(key) != value for key, value in shared.items()):
        raise ValueError("command identity mismatch")
    expected_status = dict(shared)
    expected_status.update({
        "arm": task["arm"], "scenario": task["scenario"],
        "route": task["route"], "carla_seed": task["carla_seed"],
        "traffic_seed": task["traffic_seed"],
        "replay_seed": task["replay_seed"], "repeat": task["repeat"],
        "config_sha256": config_sha,
    })
    if status.get("schema") != STATUS_SCHEMA or any(
            status.get(key) != value for key, value in expected_status.items()):
        raise ValueError("status identity mismatch")
    if evidence.get("gpu") not in (6, 7):
        raise ValueError("GPU identity is outside the stage-one allocation")
    return attempt, config_sha, policy_sha


def _event_metrics(records, task, policy_sha):
    streams = {}
    cold_start = 0
    for record in records:
        record = DitlAuditRecord(record).values
        if record.get("route_id") != "town05_short_r%d" % task["route"]:
            raise ValueError("event audit route identity mismatch")
        if record.get("policy_seed") != task["replay_seed"] or \
                record.get("policy_config_hash") != policy_sha:
            raise ValueError("event audit policy identity mismatch")
        stream_id = record.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("event audit stream identity is missing")
        accepted = _nonnegative_int(record.get("accepted_count"), "accepted_count")
        dropped = _nonnegative_int(record.get("dropped_count"), "dropped_count")
        state = streams.setdefault(stream_id, {
            "accepted": 0, "dropped": 0, "started": False, "aoi_ms": [],
            "zoh": 0, "observations": 0,
        })
        if accepted < state["accepted"] or dropped < state["dropped"]:
            raise ValueError("event audit counters are not monotonic")
        state["accepted"], state["dropped"] = accepted, dropped
        startup = record.get("is_startup_ego_only")
        zoh = record.get("is_zoh")
        if type(startup) is not bool or type(zoh) is not bool:
            raise ValueError("event audit startup/ZOH flags must be boolean")
        aoi = record.get("aoi_ns")
        if startup or aoi is None:
            if not state["started"]:
                cold_start += 1
            continue
        state["started"] = True
        aoi_ms = _finite(aoi, "aoi_ns") / 1000000.0
        if aoi_ms < 0:
            raise ValueError("AoI must be non-negative")
        state["aoi_ms"].append(aoi_ms)
        state["observations"] += 1
        state["zoh"] += int(zoh)
    if not streams:
        raise ValueError("event audit has no streams")
    accepted = sum(row["accepted"] for row in streams.values())
    dropped = sum(row["dropped"] for row in streams.values())
    aoi_values = [value for row in streams.values() for value in row["aoi_ms"]]
    observations = len(aoi_values)
    if accepted + dropped <= 0 or observations <= 0:
        raise ValueError("event audit lacks final counters or warm observations")
    return {
        "accepted_count": accepted,
        "dropped_count": dropped,
        "actual_drop_rate": float(dropped) / (accepted + dropped),
        "aoi_samples_ms": aoi_values,
        "mean_aoi_ms": sum(aoi_values) / observations,
        "severe_stale_rate": float(sum(value >= 600.0 for value in aoi_values)) / observations,
        "zoh_rate": float(sum(row["zoh"] for row in streams.values())) / observations,
        "cold_start_count": cold_start,
        "observation_count": observations,
        "stream_count": len(streams),
    }


def _has_collision(result_path):
    document = _read_json(result_path)
    infractions = document["_checkpoint"]["global_record"].get("infractions", {})
    if not isinstance(infractions, dict):
        raise ValueError("route result infractions must be a mapping")
    for name, value in infractions.items():
        if not isinstance(name, str) or not name.startswith("collisions_"):
            continue
        if isinstance(value, list):
            if value:
                return True
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if _finite(value, name) > 0:
                return True
        elif value is not None:
            raise ValueError("collision infraction has an invalid value")
    return False


def _infra(task, attempt_dir, reason, attempt=None, cohort=None):
    return {
        "task_key": task["key"], "arm": task["arm"],
        "scenario": task["scenario"], "route": task["route"],
        "attempt": attempt, "classification": "infrastructure_failure",
        "failure_reason": str(reason), "evidence_dir": str(attempt_dir.resolve()),
        "cohort_path": None if cohort is None else str(cohort),
        "DS": None, "RC": None, "collisions": None, "route_completed": None,
        "event_metrics": None, "provenance": {},
    }


def _reuse_provenance(evidence):
    payload = evidence.get("reuse_provenance")
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("schema") != \
            "ditl.packet-loss-stage1-reuse-provenance.v1":
        raise ValueError("reuse provenance schema mismatch")
    if payload.get("source_kind") not in ("canonical", "legacy_r1"):
        raise ValueError("reuse provenance source kind is invalid")
    for field in ("source_kind", "source_cohort", "source_attempt_dir"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError("reuse provenance is incomplete")
    hashes = payload.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("reuse provenance source hashes are missing")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or \
                len(digest) != 64:
            raise ValueError("reuse provenance source hash is invalid")
        try:
            int(digest, 16)
        except ValueError:
            raise ValueError("reuse provenance source hash is invalid")
    return copy.deepcopy(payload)


def classify_task_evidence(task, attempt_dir):
    """Classify one route independently from other routes in its batch."""
    attempt_dir = Path(attempt_dir)
    evidence_path = attempt_dir / "evidence.json"
    attempt = None
    cohort = _infer_cohort(attempt_dir)
    try:
        if attempt_dir.is_symlink() or not attempt_dir.is_dir() or \
                evidence_path.is_symlink() or not evidence_path.is_file():
            raise ValueError("attempt evidence is missing")
        evidence = _read_json(evidence_path)
        if not isinstance(evidence, dict) or evidence.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError("evidence schema mismatch")
        attempt = evidence.get("attempt")
        claimed_cohort = Path(evidence.get("cohort_path", "")).resolve()
        if cohort is not None and claimed_cohort != cohort:
            raise ValueError("evidence cohort identity mismatch")
        cohort = claimed_cohort
        if not cohort.is_absolute() or not _inside(attempt_dir, cohort):
            raise ValueError("evidence source is outside its cohort")
        artifacts = evidence.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
            raise ValueError("evidence artifact set is incomplete")
        paths = {name: _artifact_path(attempt_dir, artifacts[name], name)
                 for name in ARTIFACT_NAMES}
        status = _read_json(paths["status.json"])
        identity = _read_json(paths["command_identity.json"])
        attempt, config_sha, policy_sha = _validate_execution_identity(
            task, evidence, status, identity)
        if status.get("returncode") != 0 or status.get("state") != "completed":
            raise ValueError("evaluator or agent execution failed")
        parsed = parse_route_result(str(paths["result.json"]))
        if not parsed.get("countable"):
            raise ValueError(parsed.get("error", "route result is not countable"))
        events = _event_metrics(_read_jsonl(paths["event_audit.jsonl"]),
                                task, policy_sha)
        completed = parsed["status"] == "Completed"
        provenance = {
            "worker": evidence["worker"], "gpu": evidence["gpu"],
            "ports": list(evidence["ports"]),
            "config_sha256": config_sha,
            "policy_config_hash": policy_sha,
            "artifact_sha256": {
                name: artifacts[name]["sha256"] for name in ARTIFACT_NAMES
            },
        }
        reuse = _reuse_provenance(evidence)
        if reuse is not None:
            provenance["reuse_provenance"] = reuse
        return {
            "task_key": task["key"], "arm": task["arm"],
            "scenario": task["scenario"], "route": task["route"],
            "attempt": attempt, "classification": "countable",
            "failure_reason": None, "evidence_dir": str(attempt_dir.resolve()),
            "cohort_path": str(cohort), "DS": float(parsed["DS"]),
            "RC": float(parsed["RC"]),
            "collisions": _has_collision(paths["result.json"]),
            "route_completed": completed, "event_metrics": events,
            "provenance": provenance,
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return _infra(task, attempt_dir, error, attempt=attempt, cohort=cohort)


def _manifest_task_map(manifest):
    validated = validate_stage1_manifest(manifest)
    return validated, {task["key"]: task for task in validated["tasks"]}


def plan_retry_task_keys(manifest, first_attempt):
    """Return first-pass infrastructure failures in canonical task order."""
    validated, task_map = _manifest_task_map(manifest)
    unknown = set(first_attempt).difference(task_map)
    if unknown:
        raise ValueError("first-attempt mapping contains an unknown task")
    result = []
    for task in validated["tasks"]:
        row = first_attempt.get(task["key"])
        if row is not None and row.get("classification") == "infrastructure_failure":
            result.append(task["key"])
    # Reuse the Task 1 batch builder as the canonical grouping/order oracle.
    flattened = tuple(
        key for batch in build_batches(validated, result)
        for key in batch["task_keys"]
    )
    if set(flattened) != set(result):
        raise ValueError("retry tasks do not form canonical manifest batches")
    return flattened


def finalize_task_results(manifest, first_attempt, retry_attempt):
    """Apply the strict one-retry then zero-DS policy per route."""
    validated, task_map = _manifest_task_map(manifest)
    known = set(task_map)
    if set(first_attempt).difference(known) or set(retry_attempt).difference(known):
        raise ValueError("attempt mapping contains an unknown task")
    cohorts = {row.get("cohort_path") for row in first_attempt.values()
               if row.get("cohort_path") is not None}
    if len(cohorts) > 1:
        raise ValueError("first-pass evidence must come from a single cohort")
    for row in first_attempt.values():
        if row.get("cohort_path") is None or not _inside(
                row.get("evidence_dir", ""), row["cohort_path"]):
            raise ValueError("first-pass evidence is external to the current cohort")
    retry_keys = set(plan_retry_task_keys(validated, first_attempt))
    if set(retry_attempt) != retry_keys:
        raise ValueError("retry evidence must exactly match first-pass infrastructure failures")
    finalized = {}
    for task in validated["tasks"]:
        key = task["key"]
        first = first_attempt.get(key)
        if first is None:
            continue
        if first.get("attempt") not in (1, None):
            raise ValueError("first-pass evidence has an invalid attempt")
        selected = first
        source = "first_attempt"
        if key in retry_keys:
            retry = retry_attempt[key]
            if retry.get("attempt") not in (2, None):
                raise ValueError("retry evidence has an invalid attempt")
            if retry.get("cohort_path") != first.get("cohort_path") or not _inside(
                    retry.get("evidence_dir", ""), retry.get("cohort_path", "")):
                raise ValueError("retry evidence is external to the current cohort")
            selected, source = retry, "retry_attempt"
        if selected.get("classification") == "countable":
            row = copy.deepcopy(selected)
            row.update({"final_status": "countable", "source": source})
        else:
            row = {
                "task_key": key, "arm": task["arm"],
                "scenario": task["scenario"], "route": task["route"],
                "final_status": "zeroed_after_retry", "source": source,
                "DS": 0.0, "RC": None, "collisions": None,
                "route_completed": None, "event_metrics": None,
                "failure_reason": selected.get("failure_reason"),
                "provenance": copy.deepcopy(selected.get("provenance", {})),
            }
        finalized[key] = row
    return finalized


def _mean(values):
    return sum(values) / float(len(values))


def _median(values):
    return float(statistics.median(values))


def _nearest_rank(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, int(math.ceil(probability * len(ordered))))
    return ordered[rank - 1]


def _mixture_quantile(route_samples, probability):
    """Quantile of an empirical mixture assigning equal mass to each route."""
    if not route_samples:
        return None
    weighted = []
    route_weight = 1.0 / len(route_samples)
    for samples in route_samples:
        if not samples:
            raise ValueError("equal-route mixture contains an empty route")
        sample_weight = route_weight / len(samples)
        weighted.extend((float(value), sample_weight) for value in samples)
    cumulative = 0.0
    ordered = sorted(weighted, key=lambda item: item[0])
    for value, weight in ordered:
        cumulative += weight
        if cumulative + 1e-15 >= probability:
            return value
    return ordered[-1][0]


def _derived_seed(seed, label):
    digest = hashlib.sha256(("%d:%s" % (seed, label)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def _bootstrap_mean_ci(values, seed, repetitions, label):
    if not values:
        return None
    if repetitions <= 0 or isinstance(repetitions, bool):
        raise ValueError("bootstrap repetitions must be positive")
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    generator = random.Random(_derived_seed(seed, label))
    count = len(values)
    estimates = []
    for _unused in range(repetitions):
        estimates.append(_mean([values[generator.randrange(count)]
                                for _index in range(count)]))
    return [_nearest_rank(estimates, 0.025), _nearest_rank(estimates, 0.975)]


def _metric(values, seed, repetitions, label, include_distribution=True):
    values = [float(value) for value in values]
    result = {
        "value": None if not values else _mean(values),
        "ci95": _bootstrap_mean_ci(values, seed, repetitions, label),
        "N_valid": len(values),
    }
    if include_distribution:
        result["unit_values"] = values
    return result


def _point(value, count):
    return {"value": None if value is None else float(value), "N_valid": count}


def _science_route_row(task, result):
    events = result.get("event_metrics")
    return {
        "task_key": task["key"], "arm": task["arm"],
        "scenario": task["scenario"], "route": task["route"],
        "final_status": result.get("final_status"),
        "DS": result.get("DS"), "RC": result.get("RC"),
        "route_completed": result.get("route_completed"),
        "collision": result.get("collisions"),
        "actual_drop_rate": None if events is None else events["actual_drop_rate"],
        "mean_aoi_ms": None if events is None else events["mean_aoi_ms"],
        "severe_stale_rate": None if events is None else events["severe_stale_rate"],
        "zoh_rate": None if events is None else events["zoh_rate"],
        "cold_start_count": None if events is None else events["cold_start_count"],
        "observation_count": None if events is None else events["observation_count"],
        "accepted_count": None if events is None else events["accepted_count"],
        "dropped_count": None if events is None else events["dropped_count"],
    }


def _arm_summary(arm, rows, seed, repetitions):
    countable = [row for row in rows if row.get("final_status") == "countable"]
    events = [row["event_metrics"] for row in countable
              if row.get("event_metrics") is not None]
    macro_drop = [row["actual_drop_rate"] for row in events]
    accepted = sum(row["accepted_count"] for row in events)
    dropped = sum(row["dropped_count"] for row in events)
    total = accepted + dropped
    route_samples = [row["aoi_samples_ms"] for row in events]
    mechanism = {
        "actual_drop_rate_macro": _metric(
            macro_drop, seed, repetitions, arm + ":drop-macro"),
        "actual_drop_rate_micro": _point(
            None if total == 0 else float(dropped) / total, len(events)),
        "mean_aoi_ms": _metric(
            [row["mean_aoi_ms"] for row in events], seed, repetitions,
            arm + ":mean-aoi"),
        "aoi_p95_ms": _point(_mixture_quantile(route_samples, 0.95), len(events)),
        "aoi_p99_ms": _point(_mixture_quantile(route_samples, 0.99), len(events)),
        "severe_stale_rate": _metric(
            [row["severe_stale_rate"] for row in events], seed, repetitions,
            arm + ":severe"),
        "zoh_rate": _metric(
            [row["zoh_rate"] for row in events], seed, repetitions,
            arm + ":zoh"),
    }
    ds_values = [row["DS"] for row in rows if row.get("DS") is not None]
    rc_values = [row["RC"] for row in countable if row.get("RC") is not None]
    completion = [float(bool(row["route_completed"])) for row in countable
                  if row.get("route_completed") is not None]
    collision = [float(bool(row["collisions"])) for row in countable
                 if row.get("collisions") is not None]
    driving = {
        "DS_mean": _metric(ds_values, seed, repetitions, arm + ":ds"),
        "DS_median": _point(None if not ds_values else _median(ds_values),
                            len(ds_values)),
        "RC_mean": _metric(rc_values, seed, repetitions, arm + ":rc"),
        "route_completion_rate": _metric(
            completion, seed, repetitions, arm + ":completion"),
        "collision_rate": _metric(
            collision, seed, repetitions, arm + ":collision"),
    }
    return {"mechanism": mechanism, "driving": driving,
            "countable": len(countable), "zeroed_after_retry": sum(
                row.get("final_status") == "zeroed_after_retry" for row in rows)}


def _paired_view(deltas, seed, repetitions, label):
    return {
        "mean_delta_DS": None if not deltas else _mean(deltas),
        "median_delta_DS": None if not deltas else _median(deltas),
        "ci95": _bootstrap_mean_ci(deltas, seed, repetitions, label),
        "N_valid": len(deltas),
        "unit_delta_DS": [float(value) for value in deltas],
    }


def _paired_summaries(manifest, finalized, seed, repetitions):
    def unit_identity(task):
        return tuple(task[field] for field in UNIT_IDENTITY_FIELDS)

    task_by_arm_cell = {
        (task["arm"],) + unit_identity(task): task
        for task in manifest["tasks"]
    }
    paired = {}
    for arm in ("drop_010", "drop_020", "drop_030"):
        primary = []
        sensitivity = []
        for clean in (task for task in manifest["tasks"] if task["arm"] == "clean"):
            cell = unit_identity(clean)
            drop = task_by_arm_cell.get((arm,) + cell)
            clean_row = finalized.get(clean["key"])
            drop_row = None if drop is None else finalized.get(drop["key"])
            if clean_row is None or drop_row is None or \
                    clean_row.get("DS") is None or drop_row.get("DS") is None:
                continue
            delta = float(drop_row["DS"]) - float(clean_row["DS"])
            primary.append(delta)
            if clean_row.get("final_status") == "countable" and \
                    drop_row.get("final_status") == "countable":
                sensitivity.append(delta)
        paired[arm] = {
            "primary": _paired_view(
                primary, seed, repetitions, arm + ":paired-primary"),
            "sensitivity": _paired_view(
                sensitivity, seed, repetitions, arm + ":paired-sensitivity"),
        }
    return paired


def summarize_stage1(manifest, finalized, bootstrap_seed=2000,
                     bootstrap_repetitions=10000):
    """Build route-balanced mechanism and paired driving summaries."""
    validated, task_map = _manifest_task_map(manifest)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if set(finalized).difference(task_map):
        raise ValueError("finalized results contain an unknown task")
    ordered = [(task, finalized[task["key"]]) for task in validated["tasks"]
               if task["key"] in finalized]
    by_arm = {arm["id"]: [] for arm in validated["arms"]}
    route_metrics = []
    provenance = {}
    for task, row in ordered:
        if row.get("task_key") != task["key"] or row.get("arm") != task["arm"] or \
                row.get("scenario") != task["scenario"] or \
                row.get("route") != task["route"]:
            raise ValueError("finalized task identity mismatch")
        by_arm[task["arm"]].append(row)
        route_metrics.append(_science_route_row(task, row))
        provenance[task["key"]] = copy.deepcopy(row.get("provenance", {}))
    arms = {
        arm: _arm_summary(arm, rows, bootstrap_seed,
                          bootstrap_repetitions)
        for arm, rows in by_arm.items()
    }
    monotonic_values = [
        arms[arm]["mechanism"]["actual_drop_rate_macro"]["value"]
        for arm in ("clean", "drop_010", "drop_020", "drop_030")
    ]
    monotonic = None if any(value is None for value in monotonic_values) else all(
        left < right for left, right in zip(monotonic_values, monotonic_values[1:]))
    statuses = {
        "countable": sum(row.get("final_status") == "countable"
                         for _task, row in ordered),
        "zeroed_after_retry": sum(row.get("final_status") == "zeroed_after_retry"
                                  for _task, row in ordered),
    }
    return {
        "schema": "ditl.packet-loss-stage1-summary.v1",
        "bootstrap": {"seed": bootstrap_seed,
                      "repetitions": bootstrap_repetitions},
        "integrity": {
            "planned_tasks": len(validated["tasks"]),
            "finalized_tasks": len(ordered), "statuses": statuses,
            "actual_drop_rate_strictly_increasing": monotonic,
            "shared_gpu_performance_excluded": True,
        },
        "route_metrics": route_metrics,
        "arms": arms,
        "paired": _paired_summaries(
            validated, finalized, bootstrap_seed, bootstrap_repetitions),
        "provenance": {
            "shared_gpu_notice": (
                "Latency, throughput, energy, and GPU utilization are archival "
                "provenance only and are excluded from scientific metrics."
            ),
            "task_provenance": provenance,
        },
    }


def _safe_name(value):
    return value.replace(":", "__").replace("/", "_")


def _canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _csv_bytes(fieldnames, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _arm_csv_rows(summary):
    rows = []
    for arm in ("clean", "drop_010", "drop_020", "drop_030"):
        values = {"arm": arm}
        arm_summary = summary["arms"][arm]
        for group in ("mechanism", "driving"):
            for name, metric in arm_summary[group].items():
                values[group + "." + name] = metric["value"]
                values[group + "." + name + ".N_valid"] = metric["N_valid"]
        rows.append(values)
    return rows


def generate_artifacts(cohort):
    """Validate a completed cohort and atomically publish stable artifacts."""
    cohort = Path(cohort).resolve()
    if cohort.is_symlink() or not cohort.is_dir():
        raise ValueError("cohort must be a regular directory")
    manifest_path = cohort / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("cohort manifest is missing")
    manifest = validate_stage1_manifest(_read_json(manifest_path))
    first = {}
    for task in manifest["tasks"]:
        path = cohort / "attempts" / "attempt-1" / _safe_name(task["key"])
        first[task["key"]] = classify_task_evidence(task, path)
    retry_keys = plan_retry_task_keys(manifest, first)
    retry = {}
    task_map = {task["key"]: task for task in manifest["tasks"]}
    for key in retry_keys:
        path = cohort / "attempts" / "attempt-2" / _safe_name(key)
        retry[key] = classify_task_evidence(task_map[key], path)
    finalized = finalize_task_results(manifest, first, retry)
    summary = summarize_stage1(manifest, finalized)
    task_results = {
        "schema": "ditl.packet-loss-stage1-task-results.v1",
        "tasks": [finalized[task["key"]] for task in manifest["tasks"]],
    }
    integrity = dict(summary["integrity"])
    integrity["schema"] = "ditl.packet-loss-stage1-integrity.v1"
    route_fields = (
        "task_key", "arm", "scenario", "route", "final_status", "DS", "RC",
        "route_completed", "collision", "actual_drop_rate", "mean_aoi_ms",
        "severe_stale_rate", "zoh_rate", "cold_start_count",
        "observation_count", "accepted_count", "dropped_count",
    )
    arm_rows = _arm_csv_rows(summary)
    arm_fields = list(arm_rows[0])
    contents = {
        "stage1_integrity.json": _canonical_json(integrity),
        "stage1_task_results.json": _canonical_json(task_results),
        "stage1_route_metrics.csv": _csv_bytes(route_fields,
                                                  summary["route_metrics"]),
        "stage1_arm_summary.csv": _csv_bytes(arm_fields, arm_rows),
        "stage1_summary.json": _canonical_json(summary),
    }
    for name, content in contents.items():
        _atomic_write(cohort / name, content)
    return {
        name: {"path": str(cohort / name),
               "sha256": hashlib.sha256(content).hexdigest(),
               "size_bytes": len(content)}
        for name, content in contents.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Resolve canonical packet-loss evaluator output.")
    parser.add_argument("--resolve-ego-result", required=True,
                        metavar="CHECKPOINT")
    arguments = parser.parse_args(argv)
    try:
        selected = resolve_ego_result(arguments.resolve_ego_result)
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as error:
        print("invalid ego result: %s" % error, file=sys.stderr)
        return 1
    print(str(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
