#!/usr/bin/env python
"""Build the frozen Deployment-in-the-Loop packet-loss stage-one manifest."""

from __future__ import print_function

import copy
import hashlib
import json
import math
from collections.abc import Sequence

from scripts.action_fifo_full_manifest import SUBSETS
from simulation.leaderboard.team_code.closedloop.ditl_impairment_policy import (
    RNG_VERSION,
)


AUTHORITY_SHA256 = "49cfad0e0962bb62a4f195b7a4c26b00c3337a66aa363665707ffe58ce1d4105"
PARTITION_ALGORITHM = "sha256-per-scenario-top9-v1"
MANIFEST_SCHEMA = "ditl.packet-loss-stage1-manifest.v1"
ARMS = (
    ("clean", 0.0),
    ("drop_010", 0.1),
    ("drop_020", 0.2),
    ("drop_030", 0.3),
)
_DOCUMENT_FIELDS = frozenset((
    "schema", "rng_version", "partition_sha256", "arms", "tasks", "batches",
))
_ARM_FIELDS = frozenset(("id", "drop_rate"))
_TASK_FIELDS = frozenset((
    "key", "arm", "drop_rate", "subset", "scenario", "profile", "route",
    "carla_seed", "traffic_seed", "replay_seed", "repeat", "skip_frames",
))
_BATCH_FIELDS = frozenset(("id", "arm", "scenario", "task_keys"))


def canonical_json_bytes(value):
    """Return the canonical, newline-terminated ASCII JSON representation."""
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("ascii")


def _validate_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % name)


def _authority_rows(authority):
    if not isinstance(authority, (tuple, list)) or not authority:
        raise ValueError("authority must be a non-empty sequence")
    rows = []
    seen_scenarios = set()
    seen_cells = set()
    for item in authority:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            raise ValueError("authority entries must have four fields")
        subset, scenario, profile, routes = item
        if (not isinstance(subset, str) or not subset or
                not isinstance(scenario, str) or not scenario.startswith("_") or
                not isinstance(profile, str) or not profile or
                not isinstance(routes, (tuple, list)) or len(routes) < 9):
            raise ValueError("invalid authority entry")
        if scenario in seen_scenarios:
            raise ValueError("duplicate scenario in authority")
        seen_scenarios.add(scenario)
        copied_routes = []
        for route in routes:
            if isinstance(route, bool) or not isinstance(route, int) or route < 0:
                raise ValueError("routes must be non-negative integers")
            cell = (scenario, route)
            if cell in seen_cells:
                raise ValueError("duplicate authority cell")
            seen_cells.add(cell)
            copied_routes.append(route)
        rows.append((subset, scenario, profile, tuple(copied_routes)))
    return tuple(rows)


def build_partition(authority=SUBSETS, seed=2000):
    """Split each official scenario into nine stage-one and remaining cells."""
    _validate_integer(seed, "seed")
    rows = _authority_rows(authority)
    if rows != tuple(SUBSETS):
        raise ValueError("authority must equal the official SUBSETS")
    records = []
    for subset, scenario, profile, routes in rows:
        digests = {
            route: hashlib.sha256(
                ("%d:%s:%d" % (seed, scenario, route)).encode("ascii")
            ).hexdigest()
            for route in routes
        }
        stage1 = set(sorted(routes, key=lambda route: (digests[route], route))[:9])
        for route in routes:
            records.append({
                "subset": subset,
                "scenario": scenario,
                "profile": profile,
                "route": route,
                "rank_key": "%d:%s:%d" % (seed, scenario, route),
                "rank_sha256": digests[route],
                "stage": 1 if route in stage1 else 2,
            })
    stage1_count = sum(row["stage"] == 1 for row in records)
    return {
        "algorithm": PARTITION_ALGORITHM,
        "authority_sha256": AUTHORITY_SHA256,
        "seed": seed,
        "counts": {
            "stage1": stage1_count,
            "stage2": len(records) - stage1_count,
            "total": len(records),
        },
        "records": records,
    }


def _partition_sha256(partition):
    return hashlib.sha256(canonical_json_bytes(partition["records"])).hexdigest()


def _canonical_partition(partition):
    if not isinstance(partition, dict):
        raise TypeError("partition must be a mapping")
    expected = build_partition()
    if partition != expected:
        raise ValueError("partition must equal the frozen stage-one partition")
    return expected


def _validate_run_integer(value, name):
    _validate_integer(value, name)
    if value < 0:
        raise ValueError("%s must be non-negative" % name)


def _task_key(arm, scenario, route, carla_seed, traffic_seed,
              replay_seed, repeat):
    return (
        "stage1:%s:%s:r%d:cseed%d:tseed%d:replay%d:rep%d"
        % (arm, scenario, route, carla_seed, traffic_seed,
           replay_seed, repeat)
    )


def _batch_rows(tasks):
    rows = []
    for arm, _drop_rate in ARMS:
        for _subset, scenario, _profile, _routes in SUBSETS:
            task_keys = [
                task["key"] for task in tasks
                if task["arm"] == arm and task["scenario"] == scenario
            ]
            if task_keys:
                rows.append({
                    "id": "stage1:%s:%s" % (arm, scenario),
                    "arm": arm,
                    "scenario": scenario,
                    "task_keys": task_keys,
                })
    return rows


def _make_manifest(partition, carla_seed, traffic_seed, replay_seed, repeat):
    stage1_records = [row for row in partition["records"] if row["stage"] == 1]
    tasks = []
    for arm, drop_rate in ARMS:
        for cell in stage1_records:
            tasks.append({
                "key": _task_key(
                    arm, cell["scenario"], cell["route"], carla_seed,
                    traffic_seed, replay_seed, repeat,
                ),
                "arm": arm,
                "drop_rate": drop_rate,
                "subset": cell["subset"],
                "scenario": cell["scenario"],
                "profile": cell["profile"],
                "route": cell["route"],
                "carla_seed": carla_seed,
                "traffic_seed": traffic_seed,
                "replay_seed": replay_seed,
                "repeat": repeat,
                "skip_frames": 2,
            })
    return {
        "schema": MANIFEST_SCHEMA,
        "rng_version": RNG_VERSION,
        "partition_sha256": _partition_sha256(partition),
        "arms": [
            {"id": arm, "drop_rate": drop_rate}
            for arm, drop_rate in ARMS
        ],
        "tasks": tasks,
        "batches": _batch_rows(tasks),
    }


def build_stage1_manifest(partition, carla_seed=2000, traffic_seed=2000,
                          replay_seed=2000, repeat=0):
    """Cross the frozen 45 stage-one cells with the four loss arms."""
    canonical_partition = _canonical_partition(partition)
    for name, value in (
            ("carla_seed", carla_seed),
            ("traffic_seed", traffic_seed),
            ("replay_seed", replay_seed),
            ("repeat", repeat)):
        _validate_run_integer(value, name)
    return _make_manifest(
        canonical_partition, carla_seed, traffic_seed, replay_seed, repeat
    )


def _require_fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("%s fields do not match the schema" % name)


def validate_stage1_manifest(document):
    """Validate a complete frozen manifest and return an independent copy."""
    _require_fields(document, _DOCUMENT_FIELDS, "manifest")
    if document["schema"] != MANIFEST_SCHEMA:
        raise ValueError("unexpected manifest schema")
    if document["rng_version"] != RNG_VERSION:
        raise ValueError("unexpected impairment RNG version")
    if (not isinstance(document["arms"], list) or
            not isinstance(document["tasks"], list) or
            not isinstance(document["batches"], list)):
        raise ValueError("arms, tasks, and batches must be arrays")
    for arm in document["arms"]:
        _require_fields(arm, _ARM_FIELDS, "arm")
        if (isinstance(arm["drop_rate"], bool) or
                not isinstance(arm["drop_rate"], (int, float)) or
                not math.isfinite(float(arm["drop_rate"]))):
            raise ValueError("drop_rate must be finite")
    for task in document["tasks"]:
        _require_fields(task, _TASK_FIELDS, "task")
        if (isinstance(task["drop_rate"], bool) or
                not isinstance(task["drop_rate"], (int, float)) or
                not math.isfinite(float(task["drop_rate"]))):
            raise ValueError("task drop_rate must be finite")
    for batch in document["batches"]:
        _require_fields(batch, _BATCH_FIELDS, "batch")
        if not isinstance(batch["task_keys"], list):
            raise ValueError("batch task_keys must be an array")

    if not document["tasks"]:
        raise ValueError("manifest must contain tasks")
    first = document["tasks"][0]
    for name in ("carla_seed", "traffic_seed", "replay_seed", "repeat"):
        _validate_run_integer(first[name], name)
    expected = _make_manifest(
        build_partition(), first["carla_seed"], first["traffic_seed"],
        first["replay_seed"], first["repeat"],
    )
    if document != expected:
        raise ValueError("manifest differs from the canonical stage-one manifest")
    return copy.deepcopy(document)


def build_batches(document, task_keys=None):
    """Return canonical full or retry batches for selected manifest task keys."""
    validated = validate_stage1_manifest(document)
    known_keys = [task["key"] for task in validated["tasks"]]
    if task_keys is None:
        selected = set(known_keys)
    else:
        if (isinstance(task_keys, (str, bytes)) or
                not isinstance(task_keys, Sequence)):
            raise TypeError("task_keys must be a sequence")
        requested = list(task_keys)
        if any(not isinstance(key, str) or not key for key in requested):
            raise ValueError("task keys must be non-empty strings")
        if len(requested) != len(set(requested)):
            raise ValueError("task keys must be unique")
        selected = set(requested)
        unknown = selected.difference(known_keys)
        if unknown:
            raise ValueError("unknown task key")

    batches = []
    for batch in validated["batches"]:
        keys = [key for key in batch["task_keys"] if key in selected]
        if keys:
            row = dict(batch)
            row["task_keys"] = keys
            batches.append(row)
    return tuple(batches)
