#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate a Task 6 cohort and publish strict perturbation trend artifacts."""
from __future__ import print_function

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = (
    "task6_results.json", "task6_results.csv",
    "communication_trend.png", "spatial_trend.png", "compute_trend.png",
    "task6_acceptance.json",
)
FAMILIES = ("communication", "spatial", "compute")
ROUTE_STEM = "town05_short_r146"
RESULT_RELATIVE = Path(
    "v2x_final/town05_short_collab/r146_repeat0/ego_vehicle_0/results.json"
)
IMAGE_ROUTE_RELATIVE = Path("image/v2x_final/town05_short_collab/r146_repeat0")


def _load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("task6_result_" + name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reject(message):
    raise ValueError("invalid Task 6 result cohort: %s" % message)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _reject("duplicate JSON key %r" % key)
        value[key] = item
    return value


def _constant(value):
    _reject("non-finite JSON constant %s" % value)


def _json_text(content, label):
    try:
        return json.loads(content, object_pairs_hook=_pairs, parse_constant=_constant)
    except (TypeError, ValueError) as error:
        _reject("malformed %s: %s" % (label, error))


def _read_json(path, label):
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        _reject("cannot read %s: %s" % (label, error))
    return _json_text(content, label)


def _read_jsonl(path, label):
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        _reject("cannot read %s: %s" % (label, error))
    if not content or not content.endswith("\n"):
        _reject("empty or truncated %s" % label)
    records = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line:
            _reject("blank line in %s at line %d" % (label, number))
        value = _json_text(line, "%s line %d" % (label, number))
        if not isinstance(value, dict):
            _reject("%s line %d must be an object" % (label, number))
        records.append(value)
    return records


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("%s must be a finite number" % label)
    number = float(value)
    if not math.isfinite(number):
        _reject("%s must be finite" % label)
    return number


def _nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _reject("%s must be a non-negative integer" % label)
    return value


def _sha_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _reject("cannot hash %s: %s" % (path, error))
    return digest.hexdigest()


def _regular(path, label):
    try:
        value = Path(path)
    except (TypeError, ValueError) as error:
        _reject("%s has an invalid path: %s" % (label, error))
    if value.is_symlink() or not value.is_file():
        _reject("%s must be a regular non-symlink file: %s" % (label, value))
    return value


def _directory(path, label):
    try:
        value = Path(path)
    except (TypeError, ValueError) as error:
        _reject("%s has an invalid path: %s" % (label, error))
    if value.is_symlink() or not value.is_dir():
        _reject("%s must be a non-symlink directory: %s" % (label, value))
    return value


def _nearest_rank(values, percentile, label):
    numbers = sorted(_finite(value, label) for value in values)
    if not numbers:
        _reject("%s has no measured values" % label)
    rank = int(math.ceil(percentile * len(numbers)))
    return numbers[max(0, rank - 1)]


def _policy_hash(impairment):
    try:
        canonical = {
            "extra_delay_ns": int(round(_finite(impairment["extra_delay_ms"], "extra_delay_ms") * 1000000)),
            "jitter_ns": int(round(_finite(impairment["jitter_ms"], "jitter_ms") * 1000000)),
            "drop_rate": _finite(impairment["drop_rate"], "drop_rate"),
            "burst": {
                "enabled": impairment["burst"]["enabled"],
                "enter_rate": _finite(impairment["burst"]["enter_rate"], "enter_rate"),
                "exit_rate": _finite(impairment["burst"]["exit_rate"], "exit_rate"),
            },
            "reorder_rate": _finite(impairment["reorder_rate"], "reorder_rate"),
            "reorder_extra_ns": int(round(_finite(impairment["reorder_extra_ms"], "reorder_extra_ms") * 1000000)),
            "duplicate_rate": _finite(impairment["duplicate_rate"], "duplicate_rate"),
        }
    except (KeyError, TypeError):
        _reject("config impairment does not match the replay policy schema")
    if type(canonical["burst"]["enabled"]) is not bool:
        _reject("burst.enabled must be a boolean")
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(cohort):
    manifest_path = _regular(cohort / "manifest.json", "cohort manifest")
    manifest = _read_json(manifest_path, "cohort manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != "ditl.task6-cohort.v1":
        _reject("cohort manifest schema is unsupported")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or \
            any(char not in "0123456789abcdef" for char in source_commit):
        _reject("source_commit must be a full lowercase commit ID")
    route = manifest.get("route")
    if not isinstance(route, dict) or route.get("id") != 146 or \
            route.get("scenario_suffix") != "_1" or route.get("repeat") != 0:
        _reject("route contract does not identify route 146/_1/repeat 0")
    rows = manifest.get("arms")
    if not isinstance(rows, list):
        _reject("arms must be a list")
    task6 = _load_script("ditl_task6_manifest")
    expected = task6.validate_task6_arms(task6.build_task6_arms())
    if len(rows) != len(expected):
        _reject("missing or duplicate clean/perturbation arm: exactly ten are required")
    seen = set()
    validated = []
    for sequence, (row, arm) in enumerate(zip(rows, expected)):
        if not isinstance(row, dict):
            _reject("arm manifest row %d must be an object" % sequence)
        arm_id = row.get("arm_id")
        if arm_id in seen:
            _reject("duplicate arm %s" % arm_id)
        seen.add(arm_id)
        if arm_id != arm.arm_id or row.get("sequence") != sequence:
            _reject("arm order/identity mismatch at sequence %d" % sequence)
        if row.get("contract") != dict(vars(arm)):
            _reject("arm contract mismatch for %s" % arm_id)
        validated.append((row, arm))
    if tuple(row[0]["arm_id"] for row in validated) != tuple(task6.ARM_IDS):
        _reject("missing or duplicate clean arm")
    return manifest, tuple(validated)


def _validate_config(cohort, row, arm):
    expected_path = cohort / "configs" / ("%02d_%s.yaml" % (row["sequence"], arm.arm_id))
    path = _regular(row.get("config_path", ""), "config for %s" % arm.arm_id)
    if path != expected_path:
        _reject("config path mismatch for %s" % arm.arm_id)
    if _sha_file(path) != row.get("config_sha256"):
        _reject("config hash mismatch for %s" % arm.arm_id)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        _reject("malformed config for %s: %s" % (arm.arm_id, error))
    try:
        replay = config["simulation"]["ditl_replay"]
        impairment = replay["impairment"]
    except (KeyError, TypeError):
        _reject("config replay block is missing for %s" % arm.arm_id)
    if replay.get("bundle") != row.get("bundle_path"):
        _reject("bundle ID/path mismatch between config and manifest for %s" % arm.arm_id)
    if replay.get("seed") != 2000:
        _reject("config replay seed mismatch for %s" % arm.arm_id)
    actual_drop = _finite(impairment.get("drop_rate"), "configured drop rate")
    if actual_drop != arm.drop_rate:
        _reject("configured drop rate mismatch for %s" % arm.arm_id)
    spatial = config["simulation"].get("ditl_spatial_offset")
    if arm.family == "spatial":
        expected_spatial = {"dx_m": arm.spatial_dx_m, "dy_m": 0.0, "dyaw_deg": 0.0}
        if spatial != expected_spatial:
            _reject("spatial config mismatch for %s" % arm.arm_id)
    elif spatial is not None:
        _reject("non-spatial arm contains a spatial config for %s" % arm.arm_id)
    return replay, _policy_hash(impairment)


def _bundle_metrics(row, cache):
    bundle = _directory(row.get("bundle_path", ""), "bundle")
    feature_id = row.get("bundle_feature_id")
    if bundle.name != feature_id or not isinstance(feature_id, str) or len(feature_id) != 64:
        _reject("bundle ID mismatch for %s" % row.get("arm_id"))
    manifest_path = _regular(bundle / "manifest.json", "bundle manifest")
    if _sha_file(manifest_path) != row.get("bundle_manifest_sha256"):
        _reject("bundle manifest hash mismatch for %s" % row.get("arm_id"))
    if feature_id in cache:
        return cache[feature_id]
    manifest = _read_json(manifest_path, "bundle manifest")
    if manifest.get("feature_id") != feature_id:
        _reject("bundle manifest feature ID mismatch")
    if manifest.get("profile_id") != row.get("contract", {}).get("trace_profile_id"):
        _reject("bundle profile ID mismatch")
    trace_path = _regular(bundle / "raw" / "trace.jsonl", "bundle raw trace")
    trace_sha256 = _sha_file(trace_path)
    if manifest.get("source_trace_sha256") != trace_sha256:
        _reject("bundle raw trace hash mismatch")
    records = _read_jsonl(trace_path, "bundle raw trace")
    values = []
    for record in records:
        if record.get("warmup") is True:
            continue
        if record.get("status", {}).get("outcome") != "consumed":
            continue
        try:
            values.append(record["durations_ns"]["gpu_inference"])
        except (KeyError, TypeError):
            _reject("bundle raw trace lacks gpu_inference")
    if len(values) != manifest.get("accepted_count"):
        _reject("bundle accepted sample count mismatch")
    source = _read_json(bundle / "provenance" / "command_manifest.txt",
                        "bundle command manifest")
    source_commit = source.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        _reject("bundle source commit is missing")
    metrics = {
        "gpu_inference_p95_ms": _nearest_rank(values, 0.95, "gpu_inference") / 1000000.0,
        "bundle_source_commit": source_commit,
        "bundle_trace_sha256": trace_sha256,
    }
    cache[feature_id] = metrics
    return metrics


def _status(cohort, arm_id):
    path = _regular(cohort / "status" / (arm_id + ".json"), "arm status")
    value = _read_json(path, "status for %s" % arm_id)
    if value.get("arm_id") != arm_id:
        _reject("arm identity mismatch in status for %s" % arm_id)
    if value.get("status") != "completed" or value.get("returncode") != 0:
        _reject("arm %s is non-terminal or failed" % arm_id)
    for key in ("cleanup_error", "execution_error", "postflight_error"):
        if value.get(key) is not None:
            _reject("arm %s status contains %s" % (arm_id, key))
    return path


def _validate_consumed_route_identities(checkpoint, arm_id):
    """Bind every route-bearing record consumed by the score parser."""
    if not isinstance(checkpoint, dict):
        _reject("route identity checkpoint is malformed for %s" % arm_id)
    global_record = checkpoint.get("global_record")
    route_records = checkpoint.get("records")
    if not isinstance(global_record, dict) or not isinstance(route_records, list):
        _reject("route identity records are missing for %s" % arm_id)
    # StatisticsManager creates one aggregate record with the frozen -1
    # sentinel. Accepting an arbitrary RouteScenario here could bind the
    # consumed global DS/RC/status to a different route.
    if type(global_record.get("route_id")) is not int or \
            global_record.get("route_id") != -1:
        _reject("route identity mismatch in global record for %s" % arm_id)
    for index, route_record in enumerate(route_records):
        if not isinstance(route_record, dict) or \
                route_record.get("route_id") != "RouteScenario_146":
            _reject("route identity mismatch in record %d for %s" %
                    (index, arm_id))
    if len(route_records) != 1:
        _reject("route identity requires exactly one route record for %s" % arm_id)
    meta = global_record.get("meta")
    if not isinstance(meta, dict):
        _reject("route identity metadata is malformed for %s" % arm_id)
    exceptions = meta.get("exceptions", [])
    if not isinstance(exceptions, list):
        _reject("route identity exceptions are malformed for %s" % arm_id)
    for index, exception in enumerate(exceptions):
        if not isinstance(exception, (list, tuple)) or len(exception) != 3 or \
                exception[0] != "RouteScenario_146":
            _reject("route identity mismatch in exception %d for %s" %
                    (index, arm_id))


def _result_metrics(result_root, arm_id):
    canonical = result_root / RESULT_RELATIVE
    route_results = canonical.parents[1]
    matches = tuple(route_results.glob("ego_vehicle_*/results.json"))
    if len(matches) != 1 or matches[0] != canonical:
        _reject("missing or duplicate result for %s" % arm_id)
    _regular(canonical, "route result")
    document = _read_json(canonical, "route result")
    try:
        checkpoint = document["_checkpoint"]
    except (KeyError, TypeError):
        _reject("route identity checkpoint is missing for %s" % arm_id)
    _validate_consumed_route_identities(checkpoint, arm_id)
    parser = _load_script("dual_fifo_result")
    parsed = parser.parse_route_result(str(canonical))
    if not parsed.get("countable"):
        _reject("route result is nonterminal or not countable for %s: %s" %
                (arm_id, parsed.get("error")))
    return canonical, parsed


def _episode_meta(result_root, arm_id):
    route_root = result_root / IMAGE_ROUTE_RELATIVE
    if route_root.is_symlink() or not route_root.is_dir():
        _reject("image route output is missing for %s" % arm_id)
    episodes = tuple(path for path in route_root.iterdir() if path.is_dir() and not path.is_symlink())
    if len(episodes) != 1:
        _reject("arm %s must contain exactly one result episode" % arm_id)
    return episodes[0], _directory(episodes[0] / "meta", "episode meta")


def _event_metrics(path, episode_name, policy_hash, arm_id):
    records = _read_jsonl(path, "event audit for %s" % arm_id)
    accepted_previous = 0
    dropped_previous = 0
    expired_previous = 0
    reasons = {}
    aoi_values = []
    for index, record in enumerate(records, 1):
        if record.get("schema_version") != "ditl.event-audit.v1":
            _reject("event audit schema mismatch for %s" % arm_id)
        if record.get("run_id") != episode_name:
            _reject("arm/run identity mismatch in event audit for %s" % arm_id)
        if record.get("route_id") != ROUTE_STEM:
            _reject("route identity mismatch in event audit for %s" % arm_id)
        if record.get("policy_config_hash") != policy_hash:
            _reject("config hash mismatch in event audit for %s" % arm_id)
        accepted = _nonnegative_integer(record.get("accepted_count"), "accepted_count")
        dropped = _nonnegative_integer(record.get("dropped_count"), "dropped_count")
        expired = _nonnegative_integer(record.get("expired_count"), "expired_count")
        accepted_delta = accepted - accepted_previous
        dropped_delta = dropped - dropped_previous
        expired_delta = expired - expired_previous
        if accepted_delta < 0 or dropped_delta < 0 or expired_delta < 0:
            _reject("event audit cumulative counters are not monotonic for %s" % arm_id)
        if accepted_delta > 1 or dropped_delta > 1 or expired_delta > 1:
            _reject("event audit counters advanced more than once at line %d for %s" %
                    (index, arm_id))
        reason = record.get("drop_reason")
        if dropped_delta == 1:
            if not isinstance(reason, str) or not reason:
                _reject("dropped submission lacks a reason for %s" % arm_id)
            reasons[reason] = reasons.get(reason, 0) + 1
        elif reason is not None:
            _reject("accepted submission carries a drop reason for %s" % arm_id)
        aoi = record.get("aoi_ns")
        if aoi is not None:
            aoi_values.append(_finite(aoi, "aoi_ns"))
        accepted_previous = accepted
        dropped_previous = dropped
        expired_previous = expired
    if not records or accepted_previous + dropped_previous <= 0:
        _reject("missing final audit counters for %s" % arm_id)
    if reasons.get("bernoulli", 0) != dropped_previous or sum(reasons.values()) != dropped_previous:
        _reject("final drop_reason_counts.bernoulli does not explain dropped submissions for %s" % arm_id)
    return {
        "accepted_count": accepted_previous,
        "dropped_count": dropped_previous,
        "drop_reason_counts": reasons,
        "actual_drop_rate": float(dropped_previous) / (accepted_previous + dropped_previous),
        "aoi_p95_ms": _nearest_rank(aoi_values, 0.95, "AoI") / 1000000.0,
    }


def _pose(value, label):
    if not isinstance(value, list) or len(value) != 6:
        _reject("%s must be a six-element pose" % label)
    return [_finite(item, "%s[%d]" % (label, index)) for index, item in enumerate(value)]


def _spatial_metrics(path, arm):
    records = _read_jsonl(path, "spatial audit for %s" % arm.arm_id)
    offsets = []
    for record in records:
        if record.get("schema") != "ditl.spatial-audit.v1" or record.get("route") != ROUTE_STEM:
            _reject("route/schema mismatch in spatial audit for %s" % arm.arm_id)
        configured = record.get("configured_ego_frame_offset")
        expected = {"dx_m": arm.spatial_dx_m, "dy_m": 0.0, "dyaw_deg": 0.0}
        if configured != expected:
            _reject("configured spatial offset mismatch for %s" % arm.arm_id)
        rsus = record.get("rsus")
        if not isinstance(rsus, list) or not rsus or record.get("applied_rsu_count") != len(rsus):
            _reject("spatial audit contains no applied RSU for %s" % arm.arm_id)
        for rsu in rsus:
            lidar_before = _pose(rsu.get("lidar_pose_before"), "lidar_pose_before")
            lidar_after = _pose(rsu.get("lidar_pose_after"), "lidar_pose_after")
            map_before = _pose(rsu.get("map_pose_before"), "map_pose_before")
            map_after = _pose(rsu.get("map_pose_after"), "map_pose_after")
            for before, after in ((lidar_before, lidar_after), (map_before, map_after)):
                if before[2:4] != after[2:4] or before[5] != after[5]:
                    _reject("clean pose components changed for %s" % arm.arm_id)
                if after[4] - before[4] != arm.spatial_dyaw_deg:
                    _reject("spatial yaw mismatch for %s" % arm.arm_id)
                offsets.append(math.hypot(after[0] - before[0], after[1] - before[1]))
    actual = _nearest_rank(offsets, 0.95, "input offset")
    expected_magnitude = math.hypot(arm.spatial_dx_m, arm.spatial_dy_m)
    if not math.isclose(actual, expected_magnitude, rel_tol=0.0, abs_tol=1e-9):
        _reject("actual input offset mismatch for %s" % arm.arm_id)
    return actual


def _strictly_increasing(values):
    return all(current > previous for previous, current in zip(values, values[1:]))


def _classify(rows, mechanism_key):
    mechanism = [row[mechanism_key] for row in rows]
    if not _strictly_increasing(mechanism):
        return "mechanism_invalid", False
    if all(abs(row["RD"]) < 0.01 for row in rows):
        return "insensitive_at_current_route_and_severity", True
    return "sensitive", True


def parse_cohort(cohort_path):
    """Purely read and validate all ten arms before returning canonical metrics."""
    cohort = _directory(Path(cohort_path), "cohort")
    manifest, arm_records = _validate_manifest(cohort)
    bundle_cache = {}
    rows = []
    for manifest_row, arm in arm_records:
        replay, policy_hash = _validate_config(cohort, manifest_row, arm)
        bundle = _bundle_metrics(manifest_row, bundle_cache)
        status_path = _status(cohort, arm.arm_id)
        expected_root = cohort / "results" / arm.arm_id
        result_root = _directory(manifest_row.get("result_root", ""), "result root")
        if result_root != expected_root:
            _reject("arm result root mismatch for %s" % arm.arm_id)
        result_path, route = _result_metrics(result_root, arm.arm_id)
        episode, meta = _episode_meta(result_root, arm.arm_id)
        event_path = _regular(meta / "ditl_event_audit.jsonl", "event audit")
        event = _event_metrics(event_path, episode.name, policy_hash, arm.arm_id)
        spatial_path = meta / "ditl_spatial_audit.jsonl"
        actual_offset = 0.0 if arm.arm_id == "clean" else None
        if arm.family == "spatial":
            actual_offset = _spatial_metrics(_regular(spatial_path, "spatial audit"), arm)
        elif spatial_path.exists():
            _reject("non-spatial arm has a spatial audit for %s" % arm.arm_id)
        rows.append({
            "sequence": manifest_row["sequence"], "arm_id": arm.arm_id,
            "family": arm.family, "configured_severity": arm.severity,
            "configured_drop_rate": arm.drop_rate,
            "configured_spatial_dx_m": arm.spatial_dx_m,
            "trace_profile_id": arm.trace_profile_id,
            "configured_severity_value": (
                arm.drop_rate if arm.family == "communication" else
                arm.spatial_dx_m if arm.family == "spatial" else
                arm.trace_profile_id if arm.family == "compute" else None
            ),
            "actual_severity_value": (
                event["actual_drop_rate"] if arm.family == "communication" else
                actual_offset if arm.family == "spatial" else
                bundle["gpu_inference_p95_ms"] if arm.family == "compute" else None
            ),
            "actual_drop_rate": event["actual_drop_rate"],
            "actual_input_offset_m": actual_offset,
            "accepted_count": event["accepted_count"],
            "dropped_count": event["dropped_count"],
            "drop_reason_counts": event["drop_reason_counts"],
            "aoi_p95_ms": event["aoi_p95_ms"],
            "gpu_inference_p95_ms": bundle["gpu_inference_p95_ms"],
            "DS": _finite(route["DS"], "DS"), "RC": _finite(route["RC"], "RC"),
            "status": route["status"],
            "collision_markers": {"layout": _finite(route["coll_lay"], "coll_lay"),
                                  "vehicle": _finite(route["coll_veh"], "coll_veh")},
            "config_path": str(manifest_row["config_path"]),
            "config_sha256": manifest_row["config_sha256"],
            "event_audit_path": str(event_path),
            "event_audit_sha256": _sha_file(event_path),
            "spatial_audit_path": str(spatial_path) if arm.family == "spatial" else None,
            "spatial_audit_sha256": (_sha_file(spatial_path)
                                     if arm.family == "spatial" else None),
            "result_path": str(result_path), "status_path": str(status_path),
            "result_sha256": _sha_file(result_path),
            "status_sha256": _sha_file(status_path),
            "bundle_path": str(replay["bundle"]),
            "bundle_id": manifest_row["bundle_feature_id"],
            "bundle_manifest_sha256": manifest_row["bundle_manifest_sha256"],
            "bundle_trace_sha256": bundle["bundle_trace_sha256"],
            "source_commit": manifest["source_commit"],
            "bundle_source_commit": bundle["bundle_source_commit"],
        })
    clean = rows[0]
    if clean["arm_id"] != "clean" or clean["DS"] <= 0:
        _reject("clean DS must be finite and greater than zero")
    rows_with_rd = []
    for row in rows:
        degradation = 1.0 - row["DS"] / clean["DS"]
        _finite(degradation, "RD")
        rows_with_rd.append(dict(row, RD=degradation))
    family_specs = {
        "communication": ("actual_drop_rate", "actual drop rate"),
        "spatial": ("actual_input_offset_m", "input offset (m)"),
        "compute": ("gpu_inference_p95_ms", "GPU inference p95 (ms)"),
    }
    families = {}
    clean_with_rd = rows_with_rd[0]
    for family in FAMILIES:
        key, label = family_specs[family]
        impaired = [row for row in rows_with_rd if row["family"] == family]
        trend_rows = [clean_with_rd] + impaired
        classification, valid = _classify(trend_rows, key)
        families[family] = {
            "arm_ids": ["clean"] + [row["arm_id"] for row in impaired],
            "mechanism_metric": key, "mechanism_axis_label": label,
            "mechanism_valid": valid, "classification": classification,
        }
    final_rows = []
    for row in rows_with_rd:
        if row["arm_id"] == "clean":
            final_rows.append(dict(row, sensitivity_classification="shared_baseline",
                                   mechanism_valid=True))
        else:
            classification = families[row["family"]]["classification"]
            valid = families[row["family"]]["mechanism_valid"]
            final_rows.append(dict(row, sensitivity_classification=classification,
                                   mechanism_valid=valid))
    return {
        "schema": "ditl.task6-results.v1", "cohort": str(cohort),
        "source_commit": manifest["source_commit"], "clean_DS": clean["DS"],
        "rows": final_rows, "families": families,
    }


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True,
                                ensure_ascii=True, allow_nan=False) + "\n")


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fields})


def _plot(path, report, family):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    spec = report["families"][family]
    by_id = {row["arm_id"]: row for row in report["rows"]}
    rows = [by_id[arm_id] for arm_id in spec["arm_ids"]]
    labels = [row["configured_severity"] for row in rows]
    mechanism = [row[spec["mechanism_metric"]] for row in rows]
    degradation = [row["RD"] for row in rows]
    figure, left = pyplot.subplots(figsize=(7.2, 4.4))
    right = left.twinx()
    x_values = list(range(len(rows)))
    left.plot(x_values, mechanism, "o-", color="#1f77b4", label=spec["mechanism_axis_label"])
    right.plot(x_values, degradation, "s--", color="#d62728", label="RD")
    left.set_xticks(x_values)
    left.set_xticklabels(labels)
    left.set_xlabel("configured severity")
    left.set_ylabel(spec["mechanism_axis_label"], color="#1f77b4")
    right.set_ylabel("relative degradation RD", color="#d62728")
    left.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    pyplot.close(figure)


def generate_artifacts(cohort_path, output_directory=None):
    """Validate first, then atomically publish the six canonical artifacts."""
    report = parse_cohort(cohort_path)
    output = Path(output_directory) if output_directory is not None else Path(cohort_path)
    _directory(output, "output directory")
    existing = [name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing:
        _reject("refusing to replace existing output %s" % existing[0])
    stage = Path(tempfile.mkdtemp(prefix=".task6-result-stage-", dir=str(output)))
    try:
        _write_json(stage / "task6_results.json", report)
        _write_csv(stage / "task6_results.csv", report["rows"])
        for family in FAMILIES:
            _plot(stage / (family + "_trend.png"), report, family)
        acceptance = {
            "schema": "ditl.task6-acceptance.v1", "all_results_valid": True,
            "validated_arm_count": len(report["rows"]),
            "fixed_arm_order": [row["arm_id"] for row in report["rows"]],
            "mechanisms_valid": all(value["mechanism_valid"]
                                    for value in report["families"].values()),
            "family_classification": {family: report["families"][family]["classification"]
                                      for family in FAMILIES},
            "spatial_metric_scope": "input_offset_only",
        }
        _write_json(stage / "task6_acceptance.json", acceptance)
        for name in OUTPUT_NAMES:
            os.replace(str(stage / name), str(output / name))
    finally:
        shutil.rmtree(str(stage), ignore_errors=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    generate_artifacts(args.cohort, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
