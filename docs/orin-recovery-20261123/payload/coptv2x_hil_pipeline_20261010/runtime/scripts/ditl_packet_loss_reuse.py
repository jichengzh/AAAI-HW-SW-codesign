#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate and materialize reusable packet-loss stage-one evidence."""

from __future__ import print_function

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from scripts.ditl_packet_loss_manifest import validate_stage1_manifest
import scripts.ditl_packet_loss_result as packet_result
from scripts.dual_fifo_result import parse_route_result


PLAN_SCHEMA = "ditl.packet-loss-stage1-reuse-plan.v1"
PROVENANCE_SCHEMA = "ditl.packet-loss-stage1-reuse-provenance.v1"
_COHORT_TIMESTAMP = re.compile(r"packet-loss-stage1-([0-9]{8}T[0-9]{6}Z)-")
_SOURCE_SUFFIX = re.compile(r"-(r1|r2|r4)$")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value):
    return value.replace(":", "__").replace("/", "_")


def _timestamp(path):
    match = _COHORT_TIMESTAMP.search(Path(path).name)
    if match is None:
        raise ValueError("source cohort name lacks a canonical timestamp")
    return match.group(1)


def _source_kind(path, requested=None):
    match = _SOURCE_SUFFIX.search(Path(path).name)
    if match is None:
        raise ValueError("source cohort must end in r1, r2, or r4")
    inferred = "legacy_r1" if match.group(1) == "r1" else "canonical"
    if requested is not None and requested != inferred:
        raise ValueError("source kind does not match cohort generation")
    return inferred


def _expected_hash(expected_config_sha256, task_key):
    if isinstance(expected_config_sha256, str):
        value = expected_config_sha256
    elif isinstance(expected_config_sha256, dict):
        if task_key not in expected_config_sha256:
            raise ValueError("target config hash is missing for %s" % task_key)
        value = expected_config_sha256[task_key]
    else:
        raise ValueError("expected config hashes must be a string or mapping")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("target config hash is invalid for %s" % task_key)
    return value


def _validate_source_path(cohort_path, authorized_roots):
    source = Path(cohort_path)
    if not source.is_absolute() or ".." in source.parts:
        raise ValueError("source cohort path must be absolute without traversal")
    if source.is_symlink():
        raise ValueError("source cohort must not be a symlink")
    roots = [Path(root).resolve() for root in authorized_roots]
    if source.parent.resolve() not in roots:
        raise ValueError("source cohort must be a direct child of an authorized root")
    if not source.is_dir():
        raise ValueError("source cohort is missing")
    return source.resolve()


def _validate_config(cohort, task, evidence, expected_config_sha256):
    expected = _expected_hash(expected_config_sha256, task["key"])
    config = cohort / "configs" / (_safe_name(task["key"]) + ".yaml")
    if config.is_symlink() or not config.is_file() or not packet_result._inside(
            config, cohort):
        raise ValueError("source task config is missing or external")
    actual = _sha(config)
    if actual != expected:
        raise ValueError("source config hash does not match target config hash")
    if evidence.get("config_sha256") != expected:
        raise ValueError("evidence config hash does not match target config hash")
    provenance = cohort / "provenance.json"
    if provenance.is_symlink() or not provenance.is_file():
        raise ValueError("source provenance is missing or symlinked")
    provenance_payload = packet_result._read_json(provenance)
    if provenance_payload.get("config_sha256", {}).get(task["key"]) != expected:
        raise ValueError("source provenance config hash does not match target")
    return expected, config, provenance


def _source_hashes(attempt_dir, evidence, paths, config_path, provenance_path,
                   legacy_result=None):
    values = {"evidence.json": _sha(attempt_dir / "evidence.json"),
              "config.yaml": _sha(config_path),
              "provenance.json": _sha(provenance_path)}
    for name, path in paths.items():
        values[name] = _sha(path)
    if legacy_result is not None:
        values["legacy_result.json"] = _sha(legacy_result)
    return values


def _candidate_base(task, attempt_dir, source_kind):
    cohort = packet_result._infer_cohort(attempt_dir)
    return {
        "task_key": task["key"], "source_kind": source_kind,
        "source_cohort": None if cohort is None else str(cohort),
        "source_attempt_dir": str(Path(attempt_dir).resolve()),
        "cohort_timestamp": None if cohort is None else _timestamp(cohort),
    }


def _rejected(base, error):
    row = dict(base)
    row.update({"status": "rejected", "rejection_reason": str(error),
                "DS": None, "RC": None, "source_hashes": {},
                "materialization_artifacts": {}})
    return row


def _legacy_result_path(attempt_dir, task):
    attempt_dir = Path(attempt_dir)
    if task.get("repeat") != 0:
        raise ValueError("legacy recovery supports only the exact repeat0 result")
    result = (attempt_dir / "work" / "v2x_final" /
              "town05_short_collab" /
              ("r%d_repeat0" % task["route"]) /
              "ego_vehicle_0" / "results.json")
    current = attempt_dir
    for part in result.relative_to(attempt_dir).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("legacy direct result path contains a symlink")
    if not result.is_file() or not packet_result._inside(result, attempt_dir):
        raise ValueError("legacy direct ego result is missing or external")
    result_root = result.parent.parent
    matches = sorted(result_root.glob("ego_vehicle_*/results.json"))
    if matches != [result]:
        raise ValueError("legacy route must contain exactly one ego result")
    return result


def inspect_reuse_candidate(task, attempt_dir, source_kind,
                            expected_config_sha256):
    """Inspect one canonical or r1 candidate and return a JSON-safe record."""
    attempt_dir = Path(attempt_dir)
    base = _candidate_base(task, attempt_dir, source_kind)
    try:
        if source_kind not in ("canonical", "legacy_r1"):
            raise ValueError("unknown reuse source kind")
        cohort = packet_result._infer_cohort(attempt_dir)
        if cohort is None or attempt_dir.is_symlink() or not attempt_dir.is_dir():
            raise ValueError("attempt directory is missing or outside a cohort")
        if _source_kind(cohort, source_kind) != source_kind:
            raise ValueError("source kind mismatch")
        evidence_path = attempt_dir / "evidence.json"
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise ValueError("source evidence is missing or symlinked")
        evidence = packet_result._read_json(evidence_path)
        if evidence.get("schema") != packet_result.EVIDENCE_SCHEMA:
            raise ValueError("evidence schema mismatch")
        if Path(evidence.get("cohort_path", "")).resolve() != cohort:
            raise ValueError("evidence cohort identity mismatch")
        artifacts = evidence.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(
                packet_result.ARTIFACT_NAMES):
            raise ValueError("evidence artifact set is incomplete")
        paths = {name: packet_result._artifact_path(
                    attempt_dir, artifacts[name], name)
                 for name in packet_result.ARTIFACT_NAMES}
        status = packet_result._read_json(paths["status.json"])
        identity = packet_result._read_json(paths["command_identity.json"])
        attempt, config_sha, policy_sha = packet_result._validate_execution_identity(
            task, evidence, status, identity)
        if attempt != 1:
            raise ValueError("only first-attempt evidence can be reused")
        if status.get("returncode") != 0 or status.get("state") != "completed":
            raise ValueError("evaluator or agent execution failed")
        expected, config_path, provenance_path = _validate_config(
            cohort, task, evidence, expected_config_sha256)
        if config_sha != expected:
            raise ValueError("identity config hash does not match target config hash")

        legacy_result = None
        if source_kind == "canonical":
            classified = packet_result.classify_task_evidence(task, attempt_dir)
            if classified.get("classification") != "countable":
                raise ValueError("canonical classifier rejected evidence: %s" %
                                 classified.get("failure_reason"))
            if classified.get("provenance", {}).get("config_sha256") != expected:
                raise ValueError("source provenance config hash mismatch")
            result_path = paths["result.json"]
            ds, rc = classified["DS"], classified["RC"]
        else:
            legacy_result = _legacy_result_path(attempt_dir, task)
            parsed = parse_route_result(str(legacy_result))
            if not parsed.get("countable"):
                raise ValueError(parsed.get(
                    "error", "legacy direct result is not terminal/countable"))
            packet_result._event_metrics(
                packet_result._read_jsonl(paths["event_audit.jsonl"]),
                task, policy_sha)
            result_path = legacy_result
            ds, rc = float(parsed["DS"]), float(parsed["RC"])

        source_hashes = _source_hashes(
            attempt_dir, evidence, paths, config_path, provenance_path,
            legacy_result)
        materialization = {}
        for name in packet_result.ARTIFACT_NAMES:
            source_path = result_path if name == "result.json" else paths[name]
            materialization[name] = {
                "source_path": str(source_path), "sha256": _sha(source_path),
            }
        row = dict(base)
        row.update({
            "status": "accepted", "rejection_reason": None,
            "DS": ds, "RC": rc,
            "target_config_sha256": expected,
            "source_hashes": source_hashes,
            "materialization_artifacts": materialization,
        })
        return row
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return _rejected(base, error)


def _load_source_manifest(cohort):
    path = cohort / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("source manifest is missing or symlinked")
    return validate_stage1_manifest(packet_result._read_json(path))


def validate_reuse_source(manifest, cohort_path, expected_config_sha256,
                          authorized_roots, source_kind=None):
    """Validate a source cohort and inspect every task directory it contains."""
    target = validate_stage1_manifest(copy.deepcopy(manifest))
    cohort = _validate_source_path(cohort_path, authorized_roots)
    kind = _source_kind(cohort, source_kind)
    timestamp = _timestamp(cohort)
    source_manifest = _load_source_manifest(cohort)
    target_tasks = {task["key"]: task for task in target["tasks"]}
    source_tasks = {task["key"]: task for task in source_manifest["tasks"]}
    if source_tasks != target_tasks:
        raise ValueError("source manifest task identities do not match target manifest")
    attempt_root = cohort / "attempts" / "attempt-1"
    candidates = []
    if attempt_root.is_dir() and not attempt_root.is_symlink():
        for directory in sorted(attempt_root.iterdir(), key=lambda value: str(value)):
            if not directory.is_dir() or directory.is_symlink():
                continue
            evidence_path = directory / "evidence.json"
            try:
                evidence = packet_result._read_json(evidence_path)
                task_key = evidence.get("task_identity", {}).get("key")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                task_key = None
            if task_key not in target_tasks:
                continue
            expected = _expected_hash(expected_config_sha256, task_key)
            candidates.append(inspect_reuse_candidate(
                target_tasks[task_key], directory, kind, expected))
    candidates.sort(key=lambda row: (row["task_key"], row["source_attempt_dir"]))
    return {
        "source_cohort": str(cohort), "source_kind": kind,
        "cohort_timestamp": timestamp,
        "manifest_sha256": _sha(cohort / "manifest.json"),
        "candidates": candidates,
        "accepted_count": sum(row["status"] == "accepted" for row in candidates),
        "rejected_count": sum(row["status"] == "rejected" for row in candidates),
    }


def _selection_key(candidate):
    # Negative decimal timestamp implements newest-first without locale/timezone.
    timestamp = int(candidate["cohort_timestamp"].replace("T", "").replace("Z", ""))
    return (0 if candidate["source_kind"] == "canonical" else 1,
            -timestamp, candidate["source_attempt_dir"])


def build_reuse_plan(manifest, source_cohorts, expected_config_sha256,
                     authorized_roots):
    """Build a deterministic, JSON-serializable evidence selection plan."""
    validated = validate_stage1_manifest(copy.deepcopy(manifest))
    sources = [validate_reuse_source(
        validated, path, expected_config_sha256, authorized_roots)
        for path in sorted(set(str(Path(item)) for item in source_cohorts))]
    candidates = sorted(
        [row for source in sources for row in source["candidates"]],
        key=lambda row: (row["task_key"], _selection_key(row)))
    by_task = {}
    rejections = []
    for candidate in candidates:
        if candidate["status"] != "accepted":
            rejections.append(copy.deepcopy(candidate))
            continue
        by_task.setdefault(candidate["task_key"], []).append(candidate)
    selected = {}
    for task in validated["tasks"]:
        rows = sorted(by_task.get(task["key"], ()), key=_selection_key)
        if not rows:
            continue
        selected[task["key"]] = copy.deepcopy(rows[0])
        for row in rows[1:]:
            rejected = copy.deepcopy(row)
            rejected["status"] = "rejected"
            rejected["rejection_reason"] = "superseded by deterministic selection"
            rejections.append(rejected)
    ordered_selected = {task["key"]: selected[task["key"]]
                        for task in validated["tasks"]
                        if task["key"] in selected}
    missing = [task["key"] for task in validated["tasks"]
               if task["key"] not in selected]
    counts = {}
    for row in ordered_selected.values():
        counts[row["source_kind"]] = counts.get(row["source_kind"], 0) + 1
    rejections.sort(key=lambda row: (row["task_key"], row["source_attempt_dir"],
                                     row.get("rejection_reason") or ""))
    return {
        "schema": PLAN_SCHEMA,
        "manifest_sha256": hashlib.sha256(_json_bytes(validated)).hexdigest(),
        "sources": sources, "selected": ordered_selected,
        "selected_count": len(ordered_selected),
        "missing_task_keys": missing, "source_counts": counts,
        "rejections": rejections,
    }


def _copy_verified(source, target, expected_sha):
    source = Path(source)
    if source.is_symlink() or not source.is_file() or _sha(source) != expected_sha:
        raise ValueError("reuse source artifact changed after planning: %s" % source)
    payload = source.read_bytes()
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)


def materialize_reuse_plan(manifest, plan, target_cohort):
    """Copy selected evidence into target attempt-1 and reclassify each task."""
    validated = validate_stage1_manifest(copy.deepcopy(manifest))
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("reuse plan schema mismatch")
    if plan.get("manifest_sha256") != hashlib.sha256(
            _json_bytes(validated)).hexdigest():
        raise ValueError("reuse plan manifest mismatch")
    target = Path(target_cohort)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise ValueError("target cohort must be an existing absolute regular directory")
    target = target.resolve()
    task_map = {task["key"]: task for task in validated["tasks"]}
    selected = plan.get("selected")
    if not isinstance(selected, dict) or set(selected).difference(task_map):
        raise ValueError("reuse plan selection is malformed")
    attempt_root = target / "attempts" / "attempt-1"
    attempt_root.mkdir(parents=True, exist_ok=True)
    materialized = []
    for task in validated["tasks"]:
        candidate = selected.get(task["key"])
        if candidate is None:
            continue
        destination = attempt_root / _safe_name(task["key"])
        staging = attempt_root / (".reuse-%s" % _safe_name(task["key"]))
        if destination.exists() or destination.is_symlink() or staging.exists():
            raise FileExistsError("reuse destination already exists: %s" % destination)
        staging.mkdir()
        try:
            fresh = inspect_reuse_candidate(
                task, candidate.get("source_attempt_dir"),
                candidate.get("source_kind"),
                candidate.get("target_config_sha256"))
            if fresh.get("status") != "accepted" or \
                    fresh.get("source_hashes") != candidate.get("source_hashes") or \
                    fresh.get("materialization_artifacts") != candidate.get(
                        "materialization_artifacts"):
                raise ValueError("reuse candidate changed or failed revalidation")
            descriptors = candidate.get("materialization_artifacts", {})
            if set(descriptors) != set(packet_result.ARTIFACT_NAMES):
                raise ValueError("reuse artifact selection is incomplete")
            for name in packet_result.ARTIFACT_NAMES:
                descriptor = descriptors[name]
                _copy_verified(descriptor["source_path"], staging / name,
                               descriptor["sha256"])
            source_evidence = Path(candidate["source_attempt_dir"]) / "evidence.json"
            if source_evidence.is_symlink() or _sha(source_evidence) != \
                    candidate["source_hashes"]["evidence.json"]:
                raise ValueError("source evidence changed after planning")
            evidence = packet_result._read_json(source_evidence)
            evidence["cohort_path"] = str(target)
            evidence["artifacts"] = {
                name: {"path": name, "sha256": _sha(staging / name)}
                for name in packet_result.ARTIFACT_NAMES
            }
            evidence["reuse_provenance"] = {
                "schema": PROVENANCE_SCHEMA,
                "source_kind": candidate["source_kind"],
                "source_cohort": candidate["source_cohort"],
                "source_attempt_dir": candidate["source_attempt_dir"],
                "source_hashes": copy.deepcopy(candidate["source_hashes"]),
            }
            evidence_path = staging / "evidence.json"
            with evidence_path.open("xb") as stream:
                stream.write(_json_bytes(evidence)); stream.flush(); os.fsync(stream.fileno())
            evidence_path.chmod(0o400)
            os.rename(str(staging), str(destination))
            classified = packet_result.classify_task_evidence(task, destination)
            if classified.get("classification") != "countable":
                shutil.rmtree(str(destination))
                raise ValueError("materialized evidence is not countable: %s" %
                                 classified.get("failure_reason"))
            materialized.append({
                "task_key": task["key"], "evidence_dir": str(destination),
                "classification": "countable", "DS": classified["DS"],
                "RC": classified["RC"], "source_kind": candidate["source_kind"],
            })
        except Exception:
            if staging.exists():
                shutil.rmtree(str(staging))
            raise
    keys = [row["task_key"] for row in materialized]
    return {
        "selected_task_keys": keys, "selected_count": len(keys),
        "missing_task_keys": list(plan.get("missing_task_keys", [])),
        "materialized": materialized,
    }
