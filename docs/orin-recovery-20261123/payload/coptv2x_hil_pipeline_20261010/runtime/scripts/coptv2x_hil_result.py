#!/usr/bin/env python3
"""Validate and audit per-episode CARLA plus asynchronous HIL evidence."""

from __future__ import print_function

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


class HilAuditError(ValueError):
    pass


def _sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        raise HilAuditError("artifact hashes must be 64-character SHA-256 strings")
    try:
        int(value, 16)
    except ValueError as error:
        raise HilAuditError("artifact hash is not hexadecimal") from error
    return value.lower()


def validate_episode_audit(audit):
    if not isinstance(audit, dict):
        raise HilAuditError("episode HIL audit must be an object")
    required = ("batch_order", "trigger_identity", "result_identity",
                "artifact_sha256", "aoi_frames", "remote_input_tensor_bytes")
    missing = [key for key in required if key not in audit]
    if missing:
        raise HilAuditError("missing episode evidence: %s" % ", ".join(missing))
    if audit["batch_order"] != ["ego", "rsu"]:
        raise HilAuditError("batch_order must be [ego, rsu]")
    trigger, result = audit["trigger_identity"], audit["result_identity"]
    if not isinstance(trigger, dict) or not isinstance(result, dict) or not trigger:
        raise HilAuditError("trigger/result identities must be non-empty objects")
    for field in ("run_id", "route_id", "scenario_id", "arm_id"):
        if field not in trigger or result.get(field) != trigger[field]:
            raise HilAuditError("trigger/result %s identity mismatch" % field)
    for field in ("sequence_no", "source_frame"):
        if type(trigger.get(field)) is not int or type(result.get(field)) is not int:
            raise HilAuditError("trigger/result %s must be an integer" % field)
        if result[field] > trigger[field]:
            raise HilAuditError("selected HIL result cannot be newer than current trigger")
    hashes = audit["artifact_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise HilAuditError("artifact_sha256 must be a non-empty mapping")
    normalized_hashes = {str(key): _sha256(value) for key, value in hashes.items()}
    if type(audit["aoi_frames"]) is not int or audit["aoi_frames"] < 0:
        raise HilAuditError("aoi_frames must be a non-negative integer")
    if audit["remote_input_tensor_bytes"] != 0:
        raise HilAuditError("remote_input_tensor_bytes must equal zero")
    return dict(audit, artifact_sha256=normalized_hashes)


def validate_task_artifacts(task, audit):
    """Fail closed unless fusion evidence used the exact frozen task artifacts."""
    expected = task.get("artifact_sha256")
    actual = audit.get("artifact_sha256") if isinstance(audit, dict) else None
    if not isinstance(expected, dict) or not expected or not isinstance(actual, dict):
        raise HilAuditError("task and fusion audit artifact hashes are required")
    normalized_expected = {str(key): _sha256(value) for key, value in expected.items()}
    normalized_actual = {str(key): _sha256(value) for key, value in actual.items()}
    if normalized_actual != normalized_expected:
        raise HilAuditError("fusion audit artifact hashes do not match the frozen task")
    return normalized_actual


def classify_driving_result(status):
    normalized = str(status or "").strip().lower().replace("-", "_").replace(" ", "_")
    driving = {"collision", "blocked", "route_timeout", "timeout",
               "route_not_completed", "not_completed", "failed_driving"}
    if normalized in driving:
        return "DRIVING_TERMINAL"
    if normalized in ("completed", "complete", "success"):
        return "COMPLETED"
    return "INFRA_RETRY"


def paired_row(task_id, arm_id, classification, carla_result):
    if classification in ("INFRA_RETRY", "INFRA_INVALID"):
        return {"task_id": task_id, "arm_id": arm_id, "status": "INFRA_INVALID",
                "DS": None, "RC": None, "countable": False}
    if isinstance(carla_result, dict):
        scores = carla_result.get("scores", {})
        return {"task_id": task_id, "arm_id": arm_id,
                "status": "Completed" if classification == "COMPLETED" else "FailedDriving",
                "DS": carla_result.get("DS", scores.get("score_composed")),
                "RC": carla_result.get("RC", scores.get("score_route")),
                "countable": True}
    return {"task_id": task_id, "arm_id": arm_id, "status": "FailedDriving",
            "DS": None, "RC": None, "countable": True}


def atomic_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                             dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(destination))
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_paired_csv(path, rows):
    fields = ("task_id", "arm_id", "status", "DS", "RC", "countable")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                             dir=str(destination.parent))
    os.close(descriptor)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(destination))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
