"""Pure validation and translation of private P6 history feedback artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    validate_profile_backend,
)


FEEDBACK_SCHEMA_VERSION = "p6_h800_coptv2x_feedback_v2"
SUCCESS_STATUS = "measured_success_gold"
FAILURE_STATUSES = frozenset(
    {"feasibility_failure", "numerical_feasibility_failure"}
)
ALLOWED_STATUSES = frozenset({SUCCESS_STATUS, *FAILURE_STATUSES})
PUBLIC_FAILURE_REASON = re.compile(r"^[a-z0-9_-]+$")
TVM_MEASUREMENT_MANIFEST_SCHEMA = "p6_tvm_measurement_manifest_v1"
TVM_TOOLCHAIN_ID = "tvm_auto"
TVM_MEASUREMENT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "hardware_profile",
        "tvm_arch",
        "tvm_cache_namespace",
        "toolchain_id",
        "candidate_id",
        "source_digest",
    }
)


class P6HistoryFeedbackValidationError(ValueError):
    """Stable internal category for malformed private feedback artifacts."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def validate_tvm_measurement_manifest(
    manifest: object,
    *,
    profile: HardwareExecutionProfile,
    candidate_id: str,
    source_digest: str,
) -> bool:
    """Return whether TVM cache evidence exactly matches the active work."""
    try:
        validate_profile_backend(profile, "tvm_auto")
        if not isinstance(candidate_id, str) or not candidate_id or not _is_sha(
            source_digest
        ):
            raise ValueError
        if _contains_tensorrt_engine_marker(manifest):
            raise P6HistoryFeedbackValidationError()
        if not isinstance(manifest, Mapping):
            return False
        expected = {
            "schema_version": TVM_MEASUREMENT_MANIFEST_SCHEMA,
            "hardware_profile": profile.profile_id,
            "tvm_arch": profile.tvm_arch,
            "tvm_cache_namespace": profile.tvm_cache_namespace,
            "toolchain_id": TVM_TOOLCHAIN_ID,
            "candidate_id": candidate_id,
            "source_digest": source_digest,
        }
        return set(manifest) == TVM_MEASUREMENT_MANIFEST_KEYS and dict(
            manifest
        ) == expected
    except P6HistoryFeedbackValidationError:
        raise
    except (TypeError, ValueError):
        raise P6HistoryFeedbackValidationError() from None


def _contains_tensorrt_engine_marker(value: object) -> bool:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            populated = child is not None and child is not False and child != ""
            if "engine" in key and populated:
                return True
            if _contains_tensorrt_engine_marker(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_tensorrt_engine_marker(child) for child in value)
    elif isinstance(value, str):
        normalized = value.lower()
        return normalized.endswith(".engine") or "tensorrt" in normalized
    return False


def translate_history_feedback(
    request: Mapping[str, Any],
    interface: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    read_json: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the complete private completion set and return safe feedback."""
    try:
        state = read_json(paths["task_state"])
        result = read_json(paths["actual_feedback"])
        receipt = read_json(paths["actual_receipt"])
        barrier = read_json(paths["finalization_barrier"])
        task_schema = interface["output_layout"]["task_state"]
        result_schema = interface["actual_feedback"]["result"]
        expected = _expected_identity(request)
        state_statuses = _validate_task_state(state, task_schema, expected)
        feedback_rows = _validate_result(
            result, result_schema, request, expected, state_statuses
        )
        _validate_completion_mapping(
            receipt, interface["actual_feedback"]["receipt"], request, expected
        )
        _validate_completion_mapping(
            barrier,
            interface["actual_feedback"]["finalization_barrier"],
            request,
            expected,
        )
        return {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": feedback_rows,
        }
    except P6HistoryFeedbackValidationError:
        raise
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryFeedbackValidationError() from None


def _expected_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        row["row_id"]: {
            "row_sha256": request["row_sha256"][row["row_id"]],
            "source_evidence_sha256": row["source_evidence_sha256"],
        }
        for row in request["rows"]
    }


def _validate_task_state(
    state: Mapping[str, Any],
    schema: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, str]:
    stage_key = schema["stage_key"]
    rows_key = schema["rows_key"]
    if (
        set(state) != {stage_key, rows_key}
        or state.get(stage_key) != schema["stage_order"][-1]
    ):
        raise ValueError
    rows = _exact_rows(state.get(rows_key), expected, schema)
    for row_id, row in rows.items():
        if row[schema["status_key"]] not in ALLOWED_STATUSES:
            raise ValueError
        _validate_identity_row(row, row_id, schema, expected)
    return {row_id: str(row[schema["status_key"]]) for row_id, row in rows.items()}


def _validate_result(
    result: Mapping[str, Any],
    schema: Mapping[str, Any],
    request: Mapping[str, Any],
    expected: Mapping[str, Any],
    state_statuses: Mapping[str, str],
) -> list[dict[str, Any]]:
    if (
        set(result) != {"measurement_request_sha256", schema["rows_key"]}
        or result.get("measurement_request_sha256")
        != request["measurement_request_sha256"]
    ):
        raise ValueError
    rows = _exact_rows(result.get(schema["rows_key"]), expected, schema, variable=True)
    translated: list[dict[str, Any]] = []
    for request_row in request["rows"]:
        row_id = request_row["row_id"]
        row = rows[row_id]
        _validate_identity_row(row, row_id, schema, expected)
        status = row[schema["status_key"]]
        if status != state_statuses.get(row_id):
            raise ValueError
        base = {"row_id": row_id, "terminal_status": status}
        if status == SUCCESS_STATUS:
            expected_keys = {
                schema["row_id_key"],
                schema["row_hash_key"],
                schema["source_evidence_key"],
                schema["status_key"],
                *schema["metric_keys"],
            }
            if set(row) != expected_keys:
                raise ValueError
            metrics = {key: row[key] for key in schema["metric_keys"]}
            if (
                any(not _finite(value) for value in metrics.values())
                or float(metrics["latency_ms"]) <= 0.0
                or float(metrics["energy_j"]) <= 0.0
                or any(
                    not 0.0 <= float(metrics[key]) <= 1.0
                    for key in ("ap30", "ap50", "ap70")
                )
            ):
                raise ValueError
            translated.append(
                {**base, **{key: float(metrics[key]) for key in schema["metric_keys"]}}
            )
        elif status in FAILURE_STATUSES:
            expected_keys = {
                schema["row_id_key"],
                schema["row_hash_key"],
                schema["source_evidence_key"],
                schema["status_key"],
                "failure_reason",
            }
            reason = row.get("failure_reason")
            if set(row) != expected_keys or not isinstance(reason, str) or not reason:
                raise ValueError
            translated.append(
                {
                    **base,
                    "failure_reason": (
                        reason
                        if PUBLIC_FAILURE_REASON.fullmatch(reason)
                        else "unspecified"
                    ),
                }
            )
        else:
            raise ValueError
    return translated


def _exact_rows(
    raw_rows: object,
    expected: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    variable: bool = False,
) -> dict[str, Mapping[str, Any]]:
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != 4
        or not all(isinstance(row, Mapping) for row in raw_rows)
    ):
        raise ValueError
    row_ids = [row.get(schema["row_id_key"]) for row in raw_rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError
    if len(set(row_ids)) != 4 or set(row_ids) != set(expected):
        raise ValueError
    rows = dict(zip(row_ids, raw_rows, strict=True))
    if not variable:
        required = {
            schema["row_id_key"],
            schema["row_hash_key"],
            schema["source_evidence_key"],
            schema["status_key"],
        }
        if any(set(row) != required for row in rows.values()):
            raise ValueError
    return rows


def _validate_identity_row(
    row: Mapping[str, Any],
    row_id: str,
    schema: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    identity = expected[row_id]
    if (
        row.get(schema["row_hash_key"]) != identity["row_sha256"]
        or row.get(schema["source_evidence_key"])
        != identity["source_evidence_sha256"]
    ):
        raise ValueError


def _validate_completion_mapping(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    request: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    request_key = schema["request_sha256_key"]
    row_hashes_key = schema["row_hashes_key"]
    evidence_key = schema["source_evidence_key"]
    if set(payload) != {request_key, row_hashes_key, evidence_key}:
        raise ValueError
    if payload.get(request_key) != request["measurement_request_sha256"]:
        raise ValueError
    row_hashes = payload.get(row_hashes_key)
    evidence = payload.get(evidence_key)
    if not isinstance(row_hashes, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError
    if set(row_hashes) != set(expected) or set(evidence) != set(expected):
        raise ValueError
    if row_hashes != {
        row_id: values["row_sha256"] for row_id, values in expected.items()
    }:
        raise ValueError
    if evidence != {
        row_id: values["source_evidence_sha256"]
        for row_id, values in expected.items()
    }:
        raise ValueError


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
