"""Pure, redacted Stage2 latency repeat-quality policy.

The policy evaluates caller-provided measurement rows in memory.  It neither
reads experiment artifacts nor returns caller identities, paths, or URIs.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage2_latency_outlier_policy_v1"
VALID_GRADES = {"calibration", "paper"}
CALIBRATION_REPEAT_MAX_MIN = 2.0
PAPER_REPEAT_MAX_MIN = 1.5
CALIBRATION_MULTI_RUN_SPREAD = 0.15
PAPER_MULTI_RUN_SPREAD = 0.10
PAPER_MULTI_RUN_ABS_MS = 0.5


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _latency_ms(row: Mapping[str, Any], field: str) -> tuple[float | None, bool]:
    if field not in row or row.get(field) is None:
        return None, True
    value = _finite_positive(row.get(field))
    return (None, False) if value is None else (value / 1000.0, True)


def _grade_thresholds(grade: str) -> dict[str, float]:
    if not isinstance(grade, str) or grade not in VALID_GRADES:
        raise ValueError("unsupported grade")
    if grade == "paper":
        return {
            "repeat_max_min": PAPER_REPEAT_MAX_MIN,
            "multi_run_spread": PAPER_MULTI_RUN_SPREAD,
            "multi_run_abs_ms": PAPER_MULTI_RUN_ABS_MS,
        }
    return {
        "repeat_max_min": CALIBRATION_REPEAT_MAX_MIN,
        "multi_run_spread": CALIBRATION_MULTI_RUN_SPREAD,
        "multi_run_abs_ms": math.inf,
    }


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """Return a private group key only when every group field is usable."""
    values = tuple(
        row.get(field) for field in ("model", "config_id", "schedule_policy")
    )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    return values  # type: ignore[return-value]


def _validated_group_p50_ms(values: Sequence[float] | object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("group_p50_ms must be a sequence")
    normalized = tuple(_finite_positive(value) for value in values)
    if any(value is None for value in normalized):
        raise ValueError("group_p50_ms must contain finite positive numbers")
    return tuple(float(value) for value in normalized if value is not None)


def _report_row(
    *,
    row_index: int,
    p50_ms: float | None,
    min_ms: float | None,
    max_ms: float | None,
    quality_flag: str,
    claim_status: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "latency_p50_ms": p50_ms,
        "latency_min_ms": min_ms,
        "latency_max_ms": max_ms,
        "quality_flag": quality_flag,
        "claim_status": claim_status,
        "reasons": list(reasons),
    }


def _invalid_row_report(row_index: int) -> dict[str, Any]:
    return _report_row(
        row_index=row_index,
        p50_ms=None,
        min_ms=None,
        max_ms=None,
        quality_flag="invalid_row",
        claim_status="no_claim",
        reasons=["row_is_not_a_mapping"],
    )


def classify_latency_row(
    row: Mapping[str, Any],
    *,
    row_index: int = 0,
    group_p50_ms: Sequence[float] = (),
    grade: str = "calibration",
    historical_anchor_ms: float | None = None,
) -> dict[str, Any]:
    """Classify one row without including its caller-provided identity.

    ``group_p50_ms`` must already contain finite positive millisecond values.
    An invalid anchor fails this row closed instead of being silently ignored.
    """
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("row_index must be a non-negative integer")
    thresholds = _grade_thresholds(grade)
    group_values = _validated_group_p50_ms(group_p50_ms)
    group_key = _group_key(row)
    reasons: list[str] = []
    quality_flag = "stable"
    claim_status = "claimable"
    p50_ms, p50_valid = _latency_ms(row, "latency_p50_us")
    min_ms, min_valid = _latency_ms(row, "latency_min_us")
    max_ms, max_valid = _latency_ms(row, "latency_max_us")

    def reject(flag: str, reason: str) -> None:
        nonlocal quality_flag, claim_status
        if quality_flag == "stable":
            quality_flag = flag
        claim_status = "no_claim"
        reasons.append(reason)

    if row.get("schema") != "latency_lut_row_v1":
        reject("not_latency", "unexpected_row_schema")
    elif row.get("measurement_status") != "measured":
        reject("not_measured", "measurement_not_measured")
    elif group_key is None:
        reject("invalid_group", "group_fields_must_be_non_empty_strings")
    elif not p50_valid or not min_valid or not max_valid:
        reject("invalid_latency", "latency_values_must_be_finite_positive_numbers")
    elif p50_ms is None or min_ms is None or max_ms is None:
        reject("missing_latency", "latency_p50_min_max_are_required")
    elif max_ms < min_ms:
        reject("invalid_latency", "latency_max_is_less_than_latency_min")
    else:
        repeat_ratio = max_ms / min_ms
        if repeat_ratio > thresholds["repeat_max_min"]:
            reject("unstable_repeat", "repeat_variability_exceeds_threshold")

        if len(group_values) >= 2:
            spread = max(group_values) - min(group_values)
            midpoint = median(group_values)
            relative_spread = spread / midpoint
            if not (
                relative_spread <= thresholds["multi_run_spread"]
                or spread <= thresholds["multi_run_abs_ms"]
            ):
                reject("unstable_multi_run", "multi_run_spread_exceeds_threshold")

        if historical_anchor_ms is not None:
            anchor = _finite_positive(historical_anchor_ms)
            if anchor is None:
                reject("invalid_anchor", "historical_anchor_must_be_finite_positive")
            else:
                drift = abs(p50_ms - anchor)
                relative_drift = drift / anchor
                if grade == "paper" and relative_drift > 0.10 and drift > 0.5:
                    reject("anchor_drift", "historical_anchor_drift_exceeds_threshold")
                elif relative_drift > 0.15 and drift > 0.5:
                    reasons.append("historical_anchor_drift_warning")

    return _report_row(
        row_index=row_index,
        p50_ms=p50_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        quality_flag=quality_flag,
        claim_status=claim_status,
        reasons=reasons,
    )


def _valid_group_values(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = _group_key(row)
        p50_ms, valid = _latency_ms(row, "latency_p50_us")
        if key is not None and valid and p50_ms is not None:
            grouped.setdefault(key, []).append(p50_ms)
    return grouped


def _validate_anchors(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("historical_anchors_ms must be a mapping")
    return value


def detect_latency_outliers(
    rows: Sequence[Mapping[str, Any] | object],
    *,
    grade: str = "calibration",
    historical_anchors_ms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a redacted, deterministic latency-quality report.

    Caller identifiers are only used internally to form repeat groups and are
    never reflected in results, reasons, or exceptions.
    """
    _grade_thresholds(grade)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a sequence")
    if not rows:
        raise ValueError("rows must not be empty")
    anchors = _validate_anchors(historical_anchors_ms)
    mapped_rows = [row for row in rows if isinstance(row, Mapping)]
    grouped = _valid_group_values(mapped_rows)
    report_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            report_rows.append(_invalid_row_report(row_index))
            continue
        key = _group_key(row)
        anchor_key = "|".join(key) if key is not None else None
        report_rows.append(
            classify_latency_row(
                row,
                row_index=row_index,
                group_p50_ms=grouped.get(key, ()) if key is not None else (),
                grade=grade,
                historical_anchor_ms=anchors.get(anchor_key) if anchor_key else None,
            )
        )
    unstable_rows = sum(
        row["quality_flag"] != "stable" or row["claim_status"] != "claimable"
        for row in report_rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "grade": grade,
        "total_rows": len(report_rows),
        "unstable_rows": unstable_rows,
        "claimable_rows": len(report_rows) - unstable_rows,
        "rows": report_rows,
    }


__all__ = ["SCHEMA_VERSION", "classify_latency_row", "detect_latency_outliers"]
