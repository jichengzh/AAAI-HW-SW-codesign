"""Pure, fail-closed feedback-update evaluation for the public Stage4 protocol.

The evaluator receives only in-memory synthetic measurements and prediction
records.  It deliberately has no file, network, hardware, or process boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


TARGETS = ("latency_ms", "energy_j", "ap70")
EXPECTED_ARMS = frozenset(
    {
        ("tvm_auto", "fp16"),
        ("tvm_auto", "int8"),
        ("trt_engine", "fp16"),
        ("trt_engine", "int8"),
    }
)
PUBLIC_VALUE_HEADS = frozenset(
    {
        "extra_trees_raw",
        "extra_trees_log",
        "lgbm_l1_raw",
        "lgbm_l1_log",
        "lgbm_huber_raw",
        "lgbm_huber_log",
        "lgbm_quantile_raw",
        "lgbm_quantile_log",
        "extra_trees_residual",
        "lgbm_l1_residual",
        "lgbm_huber_residual",
        "lgbm_quantile_residual",
    }
)
ROW_FIELDS = frozenset(
    {
        "manifest_job_id",
        "group_id",
        "training_source",
        "split",
        "dispatch_key",
        "q_mode",
        *TARGETS,
    }
)
PREDICTION_FIELDS = frozenset(
    {"manifest_job_id", "group_id", "prediction", "lower", "upper"}
)
ROLE_COUNTS = {
    ("initial_coldstart", "train"): 2,
    ("initial_coldstart", "locked_holdout"): 2,
    ("online_feedback", "online_feedback"): 4,
}
FEEDBACK_ROLE = ("online_feedback", "online_feedback")
UNCERTAINTY_METHOD = "lgbm_quantile_plus_group_conformal"
ANONYMOUS_GROUP_ID = re.compile(r"^g(?:0|[1-9][0-9]{0,3})$")


def _invalid() -> None:
    """Reject an input without reflecting caller supplied fields or values."""
    raise ValueError("invalid feedback input")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _anonymous_group_id(value: Any) -> bool:
    return isinstance(value, str) and ANONYMOUS_GROUP_ID.fullmatch(value) is not None


def _anonymous_row_id(value: Any, group_id: Any) -> bool:
    return (
        _anonymous_group_id(group_id)
        and isinstance(value, str)
        and value
        in {
            f"{group_id}|tvm_auto|fp16",
            f"{group_id}|tvm_auto|int8",
            f"{group_id}|trt_engine|fp16",
            f"{group_id}|trt_engine|int8",
        }
    )


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return value


def _validate_rows(value: Any) -> tuple[dict[str, Mapping[str, Any]], dict[str, set[str]]]:
    rows = _sequence(value)
    if rows is None:
        _invalid()

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    groups_by_role = {role: set() for role in ROLE_COUNTS}
    arms_by_group: dict[str, set[tuple[str, str]]] = {}
    role_by_group: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != ROW_FIELDS:
            _invalid()
        row_id = row.get("manifest_job_id")
        group_id = row.get("group_id")
        role = (row.get("training_source"), row.get("split"))
        arm = (row.get("dispatch_key"), row.get("q_mode"))
        if (
            not _anonymous_group_id(group_id)
            or not _anonymous_row_id(row_id, group_id)
            or role not in ROLE_COUNTS
            or arm not in EXPECTED_ARMS
            or not all(_finite_number(row.get(target)) for target in TARGETS)
            or float(row["latency_ms"]) <= 0
            or float(row["energy_j"]) <= 0
            or not 0 <= float(row["ap70"]) <= 1
            or row_id in rows_by_id
        ):
            _invalid()
        previous_role = role_by_group.get(group_id)
        if previous_role is not None and previous_role != role:
            _invalid()
        role_by_group[group_id] = role
        groups_by_role[role].add(group_id)
        arms = arms_by_group.setdefault(group_id, set())
        if arm in arms:
            _invalid()
        arms.add(arm)
        rows_by_id[row_id] = row

    if (
        set(groups_by_role) != set(ROLE_COUNTS)
        or any(len(groups_by_role[role]) != count for role, count in ROLE_COUNTS.items())
        or any(arms != EXPECTED_ARMS for arms in arms_by_group.values())
        or len(arms_by_group) != sum(ROLE_COUNTS.values())
        or len(rows_by_id) != sum(ROLE_COUNTS.values()) * len(EXPECTED_ARMS)
    ):
        _invalid()
    return rows_by_id, groups_by_role


def _validate_heads(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(TARGETS):
        _invalid()
    heads = {target: value.get(target) for target in TARGETS}
    if not all(isinstance(head, str) and head in PUBLIC_VALUE_HEADS for head in heads.values()):
        _invalid()
    return {target: str(heads[target]) for target in TARGETS}


def _prediction_summary(
    value: Any,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    feedback_groups: set[str],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, dict[str, float]]]]:
    if not isinstance(value, Mapping) or set(value) != set(TARGETS):
        _invalid()
    expected_ids = {
        row_id
        for row_id, row in rows_by_id.items()
        if (row["training_source"], row["split"]) == FEEDBACK_ROLE
    }
    if len(expected_ids) != len(feedback_groups) * len(EXPECTED_ARMS):
        _invalid()

    records_by_target: dict[str, list[dict[str, str]]] = {}
    targets: dict[str, dict[str, dict[str, float]]] = {}
    for target in TARGETS:
        prediction_rows = _sequence(value.get(target))
        if prediction_rows is None:
            _invalid()
        parsed: list[tuple[str, str, float, float, float]] = []
        seen: set[str] = set()
        for prediction in prediction_rows:
            if not isinstance(prediction, Mapping) or set(prediction) != PREDICTION_FIELDS:
                _invalid()
            row_id = prediction.get("manifest_job_id")
            group_id = prediction.get("group_id")
            lower = prediction.get("lower")
            estimate = prediction.get("prediction")
            upper = prediction.get("upper")
            source_row = rows_by_id.get(row_id) if isinstance(row_id, str) else None
            if (
                not _anonymous_group_id(group_id)
                or not _anonymous_row_id(row_id, group_id)
                or source_row is None
                or group_id != source_row["group_id"]
                or row_id in seen
                or not all(_finite_number(number) for number in (lower, estimate, upper))
                or float(lower) > float(estimate)
                or float(estimate) > float(upper)
            ):
                _invalid()
            seen.add(row_id)
            parsed.append(
                (str(row_id), str(group_id), float(lower), float(estimate), float(upper))
            )
        if seen != expected_ids:
            _invalid()

        parsed.sort(key=lambda record: record[0])
        actuals = [float(rows_by_id[row_id][target]) for row_id, _, _, _, _ in parsed]
        estimates = [estimate for _, _, _, estimate, _ in parsed]
        widths = [upper - lower for _, _, lower, _, upper in parsed]
        coverage = [
            lower <= actual <= upper
            for actual, (_, _, lower, _, upper) in zip(actuals, parsed)
        ]
        records_by_target[target] = [
            {"manifest_job_id": row_id, "group_id": group_id}
            for row_id, group_id, _, _, _ in parsed
        ]
        targets[target] = {
            "point_metrics": {
                "mae": sum(abs(actual - estimate) for actual, estimate in zip(actuals, estimates))
                / len(parsed)
            },
            "interval": {
                "mean_width": sum(widths) / len(widths),
                "row_coverage": sum(coverage) / len(coverage),
            },
        }
    return records_by_target, targets


def _build_folds(groups_by_role: Mapping[tuple[str, str], set[str]]) -> list[dict[str, list[str] | str]]:
    baseline = sorted(groups_by_role[("initial_coldstart", "train")])
    calibration = sorted(groups_by_role[("initial_coldstart", "locked_holdout")])
    feedback = sorted(groups_by_role[FEEDBACK_ROLE])
    return [
        {
            "heldout_feedback_group": heldout,
            "baseline_fit_groups": list(baseline),
            "updated_fit_groups": [
                *baseline,
                *(group for group in feedback if group != heldout),
            ],
            "added_feedback_groups": [group for group in feedback if group != heldout],
            "calibration_groups": list(calibration),
        }
        for heldout in feedback
    ]


def _phase_summary(
    records: Mapping[str, list[dict[str, str]]],
    targets: Mapping[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    return {
        "row_count": len(records[TARGETS[0]]),
        "records": {target: list(records[target]) for target in TARGETS},
        "targets": {target: dict(targets[target]) for target in TARGETS},
    }


def build_feedback_update_evaluation(
    rows: Sequence[Mapping[str, Any]],
    before_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    after_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    canonical_value_heads: Mapping[str, str],
    *,
    uncertainty_method: str = UNCERTAINTY_METHOD,
) -> dict[str, Any]:
    """Build a closure-audit-compatible report from safe, in-memory inputs.

    Inputs use fixed schemas.  Invalid inputs receive a generic failure so no
    caller-provided identity, location, configuration, or measurement value can
    be reflected in an error or output report.
    """
    if uncertainty_method != UNCERTAINTY_METHOD:
        _invalid()
    rows_by_id, groups_by_role = _validate_rows(rows)
    heads = _validate_heads(canonical_value_heads)
    feedback_groups = groups_by_role[FEEDBACK_ROLE]
    before_records, before_targets = _prediction_summary(
        before_predictions, rows_by_id, feedback_groups
    )
    after_records, after_targets = _prediction_summary(
        after_predictions, rows_by_id, feedback_groups
    )
    if any(
        {
            record["manifest_job_id"]
            for record in before_records[target]
        }
        != {
            record["manifest_job_id"]
            for record in after_records[target]
        }
        for target in TARGETS
    ):
        _invalid()
    return {
        "schema_version": "stage4_feedback_update_eval_v1",
        "canonical_value_heads": heads,
        "uncertainty_method": UNCERTAINTY_METHOD,
        "folds": _build_folds(groups_by_role),
        "before_feedback": _phase_summary(before_records, before_targets),
        "after_feedback": _phase_summary(after_records, after_targets),
    }


__all__ = ["build_feedback_update_evaluation"]
