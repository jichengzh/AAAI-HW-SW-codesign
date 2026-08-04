"""Pure, fail-closed selection completion for the public Stage4 protocol.

The module accepts only anonymous, in-memory four-arm measurements plus a
strict selection manifest.  It returns aggregate evidence suitable for the
Stage4 closure audit and deliberately never exposes row or group identities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "stage4_selection_completion_v1"
MANIFEST_SCHEMA_VERSION = "stage4_selection_completion_manifest_v1"
TARGETS = ("latency_ms", "energy_j", "ap70")
EXPECTED_ARMS = frozenset(
    {
        ("tvm_auto", "fp16"),
        ("tvm_auto", "int8"),
        ("trt_engine", "fp16"),
        ("trt_engine", "int8"),
    }
)
PUBLIC_ACQUISITION_POLICIES = frozenset(
    {"random", "predicted_frontier_diversity"}
)
CANDIDATE_FIELDS = frozenset(
    {
        "manifest_job_id",
        "group_id",
        "dispatch_key",
        "q_mode",
        *TARGETS,
    }
)
SELECTION_FIELDS = frozenset(
    {"manifest_job_id", "group_id", "dispatch_key", "q_mode"}
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "required_arm_product",
        "candidate_row_ids",
        "selection_policy",
        "folds",
        "replay_policies",
    }
)
FOLD_FIELDS = frozenset({"fold_id", "candidate_group_ids", "selected_rows"})
POLICY_METRIC_FIELDS = frozenset(
    {"groups_to_95pct_oracle_HV_median", "success_rate"}
)
ANONYMOUS_GROUP_ID = re.compile(r"^g(?:0|[1-9][0-9]{0,3})$")
ANONYMOUS_FOLD_ID = re.compile(r"^f(?:0|[1-9][0-9]{0,3})$")
MAX_GROUPS = 10_000
MAX_MEASUREMENT = 1_000_000_000.0


def _invalid() -> None:
    """Reject a caller-controlled payload without reflecting any of it."""
    raise ValueError("invalid selection input")


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return value


def _exact_mapping(value: Any, fields: frozenset[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == fields


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _anonymous_group_id(value: Any) -> bool:
    return isinstance(value, str) and ANONYMOUS_GROUP_ID.fullmatch(value) is not None


def _anonymous_fold_id(value: Any) -> bool:
    return isinstance(value, str) and ANONYMOUS_FOLD_ID.fullmatch(value) is not None


def _anonymous_row_id(
    value: Any, group_id: Any, dispatch_key: Any, q_mode: Any
) -> bool:
    return (
        _anonymous_group_id(group_id)
        and isinstance(dispatch_key, str)
        and isinstance(q_mode, str)
        and isinstance(value, str)
        and value == f"{group_id}|{dispatch_key}|{q_mode}"
    )


def _valid_measurements(row: Mapping[str, Any]) -> bool:
    latency = row.get("latency_ms")
    energy = row.get("energy_j")
    ap70 = row.get("ap70")
    return (
        all(_finite_number(value) for value in (latency, energy, ap70))
        and 0.0 < float(latency) <= MAX_MEASUREMENT
        and 0.0 < float(energy) <= MAX_MEASUREMENT
        and 0.0 <= float(ap70) <= 1.0
    )


def _validate_candidate_rows(
    value: Any,
) -> tuple[dict[str, tuple[str, str, str]], set[str]]:
    rows = _sequence(value)
    if rows is None or not 1 <= len(rows) <= MAX_GROUPS * len(EXPECTED_ARMS):
        _invalid()

    candidates: dict[str, tuple[str, str, str]] = {}
    arms_by_group: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if not _exact_mapping(row, CANDIDATE_FIELDS):
            _invalid()
        row_id = row.get("manifest_job_id")
        group_id = row.get("group_id")
        dispatch_key = row.get("dispatch_key")
        q_mode = row.get("q_mode")
        arm = (dispatch_key, q_mode)
        if (
            not _anonymous_group_id(group_id)
            or not isinstance(dispatch_key, str)
            or not isinstance(q_mode, str)
            or arm not in EXPECTED_ARMS
            or not _anonymous_row_id(row_id, group_id, dispatch_key, q_mode)
            or not _valid_measurements(row)
            or row_id in candidates
        ):
            _invalid()
        arms = arms_by_group.setdefault(group_id, set())
        if arm in arms:
            _invalid()
        arms.add(arm)
        candidates[row_id] = (group_id, dispatch_key, q_mode)

    group_ids = set(arms_by_group)
    if (
        not 1 <= len(group_ids) <= MAX_GROUPS
        or len(candidates) != len(group_ids) * len(EXPECTED_ARMS)
        or any(arms != EXPECTED_ARMS for arms in arms_by_group.values())
    ):
        _invalid()
    return candidates, group_ids


def _validate_required_arms(value: Any) -> None:
    raw_arms = _sequence(value)
    if raw_arms is None or len(raw_arms) != len(EXPECTED_ARMS):
        _invalid()
    arms: set[tuple[str, str]] = set()
    for raw_arm in raw_arms:
        arm = _sequence(raw_arm)
        if arm is None or len(arm) != 2 or not all(isinstance(item, str) for item in arm):
            _invalid()
        pair = (arm[0], arm[1])
        if pair in arms:
            _invalid()
        arms.add(pair)
    if arms != EXPECTED_ARMS:
        _invalid()


def _validate_manifest_candidate_binding(
    value: Any, candidate_ids: set[str]
) -> None:
    row_ids = _sequence(value)
    if row_ids is None or len(row_ids) != len(candidate_ids):
        _invalid()
    if (
        any(not isinstance(row_id, str) for row_id in row_ids)
        or len(set(row_ids)) != len(row_ids)
        or set(row_ids) != candidate_ids
    ):
        _invalid()


def _validate_selected_row(
    value: Any,
    candidates: Mapping[str, tuple[str, str, str]],
    fold_groups: set[str],
) -> tuple[str, str, str, str]:
    if not _exact_mapping(value, SELECTION_FIELDS):
        _invalid()
    row_id = value.get("manifest_job_id")
    group_id = value.get("group_id")
    dispatch_key = value.get("dispatch_key")
    q_mode = value.get("q_mode")
    if (
        not _anonymous_group_id(group_id)
        or not isinstance(dispatch_key, str)
        or not isinstance(q_mode, str)
        or (dispatch_key, q_mode) not in EXPECTED_ARMS
        or not _anonymous_row_id(row_id, group_id, dispatch_key, q_mode)
        or group_id not in fold_groups
        or candidates.get(row_id) != (group_id, dispatch_key, q_mode)
    ):
        _invalid()
    return row_id, group_id, dispatch_key, q_mode


def _validate_folds(
    value: Any,
    candidates: Mapping[str, tuple[str, str, str]],
    candidate_groups: set[str],
) -> list[tuple[str, str]]:
    folds = _sequence(value)
    if folds is None or not 1 <= len(folds) <= len(candidate_groups):
        _invalid()

    seen_fold_ids: set[str] = set()
    bound_groups: set[str] = set()
    selected_ids: set[str] = set()
    selected_by_group: dict[str, tuple[str, str]] = {}
    for fold in folds:
        if not _exact_mapping(fold, FOLD_FIELDS):
            _invalid()
        fold_id = fold.get("fold_id")
        raw_groups = _sequence(fold.get("candidate_group_ids"))
        raw_selected = _sequence(fold.get("selected_rows"))
        if (
            not _anonymous_fold_id(fold_id)
            or fold_id in seen_fold_ids
            or raw_groups is None
            or raw_selected is None
            or not raw_groups
            or len(raw_groups) != len(raw_selected)
            or any(not _anonymous_group_id(group_id) for group_id in raw_groups)
        ):
            _invalid()
        seen_fold_ids.add(fold_id)
        fold_groups = set(raw_groups)
        if len(fold_groups) != len(raw_groups) or fold_groups & bound_groups:
            _invalid()
        bound_groups.update(fold_groups)

        fold_selected_groups: set[str] = set()
        for selected in raw_selected:
            row_id, group_id, dispatch_key, q_mode = _validate_selected_row(
                selected, candidates, fold_groups
            )
            if row_id in selected_ids or group_id in fold_selected_groups:
                _invalid()
            selected_ids.add(row_id)
            fold_selected_groups.add(group_id)
            selected_by_group[group_id] = (dispatch_key, q_mode)
        if fold_selected_groups != fold_groups:
            _invalid()

    if bound_groups != candidate_groups or set(selected_by_group) != candidate_groups:
        _invalid()
    return [selected_by_group[group_id] for group_id in sorted(candidate_groups)]


def _validate_replay_policies(value: Any, group_count: int) -> dict[str, dict[str, float | int]]:
    if not _exact_mapping(value, PUBLIC_ACQUISITION_POLICIES):
        _invalid()
    policies: dict[str, dict[str, float | int]] = {}
    for policy in sorted(PUBLIC_ACQUISITION_POLICIES):
        metrics = value.get(policy)
        if not _exact_mapping(metrics, POLICY_METRIC_FIELDS):
            _invalid()
        median = metrics.get("groups_to_95pct_oracle_HV_median")
        success_rate = metrics.get("success_rate")
        if (
            not isinstance(median, int)
            or isinstance(median, bool)
            or not 1 <= median <= group_count
            or not _finite_number(success_rate)
            or not 0.0 <= float(success_rate) <= 1.0
        ):
            _invalid()
        policies[policy] = {
            "groups_to_95pct_oracle_HV_median": median,
            "success_rate": float(success_rate),
        }
    return policies


def _validate_manifest(
    value: Any,
    candidates: Mapping[str, tuple[str, str, str]],
    candidate_groups: set[str],
) -> tuple[str, list[tuple[str, str]], dict[str, dict[str, float | int]], int]:
    if not _exact_mapping(value, MANIFEST_FIELDS):
        _invalid()
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        _invalid()
    policy = value.get("selection_policy")
    if not isinstance(policy, str) or policy not in PUBLIC_ACQUISITION_POLICIES:
        _invalid()
    _validate_required_arms(value.get("required_arm_product"))
    _validate_manifest_candidate_binding(value.get("candidate_row_ids"), set(candidates))
    selected_arms = _validate_folds(value.get("folds"), candidates, candidate_groups)
    replay_policies = _validate_replay_policies(value.get("replay_policies"), len(candidate_groups))
    folds = _sequence(value.get("folds"))
    if folds is None:
        _invalid()
    return policy, selected_arms, replay_policies, len(folds)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def build_selection_completion_report(
    candidate_rows: Sequence[Mapping[str, Any]], selection_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return a frozen, redacted selection-completion report.

    ``candidate_rows`` are complete measured four-arm groups.  The manifest
    commits the candidate IDs, arm product, fold membership, per-fold selected
    rows, permitted selection policy, and replay aggregates.  Any contract
    deviation fails closed with a generic error.
    """
    candidates, candidate_groups = _validate_candidate_rows(candidate_rows)
    policy, selected_arms, replay_policies, fold_count = _validate_manifest(
        selection_manifest, candidates, candidate_groups
    )
    arm_counts: dict[str, int] = {}
    for dispatch_key, q_mode in selected_arms:
        arm_name = f"{dispatch_key}|{q_mode}"
        arm_counts[arm_name] = arm_counts.get(arm_name, 0) + 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "evaluation_row_count": len(candidates),
            "evaluation_group_count": len(candidate_groups),
            "retain_ranker": False,
            "selected_uncertainty_method": "lgbm_quantile",
        },
        "selection": {
            "policy": policy,
            "selected_group_count": len(selected_arms),
            "selected_arm_counts": dict(sorted(arm_counts.items())),
        },
        "replay": {"policies": replay_policies},
        "binding": {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "fold_count": fold_count,
            "required_arm_count": len(EXPECTED_ARMS),
        },
    }
    return _deep_freeze(report)


def run_stage4_completion(
    candidate_rows: Sequence[Mapping[str, Any]], selection_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Compatibility entrypoint for the public selection-completion report."""
    return build_selection_completion_report(candidate_rows, selection_manifest)


__all__ = [
    "EXPECTED_ARMS",
    "MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_selection_completion_report",
    "run_stage4_completion",
]
