"""Public, prediction-only ranking and Pareto selection for Stage4.

This module is deliberately small and self-contained.  It validates a closed
in-memory prediction contract before it ranks candidates, so measured labels,
artifact locations, and other execution context cannot enter a selection.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "stage4_ranking_pareto_v1"
PREDICTION_SCHEMA_VERSION = "stage4_candidate_predictions_v1"
MANIFEST_SCHEMA_VERSION = "stage4_ranking_pareto_manifest_v1"
FOLD_SCHEMA_VERSION = "stage4_ranking_pareto_fold_v1"
POLICY = "predicted_frontier_diversity"
# Pareto construction compares candidate pairs, so this cap also bounds its
# worst-case work to 1,024 candidates rather than admitting a quadratic DoS.
MAX_GROUPS = 256

ARMS = frozenset(
    {
        ("tvm_auto", "fp16"),
        ("tvm_auto", "int8"),
        ("trt_engine", "fp16"),
        ("trt_engine", "int8"),
    }
)
WEIGHT_KEYS = frozenset({"latency_ms", "energy_j", "ap70", "uncertainty"})
DEFAULT_WEIGHTS = {
    "latency_ms": 1.0,
    "energy_j": 1.0,
    "ap70": 1.0,
    "uncertainty": 1.0,
}
_GROUP_ID = re.compile(r"^g(?:0|[1-9][0-9]{0,3})$")
_FOLD_ID = re.compile(r"^f(?:0|[1-9][0-9]{0,3})$")
_ERROR = "invalid public ranking/Pareto contract"


class FrozenDict(dict[str, Any]):
    """A JSON-serializable mapping that rejects public mutation operations."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("ranking/Pareto results are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def copy(self) -> "FrozenDict":
        return self


def _fail() -> None:
    raise ValueError(_ERROR)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _record_sequence(
    value: Any, *, max_length: int | None = None
) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    try:
        if max_length is not None and len(value) > max_length:
            return None
        records = tuple(value)
    except Exception:
        return None
    if max_length is not None and len(records) > max_length:
        return None
    if not all(isinstance(record, Mapping) for record in records):
        return None
    return records


def _anonymous_group_id(value: Any) -> bool:
    return isinstance(value, str) and _GROUP_ID.fullmatch(value) is not None


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _validated_group_ids(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    try:
        if len(value) > MAX_GROUPS:
            return None
        values = tuple(value)
    except Exception:
        return None
    if len(values) > MAX_GROUPS:
        return None
    if not values or not all(_anonymous_group_id(group_id) for group_id in values):
        return None
    if len(set(values)) != len(values):
        return None
    return values


def _validate_manifest(manifest: Any, actual_groups: frozenset[str]) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != {"schema_version", "group_ids"}:
        _fail()
    group_ids = _validated_group_ids(manifest.get("group_ids"))
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or group_ids is None
        or frozenset(group_ids) != actual_groups
    ):
        _fail()


def _validate_fold(fold: Any, actual_groups: frozenset[str]) -> None:
    if not isinstance(fold, Mapping) or set(fold) != {
        "schema_version",
        "fold_id",
        "group_ids",
    }:
        _fail()
    group_ids = _validated_group_ids(fold.get("group_ids"))
    if (
        fold.get("schema_version") != FOLD_SCHEMA_VERSION
        or not isinstance(fold.get("fold_id"), str)
        or _FOLD_ID.fullmatch(fold["fold_id"]) is None
        or group_ids is None
        or frozenset(group_ids) != actual_groups
    ):
        _fail()


def _validate_weights(weights: Any) -> dict[str, float]:
    if not isinstance(weights, Mapping) or set(weights) != WEIGHT_KEYS:
        _fail()
    normalized = {name: float(value) for name, value in weights.items() if _finite_number(value)}
    if len(normalized) != len(WEIGHT_KEYS) or any(value < 0.0 for value in normalized.values()):
        _fail()
    if not any(value > 0.0 for value in normalized.values()):
        _fail()
    return normalized


def _validate_predictions(prediction_set: Any) -> tuple[tuple[dict[str, Any], ...], frozenset[str]]:
    if not isinstance(prediction_set, Mapping) or set(prediction_set) != {
        "schema_version",
        "candidates",
    }:
        _fail()
    candidates = _record_sequence(
        prediction_set.get("candidates"), max_length=MAX_GROUPS * len(ARMS)
    )
    if prediction_set.get("schema_version") != PREDICTION_SCHEMA_VERSION or not candidates:
        _fail()

    normalized: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    arms_by_group: dict[str, set[tuple[str, str]]] = {}
    for candidate in candidates:
        if set(candidate) != {
            "candidate_id",
            "group_id",
            "backend",
            "precision",
            "prediction",
        }:
            _fail()
        group_id = candidate.get("group_id")
        backend = candidate.get("backend")
        precision = candidate.get("precision")
        candidate_id = candidate.get("candidate_id")
        prediction = candidate.get("prediction")
        if (
            not _anonymous_group_id(group_id)
            or not isinstance(backend, str)
            or not isinstance(precision, str)
            or (backend, precision) not in ARMS
            or not isinstance(candidate_id, str)
            or candidate_id != f"{group_id}:{backend}:{precision}"
            or candidate_id in candidate_ids
            or not isinstance(prediction, Mapping)
            or set(prediction) != {"latency_ms", "energy_j", "ap70", "uncertainty"}
        ):
            _fail()
        latency_ms = prediction.get("latency_ms")
        energy_j = prediction.get("energy_j")
        ap70 = prediction.get("ap70")
        uncertainty = prediction.get("uncertainty")
        if (
            not _finite_number(latency_ms)
            or float(latency_ms) <= 0.0
            or not _finite_number(energy_j)
            or float(energy_j) <= 0.0
            or not _finite_number(ap70)
            or not 0.0 <= float(ap70) <= 1.0
            or not _finite_number(uncertainty)
            or float(uncertainty) < 0.0
        ):
            _fail()
        arms = arms_by_group.setdefault(group_id, set())
        if len(arms_by_group) > MAX_GROUPS:
            _fail()
        arm = (backend, precision)
        if arm in arms:
            _fail()
        arms.add(arm)
        candidate_ids.add(candidate_id)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "group_id": group_id,
                "latency_ms": float(latency_ms),
                "energy_j": float(energy_j),
                "ap70": float(ap70),
                "uncertainty": float(uncertainty),
            }
        )
    if any(arms != ARMS for arms in arms_by_group.values()):
        _fail()
    return tuple(normalized), frozenset(arms_by_group)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        left["latency_ms"] <= right["latency_ms"]
        and left["energy_j"] <= right["energy_j"]
        and left["ap70"] >= right["ap70"]
        and left["uncertainty"] <= right["uncertainty"]
    )
    strictly_better = (
        left["latency_ms"] < right["latency_ms"]
        or left["energy_j"] < right["energy_j"]
        or left["ap70"] > right["ap70"]
        or left["uncertainty"] < right["uncertainty"]
    )
    return bool(no_worse and strictly_better)


def _utility(candidate: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return (
        weights["ap70"] * candidate["ap70"]
        - weights["latency_ms"] * candidate["latency_ms"]
        - weights["energy_j"] * candidate["energy_j"]
        - weights["uncertainty"] * candidate["uncertainty"]
    )


def build_stage4_ranking_pareto_report(
    prediction_set: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    fold: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    policy: str = POLICY,
) -> FrozenDict:
    """Rank validated anonymous candidates and compute the prediction Pareto frontier.

    The accepted input is a ``stage4_candidate_predictions_v1`` envelope.  Every
    anonymous group must contain exactly the four public backend/precision arms;
    candidate data can contain only predicted latency, energy, AP, and
    uncertainty.  The immutable return value intentionally contains selections
    and aggregate counts only, never the submitted prediction values.
    """
    if policy != POLICY:
        _fail()
    candidates, group_ids = _validate_predictions(prediction_set)
    _validate_manifest(manifest, group_ids)
    _validate_fold(fold, group_ids)
    active_weights = _validate_weights(DEFAULT_WEIGHTS if weights is None else weights)

    ordered_candidates = tuple(sorted(candidates, key=lambda candidate: candidate["candidate_id"]))
    frontier_ids = frozenset(
        candidate["candidate_id"]
        for candidate in ordered_candidates
        if not any(
            other["candidate_id"] != candidate["candidate_id"]
            and _dominates(other, candidate)
            for other in ordered_candidates
        )
    )
    ranked = tuple(
        candidate["candidate_id"]
        for candidate in sorted(
            ordered_candidates,
            key=lambda candidate: (
                candidate["candidate_id"] not in frontier_ids,
                -_utility(candidate, active_weights),
                candidate["candidate_id"],
            ),
        )
    )
    frontier = tuple(sorted(frontier_ids))
    frontier_groups = frozenset(candidate_id.split(":", 1)[0] for candidate_id in frontier)
    result = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(ordered_candidates),
        "group_count": len(group_ids),
        "policy": POLICY,
        "ranked_candidate_ids": ranked,
        "pareto": {
            "frontier_candidate_ids": frontier,
            "summary": {
                "candidate_count": len(ordered_candidates),
                "frontier_candidate_count": len(frontier),
                "frontier_group_count": len(frontier_groups),
            },
        },
        "ranker_value_comparison": {
            "retain_ranker": False,
            "decision": "prediction_value_heads_only",
        },
        "summary": {
            "retain_ranker": False,
            "selection_policy": POLICY,
            "candidate_count": len(ordered_candidates),
            "group_count": len(group_ids),
            "frontier_candidate_count": len(frontier),
        },
    }
    return _frozen(result)


def rank_pareto_candidates(
    prediction_set: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    fold: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    policy: str = POLICY,
) -> FrozenDict:
    """Public alias for :func:`build_stage4_ranking_pareto_report`."""
    return build_stage4_ranking_pareto_report(
        prediction_set,
        manifest=manifest,
        fold=fold,
        weights=weights,
        policy=policy,
    )


__all__ = [
    "FOLD_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_GROUPS",
    "POLICY",
    "PREDICTION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_stage4_ranking_pareto_report",
    "rank_pareto_candidates",
]
