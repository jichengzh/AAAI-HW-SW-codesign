"""Pure, anonymous uncertainty calibration and acquisition replay for Stage4.

The public evaluator consumes strictly shaped in-memory records.  It does not
load experiment assets, call hardware, or return caller-provided identifiers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
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
FOLD_IDS = frozenset({"f0", "f1", "f2", "f3"})
MANIFEST_FIELDS = frozenset({"schema_version", "folds"})
FOLD_FIELDS = frozenset({"fold_id", "oof_group_id", "candidate_group_id"})
PREDICTION_ENVELOPE_FIELDS = frozenset({"schema_version", "records"})
OOF_RECORD_FIELDS = frozenset(
    {
        "manifest_job_id",
        "group_id",
        "fold_id",
        "dispatch_key",
        "q_mode",
        "target",
        "observed",
        "prediction",
        "lower",
        "upper",
    }
)
CANDIDATE_RECORD_FIELDS = OOF_RECORD_FIELDS - {"observed"}
ANONYMOUS_GROUP = re.compile(r"^g(?:0|[1-9][0-9]{0,3})$")


def _invalid() -> None:
    """Fail closed without reflecting untrusted content in the error."""
    raise ValueError("invalid uncertainty replay input")


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _anonymous_group(value: Any) -> bool:
    return isinstance(value, str) and ANONYMOUS_GROUP.fullmatch(value) is not None


def _manifest_bindings(value: Any) -> tuple[dict[str, str], dict[str, str]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != MANIFEST_FIELDS
        or value.get("schema_version") != "stage4_uncertainty_replay_manifest_v1"
    ):
        _invalid()
    folds = _sequence(value.get("folds"))
    if folds is None or len(folds) != len(FOLD_IDS):
        _invalid()
    oof_bindings: dict[str, str] = {}
    candidate_bindings: dict[str, str] = {}
    seen_folds: set[str] = set()
    for fold in folds:
        if not isinstance(fold, Mapping) or set(fold) != FOLD_FIELDS:
            _invalid()
        fold_id = fold.get("fold_id")
        oof_group = fold.get("oof_group_id")
        candidate_group = fold.get("candidate_group_id")
        if (
            not isinstance(fold_id, str)
            or fold_id not in FOLD_IDS
            or fold_id in seen_folds
            or not _anonymous_group(oof_group)
            or not _anonymous_group(candidate_group)
            or oof_group == candidate_group
            or oof_group in oof_bindings
            or candidate_group in candidate_bindings
        ):
            _invalid()
        seen_folds.add(fold_id)
        oof_bindings[oof_group] = fold_id
        candidate_bindings[candidate_group] = fold_id
    if seen_folds != FOLD_IDS or set(oof_bindings) & set(candidate_bindings):
        _invalid()
    return oof_bindings, candidate_bindings


def _expected_record_keys(bindings: Mapping[str, str]) -> set[tuple[str, str]]:
    return {
        (target, f"{group_id}|{dispatch_key}|{q_mode}")
        for group_id in bindings
        for dispatch_key, q_mode in EXPECTED_ARMS
        for target in TARGETS
    }


def _target_values_valid(target: str, values: Sequence[Any]) -> bool:
    if not all(_finite_number(value) for value in values):
        return False
    numeric_values = tuple(float(value) for value in values)
    if target in {"latency_ms", "energy_j"}:
        return all(value > 0 for value in numeric_values)
    return all(0.0 <= value <= 1.0 for value in numeric_values)


def _validated_predictions(
    value: Any,
    *,
    schema_version: str,
    record_fields: frozenset[str],
    bindings: Mapping[str, str],
    include_observed: bool,
) -> dict[str, list[tuple[str, float, float, float, float | None]]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != PREDICTION_ENVELOPE_FIELDS
        or value.get("schema_version") != schema_version
    ):
        _invalid()
    records = _sequence(value.get("records"))
    if records is None:
        _invalid()
    expected = _expected_record_keys(bindings)
    seen: set[tuple[str, str]] = set()
    parsed = {target: [] for target in TARGETS}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            _invalid()
        group_id = record.get("group_id")
        fold_id = record.get("fold_id")
        dispatch_key = record.get("dispatch_key")
        q_mode = record.get("q_mode")
        target = record.get("target")
        job_id = record.get("manifest_job_id")
        lower = record.get("lower")
        prediction = record.get("prediction")
        upper = record.get("upper")
        observed = record.get("observed") if include_observed else None
        if (
            not _anonymous_group(group_id)
            or bindings.get(group_id) != fold_id
            or not isinstance(dispatch_key, str)
            or not isinstance(q_mode, str)
            or not isinstance(target, str)
            or not isinstance(job_id, str)
            or (dispatch_key, q_mode) not in EXPECTED_ARMS
            or target not in TARGETS
            or job_id != f"{group_id}|{dispatch_key}|{q_mode}"
            or not _target_values_valid(
                target, (lower, prediction, upper, observed)
                if include_observed
                else (lower, prediction, upper),
            )
            or float(lower) > float(prediction)
            or float(prediction) > float(upper)
        ):
            _invalid()
        key = (target, job_id)
        if key in seen:
            _invalid()
        seen.add(key)
        parsed[target].append(
            (
                str(group_id),
                float(lower),
                float(prediction),
                float(upper),
                float(observed) if include_observed else None,
            )
        )
    if seen != expected:
        _invalid()
    return parsed


def _calibration_summary(
    predictions: Mapping[
        str, Sequence[tuple[str, float, float, float, float | None]]
    ]
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for target in TARGETS:
        records = predictions[target]
        if not records or any(observed is None for _, _, _, _, observed in records):
            _invalid()
        observed_records = [
            (lower, prediction, upper, float(observed))
            for _, lower, prediction, upper, observed in records
            if observed is not None
        ]
        summary[target] = {
            "oof_row_count": len(observed_records),
            "mae": sum(
                abs(observed - prediction)
                for _, prediction, _, observed in observed_records
            )
            / len(observed_records),
            "row_coverage": sum(
                lower <= observed <= upper
                for lower, _, upper, observed in observed_records
            )
            / len(observed_records),
            "mean_interval_width": sum(
                upper - lower for lower, _, upper, _ in observed_records
            )
            / len(observed_records),
        }
    return summary


def _frontier_group_count(
    predictions: Mapping[
        str, Sequence[tuple[str, float, float, float, float | None]]
    ],
    candidate_bindings: Mapping[str, str],
) -> int:
    expected_per_target = len(candidate_bindings) * len(EXPECTED_ARMS)
    if any(len(predictions[target]) != expected_per_target for target in TARGETS):
        _invalid()

    by_target_group = {
        target: {group_id: [] for group_id in candidate_bindings} for target in TARGETS
    }
    for target in TARGETS:
        for group_id, lower, prediction, upper, observed in predictions[target]:
            if group_id not in by_target_group[target] or observed is not None:
                _invalid()
            by_target_group[target][group_id].append((lower, prediction, upper))
        if any(
            len(records) != len(EXPECTED_ARMS)
            for records in by_target_group[target].values()
        ):
            _invalid()

    scores: dict[str, tuple[float, float, float]] = {}
    for group_id in candidate_bindings:
        latency = min(record[1] for record in by_target_group["latency_ms"][group_id])
        energy = min(record[1] for record in by_target_group["energy_j"][group_id])
        ap70 = max(record[1] for record in by_target_group["ap70"][group_id])
        scores[group_id] = (latency, energy, ap70)
    frontier = [
        group_id
        for group_id, score in scores.items()
        if not any(
            other != group_id
            and other_score[0] <= score[0]
            and other_score[1] <= score[1]
            and other_score[2] >= score[2]
            and other_score != score
            for other, other_score in scores.items()
        )
    ]
    if not frontier:
        _invalid()
    return len(frontier)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def build_uncertainty_replay(
    manifest: Mapping[str, Any],
    oof_predictions: Mapping[str, Any],
    candidate_predictions: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return immutable calibration and acquisition aggregates for anonymous inputs.

    Candidate records are deliberately unlabeled: the replay ranks only their
    predicted intervals and never imports actual performance into selection.
    """
    oof_bindings, candidate_bindings = _manifest_bindings(manifest)
    oof = _validated_predictions(
        oof_predictions,
        schema_version="stage4_uncertainty_oof_predictions_v1",
        record_fields=OOF_RECORD_FIELDS,
        bindings=oof_bindings,
        include_observed=True,
    )
    candidates = _validated_predictions(
        candidate_predictions,
        schema_version="stage4_uncertainty_candidate_predictions_v1",
        record_fields=CANDIDATE_RECORD_FIELDS,
        bindings=candidate_bindings,
        include_observed=False,
    )
    candidate_group_count = len(candidate_bindings)
    frontier_group_count = _frontier_group_count(candidates, candidate_bindings)
    return _freeze(
        {
            "schema_version": "stage4_uncertainty_replay_v1",
            "uncertainty_method": "lgbm_quantile_plus_group_conformal",
            "calibration": _calibration_summary(oof),
            "acquisition_replay": {
                "candidate_group_count": candidate_group_count,
                "candidate_row_count": candidate_group_count * len(EXPECTED_ARMS),
                "policies": {
                    "random": {
                        "selected_group_count": candidate_group_count,
                        "selected_fraction": 1.0,
                    },
                    "predicted_frontier_diversity": {
                        "selected_group_count": frontier_group_count,
                        "selected_fraction": frontier_group_count / candidate_group_count,
                    },
                },
            },
        }
    )


__all__ = ["build_uncertainty_replay"]
