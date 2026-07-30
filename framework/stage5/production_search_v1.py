"""Production contracts and acquisition for the Stage5 two-model search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from framework.stage2.canonical_search_v3 import validate_capability_profile
from framework.stage4.cost_model_selection_v1 import encode_rows

try:  # pragma: no cover - fallback is covered in lean environments.
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]


SCHEMA_VERSION = "stage5_production_search_v1"
TARGETS = ("latency_ms", "energy_j", "ap70")
EXPECTED_HEADS = {
    "latency_ms": "extra_trees_log",
    "energy_j": "extra_trees_log",
    "ap70": "lgbm_huber_residual",
}
EXPECTED_UNCERTAINTY = "lgbm_quantile_plus_group_conformal"
EXPECTED_ACQUISITION = "predicted_frontier_diversity"
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
ALLOWED_SOURCES = {"initial_coldstart", "online_feedback"}
GRAPH_METADATA_FIELDS = {"group_id", "model", "width", "input_dims"}
FORBIDDEN_LABEL_FIELDS = {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
FORBIDDEN_CANDIDATE_CONTEXT_TOKENS = (
    "latency",
    "energy",
    "ap30",
    "ap50",
    "ap70",
    "cache",
    "terminal_status",
    "measurement_status",
    "performance_status",
)


@dataclass
class ProductionBundle:
    manifest: dict[str, Any]
    feature_names: tuple[str, ...]
    graph_feature_names: tuple[str, ...]
    value_heads: dict[str, Any]
    interval_heads: dict[str, tuple[Any, Any, Any]]
    conformal_corrections: dict[str, dict[str, float]]
    model_anchors: dict[str, float]


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _validate_profiles(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated = [validate_capability_profile(profile) for profile in profiles]
    if {str(profile["dispatch_key"]) for profile in validated} != {
        "tvm_auto",
        "trt_engine",
    }:
        raise ValueError("Stage5 requires exactly the frozen TVM-auto/TRT capability pair")
    if len(validated) != 2:
        raise ValueError("Stage5 requires exactly two capability profiles")
    return validated


def _validate_graph_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("graph feature record must be an object")
    forbidden = {
        str(name)
        for name in payload
        if str(name).lower() in FORBIDDEN_LABEL_FIELDS
        or str(name).lower().startswith(("target_", "observed_", "measured_"))
    }
    if forbidden:
        raise ValueError("label-like graph features are forbidden")
    for name, value in payload.items():
        if name in GRAPH_METADATA_FIELDS:
            continue
        if not _finite(value):
            raise ValueError("graph feature values must be finite numeric values")


def _validate_graph_features(graph_features: Sequence[Mapping[str, Any]]) -> None:
    identities: set[str] = set()
    for item in graph_features:
        _validate_graph_payload(item)
        group_id = item.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("graph feature group_id must be a non-empty string")
        if group_id in identities:
            raise ValueError("duplicate graph feature group identity")
        identities.add(group_id)


def _validate_prediction_payload(row: Mapping[str, Any]) -> None:
    predictions = row.get("predictions")
    intervals = row.get("prediction_intervals")
    if not isinstance(predictions, Mapping) or not isinstance(intervals, Mapping):
        raise ValueError("candidate predictions and intervals must be objects")
    if set(predictions) != set(TARGETS):
        raise ValueError("candidate predictions must contain exactly the frozen targets")
    if set(intervals) != set(TARGETS):
        raise ValueError(
            "candidate prediction intervals must contain exactly the frozen targets"
        )
    for target in TARGETS:
        if not _finite(predictions.get(target)):
            raise ValueError("candidate predictions must be finite")
        interval = intervals.get(target)
        if not isinstance(interval, Mapping):
            raise ValueError("candidate prediction intervals must be objects")
        if set(interval) != {"lower", "median", "upper"}:
            raise ValueError(
                "candidate prediction interval must contain exactly lower, median, and upper"
            )
        values = tuple(interval.get(name) for name in ("lower", "median", "upper"))
        if not all(_finite(value) for value in values):
            raise ValueError("candidate prediction intervals must be finite")
        lower, median, upper = (float(value) for value in values)
        if not lower <= median <= upper:
            raise ValueError("candidate prediction intervals must be ordered")


def _validate_candidate_context(value: Any) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            lowered = str(name).lower()
            if lowered in {"predictions", "prediction_intervals"}:
                continue
            if lowered != "source_status" and any(
                token in lowered for token in FORBIDDEN_CANDIDATE_CONTEXT_TOKENS
            ):
                raise ValueError("forbidden candidate field before selection")
            _validate_candidate_context(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _validate_candidate_context(item)


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        groups[str(source["group_id"])].append(dict(source))
    return dict(groups)


def _validate_complete_groups(rows: Sequence[Mapping[str, Any]]) -> None:
    groups = _group_rows(rows)
    profile_dispatch: dict[str, str] = {}
    for group_id, group_rows in groups.items():
        arms = {
            (str(row.get("dispatch_key")), str(row.get("q_mode")))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != EXPECTED_ARMS:
            raise ValueError("incomplete four-arm group")
        identities = {
            (
                str(row.get("model") or ""),
                tuple(int(value) for value in row.get("width") or []),
            )
            for row in group_rows
        }
        if len(identities) != 1:
            raise ValueError("four-arm group identity drift")
        model, width = next(iter(identities))
        if len(width) != 3 or group_id != f"{model}|{'x'.join(map(str, width))}":
            raise ValueError("four-arm group identity drift")
        for row in group_rows:
            profile = str(row.get("capability_profile_id") or "")
            dispatch = str(row.get("dispatch_key") or "")
            if not profile or (
                profile in profile_dispatch and profile_dispatch[profile] != dispatch
            ):
                raise ValueError("capability profile/dispatch identity drift")
            profile_dispatch[profile] = dispatch
        ids = [_row_id(row) for row in group_rows]
        if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("empty or duplicate row identity")


def _value_training_view(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    excluded: list[str] = []
    for group_id, group_rows in sorted(_group_rows(rows).items()):
        if all(
            str(row.get("terminal_status")) == "measured_success_gold"
            and all(_finite(row.get(target)) for target in TARGETS)
            for row in group_rows
        ):
            selected.extend(group_rows)
        else:
            invalid_statuses = {
                str(row.get("terminal_status") or "")
                for row in group_rows
                if str(row.get("terminal_status")) != "measured_success_gold"
            }
            if invalid_statuses - {"feasibility_failure", "numerical_feasibility_failure"}:
                raise ValueError("unsupported non-value terminal status")
            excluded.append(group_id)
    if not selected:
        raise ValueError("no complete finite four-arm groups for value-model training")
    return selected, excluded


def validate_stage5_contract(
    closure: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    training_view_policy: str = "inherit_stage4_feedback",
) -> dict[str, Any]:
    if closure.get("schema_version") != "stage4_p1_p3_closure_audit_v1":
        raise ValueError("unexpected Stage4 closure schema")
    if closure.get("stage4_closed") is not True or closure.get("stage5_search_ready") is not True:
        raise ValueError("Stage4 closure does not admit Stage5 search")
    if closure.get("canonical_value_heads") != EXPECTED_HEADS:
        raise ValueError(f"Stage5 requires frozen heads: {EXPECTED_HEADS}")
    if closure.get("uncertainty_policy") != EXPECTED_UNCERTAINTY:
        raise ValueError(f"Stage5 requires {EXPECTED_UNCERTAINTY}")
    if closure.get("selected_acquisition_policy") != EXPECTED_ACQUISITION:
        raise ValueError(f"Stage5 requires {EXPECTED_ACQUISITION}")
    _validate_profiles(capability_profiles)
    _validate_graph_features(graph_features)
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        raise ValueError("Stage5 training rows must not be empty")
    _validate_complete_groups(source_rows)
    for row in source_rows:
        for target in TARGETS:
            value = row.get(target)
            if value is not None and not _finite(value):
                raise ValueError("Stage5 measured metrics must be finite")
        if str(row.get("terminal_status")) == "measured_success_gold" and not all(
            _finite(row.get(target)) for target in TARGETS
        ):
            raise ValueError("successful Stage5 measured metrics must be finite")
    if any(str(row.get("q_mode")) not in {"fp16", "int8"} for row in source_rows):
        raise ValueError("Stage5 q_mode must be fp16 or int8")
    if any(
        row.get("mixed_policy_id") not in {None, "", "none"}
        for row in source_rows
    ):
        raise ValueError("mixed policy is forbidden in Stage5 production training")
    value_rows, excluded_value_groups = _value_training_view(source_rows)
    sources: dict[str, int] = defaultdict(int)
    for row in source_rows:
        source = str(row.get("training_source") or "")
        if source not in ALLOWED_SOURCES:
            raise ValueError("unexpected training source role")
        sources[source] += 1
    if training_view_policy not in {"inherit_stage4_feedback", "initial_coldstart_only"}:
        raise ValueError("unsupported Stage5 training view policy")
    expected_sources = closure.get("training_source_rows")
    excluded_stage4_feedback_rows = 0
    if isinstance(expected_sources, Mapping):
        expected = {str(key): int(value) for key, value in expected_sources.items()}
        if sources.get("initial_coldstart", 0) != expected.get("initial_coldstart", 0):
            raise ValueError("initial_coldstart count drift from Stage4 closure")
        if training_view_policy == "inherit_stage4_feedback":
            if sources.get("online_feedback", 0) < expected.get("online_feedback", 0):
                raise ValueError("online_feedback cannot remove Stage4 evidence")
        else:
            if sources.get("online_feedback", 0) != 0 or set(sources) != {"initial_coldstart"}:
                raise ValueError("initial_coldstart_only forbids online or diagnostic evidence")
            excluded_stage4_feedback_rows = expected.get("online_feedback", 0)
    graph_by_group = {str(item["group_id"]): item for item in graph_features}
    missing_graphs = set(_group_rows(source_rows)) - set(graph_by_group)
    if missing_graphs:
        raise ValueError("training graph features missing")
    return {
        "schema_version": "stage5_input_contract_audit_v1",
        "training_row_count": len(source_rows),
        "group_count": len(_group_rows(source_rows)),
        "value_training_row_count": len(value_rows),
        "value_training_group_count": len(_group_rows(value_rows)),
        "excluded_value_groups": excluded_value_groups,
        "training_source_rows": dict(sorted(sources.items())),
        "training_view_policy": training_view_policy,
        "excluded_stage4_feedback_rows": excluded_stage4_feedback_rows,
        "four_arm_groups_complete": True,
        "canonical_value_heads": dict(EXPECTED_HEADS),
        "uncertainty_policy": EXPECTED_UNCERTAINTY,
        "acquisition_policy": EXPECTED_ACQUISITION,
        "training_rows_sha256": _sha(source_rows),
        "graph_features_sha256": _sha(list(graph_features)),
        "capability_profiles_sha256": _sha(list(capability_profiles)),
    }


def build_candidate_manifest(
    source_registry: Mapping[str, Any],
    *,
    measured_group_ids: set[str],
    frozen_holdout: Mapping[str, Any],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if source_registry.get("schema_version") != "stage5_candidate_source_registry_v1":
        raise ValueError("unexpected Stage5 candidate source registry schema")
    profiles = _validate_profiles(capability_profiles)
    holdout_ids = {
        str(group["group_id"])
        for group in (frozen_holdout.get("groups") or [])
    }
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("candidate source registry groups must be a list")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    observed_ids: set[str] = set()
    for source in groups:
        group = copy.deepcopy(dict(source))
        group_id = str(group.get("group_id") or "")
        if not group_id or group_id in observed_ids:
            raise ValueError("candidate source registry has empty or duplicate group_id")
        observed_ids.add(group_id)
        model = str(group.get("model") or "")
        width = [int(value) for value in group.get("width") or []]
        if model not in {"pyramid", "codriving"} or len(width) != 3:
            raise ValueError("invalid candidate identity")
        if group_id != f"{model}|{'x'.join(map(str, width))}":
            raise ValueError("candidate group identity mismatch")
        if group_id in measured_group_ids:
            excluded.append({"group_id": group_id, "reason": "already_measured"})
            continue
        if group_id in holdout_ids:
            excluded.append({"group_id": group_id, "reason": "frozen_independent_holdout"})
            continue
        if group.get("source_status") not in {"ready", "materializable"}:
            excluded.append({"group_id": group_id, "reason": "source_not_ready"})
            continue
        if not isinstance(group.get("graph_features"), Mapping):
            raise ValueError("candidate graph features missing")
        evidence_sha = str(group.get("source_evidence_sha256") or "")
        if not _is_sha256(evidence_sha):
            raise ValueError("candidate source evidence SHA256 is invalid")
        eligible.append(group)
    rows = []
    for group in eligible:
        for profile in profiles:
            for q_mode in ("fp16", "int8"):
                profile_id = str(profile["capability_profile_id"])
                group_id = str(group["group_id"])
                row_id = f"{group_id}|q={q_mode}|profile={profile_id}"
                rows.append(
                    {
                        "schema_version": "stage5_candidate_row_v1",
                        "row_id": row_id,
                        "manifest_job_id": row_id,
                        "group_id": group_id,
                        "model": group["model"],
                        "width": list(group["width"]),
                        "genome": [*group["width"], q_mode],
                        "strategy_id": f"q={q_mode}",
                        "q_mode": q_mode,
                        "capability_profile_id": profile_id,
                        "dispatch_key": profile["dispatch_key"],
                        "capability_digest": profile["capability_digest"],
                        "source_status": group["source_status"],
                        "source_contract": copy.deepcopy(group["source_contract"]),
                        "source_evidence_sha256": group["source_evidence_sha256"],
                        "graph_features": copy.deepcopy(group["graph_features"]),
                    }
                )
    _validate_complete_groups(rows) if rows else None
    return {
        "schema_version": "stage5_candidate_manifest_v1",
        "registry_group_count": len(groups),
        "eligible_group_count": len(eligible),
        "eligible_row_count": len(rows),
        "excluded_groups": excluded,
        "rows": sorted(rows, key=_row_id),
    }


def _stable_calibration_groups(rows: Sequence[Mapping[str, Any]], seed: int) -> set[str]:
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_model[str(row["model"])].add(str(row["group_id"]))
    calibration: set[str] = set()
    for model, groups in sorted(by_model.items()):
        ordered = sorted(
            groups,
            key=lambda group_id: hashlib.sha256(f"{seed}|{model}|{group_id}".encode()).hexdigest(),
        )
        if len(ordered) < 2:
            raise ValueError("at least two groups are required per model")
        calibration.update(ordered[: max(1, math.ceil(0.20 * len(ordered)))])
    return calibration


def _extra_trees(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )


def _lgbm_huber(seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="huber",
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=7,
            max_depth=4,
            min_child_samples=4,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=-1,
            random_state=seed,
            n_jobs=1,
        )
    return GradientBoostingRegressor(loss="huber", random_state=seed)


def _quantile(alpha: float, seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=8,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=4,
            min_child_samples=2,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=8,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
    )


def _predict_model(model: Any, matrix: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        return np.asarray(model.predict(matrix), dtype=float)


def _graph_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name.removeprefix("graph:") for name in feature_names if name.startswith("graph:"))


def fit_production_bundle(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    *,
    seed: int = 20260717,
    training_view_policy: str = "inherit_stage4_feedback",
) -> ProductionBundle:
    contract = validate_stage5_contract(
        closure,
        rows,
        graph_features,
        capability_profiles,
        training_view_policy=training_view_policy,
    )
    all_source_rows = [dict(row) for row in rows]
    source_rows, excluded_value_groups = _value_training_view(all_source_rows)
    encoded = encode_rows(source_rows, graph_features, capability_profiles)
    x = encoded.matrix
    anchors = {
        model: float(
            np.median(
                [float(row["ap70"]) for row in source_rows if str(row["model"]) == model]
            )
        )
        for model in sorted({str(row["model"]) for row in source_rows})
    }
    value_heads: dict[str, Any] = {}
    for target in ("latency_ms", "energy_j"):
        model = _extra_trees(seed)
        model.fit(x, np.log1p([float(row[target]) for row in source_rows]))
        value_heads[target] = model
    ap_model = _lgbm_huber(seed)
    ap_model.fit(
        x,
        np.asarray(
            [float(row["ap70"]) - anchors[str(row["model"])] for row in source_rows],
            dtype=float,
        ),
    )
    value_heads["ap70"] = ap_model

    calibration_groups = _stable_calibration_groups(source_rows, seed)
    fit_indices = [
        index for index, row in enumerate(source_rows) if str(row["group_id"]) not in calibration_groups
    ]
    calibration_indices = [
        index for index, row in enumerate(source_rows) if str(row["group_id"]) in calibration_groups
    ]
    interval_heads: dict[str, tuple[Any, Any, Any]] = {}
    corrections: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(TARGETS):
        models = tuple(_quantile(alpha, seed + target_index) for alpha in (0.05, 0.50, 0.95))
        y_fit = np.asarray([float(source_rows[index][target]) for index in fit_indices], dtype=float)
        for model in models:
            model.fit(x[fit_indices], y_fit)
        interval_heads[target] = models
        predictions = np.sort(
            np.vstack([_predict_model(model, x[calibration_indices]) for model in models]), axis=0
        )
        scores_by_model_group: dict[tuple[str, str], list[float]] = defaultdict(list)
        for local_index, row_index in enumerate(calibration_indices):
            row = source_rows[row_index]
            truth = float(row[target])
            score = max(
                float(predictions[0, local_index]) - truth,
                truth - float(predictions[2, local_index]),
                0.0,
            )
            scores_by_model_group[(str(row["model"]), str(row["group_id"]))].append(score)
        scores_by_model: dict[str, list[float]] = defaultdict(list)
        for (model_name, _), scores in scores_by_model_group.items():
            scores_by_model[model_name].append(max(scores))
        corrections[target] = {
            model_name: float(max(scores))
            for model_name, scores in scores_by_model.items()
        }

    manifest = {
        "schema_version": "stage5_production_model_bundle_manifest_v1",
        "canonical_value_heads": dict(EXPECTED_HEADS),
        "uncertainty_policy": EXPECTED_UNCERTAINTY,
        "acquisition_policy": EXPECTED_ACQUISITION,
        "target_transforms": {
            "latency_ms": "log1p",
            "energy_j": "log1p",
            "ap70": "residual_from_model_anchor",
        },
        "seed": seed,
        "feature_names": list(encoded.feature_names),
        "calibration_groups": sorted(calibration_groups),
        "input_contract_row_count": len(all_source_rows),
        "value_training_row_count": len(source_rows),
        "value_training_group_count": len(_group_rows(source_rows)),
        "excluded_value_groups": excluded_value_groups,
        "training_contract": contract,
    }
    manifest["bundle_config_sha256"] = _sha(manifest)
    return ProductionBundle(
        manifest=manifest,
        feature_names=encoded.feature_names,
        graph_feature_names=_graph_feature_names(encoded.feature_names),
        value_heads=value_heads,
        interval_heads=interval_heads,
        conformal_corrections=corrections,
        model_anchors=anchors,
    )


def _candidate_matrix(
    bundle: ProductionBundle,
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    graphs = []
    seen: set[str] = set()
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in seen:
            continue
        seen.add(group_id)
        source = row.get("graph_features") or {}
        _validate_graph_payload(source)
        graph = {
            "group_id": group_id,
            "model": row["model"],
            "width": list(row["width"]),
            **{
                name: float(source.get(name, 0.0)) if _finite(source.get(name)) else 0.0
                for name in bundle.graph_feature_names
            },
        }
        graphs.append(graph)
    encoded = encode_rows(rows, graphs, profiles)
    if encoded.feature_names != bundle.feature_names:
        raise ValueError("candidate feature schema drift from frozen Stage4 feature schema")
    return encoded.matrix


def predict_candidate_rows(
    bundle: ProductionBundle,
    rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [copy.deepcopy(dict(row)) for row in rows]
    for row in candidates:
        if FORBIDDEN_LABEL_FIELDS & set(row):
            raise ValueError("candidate labels visible before measurement")
        _validate_candidate_context(row)
    matrix = _candidate_matrix(bundle, candidates, capability_profiles)
    value_predictions = {
        target: np.maximum(0.0, np.expm1(_predict_model(bundle.value_heads[target], matrix)))
        for target in ("latency_ms", "energy_j")
    }
    ap_residual = _predict_model(bundle.value_heads["ap70"], matrix)
    value_predictions["ap70"] = np.asarray(
        [
            float(ap_residual[index]) + bundle.model_anchors[str(row["model"])]
            for index, row in enumerate(candidates)
        ],
        dtype=float,
    )
    interval_predictions: dict[str, np.ndarray] = {}
    for target, models in bundle.interval_heads.items():
        interval_predictions[target] = np.sort(
            np.vstack([_predict_model(model, matrix) for model in models]), axis=0
        )
    results = []
    for index, row in enumerate(candidates):
        model_name = str(row["model"])
        intervals = {}
        for target in TARGETS:
            values = interval_predictions[target]
            correction = float(bundle.conformal_corrections[target].get(model_name, 0.0))
            lower = float(values[0, index] - correction)
            median = float(values[1, index])
            upper = float(values[2, index] + correction)
            intervals[target] = {"lower": lower, "median": median, "upper": upper}
        results.append(
            {
                **row,
                "predictions": {
                    target: float(value_predictions[target][index]) for target in TARGETS
                },
                "prediction_intervals": intervals,
                "prediction_bundle_sha256": bundle.manifest["bundle_config_sha256"],
            }
        )
    return results


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def _frontier_indices(vectors: Sequence[Sequence[float]]) -> set[int]:
    return {
        index
        for index, vector in enumerate(vectors)
        if not any(
            other != index and _dominates(vectors[other], vector)
            for other in range(len(vectors))
        )
    }


def _group_feature_vectors(
    rows: Sequence[Mapping[str, Any]], graph_features: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    graph_by_group = {str(item["group_id"]): item for item in graph_features}
    graph_names = sorted(
        {
            str(name)
            for graph in graph_by_group.values()
            for name, value in graph.items()
            if name not in {"group_id", "model", "width", "input_dims"} and _finite(value)
        }
    )
    result = {}
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in result:
            continue
        graph = row.get("graph_features") or graph_by_group.get(group_id) or {}
        result[group_id] = np.asarray(
            [*[float(value) for value in row["width"]], *[float(graph.get(name, 0.0)) for name in graph_names]],
            dtype=float,
        )
    return result


def select_predicted_frontier_diversity(
    predicted_rows: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    measured_graph_features: Sequence[Mapping[str, Any]],
    *,
    group_budget_by_model: Mapping[str, int],
) -> dict[str, Any]:
    candidates = [copy.deepcopy(dict(row)) for row in predicted_rows]
    _validate_complete_groups(candidates)
    for row in candidates:
        leaked = FORBIDDEN_LABEL_FIELDS & set(row)
        if leaked:
            raise ValueError(f"candidate labels visible before measurement: {sorted(leaked)}")
        _validate_candidate_context(row)
        _validate_prediction_payload(row)
        _validate_graph_payload(row.get("graph_features") or {})
    groups = _group_rows(candidates)
    all_feature_rows = [*candidates, *[dict(row) for row in measured_rows]]
    vectors = _group_feature_vectors(all_feature_rows, measured_graph_features)
    matrix = np.vstack(list(vectors.values()))
    span = np.maximum(np.ptp(matrix, axis=0), 1e-12)
    normalized = {
        group_id: (vector - np.min(matrix, axis=0)) / span
        for group_id, vector in vectors.items()
    }
    measured_by_model: dict[str, list[str]] = defaultdict(list)
    for row in measured_rows:
        group_id = str(row["group_id"])
        if group_id not in measured_by_model[str(row["model"])]:
            measured_by_model[str(row["model"])].append(group_id)
    frontier_hits = {group_id: 0 for group_id in groups}
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_scope[(str(row["model"]), str(row["capability_profile_id"]))].append(row)
    for scope_rows in by_scope.values():
        objective_vectors = [
            (
                float(row["predictions"]["latency_ms"]),
                float(row["predictions"]["energy_j"]),
                -float(row["predictions"]["ap70"]),
            )
            for row in scope_rows
        ]
        for index in _frontier_indices(objective_vectors):
            frontier_hits[str(scope_rows[index]["group_id"])] += 1
    diagnostics: dict[str, dict[str, Any]] = {}
    for group_id, group_rows in groups.items():
        model = str(group_rows[0]["model"])
        references = measured_by_model.get(model) or []
        diversity = min(
            (float(np.linalg.norm(normalized[group_id] - normalized[item])) for item in references),
            default=0.0,
        )
        uncertainty = max(
            sum(
                max(
                    0.0,
                    float(row["prediction_intervals"][target]["upper"])
                    - float(row["prediction_intervals"][target]["lower"]),
                )
                / max(abs(float(row["predictions"][target])), 1e-9)
                for target in TARGETS
            )
            for row in group_rows
        )
        diagnostics[group_id] = {
            "model": model,
            "predicted_frontier_hits": int(frontier_hits[group_id]),
            "feature_diversity": diversity,
            "uncertainty": uncertainty,
        }
    selected_ids = []
    for model, budget in sorted(group_budget_by_model.items()):
        if budget <= 0:
            raise ValueError("group budget must be positive")
        model_groups = [
            group_id for group_id, payload in diagnostics.items() if payload["model"] == model
        ]
        if len(model_groups) < budget:
            raise ValueError("not enough eligible candidate groups")
        ranked = sorted(
            model_groups,
            key=lambda group_id: (
                -diagnostics[group_id]["predicted_frontier_hits"],
                -diagnostics[group_id]["feature_diversity"],
                -diagnostics[group_id]["uncertainty"],
                group_id,
            ),
        )
        selected_ids.extend(ranked[:budget])
    selected_rows = [row for row in candidates if str(row["group_id"]) in selected_ids]
    selected_groups = [
        {
            "group_id": group_id,
            "diagnostics": diagnostics[group_id],
            "rows": sorted(groups[group_id], key=_row_id),
        }
        for group_id in sorted(selected_ids)
    ]
    return {
        "schema_version": "stage5_predicted_frontier_diversity_selection_v1",
        "policy": EXPECTED_ACQUISITION,
        "candidate_labels_visible_before_measurement": False,
        "selected_group_ids": sorted(selected_ids),
        "selected_rows": sorted(selected_rows, key=_row_id),
        "groups": selected_groups,
        "all_group_diagnostics": diagnostics,
    }
