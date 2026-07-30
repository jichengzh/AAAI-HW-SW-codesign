"""Hardware-independent contracts for the Stage7 online ablation experiment.

This public slice deliberately contains no execution, cache, compiler, or
cluster interfaces.  It protects the inputs used *before* selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage7_online_component_ablation_v1"
SEEDS = (20260718, 20260719, 20260720)
CORE_VARIANTS = ("full", "without_surrogate", "without_measured_feedback", "backend_blind")
WIDTH_SCHEMA = ("w0", "w1", "w2")
_LABEL_TOKENS = ("latency", "energy", "terminal_status", "failure_reason", "cache", "pareto", "frontier")
_BACKEND_TOKENS = ("backend", "capability", "profile", "dispatch")
_TRUSTED_PROVENANCE = {"architecture", "model_config", "model-config", "static_config", "static-config"}


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def frozen_experiment_contracts() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return independent copies of Full and the three published ablations."""
    full = {
        "schema_version": SCHEMA_VERSION,
        "variant": "full",
        "seeds": list(SEEDS),
        "batch_size": 4,
        "round_count": 4,
        "sample_budget": 16,
        "policy_name": "predicted_frontier_diversity",
        "surrogate_acquisition": "enabled",
        "feedback_refit": "enabled",
        "actual_graph_feature_feedback": "enabled",
        "backend_model_features": "enabled",
    }
    variants = {
        "without_surrogate": {**full, "variant": "without_surrogate", "surrogate_acquisition": "uniform_random_without_replacement"},
        "without_measured_feedback": {**full, "variant": "without_measured_feedback", "feedback_refit": "frozen_initial_bundle", "actual_graph_feature_feedback": "off"},
        "backend_blind": {**full, "variant": "backend_blind", "backend_model_features": "off"},
    }
    return copy.deepcopy(full), copy.deepcopy(variants)


def audit_single_variable_isolation(full: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    """Reject an ablation that changes a non-published control variable."""
    allowed = {
        "without_surrogate": {"variant", "surrogate_acquisition"},
        "without_measured_feedback": {"variant", "feedback_refit", "actual_graph_feature_feedback"},
        "backend_blind": {"variant", "backend_model_features"},
    }
    name = str(variant.get("variant") or "")
    if name not in allowed:
        raise ValueError("unknown Stage7 variant")
    changed = sorted(key for key in set(full) | set(variant) if full.get(key) != variant.get(key))
    if set(changed) != allowed[name]:
        raise ValueError("variant changes fields outside its published ablation")
    return {"variant": name, "changed_fields": changed, "verdict": "pass", "audit_sha256": _sha(changed)}


def _identity(value: Mapping[str, Any] | str) -> str:
    return value if isinstance(value, str) else str(value.get("row_id") or value.get("manifest_job_id") or "")


def a1_select_unselected(scan_pass_candidates: Sequence[Mapping[str, Any] | str], selected_ids: set[str], *, seed: int, batch_size: int = 4) -> list[str]:
    """Select a uniform, unique A1 batch with a local seeded RNG only."""
    if seed not in SEEDS:
        raise ValueError("A1 seed must be one of the frozen Stage7 seeds")
    if batch_size != 4:
        raise ValueError("Stage7 A1 batch size is frozen at four")
    ids = [_identity(candidate) for candidate in scan_pass_candidates]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidates require unique non-empty identities")
    available = [value for value in ids if value not in selected_ids]
    if len(available) < batch_size:
        raise ValueError("fewer than four unselected candidates")
    return random.Random(seed).sample(available, batch_size)


def project_a2_feedback(initial_training_view: Sequence[Mapping[str, Any]], initial_graph_feature_view: Mapping[str, Mapping[str, Any]], *, anchor: Mapping[str, Any], bundle_sha256: str, selected_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Record feedback without changing A2's frozen fitting or graph inputs."""
    if not _is_sha(bundle_sha256):
        raise ValueError("A2 bundle SHA256 must be a 64-character digest")
    recorded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in selected_results:
        row = copy.deepcopy(dict(value))
        identity = _identity(row)
        if not identity:
            raise ValueError("selected feedback results require an identity")
        if identity not in seen:
            recorded.append(row)
            seen.add(identity)
    return {
        "training_view": copy.deepcopy([dict(value) for value in initial_training_view]),
        "graph_feature_view": copy.deepcopy(dict(initial_graph_feature_view)),
        "anchor": copy.deepcopy(dict(anchor)), "bundle_sha256": bundle_sha256,
        "recorded_results": recorded, "feedback_refit": False, "actual_graph_feature_feedback": False,
    }


def _canon(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _is_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _contains_label(value: Any, *, predicted: bool = False) -> bool:
    if isinstance(value, Mapping):
        def forbidden_key(key: Any) -> bool:
            name = _canon(key)
            return (
                any(token in name for token in _LABEL_TOKENS)
                or name == "delta_hv"
                or name.startswith("map")
                or any(part == "ap" or re.fullmatch(r"ap[0-9]+", part) for part in name.split("_") if part)
            )
        return any(
            (not predicted and forbidden_key(key))
            or _contains_label(
                item,
                predicted=predicted or str(key) in {"predictions", "prediction_intervals"},
            )
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_label(item, predicted=predicted) for item in value)
    return False


def selection_candidate_view(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a label/cache-free copy or reject a candidate that leaks one."""
    rows = [copy.deepcopy(dict(value)) for value in candidates]
    if not rows or any(not _identity(row) for row in rows):
        raise ValueError("selection candidates require identities")
    if len({_identity(row) for row in rows}) != len(rows):
        raise ValueError("selection candidates require unique identities")
    if any(_contains_label(row) for row in rows):
        raise ValueError("forbidden label or cache field before selection")
    return rows


def project_backend_blind_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Drop only backend/capability/profile-derived feature columns."""
    full_rows: list[dict[str, float]] = []
    removed: set[str] = set()
    for source in rows:
        row = copy.deepcopy(dict(source))
        width = row.get("width")
        if not isinstance(width, (list, tuple)) or len(width) != 3:
            raise ValueError("backend-blind rows require three widths")
        q_mode = row.get("q_mode")
        if q_mode not in {"fp16", "int8"}:
            raise ValueError("backend-blind rows require fp16 or int8 q_mode")
        provenance = row.get("feature_provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("feature_provenance must be a mapping")
        features = {f"width:{name}": float(value) for name, value in zip(WIDTH_SCHEMA, width)}
        features["q_mode"] = float(q_mode == "int8")
        for container, prefix in (("graph_features", "graph"), ("model_features", "model"), ("features", "feature")):
            values = row.get(container) or {}
            if not isinstance(values, Mapping):
                raise ValueError(f"{container} must be a mapping")
            for name, value in values.items():
                feature = f"{prefix}:{name}"
                source_name = str(provenance.get(name, provenance.get(feature, "")))
                if container != "graph_features" and not source_name:
                    raise ValueError(f"{container} feature provenance missing: {name}")
                if container != "graph_features" and not any(token in _canon(name) or token in _canon(source_name) for token in _BACKEND_TOKENS) and _canon(source_name) not in _TRUSTED_PROVENANCE:
                    raise ValueError(f"{container} feature provenance is unknown: {name}")
                if any(token in _canon(name) or token in _canon(source_name) for token in _BACKEND_TOKENS):
                    removed.add(feature)
                features[feature] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
        full_rows.append(features)
    full_schema = sorted({name for row in full_rows for name in row})
    blind_schema = [name for name in full_schema if name not in removed]
    return {
        "full_schema": full_schema, "blind_schema": blind_schema, "removed_names": sorted(removed),
        "full_matrix_sha256": _sha([[row.get(name, 0.0) for name in full_schema] for row in full_rows]),
        "blind_matrix": [[row.get(name, 0.0) for name in blind_schema] for row in full_rows],
        "blind_matrix_sha256": _sha([[row.get(name, 0.0) for name in blind_schema] for row in full_rows]),
        "leakage_verdict": "no_forbidden_features",
    }
