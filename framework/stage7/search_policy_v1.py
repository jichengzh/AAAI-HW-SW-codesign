"""Selection-only adapters for the frozen Stage7 online-ablation contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from framework.stage7.online_component_ablation_v1 import (
    SEEDS,
    a1_select_unselected,
    frozen_experiment_contracts,
    project_backend_blind_features,
    project_a2_feedback,
    selection_candidate_view,
)


SCHEMA_VERSION = "stage7_search_policy_v1"


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _variant_contract(variant: str) -> dict[str, Any]:
    full, variants = frozen_experiment_contracts()
    contracts = {"full": full, **variants}
    if variant not in contracts:
        raise ValueError("unknown Stage7 variant")
    return contracts[variant]


def build_trajectory_contract(*, variant: str, seed: int, result_root: str | Path, frozen_input_sha256: str, candidate_pool_sha256: str) -> dict[str, Any]:
    """Bind one fixed variant/seed trajectory to caller-provided immutable inputs."""
    root = Path(result_root)
    if not root.is_absolute():
        raise ValueError("result_root must be absolute")
    if seed not in SEEDS:
        raise ValueError("seed is not one of the frozen Stage7 seeds")
    _variant_contract(variant)
    if not _is_sha(frozen_input_sha256) or not _is_sha(candidate_pool_sha256):
        raise ValueError("trajectory inputs require SHA256 digests")
    payload = {
        "schema_version": SCHEMA_VERSION + "_trajectory_contract", "variant": variant, "seed": seed,
        "result_root": str(root), "trajectory_dir": str(root / "variants" / variant / f"seed_{seed}"),
        "frozen_input_sha256": frozen_input_sha256, "candidate_pool_sha256": candidate_pool_sha256,
        "policy_name": "predicted_frontier_diversity",
    }
    return {**payload, "trajectory_contract_sha256": canonical_sha256(payload)}


def _measurement_request(*, variant: str, seed: int, round_index: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [copy.deepcopy(dict(row)) for row in rows]
    ids = [_identity(row) for row in selected]
    if len(ids) != 4 or any(not value for value in ids) or len(set(ids)) != 4:
        raise ValueError("selection must contain exactly four unique candidates")
    payload = {
        "schema_version": SCHEMA_VERSION + "_measurement_request", "variant": variant, "seed": seed,
        "round_index": round_index, "rows": selected, "selected_row_ids": ids,
    }
    request_identity = canonical_sha256(payload)
    return {**payload, "request_identity": request_identity, "measurement_request_sha256": request_identity}


def _ranked_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically consume caller-provided surrogate scores without labels."""
    def rank(row: Mapping[str, Any]) -> tuple[float, str]:
        value = row.get("score", row.get("acquisition_score", 0.0))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("surrogate score must be numeric")
        return (-float(value), _identity(row))
    return [copy.deepcopy(dict(row)) for row in sorted(rows, key=rank)]


def _a2_frozen(*, seed: int, candidate_pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = _ranked_rows(selection_candidate_view(candidate_pool))
    payload = {
        "schema_version": SCHEMA_VERSION + "_a2_frozen", "seed": seed,
        "bundle_sha256": canonical_sha256({"seed": seed, "pool": predictions}),
        "candidate_predictions": predictions,
    }
    return {**payload, "frozen_payload_sha256": canonical_sha256(payload)}


def _validate_a2_frozen(value: Mapping[str, Any], *, seed: int, expected_sha256: str | None) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(value))
    recorded = frozen.pop("frozen_payload_sha256", None)
    if recorded != canonical_sha256(frozen) or not _is_sha(frozen.get("bundle_sha256")):
        raise ValueError("A2 frozen bundle identity drift")
    if not _is_sha(expected_sha256) or recorded != expected_sha256:
        raise ValueError("A2 frozen identity does not match round-zero bundle")
    if frozen.get("seed") != seed or not isinstance(frozen.get("candidate_predictions"), list):
        raise ValueError("A2 frozen payload drift")
    selection_candidate_view(frozen["candidate_predictions"])
    return {**frozen, "frozen_payload_sha256": recorded}


def _backend_blind_ranked_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score candidates from the backend-blind numeric projection only."""
    audit = project_backend_blind_features(rows)
    projected = list(zip(rows, audit["blind_matrix"]))
    ranked = [copy.deepcopy(dict(row)) for row, _ in sorted(projected, key=lambda item: (sum(float(value) for value in item[1]), _identity(item[0])))]
    return ranked, audit


def select_stage7_round(*, variant: str, seed: int, round_index: int, candidate_pool: Sequence[Mapping[str, Any]], selected_ids: set[str] | None = None, feedback_rows: Sequence[Mapping[str, Any]] = (), previous_measurement_request: Mapping[str, Any] | None = None, a2_frozen: Mapping[str, Any] | None = None, expected_a2_frozen_sha256: str | None = None) -> dict[str, Any]:
    """Publish four selection requests only; no measurement/cache/execution occurs here."""
    _variant_contract(variant)
    if seed not in SEEDS or round_index not in range(4):
        raise ValueError("Stage7 seed or round index is outside the frozen contract")
    selected = set(selected_ids or set())
    candidates = [row for row in selection_candidate_view(candidate_pool) if _identity(row) not in selected]
    if len(candidates) < 4:
        raise ValueError("fewer than four unselected candidates remain")
    if round_index > 0:
        if previous_measurement_request is None:
            raise ValueError("later rounds require explicit previous request")
        validate_feedback_jsonl(feedback_rows, previous_measurement_request)
    elif feedback_rows or previous_measurement_request is not None:
        raise ValueError("round zero cannot accept previous feedback")
    frozen: dict[str, Any] | None = None
    backend_audit: dict[str, Any] | None = None
    if variant == "without_surrogate":
        ids = a1_select_unselected(candidates, set(), seed=seed)
        by_id = {_identity(row): row for row in candidates}
        chosen = [by_id[value] for value in ids]
        acquisition = {"policy": "uniform_random_without_replacement", "selected_row_ids": ids, "selected_rows": copy.deepcopy(chosen), "candidate_labels_visible_before_measurement": False}
    else:
        if variant == "without_measured_feedback":
            frozen = _a2_frozen(seed=seed, candidate_pool=candidates) if round_index == 0 else _validate_a2_frozen(a2_frozen or {}, seed=seed, expected_sha256=expected_a2_frozen_sha256)
            predicted_by_id = {_identity(row): row for row in frozen["candidate_predictions"]}
            ranked = [predicted_by_id[_identity(row)] for row in candidates if _identity(row) in predicted_by_id]
        elif variant == "backend_blind":
            ranked, backend_audit = _backend_blind_ranked_rows(candidates)
        else:
            ranked = _ranked_rows(candidates)
        chosen = ranked[:4]
        acquisition = {"policy": "predicted_frontier_diversity", "selected_row_ids": [_identity(row) for row in chosen], "selected_rows": copy.deepcopy(chosen), "candidate_labels_visible_before_measurement": False}
    request = _measurement_request(variant=variant, seed=seed, round_index=round_index, rows=chosen)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION + "_round_selection", "variant": variant, "seed": seed, "round_index": round_index, "acquisition": acquisition, "measurement_request": request}
    if frozen is not None:
        result["a2_frozen"] = frozen
        result["a2_feedback_projection"] = project_a2_feedback([], {}, anchor={}, bundle_sha256=frozen["bundle_sha256"], selected_results=feedback_rows)
    if backend_audit is not None:
        result["backend_blind_audit"] = backend_audit
    return result


def validate_feedback_jsonl(rows: Sequence[Mapping[str, Any]], measurement_request: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only a complete feedback batch carrying the exact request identity."""
    request_identity = measurement_request.get("request_identity")
    expected_ids = [_identity(row) for row in measurement_request.get("rows", [])]
    if not _is_sha(request_identity) or len(expected_ids) != 4 or len(set(expected_ids)) != 4:
        raise ValueError("measurement request identity is invalid")
    received = [copy.deepcopy(dict(row)) for row in rows]
    if len(received) != 4:
        raise ValueError("feedback must contain exactly four rows")
    ids = [_identity(row) for row in received]
    if set(ids) != set(expected_ids) or len(set(ids)) != 4:
        raise ValueError("feedback row identities do not exactly match request")
    if any(row.get("request_identity") != request_identity for row in received):
        raise ValueError("feedback request identity drift")
    if any(row.get("terminal_status") not in {"completed", "failed"} for row in received):
        raise ValueError("feedback terminal status is invalid")
    return {"feedback_row_count": 4, "request_identity": request_identity, "row_ids": sorted(ids)}


def build_request_binding(trajectory_contract: Mapping[str, Any], measurement_request: Mapping[str, Any], *, round_index: int) -> dict[str, Any]:
    """Hash-bind an output request to exactly one published trajectory."""
    if round_index not in range(4) or measurement_request.get("round_index") != round_index:
        raise ValueError("request binding round identity drift")
    expected = {"variant": trajectory_contract.get("variant"), "seed": trajectory_contract.get("seed"), "round_index": round_index}
    if any(measurement_request.get(key) != value for key, value in expected.items()):
        raise ValueError("request binding identity drift")
    if not _is_sha(trajectory_contract.get("trajectory_contract_sha256")) or not _is_sha(measurement_request.get("request_identity")):
        raise ValueError("request binding requires SHA identities")
    payload = {"schema_version": SCHEMA_VERSION + "_request_binding", "trajectory_contract_sha256": trajectory_contract["trajectory_contract_sha256"], "measurement_request_sha256": measurement_request["request_identity"], "variant": expected["variant"], "seed": expected["seed"], "round_index": round_index, "selected_row_ids": [_identity(row) for row in measurement_request["rows"]]}
    return {**payload, "request_binding_sha256": canonical_sha256(payload)}


def validate_request_binding(binding: Mapping[str, Any], measurement_request: Mapping[str, Any], trajectory_contract: Mapping[str, Any]) -> dict[str, Any]:
    recorded = dict(binding)
    digest = recorded.pop("request_binding_sha256", None)
    if digest != canonical_sha256(recorded):
        raise ValueError("request binding SHA drift")
    expected = build_request_binding(trajectory_contract, measurement_request, round_index=int(measurement_request.get("round_index", -1)))
    if dict(binding) != expected:
        raise ValueError("request binding identity drift")
    return {"verdict": "pass", "request_binding_sha256": digest}
