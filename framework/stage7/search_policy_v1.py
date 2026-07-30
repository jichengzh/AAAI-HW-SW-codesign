"""Selection-only adapters for the frozen Stage7 online-ablation contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_measurement_request as build_stage5_measurement_request,
    select_task_batch,
    validate_search_task,
)
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


def _stage5_request(
    *, task: SearchTask, variant: str, seed: int, round_index: int, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Wrap a Stage5 request in one canonical Stage7 feedback identity body."""
    stage5_request = build_stage5_measurement_request(
        task=task, selected_rows=rows, round_index=round_index
    )
    selected = [copy.deepcopy(dict(row)) for row in stage5_request["rows"]]
    ids = [_identity(row) for row in selected]
    if len(ids) != 4 or any(not value for value in ids) or len(set(ids)) != 4:
        raise ValueError("selection must contain exactly four unique candidates")
    payload = {
        "schema_version": SCHEMA_VERSION + "_measurement_request", "variant": variant,
        "seed": seed, "round_index": round_index, "task_sha256": stage5_request["task_sha256"],
        "stage5_measurement_request_sha256": stage5_request["measurement_request_sha256"],
        "rows": selected, "selected_row_ids": ids,
    }
    request_identity = canonical_sha256(payload)
    return {**payload, "request_identity": request_identity, "measurement_request_sha256": request_identity}


_REQUEST_BODY_FIELDS = (
    "schema_version", "variant", "seed", "round_index", "task_sha256",
    "stage5_measurement_request_sha256", "rows", "selected_row_ids",
)


def validate_measurement_request(measurement_request: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute both published request identities over one canonical request body."""
    if set(measurement_request) != {*_REQUEST_BODY_FIELDS, "request_identity", "measurement_request_sha256"}:
        raise ValueError("measurement request canonical body fields drift")
    body = {field: copy.deepcopy(measurement_request.get(field)) for field in _REQUEST_BODY_FIELDS}
    canonical = canonical_sha256(body)
    if (
        measurement_request.get("request_identity") != canonical
        or measurement_request.get("measurement_request_sha256") != canonical
        or not _is_sha(body["task_sha256"])
        or not _is_sha(body["stage5_measurement_request_sha256"])
    ):
        raise ValueError("measurement request canonical identity drift")
    rows = body["rows"]
    ids = body["selected_row_ids"]
    if (
        not isinstance(rows, list) or not isinstance(ids, list) or len(rows) != 4
        or [_identity(row) for row in rows if isinstance(row, Mapping)] != ids
        or len(set(ids)) != 4 or any(not isinstance(value, str) or not value for value in ids)
    ):
        raise ValueError("measurement request canonical row identity drift")
    return {**body, "request_identity": canonical, "measurement_request_sha256": canonical}


def _stage7_task_contract(task: SearchTask) -> dict[str, Any]:
    contract = validate_search_task(task)
    if (task.task_id, task.target_model, task.hardware_id, task.sample_budget, task.batch_size, task.round_count) != (
        "S7-PYR-TVM", "pyramid", "h800", 16, 4, 4,
    ):
        raise ValueError("search task is not the frozen Stage7 task")
    return contract


def build_stage7_task(raw_profile: Mapping[str, Any]) -> SearchTask:
    """Construct the sole public Stage7 task from an explicit capability profile."""
    task = SearchTask(
        task_id="S7-PYR-TVM", target_model="pyramid", hardware_id="h800",
        capability_profile=copy.deepcopy(dict(raw_profile)), sample_budget=16,
        batch_size=4, round_count=4,
    )
    _stage7_task_contract(task)
    return task


def _sorted_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((copy.deepcopy(dict(row)) for row in rows), key=_identity)


def _stage5_acquisition(
    *, task: SearchTask, candidates: Sequence[Mapping[str, Any]], measured_rows: Sequence[Mapping[str, Any]],
    measured_graph_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return select_task_batch(
        _sorted_candidates(candidates),
        [copy.deepcopy(dict(row)) for row in measured_rows],
        [copy.deepcopy(dict(row)) for row in measured_graph_features], task=task,
    )


def _a2_frozen(
    *, task: SearchTask, seed: int, candidate_pool: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]], measured_graph_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = _sorted_candidates(selection_candidate_view(candidate_pool))
    acquisition = _stage5_acquisition(
        task=task, candidates=predictions, measured_rows=measured_rows,
        measured_graph_features=measured_graph_features,
    )
    payload = {
        "schema_version": SCHEMA_VERSION + "_a2_frozen", "seed": seed,
        "task_sha256": _stage7_task_contract(task)["task_sha256"],
        "bundle_sha256": canonical_sha256({"seed": seed, "pool": predictions}),
        "candidate_predictions": predictions,
        "measured_rows": [copy.deepcopy(dict(row)) for row in measured_rows],
        "measured_graph_features": [copy.deepcopy(dict(row)) for row in measured_graph_features],
        "round_zero_acquisition": acquisition,
        "round_zero_acquisition_sha256": canonical_sha256(acquisition),
    }
    return {**payload, "frozen_payload_sha256": canonical_sha256(payload)}


def _validate_a2_frozen(
    value: Mapping[str, Any], *, task: SearchTask, seed: int, expected_sha256: str | None,
    candidate_pool: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(value))
    recorded = frozen.pop("frozen_payload_sha256", None)
    if recorded != canonical_sha256(frozen) or not _is_sha(frozen.get("bundle_sha256")):
        raise ValueError("A2 frozen bundle identity drift")
    if not _is_sha(expected_sha256) or recorded != expected_sha256:
        raise ValueError("A2 frozen identity does not match round-zero bundle")
    if (
        frozen.get("seed") != seed
        or frozen.get("task_sha256") != _stage7_task_contract(task)["task_sha256"]
        or not isinstance(frozen.get("candidate_predictions"), list)
        or not isinstance(frozen.get("measured_rows"), list)
        or not isinstance(frozen.get("measured_graph_features"), list)
    ):
        raise ValueError("A2 frozen payload drift")
    predictions = selection_candidate_view(frozen["candidate_predictions"])
    supplied = selection_candidate_view(candidate_pool)
    if {_identity(row) for row in predictions} != {_identity(row) for row in supplied}:
        raise ValueError("A2 frozen candidate pool identity drift")
    if frozen.get("round_zero_acquisition_sha256") != canonical_sha256(frozen.get("round_zero_acquisition")):
        raise ValueError("A2 frozen acquisition identity drift")
    return {**frozen, "frozen_payload_sha256": recorded}


def _backend_field(name: Any) -> bool:
    lowered = str(name).lower().replace("_", " ")
    return any(token in lowered for token in ("backend", "capability", "profile", "dispatch"))


def _backend_blind_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep Stage5 identity fields but remove only model-facing derived features."""
    copied = copy.deepcopy(dict(row))
    graph = copied.get("graph_features")
    if isinstance(graph, Mapping):
        copied["graph_features"] = {
            key: copy.deepcopy(value) for key, value in graph.items()
            if key in {"group_id", "model", "width", "input_dims"} or not _backend_field(key)
        }
    for container in ("model_features", "features"):
        values = copied.get(container)
        if isinstance(values, Mapping):
            copied[container] = {
                key: copy.deepcopy(value) for key, value in values.items() if not _backend_field(key)
            }
    provenance = copied.get("feature_provenance")
    if isinstance(provenance, Mapping):
        copied["feature_provenance"] = {
            key: copy.deepcopy(value) for key, value in provenance.items()
            if not _backend_field(key) and not _backend_field(value)
        }
    return copied


def _backend_blind_inputs(
    rows: Sequence[Mapping[str, Any]], measured_rows: Sequence[Mapping[str, Any]],
    measured_graph_features: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the same blind projection before the unchanged Stage5 acquisition."""
    audit = project_backend_blind_features(rows)
    blind_rows = [_backend_blind_row(row) for row in rows]
    blind_measured = [_backend_blind_row(row) for row in measured_rows]
    blind_graphs = []
    for graph in measured_graph_features:
        blind_graphs.append({
            key: copy.deepcopy(value) for key, value in dict(graph).items()
            if key in {"group_id", "model", "width", "input_dims"} or not _backend_field(key)
        })
    return blind_rows, blind_measured, blind_graphs, audit


def select_stage7_round(*, variant: str, seed: int, round_index: int, task: SearchTask, candidate_pool: Sequence[Mapping[str, Any]], measured_rows: Sequence[Mapping[str, Any]], measured_graph_features: Sequence[Mapping[str, Any]], selected_ids: set[str] | None = None, feedback_rows: Sequence[Mapping[str, Any]] = (), previous_measurement_request: Mapping[str, Any] | None = None, a2_frozen: Mapping[str, Any] | None = None, expected_a2_frozen_sha256: str | None = None) -> dict[str, Any]:
    """Publish four selection requests only; no measurement/cache/execution occurs here."""
    _variant_contract(variant)
    if seed not in SEEDS or round_index not in range(4):
        raise ValueError("Stage7 seed or round index is outside the frozen contract")
    _stage7_task_contract(task)
    selected = set(selected_ids or set())
    all_candidates = _sorted_candidates(selection_candidate_view(candidate_pool))
    candidates = [row for row in all_candidates if _identity(row) not in selected]
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
            frozen = _a2_frozen(
                task=task, seed=seed, candidate_pool=all_candidates,
                measured_rows=measured_rows, measured_graph_features=measured_graph_features,
            ) if round_index == 0 else _validate_a2_frozen(
                a2_frozen or {}, task=task, seed=seed,
                expected_sha256=expected_a2_frozen_sha256, candidate_pool=all_candidates,
            )
            predicted_by_id = {_identity(row): row for row in frozen["candidate_predictions"]}
            ranked = [
                predicted_by_id[_identity(row)] for row in frozen["candidate_predictions"]
                if _identity(row) not in selected
            ]
            acquisition = _stage5_acquisition(
                task=task, candidates=ranked, measured_rows=frozen["measured_rows"],
                measured_graph_features=frozen["measured_graph_features"],
            )
        elif variant == "backend_blind":
            blind_candidates, blind_measured, blind_graphs, backend_audit = _backend_blind_inputs(
                candidates, measured_rows, measured_graph_features
            )
            acquisition = _stage5_acquisition(
                task=task, candidates=blind_candidates, measured_rows=blind_measured,
                measured_graph_features=blind_graphs,
            )
        else:
            acquisition = _stage5_acquisition(
                task=task, candidates=candidates, measured_rows=measured_rows,
                measured_graph_features=measured_graph_features,
            )
        chosen = acquisition["selected_rows"]
    request = _stage5_request(task=task, variant=variant, seed=seed, round_index=round_index, rows=chosen)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION + "_round_selection", "variant": variant, "seed": seed, "round_index": round_index, "acquisition": acquisition, "measurement_request": request}
    if frozen is not None:
        result["a2_frozen"] = frozen
        result["a2_feedback_projection"] = project_a2_feedback([], {}, anchor={}, bundle_sha256=frozen["bundle_sha256"], selected_results=feedback_rows)
    if backend_audit is not None:
        result["backend_blind_audit"] = backend_audit
    return result


def validate_feedback_jsonl(rows: Sequence[Mapping[str, Any]], measurement_request: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only a complete feedback batch carrying the exact request identity."""
    verified_request = validate_measurement_request(measurement_request)
    request_identity = verified_request["request_identity"]
    expected_ids = [_identity(row) for row in verified_request["rows"]]
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
    verified_request = validate_measurement_request(measurement_request)
    if round_index not in range(4) or verified_request.get("round_index") != round_index:
        raise ValueError("request binding round identity drift")
    expected = {"variant": trajectory_contract.get("variant"), "seed": trajectory_contract.get("seed"), "round_index": round_index}
    if any(verified_request.get(key) != value for key, value in expected.items()):
        raise ValueError("request binding identity drift")
    if not _is_sha(trajectory_contract.get("trajectory_contract_sha256")):
        raise ValueError("request binding requires SHA identities")
    payload = {"schema_version": SCHEMA_VERSION + "_request_binding", "trajectory_contract_sha256": trajectory_contract["trajectory_contract_sha256"], "measurement_request_sha256": verified_request["request_identity"], "variant": expected["variant"], "seed": expected["seed"], "round_index": round_index, "selected_row_ids": [_identity(row) for row in verified_request["rows"]]}
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
