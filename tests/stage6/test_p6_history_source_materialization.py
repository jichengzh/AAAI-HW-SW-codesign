from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    project_source_materialization_request,
)


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shared_paths(group_slug: str) -> dict[str, str]:
    return {
        key: f"/private/synthetic/materialized/{group_slug}/{key}"
        for key in SHARED_SOURCE_PATH_KEYS
    }


def _source_contract(width: list[int]) -> dict[str, Any]:
    group_id = f"pyramid|{width[0]}x{width[1]}x{width[2]}"
    evidence = _sha({"evidence": group_id})
    return {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "artifact_id": f"pyramid-{width[0]}-{width[1]}-{width[2]}",
        "source_status": "ready",
        "source_evidence_sha256": evidence,
        "materialization_scope": "synthetic_fixture",
        "stage_widths": {
            "stage1_width": width[0],
            "stage2_width": width[1],
            "stage3_width": width[2],
        },
        "base_checkpoint_path": "/private/synthetic/base/model.ckpt",
        "dataset_root": "/private/synthetic/dataset",
        "training_parameters": {"epochs": 7, "optimizer": "synthetic"},
        "shared_source_paths": _shared_paths("-".join(map(str, width))),
    }


def _row(width: list[int], q_mode: str, index: int, task_sha: str) -> dict[str, Any]:
    contract = _source_contract(width)
    group_id = contract["group_id"]
    evidence = contract["source_evidence_sha256"]
    row_id = f"{group_id}|q={q_mode}|candidate={index}"
    return {
        "schema_version": "stage5_candidate_row_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "row_id": row_id,
        "manifest_job_id": row_id,
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "width_schema": ["w0", "w1", "w2"],
        "structure_widths": {"w0": width[0], "w1": width[1], "w2": width[2]},
        "genome": [*width, q_mode],
        "strategy_id": f"q={q_mode}",
        "q_mode": q_mode,
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm-auto",
        "capability_digest": _sha({"profile": "h800-tvm-auto"}),
        "dispatch_key": "tvm_auto",
        "source_status": "ready",
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "source_contract": contract,
        "source_contract_sha256": _sha(contract),
        "source_evidence_sha256": evidence,
        "graph_features": {"group_id": group_id, "model": "pyramid", "width": width},
    }


def _request() -> dict[str, Any]:
    task_sha = _sha({"task": "P6-H800-PYRAMID"})
    rows = [
        _row([17, 31, 63], "fp16", 0, task_sha),
        _row([17, 31, 63], "int8", 1, task_sha),
        _row([23, 47, 95], "fp16", 2, task_sha),
        _row([29, 53, 101], "int8", 3, task_sha),
    ]
    body = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**body, "measurement_request_sha256": _sha(body)}


def _rehash_request(request: dict[str, Any]) -> None:
    request["row_sha256"] = {
        row["row_id"]: _sha(row) for row in request["rows"]
    }
    body = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(body)


def _rehash_row_contract(request: dict[str, Any], row_index: int) -> None:
    row = request["rows"][row_index]
    row["source_contract_sha256"] = _sha(row["source_contract"])
    _rehash_request(request)


def test_projection_flattens_shared_paths_and_recomputes_only_existing_hashes() -> None:
    """Catches mutation, q-mode path splitting, static-field loss, and stale hashes."""
    request = _request()
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    assert request == original
    assert projected.request is not request
    assert projected.request["rows"][0] is not request["rows"][0]
    assert projected.ordered_group_ids == (
        "pyramid|17x31x63",
        "pyramid|23x47x95",
        "pyramid|29x53x101",
    )
    first, second = projected.request["rows"][:2]
    assert first["q_mode"] == "fp16"
    assert second["q_mode"] == "int8"
    assert first["strategy_id"] != second["strategy_id"]
    assert first["genome"] != second["genome"]
    for row in projected.request["rows"]:
        contract = row["source_contract"]
        assert "shared_source_paths" not in contract
        assert contract["base_checkpoint_path"] == "/private/synthetic/base/model.ckpt"
        assert contract["training_parameters"] == {
            "epochs": 7,
            "optimizer": "synthetic",
        }
        assert row["source_contract_sha256"] == _sha(contract)
        assert projected.request["row_sha256"][row["row_id"]] == _sha(row)
    assert {
        key: first["source_contract"][key] for key in SHARED_SOURCE_PATH_KEYS
    } == {
        key: second["source_contract"][key] for key in SHARED_SOURCE_PATH_KEYS
    }
    request_body = {
        key: value
        for key, value in projected.request.items()
        if key != "measurement_request_sha256"
    }
    assert projected.request["measurement_request_sha256"] == _sha(request_body)
    assert set(projected.request) == set(original)
    assert set(projected.request["rows"][0]) == set(original["rows"][0])
    projected.request["rows"][0]["source_contract"]["training_parameters"][
        "epochs"
    ] = 99
    assert request["rows"][0]["source_contract"]["training_parameters"][
        "epochs"
    ] == 7


@pytest.mark.parametrize("mutation", ["same_group_mismatch", "cross_group_collision"])
def test_projection_rejects_shared_path_mismatch_or_collision(mutation: str) -> None:
    request = _request()
    if mutation == "same_group_mismatch":
        request["rows"][1]["source_contract"]["shared_source_paths"][
            "checkpoint_path"
        ] += "-drift"
        _rehash_row_contract(request, 1)
    else:
        first_path = request["rows"][0]["source_contract"]["shared_source_paths"][
            "checkpoint_path"
        ]
        request["rows"][2]["source_contract"]["shared_source_paths"][
            "checkpoint_path"
        ] = first_path
        _rehash_row_contract(request, 2)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_rejects_same_group_static_contract_drift() -> None:
    """Catches one deduplicated group carrying conflicting private training inputs."""
    request = _request()
    request["rows"][1]["source_contract"]["training_parameters"]["epochs"] = 8
    _rehash_row_contract(request, 1)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_rejects_legacy_q_mode_outputs_in_shared_contract() -> None:
    """Catches a v2 request retaining a second q-mode-specific source truth."""
    request = _request()
    request["rows"][2]["source_contract"][
        "materialization_outputs_by_q_mode"
    ] = {"fp16": {"checkpoint_path": "/private/untrusted/fp16.ckpt"}}
    _rehash_row_contract(request, 2)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


@pytest.mark.parametrize("mutation", ["identity", "missing_path", "extra_path"])
def test_projection_rejects_registry_identity_or_shared_key_drift(mutation: str) -> None:
    request = _request()
    contract: dict[str, Any] = request["rows"][0]["source_contract"]
    if mutation == "identity":
        contract["artifact_id"] = "pyramid-identity-drift"
    elif mutation == "missing_path":
        contract["shared_source_paths"].pop("onnx_report_path")
    else:
        contract["shared_source_paths"]["untrusted_extra_path"] = (
            "/private/synthetic/untrusted"
        )
    _rehash_row_contract(request, 0)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_preserves_legacy_request_without_shared_bundle() -> None:
    """Catches breaking the recipe-v1/static registry measurement path."""
    request = _request()
    for row in request["rows"]:
        row["source_contract"].pop("shared_source_paths")
        row["source_contract_sha256"] = _sha(row["source_contract"])
    _rehash_request(request)
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    assert projected.request == original
    assert projected.request is not request
    assert request == original


def test_projection_rejects_mixed_legacy_and_shared_requests() -> None:
    request = _request()
    request["rows"][0]["source_contract"].pop("shared_source_paths")
    _rehash_row_contract(request, 0)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
