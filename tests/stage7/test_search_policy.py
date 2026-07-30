"""Round and feedback contracts for public Stage7 selection."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Mapping

import pytest

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5 import single_target_search_v2 as single
from framework.stage7 import search_policy_v1 as policy


def _task() -> single.SearchTask:
    return single.SearchTask("S7-PYR-TVM", "pyramid", "h800", _profile())


def _profile() -> dict[str, object]:
    return build_capability_profile(
        capability_profile_id="h800-tvm", hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"tvm").hexdigest(), dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )


def _source_group(width: list[int]) -> dict[str, object]:
    group_id = f"pyramid|{'x'.join(map(str, width))}"
    evidence = hashlib.sha256(group_id.encode()).hexdigest()
    contract = {
        "schema_version": "stage5_source_contract_v1", "group_id": group_id, "model": "pyramid",
        "width": width, "artifact_id": f"fixture-{width[0]}", "source_status": "ready",
        "source_evidence_sha256": evidence, "materialization_scope": "public_test_fixture",
    }
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "group_id": group_id, "model": "pyramid", "width": width, "source_status": "ready",
        "source_evidence_sha256": evidence, "source_contract": contract, "source_contract_sha256": contract_sha,
        "graph_features": {"group_id": group_id, "model": "pyramid", "width": width, "conv_count": 20, "conv_macs": float(width[0] * width[1] * width[2])},
    }


def _candidates() -> list[dict[str, object]]:
    source = {"schema_version": "stage5_candidate_source_registry_v1", "groups": [
        _source_group(list(width)) for width in ([40, 80, 160], [48, 96, 192], [56, 112, 224], [64, 128, 256])
    ]}
    manifest = single.build_task_candidate_manifest(source, task=_task(), measured_row_ids=set())
    candidates: list[dict[str, object]] = []
    for index, row in enumerate(manifest["rows"]):
        predicted = copy.deepcopy(row)
        values = {"latency_ms": 1.0 + index, "energy_j": 0.2 + index / 10, "ap70": 0.8 - index / 100}
        predicted["predictions"] = values
        predicted["prediction_intervals"] = {
            key: {"lower": value - 0.1, "median": value, "upper": value + 0.1}
            for key, value in values.items()
        }
        candidates.append(predicted)
    return candidates


def _select(**kwargs: object) -> dict[str, object]:
    return policy.select_stage7_round(task=_task(), measured_rows=[], measured_graph_features=[], **kwargs)


def _feedback(selection: Mapping[str, object]) -> list[dict[str, str]]:
    request = selection["measurement_request"]
    return [
        {"row_id": row["row_id"], "request_identity": request["request_identity"], "terminal_status": "completed"}
        for row in request["rows"]
    ]


def _rehash_request(request: dict[str, object]) -> dict[str, object]:
    body = {field: request[field] for field in policy._REQUEST_BODY_FIELDS}
    identity = policy.canonical_sha256(body)
    return {**request, "request_identity": identity, "measurement_request_sha256": identity}


def test_round_zero_selection_excludes_existing_ids_and_is_deterministic() -> None:
    """Catches reselecting measured work or allowing score ordering to vary by input order."""
    first = _select(
        variant="full", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    second = _select(
        variant="full", seed=20260718, round_index=0, candidate_pool=list(reversed(_candidates()))
    )

    assert first["acquisition"]["selected_row_ids"] == second["acquisition"]["selected_row_ids"]
    assert len(first["measurement_request"]["rows"]) == 4


def test_a2_later_round_reuses_exact_frozen_bundle() -> None:
    """Catches A2 accepting a different later-round prediction or bundle identity."""
    first = _select(
        variant="without_measured_feedback", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    later = _select(
        variant="without_measured_feedback", seed=20260718, round_index=1, candidate_pool=_candidates(),
        selected_ids=set(first["acquisition"]["selected_row_ids"]), a2_frozen=first["a2_frozen"],
        expected_a2_frozen_sha256=first["a2_frozen"]["frozen_payload_sha256"],
        previous_measurement_request=first["measurement_request"],
        feedback_rows=[{"row_id": row_id, "request_identity": first["measurement_request"]["request_identity"], "terminal_status": "completed"}
                       for row_id in first["acquisition"]["selected_row_ids"]],
    )

    assert later["a2_frozen"] == first["a2_frozen"]
    assert later["a2_feedback_projection"]["feedback_refit"] is False
    tampered = copy.deepcopy(first["a2_frozen"])
    tampered["bundle_sha256"] = "b" * 64
    feedback = [
        {"row_id": row_id, "request_identity": first["measurement_request"]["request_identity"], "terminal_status": "completed"}
        for row_id in first["acquisition"]["selected_row_ids"]
    ]
    with pytest.raises(ValueError, match="A2"):
        _select(
            variant="without_measured_feedback", seed=20260718, round_index=1,
            candidate_pool=_candidates(), selected_ids=set(first["acquisition"]["selected_row_ids"]), a2_frozen=tampered,
            expected_a2_frozen_sha256=first["a2_frozen"]["frozen_payload_sha256"],
            previous_measurement_request=first["measurement_request"], feedback_rows=feedback,
        )


def test_a2_requires_the_explicit_round_zero_frozen_identity() -> None:
    """Catches a self-hashed replacement bundle changing later A2 selection."""
    first = _select(
        variant="without_measured_feedback", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    replacement = copy.deepcopy(first["a2_frozen"])
    replacement["candidate_predictions"].reverse()
    unsigned = dict(replacement)
    unsigned.pop("frozen_payload_sha256")
    replacement["frozen_payload_sha256"] = policy.canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="frozen identity"):
        _select(
            variant="without_measured_feedback", seed=20260718, round_index=1, candidate_pool=_candidates(),
            selected_ids=set(first["acquisition"]["selected_row_ids"]), a2_frozen=replacement,
            expected_a2_frozen_sha256=first["a2_frozen"]["frozen_payload_sha256"],
            previous_measurement_request=first["measurement_request"],
            feedback_rows=[
                {"row_id": row_id, "request_identity": first["measurement_request"]["request_identity"], "terminal_status": "completed"}
                for row_id in first["acquisition"]["selected_row_ids"]
            ],
        )


def test_feedback_requires_exact_request_identity_and_complete_unique_rows() -> None:
    """Catches feedback from another request, duplicate rows, or partial completions."""
    selection = _select(
        variant="full", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    request = selection["measurement_request"]
    rows = [
        {"row_id": row["row_id"], "request_identity": request["request_identity"], "terminal_status": "completed"}
        for row in request["rows"]
    ]

    accepted = policy.validate_feedback_jsonl(rows, request)
    assert accepted["feedback_row_count"] == 4
    wrong = copy.deepcopy(rows)
    wrong[0]["request_identity"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        policy.validate_feedback_jsonl(wrong, request)
    with pytest.raises(ValueError, match="exactly four"):
        policy.validate_feedback_jsonl(rows[:3], request)

    with pytest.raises(ValueError, match="previous request"):
        _select(
            variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(), feedback_rows=rows,
        )
    wrong[0]["terminal_status"] = "completed"
    with pytest.raises(ValueError, match="identity"):
        _select(
            variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(), feedback_rows=wrong,
            previous_measurement_request=request,
        )


def test_backend_blind_selection_cannot_consume_score_or_derived_features() -> None:
    """Catches backend-blind selection being driven by unprojected backend scores."""
    rows = _candidates()
    for index, row in enumerate(rows):
        row["model_features"] = {"capability_score": float(index), "static_depth": 2.0}
        row["feature_provenance"] = {"capability_score": "capability_profile-derived", "static_depth": "static_config"}
        row["score"] = float(index)
    altered = copy.deepcopy(rows)
    for row in altered:
        row["score"] = -float(row["score"])

    bundle = policy.build_backend_blind_prediction_bundle(
        task=_task(), candidate_pool=rows, model_bundle_sha256="a" * 64
    )
    first = _select(
        variant="backend_blind", seed=20260718, round_index=0, candidate_pool=rows,
        blind_prediction_bundle=bundle, expected_blind_prediction_bundle_sha256=bundle["bundle_sha256"],
    )
    second = _select(
        variant="backend_blind", seed=20260718, round_index=0, candidate_pool=altered,
        blind_prediction_bundle=bundle, expected_blind_prediction_bundle_sha256=bundle["bundle_sha256"],
    )

    assert first["acquisition"]["selected_row_ids"] == second["acquisition"]["selected_row_ids"]
    assert "model:capability_score" in first["backend_blind_audit"]["removed_names"]


def test_later_round_uses_verified_history_when_caller_selected_ids_are_empty() -> None:
    """Catches an empty or stale caller list reselecting the verified previous batch."""
    first = _select(variant="full", seed=20260718, round_index=0, candidate_pool=_candidates())
    with pytest.raises(ValueError, match="selected history"):
        _select(
            variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(), selected_ids=set(),
            previous_measurement_request=first["measurement_request"], feedback_rows=_feedback(first),
        )
    later = _select(
        variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(),
        selected_ids=set(first["acquisition"]["selected_row_ids"]),
        previous_measurement_request=first["measurement_request"], feedback_rows=_feedback(first),
    )
    assert not set(later["acquisition"]["selected_row_ids"]) & set(first["acquisition"]["selected_row_ids"])


@pytest.mark.parametrize("field", ["variant", "seed", "round_index", "task_sha256"])
def test_later_round_rejects_self_consistent_wrong_previous_request(field: str) -> None:
    """Catches accepting an internally rehashed request from a different trajectory."""
    first = _select(variant="full", seed=20260718, round_index=0, candidate_pool=_candidates())
    wrong = copy.deepcopy(first["measurement_request"])
    wrong[field] = {"variant": "backend_blind", "seed": 20260719, "round_index": 1, "task_sha256": "b" * 64}[field]
    if field == "round_index":
        wrong["selected_history_ids"] = [f"prior-{index}" for index in range(4)] + list(
            wrong["selected_row_ids"]
        )
    wrong = _rehash_request(wrong)
    feedback = _feedback(first)
    for row in feedback:
        row["request_identity"] = wrong["request_identity"]
    with pytest.raises(ValueError, match="previous request trajectory"):
        _select(
            variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(),
            selected_ids=set(first["acquisition"]["selected_row_ids"]),
            previous_measurement_request=wrong, feedback_rows=feedback,
        )


def test_backend_blind_uses_only_explicit_bound_prediction_bundle() -> None:
    """Catches caller predictions overriding or replacing a blind prediction bundle."""
    candidates = _candidates()
    bundle = policy.build_backend_blind_prediction_bundle(
        task=_task(), candidate_pool=candidates, model_bundle_sha256="a" * 64
    )
    baseline = _select(
        variant="backend_blind", seed=20260718, round_index=0, candidate_pool=candidates,
        blind_prediction_bundle=bundle, expected_blind_prediction_bundle_sha256=bundle["bundle_sha256"],
    )
    altered = copy.deepcopy(candidates)
    for row in altered:
        row["predictions"]["latency_ms"] *= 1000
    same = _select(
        variant="backend_blind", seed=20260718, round_index=0, candidate_pool=altered,
        blind_prediction_bundle=bundle, expected_blind_prediction_bundle_sha256=bundle["bundle_sha256"],
    )
    assert baseline["acquisition"]["selected_row_ids"] == same["acquisition"]["selected_row_ids"]
    tampered = copy.deepcopy(bundle)
    tampered["predictions"][0]["predictions"]["latency_ms"] *= 2
    unsigned = dict(tampered)
    unsigned.pop("bundle_sha256")
    tampered["bundle_sha256"] = policy.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="blind prediction bundle identity"):
        _select(
            variant="backend_blind", seed=20260718, round_index=0, candidate_pool=candidates,
            blind_prediction_bundle=tampered, expected_blind_prediction_bundle_sha256=bundle["bundle_sha256"],
        )


def test_request_binding_rejects_trajectory_identity_drift() -> None:
    """Catches request-binding reuse under another variant, seed, directory, or round."""
    trajectory = policy.build_trajectory_contract(
        variant="full", seed=20260718, result_root="/tmp/stage7",
        frozen_input_sha256="a" * 64, candidate_pool_sha256="b" * 64,
    )
    selection = _select(
        variant="full", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    binding = policy.build_request_binding(trajectory, selection["measurement_request"], round_index=0)
    assert policy.validate_request_binding(binding, selection["measurement_request"], trajectory)["verdict"] == "pass"
    drifted = dict(trajectory, variant="backend_blind")
    with pytest.raises(ValueError, match="identity"):
        policy.validate_request_binding(binding, selection["measurement_request"], drifted)


def test_a2_uses_frozen_prediction_order_and_excludes_selected_ids() -> None:
    """Catches A2 rebuilding ranks from a reordered caller pool or retaining selected IDs."""
    first = _select(variant="without_measured_feedback", seed=20260718, round_index=0, candidate_pool=_candidates())
    feedback = [
        {"row_id": row_id, "request_identity": first["measurement_request"]["request_identity"], "terminal_status": "completed"}
        for row_id in first["acquisition"]["selected_row_ids"]
    ]
    common = {
        "variant": "without_measured_feedback", "seed": 20260718, "round_index": 1,
        "selected_ids": set(first["acquisition"]["selected_row_ids"]), "a2_frozen": first["a2_frozen"],
        "expected_a2_frozen_sha256": first["a2_frozen"]["frozen_payload_sha256"],
        "previous_measurement_request": first["measurement_request"], "feedback_rows": feedback,
    }
    normal = _select(candidate_pool=_candidates(), **common)
    reordered = _select(candidate_pool=list(reversed(_candidates())), **common)

    assert normal["acquisition"]["selected_row_ids"] == reordered["acquisition"]["selected_row_ids"]
    assert not set(normal["acquisition"]["selected_row_ids"]) & set(first["acquisition"]["selected_row_ids"])


@pytest.mark.parametrize("field", ["rows", "variant", "seed", "round_index"])
def test_feedback_rejects_forged_canonical_request_body(field: str) -> None:
    """Catches a 64-hex request identity being reused after its body has changed."""
    selection = _select(variant="full", seed=20260718, round_index=0, candidate_pool=_candidates())
    forged = copy.deepcopy(selection["measurement_request"])
    if field == "rows":
        forged["rows"] = list(reversed(forged["rows"]))
    elif field == "variant":
        forged[field] = "backend_blind"
    elif field == "seed":
        forged[field] = 20260719
    else:
        forged[field] = 1
    feedback = [
        {"row_id": row["row_id"], "request_identity": forged["request_identity"], "terminal_status": "completed"}
        for row in forged["rows"]
    ]

    with pytest.raises(ValueError, match="canonical"):
        policy.validate_feedback_jsonl(feedback, forged)
