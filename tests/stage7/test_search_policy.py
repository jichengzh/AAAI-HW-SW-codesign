"""Round and feedback contracts for public Stage7 selection."""

from __future__ import annotations

import copy

import pytest

from framework.stage7 import search_policy_v1 as policy


def _candidates() -> list[dict[str, object]]:
    return [
        {"row_id": f"candidate-{number}", "score": float(number), "width": [16, 32, 64], "q_mode": "fp16"}
        for number in range(8)
    ]


def test_round_zero_selection_excludes_existing_ids_and_is_deterministic() -> None:
    """Catches reselecting measured work or allowing score ordering to vary by input order."""
    first = policy.select_stage7_round(
        variant="full", seed=20260718, round_index=0, candidate_pool=_candidates(), selected_ids={"candidate-7"}
    )
    second = policy.select_stage7_round(
        variant="full", seed=20260718, round_index=0, candidate_pool=list(reversed(_candidates())), selected_ids={"candidate-7"}
    )

    assert first["acquisition"]["selected_row_ids"] == second["acquisition"]["selected_row_ids"]
    assert "candidate-7" not in first["acquisition"]["selected_row_ids"]
    assert len(first["measurement_request"]["rows"]) == 4


def test_a2_later_round_reuses_exact_frozen_bundle() -> None:
    """Catches A2 accepting a different later-round prediction or bundle identity."""
    first = policy.select_stage7_round(
        variant="without_measured_feedback", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    later = policy.select_stage7_round(
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
        policy.select_stage7_round(
            variant="without_measured_feedback", seed=20260718, round_index=1,
            candidate_pool=_candidates(), selected_ids=set(first["acquisition"]["selected_row_ids"]), a2_frozen=tampered,
            expected_a2_frozen_sha256=first["a2_frozen"]["frozen_payload_sha256"],
            previous_measurement_request=first["measurement_request"], feedback_rows=feedback,
        )


def test_a2_requires_the_explicit_round_zero_frozen_identity() -> None:
    """Catches a self-hashed replacement bundle changing later A2 selection."""
    first = policy.select_stage7_round(
        variant="without_measured_feedback", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    replacement = copy.deepcopy(first["a2_frozen"])
    replacement["candidate_predictions"].reverse()
    unsigned = dict(replacement)
    unsigned.pop("frozen_payload_sha256")
    replacement["frozen_payload_sha256"] = policy.canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="frozen identity"):
        policy.select_stage7_round(
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
    selection = policy.select_stage7_round(
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
        policy.select_stage7_round(
            variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(), feedback_rows=rows,
        )
    wrong[0]["terminal_status"] = "completed"
    with pytest.raises(ValueError, match="identity"):
        policy.select_stage7_round(
            variant="full", seed=20260718, round_index=1, candidate_pool=_candidates(), feedback_rows=wrong,
            previous_measurement_request=request,
        )


def test_backend_blind_selection_cannot_consume_score_or_derived_features() -> None:
    """Catches backend-blind selection being driven by unprojected backend scores."""
    rows = [
        {
            "row_id": f"candidate-{index}", "score": float(index), "width": [16 + index, 32, 64], "q_mode": "fp16",
            "model_features": {"capability_score": float(index), "static_depth": 2.0},
            "feature_provenance": {"capability_score": "capability_profile-derived", "static_depth": "static_config"},
        }
        for index in range(5)
    ]
    altered = copy.deepcopy(rows)
    for row in altered:
        row["score"] = -float(row["score"])

    first = policy.select_stage7_round(variant="backend_blind", seed=20260718, round_index=0, candidate_pool=rows)
    second = policy.select_stage7_round(variant="backend_blind", seed=20260718, round_index=0, candidate_pool=altered)

    assert first["acquisition"]["selected_row_ids"] == second["acquisition"]["selected_row_ids"]
    assert "model:capability_score" in first["backend_blind_audit"]["removed_names"]


def test_request_binding_rejects_trajectory_identity_drift() -> None:
    """Catches request-binding reuse under another variant, seed, directory, or round."""
    trajectory = policy.build_trajectory_contract(
        variant="full", seed=20260718, result_root="/tmp/stage7",
        frozen_input_sha256="a" * 64, candidate_pool_sha256="b" * 64,
    )
    selection = policy.select_stage7_round(
        variant="full", seed=20260718, round_index=0, candidate_pool=_candidates()
    )
    binding = policy.build_request_binding(trajectory, selection["measurement_request"], round_index=0)
    assert policy.validate_request_binding(binding, selection["measurement_request"], trajectory)["verdict"] == "pass"
    drifted = dict(trajectory, variant="backend_blind")
    with pytest.raises(ValueError, match="identity"):
        policy.validate_request_binding(binding, selection["measurement_request"], drifted)
