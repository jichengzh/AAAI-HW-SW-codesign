from __future__ import annotations

import pytest

from scripts.fcooper_close_formal_t16_v2 import validate_round_identity


def _request(round_index: int) -> dict:
    rows = [
        {
            "row_id": f"r{round_index}-{index}",
            "manifest_job_id": f"r{round_index}-{index}",
            "task_id": "S5-FCO-TRT-V2",
        }
        for index in range(4)
    ]
    return {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S5-FCO-TRT-V2",
        "round_index": round_index,
        "rows": rows,
    }


def test_round_identity_requires_exact_requested_released_rows() -> None:
    request = _request(2)
    feedback = [
        {**row, "round_index": 2, "training_source": "online_feedback"}
        for row in request["rows"]
    ]
    audit = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": 4,
        "released_feedback_rows": feedback,
    }

    assert validate_round_identity(request, feedback, audit, round_index=2) == {
        "r2-0",
        "r2-1",
        "r2-2",
        "r2-3",
    }

    audit["feedback_released"] = False
    with pytest.raises(ValueError, match="atomic"):
        validate_round_identity(request, feedback, audit, round_index=2)


def test_round_identity_rejects_cross_round_feedback() -> None:
    request = _request(1)
    feedback = [
        {**row, "round_index": 0, "training_source": "online_feedback"}
        for row in request["rows"]
    ]
    audit = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": 4,
        "released_feedback_rows": feedback,
    }

    with pytest.raises(ValueError, match="round"):
        validate_round_identity(request, feedback, audit, round_index=1)
