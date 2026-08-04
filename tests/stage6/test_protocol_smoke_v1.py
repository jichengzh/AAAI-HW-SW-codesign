from __future__ import annotations

from framework.stage6.protocol_smoke_v1 import (
    lock_compress_then_tune,
    record_reverse_transfer_attempts,
    select_hardware_blind_batch,
)


def test_hardware_blind_selection_ignores_backend_labels() -> None:
    candidates = [
        {
            "candidate_id": str(index),
            "width": [16 + index, 32, 64],
            "q_mode": "fp16" if index % 2 else "int8",
            "parameter_count": 100 - index,
            "flops": 200 - index,
            "ap_surrogate": 0.5 + index / 100,
            "latency_ms": -1000 * index,
            "energy_j": -1000 * index,
        }
        for index in range(8)
    ]
    for index, row in enumerate(candidates):
        row.update(
            {
                "capability_profile_id": f"profile-{index}",
                "manifest_job_id": f"job-{index}",
                "source_contract": {"artifact_id": f"artifact-{index}"},
            }
        )
    selected = select_hardware_blind_batch(candidates, batch_size=4)
    perturbed = [
        {**row, "latency_ms": 10**9 + index, "energy_j": 10**9 - index}
        for index, row in enumerate(candidates)
    ]
    selected_perturbed = select_hardware_blind_batch(perturbed, batch_size=4)

    assert [row["candidate_id"] for row in selected] == [
        row["candidate_id"] for row in selected_perturbed
    ]
    assert all("latency_ms" not in row and "energy_j" not in row for row in selected)
    allowed_fields = {
        "candidate_id",
        "width",
        "q_mode",
        "parameter_count",
        "flops",
        "ap_surrogate",
        "hardware_blind_acquisition_score",
    }
    assert all(set(row) <= allowed_fields for row in selected)


def test_forward_serial_locks_four_after_exactly_twelve() -> None:
    screened = [{"candidate_id": str(index), "screen_rank": index} for index in range(12)]

    result = lock_compress_then_tune(screened)

    assert result["screen_count"] == 12
    assert result["locked_count"] == 4
    assert result["lock_precedes_tuning"]


def test_reverse_transfer_keeps_failures_without_fallback() -> None:
    result = record_reverse_transfer_attempts(
        [
            {"candidate_id": "a", "applicable": False, "reason": "shape_mismatch"},
            {"candidate_id": "b", "applicable": True},
        ]
    )

    assert result["failure_count"] == 1
    assert result["fallback_count"] == 0
    assert result["retune_count"] == 0


def test_reverse_transfer_preserves_fallback_and_retune_violations() -> None:
    result = record_reverse_transfer_attempts(
        [
            {
                "candidate_id": "bad",
                "applicable": False,
                "fallback_used": True,
                "compressed_shape_retuned": True,
            }
        ]
    )

    assert result["fallback_count"] == 1
    assert result["retune_count"] == 1
