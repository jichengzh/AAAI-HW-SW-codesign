from __future__ import annotations

import pytest

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


def test_hardware_blind_selection_never_echoes_extra_context_or_path_like_ids() -> None:
    private_marker = "/private/capability-profile/manifest.json"
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "width": [16 + index, 32, 64],
            "q_mode": "fp16",
            "parameter_count": 100 - index,
            "flops": 200 - index,
            "ap_surrogate": 0.5 + index / 100,
            "capability_profile": private_marker,
            "manifest": {"source_contract": private_marker},
            "source_contract": {"path": private_marker},
        }
        for index in range(4)
    ]

    selected = select_hardware_blind_batch(candidates, batch_size=4)

    assert private_marker not in repr(selected)
    assert all(set(row) <= {
        "candidate_id",
        "width",
        "q_mode",
        "parameter_count",
        "flops",
        "ap_surrogate",
        "hardware_blind_acquisition_score",
    } for row in selected)

    candidates[0]["candidate_id"] = private_marker
    with pytest.raises(ValueError, match="invalid hardware-blind candidate"):
        select_hardware_blind_batch(candidates, batch_size=4)


def test_hardware_blind_selection_rejects_malformed_static_output_fields() -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "width": [16 + index, 32, 64],
            "q_mode": "fp16",
            "parameter_count": 100 - index,
            "flops": 200 - index,
            "ap_surrogate": 0.5 + index / 100,
        }
        for index in range(4)
    ]
    candidates[0]["width"] = {"source_contract": "private-marker"}

    with pytest.raises(ValueError, match="invalid hardware-blind candidate"):
        select_hardware_blind_batch(candidates, batch_size=4)


def test_forward_serial_locks_four_after_exactly_twelve() -> None:
    screened = [{"candidate_id": str(index), "screen_rank": index} for index in range(12)]

    result = lock_compress_then_tune(screened)

    assert result["screen_count"] == 12
    assert result["locked_count"] == 4
    assert result["lock_precedes_tuning"]


def test_forward_serial_lock_rejects_path_like_candidate_ids_without_echoing_them() -> None:
    private_marker = "/private/stage6/lock-source.json"
    screened = [
        {
            "candidate_id": f"candidate-{index}",
            "screen_rank": index,
            "source_contract": {"path": private_marker},
        }
        for index in range(12)
    ]
    screened[0]["candidate_id"] = private_marker

    with pytest.raises(ValueError, match="invalid hardware-blind candidate") as error:
        lock_compress_then_tune(screened)

    assert private_marker not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


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


def test_reverse_transfer_report_uses_fixed_public_fields_and_reasons() -> None:
    private_marker = "/private/stage6/reverse-transfer.json"
    result = record_reverse_transfer_attempts(
        [
            {
                "candidate_id": private_marker,
                "applicable": False,
                "fallback_used": True,
                "compressed_shape_retuned": True,
                "reason": private_marker,
                "source_contract": {"path": private_marker},
                "trace_path": private_marker,
            },
            {
                "candidate_id": "safe-candidate",
                "applicable": True,
                "reason": private_marker,
            },
        ]
    )

    assert private_marker not in repr(result)
    assert result["attempts"] == [
        {
            "attempt_index": 0,
            "terminal_status": "feasibility_failure",
            "reason_code": "transfer_inapplicable",
            "fallback_used": True,
            "compressed_shape_retuned": True,
        },
        {
            "attempt_index": 1,
            "terminal_status": "transferred_success",
            "reason_code": "transfer_applicable",
            "fallback_used": False,
            "compressed_shape_retuned": False,
        },
    ]


def test_hardware_blind_invalid_number_has_no_exception_chain() -> None:
    private_marker = "/private/stage6/rejected-number"

    class PrivateFailure:
        def __float__(self) -> float:
            raise ValueError(private_marker)

    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "parameter_count": 100 - index,
            "flops": 200 - index,
            "ap_surrogate": 0.5 + index / 100,
        }
        for index in range(4)
    ]
    candidates[0]["flops"] = PrivateFailure()

    with pytest.raises(ValueError, match="invalid hardware-blind candidate") as error:
        select_hardware_blind_batch(candidates, batch_size=4)

    assert private_marker not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
