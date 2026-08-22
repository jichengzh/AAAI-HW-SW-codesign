"""Public planning versus runtime resolution for private history rounds."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework.stage6.p6_history_measurement_v1 import (
    P6HistoryMeasurementError,
    plan_validated_history_round_paths,
    resolve_validated_history_round_paths,
)


def _interface() -> dict[str, object]:
    return {
        "output_layout": {
            "round_root_template": "private-runs/{round_id}",
            "task_state": {"path_template": "private-runs/{round_id}/state.json"},
        },
        "actual_feedback": {
            "result": {"path_template": "private-runs/{round_id}/result.json"},
            "receipt": {"path_template": "private-runs/{round_id}/receipt.json"},
            "finalization_barrier": {"path_template": "private-runs/{round_id}/barrier.json"},
        },
    }


def test_planner_allows_absent_private_round_but_runtime_requires_public_round(
    tmp_path: Path,
) -> None:
    """A planner must not create or require a destination it merely describes."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    planned = plan_validated_history_round_paths(_interface(), private_root, 2)

    assert planned["round_root"] == private_root / "private-runs/2"
    assert planned["measurement_request"] == private_root / "private-runs/2/measurement-request.json"
    assert not (private_root / "private-runs").exists()

    public_round = tmp_path / "local-output/round-02"
    with pytest.raises(P6HistoryMeasurementError):
        resolve_validated_history_round_paths(_interface(), private_root, public_round, 2)

    public_round.mkdir(parents=True)
    resolved = resolve_validated_history_round_paths(
        _interface(), private_root, public_round, 2
    )
    assert resolved["local_output_root"] == public_round.parent.resolve(strict=True)
    assert {key: value for key, value in resolved.items() if key not in {"history_root", "local_output_root"}} == planned


@pytest.mark.parametrize("round_index", [-1, 4, True])
def test_planner_rejects_noncanonical_round_indices(tmp_path: Path, round_index: object) -> None:
    """Only the fixed four P6 rounds may be rendered into private destinations."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    with pytest.raises(P6HistoryMeasurementError):
        plan_validated_history_round_paths(_interface(), private_root, round_index)  # type: ignore[arg-type]
