"""Release gate for the offline dynamic P6 source-reuse lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.coptv2x_h800_search_v2 import run_p6_coptv2x_search
from framework.stage6.p6_source_reuse_evidence_v1 import (
    RUN_CONTEXT_RELATIVE_PATH,
    canonical_json_sha256,
    load_fresh_run_context,
    plan_source_reuse_paths,
    receipt_path_for_group,
    require_selected_groups_ready_current_run,
)
from tests.release.p6_source_reuse_lifecycle_fixture import (
    build_offline_reuse_lifecycle,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_row_ids(
    selected_rows_by_round: tuple[tuple[tuple[str, str], ...], ...],
) -> set[str]:
    return {f"{group_id}|q={q_mode}|profile=h800-tvm-auto" for rows in selected_rows_by_round for group_id, q_mode in rows}


def _expected_candidate_count_from_fixture(fixture: Any) -> int:
    plan = _read_json(fixture.local_output_root / "pyramid_candidate_plan.json")
    return len(plan["candidates"])


def test_zero_gpu_dynamic_lifecycle_reuses_group_source_and_runs_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch source re-materialization, missing context/receipts, and skipped rows."""
    fixture = build_offline_reuse_lifecycle(tmp_path, monkeypatch)
    state = run_p6_coptv2x_search(**fixture.call_kwargs)

    assert state.status == "completed"
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    selected = [item for round_rows in fixture.selected_rows_by_round for item in round_rows]
    assert len(selected) == len(set(selected)) == 16
    selected_group_ids = {group_id for group_id, _q_mode in selected}
    assert len(fixture.source_calls) == len(selected_group_ids)
    assert len(fixture.source_calls) < len(selected)
    assert {call["group_id"] for call in fixture.source_calls} == selected_group_ids
    assert all(
        call["argv"][1::2] == ("--request", "--model", "--group-id", "--gpu")
        and len(call["env"]) == 5
        for call in fixture.source_calls
    )
    assert all(
        len({call["group_id"] for call in fixture.source_calls if call["round"] == round_index})
        == len({
            group_id for group_id, _q_mode in fixture.selected_rows_by_round[round_index]
            if group_id not in {
                prior_group
                for prior_round in fixture.selected_rows_by_round[:round_index]
                for prior_group, _prior_q in prior_round
            }
        })
        for round_index in range(4)
    )

    repeated_later_group = fixture.selected_rows_by_round[0][0][0]
    same_round_group = fixture.selected_rows_by_round[1][0][0]
    assert (repeated_later_group, "int8") in fixture.selected_rows_by_round[2]
    assert {
        q_mode for group_id, q_mode in fixture.selected_rows_by_round[1]
        if group_id == same_round_group
    } == {"fp16", "int8"}
    assert len([call for call in fixture.source_calls if call["group_id"] == repeated_later_group]) == 1
    assert len([call for call in fixture.source_calls if call["group_id"] == same_round_group]) == 1
    for stage in ("quantization", "performance", "ap", "finalization"):
        assert len(fixture.downstream_rows[stage]) == 4
        assert {row_id for batch in fixture.downstream_rows[stage] for row_id in batch} == _selected_row_ids(fixture.selected_rows_by_round)

    registry = _read_json(fixture.local_output_root / "source_registry.json")
    plan = _read_json(fixture.local_output_root / "pyramid_candidate_plan.json")
    expected_candidate_count = _expected_candidate_count_from_fixture(fixture)
    assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
    assert len(registry["groups"]) == plan["structure_count"]
    assert sum(len(group["available_q_modes"]) for group in registry["groups"]) == plan["candidate_count"]
    assert plan["candidate_count"] == expected_candidate_count
    assert expected_candidate_count >= 16
    assert plan["candidate_count"] not in {343, 686}
    assert fixture.call_kwargs["local"].candidate_source_mode == "framework_stage2_search_space"

    context = _read_json(fixture.local_output_root / RUN_CONTEXT_RELATIVE_PATH)
    assert context["created_before_round_index"] == 0
    assert context["candidate_plan_sha256"] == canonical_json_sha256(plan)
    assert context["source_registry_sha256"] == canonical_json_sha256(registry)
    assert sum(record.get("event") == "context_publish" for record in fixture.process_records) == 1

    measured_row_ids: set[str] = set()
    for round_index in range(4):
        request = _read_json(fixture.local_output_root / f"round-{round_index:02d}/measurement_request.json")
        assert request["round_index"] == round_index
        assert len(request["rows"]) == 4
        assert all(row["source_contract"]["training_required"] is True for row in request["rows"])
        assert request["measurement_request_sha256"] == canonical_json_sha256({key: value for key, value in request.items() if key != "measurement_request_sha256"})
        measured_row_ids.update(row["row_id"] for row in request["rows"])
    assert len(measured_row_ids) == 16
    assert measured_row_ids.isdisjoint(fixture.gold176_row_ids)
    assert not any(row_id in fixture.gold176_row_ids for stage_batches in fixture.downstream_rows.values() for batch in stage_batches for row_id in batch)

    assert fixture.fit_input_counts == [176, 180, 184, 188]
    assert "receipt_count" not in _read_json(fixture.local_output_root / "state.json")
    paths = plan_source_reuse_paths(fixture.local_output_root)
    receipt_bytes = fixture.receipt_bytes_before_reuse[repeated_later_group]
    run_context = load_fresh_run_context(
        local_output_root=fixture.local_output_root,
        expected_task_id=fixture.call_kwargs["contract"].search_id,
        expected_task_sha256=fixture.task_sha256,
    )
    for group_id in selected_group_ids:
        request = next(
            _read_json(fixture.local_output_root / f"round-{round_index:02d}/measurement_request.json")
            for round_index, rows in enumerate(fixture.selected_rows_by_round)
            if any(candidate_group == group_id for candidate_group, _q_mode in rows)
        )
        receipts = require_selected_groups_ready_current_run(
            request,
            run_context=run_context,
            local_output_root=fixture.local_output_root,
            interface=fixture.binding["execution_interface"],
            private_root=fixture.private_root,
        )
        assert any(receipt.group_id == group_id for receipt in receipts)
    assert receipt_path_for_group(paths, repeated_later_group).read_bytes() == receipt_bytes

    for record in fixture.process_records:
        assert record.get("kind") == "injected_boundary"
        assert record.get("launched") is False
    assert not any(record.get("stage") in {"gpu", "training", "tvm", "ap", "latency", "energy", "history"} and record.get("launched") for record in fixture.process_records)
    assert all("receipt" not in call["argv"] and "context" not in call["env"] for call in fixture.source_calls)
    public_bytes = (fixture.local_output_root / "state.json").read_bytes()
    assert b"PRIVATE-OFFLINE-REUSE-TOKEN" not in public_bytes
