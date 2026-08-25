from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_ap_round_adapter_v1 import (
    P6APRoundAdapterError,
    run_ap_round,
)
from tests.stage6.test_p6_ap_round_adapter import (
    _APRunner,
    _materialize_native_ap_report,
    _native_ap_plan_rows,
    _profile,
    _read_json,
    _set_runtime_env,
    _write_performance_outputs,
    _write_performance_task_state,
    _write_round_request,
    _write_stale_native_plan,
)


@pytest.mark.parametrize(
    "stale_writer",
    [
        lambda round_root, request: _write_stale_native_plan(round_root, request),
        lambda round_root, request: (round_root / "ap/ap_plan_shard_0.jsonl").write_text(
            "{}", encoding="utf-8"
        ),
        lambda round_root, request: (round_root / "ap/ap_state_shard_0.jsonl").write_text(
            "{}", encoding="utf-8"
        ),
        lambda round_root, request: (round_root / "ap/ap_state.jsonl").write_text(
            "{}", encoding="utf-8"
        ),
    ],
)
def test_ap_round_rejects_preexisting_ap_outputs_before_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_writer: Any,
) -> None:
    """Break caught: stale AP plan/shard/state outputs are consumed after planner rc=1."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    (round_root / "ap").mkdir(parents=True, exist_ok=True)
    stale_writer(round_root, request)
    original_state = _read_json(task_state)
    runner = _APRunner(planner_returncode=1, planner_writes_outputs=False)

    with pytest.raises(P6APRoundAdapterError):
        run_ap_round(profile, task_state, round_root, runner)

    assert runner.calls == []
    assert _read_json(task_state) == original_state


def test_ap_round_rejects_preexisting_execution_reports_before_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: rc=0 shards bind fresh state to stale AP report bytes."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    rows = _native_ap_plan_rows(request)
    for row in rows:
        _materialize_native_ap_report(row, "sanity", round_root / "ap_execution")
        _materialize_native_ap_report(row, "full", round_root / "ap_execution")
    original_state = _read_json(task_state)
    runner = _APRunner(plan_rows=rows, reuse_existing_reports=True)

    with pytest.raises(P6APRoundAdapterError):
        run_ap_round(profile, task_state, round_root, runner)

    assert runner.calls == []
    assert _read_json(task_state) == original_state


def test_ap_round_allows_empty_execution_directories_and_unrelated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The freshness boundary covers historical report outputs, not the whole root."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    empty = round_root / "ap_execution/sanity/shard_0/empty"
    empty.mkdir(parents=True)
    (round_root / "ap_execution/operator-note.txt").write_text("unrelated", encoding="utf-8")

    run_ap_round(profile, task_state, round_root, _APRunner())

    assert empty.is_dir()
    assert _read_json(task_state)["stage"] == "ap"


@pytest.mark.parametrize(
    ("sanity_returncode", "full_returncode", "expected_stage"),
    [(6, 0, "sanity"), (0, 7, "full")],
)
def test_ap_round_keeps_performance_state_on_leaf_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sanity_returncode: int,
    full_returncode: int,
    expected_stage: str,
) -> None:
    """Break caught: task state advances after a nonzero AP shard leaf."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    original_state = _read_json(task_state)
    runner = _APRunner(sanity_returncode=sanity_returncode, full_returncode=full_returncode)

    with pytest.raises(P6APRoundAdapterError):
        run_ap_round(profile, task_state, round_root, runner)

    assert expected_stage in [call["stage"] for call in runner.calls]
    assert _read_json(task_state) == original_state


@pytest.mark.parametrize(
    "plan_rows",
    [
        lambda request: _native_ap_plan_rows(request)[:3],
        lambda request: [
            {**row, "manifest_job_id": "stale"} if index == 0 else row
            for index, row in enumerate(_native_ap_plan_rows(request))
        ],
    ],
)
def test_ap_round_rejects_missing_or_stale_native_plan_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_rows: Any,
) -> None:
    """Break caught: AP plan rows can disappear or drift from the four request rows."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    original_state = _read_json(task_state)

    with pytest.raises(P6APRoundAdapterError):
        run_ap_round(profile, task_state, round_root, _APRunner(plan_rows=plan_rows(request)))

    assert _read_json(task_state) == original_state


def test_ap_round_rejects_missing_execute_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: adapter runs without the native AP executor leaf binding."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path, execute_name="performance_execute")
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)

    with pytest.raises(P6APRoundAdapterError):
        run_ap_round(profile, task_state, round_root, _APRunner())
