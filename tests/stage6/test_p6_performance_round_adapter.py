from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

import pytest

from framework.stage6.p6_performance_round_adapter_v1 import (
    P6PerformanceRoundAdapterError,
    run_performance_round,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from tests.stage6.test_p6_quantization_round_adapter import (
    _read_json,
    _write_json,
    _write_round_request,
    _write_task_state,
)


class _Result:
    returncode = 0


class _PerformanceRunner:
    def __init__(
        self,
        *,
        planner_returncode: int = 0,
        executor_returncode: int = 0,
        state_statuses: Sequence[str] = ("success", "success", "confirmed_failure", "success"),
        manifest_row_ids: Sequence[str] | None = None,
        job_row_ids: Sequence[str] | None = None,
    ) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self._planner_returncode = planner_returncode
        self._executor_returncode = executor_returncode
        self._state_statuses = tuple(state_statuses)
        self._manifest_row_ids = tuple(manifest_row_ids) if manifest_row_ids is not None else None
        self._job_row_ids = tuple(job_row_ids) if job_row_ids is not None else None

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        if shell is not False:
            raise AssertionError("performance adapter must use direct argv")
        self.calls.append(
            MappingProxyType(
                {
                    "argv": tuple(argv),
                    "cwd": cwd,
                    "env": MappingProxyType(dict(env)),
                    "shell": shell,
                }
            )
        )
        if "--output-dir" in argv:
            return self._run_planner(argv)
        if "--state-jsonl" in argv:
            return self._run_executor(argv)
        raise AssertionError(f"unexpected argv: {argv!r}")

    def _run_planner(self, argv: Sequence[str]) -> _Result:
        result = _Result()
        result.returncode = self._planner_returncode
        if result.returncode != 0:
            return result
        request = _read_json(Path(argv[argv.index("--request-json") + 1]))
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        rows = request["rows"]
        manifest_ids = self._manifest_row_ids or tuple(row["row_id"] for row in rows)
        job_ids = self._job_row_ids or tuple(row["row_id"] for row in rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "performance_manifest.json",
            {
                "schema_version": "stage5_performance_manifest_v2",
                "row_count": 4,
                "rows": [
                    {"row_id": row_id, "manifest_job_id": row_id}
                    for row_id in manifest_ids
                ],
            },
        )
        (output_dir / "performance_jobs.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "job_id": row_id,
                        "row_id": row_id,
                        "manifest_job_id": row_id,
                        "max_attempts": 2,
                    },
                    sort_keys=True,
                )
                + "\n"
                for row_id in job_ids
            ),
            encoding="utf-8",
        )
        return result

    def _run_executor(self, argv: Sequence[str]) -> _Result:
        result = _Result()
        result.returncode = self._executor_returncode
        if result.returncode != 0:
            return result
        jobs = [
            json.loads(line)
            for line in Path(argv[argv.index("--jobs-jsonl") + 1]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        state_path = Path(argv[argv.index("--state-jsonl") + 1])
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "".join(
                json.dumps({"job_id": job["job_id"], "status": status}, sort_keys=True) + "\n"
                for job, status in zip(jobs, self._state_statuses, strict=True)
            ),
            encoding="utf-8",
        )
        return result


def test_performance_round_plans_once_then_executes_with_historical_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: planner/executor argv, order, GPU CSV, or worker count drift."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    original_rows = _read_json(task_state)["rows"]
    runner = _PerformanceRunner()

    run_performance_round(profile, task_state, round_root, runner)

    planner, executor = profile.leaves
    assert [call["argv"] for call in runner.calls] == _expected_argv(
        profile,
        planner,
        executor,
        round_root,
    )
    assert "--tvm-fp16-max-trials" not in runner.calls[0]["argv"]
    assert [call["cwd"] for call in runner.calls] == [
        planner.implementation_cwd,
        executor.implementation_cwd,
    ]
    for call in runner.calls:
        assert call["shell"] is False
        assert call["env"] == _expected_env(profile, tmp_path, task_state, round_root)
    assert _read_json(task_state) == {"stage": "performance", "rows": original_rows}


@pytest.mark.parametrize(
    "statuses",
    [
        ("success", "ready", "confirmed_failure", "success"),
        ("success", "missing", "confirmed_failure", "success"),
        ("success", "failed", "confirmed_failure", "success"),
    ],
)
def test_performance_round_rejects_nonterminal_native_state_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statuses: tuple[str, str, str, str],
) -> None:
    """Break caught: ready/missing/incomplete native rows advance task state."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    original_state = _read_json(task_state)

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(
            profile,
            task_state,
            round_root,
            _PerformanceRunner(state_statuses=statuses),
        )

    assert _read_json(task_state) == original_state


@pytest.mark.parametrize(
    ("planner_returncode", "executor_returncode", "expected_calls"),
    [(9, 0, 1), (0, 8, 2)],
)
def test_performance_round_keeps_quantization_state_on_leaf_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner_returncode: int,
    executor_returncode: int,
    expected_calls: int,
) -> None:
    """Break caught: nonzero planner/executor return advances the round."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    original_state = _read_json(task_state)
    runner = _PerformanceRunner(
        planner_returncode=planner_returncode,
        executor_returncode=executor_returncode,
    )

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(profile, task_state, round_root, runner)

    assert len(runner.calls) == expected_calls
    assert _read_json(task_state) == original_state


@pytest.mark.parametrize(
    "runner",
    [
        _PerformanceRunner(manifest_row_ids=("wrong", "b", "c", "d")),
        _PerformanceRunner(job_row_ids=("wrong", "b", "c", "d")),
    ],
)
def test_performance_round_rejects_manifest_or_job_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: _PerformanceRunner,
) -> None:
    """Break caught: native plan identities drift from the four requested rows."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    original_state = _read_json(task_state)

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(profile, task_state, round_root, runner)

    assert _read_json(task_state) == original_state


def test_performance_round_rejects_missing_execute_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: adapter runs without the native executor leaf binding."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path, execute_name="ap_execute")
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    original_state = _read_json(task_state)

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(profile, task_state, round_root, _PerformanceRunner())

    assert _read_json(task_state) == original_state


def _profile(
    tmp_path: Path,
    *,
    plan_name: str = "performance_plan",
    execute_name: str = "performance_execute",
) -> ValidatedPostSourceAdapterProfile:
    private_root = tmp_path / "private"
    plan_cwd = private_root / "plan-cwd"
    execute_cwd = private_root / "execute-cwd"
    plan_cwd.mkdir(parents=True)
    execute_cwd.mkdir(parents=True)
    plan = _write_leaf(plan_cwd / "stage5_build_performance_plan_v2.py")
    execute = _write_leaf(execute_cwd / "stage3_execute_performance_plan_v3.py")
    return ValidatedPostSourceAdapterProfile(
        schema_version="p6_post_source_adapter_profile_v1",
        private_root=private_root,
        project_python=Path(sys.executable),
        adapters=(),
        leaves=(
            PostSourceLeaf(plan_name, plan, plan_cwd, "0" * 64),
            PostSourceLeaf(execute_name, execute, execute_cwd, "1" * 64),
        ),
    )


def _write_leaf(path: Path) -> Path:
    path.write_text("# fake performance leaf\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _expected_argv(
    profile: ValidatedPostSourceAdapterProfile,
    planner: PostSourceLeaf,
    executor: PostSourceLeaf,
    round_root: Path,
) -> list[tuple[str, ...]]:
    performance_root = round_root / "performance"
    return [
        (
            str(profile.project_python),
            str(planner.implementation),
            "--request-json",
            str(round_root / "measurement-request.json"),
            "--remote-artifact-root",
            str(performance_root / "artifacts"),
            "--output-dir",
            str(performance_root),
            "--quant-contract-root",
            str(round_root / "quant_contracts"),
            "--gpus",
            "2,5,7",
        ),
        (
            str(profile.project_python),
            str(executor.implementation),
            "--jobs-jsonl",
            str(performance_root / "performance_jobs.jsonl"),
            "--state-jsonl",
            str(performance_root / "performance_state.jsonl"),
            "--gpus",
            "2,5,7",
            "--max-workers",
            "3",
        ),
    ]


def _expected_env(
    profile: ValidatedPostSourceAdapterProfile,
    tmp_path: Path,
    task_state: Path,
    round_root: Path,
) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "2,5,7",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(tmp_path / "private"),
        "P6_HISTORY_TASK_STATE": str(task_state),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
        "PATH": f"{profile.project_python.parent}:/usr/bin:/bin",
        "PYTHONPATH": f"{profile.private_root}:{Path.cwd()}",
    }


def _write_quantized_task_state(round_root: Path, request: Mapping[str, Any]) -> Path:
    task_state = _write_task_state(round_root, request)
    state = _read_json(task_state)
    state["stage"] = "quantization"
    _write_json(task_state, state)
    return task_state


def _set_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    round_root = tmp_path / "round"
    task_state = round_root / "state" / "task-state.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
    monkeypatch.setenv("P6_HISTORY_RUN_MODE", "bound")
    monkeypatch.setenv("P6_HISTORY_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("P6_HISTORY_TASK_STATE", str(task_state))
    monkeypatch.setenv("P6_HISTORY_ROUND_OUTPUT_ROOT", str(round_root))
