from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from threading import Lock
from time import monotonic, sleep
from types import MappingProxyType
from typing import Any

import pytest

from framework.stage6.p6_ap_round_adapter_v1 import (
    P6APRoundAdapterError,
    run_ap_round,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from tests.stage6.test_p6_performance_round_adapter import (
    _expected_env,
    _native_performance_job,
    _native_state_row,
    _write_quantized_task_state,
)
from tests.stage6.test_p6_quantization_round_adapter import (
    _read_json,
    _write_json,
    _write_round_request,
)


class _Result:
    returncode = 0


class _APRunner:
    def __init__(
        self,
        *,
        plan_rows: Sequence[Mapping[str, Any]] | None = None,
        planner_returncode: int = 0,
        sanity_returncode: int = 0,
        full_returncode: int = 0,
    ) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self.max_active_per_gpu: dict[str, int] = {}
        self._active_per_gpu: dict[str, int] = {}
        self._lock = Lock()
        self._plan_rows = tuple(dict(row) for row in plan_rows) if plan_rows else None
        self._planner_returncode = planner_returncode
        self._sanity_returncode = sanity_returncode
        self._full_returncode = full_returncode

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        if shell is not False:
            raise AssertionError("AP adapter must use direct argv")
        stage = self._stage(argv)
        started = monotonic()
        call_index = self._record_call(argv, cwd, env, shell, stage, started)
        result = _Result()
        if stage == "planner":
            result.returncode = self._planner_returncode
            if result.returncode == 0:
                self._write_plan(argv)
            self._finish_call(call_index)
            return result
        gpu = str(env["CUDA_VISIBLE_DEVICES"])
        with self._lock:
            self._active_per_gpu[gpu] = self._active_per_gpu.get(gpu, 0) + 1
            self.max_active_per_gpu[gpu] = max(
                self.max_active_per_gpu.get(gpu, 0),
                self._active_per_gpu[gpu],
            )
        sleep(0.02)
        result.returncode = self._sanity_returncode if stage == "sanity" else self._full_returncode
        if result.returncode == 0:
            self._write_state(argv)
        with self._lock:
            self._active_per_gpu[gpu] -= 1
        self._finish_call(call_index)
        return result

    def _record_call(
        self,
        argv: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
        stage: str,
        started: float,
    ) -> int:
        with self._lock:
            self.calls.append(
                {
                    "argv": tuple(argv),
                    "cwd": cwd,
                    "env": MappingProxyType(dict(env)),
                    "shell": shell,
                    "stage": stage,
                    "started": started,
                    "finished": None,
                }
            )
            return len(self.calls) - 1

    def _finish_call(self, index: int) -> None:
        with self._lock:
            self.calls[index]["finished"] = monotonic()

    @staticmethod
    def _stage(argv: Sequence[str]) -> str:
        if "--output-jsonl" in argv:
            return "planner"
        if "--stage" in argv:
            return str(argv[argv.index("--stage") + 1])
        raise AssertionError(f"unexpected argv: {argv!r}")

    def _write_plan(self, argv: Sequence[str]) -> None:
        output_json = Path(argv[argv.index("--output-json") + 1])
        output_jsonl = Path(argv[argv.index("--output-jsonl") + 1])
        rows = list(self._plan_rows or _native_ap_plan_rows(_read_json(output_json.parent.parent / "measurement-request.json")))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_json,
            {
                "schema_version": "stage5_ap_plan_v2",
                "row_count": len(rows),
                "jobs": rows,
            },
        )
        output_jsonl.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_state(self, argv: Sequence[str]) -> None:
        stage = str(argv[argv.index("--stage") + 1])
        plan_rows = _read_jsonl(Path(argv[argv.index("--ap-plan-jsonl") + 1]))
        state_path = Path(argv[argv.index("--state-jsonl") + 1])
        existing = _read_jsonl(state_path) if state_path.is_file() else []
        additions = [
            _native_ap_terminal(row, stage)
            for row in plan_rows
            if row.get("ap_terminal") == "ready"
        ]
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in [*existing, *additions]
            ),
            encoding="utf-8",
        )


def test_ap_round_plans_once_partitions_ready_rows_and_runs_sanity_before_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: AP planner/order/shards/GPU binding drift from native controller."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    original_rows = _read_json(task_state)["rows"]
    runner = _APRunner()

    run_ap_round(profile, task_state, round_root, runner)

    assert [call["stage"] for call in runner.calls].count("planner") == 1
    assert {call["stage"] for call in runner.calls[1:4]} == {"sanity"}
    assert {call["stage"] for call in runner.calls[4:]} == {"full"}
    assert max(call["finished"] for call in runner.calls if call["stage"] == "sanity") <= min(
        call["started"] for call in runner.calls if call["stage"] == "full"
    )
    assert runner.max_active_per_gpu == {"2": 1, "5": 1, "7": 1}
    assert _read_jsonl(round_root / "ap/ap_plan_shard_0.jsonl") == [
        _native_ap_plan_rows(request)[0],
        _native_ap_plan_rows(request)[3],
    ]
    assert _read_jsonl(round_root / "ap/ap_plan_shard_1.jsonl") == [
        _native_ap_plan_rows(request)[1],
    ]
    assert _read_jsonl(round_root / "ap/ap_plan_shard_2.jsonl") == [
        _native_ap_plan_rows(request)[2],
    ]
    _assert_historical_call_contract(runner.calls, profile, tmp_path, task_state, round_root)
    assert _read_json(task_state) == {"stage": "ap", "rows": original_rows}


def _assert_historical_call_contract(
    calls: Sequence[Mapping[str, Any]],
    profile: ValidatedPostSourceAdapterProfile,
    tmp_path: Path,
    task_state: Path,
    round_root: Path,
) -> None:
    ordered_stage_calls = _ordered_stage_calls(calls)
    assert [call["env"]["CUDA_VISIBLE_DEVICES"] for call in ordered_stage_calls] == [
        "2",
        "5",
        "7",
        "2",
        "5",
        "7",
    ]
    assert calls[0]["argv"] == _expected_argv(profile, round_root)[0]
    assert [call["argv"] for call in ordered_stage_calls] == _expected_argv(profile, round_root)[1:]
    for call in calls:
        assert call["cwd"] == (
            profile.leaves[0].implementation_cwd
            if call["stage"] == "planner"
            else profile.leaves[1].implementation_cwd
        )
        assert call["shell"] is False
        assert call["env"] == _expected_call_env(profile, tmp_path, task_state, round_root, call)


def test_ap_round_preserves_numerical_skip_and_performance_blocked_native_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: native AP terminal statuses are aliased or over-required."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    rows = _native_ap_plan_rows(request)
    rows[1]["runner_key"] = "codriving_tvm_int8_numeric_gate"
    rows[2]["ap_terminal"] = "blocked_performance_not_success"
    runner = _APRunner(plan_rows=rows)

    run_ap_round(profile, task_state, round_root, runner)

    state = _read_jsonl(round_root / "ap/ap_state.jsonl")
    latest = {(row["job_id"], row["stage"]): row for row in state if row["record_type"] == "job_terminal"}
    assert latest[(rows[0]["manifest_job_id"], "full")]["status"] == "success"
    assert latest[(rows[1]["manifest_job_id"], "sanity")]["status"] == "failed"
    assert latest[(rows[1]["manifest_job_id"], "sanity")]["failure_reason"] == "numerical_feasibility_failure"
    assert latest[(rows[1]["manifest_job_id"], "full")]["status"] == "skipped_numerical_feasibility"
    assert latest[(rows[1]["manifest_job_id"], "full")]["failure_reason"] == "numerical_feasibility_failure"
    assert (rows[2]["manifest_job_id"], "full") not in latest
    assert _read_json(task_state)["stage"] == "ap"


def test_ap_round_accepts_all_performance_blocked_native_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: adapter requires AP state for native blocked_* plan rows."""
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_performance_task_state(round_root, request)
    _write_performance_outputs(round_root, request)
    rows = [
        {**row, "ap_terminal": "blocked_performance_not_success"}
        for row in _native_ap_plan_rows(request)
    ]

    run_ap_round(profile, task_state, round_root, _APRunner(plan_rows=rows))

    assert (round_root / "ap/ap_state.jsonl").read_text(encoding="utf-8") == ""
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


def _profile(
    tmp_path: Path,
    *,
    plan_name: str = "ap_plan",
    execute_name: str = "ap_execute",
) -> ValidatedPostSourceAdapterProfile:
    private_root = tmp_path / "private"
    plan_cwd = private_root / "plan-cwd"
    execute_cwd = private_root / "execute-cwd"
    plan_cwd.mkdir(parents=True)
    execute_cwd.mkdir(parents=True)
    return ValidatedPostSourceAdapterProfile(
        schema_version="p6_post_source_adapter_profile_v1",
        private_root=private_root,
        project_python=Path(sys.executable),
        adapters=(),
        leaves=(
            PostSourceLeaf(plan_name, _write_leaf(plan_cwd / "stage5_ap_plan_v2.py"), plan_cwd, "0" * 64),
            PostSourceLeaf(execute_name, _write_leaf(execute_cwd / "stage3_execute_ap_plan_v3.py"), execute_cwd, "1" * 64),
        ),
    )


def _write_leaf(path: Path) -> Path:
    path.write_text("# fake AP leaf\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _write_performance_task_state(round_root: Path, request: Mapping[str, Any]) -> Path:
    task_state = _write_quantized_task_state(round_root, request)
    state = _read_json(task_state)
    state["stage"] = "performance"
    _write_json(task_state, state)
    return task_state


def _write_performance_outputs(round_root: Path, request: Mapping[str, Any]) -> None:
    performance_root = round_root / "performance"
    performance_root.mkdir(parents=True)
    rows = request["rows"]
    _write_json(
        performance_root / "performance_manifest.json",
        {
            "schema_version": "stage5_performance_manifest_v2",
            "row_count": 4,
            "jobs": [_native_performance_manifest_row(row) for row in rows],
        },
    )
    jobs = [_native_performance_job(row) for row in rows]
    (performance_root / "performance_jobs.jsonl").write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    (performance_root / "performance_state.jsonl").write_text(
        "".join(
            json.dumps(_native_state_row(job["job_id"], "success"), sort_keys=True) + "\n"
            for job in jobs
        ),
        encoding="utf-8",
    )


def _native_performance_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "schema_version": "stage5_performance_manifest_row_v1",
        "job_id": row["manifest_job_id"],
        "split": "online_feedback",
        "source_pool": "stage5_online_feedback",
        "required_metrics": ["latency", "energy", "ap"],
        "terminal_status": "pending",
    }


def _native_ap_plan_rows(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_native_ap_plan_row(row) for row in request["rows"]]


def _native_ap_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    job_id = str(row["manifest_job_id"])
    runner_key = "pyramid_tvm_int8_numeric_gate" if row["q_mode"] == "int8" else "pyramid_tvm_fp16_bridge"
    return {
        "schema_version": "stage5_ap_plan_v2",
        "manifest_job_id": job_id,
        "model": row["model"],
        "width": list(row["width"]),
        "q": row["q_mode"],
        "profile": row["capability_profile_id"],
        "required_metrics": ["latency", "energy", "ap"],
        "source_contract": dict(row["source_contract"]),
        "performance_terminal": "success",
        "performance_job_id": f"{row['group_id']}|tvm",
        "performance_result_json": f"/native/perf/{job_id}.json",
        "runner_key": runner_key,
        "compiled_artifact": f"/native/artifacts/{job_id}.so",
        "compiled_artifact_path": f"/native/artifacts/{job_id}.so",
        "compiled_artifact_digest": "a" * 64,
        "sanity_command": ["python3", "ap.py", "--report-json", f"/native/ap/{job_id}/sanity.json"],
        "full_command": ["python3", "ap.py", "--report-json", f"/native/ap/{job_id}/full.json"],
        "full_command_state_bindings": None,
        "ap_terminal": "ready",
    }


def _native_ap_terminal(row: Mapping[str, Any], stage: str) -> dict[str, Any]:
    job_id = str(row["manifest_job_id"])
    if row.get("runner_key") == "codriving_tvm_int8_numeric_gate" and stage == "sanity":
        status = "failed"
        failure = "numerical_feasibility_failure"
    elif row.get("runner_key") == "codriving_tvm_int8_numeric_gate" and stage == "full":
        status = "skipped_numerical_feasibility"
        failure = "numerical_feasibility_failure"
    else:
        status = "success"
        failure = None
    return {
        "record_type": "job_terminal",
        "job_id": job_id,
        "model": row["model"],
        "stage": stage,
        "status": status,
        "attempts": 0 if status == "skipped_numerical_feasibility" else 1,
        "report_path": f"/native/ap/{job_id}/{stage}.json",
        "report_sha256": "b" * 64,
        "ap": {} if stage == "sanity" else {"ap30": 0.3, "ap50": 0.5, "ap70": 0.7},
        "failure_reason": failure,
        "plan_fingerprint": f"fingerprint-{job_id}-{stage}",
        "timestamp": "2026-08-25T00:00:00+00:00",
    }


def _expected_argv(
    profile: ValidatedPostSourceAdapterProfile,
    round_root: Path,
) -> list[tuple[str, ...]]:
    ap_root = round_root / "ap"
    return [
        (
            str(profile.project_python),
            str(profile.leaves[0].implementation),
            "--manifest-json",
            str(round_root / "performance/performance_manifest.json"),
            "--performance-jobs-jsonl",
            str(round_root / "performance/performance_jobs.jsonl"),
            "--performance-state-jsonl",
            str(round_root / "performance/performance_state.jsonl"),
            "--output-root",
            str(round_root / "ap_execution"),
            "--output-json",
            str(ap_root / "ap_plan.json"),
            "--output-jsonl",
            str(ap_root / "ap_plan.jsonl"),
        ),
        *_stage_argv(profile, round_root, "sanity"),
        *_stage_argv(profile, round_root, "full"),
    ]


def _stage_argv(
    profile: ValidatedPostSourceAdapterProfile,
    round_root: Path,
    stage: str,
) -> list[tuple[str, ...]]:
    return [
        (
            str(profile.project_python),
            str(profile.leaves[1].implementation),
            "--ap-plan-jsonl",
            str(round_root / f"ap/ap_plan_shard_{index}.jsonl"),
            "--stage",
            stage,
            "--state-jsonl",
            str(round_root / f"ap/ap_state_shard_{index}.jsonl"),
            "--gpu",
            gpu,
            "--univ2x-python",
            str(profile.project_python),
            "--artifact-root",
            str(round_root / f"ap_execution/{stage}/shard_{index}"),
        )
        for index, gpu in enumerate(("2", "5", "7"))
    ]


def _expected_call_env(
    profile: ValidatedPostSourceAdapterProfile,
    tmp_path: Path,
    task_state: Path,
    round_root: Path,
    call: Mapping[str, Any],
) -> dict[str, str]:
    env = _expected_env(profile, tmp_path, task_state, round_root)
    if call["stage"] != "planner":
        env["CUDA_VISIBLE_DEVICES"] = call["argv"][call["argv"].index("--gpu") + 1]
    return env


def _ordered_stage_calls(calls: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    stage_order = {"sanity": 0, "full": 1}
    return sorted(
        [call for call in calls if call["stage"] != "planner"],
        key=lambda call: (stage_order[call["stage"]], _call_shard_index(call)),
    )


def _call_shard_index(call: Mapping[str, Any]) -> int:
    argv = call["argv"]
    return int(Path(argv[argv.index("--ap-plan-jsonl") + 1]).stem.rsplit("_", 1)[1])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _set_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
    monkeypatch.setenv("P6_HISTORY_RUN_MODE", "bound")
    monkeypatch.setenv("P6_HISTORY_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("P6_HISTORY_TASK_STATE", str(tmp_path / "round/state/task-state.json"))
    monkeypatch.setenv("P6_HISTORY_ROUND_OUTPUT_ROOT", str(tmp_path / "round"))
