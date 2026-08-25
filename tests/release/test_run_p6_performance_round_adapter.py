from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from framework.stage6.p6_post_source_adapter_profile_v1 import (
    POST_SOURCE_ADAPTER_STAGES,
    POST_SOURCE_LEAF_NAMES,
)
from tests.stage6.test_p6_quantization_round_adapter import (
    _read_json,
    _write_json,
    _write_round_request,
    _write_task_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/run_p6_performance_round_adapter.py"


def test_performance_cli_accepts_generated_wrapper_argv_and_executes_native_leaves(
    tmp_path: Path,
) -> None:
    """Break caught: wrapper-compatible CLI shape stops invoking the native leaves."""
    private_root = tmp_path / "private"
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    profile = _write_profile(private_root)

    result = _run_cli(profile, task_state, round_root)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "performance_complete\n"
    assert result.stderr == ""
    assert _read_json(task_state)["stage"] == "performance"
    log = [
        json.loads(line)
        for line in (round_root / "performance-leaves.log").read_text(encoding="utf-8").splitlines()
    ]
    assert log == _expected_leaf_log(round_root)


def test_performance_cli_reports_argument_error_without_mutating_state(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    original_state = _read_json(task_state)
    profile = _write_profile(private_root)

    result = subprocess.run(
        [sys.executable, str(CLI), "--profile", str(profile), str(task_state)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=_env(task_state, round_root),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert _read_json(task_state) == original_state


def _run_cli(
    profile: Path,
    task_state: Path,
    round_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--profile",
            str(profile),
            str(task_state),
            str(round_root),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=_env(task_state, round_root),
    )


def _env(task_state: Path, round_root: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "2,5,7",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(round_root.parent / "private"),
        "P6_HISTORY_TASK_STATE": str(task_state),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPOSITORY_ROOT),
    }


def _expected_leaf_log(round_root: Path) -> list[dict[str, object]]:
    performance_root = round_root / "performance"
    return [
        {
            "leaf": "plan",
            "argv": [
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
            ],
            "cuda": "2,5,7",
        },
        {
            "leaf": "execute",
            "argv": [
                "--jobs-jsonl",
                str(performance_root / "performance_jobs.jsonl"),
                "--state-jsonl",
                str(performance_root / "performance_state.jsonl"),
                "--gpus",
                "2,5,7",
                "--max-workers",
                "3",
            ],
            "cuda": "2,5,7",
        },
    ]


def _write_profile(private_root: Path) -> Path:
    adapter_cwd = private_root / "adapter-cwd"
    leaf_cwd = private_root / "leaf-cwd"
    adapter_cwd.mkdir(parents=True)
    leaf_cwd.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(private_root)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    adapters = {
        stage: _profile_entry(
            private_root,
            _write_executable(adapter_cwd / f"{stage}.py", "# adapter\n"),
            adapter_cwd,
        )
        for stage in POST_SOURCE_ADAPTER_STAGES
    }
    leaves = {}
    for name in POST_SOURCE_LEAF_NAMES:
        body = "# leaf\n"
        if name == "performance_plan":
            body = _plan_leaf_body()
        if name == "performance_execute":
            body = _execute_leaf_body()
        implementation = _write_executable(leaf_cwd / f"{name}.py", body)
        leaves[name] = {
            **_profile_entry(private_root, implementation, leaf_cwd),
            "sha256": _sha256(implementation),
        }
    profile = {
        "schema_version": "p6_post_source_adapter_profile_v1",
        "target": {"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"},
        "runner_interface_schema_version": "p6_history_runner_interface_v1",
        "project_python": str(Path(sys.executable).resolve(strict=True)),
        "adapters": adapters,
        "leaves": leaves,
    }
    path = private_root / "post-source-adapter-profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def _profile_entry(private_root: Path, implementation: Path, cwd: Path) -> dict[str, str]:
    return {
        "implementation_relative_path": implementation.relative_to(private_root).as_posix(),
        "implementation_cwd_relative_path": cwd.relative_to(private_root).as_posix(),
    }


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o700)
    return path


def _plan_leaf_body() -> str:
    return (
        _plan_leaf_imports()
        + _plan_leaf_manifest_writer()
        + _plan_leaf_jobs_writer()
        + _plan_leaf_log_writer()
    )


def _plan_leaf_imports() -> str:
    return r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[sys.argv.index("--request-json") + 1]).read_text(encoding="utf-8"))
output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
output_dir.mkdir(parents=True, exist_ok=True)
rows = request["rows"]
"""


def _plan_leaf_manifest_writer() -> str:
    return r"""
(output_dir / "performance_manifest.json").write_text(json.dumps({
    "schema_version": "stage5_performance_manifest_v2",
    "source_request_schema": request["schema_version"],
    "source_request_sha256": request["measurement_request_sha256"],
    "task_id": request["task_id"],
    "task_sha256": request["task_sha256"],
    "source_pool": "stage5_online_feedback",
    "genome_count": 4,
    "row_count": 4,
    "group_count": 4,
    "group_ids": [row["group_id"] for row in rows],
    "jobs": [{
        **row,
        "schema_version": "stage5_performance_manifest_row_v1",
        "job_id": row["manifest_job_id"],
        "manifest_job_id": row["manifest_job_id"],
        "split": "online_feedback",
        "source_pool": "stage5_online_feedback",
        "required_metrics": ["latency", "energy", "ap"],
        "source_status": "ready",
        "source_evidence_path": "/native/evidence/" + row["manifest_job_id"] + ".json",
        "source_evidence_sha256": "d" * 64,
        "terminal_status": "pending",
    } for row in rows],
}, sort_keys=True), encoding="utf-8")
"""


def _plan_leaf_jobs_writer() -> str:
    return r"""
(output_dir / "performance_jobs.jsonl").write_text("".join(
    json.dumps({
        "schema_version": "stage35_gold32_performance_job_v1",
        "job_id": row["group_id"] + "|" + ("tvm_int8" if row["q_mode"] == "int8" else "tvm_fp16"),
        "manifest_job_id": row["manifest_job_id"],
        "group_id": row["group_id"],
        "model": row["model"],
        "width_key": "x".join(str(item) for item in row["width"]),
        "q_mode": row["q_mode"],
        "runner_key": "tvm_int8" if row["q_mode"] == "int8" else "tvm_fp16",
        "dispatch_key": row["dispatch_key"],
        "split": "online_feedback",
        "onnx_path": row["source_contract"]["onnx_path"],
        "calibration_root": "/native/calibration/" + row["manifest_job_id"],
        "source_contract": row["source_contract"],
        "command": ["/native/python", "measure.py", "--gpu", "2"],
        "assigned_gpu": 2,
        "gpu_pool": "2,5,7",
        "remote_artifact_root": "/native/artifacts",
        "expected_result_json": "/native/artifacts/" + row["manifest_job_id"] + "/result.json",
        "max_attempts": 2,
        "terminal_status": "pending",
    }, sort_keys=True) + "\n"
    for row in rows
), encoding="utf-8")
"""


def _plan_leaf_log_writer() -> str:
    return r"""
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "performance-leaves.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "leaf": "plan",
        "argv": sys.argv[1:],
        "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
    }, sort_keys=True) + "\n")
"""


def _execute_leaf_body() -> str:
    return r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

jobs = [
    json.loads(line)
    for line in Path(sys.argv[sys.argv.index("--jobs-jsonl") + 1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
state_path = Path(sys.argv[sys.argv.index("--state-jsonl") + 1])
state_path.write_text("".join(
    json.dumps({
        "schema_version": "stage3_execute_performance_plan_v3_state",
        "job_id": row["job_id"],
        "attempt": 1,
        "status": "success",
        "returncode": 0,
        "start_time_unix": 1.0,
        "end_time_unix": 2.0,
        "elapsed_s": 1.0,
        "stdout_path": "/native/logs/" + row["job_id"] + ".stdout.txt",
        "stderr_path": "/native/logs/" + row["job_id"] + ".stderr.txt",
        "result_json": "/native/results/" + row["job_id"] + ".json",
        "result_sha256": "e" * 64,
        "failure_reasons": [],
    }, sort_keys=True) + "\n"
    for row in jobs
), encoding="utf-8")
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "performance-leaves.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "leaf": "execute",
        "argv": sys.argv[1:],
        "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
    }, sort_keys=True) + "\n")
"""


def _write_quantized_task_state(round_root: Path, request: dict[str, object]) -> Path:
    task_state = _write_task_state(round_root, request)
    state = _read_json(task_state)
    state["stage"] = "quantization"
    _write_json(task_state, state)
    return task_state


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
