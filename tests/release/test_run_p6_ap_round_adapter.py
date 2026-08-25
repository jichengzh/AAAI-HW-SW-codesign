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
from tests.stage6.test_p6_ap_round_adapter import _write_performance_outputs
from tests.stage6.test_p6_performance_round_adapter import _write_quantized_task_state
from tests.stage6.test_p6_quantization_round_adapter import (
    _read_json,
    _write_round_request,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/run_p6_ap_round_adapter.py"


def test_ap_cli_accepts_generated_wrapper_argv_and_executes_native_leaves(
    tmp_path: Path,
) -> None:
    """Break caught: wrapper-compatible AP CLI stops invoking historical leaves."""
    private_root = tmp_path / "private"
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    state = _read_json(task_state)
    state["stage"] = "performance"
    task_state.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    _write_performance_outputs(round_root, request)
    profile = _write_profile(private_root)

    result = _run_cli(profile, task_state, round_root)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ap_complete\n"
    assert result.stderr == ""
    assert _read_json(task_state)["stage"] == "ap"
    log = [
        json.loads(line)
        for line in (round_root / "ap-leaves.log").read_text(encoding="utf-8").splitlines()
    ]
    assert _ordered_log(log) == _expected_leaf_log(round_root)


def test_ap_cli_reports_argument_error_without_mutating_state(
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
        "CUDA_VISIBLE_DEVICES": ",".join(("2", "5", "7")),
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(round_root.parent / "private"),
        "P6_HISTORY_TASK_STATE": str(task_state),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPOSITORY_ROOT),
    }


def _expected_leaf_log(round_root: Path) -> list[dict[str, object]]:
    ap_root = round_root / "ap"
    return [
        {
            "leaf": "plan",
            "argv": [
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
            ],
            "cuda": "2,5,7",
        },
        *_stage_logs(round_root, "sanity"),
        *_stage_logs(round_root, "full"),
    ]


def _stage_logs(round_root: Path, stage: str) -> list[dict[str, object]]:
    return [
        {
            "leaf": "execute",
            "argv": [
                "--ap-plan-jsonl",
                str(round_root / f"ap/ap_plan_shard_{index}.jsonl"),
                "--stage",
                stage,
                "--state-jsonl",
                str(round_root / f"ap/ap_state_shard_{index}.jsonl"),
                "--gpu",
                gpu,
                "--univ2x-python",
                str(Path(sys.executable).resolve(strict=True)),
                "--artifact-root",
                str(round_root / f"ap_execution/{stage}/shard_{index}"),
            ],
            "cuda": gpu,
        }
        for index, gpu in enumerate(("2", "5", "7"))
    ]


def _ordered_log(log: list[dict[str, object]]) -> list[dict[str, object]]:
    planner = [row for row in log if row["leaf"] == "plan"]
    execute = sorted(
        [row for row in log if row["leaf"] == "execute"],
        key=lambda row: (_log_stage_order(row), _log_shard_index(row)),
    )
    return [*planner, *execute]


def _log_stage_order(row: dict[str, object]) -> int:
    argv = row["argv"]
    assert isinstance(argv, list)
    return {"sanity": 0, "full": 1}[str(argv[argv.index("--stage") + 1])]


def _log_shard_index(row: dict[str, object]) -> int:
    argv = row["argv"]
    assert isinstance(argv, list)
    return int(Path(str(argv[argv.index("--ap-plan-jsonl") + 1])).stem.rsplit("_", 1)[1])


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
        if name == "ap_plan":
            body = _plan_leaf_body()
        if name == "ap_execute":
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
    return r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[sys.argv.index("--manifest-json") + 1]).read_text(encoding="utf-8"))
output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
output_json = Path(sys.argv[sys.argv.index("--output-json") + 1])
output_jsonl = Path(sys.argv[sys.argv.index("--output-jsonl") + 1])
rows = [{
    "schema_version": "stage5_ap_plan_v2",
    "manifest_job_id": row["manifest_job_id"],
    "model": row["model"],
    "width": row["width"],
    "q": row["q_mode"],
    "profile": row["capability_profile_id"],
    "required_metrics": ["latency", "energy", "ap"],
    "source_contract": row["source_contract"],
    "performance_terminal": "success",
    "performance_job_id": row["manifest_job_id"] + "|tvm",
    "performance_result_json": "/native/perf/" + row["manifest_job_id"] + ".json",
    "runner_key": "pyramid_tvm_int8_numeric_gate" if row["q_mode"] == "int8" else "pyramid_tvm_fp16_bridge",
    "compiled_artifact": "/native/artifacts/" + row["manifest_job_id"] + ".so",
    "compiled_artifact_path": "/native/artifacts/" + row["manifest_job_id"] + ".so",
    "compiled_artifact_digest": "a" * 64,
    "sanity_command": ["python3", "ap.py", "--report-json", str(output_root / "ap" / row["manifest_job_id"] / "sanity_16" / "full_ap_eval_report.json")],
    "full_command": ["python3", "ap.py", "--report-json", str(output_root / "ap" / row["manifest_job_id"] / "full_1789" / "full_ap_eval_report.json")],
    "full_command_state_bindings": None,
    "ap_terminal": "ready",
} for row in manifest["jobs"]]
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps({"schema_version": "stage5_ap_plan_v2", "row_count": len(rows), "jobs": rows}, sort_keys=True), encoding="utf-8")
output_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "ap-leaves.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"leaf": "plan", "argv": sys.argv[1:], "cuda": os.environ["CUDA_VISIBLE_DEVICES"]}, sort_keys=True) + "\n")
"""


def _execute_leaf_body() -> str:
    return (
        _execute_leaf_imports()
        + _execute_leaf_fingerprint_helpers()
        + _execute_leaf_terminal_writer()
        + _execute_leaf_log_writer()
    )


def _execute_leaf_imports() -> str:
    return r"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

stage = sys.argv[sys.argv.index("--stage") + 1]
rows = [
    json.loads(line)
    for line in Path(sys.argv[sys.argv.index("--ap-plan-jsonl") + 1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
state_path = Path(sys.argv[sys.argv.index("--state-jsonl") + 1])
existing = []
if state_path.is_file():
    existing = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()]
"""


def _execute_leaf_fingerprint_helpers() -> str:
    return r"""

def report_path_from_command(command):
    for index, token in enumerate(command):
        if token == "--report-json" and index + 1 < len(command):
            return command[index + 1]
    return None

def fingerprint(row, stage):
    command = list(map(str, row[stage + "_command"]))
    binding = {
        "runner_key": row.get("runner_key"),
        "compiled_artifact_digest": row.get("compiled_artifact_digest"),
        "compiled_artifact_path": row.get("compiled_artifact_path") or row.get("compiled_artifact"),
        "stage": stage,
        "stage_command": command,
        "report_path": report_path_from_command(command),
    }
    return hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
"""


def _execute_leaf_terminal_writer() -> str:
    return r"""
def report_payload(stage):
    payload = {"status": "success", "processed_samples": 16 if stage == "sanity" else 1789, "fallback_samples": 0, "failed_samples": 0}
    if stage == "full":
        payload.update({"ap": {"ap30": 0.3, "ap50": 0.5, "ap70": 0.7}, "ap_measured": True, "smoke_gate_passed": True})
    return payload

def terminal(row):
    report_path = Path(report_path_from_command(row[stage + "_command"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload(stage), sort_keys=True), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return {
        "record_type": "job_terminal",
        "job_id": row["manifest_job_id"],
        "model": row["model"],
        "stage": stage,
        "status": "success",
        "attempts": 1,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "ap": {} if stage == "sanity" else {"ap30": 0.3, "ap50": 0.5, "ap70": 0.7},
        "failure_reason": None,
        "plan_fingerprint": fingerprint(row, stage),
        "timestamp": "2026-08-25T00:00:00+00:00",
    }

state_path.write_text("".join(
    json.dumps(row, sort_keys=True) + "\n"
    for row in [*existing, *[terminal(row) for row in rows if row["ap_terminal"] == "ready"]]
), encoding="utf-8")
"""


def _execute_leaf_log_writer() -> str:
    return r"""
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "ap-leaves.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"leaf": "execute", "argv": sys.argv[1:], "cuda": os.environ["CUDA_VISIBLE_DEVICES"]}, sort_keys=True) + "\n")
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
