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
    _write_round_request,
    _write_task_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/run_p6_quantization_round_adapter.py"


def test_quantization_cli_accepts_generated_wrapper_argv_and_executes_leaf(
    tmp_path: Path,
) -> None:
    """Break caught: wrapper-compatible CLI shape stops invoking the real leaf."""
    private_root = tmp_path / "private"
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "fp16"))
    task_state = _write_task_state(round_root, request)
    profile = _write_profile(private_root)

    result = _run_cli(profile, task_state, round_root)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "quantization_complete\n"
    assert result.stderr == ""
    assert _read_json(task_state)["stage"] == "quantization"
    log = [
        json.loads(line)
        for line in (round_root / "quant-leaf.log").read_text(encoding="utf-8").splitlines()
    ]
    assert log == [
        {
            "argv": [
                "--onnx",
                request["rows"][1]["source_contract"]["onnx_path"],
                "--calibration-npz",
                request["rows"][1]["source_contract"]["calibration_npz"],
                "--calibration-summary",
                request["rows"][1]["source_contract"]["calibration_summary"],
                "--output-json",
                str(round_root / "quant_contracts/16x32x64/tensor_quant_params.json"),
            ],
            "cuda": "2",
            "round_root": str(round_root),
        }
    ]


def test_quantization_cli_reports_argument_error_without_mutating_state(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("int8", "fp16", "fp16", "fp16"))
    task_state = _write_task_state(round_root, request)
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
    leaves = {
        name: {
            **_profile_entry(
                private_root,
                _write_executable(
                    leaf_cwd / f"{name}.py",
                    _quant_leaf_body() if name == "quant_contract" else "# leaf\n",
                ),
                leaf_cwd,
            ),
            "sha256": _sha256(leaf_cwd / f"{name}.py"),
        }
        for name in POST_SOURCE_LEAF_NAMES
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


def _quant_leaf_body() -> str:
    return r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

output_path = Path(sys.argv[sys.argv.index("--output-json") + 1])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps({"schema": "stage3_tvm_int8_quant_contract_v3", "scales": [1.0]}),
    encoding="utf-8",
)
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "quant-leaf.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
        "round_root": str(round_root),
    }, sort_keys=True) + "\n")
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
