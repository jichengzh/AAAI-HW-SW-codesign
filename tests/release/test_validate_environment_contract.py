"""Black-box tests for the offline environment contract CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "tools" / "release" / "validate_environment_contract.py"
TEST_CAPABILITY = REPOSITORY_ROOT / "configs" / "hardware" / "task3" / "rtx4090.yaml"


@pytest.fixture(autouse=True)
def temporary_capability() -> None:
    """Create the fixed-root capability fixture required by the subprocess CLI."""
    TEST_CAPABILITY.parent.mkdir(parents=True, exist_ok=True)
    TEST_CAPABILITY.write_text(
        yaml.safe_dump(
            {
                "name": "P5 Task 3 Test Capability",
                "arch": {"family": "Ada", "sm": "sm89"},
                "ips": {"gpu": {"precisions": ["FP16"]}},
            }
        ),
        encoding="utf-8",
    )
    try:
        yield
    finally:
        TEST_CAPABILITY.unlink(missing_ok=True)
        try:
            TEST_CAPABILITY.parent.rmdir()
        except OSError:
            pass


def _write_rtx_inputs(tmp_path: Path, *, cuda: str) -> tuple[Path, Path, Path]:
    root = tmp_path / "repository"
    root.mkdir()
    runtime = {
        "python": "3.11",
        "cuda": "12.1",
        "driver": "535.104",
        "framework": {"name": "torch", "version": "2.4"},
    }
    contract = root / "environment-contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "schema": "environment_contract_v1",
                "target": "rtx4090",
                "hardware_capability": "configs/hardware/task3/rtx4090.yaml",
                "requires_gpu": True,
                "runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    observation = root / "environment-observation.json"
    observation.write_text(
        json.dumps(
            {
                "schema": "environment_observation_v1",
                "target": "rtx4090",
                "runtime": {**runtime, "cuda": cuda},
                "gpu_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return root, contract, observation


def _run_cli(contract: Path, observation: Path, output: Path, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--contract",
            str(contract),
            "--observation",
            str(observation),
            "--output",
            str(output),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_writes_redacted_failure_report_and_returns_one(tmp_path: Path) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="11.8")
    output = tmp_path / "private-marker" / "report.json"

    result = _run_cli(contract, observation, output, cwd=root)

    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "environment_contract_report_v1",
        "target": "rtx4090",
        "passed": False,
        "failures": [{"code": "runtime.cuda.out_of_range", "field": "runtime.cuda"}],
    }
    assert "private-marker" not in result.stdout + result.stderr


def test_cli_returns_zero_and_writes_passing_report(tmp_path: Path) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="12.1")
    output = tmp_path / "report.json"

    result = _run_cli(contract, observation, output, cwd=root)

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "environment_contract_report_v1",
        "target": "rtx4090",
        "passed": True,
        "failures": [],
    }


def test_cli_reports_nested_framework_failure_fields(tmp_path: Path) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="12.1")
    document = json.loads(observation.read_text(encoding="utf-8"))
    document["runtime"]["framework"] = {"name": "tensorflow", "version": "1.0"}
    observation.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "report.json"

    result = _run_cli(contract, observation, output, cwd=root)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert report == {
        "schema": "environment_contract_report_v1",
        "target": "rtx4090",
        "passed": False,
        "failures": [
            {"code": "runtime.framework.name.mismatch", "field": "runtime.framework.name"},
            {
                "code": "runtime.framework.version.out_of_range",
                "field": "runtime.framework.version",
            },
        ],
    }
    assert "framework_name" not in json.dumps(report)
    assert "framework_version" not in json.dumps(report)


def test_cli_writes_invalid_report_and_returns_two_for_invalid_observation(tmp_path: Path) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="12.1")
    observation.write_text("{", encoding="utf-8")
    output = tmp_path / "report.json"

    result = _run_cli(contract, observation, output, cwd=root)

    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "environment_contract_report_v1",
        "target": None,
        "passed": False,
        "failures": [{"code": "input.invalid", "field": "input"}],
    }


@pytest.mark.parametrize("input_kind", ["python_version", "json_gpu_count", "yaml_integer"])
def test_cli_normalizes_oversized_numeric_inputs_to_redacted_invalid_report(
    tmp_path: Path, input_kind: str
) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="12.1")
    output = tmp_path / "private-marker" / "report.json"
    oversized_number = "9" * 5_000

    if input_kind == "python_version":
        document = json.loads(observation.read_text(encoding="utf-8"))
        document["runtime"]["python"] = oversized_number
        observation.write_text(json.dumps(document), encoding="utf-8")
    elif input_kind == "json_gpu_count":
        document = json.loads(observation.read_text(encoding="utf-8"))
        observation.write_text(
            json.dumps({key: value for key, value in document.items() if key != "gpu_count"})
            .removesuffix("}")
            + ', "gpu_count": '
            + oversized_number
            + "}",
            encoding="utf-8",
        )
    else:
        contract.write_text(
            contract.read_text(encoding="utf-8").replace(
                "requires_gpu: true", f"requires_gpu: {oversized_number}"
            ),
            encoding="utf-8",
        )

    result = _run_cli(contract, observation, output, cwd=root)
    output_text = result.stdout + result.stderr

    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "environment_contract_report_v1",
        "target": None,
        "passed": False,
        "failures": [{"code": "input.invalid", "field": "input"}],
    }
    assert "Traceback" not in output_text
    assert str(tmp_path) not in output_text
    assert str(REPOSITORY_ROOT) not in output_text


def test_cli_rejects_abbreviated_arguments_without_writing_a_report(tmp_path: Path) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="12.1")
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--contra",
            str(contract),
            "--observ",
            str(observation),
            "--out",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert not output.exists()


def test_cli_source_does_not_probe_the_host() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for forbidden in ("os.environ", "subprocess", "nvidia-smi", "torch.cuda"):
        assert forbidden not in source
