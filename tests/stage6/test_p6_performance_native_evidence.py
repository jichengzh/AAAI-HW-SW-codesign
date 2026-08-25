"""Evidence-specific gates for the P6 native performance round."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_performance_round_adapter_v1 import (
    P6PerformanceRoundAdapterError,
    _validate_terminal_rows,
    run_performance_round,
)
from tests.stage6.p6_performance_native_fixture import (
    native_state_rows,
    quant_contract_path,
    sha256_file,
)
from tests.stage6.test_p6_performance_round_adapter import (
    _PerformanceRunner,
    _native_source_contract,
    _profile,
    _set_runtime_env,
    _write_quantized_task_state,
)
from tests.stage6.test_p6_quantization_round_adapter import (
    _read_json,
    _write_json,
    _write_round_request,
)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "schema",
        "onnx_path",
        "onnx_sha256",
        "calibration_npz",
        "calibration_npz_sha256",
        "calibration_summary",
        "calibration_summary_sha256",
    ),
)
def test_performance_round_rejects_unbound_int8_quant_contract_before_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    contract_path = quant_contract_path(round_root, request["rows"][1])
    if mutation == "missing":
        contract_path.unlink()
    else:
        contract = _read_json(contract_path)
        contract[mutation] = "wrong" if not mutation.endswith("sha256") else "0" * 64
        _write_json(contract_path, contract)
    original_state = _read_json(task_state)
    runner = _PerformanceRunner()

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(profile, task_state, round_root, runner)

    assert runner.calls == []
    assert _read_json(task_state) == original_state


def test_performance_round_does_not_require_quant_contract_for_fp16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16",) * 4)
    task_state = _write_quantized_task_state(round_root, request)

    run_performance_round(profile, task_state, round_root, _PerformanceRunner())

    assert _read_json(task_state)["stage"] == "performance"


@pytest.mark.parametrize(
    "state_overrides",
    (
        {"schema_version": "wrong"},
        {"returncode": 9},
        {"result_json": "/unbound/result.json"},
        {"result_sha256": "0" * 64},
    ),
)
def test_performance_round_rejects_unbound_native_success_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_overrides: Mapping[str, Any],
) -> None:
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
            _PerformanceRunner(first_state_overrides=state_overrides),
        )

    assert _read_json(task_state) == original_state


def test_performance_round_rejects_success_without_native_result_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            _PerformanceRunner(
                state_statuses=("success",) * 4,
                write_result_artifacts=False,
            ),
        )

    assert _read_json(task_state) == original_state


def test_performance_round_rejects_invalid_native_success_result_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(
            profile,
            task_state,
            round_root,
            _PerformanceRunner(result_payload_overrides={"energy_j": None}),
        )


def test_performance_round_rejects_planner_quant_output_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("int8", "fp16", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    row = request["rows"][0]
    contract = _native_source_contract(
        row,
        row["manifest_job_id"],
        quant_root=round_root / "quant_contracts",
    )
    drifted = {
        **contract,
        "tensor_quant_params_json": str(round_root / "wrong-quant.json"),
        "tensor_quant_params_sha256": "0" * 64,
    }

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(
            profile,
            task_state,
            round_root,
            _PerformanceRunner(
                first_manifest_row_overrides={"source_contract": drifted},
                first_job_overrides={"source_contract": drifted},
            ),
        )


@pytest.mark.parametrize(
    ("flag", "relative_result"),
    (
        ("--out", "proxy.json"),
        ("--out-dir", "proxy-dir/route_b_fp16_auto_result.json"),
    ),
)
def test_performance_round_rejects_external_command_result_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    relative_result: str,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)
    result_path = tmp_path / "external" / relative_result
    _write_native_success_result(result_path)
    command_output = result_path if flag == "--out" else result_path.parent

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(
            profile,
            task_state,
            round_root,
            _PerformanceRunner(
                first_job_overrides={"command": ["python", "measure.py", flag, str(command_output)]},
                first_state_overrides={
                    "result_json": str(result_path),
                    "result_sha256": sha256_file(result_path),
                },
            ),
        )


@pytest.mark.parametrize(
    ("first_q_mode", "result_name"),
    (
        ("fp16", "route_b_fp16_auto_result.json"),
        ("int8", "route_b_int8_auto_decomp_result.json"),
    ),
)
def test_performance_round_accepts_native_tvm_out_dir_result_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_q_mode: str,
    result_name: str,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    other_q_mode = "int8" if first_q_mode == "fp16" else "fp16"
    request = _write_round_request(
        round_root,
        (first_q_mode, other_q_mode, "fp16", "int8"),
    )
    task_state = _write_quantized_task_state(round_root, request)
    output_dir = round_root / f"performance/artifacts/batch_01/pyramid/tvm_{first_q_mode}"
    result_path = output_dir / result_name

    run_performance_round(
        profile,
        task_state,
        round_root,
        _PerformanceRunner(
            first_job_overrides={
                "command": [
                    "python",
                    "measure.py",
                    "--out-dir",
                    str(output_dir),
                    *(
                        [
                            "--tensor-quant-params-json",
                            str(quant_contract_path(round_root, request["rows"][0])),
                        ]
                        if first_q_mode == "int8"
                        else []
                    ),
                ]
            },
            first_result_path=result_path,
        ),
    )

    assert _read_json(task_state)["stage"] == "performance"


@pytest.mark.parametrize(
    ("first_q_mode", "result_name"),
    (
        ("fp16", "result.json"),
        ("fp16", "route_b_int8_auto_decomp_result.json"),
        ("int8", "route_b_fp16_auto_result.json"),
    ),
)
def test_performance_round_rejects_tvm_proxy_or_cross_precision_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_q_mode: str,
    result_name: str,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    other_q_mode = "int8" if first_q_mode == "fp16" else "fp16"
    request = _write_round_request(
        round_root,
        (first_q_mode, other_q_mode, "fp16", "int8"),
    )
    task_state = _write_quantized_task_state(round_root, request)
    output_dir = (
        round_root
        / "performance/artifacts"
        / str(request["rows"][0]["manifest_job_id"])
    )
    result_path = output_dir / result_name

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(
            profile,
            task_state,
            round_root,
            _PerformanceRunner(
                first_job_overrides={
                    "command": [
                        "python",
                        "measure.py",
                        "--out-dir",
                        str(output_dir),
                        *(
                            [
                                "--tensor-quant-params-json",
                                str(quant_contract_path(round_root, request["rows"][0])),
                            ]
                            if first_q_mode == "int8"
                            else []
                        ),
                    ]
                },
                first_result_path=result_path,
            ),
        )


def test_performance_round_rejects_confirmed_failure_without_failed_attempt_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime_env(monkeypatch, tmp_path)
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_quantized_task_state(round_root, request)

    with pytest.raises(P6PerformanceRoundAdapterError):
        run_performance_round(
            profile,
            task_state,
            round_root,
            _PerformanceRunner(native_confirmed_history=False),
        )


def _write_native_success_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "status": "success",
            "numerical_finite": True,
            "lat_p50_ms": 1.25,
            "energy_j": 2.5,
        },
    )


@pytest.mark.parametrize("mutation", ("missing_attempt", "wrong_order", "copy_drift"))
def test_confirmed_failure_requires_exact_native_failed_attempt_history(
    mutation: str,
) -> None:
    job = {
        "job_id": "pyramid|16x32x64|tvm_fp16",
        "max_attempts": 2,
        "expected_result_json": "/native/results/result.json",
    }
    rows = native_state_rows(
        job,
        job["job_id"],
        "confirmed_failure",
        write_result=False,
        result_payload_overrides={},
    )
    if mutation == "missing_attempt":
        rows = rows[1:]
    elif mutation == "wrong_order":
        rows = [rows[1], rows[0], rows[2]]
    else:
        rows[2] = {**rows[2], "failure_reasons": ["different-final-failure"]}

    with pytest.raises(P6PerformanceRoundAdapterError):
        _validate_terminal_rows(rows, job)
