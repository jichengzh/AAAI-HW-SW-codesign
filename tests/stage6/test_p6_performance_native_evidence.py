"""Evidence-specific gates for the P6 native performance round."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_performance_round_adapter_v1 import (
    P6PerformanceRoundAdapterError,
    run_performance_round,
)
from tests.stage6.p6_performance_native_fixture import quant_contract_path
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
