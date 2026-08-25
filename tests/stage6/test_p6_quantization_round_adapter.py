from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

import pytest

from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from framework.stage6.p6_quantization_round_adapter_v1 import (
    P6QuantizationRoundAdapterError,
    run_quantization_round,
)


class _Result:
    returncode = 0


class _FakeLeafRunner:
    def __init__(self, *, write_invalid_contract: bool = False) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self._write_invalid_contract = write_invalid_contract

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        if shell is not False:
            raise AssertionError("quant adapter must use direct argv")
        output_path = Path(argv[argv.index("--output-json") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            {"schema": "private_wrong_schema"}
            if self._write_invalid_contract
            else {"schema": "stage3_tvm_int8_quant_contract_v3", "scales": [1.0]}
        )
        output_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
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
        return _Result()


@pytest.mark.parametrize(
    ("q_modes", "expected_call_indices"),
    [
        (("fp16", "fp16", "fp16", "fp16"), ()),
        (("int8", "int8", "int8", "int8"), (0, 1, 2, 3)),
        (("fp16", "int8", "fp16", "int8"), (1, 3)),
        (("fp16", "int8", "fp16", "fp16"), (1,)),
    ],
)
def test_quantization_round_fans_out_only_int8_rows_in_request_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    q_modes: tuple[str, str, str, str],
    expected_call_indices: tuple[int, ...],
) -> None:
    """Break caught: FP16 rows launch quant leaves or INT8 GPU order drifts."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
    monkeypatch.setenv("P6_HISTORY_RUN_MODE", "bound")
    monkeypatch.setenv("P6_HISTORY_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("P6_HISTORY_TASK_STATE", str(tmp_path / "round/state/task-state.json"))
    monkeypatch.setenv("P6_HISTORY_ROUND_OUTPUT_ROOT", str(tmp_path / "round"))
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, q_modes)
    task_state = _write_task_state(round_root, request)
    original_state = _read_json(task_state)
    runner = _FakeLeafRunner()

    run_quantization_round(profile, task_state, round_root, runner)

    rows = request["rows"]
    leaf = next(item for item in profile.leaves if item.name == "quant_contract")
    expected_gpu = ["2", "5", "7"]
    assert [call["argv"] for call in runner.calls] == [
        (
            str(profile.project_python),
            str(leaf.implementation),
            "--onnx",
            rows[index]["source_contract"]["onnx_path"],
            "--calibration-npz",
            rows[index]["source_contract"]["calibration_npz"],
            "--calibration-summary",
            rows[index]["source_contract"]["calibration_summary"],
            "--output-json",
            str(_quant_output(round_root, rows[index]["width"])),
        )
        for index in expected_call_indices
    ]
    assert [call["cwd"] for call in runner.calls] == [
        leaf.implementation_cwd for _ in expected_call_indices
    ]
    assert [call["env"]["CUDA_VISIBLE_DEVICES"] for call in runner.calls] == [
        expected_gpu[index % len(expected_gpu)]
        for index, _ in enumerate(expected_call_indices)
    ]
    for call in runner.calls:
        assert set(call["env"]) == {
            "CUDA_VISIBLE_DEVICES",
            "P6_HISTORY_RUN_MODE",
            "P6_HISTORY_PRIVATE_ROOT",
            "P6_HISTORY_TASK_STATE",
            "P6_HISTORY_ROUND_OUTPUT_ROOT",
            "PATH",
            "PYTHONPATH",
        }
        assert call["shell"] is False
    assert sorted(
        path.relative_to(round_root).as_posix()
        for path in (round_root / "quant_contracts").glob("*/*")
    ) == [
        _quant_output(round_root, rows[index]["width"]).relative_to(round_root).as_posix()
        for index in expected_call_indices
    ]
    state = _read_json(task_state)
    assert state == {"stage": "quantization", "rows": original_state["rows"]}


def test_quantization_round_keeps_state_initialized_until_native_contracts_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: task state advances even though the native leaf output is wrong."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
    monkeypatch.setenv("P6_HISTORY_RUN_MODE", "bound")
    monkeypatch.setenv("P6_HISTORY_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("P6_HISTORY_TASK_STATE", str(tmp_path / "round/state/task-state.json"))
    monkeypatch.setenv("P6_HISTORY_ROUND_OUTPUT_ROOT", str(tmp_path / "round"))
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("int8", "fp16", "fp16", "fp16"))
    task_state = _write_task_state(round_root, request)
    original_state = _read_json(task_state)

    with pytest.raises(P6QuantizationRoundAdapterError):
        run_quantization_round(
            profile,
            task_state,
            round_root,
            _FakeLeafRunner(write_invalid_contract=True),
        )

    assert _read_json(task_state) == original_state


def _profile(tmp_path: Path) -> ValidatedPostSourceAdapterProfile:
    private_root = tmp_path / "private"
    leaf_cwd = private_root / "leaf-cwd"
    leaf_cwd.mkdir(parents=True)
    implementation = leaf_cwd / "quant_contract.py"
    implementation.write_text("# fake leaf\n", encoding="utf-8")
    implementation.chmod(0o700)
    return ValidatedPostSourceAdapterProfile(
        schema_version="p6_post_source_adapter_profile_v1",
        private_root=private_root,
        project_python=Path(sys.executable),
        adapters=(),
        leaves=(
            PostSourceLeaf(
                name="quant_contract",
                implementation=implementation,
                implementation_cwd=leaf_cwd,
                sha256="0" * 64,
            ),
        ),
    )


def _write_round_request(round_root: Path, q_modes: Sequence[str]) -> dict[str, Any]:
    round_root.mkdir(parents=True)
    widths = ([16, 32, 64], [24, 40, 72], [32, 48, 80], [40, 56, 88])
    if tuple(q_modes) == ("fp16", "int8", "fp16", "fp16"):
        widths = ([16, 32, 64], [16, 32, 64], [32, 48, 80], [40, 56, 88])
    rows = [
        _row(round_root, index=index, width=widths[index], q_mode=q_modes[index])
        for index in range(4)
    ]
    request: dict[str, Any] = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "task-p6",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {},
        "rows": rows,
    }
    request["row_sha256"] = {row["row_id"]: _canonical_sha(row) for row in rows}
    request["measurement_request_sha256"] = _canonical_sha(request)
    _write_json(round_root / "measurement-request.json", request)
    return request


def _row(round_root: Path, *, index: int, width: Sequence[int], q_mode: str) -> dict[str, Any]:
    width_label = "x".join(str(value) for value in width)
    row_id = f"pyramid|{width_label}|q={q_mode}|row={index}"
    source_root = round_root / "materialized" / width_label
    contract = {
        "onnx_path": str(source_root / "onnx" / "model.onnx"),
        "calibration_npz": str(source_root / "calibration" / "cache.npz"),
        "calibration_summary": str(source_root / "calibration" / "summary.json"),
    }
    return {
        "schema_version": "stage5_candidate_row_v2",
        "task_id": "task-p6",
        "task_sha256": "a" * 64,
        "row_id": row_id,
        "manifest_job_id": row_id,
        "group_id": f"pyramid|{width_label}",
        "model": "pyramid",
        "width": list(width),
        "width_schema": "pyramid_stage_widths_v1",
        "structure_widths": {
            "stage1_width": width[0],
            "stage2_width": width[1],
            "stage3_width": width[2],
        },
        "genome": [*width, q_mode],
        "strategy_id": f"q={q_mode}",
        "q_mode": q_mode,
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm-auto",
        "capability_digest": "b" * 64,
        "dispatch_key": "tvm_auto",
        "source_status": "ready",
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "source_contract": contract,
        "source_contract_sha256": _canonical_sha(contract),
        "source_evidence_sha256": "c" * 64,
        "graph_features": {
            "group_id": f"pyramid|{width_label}",
            "model": "pyramid",
            "width": list(width),
        },
    }


def _write_task_state(round_root: Path, request: Mapping[str, Any]) -> Path:
    path = round_root / "state" / "task-state.json"
    path.parent.mkdir(parents=True)
    _write_json(
        path,
        {
            "stage": "initialized",
            "rows": [
                {
                    "row_id": row["row_id"],
                    "row_sha256": request["row_sha256"][row["row_id"]],
                    "source_evidence_sha256": row["source_evidence_sha256"],
                    "terminal_status": "pending",
                }
                for row in request["rows"]
            ],
        },
    )
    return path


def _quant_output(round_root: Path, width: Sequence[int]) -> Path:
    return (
        round_root
        / "quant_contracts"
        / "x".join(str(value) for value in width)
        / "tensor_quant_params.json"
    )


def _canonical_sha(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
