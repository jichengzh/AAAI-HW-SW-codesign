"""Native quant/result evidence helpers for P6 performance adapter tests."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def write_source_and_quant_evidence(
    round_root: Path,
    request: Mapping[str, Any],
) -> None:
    for row in request["rows"]:
        source = row["source_contract"]
        onnx_path = Path(str(source["onnx_path"]))
        calibration_npz = Path(str(source["calibration_npz"]))
        calibration_summary = Path(str(source["calibration_summary"]))
        for path, data in (
            (onnx_path, b"onnx"),
            (calibration_npz, b"npz"),
            (calibration_summary, b"{}"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if row["q_mode"] == "int8":
            _write_quant_contract(round_root, row, onnx_path, calibration_npz, calibration_summary)


def _write_quant_contract(
    round_root: Path,
    row: Mapping[str, Any],
    onnx_path: Path,
    calibration_npz: Path,
    calibration_summary: Path,
) -> None:
    path = quant_contract_path(round_root, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage3_tvm_int8_quant_contract_v3",
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": sha256_file(onnx_path),
        "calibration_npz": str(calibration_npz.resolve()),
        "calibration_npz_sha256": sha256_file(calibration_npz),
        "calibration_summary": str(calibration_summary.resolve()),
        "calibration_summary_sha256": sha256_file(calibration_summary),
        "params": {"spatial_features": {"scale": 0.5}},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def quant_contract_path(round_root: Path, row: Mapping[str, Any]) -> Path:
    width = "x".join(str(item) for item in row["width"])
    return round_root / "quant_contracts" / width / "tensor_quant_params.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_file_or_sentinel(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "e" * 64


def native_state_row(
    job: Mapping[str, Any] | str,
    job_id: str,
    status: str | None = None,
    *,
    write_result: bool = True,
    result_payload_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    native_job_id, native_status, result_path = _state_identity(job, job_id, status)
    if not isinstance(job, Mapping):
        write_result = False
    if native_status == "success" and write_result:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "success",
            "numerical_finite": True,
            "lat_p50_ms": 1.25,
            "energy_j": 2.5,
            **dict(result_payload_overrides or {}),
        }
        result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    success = native_status == "success"
    return {
        "schema_version": "stage3_execute_performance_plan_v3_state",
        "job_id": native_job_id,
        "attempt": 2 if native_status == "confirmed_failure" else 1,
        "status": native_status,
        "returncode": 0 if success else 1,
        "start_time_unix": 1.0,
        "end_time_unix": 2.0,
        "elapsed_s": 1.0,
        "stdout_path": f"/native/logs/{native_job_id}.stdout.txt",
        "stderr_path": f"/native/logs/{native_job_id}.stderr.txt",
        "result_json": str(result_path) if success else None,
        "result_sha256": sha256_file_or_sentinel(result_path) if success else None,
        "failure_reasons": [] if success else ["returncode=1"],
    }


def native_state_rows(
    job: Mapping[str, Any],
    job_id: str,
    status: str,
    *,
    write_result: bool,
    result_payload_overrides: Mapping[str, Any],
) -> list[dict[str, Any]]:
    terminal = native_state_row(
        job,
        job_id,
        status,
        write_result=write_result,
        result_payload_overrides=result_payload_overrides,
    )
    if status != "confirmed_failure":
        return [terminal]
    failed = {**terminal, "status": "failed"}
    return [{**failed, "attempt": 1}, failed, terminal]


def _state_identity(
    job: Mapping[str, Any] | str,
    job_id: str,
    status: str | None,
) -> tuple[str, str, Path]:
    if isinstance(job, Mapping):
        command = list(job.get("command") or [])
        if str(job.get("runner_key") or "").startswith("tvm_") and "--out-dir" in command:
            output_dir = Path(command[command.index("--out-dir") + 1])
            name = (
                "route_b_int8_auto_decomp_result.json"
                if job.get("runner_key") == "tvm_int8"
                else "route_b_fp16_auto_result.json"
            )
            return job_id, str(status), output_dir / name
        return job_id, str(status), Path(str(job["expected_result_json"]))
    native_job_id = str(job)
    return native_job_id, job_id, Path(f"/native/results/{native_job_id}.json")
