from __future__ import annotations

import base64
import hashlib
import json

import pytest

from framework.stage6.p6_capability_probe_worker_v1 import (
    P6CompilerRejection,
    run_probe_family,
)


def _counts(q_mode: str) -> dict[str, int | None]:
    return {
        "int8_propagated_ops": 1 if q_mode == "int8" else None,
        "precision_eligible_ops": 2 if q_mode == "int8" else None,
        "qdq_folded_pairs": 1 if q_mode == "int8" else None,
        "qdq_pairs": 1 if q_mode == "int8" else None,
        "reformat_ops": 0,
        "total_ops": 2,
        "fused_ops": 0,
        "fusible_ops": 1,
    }


def test_probe_family_runs_exact_fp16_int8_partition_and_seals_raw_bytes() -> None:
    calls: list[tuple[str, str]] = []

    def make_model(probe_id: str, q_mode: str) -> bytes:
        calls.append((probe_id, q_mode))
        return f"onnx:{probe_id}:{q_mode}".encode()

    def compile_model(payload: bytes, q_mode: str) -> tuple[dict, bytes]:
        assert hashlib.sha256(payload).hexdigest()
        return _counts(q_mode), b"real compiler IR"

    result = run_probe_family(
        family="neutral",
        probe_ids=("P1", "P2"),
        model_builder=make_model,
        compiler=compile_model,
    )

    assert calls == [
        ("P1", "fp16"),
        ("P1", "int8"),
        ("P2", "fp16"),
        ("P2", "int8"),
    ]
    assert len(result.records) == 4
    assert len(result.artifact_blobs) == 4
    assert all(row["observation_status"] == "observed_build_success" for row in result.records)


def test_probe_family_seals_a_real_compiler_failure_as_observed() -> None:
    def compiler(_payload: bytes, q_mode: str) -> tuple[dict, bytes]:
        if q_mode == "int8":
            raise P6CompilerRejection("compiler rejected probe at /private/location")
        return _counts(q_mode), b"real compiler IR"

    result = run_probe_family(
        family="pruning",
        probe_ids=("aligned_channels",),
        model_builder=lambda probe_id, q_mode: f"{probe_id}:{q_mode}".encode(),
        compiler=compiler,
    )

    failed = result.records[1]
    assert failed["build_success"] is False
    assert failed["observation_status"] == "observed_build_failure"
    assert failed["compiler_ir_sha256"] is None
    assert result.artifact_blobs[1]["compiler_output_kind"] == "error"
    error = json.loads(base64.b64decode(result.artifact_blobs[1]["compiler_output_base64"]))
    assert error["category"] == "tvm_compiler_rejection"
    assert "private" not in json.dumps(error)


def test_probe_family_does_not_relabel_malformed_observation_as_compiler_failure() -> None:
    with pytest.raises(ValueError, match="observation"):
        run_probe_family(
            family="neutral",
            probe_ids=("P1",),
            model_builder=lambda probe_id, q_mode: f"{probe_id}:{q_mode}".encode(),
            compiler=lambda payload, q_mode: ({"total_ops": 1}, b"compiler-ir"),
        )


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, ImportError])
def test_probe_family_fails_closed_on_noncompiler_infrastructure_errors(
    error_type: type[Exception],
) -> None:
    def infrastructure_failure(_payload: bytes, _q_mode: str):
        raise error_type("private infrastructure detail")

    with pytest.raises(error_type, match="private infrastructure detail"):
        run_probe_family(
            family="neutral",
            probe_ids=("P1",),
            model_builder=lambda probe_id, q_mode: f"{probe_id}:{q_mode}".encode(),
            compiler=infrastructure_failure,
        )
