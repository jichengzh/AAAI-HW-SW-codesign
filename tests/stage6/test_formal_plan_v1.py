from __future__ import annotations

import pytest

from framework.stage6.formal_plan_v1 import build_formal_plan


def _candidate(index: int, backend: str) -> dict[str, object]:
    q_mode = "int8" if index % 2 else "fp16"
    width = [16 + index, 32 + index, 64 + index]
    profile = f"h800-{backend}-probe-conditioned-v3"
    return {
        "genome": [*width, q_mode],
        "width": width,
        "q_mode": q_mode,
        "group_id": f"pyramid|{'x'.join(map(str, width))}",
        "manifest_job_id": f"pyramid|{'x'.join(map(str, width))}|q={q_mode}|profile={profile}",
        "capability_profile_id": profile,
        "dispatch_key": "tvm_auto" if backend == "tvm" else "trt_engine",
        "source_contract": {"artifact_id": f"artifact-{index}"},
        "source_evidence_sha256": str(index).zfill(64),
        "graph_features": {
            "parameter_elements": 1000 + index,
            "conv_flops": 2000 + index,
        },
        "predictions": {"ap70": 0.5 + index / 1000},
    }


def test_builds_backend_symmetric_frozen_arm_plans() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]

    plan = build_formal_plan(tvm, trt, expected_pool_size=20)

    assert plan["passed"]
    assert plan["effective_candidate_pool_size"] == 20
    assert len(plan["arms"]["compression_only"]["selected_genomes"]) == 16
    assert len(plan["arms"]["compress_then_tune"]["screened_genomes"]) == 12
    assert len(plan["arms"]["compress_then_tune"]["locked_genomes"]) == 4
    assert len(plan["arms"]["tune_then_compress"]["attempt_genomes"]) == 16
    assert plan["backend_plans"]["tvm"]["compression_only"] == plan["backend_plans"]["trt"][
        "compression_only"
    ]


def test_rejects_backend_candidate_pool_drift() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(19)]

    with pytest.raises(ValueError, match="candidate genome sets differ"):
        build_formal_plan(tvm, trt, expected_pool_size=20)


def test_rejects_backend_dependent_ap_surrogate() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    trt[0]["predictions"]["ap70"] += 0.1  # type: ignore[index, operator]

    with pytest.raises(ValueError, match="AP surrogate drift"):
        build_formal_plan(tvm, trt, expected_pool_size=20)


def test_rejects_backend_source_evidence_drift() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    trt[0]["source_evidence_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="source evidence drift"):
        build_formal_plan(tvm, trt, expected_pool_size=20)
