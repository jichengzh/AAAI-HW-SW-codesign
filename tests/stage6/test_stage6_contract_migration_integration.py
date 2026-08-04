from __future__ import annotations

from framework.stage6.contracts_v1 import build_stage6_manifest, validate_stage6_manifest
from framework.stage6.formal_plan_v1 import build_formal_plan
from framework.stage6.protocol_smoke_v1 import select_hardware_blind_batch


def _candidate(index: int, backend: str) -> dict[str, object]:
    width = [16 + index, 32 + index, 64 + index]
    q_mode = "int8" if index % 2 else "fp16"
    return {
        "genome": [*width, q_mode],
        "predictions": {"ap70": 0.5 + index / 1000},
        "graph_features": {
            "parameter_elements": 1000 + index,
            "conv_flops": 2000 + index,
        },
        "backend": backend,
        "source_evidence_sha256": str(index).zfill(64),
    }


def test_stage6_planning_contract_stays_hardware_blind_and_in_memory() -> None:
    manifest = build_stage6_manifest()
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]

    plan = build_formal_plan(tvm, trt, expected_pool_size=20)
    selected = select_hardware_blind_batch(plan["ranked_candidates"], batch_size=4)

    assert validate_stage6_manifest(manifest)["passed"]
    assert len(selected) == 4
    assert all("backend" not in row for row in selected)
    assert manifest["launch_allowed"] is False
