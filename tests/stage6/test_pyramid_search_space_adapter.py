from __future__ import annotations

import json
from itertools import product
from typing import Any

import pytest

from framework.stage6.pyramid_search_space_adapter_v1 import (
    PyramidSearchSpaceAdapterError,
    build_pyramid_candidate_plan,
)


def _candidate(
    stage: str,
    widths: list[int],
    q_modes: list[str],
    widths_by_q_mode: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    return {
        "id": stage,
        "dense_stage": stage,
        "software_points": [
            {
                "id": f"{stage}-{q_mode}-{width}",
                "width": width,
                "quant_policy": q_mode,
                "buildable": True,
                "status": "active",
            }
            for q_mode in q_modes
            for width in (widths_by_q_mode or {}).get(q_mode, widths)
        ],
    }


def _space(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": "stage2_search_space_v1",
        "model": "pyramid_lidar",
        "hardware_target": {"name": "h800"},
        "software_candidates": [
            _candidate("stage1", [16, 32], ["fp16", "int8"], {"int8": [16]}),
            _candidate("stage2", [32, 64], ["fp16", "int8"], {"int8": [32]}),
            _candidate("stage3", [64, 96], ["fp16", "int8"], {"int8": [64]}),
        ],
        "hardware_candidates": [
            {
                "id": "tvm_metaschedule_candidate",
                "backend_scope": "measured_h800_tvm",
                "hardware": "h800",
                "schedule_policy": "tuned",
            }
        ],
    }
    return {**payload, **overrides}


def test_build_pyramid_candidate_plan_uses_independent_q_mode_products() -> None:
    plan = build_pyramid_candidate_plan(_space())

    assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
    assert plan["source_schema"] == "stage2_search_space_v1"
    assert plan["target_model"] == "pyramid"
    assert plan["hardware_target"] == "h800"
    assert plan["execution_backend"] == "tvm_auto"
    assert plan["candidate_source_mode"] == "framework_stage2_search_space"
    assert plan["structure_count"] == 8
    assert plan["candidate_count"] == 9
    assert {(tuple(row["width"]), row["q_mode"]) for row in plan["candidates"]} == {
        *((width, "fp16") for width in product((16, 32), (32, 64), (64, 96))),
        ((16, 32, 64), "int8"),
    }
    assert all(row["q_mode"] in {"fp16", "int8"} for row in plan["candidates"])


def test_build_pyramid_candidate_plan_allows_missing_int8_counterparts() -> None:
    space = _space()
    for stage in space["software_candidates"]:
        stage["software_points"] = [
            point
            for point in stage["software_points"]
            if point["quant_policy"] == "fp16"
        ]
    plan = build_pyramid_candidate_plan(space)
    assert plan["candidate_count"] == 8
    assert all(row["q_mode"] == "fp16" for row in plan["candidates"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"schema": "bad"}), "schema"),
        (lambda data: data.update({"model": "codriving"}), "model"),
        (lambda data: data.update({"hardware_target": {"name": "orin"}}), "H800"),
        (lambda data: data.update({"hardware_candidates": []}), "TVM"),
        (
            lambda data: data["hardware_candidates"][0].update({"backend_scope": "cuda"}),
            "TVM",
        ),
        (lambda data: data.update({"software_candidates": data["software_candidates"][:-1]}), "stage"),
    ],
)
def test_build_pyramid_candidate_plan_rejects_unsupported_inputs(
    mutate: Any, message: str
) -> None:
    payload = _space()
    mutate(payload)
    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_unknown_quant_policy() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0]["quant_policy"] = "fp32"
    with pytest.raises(PyramidSearchSpaceAdapterError, match="quant"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_active_non_buildable_point() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0]["buildable"] = False
    with pytest.raises(PyramidSearchSpaceAdapterError, match="active.*buildable"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_non_active_buildable_point() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0].update(
        {"status": "diagnostic", "buildable": True}
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="non-active"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_ignores_non_active_non_buildable_diagnostics() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"].append(
        {
            "id": "diagnostic",
            "width": 128,
            "quant_policy": "int8",
            "buildable": False,
            "status": "diagnostic",
        }
    )
    assert build_pyramid_candidate_plan(payload)["candidate_count"] == 9


def test_build_pyramid_candidate_plan_rejects_duplicate_point_identity() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"].append(
        payload["software_candidates"][0]["software_points"][0].copy()
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="duplicate"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_no_complete_q_mode_product() -> None:
    payload = _space()
    payload["software_candidates"][1]["software_points"] = [{
        "id": "diagnostic-stage2",
        "width": 128,
        "quant_policy": "fp16",
        "buildable": False,
        "status": "diagnostic",
    }]
    with pytest.raises(PyramidSearchSpaceAdapterError, match="q-mode"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_is_canonical_when_inputs_reverse() -> None:
    first = _space()
    second = _space()
    second["software_candidates"] = list(reversed(second["software_candidates"]))
    for candidate in second["software_candidates"]:
        candidate["software_points"] = list(reversed(candidate["software_points"]))
    second["hardware_candidates"] = list(reversed(second["hardware_candidates"]))
    assert json.dumps(build_pyramid_candidate_plan(first), sort_keys=True) == json.dumps(
        build_pyramid_candidate_plan(second), sort_keys=True
    )
