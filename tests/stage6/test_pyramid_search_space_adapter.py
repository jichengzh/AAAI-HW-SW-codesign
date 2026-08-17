from __future__ import annotations

from typing import Any

import pytest

from framework.stage6.pyramid_search_space_adapter_v1 import (
    PyramidSearchSpaceAdapterError,
    build_pyramid_structure_plan,
)


def _candidate(stage: str, widths: list[int], q_modes: list[str]) -> dict[str, Any]:
    points = []
    for width in widths:
        for q_mode in q_modes:
            points.append(
                {
                    "id": f"{stage}:w{width}:{q_mode}",
                    "width": width,
                    "quant_policy": q_mode,
                    "buildable": True,
                    "status": "active",
                }
            )
    return {
        "id": stage,
        "search_group_id": stage,
        "bucket": "pyramid_backbone",
        "dense_stage": stage,
        "width_anchors": [{"width": width} for width in widths],
        "software_points": points,
    }


def _space(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": "stage2_search_space_v1",
        "model": "pyramid_lidar",
        "hardware_target": {"name": "h800"},
        "software_candidates": [
            _candidate("stage1", [16, 32], ["fp16", "int8"]),
            _candidate("stage2", [32], ["fp16", "int8"]),
            _candidate("stage3", [64], ["fp16", "int8"]),
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


def test_build_pyramid_structure_plan_maps_complete_stage2_space() -> None:
    plan = build_pyramid_structure_plan(_space())

    assert plan["schema_version"] == "p6_pyramid_structure_plan_v1"
    assert plan["source_schema"] == "stage2_search_space_v1"
    assert plan["target_model"] == "pyramid"
    assert plan["hardware_target"] == "h800"
    assert plan["execution_backend"] == "tvm_auto"
    assert plan["candidate_source_mode"] == "framework_stage2_search_space"
    assert plan["structure_count"] == 4
    assert [item["width"] for item in plan["structures"]] == [
        [16, 32, 64],
        [16, 32, 64],
        [32, 32, 64],
        [32, 32, 64],
    ]
    assert [item["q_mode"] for item in plan["structures"]] == ["fp16", "int8", "fp16", "int8"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"schema": "stage2_search_space_v0"}), "schema"),
        (lambda data: data.update({"model": "codriving"}), "model"),
        (lambda data: data.update({"hardware_target": {"name": "orin"}}), "H800"),
        (lambda data: data.update({"hardware_candidates": []}), "TVM"),
        (lambda data: data["software_candidates"].pop(), "stage"),
        (lambda data: data["software_candidates"][0]["software_points"][0].update({"status": "diagnostic_only"}), "active"),
        (lambda data: data["software_candidates"][0]["software_points"][1].update({"buildable": False}), "buildable"),
        (lambda data: data["software_candidates"][0]["software_points"][0].update({"quant_policy": "fp32"}), "quant"),
    ],
)
def test_build_pyramid_structure_plan_rejects_unmaterializable_spaces(
    mutate: Any, message: str
) -> None:
    payload = _space()
    mutate(payload)

    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_structure_plan(payload)


def test_build_pyramid_structure_plan_does_not_fall_back_to_static_grid() -> None:
    payload = _space(software_candidates=[])

    with pytest.raises(PyramidSearchSpaceAdapterError, match="stage"):
        build_pyramid_structure_plan(payload)
