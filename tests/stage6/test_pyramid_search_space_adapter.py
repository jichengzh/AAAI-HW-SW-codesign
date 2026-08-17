from __future__ import annotations

from typing import Any

import pytest
import yaml

from framework.stage1_bridge import load_stage2_search_space
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
    assert plan["structure_count"] == 2
    assert plan["candidate_count"] == 4
    assert [item["width"] for item in plan["structures"]] == [
        [16, 32, 64],
        [32, 32, 64],
    ]
    assert [item["source_point_ids_by_q_mode"] for item in plan["structures"]] == [
        {
            "fp16": ["stage1:w16:fp16", "stage2:w32:fp16", "stage3:w64:fp16"],
            "int8": ["stage1:w16:int8", "stage2:w32:int8", "stage3:w64:int8"],
        },
        {
            "fp16": ["stage1:w32:fp16", "stage2:w32:fp16", "stage3:w64:fp16"],
            "int8": ["stage1:w32:int8", "stage2:w32:int8", "stage3:w64:int8"],
        },
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"schema": "stage2_search_space_v0"}), "schema"),
        (lambda data: data.update({"model": "codriving"}), "model"),
        (lambda data: data.update({"hardware_target": {"name": "orin"}}), "H800"),
        (lambda data: data.update({"hardware_candidates": []}), "TVM"),
        (lambda data: data["software_candidates"].pop(), "stage"),
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


def _write_real_stage2_fixture(tmp_path: Any) -> dict[str, Any]:
    manifest = {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": "h800_tvm_demo"},
        "view_b1_search_groups": [
            {
                "search_group_id": f"pyramid_group.{suffix}",
                "bucket": "pyramid_backbone",
                "widths": widths,
                "round_to": 16,
                "int8_buildable_align": 64,
                "max_rate": 0.75,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": [f"pyramid_group.{suffix}"],
            }
            for suffix, widths in (("s0", [16, 32]), ("s1", [32]), ("s2", [64]))
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group.s0", "pyramid_group.s1", "pyramid_group.s2"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }
    path = tmp_path / "pyramid.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return load_stage2_search_space(path)


def test_adapter_filters_real_stage2_diagnostic_points_before_requiring_both_modes(
    tmp_path: Any,
) -> None:
    search_space = _write_real_stage2_fixture(tmp_path)
    with pytest.raises(PyramidSearchSpaceAdapterError, match="int8"):
        build_pyramid_structure_plan(search_space)


def test_adapter_rejects_stage_without_buildable_int8_provenance() -> None:
    payload = _space()
    for candidate in payload["software_candidates"]:
        candidate["software_points"] = [
            point for point in candidate["software_points"] if point["quant_policy"] == "fp16"
        ]

    with pytest.raises(PyramidSearchSpaceAdapterError, match="int8"):
        build_pyramid_structure_plan(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["software_candidates"][0]["software_points"][0].update(
                {"quant_policy": "fp32"}
            ),
            "quant",
        ),
        (
            lambda data: data["software_candidates"][0]["software_points"][0].update(
                {"buildable": False}
            ),
            "active.*buildable",
        ),
    ],
)
def test_adapter_rejects_invalid_active_software_points(
    mutate: Any, message: str
) -> None:
    payload = _space()
    mutate(payload)

    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_structure_plan(payload)


def test_adapter_rejects_duplicate_point_provenance() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"].append(
        payload["software_candidates"][0]["software_points"][0].copy()
    )

    with pytest.raises(PyramidSearchSpaceAdapterError, match="duplicate"):
        build_pyramid_structure_plan(payload)


def test_adapter_output_is_deterministic_when_input_is_reordered() -> None:
    first = _space()
    second = _space()
    second["software_candidates"] = list(reversed(second["software_candidates"]))
    for candidate in second["software_candidates"]:
        candidate["software_points"] = list(reversed(candidate["software_points"]))
    second["hardware_candidates"] = list(reversed(second["hardware_candidates"]))

    assert build_pyramid_structure_plan(first) == build_pyramid_structure_plan(second)
