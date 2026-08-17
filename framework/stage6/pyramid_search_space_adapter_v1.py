from __future__ import annotations

import copy
from collections.abc import Mapping
from itertools import product
from typing import Any


class PyramidSearchSpaceAdapterError(ValueError):
    """Raised when Stage2 search-space output cannot become Pyramid P6 candidates."""


_Q_MODES = ("fp16", "int8")
_STAGES = ("stage1", "stage2", "stage3")


def _fail(message: str) -> None:
    raise PyramidSearchSpaceAdapterError(message)


def build_pyramid_structure_plan(search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Stage2 Pyramid search space into executable structure candidates."""
    if not isinstance(search_space, Mapping):
        _fail("search space must be a mapping")
    if search_space.get("schema") != "stage2_search_space_v1":
        _fail("unsupported schema; expected stage2_search_space_v1")
    if search_space.get("model") != "pyramid_lidar":
        _fail("unsupported model; expected pyramid_lidar")

    hardware_target = search_space.get("hardware_target")
    if not isinstance(hardware_target, Mapping):
        _fail("H800 hardware target is required")
    hardware_name = hardware_target.get("name")
    if not isinstance(hardware_name, str) or not (
        hardware_name == "h800" or hardware_name.startswith("h800_")
    ):
        _fail("H800 hardware target is required")

    hardware_candidates = search_space.get("hardware_candidates")
    if not isinstance(hardware_candidates, list):
        _fail("TVM hardware candidate is required")
    required_hardware = {
        "id": "tvm_metaschedule_candidate",
        "backend_scope": "measured_h800_tvm",
        "hardware": hardware_name,
        "schedule_policy": "tuned",
    }
    if not any(
        isinstance(candidate, Mapping)
        and all(candidate.get(key) == value for key, value in required_hardware.items())
        for candidate in hardware_candidates
    ):
        _fail("TVM hardware candidate is required")

    software_candidates = search_space.get("software_candidates")
    if not isinstance(software_candidates, list):
        _fail("stage candidates are required")
    by_stage: dict[str, Mapping[str, Any]] = {}
    for candidate in software_candidates:
        if not isinstance(candidate, Mapping):
            _fail("stage candidate must be a mapping")
        stage = candidate.get("dense_stage")
        if stage not in _STAGES or stage in by_stage:
            _fail("exactly one candidate for each stage1/stage2/stage3 is required")
        by_stage[stage] = candidate
    if set(by_stage) != set(_STAGES):
        _fail("stage candidates for stage1, stage2, and stage3 are required")

    points_by_stage: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen_point_ids: set[str] = set()
    for stage in _STAGES:
        points = by_stage[stage].get("software_points")
        if not isinstance(points, list) or not points:
            _fail(f"{stage} software points are required")
        by_q_mode = {q_mode: [] for q_mode in _Q_MODES}
        seen_stage_shapes: set[tuple[str, int]] = set()
        for point in points:
            if not isinstance(point, Mapping):
                _fail(f"{stage} software point must be a mapping")
            q_mode = point.get("quant_policy")
            point_id = point.get("id")
            if isinstance(point_id, str) and point_id:
                if point_id in seen_point_ids:
                    _fail("duplicate software point id")
                seen_point_ids.add(point_id)
            if q_mode not in _Q_MODES or point.get("buildable") is not True or point.get("status") != "active":
                continue
            width = point.get("width")
            if not isinstance(point_id, str) or not point_id:
                _fail("software point id is required")
            if isinstance(width, bool) or not isinstance(width, int):
                _fail("software point width is required")
            shape = (q_mode, width)
            if shape in seen_stage_shapes:
                _fail("duplicate stage q_mode and width")
            seen_stage_shapes.add(shape)
            by_q_mode[q_mode].append({"id": point_id, "width": width})
        points_by_stage[stage] = by_q_mode

    common_q_modes = set(_Q_MODES)
    for stage in _STAGES:
        common_q_modes &= {
            q_mode for q_mode, points in points_by_stage[stage].items() if points
        }
    if not common_q_modes:
        _fail("no common quantization mode can form a complete structure")

    structures: list[dict[str, Any]] = []
    for q_mode in common_q_modes:
        stage_points = [points_by_stage[stage][q_mode] for stage in _STAGES]
        for selected in product(*stage_points):
            structures.append(
                {
                    "width": [item["width"] for item in selected],
                    "q_mode": q_mode,
                    "source_point_ids": copy.deepcopy([item["id"] for item in selected]),
                }
            )
    structures.sort(key=lambda item: (tuple(item["width"]), item["q_mode"]))

    return {
        "schema_version": "p6_pyramid_structure_plan_v1",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": hardware_name,
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": len(structures),
        "structures": structures,
    }
