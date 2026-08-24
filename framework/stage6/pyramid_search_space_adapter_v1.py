from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any
from urllib.parse import quote, unquote_to_bytes


class PyramidSearchSpaceAdapterError(ValueError):
    """Raised when Stage2 search-space output cannot become Pyramid P6 candidates."""


_Q_MODES = ("fp16", "int8")
_STAGES = ("stage1", "stage2", "stage3")
_STAGE_PROVENANCE_TOKEN_PREFIX = "p6-stage-provenance-v1:"


def _fail(message: str) -> None:
    raise PyramidSearchSpaceAdapterError(message)


def parse_pyramid_stage_provenance_token(token: str) -> tuple[str, ...]:
    """Parse one canonical multi-group stage provenance token."""
    if not isinstance(token, str) or not token.startswith(
        _STAGE_PROVENANCE_TOKEN_PREFIX
    ):
        _fail("invalid Pyramid stage provenance token")
    encoded_components = token.removeprefix(_STAGE_PROVENANCE_TOKEN_PREFIX).split(",")
    if len(encoded_components) < 2 or any(not item for item in encoded_components):
        _fail("invalid Pyramid stage provenance token")

    components: list[str] = []
    for encoded in encoded_components:
        try:
            component = unquote_to_bytes(encoded).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("invalid Pyramid stage provenance token")
        if (
            not component
            or component.startswith(_STAGE_PROVENANCE_TOKEN_PREFIX)
            or quote(component, safe="") != encoded
        ):
            _fail("invalid Pyramid stage provenance token")
        components.append(component)
    if components != sorted(set(components)):
        _fail("invalid Pyramid stage provenance token")
    return tuple(components)


def _stage_provenance(source_point_ids: list[str]) -> str:
    canonical_ids = sorted(set(source_point_ids))
    if len(canonical_ids) == 1:
        return canonical_ids[0]
    token = _STAGE_PROVENANCE_TOKEN_PREFIX + ",".join(
        quote(point_id, safe="") for point_id in canonical_ids
    )
    parse_pyramid_stage_provenance_token(token)
    return token


def _active_group_shapes(
    stage: str, group: Mapping[str, Any]
) -> dict[tuple[str, int], str]:
    points = group.get("software_points")
    if not isinstance(points, list) or not points:
        _fail(f"{stage} software points are required")
    shapes: dict[tuple[str, int], str] = {}
    for point in points:
        if not isinstance(point, Mapping):
            _fail(f"{stage} software point must be a mapping")
        q_mode = point.get("quant_policy")
        point_id = point.get("id")
        if q_mode not in _Q_MODES:
            _fail("unsupported quantization policy")
        status = point.get("status")
        buildable = point.get("buildable")
        if status == "active" and buildable is not True:
            _fail("active software point must be buildable")
        if status in {"diagnostic", "diagnostic_only"} and buildable is False:
            continue
        if status != "active" or buildable is not True:
            _fail("unsupported status/buildable combination")
        width = point.get("width")
        if not isinstance(point_id, str) or not point_id:
            _fail("software point id is required")
        if point_id.startswith(_STAGE_PROVENANCE_TOKEN_PREFIX):
            _fail("software point id uses reserved provenance token prefix")
        if isinstance(width, bool) or not isinstance(width, int):
            _fail("software point width is required")
        shape = (q_mode, width)
        if shape in shapes:
            _fail("duplicate group q_mode and width")
        shapes[shape] = point_id
    return shapes


def _common_stage_points(
    group_shapes: list[dict[tuple[str, int], str]],
) -> dict[str, list[dict[str, Any]]]:
    by_q_mode = {q_mode: [] for q_mode in _Q_MODES}
    for q_mode in _Q_MODES:
        common_widths = set.intersection(
            *(
                {width for mode, width in shapes if mode == q_mode}
                for shapes in group_shapes
            )
        )
        for width in sorted(common_widths):
            source_point_ids = [shapes[(q_mode, width)] for shapes in group_shapes]
            by_q_mode[q_mode].append(
                {"id": _stage_provenance(source_point_ids), "width": width}
            )
    return by_q_mode


def _formal_q_modes(search_space: Mapping[str, Any]) -> tuple[str, ...]:
    raw_q_modes = search_space.get("formal_q_modes")
    if not isinstance(raw_q_modes, list) or not raw_q_modes:
        _fail("formal q modes are required")
    q_modes = tuple(str(q_mode) for q_mode in raw_q_modes)
    if (
        tuple(sorted(q_modes)) != q_modes
        or len(set(q_modes)) != len(q_modes)
        or not set(q_modes) <= set(_Q_MODES)
    ):
        _fail("formal q modes must be sorted unique fp16/int8 modes")
    return q_modes


def _formal_axis_points(axis: Mapping[str, Any]) -> list[dict[str, Any]]:
    axis_id = axis.get("axis_id")
    widths = axis.get("legal_widths")
    base_width = axis.get("base_width")
    if (
        not isinstance(axis_id, str)
        or not axis_id.strip()
        or not isinstance(widths, list)
        or not widths
        or isinstance(base_width, bool)
        or not isinstance(base_width, int)
        or base_width <= 0
    ):
        _fail("formal structural axis is invalid")
    if any(isinstance(width, bool) or not isinstance(width, int) for width in widths):
        _fail("formal structural axis width is invalid")
    if widths != sorted(set(widths)) or widths[-1] > base_width:
        _fail("formal structural axis widths are not canonical")
    return [
        {
            "id": f"{axis_id}:w{width}",
            "width": width,
        }
        for width in widths
    ]


def _formal_points_by_stage(
    search_space: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]] | None:
    raw_axes = search_space.get("structural_axes")
    if raw_axes is None:
        return None
    if not isinstance(raw_axes, list) or not raw_axes:
        _fail("formal structural axes are required")
    points_by_stage: dict[str, list[dict[str, Any]]] = {}
    for axis in raw_axes:
        if not isinstance(axis, Mapping):
            _fail("formal structural axis must be a mapping")
        stage = axis.get("dense_stage")
        if stage not in _STAGES:
            _fail("formal structural axes for stage1, stage2, and stage3 are required")
        if stage in points_by_stage:
            _fail("duplicate formal structural axis stage")
        points_by_stage[stage] = _formal_axis_points(axis)
    if set(points_by_stage) != set(_STAGES):
        _fail("formal structural axes for stage1, stage2, and stage3 are required")
    return {stage: points_by_stage[stage] for stage in _STAGES}


def _build_formal_candidate_plan(
    search_space: Mapping[str, Any], hardware_name: str
) -> dict[str, Any] | None:
    points_by_stage = _formal_points_by_stage(search_space)
    if points_by_stage is None:
        return None
    candidates: list[dict[str, Any]] = []
    for q_mode in _formal_q_modes(search_space):
        for selected in product(*(points_by_stage[stage] for stage in _STAGES)):
            candidates.append(
                {
                    "width": [item["width"] for item in selected],
                    "q_mode": q_mode,
                    "source_point_ids": [item["id"] for item in selected],
                }
            )
    candidates.sort(
        key=lambda row: (
            tuple(row["width"]),
            row["q_mode"],
            tuple(row["source_point_ids"]),
        )
    )
    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": hardware_name,
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": len({tuple(row["width"]) for row in candidates}),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_pyramid_candidate_plan(search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Stage2 Pyramid search space into executable candidates."""
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

    formal_plan = _build_formal_candidate_plan(search_space, hardware_name)
    if formal_plan is not None:
        return formal_plan

    software_candidates = search_space.get("software_candidates")
    if not isinstance(software_candidates, list):
        _fail("stage candidates are required")
    groups_by_stage: dict[str, list[Mapping[str, Any]]] = {
        stage: [] for stage in _STAGES
    }
    for candidate in software_candidates:
        if not isinstance(candidate, Mapping):
            _fail("stage candidate must be a mapping")
        stage = candidate.get("dense_stage")
        if stage not in _STAGES:
            _fail("stage candidates for stage1, stage2, and stage3 are required")
        groups_by_stage[stage].append(candidate)
    if any(not groups_by_stage[stage] for stage in _STAGES):
        _fail("stage candidates for stage1, stage2, and stage3 are required")

    points_by_stage: dict[str, dict[str, list[dict[str, Any]]]] = {}
    shape_by_source_id: dict[str, tuple[str, int]] = {}
    for stage in _STAGES:
        group_shapes = [
            _active_group_shapes(stage, group) for group in groups_by_stage[stage]
        ]
        for shapes in group_shapes:
            for shape, point_id in shapes.items():
                previous_shape = shape_by_source_id.setdefault(point_id, shape)
                if previous_shape != shape:
                    _fail("software point id maps to different shapes")
        points_by_stage[stage] = _common_stage_points(group_shapes)

    candidates: list[dict[str, Any]] = []
    for q_mode in _Q_MODES:
        if any(not points_by_stage[stage][q_mode] for stage in _STAGES):
            continue
        for selected in product(
            *(points_by_stage[stage][q_mode] for stage in _STAGES)
        ):
            candidates.append(
                {
                    "width": [item["width"] for item in selected],
                    "q_mode": q_mode,
                    "source_point_ids": [item["id"] for item in selected],
                }
            )
    if not candidates:
        _fail("no q-mode has complete provenance across all stages")

    candidates.sort(
        key=lambda row: (
            tuple(row["width"]),
            row["q_mode"],
            tuple(row["source_point_ids"]),
        )
    )
    identities = {(tuple(row["width"]), row["q_mode"]) for row in candidates}
    if len(identities) != len(candidates):
        _fail("duplicate candidate width and q_mode identity")
    structure_count = len({tuple(row["width"]) for row in candidates})

    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": hardware_name,
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": structure_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_pyramid_structure_plan(search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility alias for callers using the P6.1 converter name."""
    return build_pyramid_candidate_plan(search_space)
