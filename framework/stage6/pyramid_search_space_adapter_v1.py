from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, unquote_to_bytes

from framework.stage2.formal_software_space_v1 import build_formal_software_plan
from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    default_hardware_execution_profile,
    validate_profile_backend,
)


class PyramidSearchSpaceAdapterError(ValueError):
    """Raised when Stage2 search-space output cannot become Pyramid P6 candidates."""


_Q_MODES = ("fp16", "int8")
_STAGES = ("stage1", "stage2", "stage3")
_SCANNER_STAGE_ALIASES = {
    "stage0": "stage1",
    "stage1": "stage2",
    "stage2": "stage3",
}
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


def _build_formal_candidate_plan(
    search_space: Mapping[str, Any], hardware_target_id: str
) -> dict[str, Any]:
    try:
        formal_plan = build_formal_software_plan(search_space)
    except ValueError as error:
        raise PyramidSearchSpaceAdapterError(str(error)) from error
    stage_axis_ids = _stage_axis_ids(formal_plan["axis_schema"])
    candidates = [
        _p6_candidate(row, stage_axis_ids)
        for row in formal_plan["candidates"]
    ]
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
        "hardware_target": hardware_target_id,
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": formal_plan["structure_count"],
        "candidate_count": len(candidates),
        "axis_schema": formal_plan["axis_schema"],
        "q_modes": formal_plan["q_modes"],
        "source_provenance": formal_plan["source_provenance"],
        "formal_plan_schema": formal_plan["schema"],
        "candidates": candidates,
    }


def _stage_axis_ids(axis_schema: Mapping[str, Any]) -> dict[str, str]:
    axes = axis_schema.get("free_axes")
    if not isinstance(axes, list):
        _fail("formal axis_schema.free_axes is required")
    stage_aliases = _stage_aliases(axes)
    by_stage: dict[str, str] = {}
    for axis in axes:
        if not isinstance(axis, Mapping):
            _fail("formal axis_schema.free_axes entries must be mappings")
        stage = axis.get("dense_stage")
        axis_id = axis.get("axis_id")
        if stage not in stage_aliases or not isinstance(axis_id, str) or not axis_id:
            _fail("formal Pyramid axes must map to stage1, stage2, and stage3")
        stage = stage_aliases[stage]
        if stage in by_stage:
            _fail("duplicate formal Pyramid stage axis")
        by_stage[stage] = axis_id
    if set(by_stage) != set(_STAGES):
        _fail("formal Pyramid axes must map to stage1, stage2, and stage3")
    return by_stage


def _stage_aliases(axes: list[Any]) -> Mapping[str, str]:
    stages = {
        axis.get("dense_stage")
        for axis in axes
        if isinstance(axis, Mapping)
    }
    if stages == set(_SCANNER_STAGE_ALIASES):
        return _SCANNER_STAGE_ALIASES
    return {stage: stage for stage in _STAGES}


def _p6_candidate(
    formal_candidate: Mapping[str, Any], stage_axis_ids: Mapping[str, str]
) -> dict[str, Any]:
    axis_values = formal_candidate.get("axis_values")
    if not isinstance(axis_values, Mapping):
        _fail("formal candidate axis values are required")
    width = [axis_values[stage_axis_ids[stage]] for stage in _STAGES]
    source_point_ids = [
        f"{stage_axis_ids[stage]}:w{axis_values[stage_axis_ids[stage]]}"
        for stage in _STAGES
    ]
    return {
        "width": width,
        "q_mode": formal_candidate["q_mode"],
        "source_point_ids": source_point_ids,
        "formal_candidate_id": formal_candidate["candidate_id"],
        "formal_structure_index": formal_candidate["structure_index"],
        "formal_candidate_index": formal_candidate["candidate_index"],
        "axis_order": list(formal_candidate["axis_order"]),
        "axis_values": dict(axis_values),
        "formal_identity": formal_candidate["identity"],
    }


def build_pyramid_candidate_plan(
    search_space: Mapping[str, Any],
    *,
    profile: HardwareExecutionProfile | None = None,
) -> dict[str, Any]:
    """Convert a Stage2 Pyramid search space into executable candidates."""
    selected_profile = profile or default_hardware_execution_profile()
    try:
        validate_profile_backend(selected_profile, "tvm_auto")
    except ValueError as error:
        raise PyramidSearchSpaceAdapterError(str(error)) from error
    if not isinstance(search_space, Mapping):
        _fail("search space must be a mapping")
    if search_space.get("schema") != "stage2_search_space_v1":
        _fail("unsupported schema; expected stage2_search_space_v1")
    if search_space.get("model") != "pyramid_lidar":
        _fail("unsupported model; expected pyramid_lidar")

    hardware_target = search_space.get("hardware_target")
    if not isinstance(hardware_target, Mapping):
        _fail("hardware target is required")
    hardware_name = hardware_target.get("name")
    if (
        _normalize_hardware_name(hardware_name)
        not in selected_profile.allowed_normalized_gpu_models
    ):
        _fail("hardware target does not match the hardware profile")

    hardware_candidates = search_space.get("hardware_candidates")
    if not isinstance(hardware_candidates, list):
        _fail("TVM hardware candidate is required")
    required_hardware = {
        "id": "tvm_metaschedule_candidate",
        "backend_scope": f"measured_{selected_profile.target_hardware_id}_tvm",
        "hardware": hardware_name,
        "schedule_policy": "tuned",
    }
    if not any(
        isinstance(candidate, Mapping)
        and all(candidate.get(key) == value for key, value in required_hardware.items())
        for candidate in hardware_candidates
    ):
        _fail("TVM hardware candidate is required")

    return _build_formal_candidate_plan(
        search_space, selected_profile.target_hardware_id
    )


def _normalize_hardware_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.upper() if character.isalnum())


def build_pyramid_structure_plan(
    search_space: Mapping[str, Any],
    *,
    profile: HardwareExecutionProfile | None = None,
) -> dict[str, Any]:
    """Compatibility alias for callers using the P6.1 converter name."""
    return build_pyramid_candidate_plan(search_space, profile=profile)
