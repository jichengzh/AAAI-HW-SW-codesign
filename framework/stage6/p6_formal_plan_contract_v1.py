"""Validation helpers for P6 candidate plans derived from formal axes."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from framework.stage5.genome_contract_v1 import (
    canonical_group_id,
    validate_structure_identity,
)
from framework.stage5.production_search_v1 import (
    source_group_q_modes,
    validate_source_contract,
)
from framework.stage6.p6_history_recipe_profiles_v1 import (
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
)
from framework.stage6.p6_history_training_contract_v1 import (
    P6HistoryTrainingContractError,
    validate_recipe_v2_group_training_contract,
)


ALLOWED_Q_MODES = frozenset({"fp16", "int8"})
PLAN_SCHEMA_VERSION = "p6_pyramid_candidate_plan_v2"
REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v2"
RECIPE_V1 = "p6_history_dynamic_materialization_recipe_v1"
STAGE_WIDTH_FIELDS = ("stage1_width", "stage2_width", "stage3_width")
OUTPUT_TEMPLATE_KEYS = (
    "training_path_template",
    "checkpoint_path_template",
    "onnx_path_template",
    "calibration_path_template",
)
LEGACY_OUTPUT_KEYS = tuple(
    key.removesuffix("_template") for key in OUTPUT_TEMPLATE_KEYS
)
EXPECTED_PLAN_FIELDS = {
    "schema_version": PLAN_SCHEMA_VERSION,
    "source_schema": "stage2_search_space_v1",
    "target_model": "pyramid",
    "execution_backend": "tvm_auto",
    "candidate_source_mode": "framework_stage2_search_space",
}
STAGES = ("stage1", "stage2", "stage3")


def validate_p6_candidate_plan(
    raw_plan: Mapping[str, Any],
) -> dict[tuple[tuple[int, int, int], str], tuple[str, str, str]]:
    """Validate a P6 candidate plan and return its canonical identity map."""
    _validate_plan_envelope(raw_plan)
    stage_axes = _formal_stage_axes(raw_plan)
    entries = [
        _candidate_entry(candidate, stage_axes)
        for candidate in raw_plan["candidates"]
    ]
    mapping = dict(entries)
    if len(mapping) != len(entries):
        _invalid("candidate plan identity is duplicated")
    if entries != sorted(entries, key=lambda item: (item[0][0], item[0][1], item[1])):
        _invalid("candidate plan is not canonical")
    _validate_plan_counts(raw_plan, entries, mapping)
    return mapping


def formal_base_widths(raw_plan: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Return stage-ordered formal base widths, or None for legacy non-formal plans."""
    stage_axes = _formal_stage_axes(raw_plan)
    if stage_axes is None:
        return None
    return tuple(base for _axis_id, base in stage_axes)  # type: ignore[return-value]


def _validate_plan_envelope(raw_plan: Mapping[str, Any]) -> None:
    if not isinstance(raw_plan, Mapping):
        _invalid("candidate plan must be an object")
    if any(raw_plan.get(key) != value for key, value in EXPECTED_PLAN_FIELDS.items()):
        _invalid("candidate plan contract is incompatible")
    hardware_target = raw_plan.get("hardware_target")
    if not isinstance(hardware_target, str) or not (
        hardware_target == "h800" or hardware_target.startswith("h800_")
    ):
        _invalid("candidate plan hardware is incompatible")
    candidates = raw_plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        _invalid("candidate plan candidates are invalid")


def _formal_stage_axes(raw_plan: Mapping[str, Any]) -> tuple[tuple[str, int], ...] | None:
    axis_schema = raw_plan.get("axis_schema")
    if axis_schema is None:
        return None
    if not isinstance(axis_schema, Mapping):
        _invalid("formal axis schema is invalid")
    axes = axis_schema.get("free_axes")
    if not isinstance(axes, list):
        _invalid("formal axis schema is invalid")
    by_stage: dict[str, tuple[str, int]] = {}
    for axis in axes:
        stage, axis_id, base_width = _formal_axis_summary(axis)
        if stage in by_stage:
            _invalid("formal axis schema is invalid")
        by_stage[stage] = (axis_id, base_width)
    if set(by_stage) != set(STAGES):
        _invalid("formal axis schema is invalid")
    return tuple(by_stage[stage] for stage in STAGES)


def _formal_axis_summary(axis: object) -> tuple[str, str, int]:
    if not isinstance(axis, Mapping):
        _invalid("formal axis schema is invalid")
    stage = axis.get("dense_stage")
    axis_id = axis.get("axis_id")
    base_width = axis.get("base_width")
    if (
        stage not in STAGES
        or not isinstance(axis_id, str)
        or not axis_id.strip()
        or isinstance(base_width, bool)
        or not isinstance(base_width, int)
        or base_width <= 0
    ):
        _invalid("formal axis schema is invalid")
    return str(stage), axis_id, base_width


def _candidate_entry(
    candidate: object,
    stage_axes: tuple[tuple[str, int], ...] | None,
) -> tuple[tuple[tuple[int, int, int], str], tuple[str, str, str]]:
    width, q_mode, source_point_ids = _candidate_identity(candidate)
    if stage_axes is not None:
        _validate_formal_candidate_sources(width, source_point_ids, stage_axes)
        if any(value > cap for value, (_axis_id, cap) in zip(width, stage_axes, strict=True)):
            _invalid("formal candidate width over base")
    return ((width, q_mode), source_point_ids)


def _candidate_identity(candidate: object) -> tuple[tuple[int, int, int], str, tuple[str, str, str]]:
    if not isinstance(candidate, Mapping):
        _invalid("candidate plan candidate is invalid")
    width = candidate.get("width")
    q_mode = candidate.get("q_mode")
    source_point_ids = candidate.get("source_point_ids")
    if (
        not isinstance(width, list)
        or len(width) != len(STAGES)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in width)
        or q_mode not in ALLOWED_Q_MODES
        or not isinstance(source_point_ids, list)
        or len(source_point_ids) != len(STAGES)
        or any(not isinstance(source_id, str) or not source_id.strip() for source_id in source_point_ids)
    ):
        _invalid("candidate plan identity is invalid")
    return (width[0], width[1], width[2]), str(q_mode), (
        source_point_ids[0],
        source_point_ids[1],
        source_point_ids[2],
    )


def _validate_formal_candidate_sources(
    width: tuple[int, int, int],
    source_point_ids: tuple[str, str, str],
    stage_axes: tuple[tuple[str, int], ...],
) -> None:
    expected = tuple(
        f"{axis_id}:w{value}"
        for value, (axis_id, _base_width) in zip(width, stage_axes, strict=True)
    )
    if source_point_ids != expected:
        _invalid("formal candidate source point ids drift from axis widths")


def _validate_plan_counts(
    raw_plan: Mapping[str, Any],
    entries: list[tuple[tuple[tuple[int, int, int], str], tuple[str, str, str]]],
    mapping: Mapping[tuple[tuple[int, int, int], str], tuple[str, str, str]],
) -> None:
    candidate_count = raw_plan.get("candidate_count")
    structure_count = raw_plan.get("structure_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(entries)
        or isinstance(structure_count, bool)
        or not isinstance(structure_count, int)
        or structure_count != len({identity[0] for identity in mapping})
    ):
        _invalid("candidate plan counts are inconsistent")


def materialize_p6_history_groups(
    plan_mapping: Mapping[tuple[tuple[int, int, int], str], tuple[str, str, str]],
    template: Mapping[str, Any],
    recipe: Mapping[str, Any],
    private_root: Path,
    local_output_root: Path,
    registry_output_path: Path,
) -> list[dict[str, Any]]:
    """Build registry groups from a validated formal P6 candidate plan."""
    observed_artifacts: set[str] = set()
    observed_outputs = {registry_output_path}
    recipe_version = recipe["schema_version"]
    base_stage_widths = recipe_v2_base_stage_widths(template, recipe_version)
    return [
        _materialize_group(
            width,
            provenance_by_q_mode,
            template,
            recipe,
            private_root,
            local_output_root,
            observed_artifacts,
            observed_outputs,
            base_stage_widths,
        )
        for width, provenance_by_q_mode in sorted(
            _plan_by_width(plan_mapping).items()
        )
    ]


def recipe_v2_base_stage_widths(
    template: Mapping[str, Any], recipe_version: str
) -> tuple[int, int, int] | None:
    """Return recipe-v2 materializer base widths, if present."""
    if recipe_version != RECIPE_V2:
        return None
    widths = template["external_training_binding"]["training_parameters"][
        "base_stage_widths"
    ]
    return (widths[0], widths[1], widths[2])


def _plan_by_width(
    plan_mapping: Mapping[tuple[tuple[int, int, int], str], tuple[str, str, str]],
) -> dict[tuple[int, int, int], dict[str, tuple[str, str, str]]]:
    by_width: dict[tuple[int, int, int], dict[str, tuple[str, str, str]]] = {}
    for (width, q_mode), source_point_ids in plan_mapping.items():
        by_width.setdefault(width, {})[q_mode] = source_point_ids
    return by_width


def _materialize_group(
    width: tuple[int, int, int],
    provenance_by_q_mode: Mapping[str, tuple[str, str, str]],
    template: Mapping[str, Any],
    recipe: Mapping[str, Any],
    private_root: Path,
    local_output_root: Path,
    observed_artifacts: set[str],
    observed_outputs: set[Path],
    base_stage_widths: tuple[int, int, int] | None,
) -> dict[str, Any]:
    width_values, group_id, artifact_id = _render_group_identity(
        width, recipe, observed_artifacts
    )
    q_modes = sorted(provenance_by_q_mode)
    contract = _materialize_group_contract(
        width, width_values, group_id, artifact_id, q_modes,
        template, recipe, local_output_root, observed_outputs, base_stage_widths
    )
    if recipe["schema_version"] == RECIPE_V2:
        contract = _validate_recipe_v2_group(
            contract, private_root, local_output_root, group_id
        )
    group = _registry_group_from_contract(
        width, width_values, group_id, q_modes, provenance_by_q_mode, contract
    )
    _validate_materialized_group(group)
    return group


def _render_group_identity(
    width: tuple[int, int, int],
    recipe: Mapping[str, Any],
    observed_artifacts: set[str],
) -> tuple[dict[str, int], str, str]:
    width_values = dict(zip(STAGE_WIDTH_FIELDS, width, strict=True))
    group_id = _render_template(recipe["group_id_template"], width_values)
    artifact_id = _render_template(recipe["artifact_id_template"], width_values)
    if (
        group_id != canonical_group_id("pyramid", width)
        or artifact_id in observed_artifacts
    ):
        _invalid("dynamic materialization identity collides or drifts")
    observed_artifacts.add(artifact_id)
    return width_values, group_id, artifact_id


def _materialize_group_contract(
    width: tuple[int, int, int],
    width_values: Mapping[str, int],
    group_id: str,
    artifact_id: str,
    q_modes: Sequence[str],
    template: Mapping[str, Any],
    recipe: Mapping[str, Any],
    local_output_root: Path,
    observed_outputs: set[Path],
    base_stage_widths: tuple[int, int, int] | None,
) -> dict[str, Any]:
    recipe_version = recipe["schema_version"]
    paths = _materialize_group_paths(
        recipe, width_values, group_id, artifact_id, q_modes,
        local_output_root, observed_outputs
    )
    contract = _base_group_contract(template, recipe_version)
    contract.update({
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "artifact_id": artifact_id,
        "source_status": _source_status_for_width(
            width, str(template["source_status"]), base_stage_widths
        ),
        "stage_widths": width_values,
        **paths,
    })
    return contract


def _materialize_group_paths(
    recipe: Mapping[str, Any],
    width_values: Mapping[str, int],
    group_id: str,
    artifact_id: str,
    q_modes: Sequence[str],
    local_output_root: Path,
    observed_outputs: set[Path],
) -> dict[str, Any]:
    render_values = {**width_values, "group_id": group_id, "artifact_id": artifact_id}
    if recipe["schema_version"] == RECIPE_V1:
        outputs = _render_recipe_v1_outputs(
            recipe, render_values, q_modes, local_output_root, observed_outputs
        )
        return {"materialization_outputs_by_q_mode": outputs}
    return {
        "shared_source_paths": _render_recipe_v2_shared_paths(
            recipe, render_values, local_output_root, observed_outputs
        )
    }


def _render_recipe_v1_outputs(
    recipe: Mapping[str, Any],
    render_values: Mapping[str, object],
    q_modes: Sequence[str],
    local_output_root: Path,
    observed_outputs: set[Path],
) -> dict[str, dict[str, str]]:
    outputs_by_q_mode: dict[str, dict[str, str]] = {}
    raw_outputs = recipe["output_path_templates_by_q_mode"]
    for q_mode in q_modes:
        raw_templates = raw_outputs.get(q_mode)
        if not isinstance(raw_templates, Mapping):
            _invalid("dynamic materialization output mapping is incomplete")
        rendered_outputs: dict[str, str] = {}
        for template_key in OUTPUT_TEMPLATE_KEYS:
            resolved_output = _render_unique_output(
                raw_templates[template_key],
                {**render_values, "q_mode": q_mode},
                local_output_root,
                observed_outputs,
            )
            rendered_outputs[template_key.removesuffix("_template")] = str(
                resolved_output
            )
        outputs_by_q_mode[q_mode] = rendered_outputs
    return outputs_by_q_mode


def _render_recipe_v2_shared_paths(
    recipe: Mapping[str, Any],
    render_values: Mapping[str, object],
    local_output_root: Path,
    observed_outputs: set[Path],
) -> dict[str, str]:
    shared_paths = {}
    for key in SHARED_SOURCE_PATH_KEYS:
        resolved_output = _render_unique_output(
            recipe["shared_source_path_templates"][key],
            render_values,
            local_output_root,
            observed_outputs,
        )
        shared_paths[key] = str(resolved_output)
    return shared_paths


def _render_unique_output(
    template: str,
    values: Mapping[str, object],
    local_output_root: Path,
    observed_outputs: set[Path],
) -> Path:
    resolved_output = _render_output_path(template, values, local_output_root)
    if resolved_output in observed_outputs:
        _invalid("dynamic materialization output path collides")
    observed_outputs.add(resolved_output)
    return resolved_output


def _render_output_path(
    template: str, values: Mapping[str, object], local_output_root: Path
) -> Path:
    relative = Path(_render_template(template, values))
    if relative.is_absolute() or ".." in relative.parts:
        _invalid("dynamic materialization output escapes the allowed root")
    try:
        resolved = (local_output_root / relative).resolve(strict=False)
    except OSError as error:
        raise ValueError(
            "dynamic materialization output is unavailable"
        ) from error
    if (
        resolved == local_output_root
        or not _is_relative_to(resolved, local_output_root)
        or resolved.is_symlink()
        or (resolved.exists() and resolved.is_dir())
    ):
        _invalid("dynamic materialization output escapes the allowed root")
    return resolved


def _render_template(template: str, values: Mapping[str, object]) -> str:
    try:
        rendered = template.format_map(values)
    except (KeyError, ValueError) as error:
        raise ValueError("dynamic materialization template failed") from error
    if not rendered or any(character in rendered for character in ("\x00", "\r", "\n")):
        _invalid("dynamic materialization template rendered an invalid value")
    return rendered


def _base_group_contract(
    template: Mapping[str, Any], recipe_version: str
) -> dict[str, Any]:
    contract = copy.deepcopy(dict(template))
    contract.pop("dynamic_materialization_recipe", None)
    if recipe_version == RECIPE_V2:
        contract.pop("materialization_outputs_by_q_mode", None)
        for key in (*SHARED_SOURCE_PATH_KEYS, *LEGACY_OUTPUT_KEYS):
            contract.pop(key, None)
    return contract


def _source_status_for_width(
    width: tuple[int, int, int],
    materializable_status: str,
    base_stage_widths: tuple[int, int, int] | None,
) -> str:
    return materializable_status


def _validate_recipe_v2_group(
    contract: Mapping[str, Any],
    private_root: Path,
    local_output_root: Path,
    group_id: str,
) -> dict[str, Any]:
    try:
        return validate_recipe_v2_group_training_contract(
            contract,
            private_root=private_root,
            local_output_root=local_output_root,
            group_id=group_id,
        )
    except P6HistoryTrainingContractError as error:
        raise ValueError("recipe-v2 group training contract is invalid") from error


def _registry_group_from_contract(
    width: tuple[int, int, int],
    width_values: Mapping[str, int],
    group_id: str,
    q_modes: Sequence[str],
    provenance_by_q_mode: Mapping[str, tuple[str, str, str]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "source_status": contract["source_status"],
        "source_evidence_sha256": contract["source_evidence_sha256"],
        "source_contract": contract,
        "source_contract_sha256": _canonical_sha(contract),
        "materialization_kind": "local_pyramid_tvm",
        "available_q_modes": list(q_modes),
        "source_point_ids_by_q_mode": {
            q_mode: list(provenance_by_q_mode[q_mode])
            for q_mode in q_modes
        },
        "graph_features": {
            "group_id": group_id,
            "model": "pyramid",
            "width": list(width),
            **width_values,
        },
    }


def _validate_materialized_group(group: Mapping[str, Any]) -> None:
    try:
        validate_structure_identity(group)
        validate_source_contract(group)
        source_group_q_modes(REGISTRY_SCHEMA_VERSION, group)
    except (TypeError, ValueError) as error:
        raise ValueError("derived source contract is invalid") from error


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("source contract is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _invalid(detail: str) -> None:
    raise ValueError(detail)
