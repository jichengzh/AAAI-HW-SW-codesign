"""Generic formal software-space enumeration from scanner-owned axes."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Mapping

from framework.stage1.structural_axis_digest import (
    SCANNER_AXIS_PROVENANCE_SOURCE,
    canonical_digest,
    scanner_structural_axes_digest,
)


_SEARCH_SPACE_FIELDS = frozenset(
    {
        "schema", "model", "hardware_target", "optimized_scope",
        "scanner_structural_axes_digest", "structural_axes", "axis_schema",
        "formal_q_modes", "formal_q_mode_provenance", "formal_candidate_policy",
        "software_candidates",
        "hardware_candidates", "model_search_policy", "hierarchical_blocks",
        "quant_units", "routing_segments", "skipped_blocks",
        "legal_candidate_policy", "claim_scope", "unsupported_conclusions",
    }
)
_AXIS_FIELDS = frozenset(
    {
        "axis_id", "dense_stage", "axis_kind", "base_width", "legal_widths",
        "member_b1_groups", "round_to", "provenance",
    }
)
_MEMBER_FIELDS = frozenset(
    {
        "b1_group_id", "module_path", "canonical_to_member_num",
        "canonical_to_member_den", "materializer_param", "role",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "source", "scanner_input_digest", "input_digest", "dataflow_group_ids",
        "config_digest", "checkpoint_digest", "scenario_digest",
        "scan_manifest_digest", "relevant_groups_digest",
        "structural_evidence_digest", "digest_sources",
    }
)
_DIGEST_SOURCES = {
    "config_digest": "trace_context.loaded_config",
    "checkpoint_digest": "trace_context.checkpoint_evidence",
    "scenario_digest": "scan_scenario",
    "scan_manifest_digest": "scanner_group_manifest",
    "structural_evidence_digest": "canonical_structural_axis_inputs",
}
_Q_PROVENANCE_FIELDS = frozenset(
    {"schema", "hardware_target", "sources", "formal_q_modes", "digest"}
)
_Q_SOURCE_FIELDS = frozenset({"hardware", "backend", "configured", "graph"})
_HARDWARE_TARGET_FIELDS = frozenset(
    {
        "name", "int8_align", "fp16_align", "int8_pack_factor",
        "alignment_enforcement",
    }
)
_POLICY = {
    "enumeration": "axis_schema.free_axes_x_formal_q_modes",
    "legacy_views": "diagnostic_only",
    "diagnostic_anchors_drive_formal_space": False,
    "source": "scanner_structural_axes",
}


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return canonical_digest(payload)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], field_name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} has unknown fields: {unknown}")


def _exact_fields(
    value: Mapping[str, Any], fields: frozenset[str], field_name: str
) -> None:
    _unknown_fields(value, fields, field_name)
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"{field_name} is missing fields: {missing}")


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _valid_digest(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name} must be a canonical SHA-256 digest")
    return value


def _validated_dense_stage(value: object, axis_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"axis_schema {axis_id}.dense_stage must be null or non-empty string")
    return value


def _validated_member(value: object, base: int, widths: list[int]) -> dict[str, Any]:
    member = _mapping(value, "axis member")
    _unknown_fields(member, _MEMBER_FIELDS, "axis member")
    for field_name in ("b1_group_id", "module_path", "materializer_param", "role"):
        if not isinstance(member.get(field_name), str) or not member[field_name].strip():
            raise ValueError(f"axis member.{field_name} must be a non-empty string")
    num = _positive_int(member.get("canonical_to_member_num"), "axis member ratio")
    den = _positive_int(member.get("canonical_to_member_den"), "axis member ratio")
    member_base = base * num
    if member_base % den or any(width * num % den for width in widths):
        raise ValueError("axis member ratio must map every width to an integer")
    return deepcopy(dict(member))


def _validated_axis(value: object, expected_kind: str) -> dict[str, Any]:
    axis = _mapping(value, "axis_schema axis")
    _unknown_fields(axis, _AXIS_FIELDS, "axis_schema axis")
    axis_id = axis.get("axis_id")
    if not isinstance(axis_id, str) or not axis_id.strip():
        raise ValueError("axis_schema axis_id must be a non-empty string")
    if axis.get("axis_kind") != expected_kind:
        raise ValueError(f"axis_schema {axis_id} axis_kind must be {expected_kind}")
    dense_stage = _validated_dense_stage(axis.get("dense_stage"), axis_id)
    base = _positive_int(axis.get("base_width"), f"axis_schema {axis_id}.base_width")
    widths = axis.get("legal_widths")
    if not isinstance(widths, list) or not widths:
        raise ValueError(f"axis_schema {axis_id}.legal_widths must be non-empty")
    parsed = [_positive_int(width, f"axis_schema {axis_id} positive integer") for width in widths]
    if parsed != sorted(set(parsed)):
        raise ValueError(f"axis_schema {axis_id}.legal_widths must be sorted and unique")
    if parsed[-1] != base or any(width > base for width in parsed):
        raise ValueError(f"axis_schema {axis_id}.legal_widths violate base_width")
    completed = _complete_axis(axis, parsed, base)
    completed["dense_stage"] = dense_stage
    return completed


def _complete_axis(
    axis: Mapping[str, Any], widths: list[int], base: int
) -> dict[str, Any]:
    axis_id = axis["axis_id"]
    round_to = _positive_int(axis.get("round_to"), f"axis_schema {axis_id}.round_to")
    if any(width % round_to for width in widths):
        raise ValueError(f"axis_schema {axis_id}.legal_widths violate round_to")
    provenance = _mapping(axis.get("provenance"), f"axis_schema {axis_id}.provenance")
    _validate_axis_provenance(provenance, axis_id, axis["axis_kind"])
    members = axis.get("member_b1_groups")
    if not isinstance(members, list) or not members:
        raise ValueError(f"axis_schema {axis_id}.member_b1_groups must be non-empty")
    out = deepcopy(dict(axis))
    out["member_b1_groups"] = [_validated_member(row, base, widths) for row in members]
    out["legal_widths"] = widths
    return out


def _validate_axis_provenance(
    provenance: Mapping[str, Any], axis_id: str, axis_kind: str
) -> None:
    optional = frozenset({"derived_from"}) if axis_kind == "fixed_derived" else frozenset()
    _unknown_fields(provenance, _PROVENANCE_FIELDS | optional, f"axis_schema {axis_id}.provenance")
    missing = sorted(_PROVENANCE_FIELDS - set(provenance))
    if missing:
        raise ValueError(f"axis_schema {axis_id}.provenance is missing fields: {missing}")
    if provenance.get("source") != SCANNER_AXIS_PROVENANCE_SOURCE:
        raise ValueError(f"axis_schema {axis_id}.provenance.source is invalid")
    _validate_provenance_digests(provenance, axis_id)
    _validate_provenance_groups(provenance.get("dataflow_group_ids"), axis_id)
    if provenance.get("digest_sources") != _DIGEST_SOURCES:
        raise ValueError(f"axis_schema {axis_id}.provenance.digest_sources is invalid")
    if axis_kind == "fixed_derived" and not str(provenance.get("derived_from") or "").strip():
        raise ValueError(f"axis_schema {axis_id}.provenance.derived_from is required")


def _validate_provenance_digests(
    provenance: Mapping[str, Any], axis_id: str
) -> None:
    for field_name in (
        "scanner_input_digest", "input_digest", "config_digest",
        "checkpoint_digest", "scenario_digest", "scan_manifest_digest",
        "relevant_groups_digest", "structural_evidence_digest",
    ):
        _valid_digest(provenance.get(field_name), f"axis_schema {axis_id}.{field_name}")
    if provenance["input_digest"] != provenance["scanner_input_digest"]:
        raise ValueError(f"axis_schema {axis_id}.provenance.input_digest alias mismatch")


def _validate_provenance_groups(value: object, axis_id: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"axis_schema {axis_id}.provenance.dataflow_group_ids must be non-empty")
    if any(not isinstance(group_id, str) or not group_id.strip() for group_id in value):
        raise ValueError(f"axis_schema {axis_id}.provenance.dataflow_group_ids are invalid")
    if len(value) != len(set(value)):
        raise ValueError(f"axis_schema {axis_id}.provenance.dataflow_group_ids must be unique")


def _validated_axis_schema(search_space: Mapping[str, Any]) -> dict[str, Any]:
    schema = _mapping(search_space.get("axis_schema"), "axis_schema")
    _unknown_fields(schema, frozenset({"free_axes", "fixed_derived_axes"}), "axis_schema")
    free_raw = schema.get("free_axes")
    fixed_raw = schema.get("fixed_derived_axes")
    if not isinstance(free_raw, list) or not free_raw:
        raise ValueError("axis_schema.free_axes must be a non-empty list")
    if not isinstance(fixed_raw, list):
        raise ValueError("axis_schema.fixed_derived_axes must be a list")
    free = [_validated_axis(row, "free") for row in free_raw]
    fixed = [_validated_axis(row, "fixed_derived") for row in fixed_raw]
    _validate_axis_relations(free, fixed)
    return {"free_axes": free, "fixed_derived_axes": fixed}


def _validate_axis_relations(free: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> None:
    axes = free + fixed
    axis_ids = [axis["axis_id"] for axis in axes]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("duplicate axis_id in axis_schema")
    for rows, field_name in ((free, "free_axes"), (fixed, "fixed_derived_axes")):
        ids = [axis["axis_id"] for axis in rows]
        if ids != sorted(ids):
            raise ValueError(f"axis_schema.{field_name} must use canonical axis order")
    free_ids = {axis["axis_id"] for axis in free}
    for axis in fixed:
        if axis["provenance"].get("derived_from") not in free_ids:
            raise ValueError("fixed-derived provenance must name a free axis")


def _validated_q_mode_list(value: object, field_name: str) -> list[str]:
    modes = value
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"{field_name} must be a non-empty list")
    if any(isinstance(mode, bool) or not isinstance(mode, str) for mode in modes):
        raise ValueError(f"{field_name} contains an invalid value")
    supported = [mode for mode in ("fp16", "int8") if mode in modes]
    if modes != supported or len(modes) != len(set(modes)):
        raise ValueError(f"{field_name} must be supported, unique, and canonical")
    return list(modes)


def _validated_q_modes(search_space: Mapping[str, Any]) -> list[str]:
    return _validated_q_mode_list(search_space.get("formal_q_modes"), "formal_q_modes")


def _validated_hardware_target(search_space: Mapping[str, Any]) -> dict[str, Any]:
    target = _mapping(search_space.get("hardware_target"), "formal_q_mode_provenance.hardware_target")
    _exact_fields(target, _HARDWARE_TARGET_FIELDS, "formal_q_mode_provenance.hardware_target")
    if not isinstance(target.get("name"), str) or not target["name"].strip():
        raise ValueError("formal_q_mode_provenance.hardware_target.name is invalid")
    for field_name in ("int8_align", "fp16_align", "int8_pack_factor"):
        _positive_int(target.get(field_name), f"formal_q_mode_provenance.hardware_target.{field_name}")
    enforcement = target.get("alignment_enforcement")
    if not isinstance(enforcement, str) or not enforcement.strip():
        raise ValueError("formal_q_mode_provenance.hardware_target.alignment_enforcement is invalid")
    return deepcopy(dict(target))


def _validated_q_mode_provenance(
    search_space: Mapping[str, Any], q_modes: list[str]
) -> dict[str, Any]:
    field_name = "formal_q_mode_provenance"
    provenance = _mapping(search_space.get(field_name), field_name)
    _exact_fields(provenance, _Q_PROVENANCE_FIELDS, field_name)
    if provenance.get("schema") != "formal_q_mode_provenance_v1":
        raise ValueError(f"{field_name}.schema is invalid")
    hardware_target = _validated_hardware_target(search_space)
    if provenance.get("hardware_target") != hardware_target:
        raise ValueError(f"{field_name}.hardware_target does not match search_space")
    sources = _mapping(provenance.get("sources"), f"{field_name}.sources")
    _exact_fields(sources, _Q_SOURCE_FIELDS, f"{field_name}.sources")
    parsed_sources = {
        name: _validated_q_mode_list(sources.get(name), f"{field_name}.sources.{name}")
        for name in sorted(_Q_SOURCE_FIELDS)
    }
    if sorted(set.intersection(*(set(values) for values in parsed_sources.values()))) != q_modes:
        raise ValueError(f"{field_name}.sources intersection does not match formal_q_modes")
    if _validated_q_mode_list(provenance.get("formal_q_modes"), field_name) != q_modes:
        raise ValueError(f"{field_name}.formal_q_modes alias mismatch")
    digest = _valid_digest(provenance.get("digest"), f"{field_name}.digest")
    unsigned = {key: value for key, value in provenance.items() if key != "digest"}
    if digest != _canonical_digest(unsigned):
        raise ValueError(f"{field_name}.digest mismatch")
    return deepcopy(dict(provenance))


def _validate_source_contract(
    search_space: Mapping[str, Any], axis_schema: Mapping[str, Any]
) -> None:
    policy = _mapping(search_space.get("formal_candidate_policy"), "formal_candidate_policy")
    _unknown_fields(policy, frozenset(_POLICY), "formal_candidate_policy")
    if dict(policy) != _POLICY:
        raise ValueError("formal_candidate_policy does not name scanner structural axes")
    axes = axis_schema["free_axes"] + axis_schema["fixed_derived_axes"]
    supplied = _valid_digest(
        search_space.get("scanner_structural_axes_digest"),
        "scanner_structural_axes_digest",
    )
    if supplied != scanner_structural_axes_digest(axes):
        raise ValueError("scanner structural axes digest mismatch")
    _validate_structural_axis_alias(search_space.get("structural_axes"), axes)


def _validate_structural_axis_alias(value: object, axes: list[dict[str, Any]]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError("structural_axes must be a list")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError("structural_axes entries must be objects")
    if sorted(value, key=lambda row: row.get("axis_id", "")) != sorted(
        axes, key=lambda row: row["axis_id"]
    ):
        raise ValueError("structural_axes provenance drift from axis_schema")


def _validated_contract(
    search_space: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if not isinstance(search_space, Mapping):
        raise ValueError("search_space must be an object")
    _unknown_fields(search_space, _SEARCH_SPACE_FIELDS, "search_space")
    if search_space.get("schema") != "stage2_search_space_v1":
        raise ValueError("search_space.schema must be stage2_search_space_v1")
    if not isinstance(search_space.get("model"), str) or not search_space["model"].strip():
        raise ValueError("search_space.model must be a non-empty string")
    axis_schema = _validated_axis_schema(search_space)
    q_modes = _validated_q_modes(search_space)
    q_provenance = _validated_q_mode_provenance(search_space, q_modes)
    _validate_source_contract(search_space, axis_schema)
    return axis_schema, q_modes, q_provenance


def _source_provenance(
    search_space: Mapping[str, Any],
    axis_schema: Mapping[str, Any],
    q_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    axes = axis_schema["free_axes"] + axis_schema["fixed_derived_axes"]
    return {
        "scanner_structural_axes_digest": search_space[
            "scanner_structural_axes_digest"
        ],
        "formal_candidate_policy": deepcopy(search_space["formal_candidate_policy"]),
        "axis_provenance": [
            {"axis_id": axis["axis_id"], "provenance": deepcopy(axis["provenance"])}
            for axis in axes
        ],
        "formal_q_mode_provenance": deepcopy(dict(q_provenance)),
    }


def _candidate(
    index: int,
    structure_index: int,
    width_tuple: tuple[int, ...],
    q_mode: str,
    axis_schema: dict[str, Any],
    axis_order: list[str],
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "axis_schema": deepcopy(axis_schema),
        "axis_order": list(axis_order),
        "width_tuple": list(width_tuple),
        "q_mode": q_mode,
        "source_provenance": deepcopy(source_provenance),
    }
    return {
        "candidate_index": index,
        "structure_index": structure_index,
        "candidate_id": _canonical_digest(identity),
        "axis_order": list(axis_order),
        "axis_values": dict(zip(axis_order, width_tuple, strict=True)),
        "width_tuple": list(width_tuple),
        "q_mode": q_mode,
        "identity": identity,
    }


def build_formal_software_plan(search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate ordered free-axis widths across the formal q modes."""
    axis_schema, q_modes, q_provenance = _validated_contract(search_space)
    free_axes = axis_schema["free_axes"]
    structures = list(product(*(axis["legal_widths"] for axis in free_axes)))
    axis_order = [axis["axis_id"] for axis in free_axes]
    provenance = _source_provenance(search_space, axis_schema, q_provenance)
    candidates = [
        _candidate(
            structure_index * len(q_modes) + q_index,
            structure_index,
            width_tuple,
            q_mode,
            axis_schema,
            axis_order,
            provenance,
        )
        for structure_index, width_tuple in enumerate(structures)
        for q_index, q_mode in enumerate(q_modes)
    ]
    return {
        "schema": "formal_software_plan_v1",
        "model": search_space.get("model"),
        "free_axis_count": len(free_axes),
        "structure_count": len(structures),
        "candidate_count": len(candidates),
        "axis_schema": axis_schema,
        "q_modes": q_modes,
        "source_provenance": provenance,
        "candidates": candidates,
    }
