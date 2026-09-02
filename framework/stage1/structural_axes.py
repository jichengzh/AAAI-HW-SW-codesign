"""Scanner-owned structural-axis derivation for formal search spaces."""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from framework.stage1.structural_axis_contract import (
    axis_provenance, canonical_source_dataflow_relations,
    formal_scanner_evidence_payload,
    require_free_interface_proof, scanner_digest_sources,
    scanner_scenario_payload, seal_scanner_inputs,
    validate_axis_provenance, validate_dataflow_relations,
    validated_base_widths, validated_materializer_bindings,
    validated_retained_groups, validate_scanner_inputs,
)
from framework.stage1.structural_axis_bindings import resolved_binding_authority
from framework.stage1.structural_axis_digest import (
    SCANNER_AXIS_PROVENANCE_SOURCE, canonical_digest, immutable_mapping,
    mutable_mapping,
)
from framework.stage1.structural_axis_members import (
    ResolvedAxisMember, require_independent_interface,
    resolve_axis_members, validate_derived_references, validate_group_manifest,
)
from framework.stage1.structural_axis_widths import (
    derive_legal_widths, scenario_axis_constraint, validated_width_policy,
)

if TYPE_CHECKING:
    from framework.stage1.adapters import ScanScenario, TraceContext


@dataclass(frozen=True)
class AxisMemberBinding:
    b1_group_id: str
    module_path: str
    canonical_to_member_num: int
    canonical_to_member_den: int
    materializer_param: str
    role: str


@dataclass(frozen=True)
class StructuralAxis:
    axis_id: str
    dense_stage: str | None
    axis_kind: Literal["free", "fixed_derived"]
    base_width: int
    legal_widths: tuple[int, ...]
    member_b1_groups: tuple[AxisMemberBinding, ...]
    round_to: int
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", immutable_mapping(self.provenance))


@dataclass(frozen=True)
class StructuralAxisBundle:
    axes: tuple[StructuralAxis, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", immutable_mapping(self.diagnostics))

    @property
    def free_axes(self) -> tuple[StructuralAxis, ...]:
        return tuple(axis for axis in self.axes if axis.axis_kind == "free")

    @property
    def fixed_axes(self) -> tuple[StructuralAxis, ...]:
        return tuple(axis for axis in self.axes if axis.axis_kind == "fixed_derived")


@dataclass(frozen=True)
class _ResolvedMaterializerSource:
    source: Any
    config_path: str
    base_width: int
    module_path: str
    depgraph_width: int
    group: Mapping[str, Any]
    relation: Mapping[str, Any]
    role: str
    members: tuple[ResolvedAxisMember, ...]


def build_structural_axis_inputs(
    *,
    trace_context: "TraceContext",
    prune_groups: Sequence[Mapping[str, Any]],
    scenario: "ScanScenario",
    group_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve adapter selectors into scanner-owned structural-axis evidence."""

    sources = tuple(trace_context.materializer_sources)
    _validate_materializer_sources(sources)
    groups = _normalized_depgraph_groups(prune_groups)
    scenario_evidence = scanner_scenario_payload(scenario)
    source_relations = canonical_source_dataflow_relations(
        trace_context.dataflow_relations
    )
    group_provenance = validate_group_manifest(
        groups, group_manifest, scenario_evidence, source_relations
    )
    checkpoint_digest, config_file_digest, checkpoint_widths = _checkpoint_width_evidence(
        trace_context.checkpoint_evidence)
    resolved = tuple(
        _resolve_materializer_source(source, trace_context, groups, checkpoint_widths)
        for source in sources
    )
    _reject_duplicate_physical_axes(resolved)
    return seal_scanner_inputs(
        _scanner_input_payload(
            trace_context, groups, resolved, scenario_evidence, group_manifest or {},
            group_provenance, checkpoint_digest, config_file_digest, source_relations,
        )
    )


def _scanner_input_payload(
    trace_context: "TraceContext",
    groups: Sequence[Mapping[str, Any]],
    resolved: Sequence[_ResolvedMaterializerSource],
    scenario_evidence: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    group_provenance: Mapping[str, str],
    checkpoint_digest: str,
    config_file_digest: str | None,
    source_relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scenario_digest = canonical_digest(scenario_evidence)
    config_digest = canonical_digest(trace_context.loaded_config)
    authority_sources, derived_relations, bindings = resolved_binding_authority(resolved)
    base_widths = _base_width_authority(resolved, authority_sources, bindings)
    base_provenance = _base_provenance(config_digest, checkpoint_digest, config_file_digest)
    scanner_evidence = formal_scanner_evidence_payload(
        groups, group_manifest, scenario_evidence, authority_sources, bindings,
        base_widths, base_provenance, source_relations,
    )
    return {
        "prune_groups": [dict(group) for group in groups],
        "source_dataflow_relations": authority_sources,
        "dataflow_relations": derived_relations,
        "materializer_bindings": bindings,
        "base_widths": base_widths,
        "backend_constraints": [
            scenario_axis_constraint(item.source.axis_id, scenario_evidence)
            for item in resolved
        ],
        "scanner_evidence": scanner_evidence,
        "provenance": {
            "source": SCANNER_AXIS_PROVENANCE_SOURCE,
            "checkpoint_digest": checkpoint_digest,
            "config_digest": config_digest,
            "selectors": [_resolved_provenance(item) for item in resolved],
            "scenario_digest": scenario_digest,
            "digest_sources": scanner_digest_sources(),
            **group_provenance,
            "source_relation_authority_schema": scanner_evidence[
                "source_relation_authority"]["schema"],
            "source_group_manifest_digest": group_provenance["scan_manifest_digest"],
            "scan_manifest_digest": canonical_digest(scanner_evidence),
        },
    }

def _base_provenance(
    config_digest: str,
    checkpoint_digest: str,
    config_file_digest: str | None,
) -> dict[str, str]:
    provenance = {
        "source": "trace_context.config_checkpoint_depgraph",
        "config_digest": config_digest,
        "checkpoint_digest": checkpoint_digest,
    }
    if config_file_digest is not None:
        provenance["config_file_digest"] = config_file_digest
    return provenance


def _base_width_authority(
    resolved: Sequence[_ResolvedMaterializerSource],
    sources: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_by_axis = {row["canonical_axis_id"]: row for row in sources}
    binding_by_group = {row["b1_group_id"]: row for row in bindings}
    return [
        _resolved_base_width(
            item,
            source_by_axis[item.source.axis_id],
            binding_by_group[item.group["group_id"]],
        )
        for item in resolved
    ]


def _resolve_materializer_source(
    source: Any,
    context: "TraceContext",
    groups: Sequence[Mapping[str, Any]],
    checkpoint_widths: Mapping[str, Any],
) -> _ResolvedMaterializerSource:
    config_path, base_width = _resolve_config_base(context.loaded_config, source.config_selector)
    module_path, module = _resolve_trace_module(context.trace_modules, source.module_root_selector)
    module_width = _module_width(module, source.mutation_kind)
    checkpoint_width = _checkpoint_module_width(checkpoint_widths, module_path)
    group = _matching_depgraph_group(groups, module_path)
    depgraph_width = _positive_int(group.get("cur_width"), "DepGraph cur_width")
    _require_equal_widths(base_width=base_width, module_width=module_width,
                          checkpoint_width=checkpoint_width,
                          depgraph_width=depgraph_width, axis_id=source.axis_id)
    relation = _matching_dataflow_relation(
        context.dataflow_relations, source.axis_id, module_path, group["group_id"]
    )
    require_independent_interface(source, relation)
    role = _materializer_role(source.allowed_roles)
    members = resolve_axis_members(
        relation=relation,
        groups=groups,
        trace_modules=context.trace_modules,
        checkpoint_widths=checkpoint_widths,
        canonical_group=group,
        mutation_kind=source.mutation_kind,
        canonical_role=role,
        boundary_selectors=tuple(
            item.module_root_selector for item in context.materializer_sources
        ),
    )
    return _ResolvedMaterializerSource(
        source=source,
        config_path=config_path,
        base_width=base_width,
        module_path=module_path,
        depgraph_width=depgraph_width,
        group=group,
        relation=relation,
        role=role,
        members=members,
    )


def _resolved_base_width(
    item: _ResolvedMaterializerSource,
    source: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "axis_id": item.source.axis_id,
        "width": item.base_width,
        "config_path": item.config_path,
        "config_width": item.base_width,
        "checkpoint_module_path": item.module_path,
        "checkpoint_width": item.base_width,
        "canonical_group_id": item.group["group_id"],
        "canonical_group_width": item.depgraph_width,
        "source_relation_digest": canonical_digest(source),
        "materializer_binding_digest": binding["materializer_binding_digest"],
    }


def _resolved_provenance(item: _ResolvedMaterializerSource) -> dict[str, Any]:
    return {
        "axis_id": item.source.axis_id,
        "config_selector": item.source.config_selector,
        "resolved_config_path": item.config_path,
        "module_root_selector": item.source.module_root_selector,
        "resolved_module_path": item.module_path,
        "mutation_kind": item.source.mutation_kind,
        "role": item.role,
        "adapter": dict(item.source.provenance),
    }


def _validate_materializer_sources(sources: Sequence[Any]) -> None:
    if not sources:
        raise ValueError("at least one adapter materializer binding is required")
    seen: set[str] = set()
    for source in sources:
        axis_id = _text(source.axis_id, "materializer axis_id")
        if axis_id in seen:
            raise ValueError(f"duplicate materializer axis declaration: {axis_id}")
        seen.add(axis_id)
        if not isinstance(source.provenance, Mapping) or not source.provenance:
            raise ValueError(
                f"materializer selector provenance is required for axis {axis_id}"
            )


def _reject_duplicate_physical_axes(
    resolved: Sequence[_ResolvedMaterializerSource],
) -> None:
    coordinates = (
        ("config path", lambda item: item.config_path),
        ("module path", lambda item: item.module_path),
        ("DepGraph group", lambda item: str(item.group["group_id"])),
    )
    for coordinate_name, coordinate in coordinates:
        owners: dict[str, str] = {}
        for item in resolved:
            value = coordinate(item)
            previous = owners.get(value)
            if previous is not None and previous != item.source.axis_id:
                raise ValueError(
                    "duplicate physical axis "
                    f"{coordinate_name} {value!r}: {previous}, {item.source.axis_id}"
                )
            owners[value] = item.source.axis_id


def _normalized_depgraph_groups(
    prune_groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalized = []
    seen: set[str] = set()
    for raw in prune_groups:
        group_id = _text(raw.get("group_id"), "DepGraph group_id")
        root_layer = _text(raw.get("root_layer"), "DepGraph root_layer")
        if group_id in seen:
            raise ValueError(f"duplicate DepGraph group: {group_id}")
        seen.add(group_id)
        normalized.append({**dict(raw), "group_id": group_id, "root_layer": root_layer})
    if not normalized:
        raise ValueError("DepGraph groups are required")
    return tuple(normalized)


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(child, path))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, child in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_config(child, path))
        return flattened
    return {prefix: value}


def _normalized_selector(selector: str) -> str:
    return selector.replace("[", ".").replace("]", "").strip(".")


def _resolve_config_base(
    loaded_config: Mapping[str, Any], selector: str
) -> tuple[str, int]:
    normalized = _normalized_selector(_text(selector, "config selector"))
    matches = [
        (path, value)
        for path, value in _flatten_config(loaded_config).items()
        if fnmatch.fnmatchcase(path, normalized)
    ]
    if not matches:
        raise ValueError(f"canonical base not found for config selector {selector!r}")
    if len(matches) != 1:
        paths = [path for path, _ in matches]
        raise ValueError(
            f"config selector must resolve to a unique config path: {paths}"
        )
    path, value = matches[0]
    return path, _positive_int(value, f"canonical base at {path}")


def _resolve_trace_module(
    trace_modules: Mapping[str, Any], selector: str
) -> tuple[str, Any]:
    normalized = _text(selector, "module root selector")
    matches = [
        (path, module)
        for path, module in trace_modules.items()
        if fnmatch.fnmatchcase(str(path), normalized)
    ]
    if len(matches) != 1:
        paths = [str(path) for path, _ in matches]
        raise ValueError(
            f"module selector must resolve to a unique trace module: {paths}"
        )
    return str(matches[0][0]), matches[0][1]


def _module_width(module: Any, mutation_kind: str) -> int:
    attribute = _text(mutation_kind, "mutation_kind")
    if not hasattr(module, attribute):
        raise ValueError(
            f"trace module does not expose mutation width attribute {attribute!r}"
        )
    return _positive_int(getattr(module, attribute), f"trace module {attribute}")


def _checkpoint_width_evidence(
    checkpoint_evidence: Mapping[str, Any],
) -> tuple[str, str | None, Mapping[str, Any]]:
    digest = str(checkpoint_evidence.get("digest") or "")
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("checkpoint evidence digest is required")
    widths = checkpoint_evidence.get("module_widths")
    if not isinstance(widths, Mapping) or not widths:
        raise ValueError("checkpoint evidence module_widths are required")
    config_file_digest = checkpoint_evidence.get("config_file_digest")
    if config_file_digest is not None and (
        not isinstance(config_file_digest, str)
        or len(config_file_digest) != 64
        or any(character not in "0123456789abcdef" for character in config_file_digest)
    ):
        raise ValueError("config file evidence digest is invalid")
    return digest, config_file_digest, widths


def _checkpoint_module_width(
    checkpoint_widths: Mapping[str, Any], module_path: str
) -> int:
    if module_path not in checkpoint_widths:
        raise ValueError(
            f"checkpoint evidence is missing module width for {module_path}"
        )
    return _positive_int(
        checkpoint_widths[module_path], f"checkpoint evidence for {module_path}"
    )


def _matching_depgraph_group(
    groups: Sequence[Mapping[str, Any]], module_path: str
) -> Mapping[str, Any]:
    matches = [group for group in groups if group.get("root_layer") == module_path]
    if len(matches) != 1:
        group_ids = [str(group.get("group_id")) for group in matches]
        raise ValueError(
            f"materializer binding must match exactly one DepGraph group: {group_ids}"
        )
    return matches[0]


def _matching_dataflow_relation(
    relations: Sequence[Mapping[str, Any]],
    axis_id: str,
    module_path: str,
    group_id: str,
) -> Mapping[str, Any]:
    matches = []
    for relation in relations:
        canonical_axis = str(relation.get("canonical_axis_id") or "")
        selector = relation.get("module_root_selector")
        related_group = relation.get("group_id")
        selector_matches = selector is not None and fnmatch.fnmatchcase(
            module_path, str(selector)
        )
        group_matches = related_group is not None and str(related_group) == group_id
        if canonical_axis == axis_id and (selector_matches or group_matches):
            matches.append(relation)
    if len(matches) != 1:
        raise ValueError(
            f"axis {axis_id} must have one dataflow relation for DepGraph group {group_id}"
        )
    return matches[0]


def _require_equal_widths(
    *,
    base_width: int,
    module_width: int,
    checkpoint_width: int,
    depgraph_width: int,
    axis_id: str,
) -> None:
    if checkpoint_width != module_width:
        raise ValueError(
            f"checkpoint evidence disagrees with trace module for axis {axis_id}: "
            f"{checkpoint_width} != {module_width}"
        )
    if depgraph_width != module_width:
        raise ValueError(
            f"DepGraph evidence disagrees with trace module for axis {axis_id}: "
            f"{depgraph_width} != {module_width}"
        )
    if base_width != module_width:
        raise ValueError(
            f"canonical base disagrees with checkpoint-loaded module for axis {axis_id}: "
            f"{base_width} != {module_width}"
        )


def _materializer_role(roles: Sequence[str]) -> str:
    normalized = tuple(str(role).strip() for role in roles if str(role).strip())
    if not normalized:
        raise ValueError("materializer binding requires at least one allowed role")
    return normalized[0]


def derive_structural_axes(raw_scan: Mapping[str, Any]) -> StructuralAxisBundle:
    inputs = raw_scan.get("structural_axis_inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("structural_axis_inputs are required")
    scanner_provenance = validate_scanner_inputs(inputs)
    retained_groups = validated_retained_groups(inputs)
    prune_groups = _retained_groups_by_id(retained_groups)
    binding_rows = validated_materializer_bindings(inputs)
    relation_rows = _required_list(inputs, "dataflow_relations")
    source_relations = _required_list(inputs, "source_dataflow_relations")
    scanner_evidence = inputs["scanner_evidence"]
    validate_dataflow_relations(
        relation_rows,
        source_relations,
        retained_groups,
        scanner_provenance["source_relation_authority_schema"],
    )
    dataflow = _dataflow_axes(relation_rows)
    base_rows = validated_base_widths(inputs, retained_groups, binding_rows)
    base_widths = _width_map(list(base_rows), "base_widths")
    constraints = _constraint_map(_required_backend_constraints(inputs))
    bindings = list(binding_rows)
    scenario_evidence = scanner_evidence["scenario"]

    axis_bindings = _bindings_by_axis(bindings)
    axes = [
        _derive_axis(axis_id, axis_bindings, dataflow, base_widths, constraints,
                     prune_groups, relation_rows, scanner_provenance,
                     str(inputs["scanner_input_digest"]), scenario_evidence)
        for axis_id in _ordered_axis_ids(dataflow, axis_bindings)
    ]
    validate_derived_references(axes)
    return StructuralAxisBundle(axes=tuple(axes), diagnostics={"axis_count": len(axes)})


def _bindings_by_axis(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("materializer_bindings entries must be objects")
        axis_id = _text(binding.get("axis_id"), "materializer_bindings.axis_id")
        grouped.setdefault(axis_id, []).append(binding)
    return grouped


def _derive_axis(
    axis_id: str,
    axis_bindings: Mapping[str, list[Mapping[str, Any]]],
    dataflow: Mapping[str, tuple[str, ...]],
    base_widths: Mapping[str, int],
    constraints: Mapping[str, Mapping[str, Any]],
    prune_groups: Mapping[str, Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    scanner_provenance: Mapping[str, Any],
    scanner_input_digest: str,
    scenario_evidence: Mapping[str, Any],
) -> StructuralAxis:
    bindings = axis_bindings.get(axis_id, [])
    base_width = _required_axis_value(base_widths, axis_id, "canonical base width")
    constraint = _required_axis_value(constraints, axis_id, "backend constraint")
    round_to = _positive_int(constraint.get("round_to"), "round_to")
    hardware_alignment = _positive_int(
        constraint.get("hardware_alignment", round_to), "hardware_alignment"
    )
    policy = validated_width_policy(constraint, axis_id, scenario_evidence)
    round_to, legal_widths = derive_legal_widths(
        base_width=base_width,
        policy=policy,
        round_to=round_to,
        hardware_alignment=hardware_alignment,
    )
    members = tuple(
        _member_binding(binding, prune_groups, base_width, legal_widths)
        for binding in bindings
    )
    if not members:
        raise ValueError(f"structural axis {axis_id} has no member bindings")
    axis_kind = _axis_kind(axis_id, bindings)
    if axis_kind == "free":
        require_free_interface_proof(axis_id, bindings, relation_rows)
    provenance = axis_provenance(axis_id, axis_kind, bindings, dataflow.get(axis_id, ()),
                                 scanner_provenance, scanner_input_digest)
    return StructuralAxis(
        axis_id=axis_id,
        dense_stage=_dense_stage(axis_id),
        axis_kind=axis_kind,
        base_width=base_width,
        legal_widths=legal_widths,
        member_b1_groups=members,
        round_to=round_to,
        provenance=provenance,
    )


def axis_to_dict(axis: StructuralAxis) -> dict[str, Any]:
    return {
        "axis_id": axis.axis_id,
        "dense_stage": axis.dense_stage,
        "axis_kind": axis.axis_kind,
        "base_width": axis.base_width,
        "legal_widths": list(axis.legal_widths),
        "member_b1_groups": [
            {
                "b1_group_id": member.b1_group_id,
                "module_path": member.module_path,
                "canonical_to_member_num": member.canonical_to_member_num,
                "canonical_to_member_den": member.canonical_to_member_den,
                "materializer_param": member.materializer_param,
                "role": member.role,
            }
            for member in axis.member_b1_groups
        ],
        "round_to": axis.round_to,
        "provenance": mutable_mapping(axis.provenance),
    }


def axis_to_scanner_dict(
    axis: StructuralAxis,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    payload = axis_to_dict(axis)
    validate_axis_provenance(payload["provenance"], inputs)
    return payload


def _required_list(inputs: Mapping[str, Any], key: str) -> list[Any]:
    value = inputs.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"structural_axis_inputs.{key} must be a non-empty list")
    return value


def _required_base_widths(inputs: Mapping[str, Any]) -> list[Any]:
    try:
        return _required_list(inputs, "base_widths")
    except ValueError as exc:
        raise ValueError("canonical base widths are required") from exc


def _required_backend_constraints(inputs: Mapping[str, Any]) -> list[Any]:
    try:
        return _required_list(inputs, "backend_constraints")
    except ValueError as exc:
        raise ValueError("backend constraints are required") from exc


def _required_axis_value(
    values: Mapping[str, Any],
    axis_id: str,
    description: str,
) -> Any:
    if axis_id not in values:
        raise ValueError(f"{description} is missing for structural axis {axis_id}")
    return values[axis_id]


def _retained_groups_by_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("retained prune group rows must be objects")
        row_id = _text(
            row.get("group_id", row.get("id")), "retained prune group_id"
        )
        if row_id in out:
            raise ValueError(f"duplicate retained prune group_id: {row_id}")
        out[row_id] = row
    return out


def _dataflow_axes(rows: list[Any]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("dataflow_relations entries must be objects")
        group_id = _text(row.get("group_id"), "dataflow_relations.group_id")
        axis_id = _text(row.get("canonical_axis_id"), "canonical_axis_id")
        out.setdefault(axis_id, []).append(group_id)
    return {axis_id: tuple(group_ids) for axis_id, group_ids in out.items()}


def _width_map(rows: list[Any], field_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{field_name} entries must be objects")
        axis_id = _text(row.get("axis_id"), f"{field_name}.axis_id")
        out[axis_id] = _positive_int(row.get("width"), f"{field_name}.width")
    return out


def _constraint_map(rows: list[Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("backend_constraints entries must be objects")
        axis_id = _text(row.get("axis_id"), "backend_constraints.axis_id")
        out[axis_id] = row
    return out


def _ordered_axis_ids(
    dataflow: Mapping[str, tuple[str, ...]],
    bindings: Mapping[str, list[Mapping[str, Any]]],
) -> list[str]:
    ids = []
    for axis_id in dataflow:
        if axis_id in bindings and axis_id not in ids:
            ids.append(axis_id)
    for axis_id in bindings:
        if axis_id not in ids:
            ids.append(axis_id)
    return ids


def _member_binding(
    binding: Mapping[str, Any],
    prune_groups: Mapping[str, Mapping[str, Any]],
    base_width: int,
    legal_widths: tuple[int, ...],
) -> AxisMemberBinding:
    b1_group_id = _text(binding.get("b1_group_id"), "b1_group_id")
    group = prune_groups.get(b1_group_id)
    if group is None:
        raise ValueError(f"missing prune group for binding {b1_group_id}")
    member_width = _positive_int(group.get("cur_width"), "prune_groups.cur_width")
    divisor = math.gcd(member_width, base_width)
    num = member_width // divisor
    den = base_width // divisor
    for width in legal_widths:
        mapped_numerator = width * num
        if mapped_numerator % den != 0:
            raise ValueError("member transform must produce integer widths")
        mapped_width = mapped_numerator // den
        if mapped_width <= 0 or mapped_width > member_width:
            raise ValueError("member transform exceeds member base width")
    return AxisMemberBinding(
        b1_group_id=b1_group_id,
        module_path=_text(group.get("root_layer", group.get("module_path")),
                          "prune_groups.root_layer"),
        canonical_to_member_num=num,
        canonical_to_member_den=den,
        materializer_param=_text(binding.get("param"), "materializer_bindings.param"),
        role=_text(binding.get("role"), "materializer_bindings.role"),
    )


def _axis_kind(axis_id: str, bindings: list[Mapping[str, Any]]) -> Literal["free", "fixed_derived"]:
    kinds = {str(binding.get("axis_kind") or "free") for binding in bindings}
    if kinds == {"fixed_derived"}:
        return "fixed_derived"
    if kinds <= {"free"}:
        return "free"
    raise ValueError(f"structural axis {axis_id} mixes free and fixed-derived bindings")


def _dense_stage(axis_id: str) -> str | None:
    tail = axis_id.rsplit(".", 1)[-1]
    if tail.startswith("stage") or tail.startswith("s"):
        return tail.replace("s", "stage", 1) if tail[1:].isdigit() else tail
    return None


def _text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


__all__ = ["AxisMemberBinding", "StructuralAxis", "StructuralAxisBundle",
           "axis_to_dict", "axis_to_scanner_dict", "build_structural_axis_inputs",
           "derive_structural_axes"]
