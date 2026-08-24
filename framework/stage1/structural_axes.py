"""Scanner-owned structural-axis derivation for formal search spaces."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

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


@dataclass(frozen=True)
class StructuralAxisBundle:
    axes: tuple[StructuralAxis, ...]
    diagnostics: Mapping[str, object]

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


def build_structural_axis_inputs(
    *,
    trace_context: "TraceContext",
    prune_groups: Sequence[Mapping[str, Any]],
    scenario: "ScanScenario",
) -> dict[str, Any]:
    """Resolve adapter selectors into scanner-owned structural-axis evidence."""

    sources = tuple(trace_context.materializer_sources)
    if not sources:
        raise ValueError("at least one adapter materializer binding is required")
    _validate_materializer_sources(sources)
    groups = _normalized_depgraph_groups(prune_groups)
    checkpoint_digest, checkpoint_widths = _checkpoint_width_evidence(
        trace_context.checkpoint_evidence
    )
    resolved = tuple(
        _resolve_materializer_source(source, trace_context, groups, checkpoint_widths)
        for source in sources
    )
    _reject_duplicate_physical_axes(resolved)
    payload: dict[str, Any] = {
        "prune_groups": [_resolved_prune_group(item) for item in resolved],
        "dataflow_relations": [_resolved_relation(item) for item in resolved],
        "materializer_bindings": [_resolved_binding(item) for item in resolved],
        "base_widths": [_resolved_base_width(item) for item in resolved],
        "backend_constraints": [
            _axis_constraint(item.source.axis_id, scenario.alignment, item.group)
            for item in resolved
        ],
        "provenance": {
            "source": "stage1.graph_scan.build_structural_axis_inputs",
            "checkpoint_digest": checkpoint_digest,
            "config_digest": _sha256_json(trace_context.loaded_config),
            "selectors": [_resolved_provenance(item) for item in resolved],
            "scenario_digest": _sha256_json(_scenario_payload(scenario)),
        },
    }
    payload["digest"] = _sha256_json(payload)
    return payload


def _resolve_materializer_source(
    source: Any,
    context: "TraceContext",
    groups: Sequence[Mapping[str, Any]],
    checkpoint_widths: Mapping[str, Any],
) -> _ResolvedMaterializerSource:
    config_path, base_width = _resolve_config_base(
        context.loaded_config, source.config_selector
    )
    module_path, module = _resolve_trace_module(
        context.trace_modules, source.module_root_selector
    )
    module_width = _module_width(module, source.mutation_kind)
    checkpoint_width = _checkpoint_module_width(checkpoint_widths, module_path)
    group = _matching_depgraph_group(groups, module_path)
    depgraph_width = _positive_int(group.get("cur_width"), "DepGraph cur_width")
    _require_equal_widths(
        base_width=base_width,
        module_width=module_width,
        checkpoint_width=checkpoint_width,
        depgraph_width=depgraph_width,
        axis_id=source.axis_id,
    )
    relation = _matching_dataflow_relation(
        context.dataflow_relations, source.axis_id, module_path, group["group_id"]
    )
    return _ResolvedMaterializerSource(
        source=source,
        config_path=config_path,
        base_width=base_width,
        module_path=module_path,
        depgraph_width=depgraph_width,
        group=group,
        relation=relation,
        role=_materializer_role(source.allowed_roles),
    )


def _resolved_prune_group(item: _ResolvedMaterializerSource) -> dict[str, Any]:
    return {
        "id": item.group["group_id"],
        "module_path": item.module_path,
        "cur_width": item.depgraph_width,
        "dataflow_group_id": item.source.axis_id,
    }


def _resolved_relation(item: _ResolvedMaterializerSource) -> dict[str, Any]:
    axis_kind = str(item.relation.get("axis_kind") or "free")
    return {
        "group_id": item.source.axis_id,
        "canonical_axis_id": item.source.axis_id,
        **({"axis_kind": axis_kind} if axis_kind != "free" else {}),
    }


def _resolved_binding(item: _ResolvedMaterializerSource) -> dict[str, Any]:
    binding = {
        "b1_group_id": item.group["group_id"],
        "axis_id": item.source.axis_id,
        "param": item.config_path,
        "role": item.role,
        "axis_kind": str(item.relation.get("axis_kind") or "free"),
        "write_targets": [
            {
                "selector": target.selector,
                "transform": target.transform,
                "reference_selector": target.reference_selector,
            }
            for target in item.source.write_targets
        ],
    }
    if item.relation.get("derived_from") is not None:
        binding["derived_from"] = str(item.relation["derived_from"])
    return binding


def _resolved_base_width(item: _ResolvedMaterializerSource) -> dict[str, Any]:
    return {
        "axis_id": item.source.axis_id,
        "width": item.base_width,
        "config_path": item.config_path,
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
) -> tuple[str, Mapping[str, Any]]:
    digest = str(checkpoint_evidence.get("digest") or "")
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("checkpoint evidence digest is required")
    widths = checkpoint_evidence.get("module_widths")
    if not isinstance(widths, Mapping) or not widths:
        raise ValueError("checkpoint evidence module_widths are required")
    return digest, widths


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


def _axis_constraint(
    axis_id: str,
    alignment: Mapping[str, int],
    group: Mapping[str, Any],
) -> dict[str, Any]:
    round_to = alignment.get(
        f"{axis_id}.round_to",
        alignment.get("default_round_to"),
    )
    min_width = alignment.get(
        f"{axis_id}.min_width",
        alignment.get("default_min_width"),
    )
    if round_to is None or min_width is None:
        raise ValueError(
            f"explicit alignment round_to and min_width are required for axis {axis_id}"
        )
    resolved_round_to = _positive_int(round_to, f"{axis_id} round_to")
    resolved_min_width = _positive_int(min_width, f"{axis_id} min_width")
    group_round_to = group.get("round_to_default")
    if group_round_to is not None and resolved_round_to % _positive_int(
        group_round_to, "DepGraph round_to_default"
    ) != 0:
        raise ValueError(
            f"scenario alignment conflicts with DepGraph alignment for axis {axis_id}"
        )
    return {
        "axis_id": axis_id,
        "round_to": resolved_round_to,
        "min_width": resolved_min_width,
    }


def _scenario_payload(scenario: "ScanScenario") -> dict[str, Any]:
    return {
        "hardware_precisions": list(scenario.hardware_precisions),
        "backend_precisions": list(scenario.backend_precisions),
        "compression_modes": list(scenario.compression_modes),
        "graph_quant_unit_policy": {
            key: list(value)
            for key, value in sorted(scenario.graph_quant_unit_policy.items())
        },
        "alignment": dict(sorted(scenario.alignment.items())),
    }


def _sha256_json(value: Any) -> str:
    primitive = _canonical_json_primitive(value)
    encoded = json.dumps(
        primitive,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_primitive(value: Any, path: str = "$") -> Any:
    if isinstance(value, Mapping):
        normalized = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            normalized[key] = _canonical_json_primitive(child, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_primitive(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{path} must contain only JSON primitive values")


def derive_structural_axes(raw_scan: Mapping[str, Any]) -> StructuralAxisBundle:
    inputs = raw_scan.get("structural_axis_inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("structural_axis_inputs are required")
    prune_groups = _by_id(_required_list(inputs, "prune_groups"), "id")
    dataflow = _dataflow_axes(_required_list(inputs, "dataflow_relations"))
    base_widths = _width_map(_required_base_widths(inputs), "base_widths")
    constraints = _constraint_map(_required_backend_constraints(inputs))
    bindings = _required_list(inputs, "materializer_bindings")

    axis_bindings = _bindings_by_axis(bindings)
    axes = [
        _derive_axis(axis_id, axis_bindings, dataflow, base_widths,
                     constraints, prune_groups)
        for axis_id in _ordered_axis_ids(dataflow, axis_bindings)
    ]
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
) -> StructuralAxis:
    bindings = axis_bindings.get(axis_id, [])
    base_width = _required_axis_value(base_widths, axis_id, "canonical base width")
    constraint = _required_axis_value(constraints, axis_id, "backend constraint")
    round_to = _positive_int(constraint.get("round_to"), "round_to")
    min_width = _positive_int(constraint.get("min_width"), "min_width")
    legal_widths = _legal_widths(base_width, round_to, min_width)
    members = tuple(
        _member_binding(binding, prune_groups, base_width, legal_widths)
        for binding in bindings
    )
    if not members:
        raise ValueError(f"structural axis {axis_id} has no member bindings")
    axis_kind = _axis_kind(axis_id, bindings)
    provenance = _axis_provenance(axis_id, axis_kind, bindings,
                                  dataflow.get(axis_id, ()))
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


def _axis_provenance(
    axis_id: str,
    axis_kind: Literal["free", "fixed_derived"],
    bindings: Sequence[Mapping[str, Any]],
    dataflow_group_ids: tuple[str, ...],
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "source": "structural_axis_inputs",
        "dataflow_group_ids": dataflow_group_ids,
    }
    if axis_kind != "fixed_derived":
        return provenance
    derived_from = {
        _text(binding.get("derived_from"), "fixed-derived binding derived_from")
        for binding in bindings
    }
    if len(derived_from) != 1:
        raise ValueError(
            f"fixed-derived axis {axis_id} must have one derived_from source"
        )
    provenance["derived_from"] = next(iter(derived_from))
    return provenance


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
        "provenance": dict(axis.provenance),
    }


def axis_to_scanner_dict(
    axis: StructuralAxis,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    payload = axis_to_dict(axis)
    source = inputs["provenance"]
    return {
        **payload,
        "provenance": {
            **dict(payload["provenance"]),
            "source": source["source"],
            "input_digest": inputs["digest"],
            "checkpoint_digest": source["checkpoint_digest"],
            "config_digest": source["config_digest"],
            "scenario_digest": source["scenario_digest"],
        },
    }


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


def _by_id(rows: list[Any], key: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{key} rows must be objects")
        row_id = _text(row.get(key), key)
        if row_id in out:
            raise ValueError(f"duplicate {key}: {row_id}")
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
    if member_width % base_width != 0:
        raise ValueError("member width must form an integer ratio to canonical base")
    num = member_width // base_width
    den = 1
    for width in legal_widths:
        if width * num % den != 0:
            raise ValueError("member transform must produce integer widths")
    return AxisMemberBinding(
        b1_group_id=b1_group_id,
        module_path=_text(group.get("module_path"), "prune_groups.module_path"),
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


def _legal_widths(base_width: int, round_to: int, min_width: int) -> tuple[int, ...]:
    if min_width > base_width:
        raise ValueError("min_width exceeds base_width")
    widths = tuple(
        width
        for width in range(min_width, base_width + 1, round_to)
        if width % round_to == 0
    )
    if not widths:
        raise ValueError("free structural axis has no legal widths")
    return widths


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


__all__ = [
    "AxisMemberBinding",
    "StructuralAxis",
    "StructuralAxisBundle",
    "axis_to_dict",
    "axis_to_scanner_dict",
    "build_structural_axis_inputs",
    "derive_structural_axes",
]
