"""Scanner-owned materializer-binding authority construction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from framework.stage1.structural_axis_contract import (
    canonical_source_dataflow_relations,
    seal_dataflow_relation,
    seal_materializer_binding,
)
from framework.stage1.structural_axis_digest import canonical_digest


def resolved_binding_authority(
    resolved: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bindings_by_axis = {
        item.source.axis_id: [_binding_row(item, member) for member in item.members]
        for item in resolved
    }
    raw_sources = canonical_source_dataflow_relations(
        [
            _source_relation(item, bindings_by_axis[item.source.axis_id])
            for item in resolved
        ]
    )
    sources_by_axis = {row["canonical_axis_id"]: row for row in raw_sources}
    bindings = [
        binding
        for item in resolved
        for binding in bindings_by_axis[item.source.axis_id]
    ]
    derived = [
        _derived_relation(item, binding, sources_by_axis[item.source.axis_id])
        for item in resolved
        for binding in bindings_by_axis[item.source.axis_id]
    ]
    return raw_sources, derived, bindings


def _source_relation(
    item: Any, bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    relation = dict(item.relation)
    if relation.get("member_relations") is None:
        members = tuple(
            sorted(
                (
                    {
                        "group_id": str(member.group["group_id"]),
                        "role": member.role,
                    }
                    for member in item.members
                ),
                key=lambda row: row["group_id"],
            )
        )
        group_ids = tuple(row["group_id"] for row in members)
        relation = {
            **relation,
            "member_relations_source": "scanner_inferred_nearest_boundary_v1",
            "member_relations": members,
            "declared_member_group_ids": group_ids,
            "member_relations_digest": canonical_digest(group_ids),
        }
    else:
        relation = {
            **relation,
            "member_relations_source": "adapter_declared_v1",
        }
    return {
        **relation,
        "canonical_group_id": item.group["group_id"],
        "materializer_binding_projections": list(bindings),
    }


def _binding_row(item: Any, member: Any) -> dict[str, Any]:
    row = {
        "b1_group_id": member.group["group_id"],
        "axis_id": item.source.axis_id,
        "param": item.config_path,
        "role": member.role,
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
    for field in ("derived_from", "independent_interface"):
        if item.relation.get(field) is not None:
            row[field] = item.relation[field]
    return seal_materializer_binding(row)


def _derived_relation(
    item: Any, binding: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    axis_kind = str(item.relation.get("axis_kind") or "free")
    row = {
        "group_id": binding["b1_group_id"],
        "canonical_axis_id": item.source.axis_id,
        "materializer_binding_digest": binding["materializer_binding_digest"],
        **({"axis_kind": axis_kind} if axis_kind != "free" else {}),
    }
    if item.relation.get("independent_interface") is True:
        row["independent_interface"] = True
    return seal_dataflow_relation(row, source)


__all__ = ["resolved_binding_authority"]
