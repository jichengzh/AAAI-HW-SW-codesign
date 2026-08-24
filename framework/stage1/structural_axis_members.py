"""Exact DepGraph-member resolution for scanner-owned structural axes."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from framework.stage1.structural_axis_contract import (
    formal_scanner_evidence_digest,
    scanner_group_evidence_payload,
)
from framework.stage1.structural_axis_digest import canonical_digest


@dataclass(frozen=True)
class ResolvedAxisMember:
    group: Mapping[str, Any]
    module_path: str
    width: int
    role: str


def require_independent_interface(source: Any, relation: Mapping[str, Any]) -> None:
    """Require graph proof when a free selector writes across an interface."""

    axis_kind = str(relation.get("axis_kind") or "free")
    crosses_interface = _writes_across_config_interface(source)
    if (
        crosses_interface
        and axis_kind == "free"
        and relation.get("independent_interface") is not True
    ):
        raise ValueError(
            f"free axis {source.axis_id} requires independent interface graph evidence"
        )


def validate_derived_references(axes: Sequence[Any]) -> None:
    free_axis_ids = {axis.axis_id for axis in axes if axis.axis_kind == "free"}
    for axis in axes:
        if axis.axis_kind != "fixed_derived":
            continue
        if axis.provenance.get("derived_from") not in free_axis_ids:
            raise ValueError(
                "fixed-derived derived_from must name an existing free structural axis"
            )


def validate_group_manifest(
    groups: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
    scenario: Mapping[str, Any],
    source_relations: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
    """Cross-check purified groups against the scanner manifest declaration."""

    if manifest is None:
        raise ValueError("scanner group manifest provenance is required")
    declared = manifest.get("declared_relevant_group_ids")
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ValueError("group manifest declared relevant groups are required")
    declared_ids = tuple(sorted(_text(item, "declared relevant group") for item in declared))
    actual_ids = tuple(sorted(_text(group.get("group_id"), "DepGraph group_id") for group in groups))
    if declared_ids != actual_ids or manifest.get("source_group_count") != len(actual_ids):
        raise ValueError("prune groups disagree with declared relevant groups")
    digest = _text(manifest.get("relevant_groups_digest"), "relevant groups digest")
    if digest != canonical_digest(declared_ids):
        raise ValueError("declared relevant groups digest mismatch")
    scan_digest = _text(manifest.get("scan_manifest_digest"), "scan manifest digest")
    if len(scan_digest) != 64 or any(char not in "0123456789abcdef" for char in scan_digest):
        raise ValueError("scan manifest digest must be lowercase SHA-256")
    evidence = scanner_group_evidence_payload(
        groups, manifest, scenario, source_relations
    )
    if scan_digest != formal_scanner_evidence_digest(evidence):
        raise ValueError("formal scanner evidence seal mismatch")
    return {"scan_manifest_digest": scan_digest, "relevant_groups_digest": digest}


def _writes_across_config_interface(source: Any) -> bool:
    source_parent = _selector_parent(str(source.config_selector))
    return any(
        _selector_parent(str(target.selector)) != source_parent
        for target in getattr(source, "write_targets", ())
    )


def _selector_parent(selector: str) -> str:
    normalized = selector.replace("[*]", "").split("[", 1)[0]
    return normalized.rsplit(".", 1)[0]


def resolve_axis_members(
    *,
    relation: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    trace_modules: Mapping[str, Any],
    checkpoint_widths: Mapping[str, Any],
    canonical_group: Mapping[str, Any],
    mutation_kind: str,
    canonical_role: str,
    boundary_selectors: Sequence[str] = (),
) -> tuple[ResolvedAxisMember, ...]:
    rows = relation.get("member_relations")
    if rows is None:
        rows = _inferred_member_rows(
            groups, canonical_group, canonical_role, boundary_selectors
        )
    else:
        _validate_manifest_member_declaration(relation, rows)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("dataflow member_relations must be a non-empty sequence")
    members = tuple(
        _resolved_member(
            row, groups, trace_modules, checkpoint_widths, mutation_kind
        )
        for row in rows
    )
    ids = [str(member.group["group_id"]) for member in members]
    if len(ids) != len(set(ids)):
        raise ValueError("dataflow member_relations contain duplicate DepGraph groups")
    canonical_id = str(canonical_group["group_id"])
    if canonical_id not in ids:
        raise ValueError("dataflow member_relations must include the canonical group")
    return tuple(
        sorted(
            members,
            key=lambda member: (
                str(member.group["group_id"]) != canonical_id,
                str(member.group["group_id"]),
            ),
        )
    )


def _validate_manifest_member_declaration(
    relation: Mapping[str, Any], rows: Any
) -> None:
    declared = relation.get("declared_member_group_ids")
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ValueError("explicit relations require declared member groups")
    declared_ids = tuple(sorted(_text(item, "declared member group") for item in declared))
    if len(declared_ids) != len(set(declared_ids)):
        raise ValueError("declared member groups must be unique")
    if not isinstance(rows, Sequence):
        raise ValueError("dataflow member_relations must be a sequence")
    row_ids = tuple(
        sorted(_text(row.get("group_id"), "member_relations.group_id") for row in rows)
    )
    if row_ids != declared_ids:
        raise ValueError("member_relations disagree with declared member groups")
    digest = _text(relation.get("member_relations_digest"), "member relations digest")
    if digest != canonical_digest(declared_ids):
        raise ValueError("declared member groups digest mismatch")


def _inferred_member_rows(
    groups: Sequence[Mapping[str, Any]],
    canonical_group: Mapping[str, Any],
    canonical_role: str,
    boundary_selectors: Sequence[str],
) -> tuple[dict[str, str], ...]:
    seed_id = str(canonical_group["group_id"])
    seed_root = str(canonical_group["root_layer"])
    blocked = {
        str(group["group_id"])
        for group in groups
        if str(group["root_layer"]) != seed_root
        and any(
            fnmatch.fnmatchcase(str(group["root_layer"]), selector)
            for selector in boundary_selectors
        )
    }
    selected = _connected_group_ids(groups, seed_id, blocked)
    return tuple(
        {
            "group_id": group_id,
            "role": canonical_role if group_id == seed_id else "internal",
        }
        for group_id in selected
    )


def _connected_group_ids(
    groups: Sequence[Mapping[str, Any]], seed_id: str, blocked: set[str]
) -> tuple[str, ...]:
    by_id = {str(group["group_id"]): group for group in groups}
    selected = [seed_id]
    for current_id in selected:
        current = by_id[current_id]
        for candidate_id, candidate in by_id.items():
            if candidate_id in selected or candidate_id in blocked:
                continue
            if _groups_share_layer_relation(current, candidate):
                selected.append(candidate_id)
    return tuple(selected)


def _groups_share_layer_relation(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    left_members = {str(item) for item in left.get("member_layers", ())}
    right_members = {str(item) for item in right.get("member_layers", ())}
    return (
        str(left["root_layer"]) in right_members
        or str(right["root_layer"]) in left_members
    )


def _resolved_member(
    row: Any,
    groups: Sequence[Mapping[str, Any]],
    trace_modules: Mapping[str, Any],
    checkpoint_widths: Mapping[str, Any],
    mutation_kind: str,
) -> ResolvedAxisMember:
    if not isinstance(row, Mapping):
        raise ValueError("dataflow member_relations entries must be objects")
    group_id = _text(row.get("group_id"), "member_relations.group_id")
    matches = [group for group in groups if group["group_id"] == group_id]
    if len(matches) != 1:
        raise ValueError(f"member relation must resolve one DepGraph group: {group_id}")
    group = matches[0]
    module_path = _text(group.get("root_layer"), "DepGraph root_layer")
    module = trace_modules.get(module_path)
    if module is None:
        raise ValueError(f"member relation trace module is missing: {module_path}")
    width = _module_width(module, mutation_kind)
    checkpoint_width = _checkpoint_width(checkpoint_widths, module_path)
    depgraph_width = _positive_int(group.get("cur_width"), "DepGraph cur_width")
    if width != checkpoint_width or width != depgraph_width:
        raise ValueError(f"member relation width evidence disagrees for {module_path}")
    return ResolvedAxisMember(
        group=group,
        module_path=module_path,
        width=width,
        role=_text(row.get("role"), "member_relations.role"),
    )


def _module_width(module: Any, mutation_kind: str) -> int:
    attribute = _text(mutation_kind, "mutation_kind")
    if not hasattr(module, attribute):
        raise ValueError(
            f"trace module does not expose mutation width attribute {attribute!r}"
        )
    return _positive_int(getattr(module, attribute), f"trace module {attribute}")


def _checkpoint_width(widths: Mapping[str, Any], module_path: str) -> int:
    if module_path not in widths:
        raise ValueError(f"checkpoint evidence is missing module width for {module_path}")
    return _positive_int(widths[module_path], f"checkpoint evidence for {module_path}")


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
    "ResolvedAxisMember",
    "require_independent_interface",
    "resolve_axis_members",
    "validate_group_manifest",
    "validate_derived_references",
]
