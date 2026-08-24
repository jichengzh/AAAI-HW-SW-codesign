"""Canonical materializer-binding authority validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from framework.stage1.structural_axis_digest import canonical_digest

_BINDING_FIELDS = {
    "b1_group_id",
    "axis_id",
    "param",
    "role",
    "axis_kind",
    "write_targets",
    "derived_from",
    "independent_interface",
}


def seal_materializer_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonical_materializer_binding(row)
    return {**canonical, "materializer_binding_digest": canonical_digest(canonical)}


def canonical_materializer_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    unexpected = set(row) - _BINDING_FIELDS - {"materializer_binding_digest"}
    if unexpected:
        raise ValueError("materializer binding contains unexpected fields")
    canonical = {
        field: _required_text(row.get(field), f"materializer binding {field}")
        for field in ("b1_group_id", "axis_id", "param", "role", "axis_kind")
    }
    canonical["write_targets"] = _canonical_write_targets(row.get("write_targets"))
    for field in ("derived_from", "independent_interface"):
        if row.get(field) is not None:
            canonical[field] = row[field]
    return canonical


def validate_binding_relation_authority(
    sources: Sequence[Mapping[str, Any]],
    derived_rows: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> None:
    projections, source_by_group = _binding_projection_index(sources)
    binding_by_group = {str(row["b1_group_id"]): row for row in bindings}
    derived_by_group = _derived_binding_index(derived_rows)
    if set(projections) != set(binding_by_group) or set(projections) != set(derived_by_group):
        raise ValueError("binding relation authority is not a complete bijection")
    for group_id, projection in projections.items():
        if dict(binding_by_group[group_id]) != projection:
            raise ValueError("binding relation authority disagrees with sealed binding")
        derived = derived_by_group[group_id]
        if derived.get("materializer_binding_digest") != projection[
            "materializer_binding_digest"
        ]:
            raise ValueError("derived binding relation authority digest mismatch")
        source_digest = derived["relation_provenance"].get("source_relation_digest")
        if source_digest != canonical_digest(source_by_group[group_id]):
            raise ValueError("derived binding relation authority source mismatch")


def _canonical_write_targets(value: Any) -> list[dict[str, Any]]:
    rows = _required_sequence(value, "materializer binding write_targets")
    targets = []
    for raw in rows:
        target = _required_mapping(raw, "materializer binding write target")
        if set(target) != {"selector", "transform", "reference_selector"}:
            raise ValueError("materializer binding write target fields are invalid")
        targets.append(
            {
                "selector": _required_text(target.get("selector"), "write selector"),
                "transform": _required_text(target.get("transform"), "write transform"),
                "reference_selector": target.get("reference_selector"),
            }
        )
    return targets


def _binding_projection_index(
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    projections: dict[str, dict[str, Any]] = {}
    source_by_group: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        rows = _required_sequence(
            source.get("materializer_binding_projections"),
            "binding relation authority projections",
        )
        roles = _source_member_roles(source)
        for raw in rows:
            projection = seal_materializer_binding(
                _required_mapping(raw, "binding relation authority projection")
            )
            if dict(raw) != projection:
                raise ValueError("binding relation authority projection digest mismatch")
            _validate_projection_against_source(projection, source, roles)
            group_id = projection["b1_group_id"]
            if group_id in projections:
                raise ValueError("duplicate binding relation authority identity")
            projections[group_id] = projection
            source_by_group[group_id] = source
    return projections, source_by_group


def _source_member_roles(source: Mapping[str, Any]) -> dict[str, str]:
    rows = source.get("member_relations")
    if rows is None:
        return {}
    return {
        _required_text(row.get("group_id"), "source member group_id"): _required_text(
            row.get("role"), "source member role"
        )
        for row in _required_sequence(rows, "source member relations")
    }


def _validate_projection_against_source(
    projection: Mapping[str, Any],
    source: Mapping[str, Any],
    roles: Mapping[str, str],
) -> None:
    group_id = projection["b1_group_id"]
    fields = {
        "axis_id": source.get("canonical_axis_id"),
        "axis_kind": source.get("axis_kind", "free"),
        "derived_from": source.get("derived_from"),
        "independent_interface": source.get("independent_interface"),
    }
    for field, expected in fields.items():
        if projection.get(field) != expected:
            raise ValueError(f"binding relation authority {field} mismatch")
    if roles and projection.get("role") != roles.get(group_id):
        raise ValueError("binding relation authority role mismatch")


def _derived_binding_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _required_mapping(raw, "derived binding relation authority")
        group_id = _required_text(row.get("group_id"), "derived binding group_id")
        _sha256(row.get("materializer_binding_digest"), "materializer binding digest")
        provenance = _required_mapping(
            row.get("relation_provenance"), "derived relation provenance"
        )
        if group_id in indexed:
            raise ValueError("duplicate derived binding relation authority identity")
        indexed[group_id] = {**row, "relation_provenance": provenance}
    return indexed


def _required_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} are required")
    return value


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} is required")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")
    return value


def _sha256(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be canonical SHA256")
    return text
