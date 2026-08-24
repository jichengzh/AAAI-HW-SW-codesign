"""Canonical base-width proof validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from framework.stage1.structural_axis_digest import canonical_digest


def validate_base_width_authority(
    rows: Sequence[Mapping[str, Any]],
    retained_groups: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    source_relations: Sequence[Mapping[str, Any]],
) -> None:
    groups = {_group_id(row): row for row in retained_groups}
    sources = _unique_by_axis(source_relations, "raw source relation")
    binding_groups = {str(row["b1_group_id"]): row for row in bindings}
    for row in rows:
        _validate_base_row(row, groups, binding_groups, sources)
    if {row["axis_id"] for row in rows} != {row["axis_id"] for row in bindings}:
        raise ValueError("base width authority axes disagree with binding authority")


def _validate_base_row(
    row: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> None:
    axis_id = _required_text(row.get("axis_id"), "base width authority axis_id")
    source = sources.get(axis_id)
    if source is None:
        raise ValueError("base width authority has no unique raw source relation")
    group_id = _required_text(
        row.get("canonical_group_id"), "base width authority canonical group"
    )
    if group_id != source.get("canonical_group_id"):
        raise ValueError("base width authority canonical seed mismatch")
    group = groups.get(group_id)
    binding = bindings.get(group_id)
    if group is None or binding is None or binding.get("axis_id") != axis_id:
        raise ValueError("base width authority does not match binding authority")
    _validate_coordinates(row, group, binding, source)
    _validate_widths(row, group)


def _validate_coordinates(
    row: Mapping[str, Any],
    group: Mapping[str, Any],
    binding: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    group_path = _required_text(
        group.get("root_layer", group.get("module_path")),
        "base width authority canonical group path",
    )
    expected = {
        "checkpoint_module_path": group_path,
        "config_path": binding.get("param"),
        "source_relation_digest": canonical_digest(source),
        "materializer_binding_digest": binding.get("materializer_binding_digest"),
    }
    if source.get("module_root_selector") != group_path:
        raise ValueError("base width authority source selector mismatch")
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"base width authority {field} mismatch")


def _validate_widths(
    row: Mapping[str, Any], group: Mapping[str, Any]
) -> None:
    widths = tuple(
        _positive_int(row.get(field), f"base width authority {field}")
        for field in (
            "width",
            "config_width",
            "checkpoint_width",
            "canonical_group_width",
        )
    )
    retained = _positive_int(
        group.get("cur_width"), "base width authority retained group width"
    )
    if len(set((*widths, retained))) != 1:
        raise ValueError("base width authority sources disagree")


def _unique_by_axis(
    rows: Sequence[Mapping[str, Any]], description: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        axis_id = _required_text(row.get("canonical_axis_id"), description)
        if axis_id in indexed:
            raise ValueError(f"base width authority duplicate {description}")
        indexed[axis_id] = row
    return indexed


def _group_id(row: Mapping[str, Any]) -> str:
    return _required_text(
        row.get("group_id", row.get("id")), "base width authority retained group"
    )


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


__all__ = ["validate_base_width_authority"]
