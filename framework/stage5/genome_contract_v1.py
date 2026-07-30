"""Schema-driven structure identity for Stage5 search genomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence


MODEL_WIDTH_SCHEMAS: dict[str, tuple[str, ...]] = {
    "pyramid": ("w0", "w1", "w2"),
    "codriving": ("w0", "w1", "w2"),
    "fcooper": (
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ),
}


@dataclass(frozen=True)
class StructureIdentity:
    model: str
    group_id: str
    width: tuple[int, ...]
    width_schema: tuple[str, ...]
    structure_widths: dict[str, int]


def width_schema_for_model(model: str) -> tuple[str, ...]:
    normalized = str(model).lower()
    try:
        return MODEL_WIDTH_SCHEMAS[normalized]
    except KeyError as exc:
        raise ValueError("unsupported target model") from exc


def canonical_group_id(
    model: str, width: Sequence[int], width_schema: Sequence[str] | None = None
) -> str:
    normalized = str(model).lower()
    schema = tuple(width_schema or width_schema_for_model(normalized))
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or int(value) != float(value)
        or int(value) <= 0
        for value in width
    ):
        raise ValueError("structure identity width values must be finite positive integers")
    values = tuple(int(value) for value in width)
    if len(values) != len(schema):
        raise ValueError("structure identity width/schema cardinality mismatch")
    if normalized in {"pyramid", "codriving"}:
        return f"{normalized}|{'x'.join(map(str, values))}"
    return normalized + "|" + "|".join(
        f"{name}={value}" for name, value in zip(schema, values)
    )


def validate_structure_identity(row: Mapping[str, Any]) -> StructureIdentity:
    model = str(row.get("model") or "").lower()
    expected_schema = width_schema_for_model(model)
    supplied_schema = tuple(row.get("width_schema") or expected_schema)
    if supplied_schema != expected_schema:
        raise ValueError("structure identity width_schema mismatch")
    supplied_width = row.get("width") or []
    if not isinstance(supplied_width, Sequence) or isinstance(
        supplied_width, (str, bytes)
    ):
        raise ValueError("structure identity width must be an array")
    try:
        width = tuple(int(value) for value in supplied_width)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "structure identity width values must be finite positive integers"
        ) from exc
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or int(value) != float(value)
        or int(value) <= 0
        for value in supplied_width
    ):
        raise ValueError("structure identity width values must be finite positive integers")
    if len(width) != len(expected_schema):
        raise ValueError("structure identity width/schema cardinality mismatch")
    structure_widths = {
        str(name): int(value)
        for name, value in zip(expected_schema, width)
    }
    supplied_widths = row.get("structure_widths")
    if supplied_widths is not None and dict(supplied_widths) != structure_widths:
        raise ValueError("structure identity named widths mismatch")
    group_id = str(row.get("group_id") or "")
    expected_group_id = canonical_group_id(model, width, expected_schema)
    if group_id != expected_group_id:
        raise ValueError("structure identity group_id mismatch")
    return StructureIdentity(
        model=model,
        group_id=group_id,
        width=width,
        width_schema=expected_schema,
        structure_widths=structure_widths,
    )
