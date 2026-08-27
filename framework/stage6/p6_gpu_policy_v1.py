"""Canonical GPU-index policy parsing shared by P6 execution boundaries."""

from __future__ import annotations

from collections.abc import Sequence


def canonical_gpu_indices(raw_indices: object) -> tuple[int, ...]:
    """Return a non-empty, ordered, unique tuple of non-negative integers."""
    if (
        not isinstance(raw_indices, Sequence)
        or isinstance(raw_indices, (str, bytes))
        or not raw_indices
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in raw_indices
        )
    ):
        raise ValueError("canonical GPU indices required")
    indices = tuple(raw_indices)
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("canonical GPU indices required")
    return indices


def parse_gpu_indices_csv(raw_value: object) -> tuple[int, ...]:
    """Parse an exact comma-separated representation of canonical GPU indices."""
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError("canonical GPU indices required")
    parts = tuple(raw_value.split(","))
    if any(not part.isdigit() or str(int(part)) != part for part in parts):
        raise ValueError("canonical GPU indices required")
    return canonical_gpu_indices(tuple(int(part) for part in parts))
