"""Canonical GPU-index policy parsing shared by P6 execution boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


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


def parse_runtime_gpu_pool(environment: Mapping[str, object]) -> int:
    """Read a positive GPU count exclusively from the explicit GPU_POOL key."""
    if not isinstance(environment, Mapping) or "GPU_POOL" not in environment:
        raise ValueError("canonical GPU_POOL required")
    raw_value = environment["GPU_POOL"]
    if (
        not isinstance(raw_value, str)
        or not raw_value.isdigit()
        or str(int(raw_value)) != raw_value
        or int(raw_value) <= 0
    ):
        raise ValueError("canonical GPU_POOL required")
    return int(raw_value)
