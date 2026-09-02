"""Canonical retained-group partition for materializer axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def canonical_member_partition(
    groups: Sequence[Mapping[str, Any]],
    boundary_group_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Assign every retained group to its unique nearest materializer boundary."""

    by_id = _group_index(groups)
    boundaries = tuple(str(group_id) for group_id in boundary_group_ids)
    if not boundaries or len(boundaries) != len(set(boundaries)):
        raise ValueError("materializer boundary groups must be non-empty and unique")
    if any(group_id not in by_id for group_id in boundaries):
        raise ValueError("materializer boundary is not a retained group")
    distances = {
        boundary_id: _group_distances(by_id, boundary_id)
        for boundary_id in boundaries
    }
    assignments = {boundary_id: [] for boundary_id in boundaries}
    for group_id in by_id:
        reachable = {
            boundary_id: by_group[group_id]
            for boundary_id, by_group in distances.items()
            if group_id in by_group
        }
        if not reachable:
            raise ValueError("retained group has no materializer axis")
        nearest_distance = min(reachable.values())
        nearest = tuple(
            boundary_id
            for boundary_id, distance in reachable.items()
            if distance == nearest_distance
        )
        if len(nearest) != 1:
            raise ValueError(
                "retained group is equidistant between materializer axes"
            )
        owner = nearest[0]
        assignments = {
            **assignments,
            owner: [*assignments[owner], group_id],
        }
    return {
        boundary_id: tuple(
            sorted(
                group_ids,
                key=lambda group_id: (group_id != boundary_id, group_id),
            )
        )
        for boundary_id, group_ids in assignments.items()
    }


def _group_index(
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_id = {str(group.get("group_id") or ""): group for group in groups}
    if "" in by_id or len(by_id) != len(groups):
        raise ValueError("retained group identities must be non-empty and unique")
    return by_id


def _group_distances(
    groups: Mapping[str, Mapping[str, Any]], seed_id: str
) -> dict[str, int]:
    distances = {seed_id: 0}
    frontier = (seed_id,)
    while frontier:
        additions = {
            candidate_id: distances[current_id] + 1
            for current_id in frontier
            for candidate_id, candidate in groups.items()
            if candidate_id not in distances
            and _groups_share_member_layer(groups[current_id], candidate)
        }
        frontier = tuple(additions)
        distances = {**distances, **additions}
    return distances


def _groups_share_member_layer(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    left_members = {
        str(left.get("root_layer", left.get("module_path")) or ""),
        *(str(item) for item in left.get("member_layers", ())),
    }
    right_members = {
        str(right.get("root_layer", right.get("module_path")) or ""),
        *(str(item) for item in right.get("member_layers", ())),
    }
    return bool(left_members & right_members)


__all__ = ["canonical_member_partition"]
