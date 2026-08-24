"""Exact formal-width lattice helpers for scanner-owned structural axes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from framework.stage1.structural_axis_digest import canonical_digest


@dataclass(frozen=True)
class _FormalWidthPolicy:
    max_rate: Fraction


def scenario_rate_policy(
    axis_id: str, policy: Mapping[str, int], scenario_digest: str
) -> dict[str, Any]:
    """Resolve an explicit formal pruning rate from backend policy evidence."""

    numerator = policy.get(
        f"{axis_id}.max_rate_numerator", policy.get("default_max_rate_numerator")
    )
    denominator = policy.get(
        f"{axis_id}.max_rate_denominator", policy.get("default_max_rate_denominator")
    )
    if numerator is None or denominator is None:
        raise ValueError(f"formal max_rate policy is required for axis {axis_id}")
    resolved_numerator = _strict_int(numerator, "numerator")
    resolved_denominator = _strict_int(denominator, "denominator")
    rate = _pruning_rate(Fraction(resolved_numerator, resolved_denominator))
    return {
        "max_rate_numerator": rate.numerator,
        "max_rate_denominator": rate.denominator,
        "max_rate_source": "scan_scenario.backend_policy",
        "scenario_digest": scenario_digest,
    }


def scenario_axis_constraint(
    axis_id: str, scenario: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve all formal width constraints from retained scenario evidence."""

    alignment = scenario.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("scanner scenario alignment is required")
    round_to = alignment.get(
        f"{axis_id}.round_to", alignment.get("default_round_to")
    )
    hardware = alignment.get(
        f"{axis_id}.hardware_alignment",
        alignment.get("default_hardware_alignment", round_to),
    )
    if round_to is None or hardware is None:
        raise ValueError(f"explicit alignment is required for axis {axis_id}")
    return {
        "axis_id": axis_id,
        "round_to": _strict_int(round_to, "round_to"),
        "hardware_alignment": _strict_int(hardware, "hardware_alignment"),
        **scenario_rate_policy(axis_id, alignment, canonical_digest(scenario)),
    }


def validated_width_policy(
    constraint: Mapping[str, Any], axis_id: str, scenario: Mapping[str, Any]
) -> _FormalWidthPolicy:
    expected = scenario_axis_constraint(axis_id, scenario)
    if any(constraint.get(field) != value for field, value in expected.items()):
        raise ValueError("formal max_rate policy disagrees with scanner evidence")
    if "max_rate" in constraint:
        raise ValueError("formal max_rate policy must use numerator and denominator")
    numerator = _strict_int(constraint.get("max_rate_numerator"), "numerator")
    denominator = _strict_int(constraint.get("max_rate_denominator"), "denominator")
    return _FormalWidthPolicy(_pruning_rate(Fraction(numerator, denominator)))


def derive_legal_widths(
    *,
    base_width: int,
    policy: _FormalWidthPolicy,
    round_to: int,
    hardware_alignment: int,
) -> tuple[int, tuple[int, ...]]:
    """Return the effective alignment and complete non-widening width lattice."""

    if not isinstance(policy, _FormalWidthPolicy):
        raise ValueError("verified formal max_rate policy is required")
    rate = policy.max_rate
    effective_round_to = math.lcm(round_to, hardware_alignment)
    unaligned_floor = _ceil_fraction(base_width * (1 - rate))
    first_width = max(
        effective_round_to,
        math.ceil(unaligned_floor / effective_round_to) * effective_round_to,
    )
    widths = tuple(range(first_width, base_width + 1, effective_round_to))
    if not widths or widths[-1] != base_width:
        raise ValueError("base width must align with the formal width lattice")
    return effective_round_to, widths


def _pruning_rate(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("max_rate must be a number in [0, 1)")
    try:
        rate = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("max_rate must be a number in [0, 1)") from exc
    if rate < 0 or rate >= 1:
        raise ValueError("max_rate must be a number in [0, 1)")
    return rate


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"formal max_rate policy {field} must be a positive integer")
    return value


__all__ = [
    "derive_legal_widths",
    "scenario_axis_constraint",
    "scenario_rate_policy",
    "validated_width_policy",
]
