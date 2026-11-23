from __future__ import print_function

from dataclasses import dataclass
import math
import warnings
from typing import Optional


@dataclass(frozen=True)
class DualFifoConfig:
    rsu_fifo_delay_ms: Optional[float]
    ego_output_fifo_delay_ms: float
    latency_audit: bool
    uses_new_contract: bool


def _finite_nonnegative(value, field):
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("%s must be finite and non-negative" % field)
    return numeric


def _resolve_alias(simulation, canonical, legacy, default):
    has_canonical = canonical in simulation
    has_legacy = legacy in simulation
    canonical_value = (
        _finite_nonnegative(simulation[canonical], canonical)
        if has_canonical else None
    )
    legacy_value = (
        _finite_nonnegative(simulation[legacy], legacy)
        if has_legacy else None
    )
    if has_canonical and has_legacy and canonical_value != legacy_value:
        raise ValueError("%s conflicts with legacy %s" % (canonical, legacy))
    if has_canonical:
        return canonical_value
    if has_legacy:
        return legacy_value
    return default


def _resolve_bool_alias(simulation, canonical, legacy, default):
    has_canonical = canonical in simulation
    has_legacy = legacy in simulation
    canonical_value = bool(simulation[canonical]) if has_canonical else None
    legacy_value = bool(simulation[legacy]) if has_legacy else None
    if has_canonical and has_legacy and canonical_value != legacy_value:
        raise ValueError("%s conflicts with legacy %s" % (canonical, legacy))
    if has_canonical:
        return canonical_value
    if has_legacy:
        return legacy_value
    return default


def resolve_dual_fifo_config(simulation):
    source = dict(simulation or {})
    legacy_mappings = [
        "%s -> %s" % (legacy, canonical)
        for legacy, canonical in (
            ("latency_inject_ms", "rsu_fifo_delay_ms"),
            ("action_fifo_delay_ms", "ego_output_fifo_delay_ms"),
            ("action_fifo_audit", "latency_audit"),
        ) if legacy in source
    ]
    if legacy_mappings:
        warnings.warn(
            "legacy dual-FIFO keys are deprecated: " + "; ".join(legacy_mappings),
            FutureWarning,
            stacklevel=2,
        )
    uses_new_contract = (
        "rsu_fifo_delay_ms" in source
        or "ego_output_fifo_delay_ms" in source
        or "latency_audit" in source
    )
    rsu_delay = _resolve_alias(
        source, "rsu_fifo_delay_ms", "latency_inject_ms", None
    )
    ego_delay = _resolve_alias(
        source, "ego_output_fifo_delay_ms", "action_fifo_delay_ms", 0.0
    )
    if uses_new_contract and float(source.get("tau_ego_ms", 0.0)) != 0.0:
        raise ValueError(
            "tau_ego_ms must be 0 for the dual-FIFO experiment contract"
        )
    latency_audit = _resolve_bool_alias(
        source, "latency_audit", "action_fifo_audit", False
    )
    return DualFifoConfig(
        rsu_fifo_delay_ms=rsu_delay,
        ego_output_fifo_delay_ms=ego_delay,
        latency_audit=latency_audit,
        uses_new_contract=uses_new_contract,
    )
