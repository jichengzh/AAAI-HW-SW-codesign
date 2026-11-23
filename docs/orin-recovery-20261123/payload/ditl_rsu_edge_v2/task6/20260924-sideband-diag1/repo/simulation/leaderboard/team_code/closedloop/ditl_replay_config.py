from __future__ import print_function

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class BurstConfig:
    enabled: bool
    enter_rate: float
    exit_rate: float


@dataclass(frozen=True)
class ImpairmentConfig:
    extra_delay_ns: int
    jitter_ns: int
    drop_rate: float
    burst: BurstConfig
    reorder_rate: float
    reorder_extra_ns: int
    duplicate_rate: float


@dataclass(frozen=True)
class DitlReplayConfig:
    library_root: Path
    catalog: Path
    bundle: Path
    exhaustion: str
    seed: int
    ttl_ns: Optional[int]
    deadline_ns: Optional[int]
    max_queue_events: int
    impairment: ImpairmentConfig


class DitlReplayConfigError(ValueError):
    """Raised before bundle access when replay configuration is invalid."""


_REPLAY_FIELDS = frozenset((
    "library_root",
    "catalog",
    "bundle",
    "exhaustion",
    "seed",
    "ttl_ms",
    "deadline_ms",
    "max_queue_events",
    "impairment",
))
_IMPAIRMENT_FIELDS = frozenset((
    "extra_delay_ms",
    "jitter_ms",
    "drop_rate",
    "burst",
    "reorder_rate",
    "reorder_extra_ms",
    "duplicate_rate",
))
_BURST_FIELDS = frozenset((
    "enabled",
    "enter_rate",
    "exit_rate",
))
_LEGACY_RSU_FIELDS = (
    "comm_latency",
    "rsu_fifo_delay_ms",
    "latency_inject_ms",
    "latency_trace",
)


def _mapping(value, field):
    if not isinstance(value, MappingABC):
        raise DitlReplayConfigError("%s must be a mapping" % field)
    return dict(value)


def _validate_fields(source, expected, field):
    unknown = sorted(set(source) - expected, key=repr)
    if unknown:
        raise DitlReplayConfigError(
            "%s contains unknown field %s" % (field, unknown[0])
        )
    missing = sorted(expected - set(source))
    if missing:
        raise DitlReplayConfigError(
            "%s is missing required field %s" % (field, missing[0])
        )


def _path(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise DitlReplayConfigError("%s must be an absolute path" % field)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise DitlReplayConfigError("%s must be an absolute path" % field)
    return path


def _integer(value, field, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise DitlReplayConfigError("%s must be an integer" % field)
    if positive and value <= 0:
        raise DitlReplayConfigError("%s must be a positive integer" % field)
    return value


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DitlReplayConfigError("%s must be a finite number" % field)
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise DitlReplayConfigError("%s must be a finite number" % field)
    if not math.isfinite(numeric):
        raise DitlReplayConfigError("%s must be a finite number" % field)
    return numeric


def _rate(value, field):
    numeric = _number(value, field)
    if numeric < 0.0 or numeric > 1.0:
        raise DitlReplayConfigError("%s must be in [0, 1]" % field)
    return numeric


def _milliseconds(value, field, optional=False):
    if optional and value is None:
        return None
    numeric = _number(value, field)
    if numeric < 0.0:
        raise DitlReplayConfigError(
            "%s must be finite and non-negative" % field
        )
    nanoseconds = numeric * 1000000
    if not math.isfinite(nanoseconds):
        raise DitlReplayConfigError("%s is too large" % field)
    return int(round(nanoseconds))


def _parse_burst(value):
    source = _mapping(value, "burst")
    _validate_fields(source, _BURST_FIELDS, "burst")
    enabled = source["enabled"]
    if not isinstance(enabled, bool):
        raise DitlReplayConfigError("enabled must be a boolean")
    return BurstConfig(
        enabled=enabled,
        enter_rate=_rate(source["enter_rate"], "enter_rate"),
        exit_rate=_rate(source["exit_rate"], "exit_rate"),
    )


def _parse_impairment(value):
    source = _mapping(value, "impairment")
    _validate_fields(source, _IMPAIRMENT_FIELDS, "impairment")
    return ImpairmentConfig(
        extra_delay_ns=_milliseconds(
            source["extra_delay_ms"], "extra_delay_ms"
        ),
        jitter_ns=_milliseconds(source["jitter_ms"], "jitter_ms"),
        drop_rate=_rate(source["drop_rate"], "drop_rate"),
        burst=_parse_burst(source["burst"]),
        reorder_rate=_rate(source["reorder_rate"], "reorder_rate"),
        reorder_extra_ns=_milliseconds(
            source["reorder_extra_ms"], "reorder_extra_ms"
        ),
        duplicate_rate=_rate(source["duplicate_rate"], "duplicate_rate"),
    )


def resolve_ditl_replay_config(
        simulation: Mapping[str, object]) -> Optional[DitlReplayConfig]:
    """Return None when disabled; otherwise return a fully validated copy."""
    if not isinstance(simulation, MappingABC):
        raise DitlReplayConfigError("simulation must be a mapping")
    if "ditl_replay" not in simulation:
        return None

    for field in _LEGACY_RSU_FIELDS:
        if field in simulation:
            raise DitlReplayConfigError(
                "ditl_replay conflicts with legacy %s" % field
            )

    source = _mapping(simulation["ditl_replay"], "ditl_replay")
    _validate_fields(source, _REPLAY_FIELDS, "ditl_replay")

    library_root = _path(source["library_root"], "library_root")
    catalog = _path(source["catalog"], "catalog")
    bundle = _path(source["bundle"], "bundle")
    if catalog != library_root / "catalog.json":
        raise DitlReplayConfigError(
            "catalog must equal library_root/catalog.json"
        )
    if bundle.parent != library_root / "v1" / "bundles":
        raise DitlReplayConfigError(
            "bundle must be a direct child of library_root/v1/bundles"
        )

    exhaustion = source["exhaustion"]
    if not isinstance(exhaustion, str) or exhaustion not in ("error", "cycle"):
        raise DitlReplayConfigError(
            "exhaustion must be either error or cycle"
        )

    return DitlReplayConfig(
        library_root=library_root,
        catalog=catalog,
        bundle=bundle,
        exhaustion=exhaustion,
        seed=_integer(source["seed"], "seed"),
        ttl_ns=_milliseconds(source["ttl_ms"], "ttl_ms", optional=True),
        deadline_ns=_milliseconds(
            source["deadline_ms"], "deadline_ms", optional=True
        ),
        max_queue_events=_integer(
            source["max_queue_events"], "max_queue_events", positive=True
        ),
        impairment=_parse_impairment(source["impairment"]),
    )
