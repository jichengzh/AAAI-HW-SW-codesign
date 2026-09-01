"""Public-safe provenance rules for hardware-specific P6 reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Literal

from framework.stage6.hardware_execution_profile_v1 import (
    load_hardware_execution_profile,
    validate_profile_backend,
)


REPORT_SCHEMA_VERSION = "p6_hardware_specific_report_provenance_v1"
COMPARISON_SCOPE = "hardware_specific"
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "comparison_scope",
        "hardware_profile",
        "target",
        "execution_backend",
        "tvm_arch",
        "latency_energy_hardware_profile",
        "pareto_hardware_profile",
        "ap_provenance",
    }
)
_AP_PROVENANCE_KEYS = frozenset(
    {"data_split", "checkpoint_initial_state", "seed", "metric_protocol"}
)
_PUBLIC_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class P6PublicReportError(ValueError):
    """Stable public error for invalid P6 report provenance."""


@dataclass(frozen=True)
class P6HardwareSpecificReportProvenance:
    """Redacted labels that bind performance and Pareto evidence to hardware."""

    schema_version: Literal["p6_hardware_specific_report_provenance_v1"]
    comparison_scope: Literal["hardware_specific"]
    hardware_profile: str
    target: str
    execution_backend: str
    tvm_arch: str
    latency_energy_hardware_profile: str
    pareto_hardware_profile: str
    ap_provenance: Mapping[str, str | int]


def validate_hardware_specific_report_provenance(
    raw_report: Mapping[str, object],
) -> P6HardwareSpecificReportProvenance:
    """Validate redacted report labels without accepting metric values or paths."""
    if not isinstance(raw_report, Mapping) or set(raw_report) != _REPORT_KEYS:
        raise P6PublicReportError("report provenance fields are invalid")
    if raw_report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise P6PublicReportError("report provenance schema is invalid")
    if raw_report.get("comparison_scope") != COMPARISON_SCOPE:
        raise P6PublicReportError("comparison scope must be hardware_specific")

    profile_id = _identifier(raw_report.get("hardware_profile"), "hardware profile")
    try:
        profile = load_hardware_execution_profile(profile_id)
    except ValueError:
        raise P6PublicReportError("hardware profile is invalid") from None
    target = _identifier(raw_report.get("target"), "target")
    if target != profile.target_hardware_id:
        raise P6PublicReportError("target does not match the hardware profile")
    backend = _identifier(raw_report.get("execution_backend"), "execution backend")
    try:
        validate_profile_backend(profile, backend)
    except ValueError:
        raise P6PublicReportError(
            "execution backend does not match the hardware profile"
        ) from None
    tvm_arch = _identifier(raw_report.get("tvm_arch"), "TVM architecture")
    if tvm_arch != profile.tvm_arch:
        raise P6PublicReportError("TVM architecture does not match the hardware profile")

    latency_energy_profile = raw_report.get("latency_energy_hardware_profile")
    if latency_energy_profile != profile.profile_id:
        raise P6PublicReportError(
            "latency and energy evidence must remain hardware-specific"
        )
    pareto_profile = raw_report.get("pareto_hardware_profile")
    if pareto_profile != profile.profile_id:
        raise P6PublicReportError("Pareto evidence must remain hardware-specific")
    ap_provenance = _ap_provenance(raw_report.get("ap_provenance"))

    return P6HardwareSpecificReportProvenance(
        schema_version=REPORT_SCHEMA_VERSION,
        comparison_scope=COMPARISON_SCOPE,
        hardware_profile=profile.profile_id,
        target=profile.target_hardware_id,
        execution_backend=backend,
        tvm_arch=profile.tvm_arch,
        latency_energy_hardware_profile=profile.profile_id,
        pareto_hardware_profile=profile.profile_id,
        ap_provenance=MappingProxyType(ap_provenance),
    )


def validate_cross_hardware_ap_provenance(
    raw_reports: Sequence[Mapping[str, object]],
) -> Mapping[str, str | int]:
    """Return shared AP provenance only for comparable cross-hardware reports."""
    if (
        not isinstance(raw_reports, Sequence)
        or isinstance(raw_reports, (str, bytes))
        or len(raw_reports) < 2
    ):
        raise P6PublicReportError("cross-hardware AP comparison requires reports")
    reports = [
        validate_hardware_specific_report_provenance(report)
        for report in raw_reports
    ]
    if len({report.hardware_profile for report in reports}) < 2:
        raise P6PublicReportError("cross-hardware AP comparison requires distinct profiles")
    expected = dict(reports[0].ap_provenance)
    if any(dict(report.ap_provenance) != expected for report in reports[1:]):
        raise P6PublicReportError("cross-hardware AP provenance does not match")
    return MappingProxyType(expected)


def _ap_provenance(value: object) -> dict[str, str | int]:
    if not isinstance(value, Mapping) or set(value) != _AP_PROVENANCE_KEYS:
        raise P6PublicReportError("AP provenance fields are invalid")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise P6PublicReportError("AP provenance seed is invalid")
    return {
        "data_split": _identifier(value.get("data_split"), "AP data split"),
        "checkpoint_initial_state": _identifier(
            value.get("checkpoint_initial_state"), "AP checkpoint initial state"
        ),
        "seed": seed,
        "metric_protocol": _identifier(
            value.get("metric_protocol"), "AP metric protocol"
        ),
    }


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _PUBLIC_IDENTIFIER.fullmatch(value) is None:
        raise P6PublicReportError(f"{field} is invalid")
    return value
