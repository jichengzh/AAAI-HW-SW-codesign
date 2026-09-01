"""Public-safe provenance rules for hardware-specific P6 reports."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Literal

from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    load_hardware_execution_profile,
    validate_profile_backend,
)


REPORT_SCHEMA_VERSION = "p6_hardware_specific_report_provenance_v2"
COMPARISON_SCOPE = "hardware_specific"
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "comparison_scope",
        "hardware_profile",
        "target",
        "target_model",
        "hardware_model_family",
        "gpu_count",
        "execution_backend",
        "tvm_arch",
        "tvm_cache_namespace",
        "environment_digest",
        "code_revision",
        "source_digest",
        "compiler_toolchain_digest",
        "latency_energy_hardware_profile",
        "pareto_hardware_profile",
        "ap_provenance",
    }
)
_AP_PROVENANCE_KEYS = frozenset(
    {
        "data_split",
        "checkpoint_initial_state",
        "training_config_digest",
        "seed",
        "metric_protocol",
    }
)
_PUBLIC_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class P6PublicReportError(ValueError):
    """Stable public error for invalid P6 report provenance."""


@dataclass(frozen=True)
class P6HardwareSpecificReportProvenance:
    """Redacted labels that bind performance and Pareto evidence to hardware."""

    schema_version: Literal["p6_hardware_specific_report_provenance_v2"]
    comparison_scope: Literal["hardware_specific"]
    hardware_profile: str
    target: str
    target_model: str
    hardware_model_family: str
    gpu_count: int
    execution_backend: str
    tvm_arch: str
    tvm_cache_namespace: str
    environment_digest: str
    code_revision: str
    source_digest: str
    compiler_toolchain_digest: str
    latency_energy_hardware_profile: str
    pareto_hardware_profile: str
    ap_provenance: P6APProvenance


@dataclass(frozen=True, eq=False)
class P6APProvenance(Mapping[str, str | int]):
    """Immutable, asdict-compatible external provenance for AP comparison."""

    data_split: str
    checkpoint_initial_state: str
    training_config_digest: str
    seed: int
    metric_protocol: str

    def __getitem__(self, key: str) -> str | int:
        if key not in _AP_PROVENANCE_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "data_split",
                "checkpoint_initial_state",
                "training_config_digest",
                "seed",
                "metric_protocol",
            )
        )

    def __len__(self) -> int:
        return len(_AP_PROVENANCE_KEYS)


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

    (
        profile,
        target_model,
        hardware_model_family,
        gpu_count,
        backend,
        cache_namespace,
    ) = _validated_hardware_fields(raw_report)
    environment_digest, code_revision, source_digest, toolchain_digest = (
        _validated_runtime_fields(raw_report)
    )

    _validate_hardware_specific_profiles(raw_report, profile)
    ap_provenance = _ap_provenance(raw_report.get("ap_provenance"))

    return P6HardwareSpecificReportProvenance(
        schema_version=REPORT_SCHEMA_VERSION,
        comparison_scope=COMPARISON_SCOPE,
        hardware_profile=profile.profile_id,
        target=profile.target_hardware_id,
        target_model=target_model,
        hardware_model_family=hardware_model_family,
        gpu_count=gpu_count,
        execution_backend=backend,
        tvm_arch=profile.tvm_arch,
        tvm_cache_namespace=cache_namespace,
        environment_digest=environment_digest,
        code_revision=code_revision,
        source_digest=source_digest,
        compiler_toolchain_digest=toolchain_digest,
        latency_energy_hardware_profile=profile.profile_id,
        pareto_hardware_profile=profile.profile_id,
        ap_provenance=ap_provenance,
    )


def _validate_hardware_specific_profiles(
    report: Mapping[str, object], profile: HardwareExecutionProfile
) -> None:
    if report.get("latency_energy_hardware_profile") != profile.profile_id:
        raise P6PublicReportError(
            "latency and energy evidence must remain hardware-specific"
        )
    if report.get("pareto_hardware_profile") != profile.profile_id:
        raise P6PublicReportError("Pareto evidence must remain hardware-specific")


def _validated_hardware_fields(
    report: Mapping[str, object],
) -> tuple[HardwareExecutionProfile, str, str, int, str, str]:
    profile_id = _identifier(report.get("hardware_profile"), "hardware profile")
    try:
        profile = load_hardware_execution_profile(profile_id)
    except ValueError:
        raise P6PublicReportError("hardware profile is invalid") from None
    if _identifier(report.get("target"), "target") != profile.target_hardware_id:
        raise P6PublicReportError("target does not match the hardware profile")
    target_model = _identifier(report.get("target_model"), "target model")
    family = _identifier(report.get("hardware_model_family"), "hardware model family")
    if family != profile.target_hardware_id:
        raise P6PublicReportError("hardware model family does not match the profile")
    gpu_count = report.get("gpu_count")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
        raise P6PublicReportError("GPU count is invalid")
    if profile.required_gpu_count is not None and gpu_count != profile.required_gpu_count:
        raise P6PublicReportError("GPU count does not match the hardware profile")
    backend = _identifier(report.get("execution_backend"), "execution backend")
    try:
        validate_profile_backend(profile, backend)
    except ValueError:
        raise P6PublicReportError(
            "execution backend does not match the hardware profile"
        ) from None
    if _identifier(report.get("tvm_arch"), "TVM architecture") != profile.tvm_arch:
        raise P6PublicReportError("TVM architecture does not match the hardware profile")
    cache = _identifier(report.get("tvm_cache_namespace"), "TVM cache namespace")
    if cache != profile.tvm_cache_namespace:
        raise P6PublicReportError("TVM cache namespace does not match the profile")
    return profile, target_model, family, gpu_count, backend, cache


def _validated_runtime_fields(
    report: Mapping[str, object],
) -> tuple[str, str, str, str]:
    return (
        _digest(report.get("environment_digest"), "environment"),
        _identifier(report.get("code_revision"), "code revision"),
        _digest(report.get("source_digest"), "source"),
        _digest(report.get("compiler_toolchain_digest"), "compiler toolchain"),
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
    return P6APProvenance(**expected)  # type: ignore[arg-type]


def _ap_provenance(value: object) -> P6APProvenance:
    if not isinstance(value, Mapping) or set(value) != _AP_PROVENANCE_KEYS:
        raise P6PublicReportError("AP provenance fields are invalid")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise P6PublicReportError("AP provenance seed is invalid")
    return P6APProvenance(
        data_split=_identifier(value.get("data_split"), "AP data split"),
        checkpoint_initial_state=_digest(
            value.get("checkpoint_initial_state"), "AP checkpoint initial state"
        ),
        training_config_digest=_digest(
            value.get("training_config_digest"), "AP training config"
        ),
        seed=seed,
        metric_protocol=_identifier(
            value.get("metric_protocol"), "AP metric protocol"
        ),
    )


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _PUBLIC_IDENTIFIER.fullmatch(value) is None:
        raise P6PublicReportError(f"{field} is invalid")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise P6PublicReportError(f"{field} digest is invalid")
    return value
