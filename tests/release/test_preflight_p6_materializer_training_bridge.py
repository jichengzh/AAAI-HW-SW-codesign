"""Smoke coverage for the zero-process P6 preflight public surface."""

from __future__ import annotations

from tools.release.preflight_p6_materializer_training_bridge import (
    P6MaterializerPreflightReport,
)


def test_preflight_report_has_only_public_release_fields() -> None:
    """The report shape must not grow a private execution detail field."""
    assert tuple(P6MaterializerPreflightReport.__dataclass_fields__) == (
        "schema_version",
        "status",
        "validated_round_count",
        "wrapper_marker",
        "training_required",
        "historical_process_launch_count",
        "gpu_probe_count",
    )
