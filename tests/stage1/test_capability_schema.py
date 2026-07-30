"""Behavioral contracts for normalized hardware capability schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from framework.capability_schema import HardwareCapability


def _schema_v1_payload() -> dict:
    return {
        "basic": {"name": "demo accelerator", "cost_usd": 99},
        "arch": {"family": "Demo", "sm": "sm99", "vendor": "example"},
        "ips": {
            "gpu": {"precisions": ["FP16", "INT8"]},
            "dla": {"enabled": False, "precisions": ["INT8"]},
        },
        "alignment": {"int8_channel": 64, "alignment_enforcement": "soft"},
        "features": {"dla_count": 0},
        "toolchain": {"framework": "TensorRT", "cuda": "12.0"},
    }


def test_from_yaml_flattens_schema_v1_and_excludes_disabled_ips(tmp_path: Path) -> None:
    """Schema-v1 input has a stable legacy view without enabling disabled DLA."""
    path = tmp_path / "capability.yaml"
    path.write_text(yaml.safe_dump(_schema_v1_payload()), encoding="utf-8")

    capability = HardwareCapability.from_yaml(path)

    assert capability.name == "demo accelerator"
    assert capability.arch == "Demo sm99"
    assert capability.supported_precisions == {"FP16", "INT8"}
    assert capability.has_dla is False
    assert capability.alignment.int8_channel == 64
    assert capability.alignment.alignment_enforcement == "soft"


def test_capability_requires_at_least_one_ip() -> None:
    """The schema rejects a capability that cannot execute on any IP."""
    with pytest.raises(ValidationError, match="至少要描述一个 IP"):
        HardwareCapability.model_validate({"name": "empty", "arch": "none", "ips": {}})


def test_disabled_dla_is_a_valid_descriptor_but_not_an_available_accelerator() -> None:
    """The parser preserves source metadata while the runtime view keeps DLA disabled."""
    capability = HardwareCapability.model_validate(
        {"name": "disabled", "arch": "demo", "ips": {"dla": {"enabled": False}}}
    )

    assert capability.has_dla is False
    assert capability.supported_precisions == set()


def test_dla_compatibility_accepts_legacy_count_or_new_precision_metadata() -> None:
    """Both documented DLA representations signal an available accelerator."""
    legacy = HardwareCapability.model_validate(
        {
            "name": "legacy", "arch": "demo", "ips": {"dla": {}}, "features": {"dla_count": 1}
        }
    )
    modern = HardwareCapability.model_validate(
        {
            "name": "modern",
            "arch": "demo",
            "ips": {"dla": {"enabled": True, "precisions": ["INT8"]}},
        }
    )

    assert legacy.has_dla is True
    assert modern.has_dla is True
