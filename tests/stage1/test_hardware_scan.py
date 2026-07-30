"""Public normalization contracts for the Stage1 hardware scanner."""

from __future__ import annotations

from pathlib import Path

import yaml

from framework.stage1.hardware_scan import HwCapability


def _raw_capability() -> dict:
    return {
        "basic": {"name": "demo accelerator"},
        "arch": {"family": "Demo", "sm": "sm99"},
        "ips": {
            "gpu": {"precisions": ["INT8", "FP16"]},
            "dla": {"enabled": True, "count": 2, "op_whitelist": ["Conv2d"]},
        },
        "alignment": {
            "int8_dense_channel": 64,
            "fp16_channel": 16,
            "int8_pack_factor": 8,
            "alignment_enforcement": "soft",
        },
        "quant_constraints": {
            "bit_widths_w": [8, 32],
            "granularity_w": ["per_channel"],
            "per_channel_activation_supported": True,
            "symmetric_only": False,
        },
        "memory": {"available_for_inference_gb": 12},
    }


def test_hardware_scan_normalizes_current_schema_fields(tmp_path: Path) -> None:
    """Graph scanning receives typed search constraints from a current capability YAML."""
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(_raw_capability()), encoding="utf-8")

    capability = HwCapability.from_yaml(path)

    assert capability.validated is True
    assert capability.name == "demo accelerator"
    assert capability.arch == "Demo sm99"
    assert capability.has_dla is True
    assert capability.dla_op_whitelist == ["Conv2d"]
    assert capability.gpu_precisions == ["INT8", "FP16"]
    assert capability.int8_align == 64
    assert capability.fp16_align == 16
    assert capability.int8_pack_factor == 8
    assert capability.round_to_for("int8") == 64
    assert capability.round_to_for("fp16") == 16
    assert capability.legal_bits == ["INT8"]
    assert capability.legal_granularity_w == ["per_channel"]
    assert capability.per_channel_act is True
    assert capability.symmetric_only is False
    assert capability.mem_capacity_gb == 12.0
    assert capability.summary()["schema_validated"] is True


def test_hardware_scan_retains_normalized_view_when_schema_validation_fails(tmp_path: Path) -> None:
    """An incomplete source YAML records validation failure but keeps safe defaults."""
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump({"ips": {"gpu": {}}, "alignment": {}}), encoding="utf-8")

    capability = HwCapability.from_yaml(path)

    assert capability.validated is False
    assert "ValidationError" in capability.validate_err
    assert capability.name == "unknown"
    assert capability.legal_bits == ["INT8", "FP16"]
    assert capability.int8_align == 32
    assert capability.fp16_align == 8
    assert capability.has_dla is False
