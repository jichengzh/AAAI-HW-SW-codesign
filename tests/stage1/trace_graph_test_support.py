"""Public real-torch builders shared by Stage1 graph/bridge tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from framework.stage1.adapters import MaterializerParameterSource, TraceAdapter
from framework.stage1.hardware_scan import HwCapability


class FullModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.attention_fusion = nn.Identity()
        self.cls_head = nn.Conv2d(32, 2, 1)
        self.reg_head = nn.Conv2d(32, 4, 1)

    def forward(
        self, spatial_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.attention_fusion(self.backbone(spatial_features))
        return self.cls_head(features), self.reg_head(features)


class ToyAdapter(TraceAdapter):
    name = "toy"
    skipped_modules = ["attention_fusion (attention boundary)"]

    def build_trace_net(self, device: str) -> tuple[nn.Module, torch.Tensor]:
        model = FullModel().to(device).eval()
        return model, torch.ones(1, 32, 8, 8, device=device)

    def ignored_layers(self, net: nn.Module) -> list[nn.Module]:
        return [net.cls_head, net.reg_head]  # type: ignore[attr-defined]

    def materializer_parameter_sources(
        self, loaded_config: dict
    ) -> tuple[MaterializerParameterSource, ...]:
        assert "backbone" in loaded_config["model"]
        return (
            MaterializerParameterSource(
                axis_id="backbone",
                config_selector="model.backbone.width",
                mutation_kind="out_channels",
                module_root_selector="backbone.0",
                allowed_roles=("output",),
                provenance={"adapter": "toy"},
            ),
        )


def hardware_capability() -> HwCapability:
    """Return the deterministic CPU-only test hardware capability."""
    return HwCapability(
        {
            "name": "unit hardware",
            "ips": {
                "gpu": {"precisions": ["INT8", "FP16"]},
                "dla": {"count": 1, "op_whitelist": ["Conv"]},
            },
            "alignment": {"int8_channel": 32, "fp16_channel": 16},
        },
        "unit.yaml",
        True,
    )


def formal_evidence() -> tuple[dict, dict]:
    """Return scanner-owned config and checkpoint evidence for formal scans."""
    return (
        {"model": {"backbone": {"width": 32}}},
        {"digest": "c" * 64, "module_widths": {"backbone.0": 32}},
    )
