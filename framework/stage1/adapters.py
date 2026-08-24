"""Stage1 子流程 B — 每模型 trace adapter (薄, 声明 trace 边界).

设计文档: §1 (trace_adapter 形式) + §3 S0。
每个 adapter 声明: 稠密 trace 核入口 / 跳过的不可 trace 模块 / 输出头(冻结) / 语义桶。
模型无关核 graph_scan.py 只认 TraceAdapter 接口, 不认具体模型。

4 个模型 (事实来自 multi_agent/model/*_structure_audit_v1.md + 源码核实):
  - codriving       : centerpointcodriving (V2Xverse), ResNetBEV(BasicBlock, 无 grouped), cls3/reg24
  - pyramid_lidar   : HeterPyramidCollab m1 (HEAL), ResNeXt g=32, cls2/reg14/dir4, 复用 depgraph_pyramid
  - pyramid_camera  : HeterPyramidSingle m2 (HEAL), LSS 不可 trace → 从 backbone_m2(128ch) 起 trace, 仅 OPV2V ckpt
  - v2xvit          : HeterModelBaseline (HEAL), 仅剪 backbone(transformer 不入图), 复用 depgraph_v2xvit
"""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
import torch.nn as nn

_REPO = Path(os.environ.get("STAGE1_REPO_ROOT", Path(__file__).resolve().parents[2]))
_HEAL = Path(os.environ.get("HEAL_ROOT", _REPO.parent / "HEAL"))
_HEAL_CKPT_ROOT = Path(os.environ.get("HEAL_CKPT_ROOT", _HEAL.parent / "checkpoints"))
_V2XVERSE = Path(os.environ.get("V2XVERSE_ROOT", _REPO.parent / "V2Xverse"))
_V2XVERSE_CKPT_ROOT = Path(os.environ.get("V2XVERSE_CKPT_ROOT", _V2XVERSE / "checkpoints"))


def _add_path(p: str):
    if p not in sys.path:
        sys.path.insert(0, p)


def _first_conv_in_channels(module: nn.Module, default: int = 64) -> int:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return default


def _generic_bucket(name: str) -> str:
    """层名 → 语义桶 (B2 量化视图用). 顺序敏感: 先判 neck/heads 再 backbone。"""
    n = name.lower()
    if "deblock" in n or "shrink" in n:          # 上采样 neck + shrink
        return "neck"
    if any(k in n for k in ("single_head", "cls_head", "reg_head", "dir_head")):
        return "heads"
    if "pyramid_backbone" in n:                  # HEAL ResNeXt BEV encoder
        return "bev_encoder"
    if "backbone" in n or "resnet" in n or "shrinker" in n:
        if "shrinker" in n:
            return "neck"
        return "backbone"
    return "other"


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(child) for child in value)
    return value


def _plain_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_evidence(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_evidence(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain_evidence(child) for child in value)
    return value


def _evidence_equal(left: Any, right: Any) -> bool:
    return _plain_evidence(left) == _plain_evidence(right)


def _freeze_evidence(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _deep_freeze(value)


def _config_sequence(config: Mapping[str, Any], selector: str) -> tuple[Any, ...]:
    value: Any = config
    for token in selector.split("."):
        if not isinstance(value, Mapping) or token not in value:
            raise ValueError(f"missing config sequence: {selector}")
        value = value[token]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"config selector must resolve to a non-empty sequence: {selector}")
    return tuple(value)


def _validated_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a non-string sequence")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{field_name} entries must be non-empty strings")
        normalized.append(item)
    return tuple(normalized)


def _validated_quant_policy(value: Any) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("graph_quant_unit_policy must be a non-empty mapping")
    normalized = {}
    for key, modes in value.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("graph_quant_unit_policy keys must be non-empty strings")
        normalized[key] = _validated_string_sequence(
            modes, f"graph_quant_unit_policy[{key!r}]"
        )
    return _deep_freeze(normalized)


def _validated_alignment(value: Any) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("alignment must be a non-empty mapping")
    normalized = {}
    for key, width in value.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("alignment keys must be non-empty strings")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise TypeError("alignment values must be positive non-boolean integers")
        normalized[key] = width
    return _deep_freeze(normalized)


@dataclass(frozen=True)
class ConfigWriteTarget:
    """Declarative config writeback derived from an axis's selected value."""

    selector: str
    transform: str = "identity"
    reference_selector: str | None = None


@dataclass(frozen=True)
class MaterializerParameterSource:
    """Declarative selector for one materializable canonical channel axis."""

    axis_id: str
    config_selector: str
    mutation_kind: str
    module_root_selector: str
    allowed_roles: tuple[str, ...]
    provenance: Mapping[str, object]
    write_targets: tuple[ConfigWriteTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_roles", tuple(self.allowed_roles))
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))
        object.__setattr__(self, "write_targets", tuple(self.write_targets))


@dataclass(frozen=True)
class TraceContext:
    """Immutable evidence envelope consumed by the scanner-owned resolver."""

    net: Any
    example_inputs: tuple[Any, ...]
    full_model: Any
    loaded_config: Mapping[str, Any]
    checkpoint_evidence: Mapping[str, Any]
    materializer_sources: tuple[MaterializerParameterSource, ...]
    trace_modules: Mapping[str, Any]
    dataflow_relations: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_inputs", tuple(self.example_inputs))
        object.__setattr__(self, "loaded_config", _deep_freeze(self.loaded_config))
        object.__setattr__(
            self, "checkpoint_evidence", _deep_freeze(self.checkpoint_evidence)
        )
        object.__setattr__(
            self, "materializer_sources", tuple(self.materializer_sources)
        )
        object.__setattr__(self, "trace_modules", _deep_freeze(self.trace_modules))
        object.__setattr__(
            self,
            "dataflow_relations",
            tuple(_deep_freeze(relation) for relation in self.dataflow_relations),
        )


@dataclass(frozen=True)
class ScanScenario:
    hardware_precisions: tuple[str, ...]
    backend_precisions: tuple[str, ...]
    compression_modes: tuple[str, ...]
    graph_quant_unit_policy: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    alignment: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hardware_precisions",
            _validated_string_sequence(
                self.hardware_precisions, "hardware_precisions"
            ),
        )
        object.__setattr__(
            self,
            "backend_precisions",
            _validated_string_sequence(self.backend_precisions, "backend_precisions"),
        )
        object.__setattr__(
            self,
            "compression_modes",
            _validated_string_sequence(self.compression_modes, "compression_modes"),
        )
        object.__setattr__(
            self,
            "graph_quant_unit_policy",
            _validated_quant_policy(self.graph_quant_unit_policy),
        )
        object.__setattr__(self, "alignment", _validated_alignment(self.alignment))


def _materializer_source(
    *,
    axis_id: str,
    config_selector: str,
    module_root_selector: str,
    adapter: str,
    axis_kind: str = "free",
    derived_from: str | None = None,
    write_targets: tuple[ConfigWriteTarget, ...] = (),
) -> MaterializerParameterSource:
    provenance: dict[str, object] = {
        "adapter": adapter,
        "axis_kind": axis_kind,
        "declaration": "config-sequence-derived materializer selector",
    }
    if derived_from is not None:
        provenance["derived_from"] = derived_from
    return MaterializerParameterSource(
        axis_id=axis_id,
        config_selector=config_selector,
        mutation_kind="out_channels",
        module_root_selector=module_root_selector,
        allowed_roles=("output",),
        provenance=provenance,
        write_targets=write_targets,
    )


def _source_dataflow_relation(
    source: MaterializerParameterSource,
) -> Mapping[str, Any]:
    relation: dict[str, Any] = {
        "module_root_selector": source.module_root_selector,
        "canonical_axis_id": source.axis_id,
        "axis_kind": str(source.provenance.get("axis_kind") or "free"),
    }
    derived_from = source.provenance.get("derived_from")
    if derived_from is not None:
        relation["derived_from"] = str(derived_from)
    return relation


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class TraceAdapter:
    name: str = "base"
    model_class: str = ""
    config_path: str = ""
    ckpt_path: str = ""
    ckpt_status: str = "ok"        # ok / opv2v_only_no_dair / missing
    skipped_modules: list[str] = []
    skipped_subgraphs: list[dict] = []
    trace_note: str = ""

    def build_trace_net(self, device: str) -> tuple[nn.Module, torch.Tensor]:
        raise NotImplementedError

    def build_trace_context(
        self,
        net: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        loaded_config: Mapping[str, Any],
        checkpoint_evidence: Mapping[str, Any],
    ) -> TraceContext:
        """Combine scanner-owned evidence with this adapter's selectors."""

        sources = self.materializer_parameter_sources(loaded_config)
        return TraceContext(
            net=net,
            example_inputs=example_inputs,
            full_model=net,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint_evidence,
            materializer_sources=sources,
            trace_modules=dict(net.named_modules()),
            dataflow_relations=self.structural_dataflow_relations(sources),
        )

    def materializer_parameter_sources(
        self, loaded_config: Mapping[str, Any]
    ) -> tuple[MaterializerParameterSource, ...]:
        del loaded_config
        raise ValueError(
            f"adapter {self.name!r} does not declare formal materializer selectors"
        )

    def structural_dataflow_relations(
        self, sources: tuple[MaterializerParameterSource, ...]
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(_source_dataflow_relation(source) for source in sources)

    def ignored_layers(self, net: nn.Module) -> list[nn.Module]:
        raise NotImplementedError

    def semantic_bucket(self, layer_name: str) -> str:
        return _generic_bucket(layer_name)

    def typed_skipped_subgraphs(self) -> list[dict]:
        """Return typed skipped trace boundaries while keeping legacy text skips."""

        if self.skipped_subgraphs:
            return [dict(item) for item in self.skipped_subgraphs]
        out = []
        for idx, desc in enumerate(self.skipped_modules):
            text = str(desc)
            name = text.split("(", 1)[0].strip() or f"skipped_{idx}"
            low = text.lower()
            if any(k in low for k in ("vfe", "scatter", "sparse", "quicksum", "cumsum")):
                typ = "sparse_or_geometry_preprocess"
            elif any(k in low for k in ("attention", "transformer", "where2comm", "v2vnet", "disco")):
                typ = "attention_or_routing_fusion"
            elif "fusion" in low or "warp" in low:
                typ = "fusion_or_alignment"
            else:
                typ = "custom_untraced_subgraph"
            out.append({
                "name": name,
                "type": typ,
                "description": text,
                "full_model_verdict_blocker": True,
                "blocker_gate": "trace_closure_required",
                "source": "trace_adapter.skipped_modules",
            })
        return out


# ---------------------------------------------------------------------------
# 1. CoDriving (V2Xverse)
# ---------------------------------------------------------------------------

class CoDrivingAdapter(TraceAdapter):
    name = "codriving"
    model_class = "centerpointcodriving"
    config_path = str(_V2XVERSE / "opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml")
    ckpt_path = str(_V2XVERSE_CKPT_ROOT / "codriving/perception/net_epoch_bestval_at16.pth")
    ckpt_status = "ok"
    skipped_modules = ["pillar_vfe (sparse VFE)", "scatter (sparse)", "fusion_net (CoDriving, 0-param, channel-preserving)"]
    trace_note = "trace 核 = backbone(ResNetBEV) → shrink_conv → cls/reg heads; 入口 = scatter 稠密输出"

    class _Net(nn.Module):
        def __init__(self, full):
            super().__init__()
            self.backbone = full.backbone
            self.shrink_flag = getattr(full, "shrink_flag", False)
            if self.shrink_flag:
                self.shrink_conv = full.shrink_conv
            self.cls_head = full.cls_head
            self.reg_head = full.reg_head

        def forward(self, spatial_features):
            feat = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
            if self.shrink_flag:
                feat = self.shrink_conv(feat)
            return self.cls_head(feat), self.reg_head(feat)

    def build_trace_net(self, device):
        _add_path(str(_V2XVERSE))
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.models.center_point_codriving import centerpointcodriving
        hypes = load_yaml(self.config_path)
        full = centerpointcodriving(hypes["model"]["args"])
        raw = torch.load(self.ckpt_path, map_location="cpu")
        sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        miss, unexp = full.load_state_dict(sd, strict=False)
        print(f"  [ckpt codriving] missing={len(miss)} unexpected={len(unexp)}")
        full = full.to(device).eval()
        net = self._Net(full).to(device).eval()
        cin = _first_conv_in_channels(net.backbone, 64)
        x = torch.randn(1, cin, 192, 704, device=device)
        return net, x

    def ignored_layers(self, net):
        return [net.cls_head, net.reg_head]

    def materializer_parameter_sources(
        self, loaded_config: Mapping[str, Any]
    ) -> tuple[MaterializerParameterSource, ...]:
        selector = "model.args.base_bev_backbone.num_filters"
        filters = _config_sequence(loaded_config, selector)
        backbone = tuple(
            _materializer_source(
                axis_id=f"backbone.s{index}",
                config_selector=f"{selector}[{index}]",
                module_root_selector=(
                    f"backbone.resnet.layer{index}.0.downsample.0"
                ),
                adapter=self.name,
            )
            for index, _ in enumerate(filters)
        )
        shrink_selector = "model.args.shrink_header.dim"
        _config_sequence(loaded_config, shrink_selector)
        return (
            *backbone,
            _materializer_source(
                axis_id="neck.output",
                config_selector=f"{shrink_selector}[0]",
                module_root_selector="shrink_conv.layers.0.double_conv.2",
                adapter=self.name,
                axis_kind="fixed_derived",
                derived_from=backbone[-1].axis_id,
            ),
        )


# ---------------------------------------------------------------------------
# 2. Pyramid-LiDAR (HEAL m1) — 复用 depgraph_pyramid
# ---------------------------------------------------------------------------

class PyramidLidarAdapter(TraceAdapter):
    name = "pyramid_lidar"
    model_class = "HeterPyramidCollab"
    _ckpt_dir = str(_HEAL_CKPT_ROOT / "stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29")
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch_bestval_at23.pth"
    ckpt_status = "ok"
    skipped_modules = ["encoder_m1 (PointPillar VFE, sparse)", "collab fusion warp_affine (channel-preserving)"]
    trace_note = "复用 PyramidFullTraceNet: backbone_m1→aligner→pyramid_backbone(ResNeXt g=32)→single_head→deblocks→shrink→cls/reg/dir"

    def build_trace_net(self, device):
        _add_path(str(_REPO))
        _add_path(str(_HEAL))
        from tools.configurable.depgraph_pyramid import build_full, PyramidFullTraceNet, find_ckpt
        ckpt = self.ckpt_path if Path(self.ckpt_path).is_file() else find_ckpt(self._ckpt_dir)
        full = build_full(self.config_path, ckpt, device)
        net = PyramidFullTraceNet(full).to(device).eval()
        cin = _first_conv_in_channels(net.backbone_m1, 64)
        x = torch.randn(1, cin, 128, 256, device=device)
        return net, x

    def ignored_layers(self, net):
        ign = [net.cls_head, net.reg_head, net.dir_head]
        for i in range(net.num_levels):
            ign.append(getattr(net.pyramid_backbone, f"single_head_{i}"))
        return ign

    def materializer_parameter_sources(
        self, loaded_config: Mapping[str, Any]
    ) -> tuple[MaterializerParameterSource, ...]:
        selector = "model.args.fusion_backbone.num_filters"
        filters = _config_sequence(loaded_config, selector)
        backbone = tuple(
            _materializer_source(
                axis_id=f"backbone.s{index}",
                config_selector=f"{selector}[{index}]",
                module_root_selector=(
                    "backbone_m1.resnet.layer0.0.downsample.0"
                    if index == 0
                    else f"pyramid_backbone.resnet.layer{index}.0.downsample.0"
                ),
                adapter=self.name,
            )
            for index, _ in enumerate(filters)
        )
        shrink_selector = "model.args.shrink_header.dim"
        _config_sequence(loaded_config, shrink_selector)
        return (
            *backbone,
            _materializer_source(
                axis_id="neck.output",
                config_selector=f"{shrink_selector}[0]",
                module_root_selector="shrink_conv.layers.0.double_conv.2",
                adapter=self.name,
                axis_kind="fixed_derived",
                derived_from=backbone[-1].axis_id,
            ),
        )


# ---------------------------------------------------------------------------
# 3. Pyramid-Camera (HEAL m2) — LSS 不可 trace, 从 backbone_m2 起
# ---------------------------------------------------------------------------

class PyramidCameraAdapter(TraceAdapter):
    name = "pyramid_camera"
    model_class = "HeterPyramidSingle (m2)"
    _ckpt_dir = str(_HEAL_CKPT_ROOT / "stage2/m2_alignto_m1")
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch25.pth"
    ckpt_status = "opv2v_only_no_dair"   # ⚠️ 无 DAIR camera ckpt, 仅 OPV2V
    skipped_modules = ["encoder_m2 (LiftSplatShoot: geometry proj + voxel scatter + QuickCumsum, 不可 trace)"]
    trace_note = ("LSS encoder 不可 trace → trace 核从 backbone_m2(BEV 入口)起: "
                  "backbone_m2→aligner_m2→pyramid_backbone.forward_single→shrink→cls/reg/dir; "
                  "⚠️ ckpt 仅 OPV2V(无 DAIR camera); "
                  "⚠️ aligner_m2(channel_align ConvNeXt 含 LayerNorm)排除出可剪集 —— "
                  "torch_pruning 结构化剪枝不更新 ConvNeXt LayerNorm normalized_shape(实测崩), "
                  "且其仅占 ~1.9% 参数, v0 冻结(其上游 backbone_m2 因耦合一并冻结, 占~4.7%); "
                  "主可剪路径 = pyramid_backbone(~56%)+neck(~37%)")

    class _Net(nn.Module):
        def __init__(self, full, modality="m2"):
            super().__init__()
            self.backbone = getattr(full, f"backbone_{modality}")
            self.aligner = getattr(full, f"aligner_{modality}")
            self.pyramid_backbone = full.pyramid_backbone
            self.shrink_flag = getattr(full, "shrink_flag", False)
            if self.shrink_flag:
                self.shrink_conv = full.shrink_conv
            self.cls_head = full.cls_head
            self.reg_head = full.reg_head
            self.dir_head = full.dir_head

        def forward(self, spatial_features):
            feat = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
            feat = self.aligner(feat)
            feat, occ = self.pyramid_backbone.forward_single(feat)
            if self.shrink_flag:
                feat = self.shrink_conv(feat)
            cls, reg, dir_ = self.cls_head(feat), self.reg_head(feat), self.dir_head(feat)
            if isinstance(occ, (list, tuple)) and len(occ) > 0:
                return (cls, reg, dir_, *[o for o in occ])
            return cls, reg, dir_

    def build_trace_net(self, device):
        _add_path(str(_REPO))
        _add_path(str(_HEAL))
        os.chdir(str(_HEAL))
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.models.heter_pyramid_single import HeterPyramidSingle
        hypes = load_yaml(self.config_path)
        full = HeterPyramidSingle(hypes["model"]["args"])
        raw = torch.load(self.ckpt_path, map_location="cpu")
        sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        miss, unexp = full.load_state_dict(sd, strict=False)
        print(f"  [ckpt pyramid_camera] missing={len(miss)} unexpected={len(unexp)}")
        full = full.to(device).eval()
        # 探测 modality 名 (camera 一般是 m2)
        modality = "m2"
        if not hasattr(full, f"backbone_{modality}"):
            for cand in ("m2", "m1", "m3", "m4"):
                if hasattr(full, f"backbone_{cand}"):
                    modality = cand
                    break
        net = self._Net(full, modality).to(device).eval()
        cin = _first_conv_in_channels(net.backbone, 128)
        x = torch.randn(1, cin, 256, 512, device=device)
        return net, x

    def ignored_layers(self, net):
        ign = [net.cls_head, net.reg_head, net.dir_head]
        # aligner_m2 (ConvNeXt channel_align + LayerNorm): torch_pruning 不更新
        # LayerNorm normalized_shape → 结构化剪枝后 forward 崩; v0 整体冻结。
        ign.append(net.aligner)
        for i in range(getattr(net.pyramid_backbone, "num_levels", 3)):
            sh = getattr(net.pyramid_backbone, f"single_head_{i}", None)
            if sh is not None:
                ign.append(sh)
        return ign


# ---------------------------------------------------------------------------
# 4. F-Cooper (HEAL HeterBaseline MaxFusion)
# ---------------------------------------------------------------------------

class FCooperAdapter(TraceAdapter):
    name = "fcooper"
    model_class = "HeterModelBaseline (MaxFusion)"
    _ckpt_dir = str(
        _HEAL_CKPT_ROOT
        / "baselines_hf/HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10"
    )
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch_bestval_at23.pth"
    ckpt_status = "ok"
    skipped_modules = [
        "encoder_m1 (PointPillar VFE, sparse)",
        "fusion_net (MaxFusion: parameter-free max pooling)",
    ]

    def build_trace_net(self, device):
        from framework.stage1.auto_trace import get_auto_adapter

        return get_auto_adapter(self.name).build_trace_net(device)

    def ignored_layers(self, net):
        from framework.stage1.auto_trace import get_auto_adapter

        return get_auto_adapter(self.name).ignored_layers(net)

    def materializer_parameter_sources(
        self, loaded_config: Mapping[str, Any]
    ) -> tuple[MaterializerParameterSource, ...]:
        backbone_selector = "model.args.m1.backbone_args.num_filters"
        filters = _config_sequence(loaded_config, backbone_selector)
        backbone = tuple(
            _materializer_source(
                axis_id=f"backbone.s{index}",
                config_selector=f"{backbone_selector}[{index}]",
                module_root_selector=f"body.backbone_m1.blocks.{index}.1",
                adapter=self.name,
            )
            for index, _ in enumerate(filters)
        )
        deblock_sequence = (
            "model.args.m1.backbone_args.num_upsample_filter"
        )
        _config_sequence(loaded_config, deblock_sequence)
        output_sequence = "model.args.m1.shrink_header.dim"
        _config_sequence(loaded_config, output_sequence)
        return (
            *backbone,
            _materializer_source(
                axis_id="neck.deblock",
                config_selector=f"{deblock_sequence}[0]",
                module_root_selector="body.backbone_m1.deblocks.0.0",
                adapter=self.name,
                write_targets=(
                    ConfigWriteTarget(
                        selector=f"{deblock_sequence}[*]",
                        transform="repeat_to_reference_length",
                        reference_selector=deblock_sequence,
                    ),
                    ConfigWriteTarget(
                        selector="model.args.m1.shrink_header.input_dim",
                        transform="multiply_by_reference_length",
                        reference_selector=deblock_sequence,
                    ),
                ),
            ),
            _materializer_source(
                axis_id="neck.output",
                config_selector=f"{output_sequence}[0]",
                module_root_selector="body.shrinker_m1.layers.0.double_conv.2",
                adapter=self.name,
                write_targets=(
                    ConfigWriteTarget(selector="model.args.in_head"),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# 5. V2X-ViT (HEAL) — 复用 depgraph_v2xvit, 仅剪 backbone (transformer 不入图)
# ---------------------------------------------------------------------------

class V2XViTAdapter(TraceAdapter):
    name = "v2xvit"
    model_class = "HeterModelBaseline (+ V2XTransformer)"
    _ckpt_dir = str(_HEAL_CKPT_ROOT / "baselines_hf/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch_bestval_at17.pth"
    ckpt_status = "ok"
    skipped_modules = ["encoder_m1 (PointPillar VFE, sparse)", "fusion_net (V2XTransformer: HMSA+MSwin, multi-agent, 不可 trace → 不剪)"]
    trace_note = "复用 V2XViTBackboneTraceNet: 仅 backbone_m1→shrinker_m1→cls/reg/dir; transformer 融合整体不入图(仅剪 backbone, 依据 A-2 授权 + dims_pruning §8.6)"

    def build_trace_net(self, device):
        _add_path(str(_REPO))
        _add_path(str(_HEAL))
        from tools.configurable.depgraph_v2xvit import build_model, V2XViTBackboneTraceNet
        full = build_model(device)
        net = V2XViTBackboneTraceNet(full).to(device).eval()
        cin = _first_conv_in_channels(net.backbone_m1, 64)
        x = torch.randn(1, cin, 256, 512, device=device)
        return net, x

    def ignored_layers(self, net):
        ign = [net.cls_head, net.reg_head, net.dir_head]
        # shrinker 输出层 ignored (保持 fusion 输入 256ch)
        for nm, m in net.shrinker_m1.named_modules():
            if isinstance(m, nn.Conv2d) and "double_conv.2" in nm:
                ign.append(m)
        if getattr(net, "has_shrink", False):
            ign.append(net.shrink_conv)
        return ign


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

REGISTRY = {
    "codriving": CoDrivingAdapter,
    "fcooper": FCooperAdapter,
    "pyramid_lidar": PyramidLidarAdapter,
    "pyramid_camera": PyramidCameraAdapter,
    "v2xvit": V2XViTAdapter,
}


def get_adapter(name: str) -> TraceAdapter:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}', choices={list(REGISTRY)}")
    return REGISTRY[name]()


__all__ = [
    "CoDrivingAdapter",
    "ConfigWriteTarget",
    "FCooperAdapter",
    "MaterializerParameterSource",
    "PyramidLidarAdapter",
    "REGISTRY",
    "ScanScenario",
    "TraceAdapter",
    "TraceContext",
    "get_adapter",
]
