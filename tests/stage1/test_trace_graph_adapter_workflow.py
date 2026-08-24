"""Small real-torch workflows for Stage1 boundaries, adapters, and graph scans."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from framework.stage1.adapters import (
    MaterializerParameterSource,
    ScanScenario,
    TraceAdapter,
    TraceContext,
    _first_conv_in_channels,
    _generic_bucket,
    get_adapter,
)
from framework.stage1.graph_scan import _output_shapes, _resolve_profile, scan
from framework.stage1.hardware_scan import HwCapability
from framework.stage1_bridge import load_stage2_search_space
from framework.stage1.trace_plan import (
    BoundaryValidator,
    GeneratedTraceWrapper,
    TraceBoundaryDetector,
    attach_runtime_validation,
    legacy_trace_plan_from_manifest,
)


class _FullModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.attention_fusion = nn.Identity()
        self.cls_head = nn.Conv2d(32, 2, 1)
        self.reg_head = nn.Conv2d(32, 4, 1)

    def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.attention_fusion(self.backbone(spatial_features))
        return self.cls_head(features), self.reg_head(features)


class _ToyAdapter(TraceAdapter):
    name = "toy"
    skipped_modules = ["attention_fusion (attention boundary)"]

    def build_trace_net(self, device: str) -> tuple[nn.Module, torch.Tensor]:
        return _FullModel().to(device).eval(), torch.ones(1, 32, 8, 8, device=device)

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


class _DiagnosticOnlyAdapter(_ToyAdapter):
    build_trace_context = None  # type: ignore[assignment]


class _ForgingAdapter(_ToyAdapter):
    def build_trace_context(
        self,
        net: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        loaded_config: dict,
        checkpoint_evidence: dict,
    ) -> TraceContext:
        return super().build_trace_context(
            net,
            example_inputs,
            {"model": {"backbone": {"width": 16}}},
            checkpoint_evidence,
        )


class _MutatingAdapter(_ToyAdapter):
    def materializer_parameter_sources(
        self, loaded_config: dict
    ) -> tuple[MaterializerParameterSource, ...]:
        loaded_config["model"]["backbone"]["width"] = 16
        return super().materializer_parameter_sources(loaded_config)


def _hardware() -> HwCapability:
    return HwCapability(
        {
            "name": "unit hardware",
            "ips": {"gpu": {"precisions": ["INT8", "FP16"]}, "dla": {"count": 1, "op_whitelist": ["Conv"]}},
            "alignment": {"int8_channel": 32, "fp16_channel": 16},
        },
        "unit.yaml",
        True,
    )


def _formal_evidence() -> tuple[dict, dict]:
    return (
        {"model": {"backbone": {"width": 32}}},
        {"digest": "c" * 64, "module_widths": {"backbone.0": 32}},
    )


def test_trace_detector_and_generated_wrapper_preserve_dense_head_interface() -> None:
    """The detector identifies fusion as a gate and emits an executable dense wrapper."""
    full_model = _FullModel().eval()
    plan = TraceBoundaryDetector().detect(
        full_model, model_name="attfuse", ckpt_status="loaded", input_shape=[1, 32, 8, 8]
    )
    wrapper = GeneratedTraceWrapper(full_model, plan["selected_candidate"])
    validation = BoundaryValidator().validate(wrapper, torch.ones(1, 32, 8, 8), run_prune=False)

    assert plan["review_required"] is True
    assert plan["selected_candidate"]["status"] == "selected"
    assert any(item["full_model_verdict_blocker"] for item in plan["skipped_subgraphs"])
    assert [list(tensor.shape) for tensor in wrapper(torch.ones(1, 32, 8, 8))] == [[1, 2, 8, 8], [1, 4, 8, 8]]
    assert validation["wrapper_forward_dryrun"] == "ok"
    assert validation["depgraph_build"] == "ok"
    assert validation["prune_dryrun"] == "skipped"


def test_legacy_trace_plan_and_runtime_validation_keep_coverage_boundaries() -> None:
    """Old manifest skips convert to typed blockers and gain non-mutating runtime evidence."""
    legacy = {"model": "v2xvit", "trace": {"skipped_modules": ["fusion attention", "sparse vfe"]}}
    plan = legacy_trace_plan_from_manifest(legacy)
    updated = attach_runtime_validation(
        plan, forward_status="ok", depgraph_status="ok", prune_status="fail", n_prunable_groups=3, output_shapes=[[1, 2]]
    )

    assert plan["coverage_scope"] == "dense_core_only"
    assert updated is not plan
    validation = updated["selected_candidate"]["validation"]
    assert validation["interface_invariant_check"] == "needs_review"
    assert validation["n_prunable_groups"] == 3


def test_toy_adapter_executes_full_cpu_graph_scan_without_hardware_measurement() -> None:
    """A real Conv graph reaches Stage1's B1/B2/routing/validation contracts on CPU."""
    manifest = scan(_ToyAdapter(), _hardware(), device="cpu", profile_latency_mode="off")

    assert manifest["scan_status"] == "ok"
    assert manifest["view_b1_prune_groups"]
    assert manifest["view_b1_search_groups"]
    assert manifest["view_b2_quant_units"]
    assert manifest["view_d_routing_segments"]["n_segments"] >= 1
    assert manifest["view_latency"]["status"] == "skipped"
    assert manifest["trace_plan"] is None
    assert manifest["formal_scan"]["status"] == "not_requested"
    assert "scanner_structural_axes" not in manifest


def test_formal_scan_requires_trace_context_instead_of_legacy_width_fallback() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )

    loaded_config, checkpoint = _formal_evidence()
    with pytest.raises(ValueError, match="TraceContext"):
        scan(
            _DiagnosticOnlyAdapter(),
            _hardware(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )


def test_formal_scan_requires_scanner_owned_config_and_checkpoint_evidence() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )

    with pytest.raises(ValueError, match="loaded_config.*checkpoint_evidence"):
        scan(
            _ToyAdapter(),
            _hardware(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
        )


def test_public_scan_rejects_non_scenario_objects_at_the_api_boundary() -> None:
    loaded_config, checkpoint = _formal_evidence()

    with pytest.raises(TypeError, match="ScanScenario"):
        scan(
            _ToyAdapter(),
            _hardware(),
            device="cpu",
            profile_latency_mode="off",
            scenario={"hardware_precisions": ["FP16"]},  # type: ignore[arg-type]
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )


def test_formal_scan_rejects_adapter_forged_scanner_evidence() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )
    loaded_config, checkpoint = _formal_evidence()

    with pytest.raises(ValueError, match="scanner-owned loaded_config"):
        scan(
            _ForgingAdapter(),
            _hardware(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )


def test_formal_scan_prevents_adapter_mutating_scanner_evidence() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )
    loaded_config, checkpoint = _formal_evidence()

    with pytest.raises(TypeError):
        scan(
            _MutatingAdapter(),
            _hardware(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )
    assert loaded_config["model"]["backbone"]["width"] == 32


def test_toy_scan_marks_real_partition_manifest_schema() -> None:
    manifest = scan(_ToyAdapter(), _hardware(), device="cpu", profile_latency_mode="off")

    assert manifest["schema"] == "stage1_partition_manifest_v1"
    assert manifest["stage"] == "stage1_partition"


def test_scan_scenario_drives_backend_and_compression_q_sources(tmp_path) -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16",),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={"backbone": ("FP16", "INT8"), "heads": ("FP16",)},
        alignment={"default_round_to": 32, "default_min_width": 32},
    )
    loaded_config, checkpoint = _formal_evidence()
    manifest = scan(
        _ToyAdapter(),
        _hardware(),
        device="cpu",
        profile_latency_mode="off",
        scenario=scenario,
        loaded_config=loaded_config,
        checkpoint_evidence=checkpoint,
    )
    path = tmp_path / "toy-scenario.yaml"
    import yaml

    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    assert manifest["scan_scenario"]["backend_precisions"] == ["FP16"]
    assert manifest["scanner_structural_axes"]
    assert len(manifest["scanner_structural_axes_digest"]) == 64
    assert manifest["scanner_structural_axes"][0]["provenance"]["input_digest"] == (
        manifest["scanner_structural_axes_digest"]
    )
    assert "structural_axis_inputs" not in manifest
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_adapter_helpers_and_graph_shape_modes_are_explicit() -> None:
    """Small helpers keep naming buckets and CPU profiling policy deterministic."""
    adapter = _ToyAdapter()

    assert _first_conv_in_channels(_FullModel()) == 32
    assert _generic_bucket("shrinker.block") == "neck"
    assert _generic_bucket("cls_head") == "heads"
    assert adapter.typed_skipped_subgraphs()[0]["type"] == "attention_or_routing_fusion"
    assert _resolve_profile("auto", "cpu") is False
    assert _resolve_profile("on", "cpu") is True
    assert _output_shapes({"one": torch.ones(1, 2), "two": (torch.ones(1),)}) == [[1, 2], [1]]
    with pytest.raises(KeyError, match="not-a-model"):
        get_adapter("not-a-model")
