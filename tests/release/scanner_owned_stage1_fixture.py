"""Scanner-owned Stage1 seed builders shared by release tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from framework.stage1.adapters import (
    MaterializerParameterSource,
    ScanScenario,
    TraceContext,
)
from framework.stage1.structural_axes import (
    axis_to_scanner_dict,
    build_structural_axis_inputs,
    derive_structural_axes,
)
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from tests.stage1.structural_axis_selector_test_support import selector_group_manifest


SYNTHETIC_AXIS_WIDTH = 128
SYNTHETIC_AXIS_ROUND_TO = 32
SYNTHETIC_AXIS_IDS = ("backbone.stage1", "backbone.stage2", "backbone.stage3")
SYNTHETIC_GROUP_IDS = (
    "pyramid_group.s0",
    "pyramid_group.s1",
    "pyramid_group.s2",
)
SYNTHETIC_MODULE_PATHS = (
    "pyramid_backbone.stage0",
    "pyramid_backbone.stage1",
    "pyramid_backbone.stage2",
)


@dataclass(frozen=True)
class TraceModule:
    out_channels: int


def scanner_owned_pyramid_stage1_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a release seed upgraded to scanner-owned Stage2 inputs."""
    scenario = _scenario()
    inputs = build_structural_axis_inputs(
        trace_context=_trace_context(),
        prune_groups=_prune_groups(),
        scenario=scenario,
        group_manifest=selector_group_manifest(
            _prune_groups(),
            scenario,
            list(_dataflow_relations()),
        ),
    )
    bundle = derive_structural_axes({"structural_axis_inputs": inputs})
    axes = [axis_to_scanner_dict(axis, inputs) for axis in bundle.axes]
    return {
        **deepcopy(dict(manifest)),
        "backend_support": {"precisions": list(scenario.backend_precisions)},
        "compression_modes": list(scenario.compression_modes),
        "quant_units": [
            {"id": unit, "legal_precisions": list(precisions)}
            for unit, precisions in scenario.graph_quant_unit_policy.items()
        ],
        "scanner_structural_axes": axes,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(axes),
    }


def _scenario() -> ScanScenario:
    return ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16", "INT8"),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={"pyramid_backbone": ("FP16", "INT8")},
        alignment={
            **{f"{axis_id}.round_to": SYNTHETIC_AXIS_ROUND_TO for axis_id in SYNTHETIC_AXIS_IDS},
            **{
                f"{axis_id}.hardware_alignment": SYNTHETIC_AXIS_ROUND_TO
                for axis_id in SYNTHETIC_AXIS_IDS
            },
            **{f"{axis_id}.max_rate_numerator": 1 for axis_id in SYNTHETIC_AXIS_IDS},
            **{f"{axis_id}.max_rate_denominator": 2 for axis_id in SYNTHETIC_AXIS_IDS},
        },
    )


def _trace_context() -> TraceContext:
    return TraceContext(
        net=None,
        example_inputs=(),
        full_model=None,
        loaded_config={
            "model": {
                "args": {
                    "fusion_backbone": {
                        "num_filters": [
                            SYNTHETIC_AXIS_WIDTH,
                            SYNTHETIC_AXIS_WIDTH,
                            SYNTHETIC_AXIS_WIDTH,
                        ]
                    }
                }
            }
        },
        checkpoint_evidence={
            "digest": hashlib.sha256(b"release-fixture-checkpoint").hexdigest(),
            "module_widths": {
                module_path: SYNTHETIC_AXIS_WIDTH for module_path in SYNTHETIC_MODULE_PATHS
            },
        },
        materializer_sources=tuple(
            MaterializerParameterSource(
                axis_id=axis_id,
                config_selector=f"model.args.fusion_backbone.num_filters.{index}",
                mutation_kind="out_channels",
                module_root_selector=module_path,
                allowed_roles=("output",),
                provenance={
                    "adapter": "release_fixture",
                    "declaration": "synthetic pyramid backbone output",
                },
            )
            for index, (axis_id, module_path) in enumerate(
                zip(SYNTHETIC_AXIS_IDS, SYNTHETIC_MODULE_PATHS, strict=True)
            )
        ),
        trace_modules={
            module_path: TraceModule(out_channels=SYNTHETIC_AXIS_WIDTH)
            for module_path in SYNTHETIC_MODULE_PATHS
        },
        dataflow_relations=_dataflow_relations(),
    )


def _dataflow_relations() -> tuple[dict[str, str], ...]:
    return tuple(
        {"module_root_selector": module_path, "canonical_axis_id": axis_id}
        for axis_id, module_path in zip(SYNTHETIC_AXIS_IDS, SYNTHETIC_MODULE_PATHS, strict=True)
    )


def _prune_groups() -> list[dict[str, Any]]:
    return [
        {
            "group_id": group_id,
            "root_layer": module_path,
            "cur_width": SYNTHETIC_AXIS_WIDTH,
            "round_to_default": SYNTHETIC_AXIS_ROUND_TO,
            "width_floor": SYNTHETIC_AXIS_WIDTH // 2,
            "max_rate": 0.5,
        }
        for group_id, module_path in zip(SYNTHETIC_GROUP_IDS, SYNTHETIC_MODULE_PATHS, strict=True)
    ]
