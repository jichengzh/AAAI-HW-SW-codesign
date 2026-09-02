"""Formal Stage1 scan orchestration kept separate from legacy diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from framework.stage1.adapters import (
    ScanScenario,
    TraceAdapter,
    TraceContext,
    _evidence_equal,
    _freeze_evidence,
)
from framework.stage1.structural_axes import (
    axis_to_scanner_dict,
    build_structural_axis_inputs,
    derive_structural_axes,
)
from framework.stage1.structural_axis_contract import (
    scanner_group_manifest,
    scanner_scenario_payload,
)
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest


def validate_formal_request(
    scenario: ScanScenario | None,
    loaded_config: Mapping[str, Any] | None,
    checkpoint_evidence: Mapping[str, Any] | None,
) -> bool:
    if scenario is None:
        return False
    if not isinstance(scenario, ScanScenario):
        raise TypeError("scenario must be a ScanScenario instance")
    if loaded_config is None and checkpoint_evidence is None:
        return True
    if not isinstance(loaded_config, Mapping) or not isinstance(checkpoint_evidence, Mapping):
        raise ValueError(
            "formal graph scan requires scanner-owned loaded_config and "
            "checkpoint_evidence"
        )
    return True


def formal_axis_payload(
    adapter: TraceAdapter,
    net: Any,
    example_inputs: Any,
    prune_groups: list[dict],
    scenario: ScanScenario | None,
    loaded_config: Mapping[str, Any] | None,
    checkpoint_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if scenario is None:
        return {
            "formal_scan": {
                "status": "not_requested",
                "reason": "an explicit ScanScenario is required for formal axes",
            }
        }
    if loaded_config is None or checkpoint_evidence is None:
        raise ValueError("formal scanner evidence was not validated")
    context = _verified_trace_context(
        adapter, net, example_inputs, loaded_config, checkpoint_evidence
    )
    inputs = build_structural_axis_inputs(
        trace_context=context,
        prune_groups=prune_groups,
        scenario=scenario,
        group_manifest=scanner_group_manifest(
            prune_groups,
            scanner_scenario_payload(scenario),
            context.dataflow_relations,
        ),
    )
    trusted_declarations = context.dataflow_relations
    axes = [
        axis_to_scanner_dict(
            axis,
            inputs,
            trusted_source_relation_declarations=trusted_declarations,
        )
        for axis in derive_structural_axes(
            {"structural_axis_inputs": inputs},
            trusted_source_relation_declarations=trusted_declarations,
        ).axes
    ]
    return _axis_manifest_payload(axes, inputs)


def _verified_trace_context(
    adapter: TraceAdapter,
    net: Any,
    example_inputs: Any,
    loaded_config: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
) -> TraceContext:
    context_builder = getattr(adapter, "build_trace_context", None)
    if not callable(context_builder):
        raise ValueError("formal graph scan requires an adapter-provided TraceContext")
    scanner_config = _freeze_evidence(loaded_config)
    scanner_checkpoint = _freeze_evidence(checkpoint_evidence)
    context = context_builder(
        net, (example_inputs,), scanner_config, scanner_checkpoint
    )
    if not isinstance(context, TraceContext):
        raise ValueError("build_trace_context must return TraceContext")
    if context.net is not net:
        raise ValueError("TraceContext.net must be the traced network")
    if not _evidence_equal(context.loaded_config, scanner_config):
        raise ValueError("adapter replaced scanner-owned loaded_config evidence")
    if not _evidence_equal(context.checkpoint_evidence, scanner_checkpoint):
        raise ValueError("adapter replaced scanner-owned checkpoint evidence")
    return context


def _axis_manifest_payload(
    axes: list[dict[str, Any]], inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "scanner_structural_axes": axes,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(axes),
        "formal_scan": {
            "status": "derived",
            "structural_axis_inputs_digest": inputs["digest"],
            "source_provenance": inputs["provenance"],
        },
    }


def formal_manifest_fields(
    scenario: ScanScenario | None,
    quant_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if scenario is None:
        return {}
    return {
        "scan_scenario": {
            "hardware_precisions": list(scenario.hardware_precisions),
            "backend_precisions": list(scenario.backend_precisions),
            "compression_modes": list(scenario.compression_modes),
            "graph_quant_unit_policy": {
                key: list(value)
                for key, value in scenario.graph_quant_unit_policy.items()
            },
            "alignment": dict(scenario.alignment),
        },
        "backend_support": {"precisions": list(scenario.backend_precisions)},
        "compression_modes": list(scenario.compression_modes),
        "quant_units": [
            {
                "id": unit["unit"],
                "legal_precisions": list(_quant_policy(scenario, unit["unit"])),
                "quantizable": bool(unit.get("quantizable", True)),
            }
            for unit in quant_units
        ],
    }


def formal_hardware_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep semantic hardware capability fields and omit file provenance."""

    fields = (
        "name", "arch", "schema_validated", "has_dla", "dla_op_whitelist",
        "int8_align", "fp16_align", "alignment_enforcement", "legal_bits",
        "legal_granularity_w", "per_channel_act", "symmetric_only",
        "mem_capacity_gb",
    )
    return {field: summary[field] for field in fields if field in summary}


def _quant_policy(scenario: ScanScenario, unit: str) -> tuple[str, ...]:
    return scenario.graph_quant_unit_policy.get(
        unit, scenario.graph_quant_unit_policy.get("all", ())
    )


__all__ = [
    "formal_axis_payload",
    "formal_hardware_summary",
    "formal_manifest_fields",
    "validate_formal_request",
]
