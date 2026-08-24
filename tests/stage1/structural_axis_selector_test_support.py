"""Public selector and mutation builders shared by structural-axis tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from framework.stage1.adapters import (
    ConfigWriteTarget,
    MaterializerParameterSource,
    ScanScenario,
    TraceContext,
)
from framework.stage1.structural_axes import build_structural_axis_inputs
from framework.stage1.structural_axis_contract import (
    canonical_source_dataflow_relations,
    seal_dataflow_relation,
    seal_materializer_binding,
)
from framework.stage1.structural_axis_digest import canonical_digest
from tests.stage1.structural_axis_test_support import resign_structural_inputs


def selector_context(
    *,
    sources: tuple[MaterializerParameterSource, ...] | None = None,
    loaded_config: dict | None = None,
    checkpoint_width: int = 32,
    module_selector: str = "backbone.0",
) -> TraceContext:
    """Return a scanner-owned selector context for one convolution axis."""
    net = nn.Module()
    net.backbone = nn.Sequential(nn.Conv2d(16, 32, 1))
    declared = sources
    if declared is None:
        declared = (
            MaterializerParameterSource(
                axis_id="backbone.output",
                config_selector="model.encoder.width",
                mutation_kind="out_channels",
                module_root_selector=module_selector,
                allowed_roles=("output",),
                provenance={"adapter": "unit", "declaration": "encoder output"},
            ),
        )
    return TraceContext(
        net=net,
        example_inputs=(torch.ones(1, 16, 2, 2),),
        full_model=net,
        loaded_config=loaded_config or {"model": {"encoder": {"width": 32}}},
        checkpoint_evidence={
            "digest": "b" * 64,
            "module_widths": {"backbone.0": checkpoint_width},
        },
        materializer_sources=declared,
        trace_modules=dict(net.named_modules()),
        dataflow_relations=(
            {
                "module_root_selector": "backbone.0",
                "canonical_axis_id": "backbone.output",
            },
        ),
    )


def selector_groups(root_layer: str = "backbone.0") -> list[dict]:
    """Return the canonical retained selector group."""
    return [
        {
            "group_id": "g0",
            "root_layer": root_layer,
            "cur_width": 32,
            "round_to_default": 8,
            "width_floor": 8,
            "max_rate": 0.75,
        }
    ]


def selector_scenario() -> ScanScenario:
    """Return the canonical selector scan scenario."""
    return ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16", "INT8"),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={"backbone": ("FP16", "INT8")},
        alignment={
            "default_round_to": 8,
            "default_hardware_alignment": 8,
            "default_max_rate_numerator": 3,
            "default_max_rate_denominator": 4,
        },
    )


def scenario_evidence(scenario: ScanScenario) -> dict:
    """Serialize the formal parts of a selector scenario."""
    return {
        "hardware_precisions": list(scenario.hardware_precisions),
        "backend_precisions": list(scenario.backend_precisions),
        "compression_modes": list(scenario.compression_modes),
        "graph_quant_unit_policy": {
            key: list(value)
            for key, value in sorted(scenario.graph_quant_unit_policy.items())
        },
        "alignment": dict(sorted(scenario.alignment.items())),
    }


def _scanner_evidence_seal(
    groups: list[dict],
    manifest: dict,
    serialized_scenario: dict,
    source_relations: list[dict],
) -> str:
    declaration = {
        key: manifest[key]
        for key in (
            "source_group_count",
            "declared_relevant_group_ids",
            "relevant_groups_digest",
        )
    }
    return canonical_digest(
        {
            "prune_groups": sorted(groups, key=lambda group: group["group_id"]),
            "source_dataflow_relations": canonical_source_dataflow_relations(
                source_relations
            ),
            "group_manifest": declaration,
            "scenario": serialized_scenario,
        }
    )


def selector_group_manifest(
    groups: list[dict] | None = None,
    scenario: ScanScenario | None = None,
    source_relations: list[dict] | None = None,
) -> dict:
    """Return a sealed selector group manifest."""
    evidence = groups or selector_groups()
    group_ids = tuple(sorted(group["group_id"] for group in evidence))
    manifest = {
        "source_group_count": len(evidence),
        "declared_relevant_group_ids": group_ids,
        "relevant_groups_digest": canonical_digest(group_ids),
        "structural_evidence_digest": canonical_digest(evidence),
    }
    manifest["scan_manifest_digest"] = _scanner_evidence_seal(
        evidence,
        manifest,
        scenario_evidence(scenario or selector_scenario()),
        source_relations
        or [dict(row) for row in selector_context().dataflow_relations],
    )
    return manifest


def built_selector_inputs() -> dict:
    """Build canonical single-axis structural inputs."""
    groups = selector_groups()
    return build_structural_axis_inputs(
        trace_context=selector_context(),
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(groups),
    )


def built_cross_interface_inputs() -> dict:
    """Build canonical structural inputs with a cross-interface write target."""
    source = MaterializerParameterSource(
        axis_id="backbone.output",
        config_selector="model.encoder.width",
        mutation_kind="out_channels",
        module_root_selector="backbone.0",
        allowed_roles=("output",),
        provenance={"adapter": "unit", "declaration": "cross-interface output"},
        write_targets=(
            ConfigWriteTarget(selector="model.neck.width", transform="identity"),
        ),
    )
    base = selector_context(sources=(source,))
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config=base.loaded_config,
        checkpoint_evidence=base.checkpoint_evidence,
        materializer_sources=base.materializer_sources,
        trace_modules=base.trace_modules,
        dataflow_relations=(
            {
                "module_root_selector": "backbone.0",
                "canonical_axis_id": "backbone.output",
                "independent_interface": True,
            },
        ),
    )
    groups = selector_groups()
    return build_structural_axis_inputs(
        trace_context=context,
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(
            groups,
            source_relations=[dict(row) for row in context.dataflow_relations],
        ),
    )


def set_scanner_owned_binding_fields(
    inputs: dict, index: int, **changes: object
) -> None:
    """Update one outer/sealed binding and every dependent authority seal."""
    outer = inputs["materializer_bindings"][index]
    group_id = outer["b1_group_id"]
    sealed = next(
        row
        for row in inputs["scanner_evidence"]["materializer_bindings"]
        if row["b1_group_id"] == group_id
    )
    binding = seal_materializer_binding({**sealed, **changes})
    outer.clear()
    outer.update(binding)
    sealed.clear()
    sealed.update(binding)
    canonical_sources = _update_scanner_binding_relation(inputs, group_id, binding)
    _update_base_binding_seals(inputs, group_id, binding, canonical_sources)
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    resign_structural_inputs(inputs)


def _update_scanner_binding_relation(
    inputs: dict, group_id: str, binding: dict
) -> list[dict]:
    source = next(
        row
        for row in inputs["source_dataflow_relations"]
        if group_id in row["declared_member_group_ids"]
    )
    source.update(
        {
            "canonical_axis_id": binding["axis_id"],
            "axis_kind": binding["axis_kind"],
            "materializer_binding_projections": [binding],
        }
    )
    source["member_relations"] = [{"group_id": group_id, "role": binding["role"]}]
    if "derived_from" in binding:
        source["derived_from"] = binding["derived_from"]
    else:
        source.pop("derived_from", None)
    canonical_sources = canonical_source_dataflow_relations(
        inputs["source_dataflow_relations"]
    )
    inputs["source_dataflow_relations"] = canonical_sources
    inputs["scanner_evidence"]["source_dataflow_relations"] = canonical_sources
    derived = next(
        row for row in inputs["dataflow_relations"] if row["group_id"] == group_id
    )
    unsigned = {
        key: value
        for key, value in derived.items()
        if key not in {"relation_digest", "relation_provenance"}
    }
    unsigned.update(
        canonical_axis_id=binding["axis_id"],
        axis_kind=binding["axis_kind"],
        materializer_binding_digest=binding["materializer_binding_digest"],
    )
    if "derived_from" in binding:
        unsigned["derived_from"] = binding["derived_from"]
    else:
        unsigned.pop("derived_from", None)
    derived.clear()
    derived.update(seal_dataflow_relation(unsigned, source))
    return canonical_sources


def _update_base_binding_seals(
    inputs: dict,
    group_id: str,
    binding: dict,
    canonical_sources: list[dict],
) -> None:
    canonical_source = next(
        row for row in canonical_sources if row["canonical_group_id"] == group_id
    )
    for rows in (inputs["base_widths"], inputs["scanner_evidence"]["base_widths"]):
        base = next(row for row in rows if row["canonical_group_id"] == group_id)
        base["source_relation_digest"] = canonical_digest(canonical_source)
        base["materializer_binding_digest"] = binding[
            "materializer_binding_digest"
        ]
