"""Focused scanner-owned structural-axis test builders."""

from __future__ import annotations

from fractions import Fraction

from framework.reproduction.coptv2x_paper_space_v1 import (
    derive_paper_axis_bundle,
    load_paper_scanner_evidence,
)
from framework.stage1.structural_axis_contract import (
    canonical_source_dataflow_relations,
    formal_scanner_evidence_digest,
    formal_scanner_evidence_payload,
    seal_dataflow_relation,
    seal_materializer_binding,
    seal_scanner_inputs,
    source_relation_authority_payload,
)
from framework.stage1.structural_axis_digest import canonical_digest
from framework.stage1.structural_axis_widths import scenario_axis_constraint


def paper_scanner_evidence(name: str) -> dict:
    """Load one purified paper scanner-evidence fixture via production code."""
    return load_paper_scanner_evidence(name)


def paper_axis_bundle(name: str, evidence: dict | None = None):
    """Derive a paper axis bundle via the public reproduction pipeline."""
    return derive_paper_axis_bundle(name, evidence)


def resign_structural_inputs(inputs: dict) -> None:
    """Re-seal a deliberately mutated structural-input test payload."""
    inputs["structural_evidence_digest"] = canonical_digest(
        {
            key: inputs.get(key)
            for key in (
                "prune_groups",
                "source_dataflow_relations",
                "dataflow_relations",
                "materializer_bindings",
                "base_widths",
                "backend_constraints",
            )
        }
    )
    inputs["provenance"]["structural_evidence_digest"] = inputs[
        "structural_evidence_digest"
    ]
    unsigned = {
        key: value
        for key, value in inputs.items()
        if key not in {"digest", "scanner_input_digest"}
    }
    inputs["scanner_input_digest"] = canonical_digest(unsigned)
    inputs["digest"] = inputs["scanner_input_digest"]


def refresh_source_relation_authority(inputs: dict) -> None:
    """Refresh upstream authority for a deliberate source/group fixture mutation."""

    evidence = inputs["scanner_evidence"]
    authority = evidence["source_relation_authority"]
    generated = {
        "canonical_group_id",
        "materializer_binding_projections",
        "member_relations_source",
    }
    if authority["schema"] == "scanner_retained_graph_inference_v1":
        generated |= {
            "member_relations",
            "declared_member_group_ids",
            "member_relations_digest",
        }
    declarations = [
        {key: value for key, value in source.items() if key not in generated}
        for source in inputs["source_dataflow_relations"]
    ]
    refreshed = source_relation_authority_payload(
        inputs["prune_groups"],
        evidence["group_manifest"],
        evidence["scenario"],
        declarations,
    )
    evidence["source_relation_authority"] = refreshed
    inputs["provenance"]["source_relation_authority_schema"] = refreshed["schema"]
    inputs["provenance"]["source_group_manifest_digest"] = refreshed[
        "group_manifest_digest"
    ]


def set_scanner_owned_group_width(
    inputs: dict, group_id: str, width: int
) -> None:
    """Set an outer and sealed scanner group width, then re-seal inputs."""
    outer_group = next(
        group
        for group in inputs["prune_groups"]
        if group.get("group_id", group.get("id")) == group_id
    )
    outer_group["cur_width"] = width
    sealed_group = next(
        group
        for group in inputs["scanner_evidence"]["prune_groups"]
        if group.get("group_id", group.get("id")) == group_id
    )
    sealed_group["cur_width"] = width
    refresh_source_relation_authority(inputs)
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    resign_structural_inputs(inputs)


def _scan(
    prune_groups: list[dict],
    dataflow_relations: list[dict],
    materializer_bindings: list[dict],
    base_widths: list[dict],
    backend_constraints: list[dict],
) -> dict:
    scenario, constraints = _scenario_constraints(backend_constraints)
    sealed_bindings = _scanner_owned_bindings(
        materializer_bindings, dataflow_relations
    )
    source_relations, relations = _scanner_owned_dataflow_relations(
        prune_groups, dataflow_relations, sealed_bindings
    )
    group_ids = tuple(sorted(row["id"] for row in prune_groups))
    group_manifest = {
        "source_group_count": len(prune_groups),
        "declared_relevant_group_ids": group_ids,
        "relevant_groups_digest": canonical_digest(group_ids),
    }
    config_digest = canonical_digest({"base_widths": base_widths})
    checkpoint_digest = canonical_digest({"prune_groups": prune_groups})
    base_authority = _scanner_base_width_authority(
        prune_groups, sealed_bindings, base_widths, source_relations
    )
    return _sealed_scan_payload(
        prune_groups, source_relations, relations, sealed_bindings,
        base_authority, constraints, scenario, group_manifest,
        config_digest, checkpoint_digest,
    )


def _scenario_constraints(
    backend_constraints: list[dict],
) -> tuple[dict, list[dict]]:
    alignment = {}
    for row in backend_constraints:
        rate = Fraction(str(row["max_rate"]))
        axis_id = row["axis_id"]
        alignment[f"{axis_id}.round_to"] = row["round_to"]
        alignment[f"{axis_id}.hardware_alignment"] = row.get(
            "hardware_alignment", row["round_to"]
        )
        alignment[f"{axis_id}.max_rate_numerator"] = rate.numerator
        alignment[f"{axis_id}.max_rate_denominator"] = rate.denominator
    scenario = {
        "hardware_precisions": ["FP16", "INT8"],
        "backend_precisions": ["FP16", "INT8"],
        "compression_modes": ["fp16", "int8"],
        "graph_quant_unit_policy": {"all": ["FP16", "INT8"]},
        "alignment": dict(sorted(alignment.items())),
    }
    constraints = [
        scenario_axis_constraint(row["axis_id"], scenario)
        for row in backend_constraints
    ]
    return scenario, constraints


def _sealed_scan_payload(
    prune_groups: list[dict],
    source_relations: list[dict],
    relations: list[dict],
    sealed_bindings: list[dict],
    base_authority: list[dict],
    constraints: list[dict],
    scenario: dict,
    group_manifest: dict,
    config_digest: str,
    checkpoint_digest: str,
) -> dict:
    base_provenance = {
        "source": "trace_context.config_checkpoint_depgraph",
        "config_digest": config_digest,
        "checkpoint_digest": checkpoint_digest,
    }
    generated_fields = {
        "canonical_group_id",
        "materializer_binding_projections",
        "member_relations_source",
    }
    declarations = [
        {key: value for key, value in source.items() if key not in generated_fields}
        for source in source_relations
    ]
    scanner_evidence = formal_scanner_evidence_payload(
        prune_groups, group_manifest, scenario, source_relations,
        sealed_bindings, base_authority, base_provenance, declarations,
    )
    relation_authority = scanner_evidence["source_relation_authority"]
    payload = {
        "prune_groups": prune_groups,
        "source_dataflow_relations": source_relations,
        "dataflow_relations": relations,
        "materializer_bindings": sealed_bindings,
        "base_widths": base_authority,
        "backend_constraints": constraints,
        "scanner_evidence": scanner_evidence,
        "provenance": {
            "source": "stage1.graph_scan.build_structural_axis_inputs",
            "config_digest": config_digest,
            "checkpoint_digest": checkpoint_digest,
            "scenario_digest": canonical_digest(scenario),
            "scan_manifest_digest": formal_scanner_evidence_digest(scanner_evidence),
            "source_relation_authority_schema": relation_authority["schema"],
            "source_group_manifest_digest": relation_authority[
                "group_manifest_digest"
            ],
            "digest_sources": {
                "config_digest": "trace_context.loaded_config",
                "checkpoint_digest": "trace_context.checkpoint_evidence",
                "scenario_digest": "scan_scenario",
                "scan_manifest_digest": "scanner_group_manifest",
            },
        },
    }
    return {
        "structural_axis_inputs": seal_scanner_inputs(payload)
    }


def _scanner_base_width_authority(
    groups: list[dict],
    bindings: list[dict],
    bases: list[dict],
    sources: list[dict],
) -> list[dict]:
    by_group = {group.get("group_id", group.get("id")): group for group in groups}
    rows = []
    for base in bases:
        source = next(
            row for row in sources if row["canonical_axis_id"] == base["axis_id"]
        )
        binding = next(
            row
            for row in bindings
            if row["b1_group_id"] == source["canonical_group_id"]
        )
        group = by_group[binding["b1_group_id"]]
        width = base["width"]
        rows.append(
            {
                **base,
                "config_path": binding["param"],
                "config_width": width,
                "checkpoint_module_path": group.get(
                    "root_layer", group.get("module_path")
                ),
                "checkpoint_width": width,
                "canonical_group_id": binding["b1_group_id"],
                "canonical_group_width": group["cur_width"],
                "source_relation_digest": canonical_digest(source),
                "materializer_binding_digest": binding[
                    "materializer_binding_digest"
                ],
            }
        )
    return rows


def _scanner_owned_dataflow_relations(
    prune_groups: list[dict],
    logical_relations: list[dict],
    bindings: list[dict],
) -> tuple[list[dict], list[dict]]:
    sources = _scanner_owned_sources(prune_groups, logical_relations, bindings)
    return canonical_source_dataflow_relations(sources), _derived_relations(sources)


def _scanner_owned_sources(
    prune_groups: list[dict],
    logical_relations: list[dict],
    bindings: list[dict],
) -> list[dict]:
    sources = []
    for relation in logical_relations:
        logical_group_id = relation["group_id"]
        member_ids = sorted(
            group["id"]
            for group in prune_groups
            if group.get("dataflow_group_id") == logical_group_id
        )
        if not member_ids:
            raise ValueError(f"no retained members for {logical_group_id}")
        source = {key: value for key, value in relation.items() if key != "group_id"}
        member_bindings = [
            binding
            for binding in bindings
            if binding["b1_group_id"] in member_ids
            and binding["axis_id"] == relation["canonical_axis_id"]
        ]
        canonical_binding = next(
            binding for binding in member_bindings if binding["role"] != "internal"
        )
        canonical_group = next(
            group
            for group in prune_groups
            if group["id"] == canonical_binding["b1_group_id"]
        )
        sources.append(
            {
                **source,
                "group_id": canonical_binding["b1_group_id"],
                "canonical_group_id": canonical_binding["b1_group_id"],
                "module_root_selector": canonical_group["module_path"],
                "member_relations_source": "adapter_declared_v1",
                "member_relations": [
                    {
                        "group_id": binding["b1_group_id"],
                        "role": binding["role"],
                    }
                    for binding in member_bindings
                ],
                "declared_member_group_ids": member_ids,
                "member_relations_digest": canonical_digest(tuple(member_ids)),
                "materializer_binding_projections": member_bindings,
            }
        )
    return sources


def _derived_relations(sources: list[dict]) -> list[dict]:
    derived = []
    for raw_source in sources:
        source = canonical_source_dataflow_relations([raw_source])[0]
        shared = {
            key: source[key]
            for key in ("canonical_axis_id", "axis_kind", "independent_interface")
            if key in source
        }
        projections = {
            row["b1_group_id"]: row
            for row in source["materializer_binding_projections"]
        }
        for member_id in source["declared_member_group_ids"]:
            derived.append(
                seal_dataflow_relation(
                    {
                        **shared,
                        "group_id": member_id,
                        "materializer_binding_digest": projections[member_id][
                            "materializer_binding_digest"
                        ],
                    },
                    source,
                )
            )
    return derived


def _scanner_owned_bindings(
    bindings: list[dict], relations: list[dict]
) -> list[dict]:
    source_by_axis = {row["canonical_axis_id"]: row for row in relations}
    sealed = []
    for raw in bindings:
        source = source_by_axis[raw["axis_id"]]
        row = {
            **raw,
            "axis_kind": raw.get("axis_kind", source.get("axis_kind", "free")),
            "write_targets": raw.get("write_targets", []),
        }
        for field in ("derived_from", "independent_interface"):
            if source.get(field) is not None:
                row[field] = source[field]
        sealed.append(seal_materializer_binding(row))
    return sealed


def pyramid_stage_with_output_c_and_internal_2c() -> dict:
    return _scan(
        prune_groups=[
            {
                "id": "stage3.out",
                "module_path": "backbone.stage3.out",
                "cur_width": 256,
                "dataflow_group_id": "stage3",
            },
            {
                "id": "stage3.inner",
                "module_path": "backbone.stage3.inner",
                "cur_width": 512,
                "dataflow_group_id": "stage3",
            },
        ],
        dataflow_relations=[
            {"group_id": "stage3", "canonical_axis_id": "backbone.stage3"}
        ],
        materializer_bindings=[
            {
                "b1_group_id": "stage3.out",
                "axis_id": "backbone.stage3",
                "param": "stage3",
                "role": "output",
            },
            {
                "b1_group_id": "stage3.inner",
                "axis_id": "backbone.stage3",
                "param": "stage3",
                "role": "internal",
            },
        ],
        base_widths=[{"axis_id": "backbone.stage3", "width": 256}],
        backend_constraints=[
            {
                "axis_id": "backbone.stage3",
                "round_to": 32,
                "hardware_alignment": 32,
                "max_rate": 0.75,
            }
        ],
    )


def stage_with_non_integral_member_width() -> dict:
    bad = pyramid_stage_with_output_c_and_internal_2c()
    set_scanner_owned_group_width(
        bad["structural_axis_inputs"], "stage3.inner", 300
    )
    bad["structural_axis_inputs"]["backend_constraints"][0]["round_to"] = 32
    resign_structural_inputs(bad["structural_axis_inputs"])
    return bad


def codriving_scan_with_neck_binding() -> dict:
    return _scan(
        prune_groups=[
            {"id": "s1.out", "module_path": "backbone.stage1", "cur_width": 64, "dataflow_group_id": "s1"},
            {"id": "s2.out", "module_path": "backbone.stage2", "cur_width": 128, "dataflow_group_id": "s2"},
            {"id": "s3.out", "module_path": "backbone.stage3", "cur_width": 256, "dataflow_group_id": "s3"},
            {"id": "neck.out", "module_path": "neck.output", "cur_width": 256, "dataflow_group_id": "neck_output"},
        ],
        dataflow_relations=[
            {"group_id": "s1", "canonical_axis_id": "backbone.stage1"},
            {"group_id": "s2", "canonical_axis_id": "backbone.stage2"},
            {"group_id": "s3", "canonical_axis_id": "backbone.stage3"},
            {
                "group_id": "neck_output",
                "canonical_axis_id": "neck.output",
                "axis_kind": "fixed_derived",
                "derived_from": "backbone.stage3",
            },
        ],
        materializer_bindings=[
            {"b1_group_id": "s1.out", "axis_id": "backbone.stage1", "param": "stage1", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "s2.out", "axis_id": "backbone.stage2", "param": "stage2", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "s3.out", "axis_id": "backbone.stage3", "param": "stage3", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "neck.out", "axis_id": "neck.output", "param": "stage3", "role": "derived_neck", "axis_kind": "fixed_derived", "derived_from": "backbone.stage3"},
        ],
        base_widths=[
            {"axis_id": "backbone.stage1", "width": 64},
            {"axis_id": "backbone.stage2", "width": 128},
            {"axis_id": "backbone.stage3", "width": 256},
            {"axis_id": "neck.output", "width": 256},
        ],
        backend_constraints=[
            {"axis_id": "backbone.stage1", "round_to": 8, "hardware_alignment": 8, "max_rate": 0.75},
            {"axis_id": "backbone.stage2", "round_to": 16, "hardware_alignment": 16, "max_rate": 0.75},
            {"axis_id": "backbone.stage3", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.75},
            {"axis_id": "neck.output", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.75},
        ],
    )


def fcooper_scan_with_independent_neck_bindings() -> dict:
    return _scan(
        prune_groups=[
            {"id": "b0.out", "module_path": "backbone.s0", "cur_width": 64, "dataflow_group_id": "b0"},
            {"id": "b1.out", "module_path": "backbone.s1", "cur_width": 128, "dataflow_group_id": "b1"},
            {"id": "b2.out", "module_path": "backbone.s2", "cur_width": 256, "dataflow_group_id": "b2"},
            {"id": "neck.deblock", "module_path": "neck.deblock", "cur_width": 128, "dataflow_group_id": "neck_deblock"},
            {"id": "neck.output", "module_path": "neck.output", "cur_width": 256, "dataflow_group_id": "neck_output"},
        ],
        dataflow_relations=[
            {"group_id": "b0", "canonical_axis_id": "backbone.s0"},
            {"group_id": "b1", "canonical_axis_id": "backbone.s1"},
            {"group_id": "b2", "canonical_axis_id": "backbone.s2"},
            {"group_id": "neck_deblock", "canonical_axis_id": "neck.deblock", "independent_interface": True},
            {"group_id": "neck_output", "canonical_axis_id": "neck.output", "independent_interface": True},
        ],
        materializer_bindings=[
            {"b1_group_id": "b0.out", "axis_id": "backbone.s0", "param": "backbone.s0", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "b1.out", "axis_id": "backbone.s1", "param": "backbone.s1", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "b2.out", "axis_id": "backbone.s2", "param": "backbone.s2", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "neck.deblock", "axis_id": "neck.deblock", "param": "neck.deblock", "role": "neck_interface", "axis_kind": "free"},
            {"b1_group_id": "neck.output", "axis_id": "neck.output", "param": "neck.output", "role": "neck_interface", "axis_kind": "free"},
        ],
        base_widths=[
            {"axis_id": "backbone.s0", "width": 64},
            {"axis_id": "backbone.s1", "width": 128},
            {"axis_id": "backbone.s2", "width": 256},
            {"axis_id": "neck.deblock", "width": 128},
            {"axis_id": "neck.output", "width": 256},
        ],
        backend_constraints=[
            {"axis_id": "backbone.s0", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.5},
            {"axis_id": "backbone.s1", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.75},
            {"axis_id": "backbone.s2", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.875},
            {"axis_id": "neck.deblock", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.75},
            {"axis_id": "neck.output", "round_to": 32, "hardware_alignment": 32, "max_rate": 0.75},
        ],
    )
