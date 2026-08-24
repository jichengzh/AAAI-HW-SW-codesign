"""Stage6 test-only scanner-owned Pyramid formal-space fixtures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from framework.stage1.structural_axis_digest import (
    SCANNER_AXIS_PROVENANCE_SOURCE,
    canonical_digest,
    scanner_structural_axes_digest,
)


DEFAULT_STAGE_WIDTHS = {
    "stage1": [32, 64, 96, 128],
    "stage2": [32, 64, 96, 128],
    "stage3": [32, 64, 128],
}


def scanner_owned_pyramid_stage2_space(
    stage_widths: Mapping[str, list[int]] | None = None,
    *,
    q_modes: list[str] | None = None,
    software_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a small non-paper Pyramid Stage2 space with scanner-owned axes."""
    widths = _stage_widths(stage_widths)
    modes = q_modes or ["fp16", "int8"]
    axes = _scanner_axes(widths)
    hardware_target = _hardware_target()
    return {
        "schema": "stage2_search_space_v1",
        "model": "pyramid_lidar",
        "hardware_target": hardware_target,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(axes),
        "structural_axes": deepcopy(axes),
        "axis_schema": {"free_axes": axes, "fixed_derived_axes": []},
        "formal_q_modes": list(modes),
        "formal_q_mode_provenance": _q_mode_provenance(hardware_target, modes),
        "formal_candidate_policy": {
            "enumeration": "axis_schema.free_axes_x_formal_q_modes",
            "legacy_views": "diagnostic_only",
            "diagnostic_anchors_drive_formal_space": False,
            "source": "scanner_structural_axes",
        },
        "software_candidates": (
            deepcopy(software_candidates)
            if software_candidates is not None
            else _diagnostic_software_candidates(widths, modes)
        ),
        "hardware_candidates": _hardware_candidates(),
    }


def scanner_owned_pyramid_stage1_manifest(
    stage_widths: Mapping[str, list[int]] | None = None,
    *,
    q_modes: list[str] | None = None,
) -> dict[str, Any]:
    """Return a Stage1 manifest accepted by the public bridge formal contract."""
    widths = _stage_widths(stage_widths)
    modes = q_modes or ["fp16", "int8"]
    axes = _scanner_axes(widths)
    return {
        "schema": "stage1_partition_manifest_v1",
        "stage": "stage1_partition",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {
            "name": "h800",
            "int8_align": 32,
            "fp16_align": 8,
            "int8_pack_factor": 4,
            "alignment_enforcement": "hard",
            "ips": {"gpu": {"precisions": [mode.upper() for mode in modes]}},
        },
        "backend_support": {"precisions": list(modes)},
        "compression_modes": list(modes),
        "quant_units": [
            {"id": "pyramid_backbone", "legal_precisions": list(modes)}
        ],
        "scanner_structural_axes": axes,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(axes),
        "view_b1_search_groups": _view_b1_search_groups(widths),
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": [mode.upper() for mode in modes],
                "member_groups": [f"pyramid_group.s{index}" for index in range(3)],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _stage_widths(
    stage_widths: Mapping[str, list[int]] | None,
) -> dict[str, list[int]]:
    source = stage_widths or DEFAULT_STAGE_WIDTHS
    return {stage: list(source[stage]) for stage in ("stage1", "stage2", "stage3")}


def _scanner_axes(stage_widths: Mapping[str, list[int]]) -> list[dict[str, Any]]:
    return [
        _scanner_axis(f"pyramid.backbone.s{index}", stage, stage_widths[stage])
        for index, stage in enumerate(("stage1", "stage2", "stage3"))
    ]


def _scanner_axis(axis_id: str, dense_stage: str, widths: list[int]) -> dict[str, Any]:
    return {
        "axis_id": axis_id,
        "dense_stage": dense_stage,
        "axis_kind": "free",
        "base_width": widths[-1],
        "legal_widths": list(widths),
        "member_b1_groups": [
            {
                "b1_group_id": axis_id,
                "module_path": axis_id,
                "canonical_to_member_num": 1,
                "canonical_to_member_den": 1,
                "materializer_param": f"model.{axis_id}",
                "role": "output",
            }
        ],
        "round_to": 8,
        "provenance": _provenance(axis_id),
    }


def _provenance(axis_id: str) -> dict[str, Any]:
    return {
        "source": SCANNER_AXIS_PROVENANCE_SOURCE,
        "scanner_input_digest": "a" * 64,
        "input_digest": "a" * 64,
        "dataflow_group_ids": [axis_id],
        "config_digest": "b" * 64,
        "checkpoint_digest": "c" * 64,
        "scenario_digest": "d" * 64,
        "scan_manifest_digest": "e" * 64,
        "relevant_groups_digest": "f" * 64,
        "structural_evidence_digest": "1" * 64,
        "digest_sources": {
            "config_digest": "trace_context.loaded_config",
            "checkpoint_digest": "trace_context.checkpoint_evidence",
            "scenario_digest": "scan_scenario",
            "scan_manifest_digest": "scanner_group_manifest",
            "structural_evidence_digest": "canonical_structural_axis_inputs",
        },
    }


def _hardware_target() -> dict[str, Any]:
    return {
        "name": "h800",
        "int8_align": 32,
        "fp16_align": 8,
        "int8_pack_factor": 4,
        "alignment_enforcement": "hard",
    }


def _q_mode_provenance(
    hardware_target: dict[str, Any], modes: list[str]
) -> dict[str, Any]:
    payload = {
        "schema": "formal_q_mode_provenance_v1",
        "hardware_target": deepcopy(hardware_target),
        "sources": {
            "hardware": list(modes),
            "backend": list(modes),
            "configured": list(modes),
            "graph": list(modes),
        },
        "formal_q_modes": list(modes),
    }
    return {**payload, "digest": canonical_digest(payload)}


def _diagnostic_software_candidates(
    stage_widths: Mapping[str, list[int]], q_modes: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "dense_stage": stage,
            "software_points": [
                {
                    "id": f"diagnostic:{stage}:w{width}:{q_mode}",
                    "width": width,
                    "quant_policy": q_mode,
                    "buildable": True,
                    "status": "active",
                }
                for q_mode in q_modes
                for width in stage_widths[stage]
            ],
        }
        for stage in ("stage1", "stage2", "stage3")
    ]


def _hardware_candidates() -> list[dict[str, str]]:
    return [
        {
            "id": "tvm_metaschedule_candidate",
            "backend_scope": "measured_h800_tvm",
            "hardware": "h800",
            "schedule_policy": "tuned",
        }
    ]


def _view_b1_search_groups(stage_widths: Mapping[str, list[int]]) -> list[dict[str, Any]]:
    return [
        {
            "search_group_id": f"pyramid_group.s{index}",
            "bucket": "pyramid_backbone",
            "widths": list(stage_widths[stage]),
            "round_to": 8,
            "int8_buildable_align": 32,
            "max_rate": 0.75,
            "grouped_conv": True,
            "criterion_pool": ["L1"],
            "member_b1_groups": [f"pyramid_group.s{index}"],
        }
        for index, stage in enumerate(("stage1", "stage2", "stage3"))
    ]
