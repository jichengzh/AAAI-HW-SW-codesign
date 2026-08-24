"""Public scanner-owned manifest builders for Stage1 bridge tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from framework.stage1.structural_axis_digest import (
    SCANNER_AXIS_PROVENANCE_SOURCE,
    scanner_structural_axes_digest,
)


def bridge_manifest(*, int8_buildable_align: int = 64) -> dict:
    payload = {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {
            "name": "h800_tvm_demo",
            "ips": {"gpu": {"precisions": ["FP16", "INT8"]}},
        },
        "backend_support": {"precisions": ["FP16", "INT8"]},
        "compression_modes": ["fp16", "int8"],
        "quant_units": [{"id": "all", "legal_precisions": ["FP16", "INT8"]}],
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group",
                "bucket": "pyramid_backbone",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": int8_buildable_align,
                "max_rate": 0.75,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["pyramid_group"],
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }
    payload["scanner_structural_axes"] = [
        paper_axis("pyramid_group", 64, [16, 32, 48, 64])
    ]
    seal_scanner_axes(payload)
    return payload


def write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def paper_axis(axis_id: str, base_width: int, legal_widths: list[int]) -> dict:
    return {
        "axis_id": axis_id,
        "dense_stage": None,
        "axis_kind": "free",
        "base_width": base_width,
        "legal_widths": legal_widths,
        "member_b1_groups": [
            {
                "b1_group_id": axis_id,
                "module_path": axis_id,
                "canonical_to_member_num": 1,
                "canonical_to_member_den": 1,
                "materializer_param": axis_id,
                "role": "output",
            }
        ],
        "round_to": 8,
        "provenance": _canonical_axis_provenance(axis_id),
    }


def seal_scanner_axes(manifest: dict) -> None:
    manifest["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        manifest["scanner_structural_axes"]
    )


def _canonical_axis_provenance(axis_id: str) -> dict[str, object]:
    return {
        "source": SCANNER_AXIS_PROVENANCE_SOURCE,
        "scanner_input_digest": "1" * 64,
        "input_digest": "1" * 64,
        "dataflow_group_ids": [axis_id],
        "config_digest": "2" * 64,
        "checkpoint_digest": "3" * 64,
        "scenario_digest": "4" * 64,
        "scan_manifest_digest": "5" * 64,
        "relevant_groups_digest": "6" * 64,
        "structural_evidence_digest": "7" * 64,
        "digest_sources": {
            "config_digest": "trace_context.loaded_config",
            "checkpoint_digest": "trace_context.checkpoint_evidence",
            "scenario_digest": "scan_scenario",
            "scan_manifest_digest": "scanner_group_manifest",
            "structural_evidence_digest": "canonical_structural_axis_inputs",
        },
    }
