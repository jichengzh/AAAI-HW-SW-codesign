"""Tests for deterministic, fail-closed Stage2 contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from framework.stage2.contracts import (
    Stage2EvidenceDelta,
    Stage2EvidenceRecord,
    Stage2Input,
    apply_stage1_gate,
    build_stage2_output,
)


def _manifest() -> dict:
    return {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": "h800_tvm_demo"},
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group",
                "bucket": "pyramid_backbone",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": 64,
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


def _classification(
    *, model: str = "pyramid_lidar", manifest: str = "pyramid_lidar_partition.yaml"
) -> dict:
    return {
        "schema": "stage1_model_classification_v1",
        "models": [
            {
                "model": model,
                "manifest": manifest,
                "acceleration_class": "CO_ACCELERATION_REQUIRED",
                "ckpt_status": "demo_synthetic",
                "classification": "demo",
                "evidence_level": "demo_only_not_measured",
                "scope": "demo_rsu_dense_core",
            }
        ],
    }


def _write_inputs(tmp_path: Path, classification: dict | None = None) -> tuple[Path, Path]:
    manifest_path = tmp_path / "pyramid_lidar_partition.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")
    classification_path = tmp_path / "classification.json"
    classification_path.write_text(
        json.dumps(_classification() if classification is None else classification),
        encoding="utf-8",
    )
    return manifest_path, classification_path


def test_missing_classification_fails_closed(tmp_path: Path) -> None:
    manifest_path, _ = _write_inputs(tmp_path)

    decision = apply_stage1_gate(manifest_path, None)

    assert decision.allowed is False
    assert decision.runtime_mode == "fail_closed"
    assert decision.reason == "missing_classification"


def test_classifier_record_with_matching_filename_but_wrong_model_is_rejected(
    tmp_path: Path,
) -> None:
    manifest_path, classification_path = _write_inputs(
        tmp_path,
        _classification(model="codriving"),
    )

    decision = apply_stage1_gate(manifest_path, classification_path)

    assert decision.allowed is False
    assert decision.reason == "model_identity_mismatch"


def test_fixed_input_produces_deterministic_stage2_output(tmp_path: Path) -> None:
    manifest_path, classification_path = _write_inputs(tmp_path)
    stage2_input = Stage2Input(manifest_path, classification_path)

    first = build_stage2_output(stage2_input).to_dict()
    second = build_stage2_output(stage2_input).to_dict()

    assert first == second
    assert first["optimization_status"]["mode"] == "joint"
    assert first["search_space_summary"]["n_software_candidates"] == 1


def test_evidence_delta_defensively_copies_nested_input_and_output() -> None:
    candidate_config = {"candidate": {"width": 64}}
    metric = {"latency": {"us": 123.0}}
    record = Stage2EvidenceRecord(
        backend="h800_tvm",
        hardware="H800 Hopper",
        scope="rsu_dense_core",
        evidence_kind="demo",
        provenance="test",
        candidate_config=candidate_config,
        metric=metric,
    )
    delta = Stage2EvidenceDelta(model="pyramid_lidar", records=[record])

    candidate_config["candidate"]["width"] = 16
    metric["latency"]["us"] = 999.0
    exported = delta.to_dict()
    exported["records"][0]["candidate_config"]["candidate"]["width"] = 8

    assert delta.to_dict()["records"][0]["candidate_config"]["candidate"]["width"] == 64
    assert delta.to_dict()["records"][0]["metric"]["latency"]["us"] == 123.0

    with pytest.raises(TypeError):
        record.candidate_config["candidate"]["width"] = 4


def test_manifest_must_decode_to_an_object(tmp_path: Path) -> None:
    manifest_path = tmp_path / "list.yaml"
    manifest_path.write_text("- not\n- a\n- manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        build_stage2_output(Stage2Input(manifest_path, None))
