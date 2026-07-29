"""Tests for deterministic, fail-closed Stage2 contracts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import yaml

from framework.stage2.contracts import (
    _manifest_identity,
    Stage2EvidenceDelta,
    Stage2EvidenceRecord,
    Stage2Input,
    apply_stage1_gate,
    build_stage2_output,
    resolve_safe_output_root,
    safe_output_path,
    write_json_idempotent,
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
    manifest_path = tmp_path / "framework" / "partitions" / "pyramid_lidar_partition.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")
    classification_path = tmp_path / "classification.json"
    report = _classification() if classification is None else classification
    if report["models"][0].get("manifest") == "pyramid_lidar_partition.yaml":
        report["models"][0]["manifest"] = str(manifest_path)
    report["models"][0].setdefault(
        "manifest_digest", hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    classification_path.write_text(
        json.dumps(report),
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


def test_classifier_record_without_manifest_binding_fails_closed(tmp_path: Path) -> None:
    manifest_path, classification_path = _write_inputs(tmp_path)
    report = _classification()
    del report["models"][0]["manifest"]
    classification_path.write_text(json.dumps(report), encoding="utf-8")

    decision = apply_stage1_gate(manifest_path, classification_path)

    assert decision.allowed is False
    assert decision.reason == "model_identity_mismatch"


def test_classifier_record_with_same_basename_but_different_manifest_path_fails_closed(
    tmp_path: Path,
) -> None:
    manifest_path, classification_path = _write_inputs(
        tmp_path,
        _classification(manifest="different/pyramid_lidar_partition.yaml"),
    )

    decision = apply_stage1_gate(manifest_path, classification_path)

    assert decision.allowed is False
    assert decision.reason == "model_identity_mismatch"


def test_manifest_identity_normalizes_absolute_and_relative_partition_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_root = tmp_path / "demo"
    manifest_path, classification_path = _write_inputs(demo_root)
    expected_identity = "framework/partitions/pyramid_lidar_partition.yaml"

    absolute_decision = apply_stage1_gate(manifest_path, classification_path)

    monkeypatch.chdir(tmp_path)
    demo_relative_path = Path("demo/framework/partitions/pyramid_lidar_partition.yaml")
    demo_relative_decision = apply_stage1_gate(demo_relative_path, classification_path)
    demo_relative_identity = _manifest_identity(demo_relative_path)

    monkeypatch.chdir(demo_root)
    partition_relative_path = Path("framework/partitions/pyramid_lidar_partition.yaml")
    partition_relative_decision = apply_stage1_gate(partition_relative_path, classification_path)
    partition_relative_identity = _manifest_identity(partition_relative_path)

    assert absolute_decision.allowed is True
    assert demo_relative_decision.allowed is True
    assert partition_relative_decision.allowed is True
    assert {
        _manifest_identity(manifest_path),
        demo_relative_identity,
        partition_relative_identity,
    } == {expected_identity}


def test_classifier_record_with_same_manifest_path_but_different_digest_fails_closed(
    tmp_path: Path,
) -> None:
    manifest_path, classification_path = _write_inputs(tmp_path)
    report = json.loads(classification_path.read_text(encoding="utf-8"))
    report["models"][0]["manifest_digest"] = "0" * 64
    classification_path.write_text(json.dumps(report), encoding="utf-8")

    decision = apply_stage1_gate(manifest_path, classification_path)

    assert decision.allowed is False
    assert decision.reason == "model_identity_mismatch"


def test_classifier_record_without_manifest_digest_fails_closed(tmp_path: Path) -> None:
    manifest_path, classification_path = _write_inputs(tmp_path)
    report = json.loads(classification_path.read_text(encoding="utf-8"))
    del report["models"][0]["manifest_digest"]
    classification_path.write_text(json.dumps(report), encoding="utf-8")

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


def test_safe_output_root_and_writer_are_idempotent_and_reject_escapes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_root = resolve_safe_output_root(tmp_path / "output", repository)
    output = safe_output_path(output_root, "results/output.json")

    write_json_idempotent(output, {"stable": True})
    write_json_idempotent(output, {"stable": True})
    assert output.is_file()
    with pytest.raises(ValueError, match="overwrite"):
        write_json_idempotent(output, {"stable": False})
    with pytest.raises(ValueError, match="repository root"):
        resolve_safe_output_root(repository, repository)
    with pytest.raises(ValueError, match="relative"):
        safe_output_path(output_root, "../escape.json")
