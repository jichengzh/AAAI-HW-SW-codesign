"""Behavioral guardrails for Stage1 classification and calibrated prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from framework.stage1.calibrated_predictor import (
    build_calibrated_report,
    build_predictions_v1,
    prohibited_claims,
    render_calibrated_md,
    render_predictions_md,
)
from framework.stage1.coupling_predictor import find_overpromotions, markdown_report, predict_manifest, predict_manifests
from framework.stage1.model_classifier import (
    build_classification_report,
    normalize_manifest_for_predictor,
    render_classification_md,
    write_classification_report,
)


MODELS = ("codriving", "fcooper", "attfuse", "v2xvit", "pyramid_lidar", "where2comm")


def _manifest(model: str) -> dict:
    return {
        "model": model,
        "ckpt_status": "loaded",
        "scope": "traced_dense_subgraph",
        "trace": {"skipped_modules": ["fusion.attention"] if model == "attfuse" else []},
        "trace_plan": {
            "schema": "stage1_trace_plan_v1",
            "detector": "unit",
            "ckpt_status": "loaded",
            "trace_confidence": "high",
            "coverage_scope": "trace_net_only",
            "manual_override_used": False,
            "review_required": False,
            "included_modules": ["backbone"],
            "ignored_layers": [],
            "skipped_subgraphs": [],
            "rejected_candidates": [],
        },
        "view_b1_prune_groups": [
            {"group_id": "backbone", "bucket": "backbone", "member_layers": ["backbone.conv"]}
        ],
        "view_b1_search_groups": [{"search_group_id": "backbone", "bucket": "backbone", "member_b1_groups": ["backbone"]}],
        "view_latency": {"status": "estimated_cpu"},
    }


def _write_manifests(tmp_path: Path) -> list[Path]:
    paths = []
    for model in MODELS:
        path = tmp_path / f"{model}.yaml"
        path.write_text(yaml.safe_dump(_manifest(model)), encoding="utf-8")
        paths.append(path)
    return paths


def test_safe_predictor_and_classifier_preserve_model_specific_blockers(tmp_path: Path) -> None:
    """Static inputs remain gated rather than being promoted to separable claims."""
    paths = _write_manifests(tmp_path)
    evidence_dir = tmp_path / "evidence"

    v0 = predict_manifests(paths, evidence_dir=evidence_dir)
    report = build_classification_report(paths, evidence_dir=evidence_dir)
    json_path = tmp_path / "classification.json"
    markdown_path = tmp_path / "classification.md"
    written = write_classification_report(
        manifests=paths, evidence_dir=evidence_dir, out_json=json_path, out_md=markdown_path
    )

    records = {item["model"]: item for item in v0["predictions"]}
    classes = {item["model"]: item for item in report["models"]}
    assert v0["no_overpromotion"] is True
    assert records["attfuse"]["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert records["v2xvit"]["verdict"] == "JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND"
    assert records["pyramid_lidar"]["verdict"] == "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION"
    assert classes["codriving"]["acceleration_class"] == "SEPARABLE_ACCELERATION"
    assert classes["where2comm"]["acceleration_class"] == "CO_ACCELERATION_REQUIRED"
    assert "full-model separability" in classes["attfuse"]["unsupported_conclusions"]
    assert "Safe Predictor" in markdown_report(v0)
    assert "Model Classification" in render_classification_md(report)
    assert written == report
    assert json_path.is_file() and markdown_path.is_file()


def test_normalization_is_immutable_and_overpromotion_guard_detects_invalid_claims() -> None:
    """Legacy manifests gain safe defaults without mutating the caller's data."""
    source = _manifest("unknown")
    source.pop("trace_plan")
    original = json.loads(json.dumps(source))

    normalized = normalize_manifest_for_predictor(source)
    invalid = {
        "model": "unsafe",
        "scope": "full_model_export",
        "verdict": "MEASURED_SEPARABLE",
        "blockers": ["uncovered_fusion"],
    }

    assert source == original
    assert normalized["trace"]["skipped_subgraphs"] == []
    assert normalized["view_latency"]["coverage"]["full_model_latency_pct"] is None
    assert find_overpromotions([invalid]) == [invalid]
    with pytest.raises(TypeError):
        predict_manifest(42)  # type: ignore[arg-type]


def test_calibrated_predictor_keeps_evidence_limits_for_all_model_families(tmp_path: Path) -> None:
    """Calibration applies distinct model rules while retaining the no-overpromotion invariant."""
    v0_path = tmp_path / "v0.json"
    s2_path = tmp_path / "s2.json"
    s25_path = tmp_path / "s25.json"
    s3_path = tmp_path / "s3.json"
    s4_path = tmp_path / "s4.json"
    base = {"predictions": [predict_manifest(_manifest(model), evidence_dir=tmp_path) for model in MODELS]}
    v0_path.write_text(json.dumps(base), encoding="utf-8")
    s2_path.write_text(json.dumps({"h800_anchor_scan": {"anchors": []}}), encoding="utf-8")
    s25_path.write_text(json.dumps({"gates": {}}), encoding="utf-8")
    s3_path.write_text("{}", encoding="utf-8")
    s4_path.write_text(json.dumps({"anchor_rollup": {}}), encoding="utf-8")

    v1 = build_predictions_v1(
        v0_path=v0_path, s2_path=s2_path, s2_5_path=s25_path, s3_path=s3_path, s4_path=s4_path
    )
    calibrated = build_calibrated_report(v1)
    predictions = {item["model"]: item for item in v1["predictions"]}

    assert v1["no_overpromotion"] is True
    assert predictions["codriving"]["verdict"] == "SCOPED_MEASURED_NEGATIVE_ANCHOR"
    assert predictions["fcooper"]["verdict"] == "PAIR_CALIBRATION_REQUIRED_FULL_MODEL_GATED"
    assert predictions["attfuse"]["verdict"] == "FUSION_UNCOVERED_STATIC_DENSE_PROMOTION_BLOCKED"
    assert predictions["v2xvit"]["verdict"] == "PAIR_OR_JOINT_GATE_FAKE_QUANT_TRUE_TRT_BLOCKED"
    assert predictions["pyramid_lidar"]["verdict"] == "P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND"
    assert predictions["where2comm"]["verdict"] == "ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN"
    assert "V2X-ViT Orin latency measured" in prohibited_claims()
    assert "Calibrated Predictor" in render_predictions_md(v1)
    assert calibrated["critic_status"] == "ACCEPT"
    assert "Calibrated Predictor Report" in render_calibrated_md(calibrated)
