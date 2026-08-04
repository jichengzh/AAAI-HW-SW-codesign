"""Contract tests for the public, anonymous uncertainty replay."""

from __future__ import annotations

import copy
import math
import socket
import subprocess
from typing import Any

import pytest

from framework.stage4.uncertainty_replay_v1 import build_uncertainty_replay


TARGETS = ("latency_ms", "energy_j", "ap70")
ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)
FOLDS = (
    ("f0", "g0", "g4"),
    ("f1", "g1", "g5"),
    ("f2", "g2", "g6"),
    ("f3", "g3", "g7"),
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "stage4_uncertainty_replay_manifest_v1",
        "folds": [
            {
                "fold_id": fold_id,
                "oof_group_id": oof_group_id,
                "candidate_group_id": candidate_group_id,
            }
            for fold_id, oof_group_id, candidate_group_id in FOLDS
        ],
    }


def _observed(target: str, arm_index: int) -> float:
    values = {
        "latency_ms": 1.0 + arm_index,
        "energy_j": 2.0 + arm_index,
        "ap70": 0.55 + arm_index / 100,
    }
    return values[target]


def _interval(target: str, prediction: float) -> tuple[float, float]:
    if target == "ap70":
        return max(0.0, prediction - 0.1), min(1.0, prediction + 0.1)
    return prediction - 0.25, prediction + 0.25


def _record(
    *,
    fold_id: str,
    group_id: str,
    target: str,
    dispatch_key: str,
    q_mode: str,
    arm_index: int,
    oof: bool,
) -> dict[str, object]:
    observed = _observed(target, arm_index)
    prediction = observed + (0.02 if oof else -0.02)
    lower, upper = _interval(target, prediction)
    record: dict[str, object] = {
        "manifest_job_id": f"{group_id}|{dispatch_key}|{q_mode}",
        "group_id": group_id,
        "fold_id": fold_id,
        "dispatch_key": dispatch_key,
        "q_mode": q_mode,
        "target": target,
        "prediction": prediction,
        "lower": lower,
        "upper": upper,
    }
    if oof:
        record["observed"] = observed
    return record


def _oof_predictions() -> dict[str, object]:
    return {
        "schema_version": "stage4_uncertainty_oof_predictions_v1",
        "records": [
            _record(
                fold_id=fold_id,
                group_id=group_id,
                target=target,
                dispatch_key=dispatch_key,
                q_mode=q_mode,
                arm_index=arm_index,
                oof=True,
            )
            for fold_id, group_id, _ in FOLDS
            for arm_index, (dispatch_key, q_mode) in enumerate(ARMS)
            for target in TARGETS
        ],
    }


def _candidate_predictions() -> dict[str, object]:
    return {
        "schema_version": "stage4_uncertainty_candidate_predictions_v1",
        "records": [
            _record(
                fold_id=fold_id,
                group_id=group_id,
                target=target,
                dispatch_key=dispatch_key,
                q_mode=q_mode,
                arm_index=arm_index,
                oof=False,
            )
            for fold_id, _, group_id in FOLDS
            for arm_index, (dispatch_key, q_mode) in enumerate(ARMS)
            for target in TARGETS
        ],
    }


def _build(
    *,
    manifest: dict[str, object] | None = None,
    oof: dict[str, object] | None = None,
    candidates: dict[str, object] | None = None,
) -> Any:
    return build_uncertainty_replay(
        manifest or _manifest(),
        oof or _oof_predictions(),
        candidates or _candidate_predictions(),
    )


def test_builds_an_immutable_anonymous_calibration_and_replay_summary() -> None:
    manifest = _manifest()
    oof = _oof_predictions()
    candidates = _candidate_predictions()
    before = copy.deepcopy((manifest, oof, candidates))
    report = _build(manifest=manifest, oof=oof, candidates=candidates)

    assert set(report) == {
        "schema_version",
        "uncertainty_method",
        "calibration",
        "acquisition_replay",
    }
    assert report["schema_version"] == "stage4_uncertainty_replay_v1"
    assert report["uncertainty_method"] == "lgbm_quantile_plus_group_conformal"
    assert set(report["calibration"]) == set(TARGETS)
    assert "g0" not in repr(report)
    for target in TARGETS:
        summary = report["calibration"][target]
        assert summary["oof_row_count"] == 16
        assert summary["mae"] >= 0
        assert 0.0 <= summary["row_coverage"] <= 1.0
        assert summary["mean_interval_width"] >= 0
    policies = report["acquisition_replay"]["policies"]
    assert set(policies) == {"random", "predicted_frontier_diversity"}
    assert policies["random"]["selected_group_count"] == 4
    assert 1 <= policies["predicted_frontier_diversity"]["selected_group_count"] <= 4
    with pytest.raises(TypeError):
        report["calibration"]["ap70"]["mae"] = 0.0
    assert (manifest, oof, candidates) == before


def test_replay_is_pure_in_memory_without_file_network_or_process_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)

    assert _build()["schema_version"] == "stage4_uncertainty_replay_v1"


def test_rejects_invalid_manifest_fold_or_four_arm_bindings() -> None:
    manifest = _manifest()
    manifest["folds"][0]["candidate_group_id"] = "g0"
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(manifest=manifest)

    oof = _oof_predictions()
    oof["records"] = oof["records"][3:]
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(oof=oof)


def test_rejects_duplicates_nonfinite_values_and_invalid_intervals() -> None:
    oof = _oof_predictions()
    oof["records"][1] = copy.deepcopy(oof["records"][0])
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(oof=oof)

    candidates = _candidate_predictions()
    candidates["records"][0]["prediction"] = math.nan
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(candidates=candidates)

    candidates = _candidate_predictions()
    candidates["records"][0]["lower"] = 2.0
    candidates["records"][0]["prediction"] = 1.0
    candidates["records"][0]["upper"] = 3.0
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(candidates=candidates)


def test_rejects_physical_range_failures_and_candidate_label_leakage() -> None:
    oof = _oof_predictions()
    latency = next(record for record in oof["records"] if record["target"] == "latency_ms")
    latency["observed"] = 0.0
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(oof=oof)

    candidates = _candidate_predictions()
    ap70 = next(record for record in candidates["records"] if record["target"] == "ap70")
    ap70["upper"] = 1.1
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(candidates=candidates)

    candidates = _candidate_predictions()
    candidates["records"][0]["observed"] = 0.5
    with pytest.raises(ValueError, match="invalid uncertainty replay input"):
        _build(candidates=candidates)


def test_fails_closed_without_echoing_untrusted_fields_or_values_and_mutating_inputs() -> None:
    private_marker = "private-user-root-cache-checkpoint-terminal_status-marker"
    candidates = _candidate_predictions()
    before = copy.deepcopy(candidates)
    candidates["records"][0]["root"] = private_marker

    with pytest.raises(ValueError, match="invalid uncertainty replay input") as exc_info:
        _build(candidates=candidates)

    assert private_marker not in str(exc_info.value)
    assert before["records"][1:] == candidates["records"][1:]
