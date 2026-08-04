"""Contract tests for public, prediction-only Stage4 ranking and Pareto selection."""

from __future__ import annotations

import copy
import json
import math
import socket
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from framework.stage4.ranking_pareto_v1 import build_stage4_ranking_pareto_report


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)
MAX_PUBLIC_GROUPS = 256


def _candidate(
    group_id: str,
    backend: str,
    precision: str,
    *,
    latency_ms: float,
    energy_j: float,
    ap70: float,
    uncertainty: float = 0.02,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{group_id}:{backend}:{precision}",
        "group_id": group_id,
        "backend": backend,
        "precision": precision,
        "prediction": {
            "latency_ms": latency_ms,
            "energy_j": energy_j,
            "ap70": ap70,
            "uncertainty": uncertainty,
        },
    }


def _prediction_set() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for group_id, offset in (("g1", 0.0), ("g2", 0.10)):
        for backend, precision in ARMS:
            latency_ms, energy_j, ap70, uncertainty = {
                ("tvm_auto", "fp16"): (2.1, 2.0, 0.76, 0.03),
                ("tvm_auto", "int8"): (1.2, 1.0, 0.72, 0.04),
                ("trt_engine", "fp16"): (1.8, 1.6, 0.93, 0.01),
                ("trt_engine", "int8"): (0.7, 0.8, 0.84, 0.02),
            }[(backend, precision)]
            candidates.append(
                _candidate(
                    group_id,
                    backend,
                    precision,
                    latency_ms=latency_ms - offset,
                    energy_j=energy_j - offset,
                    ap70=ap70 - offset / 10,
                    uncertainty=uncertainty,
                )
            )
    return {
        "schema_version": "stage4_candidate_predictions_v1",
        "candidates": candidates,
    }


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": "stage4_ranking_pareto_manifest_v1",
        "group_ids": ["g1", "g2"],
    }


def _fold() -> dict[str, Any]:
    return {
        "schema_version": "stage4_ranking_pareto_fold_v1",
        "fold_id": "f0",
        "group_ids": ["g1", "g2"],
    }


def _weights() -> dict[str, float]:
    return {
        "latency_ms": 0.4,
        "energy_j": 0.3,
        "ap70": 1.0,
        "uncertainty": 0.2,
    }


def _complete_prediction_set(
    group_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    group_ids = [f"g{index}" for index in range(group_count)]
    candidates = [
        _candidate(
            group_id,
            backend,
            precision,
            latency_ms=1.0 + group_index,
            energy_j=1.0 + group_index,
            ap70=0.9 - group_index / 1_000,
            uncertainty=0.01,
        )
        for group_index, group_id in enumerate(group_ids)
        for backend, precision in ARMS
    ]
    return (
        {
            "schema_version": "stage4_candidate_predictions_v1",
            "candidates": candidates,
        },
        {
            "schema_version": "stage4_ranking_pareto_manifest_v1",
            "group_ids": group_ids,
        },
        {
            "schema_version": "stage4_ranking_pareto_fold_v1",
            "fold_id": "f0",
            "group_ids": group_ids,
        },
    )


def _rename_group(
    prediction_set: dict[str, Any],
    manifest: dict[str, Any],
    fold: dict[str, Any],
    *,
    old: str,
    new: str,
) -> None:
    for candidate in prediction_set["candidates"]:
        if candidate["group_id"] == old:
            candidate["group_id"] = new
            candidate["candidate_id"] = f"{new}:{candidate['backend']}:{candidate['precision']}"
    manifest["group_ids"] = [new if group_id == old else group_id for group_id in manifest["group_ids"]]
    fold["group_ids"] = [new if group_id == old else group_id for group_id in fold["group_ids"]]


def _build(
    predictions: dict[str, Any] | None = None,
    *,
    manifest: dict[str, Any] | None = None,
    fold: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    policy: str = "predicted_frontier_diversity",
) -> Any:
    return build_stage4_ranking_pareto_report(
        predictions or _prediction_set(),
        manifest=manifest or _manifest(),
        fold=fold or _fold(),
        weights=weights or _weights(),
        policy=policy,
    )


def test_ranks_deterministically_and_reports_anonymous_pareto_summary() -> None:
    prediction_set = _prediction_set()
    reversed_prediction_set = copy.deepcopy(prediction_set)
    reversed_prediction_set["candidates"].reverse()

    first = _build(prediction_set)
    second = _build(reversed_prediction_set)

    assert first == second
    assert first["schema_version"] == "stage4_ranking_pareto_v1"
    assert first["row_count"] == 8
    assert first["group_count"] == 2
    assert first["summary"]["retain_ranker"] is False
    assert first["summary"]["selection_policy"] == "predicted_frontier_diversity"
    assert first["pareto"]["summary"] == {
        "candidate_count": 8,
        "frontier_candidate_count": 4,
        "frontier_group_count": 2,
    }
    assert first["pareto"]["frontier_candidate_ids"] == (
        "g1:trt_engine:fp16",
        "g1:trt_engine:int8",
        "g2:trt_engine:fp16",
        "g2:trt_engine:int8",
    )
    assert set(first["ranked_candidate_ids"]) == {
        candidate["candidate_id"] for candidate in prediction_set["candidates"]
    }
    rendered = repr(first).lower()
    assert "latency_ms" not in rendered
    assert "energy_j" not in rendered
    assert "ap70" not in rendered
    assert "uncertainty" not in rendered


def test_result_is_deeply_immutable_json_compatible_and_inputs_are_not_changed() -> None:
    predictions = _prediction_set()
    manifest = _manifest()
    fold = _fold()
    weights = _weights()
    original = copy.deepcopy((predictions, manifest, fold, weights))

    report = _build(predictions, manifest=manifest, fold=fold, weights=weights)

    assert (predictions, manifest, fold, weights) == original
    assert json.loads(json.dumps(report))["summary"]["retain_ranker"] is False
    with pytest.raises(TypeError):
        report["row_count"] = 9
    with pytest.raises(TypeError):
        report["summary"]["group_count"] = 9
    with pytest.raises(AttributeError):
        report["ranked_candidate_ids"].append("g9:tvm_auto:fp16")


def test_execution_is_pure_in_memory_without_file_network_or_subprocess_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)

    report = _build()

    assert report["summary"]["candidate_count"] == 8


@pytest.mark.parametrize(
    "field",
    ("truth", "measured_latency_ms", "terminal_status", "path", "uri", "root", "cache", "checkpoint"),
)
def test_rejects_label_measurement_and_location_leakage_without_echoing_input(field: str) -> None:
    predictions = _prediction_set()
    sentinel = "private-value-must-not-appear"
    predictions["candidates"][0][field] = sentinel

    with pytest.raises(ValueError) as error:
        _build(predictions)

    assert sentinel not in str(error.value)


@pytest.mark.parametrize(
    "value",
    (math.nan, math.inf, -math.inf, -0.01, 0.0, True, "1.0"),
)
def test_rejects_nonfinite_or_nonpositive_latency_and_energy(value: object) -> None:
    for metric in ("latency_ms", "energy_j"):
        predictions = _prediction_set()
        predictions["candidates"][0]["prediction"][metric] = value
        with pytest.raises(ValueError):
            _build(predictions)


@pytest.mark.parametrize("value", (math.nan, math.inf, -0.01, 1.01, True, "0.5"))
def test_rejects_ap_outside_unit_interval_or_non_numeric(value: object) -> None:
    predictions = _prediction_set()
    predictions["candidates"][0]["prediction"]["ap70"] = value

    with pytest.raises(ValueError):
        _build(predictions)


@pytest.mark.parametrize("value", (math.nan, math.inf, -0.01, True, "0.5"))
def test_rejects_negative_or_nonfinite_uncertainty(value: object) -> None:
    predictions = _prediction_set()
    predictions["candidates"][0]["prediction"]["uncertainty"] = value

    with pytest.raises(ValueError):
        _build(predictions)


def test_rejects_nonanonymous_ids_duplicate_ids_duplicate_groups_and_missing_arms() -> None:
    nonanonymous = _prediction_set()
    nonanonymous["candidates"][0]["group_id"] = "model-g1"
    with pytest.raises(ValueError):
        _build(nonanonymous)

    duplicate_id = _prediction_set()
    duplicate_id["candidates"][1]["candidate_id"] = duplicate_id["candidates"][0]["candidate_id"]
    with pytest.raises(ValueError):
        _build(duplicate_id)

    duplicate_group = _manifest()
    duplicate_group["group_ids"] = ["g1", "g1"]
    with pytest.raises(ValueError):
        _build(manifest=duplicate_group)

    missing_arm = _prediction_set()
    missing_arm["candidates"].pop()
    with pytest.raises(ValueError):
        _build(missing_arm)


@pytest.mark.parametrize("invalid_group_id", ("g00", "g10000", "g-1", "group1"))
def test_rejects_noncanonical_or_unbounded_anonymous_group_ids(invalid_group_id: str) -> None:
    predictions = _prediction_set()
    manifest = _manifest()
    fold = _fold()
    _rename_group(predictions, manifest, fold, old="g1", new=invalid_group_id)

    with pytest.raises(ValueError):
        _build(predictions, manifest=manifest, fold=fold)


@pytest.mark.parametrize("invalid_fold_id", ("f00", "f10000", "f-1", "fold1"))
def test_rejects_noncanonical_or_unbounded_fold_ids(invalid_fold_id: str) -> None:
    fold = _fold()
    fold["fold_id"] = invalid_fold_id

    with pytest.raises(ValueError):
        _build(fold=fold)


def test_accepts_canonical_anonymous_id_upper_bound() -> None:
    predictions = _prediction_set()
    manifest = _manifest()
    fold = _fold()
    _rename_group(predictions, manifest, fold, old="g1", new="g9999")
    fold["fold_id"] = "f9999"

    assert _build(predictions, manifest=manifest, fold=fold)["group_count"] == 2


class _OversizedCandidates(Sequence[Mapping[str, Any]]):
    """A sequence that proves the cap is checked before candidate materialization."""

    def __init__(self) -> None:
        self.was_materialized = False

    def __len__(self) -> int:
        return (MAX_PUBLIC_GROUPS + 1) * len(ARMS)

    def __getitem__(self, _index: int) -> Mapping[str, Any]:
        self.was_materialized = True
        raise AssertionError("oversized input must fail before materialization")


def test_rejects_more_than_public_group_limit_before_materializing_candidates() -> None:
    candidates = _OversizedCandidates()
    predictions = {
        "schema_version": "stage4_candidate_predictions_v1",
        "candidates": candidates,
    }

    with pytest.raises(ValueError):
        _build(predictions)
    assert candidates.was_materialized is False


def test_rejects_true_candidate_group_limit() -> None:
    predictions, manifest, fold = _complete_prediction_set(MAX_PUBLIC_GROUPS + 1)

    with pytest.raises(ValueError):
        _build(predictions, manifest=manifest, fold=fold)


class _OversizedGroupIds(Sequence[str]):
    """A sequence that proves metadata group IDs are bounded before copying."""

    def __init__(self) -> None:
        self.was_materialized = False

    def __len__(self) -> int:
        return MAX_PUBLIC_GROUPS + 1

    def __getitem__(self, _index: int) -> str:
        self.was_materialized = True
        raise AssertionError("oversized group metadata must fail before materialization")


@pytest.mark.parametrize("document_name", ("manifest", "fold"))
def test_rejects_oversized_manifest_or_fold_before_materializing_group_ids(
    document_name: str,
) -> None:
    manifest = _manifest()
    fold = _fold()
    group_ids = _OversizedGroupIds()
    if document_name == "manifest":
        manifest["group_ids"] = group_ids
    else:
        fold["group_ids"] = group_ids

    with pytest.raises(ValueError):
        _build(manifest=manifest, fold=fold)
    assert group_ids.was_materialized is False


def test_rejects_invalid_schema_manifest_fold_weights_and_policy() -> None:
    bad_schema = _prediction_set()
    bad_schema["schema_version"] = "wrong"
    with pytest.raises(ValueError):
        _build(bad_schema)

    bad_manifest = _manifest()
    bad_manifest["root"] = "/not-allowed"
    with pytest.raises(ValueError):
        _build(manifest=bad_manifest)

    bad_fold = _fold()
    bad_fold["group_ids"] = ["g1", "g9"]
    with pytest.raises(ValueError):
        _build(fold=bad_fold)

    with pytest.raises(ValueError):
        _build(weights={"latency_ms": 1.0})
    with pytest.raises(ValueError):
        _build(weights={key: -1.0 for key in _weights()})
    with pytest.raises(ValueError):
        _build(weights={key: 0.0 for key in _weights()})
    with pytest.raises(ValueError):
        _build(policy="ranker")
