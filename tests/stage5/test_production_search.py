"""Behavioral tests for the frozen Stage5 production predictor and selector."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5 import production_search_v1 as search


def profiles() -> list[dict[str, Any]]:
    return [
        build_capability_profile(
            capability_profile_id=f"h800-{dispatch}",
            hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(dispatch.encode()).hexdigest(),
            dispatch_key=dispatch,
            features={"int8_propagation": propagation, "qdq_fold": propagation / 2},
        )
        for dispatch, propagation in (("tvm_auto", 0.0), ("trt_engine", 1.0))
    ]


def graph(group_id: str, model: str, width: list[int]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": model,
        "width": list(width),
        "conv_count": 24 if model == "codriving" else 27,
        "conv_macs": float(np.prod(width)),
        "group_conv_count": 0 if model == "codriving" else 3,
    }


def source_lineage(
    group_id: str,
    model: str,
    width: list[int],
    *,
    source_status: str = "ready",
) -> dict[str, Any]:
    """Build fixture-only public lineage bound to stable synthetic evidence."""
    evidence_sha = hashlib.sha256(
        f"stage5-test-fixture-lineage:{group_id}".encode()
    ).hexdigest()
    contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": model,
        "width": list(width),
        "artifact_id": f"public-{model}-{width[0]}",
        "source_status": source_status,
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "external_public_test_fixture",
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "source_status": source_status,
        "source_evidence_sha256": evidence_sha,
        "source_contract": contract,
        "source_contract_sha256": contract_sha,
    }


def training_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    widths = ([16, 32, 64], [24, 48, 96], [32, 64, 128])
    for model_index, model in enumerate(("pyramid", "codriving")):
        for width_index, width in enumerate(widths):
            group_id = f"{model}|{'x'.join(map(str, width))}"
            graphs.append(graph(group_id, model, list(width)))
            for profile in profiles():
                for q_mode in ("fp16", "int8"):
                    tvm_int8 = profile["dispatch_key"] == "tvm_auto" and q_mode == "int8"
                    trt_int8 = profile["dispatch_key"] == "trt_engine" and q_mode == "int8"
                    latency = 1.0 + model_index + width_index * 0.2
                    latency *= 1.25 if tvm_int8 else 0.75 if trt_int8 else 1.0
                    ap70 = 0.76 - width_index * 0.03 - (0.01 if q_mode == "int8" else 0)
                    rows.append(
                        {
                            "manifest_job_id": (
                                f"{group_id}|q={q_mode}|profile="
                                f"{profile['capability_profile_id']}"
                            ),
                            "group_id": group_id,
                            "model": model,
                            "width": list(width),
                            "dispatch_key": profile["dispatch_key"],
                            "capability_profile_id": profile["capability_profile_id"],
                            "q_mode": q_mode,
                            "latency_ms": latency,
                            "energy_j": latency * 0.25,
                            "ap30": min(1.0, ap70 + 0.2),
                            "ap50": min(1.0, ap70 + 0.1),
                            "ap70": ap70,
                            "terminal_status": "measured_success_gold",
                            "training_source": (
                                "initial_coldstart"
                                if width_index < 2
                                else "online_feedback"
                            ),
                        }
                    )
    return rows, graphs


def closure() -> dict[str, Any]:
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "ranker_policy": "rejected_use_value_heads_only",
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 16, "online_feedback": 8},
        "frozen_holdout": {"groups": []},
    }


def registry() -> dict[str, Any]:
    groups = []
    for model in ("pyramid", "codriving"):
        for width in ([40, 80, 160], [48, 96, 192]):
            group_id = f"{model}|{'x'.join(map(str, width))}"
            groups.append(
                {
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    **source_lineage(group_id, model, width),
                    "graph_features": graph(group_id, model, width),
                }
            )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


def predicted_group(
    group_id: str,
    *,
    model: str = "pyramid",
    width: list[int],
    latency: float = 1.0,
    uncertainty: float = 0.2,
) -> list[dict[str, Any]]:
    result = []
    for profile in profiles():
        for q_mode in ("fp16", "int8"):
            row_id = (
                f"{group_id}|q={q_mode}|profile={profile['capability_profile_id']}"
            )
            result.append(
                {
                    "row_id": row_id,
                    "manifest_job_id": row_id,
                    "group_id": group_id,
                    "model": model,
                    "width": list(width),
                    "q_mode": q_mode,
                    "capability_profile_id": profile["capability_profile_id"],
                    "dispatch_key": profile["dispatch_key"],
                    "graph_features": graph(group_id, model, width),
                    "predictions": {
                        "latency_ms": latency,
                        "energy_j": latency / 4,
                        "ap70": 0.7,
                    },
                    "prediction_intervals": {
                        target: {
                            "lower": value - uncertainty,
                            "median": value,
                            "upper": value + uncertainty,
                        }
                        for target, value in {
                            "latency_ms": latency,
                            "energy_j": latency / 4,
                            "ap70": 0.7,
                        }.items()
                    },
                }
            )
    return result


def test_fit_uses_seed_bound_real_extra_trees_and_lightgbm_without_mutating_inputs() -> None:
    """Catches estimator substitution, seed drift, or caller-input mutation."""
    rows, graphs = training_data()
    profile_rows = profiles()
    closure_payload = closure()
    originals = copy.deepcopy((rows, graphs, profile_rows, closure_payload))

    first = search.fit_production_bundle(
        rows, graphs, profile_rows, closure_payload, seed=73
    )
    second = search.fit_production_bundle(
        rows, graphs, profile_rows, closure_payload, seed=73
    )

    assert isinstance(first.value_heads["latency_ms"], ExtraTreesRegressor)
    assert isinstance(first.value_heads["energy_j"], ExtraTreesRegressor)
    assert isinstance(first.value_heads["ap70"], LGBMRegressor)
    assert first.manifest == second.manifest
    assert first.manifest["seed"] == 73
    assert first.value_heads["latency_ms"].random_state == 73
    assert first.value_heads["ap70"].random_state == 73
    assert (rows, graphs, profile_rows, closure_payload) == originals


def test_fit_predict_preserves_frozen_heads_transforms_and_conformal_intervals() -> None:
    """Catches loss of model heads, target transforms, calibration, or intervals."""
    rows, graphs = training_data()
    bundle = search.fit_production_bundle(rows, graphs, profiles(), closure(), seed=7)
    manifest = search.build_candidate_manifest(
        registry(),
        measured_group_ids={row["group_id"] for row in rows},
        frozen_holdout={"groups": []},
        capability_profiles=profiles(),
    )

    predicted = search.predict_candidate_rows(bundle, manifest["rows"], profiles())

    assert bundle.manifest["canonical_value_heads"] == closure()["canonical_value_heads"]
    assert bundle.manifest["uncertainty_policy"] == (
        "lgbm_quantile_plus_group_conformal"
    )
    assert bundle.manifest["target_transforms"] == {
        "latency_ms": "log1p",
        "energy_j": "log1p",
        "ap70": "residual_from_model_anchor",
    }
    assert set(bundle.conformal_corrections) == {"latency_ms", "energy_j", "ap70"}
    assert len(predicted) == 16
    for row in predicted:
        assert set(row["predictions"]) == {"latency_ms", "energy_j", "ap70"}
        for target, value in row["predictions"].items():
            assert math.isfinite(value), target
            interval = row["prediction_intervals"][target]
            assert interval["lower"] <= interval["median"] <= interval["upper"]


def test_selection_hides_candidate_labels_is_deterministic_and_never_duplicates() -> None:
    """Catches label leakage, unstable ordering, or duplicate selections."""
    candidates = [
        *predicted_group("pyramid|40x80x160", width=[40, 80, 160], latency=1.0),
        *predicted_group("pyramid|48x96x192", width=[48, 96, 192], latency=1.2),
    ]
    original = copy.deepcopy(candidates)

    first = search.select_predicted_frontier_diversity(
        candidates,
        measured_rows=[],
        measured_graph_features=[],
        group_budget_by_model={"pyramid": 2},
    )
    second = search.select_predicted_frontier_diversity(
        list(reversed(candidates)),
        measured_rows=[],
        measured_graph_features=[],
        group_budget_by_model={"pyramid": 2},
    )

    assert first["candidate_labels_visible_before_measurement"] is False
    assert first["selected_group_ids"] == second["selected_group_ids"]
    selected_ids = [row["row_id"] for row in first["selected_rows"]]
    assert selected_ids == sorted(selected_ids)
    assert len(selected_ids) == len(set(selected_ids)) == 8
    assert candidates == original

    leaked = copy.deepcopy(candidates)
    leaked[0]["latency_ms"] = 0.01
    with pytest.raises(ValueError, match="labels visible"):
        search.select_predicted_frontier_diversity(
            leaked,
            measured_rows=[],
            measured_graph_features=[],
            group_budget_by_model={"pyramid": 1},
        )


@pytest.mark.parametrize(
    "nested_field",
    ["measured_latency_ms", "cache_status", "terminal_status"],
)
def test_selection_rejects_nested_candidate_label_cache_or_terminal_status(
    nested_field: str,
) -> None:
    """Catches forbidden pre-selection context hidden inside source contracts."""
    candidates = [
        *predicted_group("pyramid|40x80x160", width=[40, 80, 160]),
        *predicted_group("pyramid|48x96x192", width=[48, 96, 192]),
    ]
    candidates[0]["source_contract"] = {nested_field: "secret"}

    with pytest.raises(ValueError, match="forbidden candidate field"):
        search.select_predicted_frontier_diversity(
            candidates,
            measured_rows=[],
            measured_graph_features=[],
            group_budget_by_model={"pyramid": 1},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["predictions"].update(terminal_status="secret"),
        lambda row: row["prediction_intervals"].update(cache_status={"hit": True}),
        lambda row: row["prediction_intervals"]["latency_ms"].update(
            measurement_status="secret"
        ),
    ],
)
def test_selection_rejects_extra_prediction_or_interval_context(mutation) -> None:
    """Catches status/cache bypasses hidden in model-output subtrees."""
    candidates = [
        *predicted_group("pyramid|40x80x160", width=[40, 80, 160]),
        *predicted_group("pyramid|48x96x192", width=[48, 96, 192]),
    ]
    mutation(candidates[0])

    with pytest.raises(ValueError, match="prediction|interval"):
        search.select_predicted_frontier_diversity(
            candidates,
            measured_rows=[],
            measured_graph_features=[],
            group_budget_by_model={"pyramid": 1},
        )


def test_candidate_source_sha_must_be_hex_and_errors_redact_group_identity() -> None:
    """Catches pseudo-SHA lineage and public disclosure of candidate identity."""
    source_registry = registry()
    source_registry["groups"][0]["source_evidence_sha256"] = "z" * 64

    with pytest.raises(ValueError) as captured:
        search.build_candidate_manifest(
            source_registry,
            measured_group_ids=set(),
            frozen_holdout={"groups": []},
            capability_profiles=profiles(),
        )

    assert "SHA256" in str(captured.value)

    source_registry = registry()
    source_registry["groups"][0]["group_id"] = "secret-group-identity"
    with pytest.raises(ValueError) as redacted:
        search.build_candidate_manifest(
            source_registry,
            measured_group_ids=set(),
            frozen_holdout={"groups": []},
            capability_profiles=profiles(),
        )
    assert "secret-group-identity" not in str(redacted.value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda group: group.update(source_contract="not-an-object"),
        lambda group: group["source_contract"].pop("artifact_id"),
        lambda group: group["source_contract"].update(group_id="drifted"),
        lambda group: group["source_contract"].update(model="codriving"),
        lambda group: group["source_contract"].update(width=[1, 2, 3]),
        lambda group: group["source_contract"].update(source_status="materializable"),
        lambda group: group["source_contract"].update(
            source_evidence_sha256=hashlib.sha256(b"different-evidence").hexdigest()
        ),
        lambda group: group.update(
            source_contract_sha256=hashlib.sha256(b"different-contract").hexdigest()
        ),
    ],
)
def test_candidate_source_contract_is_object_identity_and_digest_bound(mutation) -> None:
    """Catches missing, drifted, or digest-unbound public source contracts."""
    source_registry = registry()
    mutation(source_registry["groups"][0])

    with pytest.raises(ValueError) as captured:
        search.build_candidate_manifest(
            source_registry,
            measured_group_ids=set(),
            frozen_holdout={"groups": []},
            capability_profiles=profiles(),
        )

    assert source_registry["groups"][0]["group_id"] not in str(captured.value)


@pytest.mark.parametrize(
    "measured_graph_features",
    [
        ["not-an-object"],
        [{"conv_count": 1}],
        [{"group_id": "measured", "model": "pyramid", "width": [1, 2, 3]}],
        [{"group_id": "measured", "observed_latency_ms": 1.0}],
        [{"group_id": "measured", "measured_energy_j": 1.0}],
        [{"group_id": "measured", "cache_status": 1.0}],
        [{"group_id": "measured", "terminal_status": 1.0}],
        [{"group_id": "measured", "conv_count": float("nan")}],
    ],
)
def test_selection_rejects_invalid_direct_measured_graph_features(
    measured_graph_features: list[Any],
) -> None:
    """Catches direct-API graph leakage, empty payloads, and malformed records."""
    candidates = [
        *predicted_group("pyramid|40x80x160", width=[40, 80, 160]),
        *predicted_group("pyramid|48x96x192", width=[48, 96, 192]),
    ]

    with pytest.raises(ValueError, match="graph"):
        search.select_predicted_frontier_diversity(
            candidates,
            measured_rows=[],
            measured_graph_features=measured_graph_features,
            group_budget_by_model={"pyramid": 1},
        )


@pytest.mark.parametrize(
    "graph_payload",
    [
        None,
        {},
        "not-an-object",
        {"group_id": "different", "conv_count": 1},
    ],
)
def test_selection_requires_nonempty_group_bound_candidate_graph_payload(
    graph_payload: Any,
) -> None:
    """Catches candidates with absent, empty, malformed, or cross-group graph data."""
    candidates = [
        *predicted_group("pyramid|40x80x160", width=[40, 80, 160]),
        *predicted_group("pyramid|48x96x192", width=[48, 96, 192]),
    ]
    candidates[0]["graph_features"] = graph_payload

    with pytest.raises(ValueError, match="graph"):
        search.select_predicted_frontier_diversity(
            candidates,
            measured_rows=[],
            measured_graph_features=[],
            group_budget_by_model={"pyramid": 1},
        )


def test_diversity_is_the_documented_tie_break_before_uncertainty_and_identity() -> None:
    """Catches replacement or reordering of the frozen diversity tie-break."""
    near = predicted_group(
        "pyramid|24x48x96", width=[24, 48, 96], latency=1.0, uncertainty=9.0
    )
    far = predicted_group(
        "pyramid|64x128x256", width=[64, 128, 256], latency=1.0, uncertainty=0.01
    )
    measured = [
        {
            "group_id": "pyramid|16x32x64",
            "model": "pyramid",
            "width": [16, 32, 64],
        }
    ]

    selected = search.select_predicted_frontier_diversity(
        [*near, *far],
        measured_rows=measured,
        measured_graph_features=[
            graph("pyramid|16x32x64", "pyramid", [16, 32, 64])
        ],
        group_budget_by_model={"pyramid": 1},
    )

    assert selected["selected_group_ids"] == ["pyramid|64x128x256"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_selection_rejects_nonfinite_predictions_and_intervals(value: float) -> None:
    """Catches NaN/Inf silently corrupting Pareto and diversity ranking."""
    candidates = predicted_group("pyramid|40x80x160", width=[40, 80, 160])
    candidates.extend(
        predicted_group("pyramid|48x96x192", width=[48, 96, 192])
    )
    candidates[0]["predictions"]["latency_ms"] = value

    with pytest.raises(ValueError, match="finite"):
        search.select_predicted_frontier_diversity(
            candidates,
            measured_rows=[],
            measured_graph_features=[],
            group_budget_by_model={"pyramid": 1},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(conv_count="not-numeric"),
        lambda item: item.update(conv_count=float("nan")),
        lambda item: item.update(observed_latency_ms=1.0),
    ],
)
def test_fit_rejects_malformed_or_label_leaking_graph_features(mutation) -> None:
    """Catches malformed graph context and nested target leakage before fitting."""
    rows, graphs = training_data()
    mutation(graphs[0])

    with pytest.raises(ValueError, match="graph|finite|numeric|label"):
        search.fit_production_bundle(rows, graphs, profiles(), closure(), seed=7)


@pytest.mark.parametrize("target", ["latency_ms", "energy_j", "ap70"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_fit_rejects_nonfinite_training_metrics(target: str, value: float) -> None:
    """Catches invalid measured labels entering target transforms."""
    rows, graphs = training_data()
    rows[0][target] = value

    with pytest.raises(ValueError, match="finite|metric|value"):
        search.fit_production_bundle(rows, graphs, profiles(), closure(), seed=7)
