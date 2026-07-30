"""Behavioral contracts for the four published Stage7 variants."""

from __future__ import annotations

import random

import pytest

from framework.stage7 import online_component_ablation_v1 as ablation


def _rows() -> list[dict[str, object]]:
    return [
        {
            "row_id": f"row-{index}",
            "width": [16, 32, 64],
            "q_mode": "fp16" if index % 2 else "int8",
            "graph_features": {"conv_count": index + 1},
            "model_features": {
                "static_depth": index + 2,
                "capability_score": 0.5,
            },
            "feature_provenance": {
                "static_depth": "static_config",
                "capability_score": "capability_profile-derived",
            },
        }
        for index in range(8)
    ]


def test_four_variants_are_isolated_with_frozen_budget() -> None:
    """Catches a fifth variant or a change beyond each published ablation."""
    full, variants = ablation.frozen_experiment_contracts()

    assert full["seeds"] == [20260718, 20260719, 20260720]
    assert (full["batch_size"], full["round_count"], full["sample_budget"]) == (4, 4, 16)
    assert full["policy_name"] == "predicted_frontier_diversity"
    assert set({"full", *variants}) == {
        "full", "without_surrogate", "without_measured_feedback", "backend_blind"
    }
    for variant in variants.values():
        assert ablation.audit_single_variable_isolation(full, variant)["verdict"] == "pass"


def test_without_surrogate_uses_seeded_uniform_sampling_without_replacement() -> None:
    """Catches A1 acquiring by a model score, a global RNG, or duplicate identity."""
    rows = _rows()
    selected = ablation.a1_select_unselected(rows, {"row-0"}, seed=20260718)

    assert selected == random.Random(20260718).sample([f"row-{index}" for index in range(1, 8)], 4)
    assert len(selected) == len(set(selected)) == 4


def test_a2_records_feedback_without_refitting_initial_views() -> None:
    """Catches A2 replacing the frozen bundle or appending feedback to its fitting views."""
    initial = [{"row_id": "gold-1", "value": 1}]
    graphs = {"gold-1": {"conv_count": 2}}
    result = ablation.project_a2_feedback(
        initial, graphs, anchor={"pyramid": 0.8}, bundle_sha256="a" * 64,
        selected_results=[{"row_id": "new-1", "latency_ms": 1.0}],
    )

    assert result["training_view"] == initial
    assert result["graph_feature_view"] == graphs
    assert result["recorded_results"] == [{"row_id": "new-1", "latency_ms": 1.0}]
    assert result["feedback_refit"] is False
    assert result["actual_graph_feature_feedback"] is False


def test_backend_blind_removes_only_backend_capability_profile_derived_features() -> None:
    """Catches backend-blind either leaking derived inputs or deleting static/graph inputs."""
    result = ablation.project_backend_blind_features(_rows())

    assert "model:capability_score" in result["removed_names"]
    assert "model:static_depth" in result["blind_schema"]
    assert "graph:conv_count" in result["blind_schema"]
    assert result["leakage_verdict"] == "no_forbidden_features"


def test_selection_view_hides_labels_and_cache_before_selection() -> None:
    """Catches performance, terminal, or cache fields becoming selection inputs."""
    candidate = _rows()[0]
    candidate.update({"latency_ms": 2.0, "cache_disposition": "hit", "terminal_status": "success"})

    with pytest.raises(ValueError, match="forbidden"):
        ablation.selection_candidate_view([candidate])

    clean = ablation.selection_candidate_view(_rows())
    assert all("latency_ms" not in row and "cache_disposition" not in row for row in clean)
    assert _rows() == _rows()  # caller construction remains independent


@pytest.mark.parametrize("label", ["mAP", "AP70", "delta_hv"])
def test_selection_view_rejects_common_metric_aliases(label: str) -> None:
    """Catches metric aliases that would leak a post-measurement label into selection."""
    candidate = _rows()[0]
    candidate[label] = 1.0

    with pytest.raises(ValueError, match="forbidden"):
        ablation.selection_candidate_view([candidate])
