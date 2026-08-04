"""Contract tests for the public, data-free Stage4 closure audit."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from framework.stage4.closure_audit_v1 import build_stage4_closure_audit


def _cost() -> dict[str, Any]:
    return {
        "targets": {
            "latency_ms": {"selected_candidate_counts": {"extra_trees_log": 5}},
            "energy_j": {"selected_candidate_counts": {"extra_trees_log": 4}},
            "ap70": {"selected_candidate_counts": {"lgbm_huber_residual": 3}},
        }
    }


def _completion(random_groups: int, selected_groups: int) -> dict[str, Any]:
    return {
        "summary": {
            "retain_ranker": False,
            "selected_uncertainty_method": "lgbm_quantile",
        },
        "replay": {
            "policies": {
                "predicted_frontier_diversity": {
                    "groups_to_95pct_oracle_HV_median": selected_groups,
                    "success_rate": 1.0,
                },
                "random": {
                    "groups_to_95pct_oracle_HV_median": random_groups,
                    "success_rate": 1.0,
                },
            }
        },
    }


def _feedback() -> dict[str, Any]:
    feedback_groups = ["pyramid|f0", "pyramid|f1", "codriving|f2", "codriving|f3"]
    records = {
        target: [
            {"manifest_job_id": f"{group}|row{arm}", "group_id": group}
            for group in feedback_groups
            for arm in range(4)
        ]
        for target in ("latency_ms", "energy_j", "ap70")
    }
    return {
        "schema_version": "stage4_feedback_update_eval_v1",
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_method": "lgbm_quantile_plus_group_conformal",
        "folds": [
            {
                "heldout_feedback_group": heldout,
                "baseline_fit_groups": ["pyramid|train", "codriving|train"],
                "updated_fit_groups": [
                    "pyramid|train",
                    "codriving|train",
                    *(group for group in feedback_groups if group != heldout),
                ],
                "added_feedback_groups": [
                    group for group in feedback_groups if group != heldout
                ],
                "calibration_groups": ["pyramid|cal", "codriving|cal"],
            }
            for heldout in feedback_groups
        ],
        "before_feedback": {
            "row_count": 16,
            "records": copy.deepcopy(records),
            "targets": {
                target: {
                    "point_metrics": {"mae": 0.04},
                    "interval": {"mean_width": 0.5, "row_coverage": 1.0},
                }
                for target in records
            },
        },
        "after_feedback": {
            "row_count": 16,
            "records": copy.deepcopy(records),
            "targets": {
                target: {
                    "point_metrics": {"mae": 0.03 if target == "ap70" else 0.05},
                    "interval": {
                        "mean_width": 0.45 if target == "ap70" else 0.55,
                        "row_coverage": 1.0,
                    },
                }
                for target in records
            },
        },
    }


def _holdout() -> dict[str, Any]:
    return {
        "schema_version": "stage4_independent_holdout_v1",
        "status": "frozen_before_source_materialization_and_labels",
        "group_count": 4,
        "row_count_after_four_arm_measurement": 16,
        "required_arm_product": [
            ["tvm_auto", "fp16"],
            ["tvm_auto", "int8"],
            ["trt_engine", "fp16"],
            ["trt_engine", "int8"],
        ],
        "groups": [
            {
                "group_id": "pyramid|new0",
                "model": "pyramid",
                "width": [1, 2, 3],
                "label_state_at_freeze": "unavailable",
            },
            {
                "group_id": "pyramid|new1",
                "model": "pyramid",
                "width": [2, 3, 4],
                "label_state_at_freeze": "unavailable",
            },
            {
                "group_id": "codriving|new0",
                "model": "codriving",
                "width": [1, 2, 3],
                "label_state_at_freeze": "unavailable",
            },
            {
                "group_id": "codriving|new1",
                "model": "codriving",
                "width": [2, 3, 4],
                "label_state_at_freeze": "unavailable",
            },
        ],
        "freeze_audit": {
            "overlap_with_gold176_or_feedback16_groups": [],
            "pre_freeze_results_or_script_reference_count_by_width": {
                "1x2x3": 0,
                "2x3x4": 0,
            },
            "forbidden_until_measurement": [
                "latency_ms",
                "energy_j",
                "ap30",
                "ap50",
                "ap70",
                "terminal_status",
            ],
        },
    }


def _reproducibility(exact: bool = True) -> dict[str, Any]:
    names = {
        "gold176_baseline_completion",
        "feedback_cost_model",
        "feedback_update_eval",
        "feedback_completion_ranking_uncertainty_pareto_replay",
    }
    return {
        "schema_version": "stage4_p1_p3_reproducibility_audit_v1",
        "all_exact_match": exact,
        "artifacts": {
            name: {
                "original_sha256": "a" * 64,
                "rerun_sha256": "a" * 64,
                "byte_exact": exact,
            }
            for name in names
        },
    }


def _merged_rows() -> list[dict[str, object]]:
    roles = [
        ("pyramid|train", "initial_coldstart", "train"),
        ("codriving|train", "initial_coldstart", "train"),
        ("pyramid|cal", "initial_coldstart", "locked_holdout"),
        ("codriving|cal", "initial_coldstart", "locked_holdout"),
        ("pyramid|f0", "online_feedback", "online_feedback"),
        ("pyramid|f1", "online_feedback", "online_feedback"),
        ("codriving|f2", "online_feedback", "online_feedback"),
        ("codriving|f3", "online_feedback", "online_feedback"),
    ]
    return [
        {
            "manifest_job_id": f"{group_id}|row{arm}",
            "group_id": group_id,
            "training_source": source,
            "split": split,
            "latency_ms": 1.0,
            "energy_j": 1.0,
            "ap70": 0.5,
        }
        for group_id, source, split in roles
        for arm in range(4)
    ]


def _build(
    *,
    baseline: dict[str, Any] | None = None,
    feedback_completion: dict[str, Any] | None = None,
    feedback_update: dict[str, Any] | None = None,
    holdout: dict[str, Any] | None = None,
    reproducibility: dict[str, Any] | None = None,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    return build_stage4_closure_audit(
        _cost(),
        baseline or _completion(10, 8),
        feedback_completion or _completion(13, 10),
        feedback_update or _feedback(),
        holdout or _holdout(),
        reproducibility_audit=reproducibility or _reproducibility(),
        merged_rows=rows or _merged_rows(),
    )


def test_closes_when_every_public_contract_gate_passes() -> None:
    report = _build()

    assert report["stage4_closed"] is True
    assert report["stage5_search_ready"] is True
    assert report["selected_acquisition_policy"] == "predicted_frontier_diversity"
    assert all(report["gates"].values())
    assert "path" not in repr(report).lower()


def test_rejects_holdout_labels_and_fails_slow_acquisition() -> None:
    holdout = _holdout()
    holdout["groups"][0]["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="label"):
        _build(holdout=holdout)

    report = _build(
        baseline=_completion(10, 12), feedback_completion=_completion(13, 15)
    )
    assert report["gates"]["acquisition_not_worse_than_random"] is False
    assert report["stage4_closed"] is False


def test_requires_exact_reproducibility_and_binding_to_merged_roles() -> None:
    report = _build(reproducibility=_reproducibility(False))
    assert report["gates"]["feedback_update_reproducible"] is False

    rows = _merged_rows()
    for row in rows:
        if row["group_id"] == "pyramid|cal":
            row["split"] = "train"
    assert _build(rows=rows)["gates"]["feedback_update_reproducible"] is False


def test_fails_closed_for_incomplete_or_malformed_protocol_fragments() -> None:
    feedback = _feedback()
    feedback["folds"] = feedback["folds"][:1]
    assert _build(feedback_update=feedback)["gates"]["feedback_update_reproducible"] is False

    holdout = _holdout()
    holdout["groups"] = holdout["groups"][:2]
    assert _build(holdout=holdout)["gates"]["independent_holdout_frozen_before_labels"] is False

    malformed_holdout = _holdout()
    malformed_holdout["groups"][0] = "not-an-object"
    assert _build(holdout=malformed_holdout)["gates"]["independent_holdout_frozen_before_labels"] is False

    malformed_width = _holdout()
    malformed_width["groups"][0]["width"] = "not-an-array"
    assert _build(holdout=malformed_width)["gates"]["independent_holdout_frozen_before_labels"] is False

    malformed_folds = _feedback()
    malformed_folds["folds"][0]["baseline_fit_groups"] = "not-an-array"
    assert _build(feedback_update=malformed_folds)["gates"]["feedback_update_reproducible"] is False


def test_fails_closed_when_random_replay_has_a_non_numeric_success_rate() -> None:
    baseline = _completion(10, 8)
    baseline["replay"]["policies"]["random"]["success_rate"] = "invalid"

    assert _build(baseline=baseline)["gates"]["acquisition_not_worse_than_random"] is False


def test_reproducibility_hash_drift_is_rejected() -> None:
    audit = _reproducibility()
    audit["artifacts"]["feedback_update_eval"]["rerun_sha256"] = "b" * 64

    report = _build(reproducibility=audit)

    assert report["reproducibility_exact_match"] is False
    assert report["gates"]["feedback_update_reproducible"] is False


def test_does_not_echo_untrusted_source_group_or_policy_values() -> None:
    private_marker = "private-location-marker"
    rows = _merged_rows()
    rows[0]["training_source"] = private_marker
    rows[0]["group_id"] = private_marker
    holdout = _holdout()
    holdout["groups"][0]["group_id"] = private_marker
    baseline = _completion(10, 8)
    feedback_completion = _completion(13, 10)
    baseline["replay"]["policies"][private_marker] = baseline["replay"]["policies"].pop(
        "predicted_frontier_diversity"
    )
    feedback_completion["replay"]["policies"][private_marker] = feedback_completion[
        "replay"
    ]["policies"].pop("predicted_frontier_diversity")

    report = _build(
        baseline=baseline,
        feedback_completion=feedback_completion,
        holdout=holdout,
        rows=rows,
    )

    assert private_marker not in repr(report)
    assert report["gates"]["acquisition_not_worse_than_random"] is False
    assert report["gates"]["training_source_roles_explicit"] is False


def test_rejects_an_unapproved_value_head_before_it_can_enter_the_report() -> None:
    cost = _cost()
    cost["targets"]["ap70"]["selected_candidate_counts"] = {
        "private-location-marker": 1
    }

    with pytest.raises(ValueError, match="approved public value head"):
        build_stage4_closure_audit(
            cost,
            _completion(10, 8),
            _completion(13, 10),
            _feedback(),
            _holdout(),
            reproducibility_audit=_reproducibility(),
            merged_rows=_merged_rows(),
        )


def test_requires_after_feedback_rows_to_match_manifest_groups_exactly() -> None:
    mismatched_group = _feedback()
    mismatched_group["after_feedback"]["records"]["ap70"][0]["group_id"] = "wrong-group"
    assert _build(feedback_update=mismatched_group)["gates"]["feedback_update_reproducible"] is False

    extra_after_row = _feedback()
    extra_after_row["after_feedback"]["records"]["ap70"].append(
        copy.deepcopy(extra_after_row["after_feedback"]["records"]["ap70"][0])
    )
    assert _build(feedback_update=extra_after_row)["gates"]["feedback_update_reproducible"] is False


def test_rejects_non_positive_group_medians_and_invalid_success_rates() -> None:
    zero_random = _completion(0, 0)
    report = _build(baseline=zero_random)
    assert report["gates"]["acquisition_not_worse_than_random"] is False

    negative_selected = _completion(10, -1)
    report = _build(baseline=negative_selected)
    assert report["gates"]["acquisition_not_worse_than_random"] is False

    invalid_rate = _completion(10, 8)
    invalid_rate["replay"]["policies"]["random"]["success_rate"] = 1.1
    report = _build(baseline=invalid_rate)
    assert report["gates"]["acquisition_not_worse_than_random"] is False


def test_invalid_acquisition_metrics_cannot_be_echoed_in_the_report() -> None:
    private_marker = "private-location-marker"
    baseline = _completion(10, 8)
    baseline["replay"]["policies"]["predicted_frontier_diversity"][
        "groups_to_95pct_oracle_HV_median"
    ] = private_marker

    report = _build(baseline=baseline)

    assert private_marker not in repr(report)
    assert report["gates"]["acquisition_not_worse_than_random"] is False


def test_rejects_non_integral_candidate_selection_counts() -> None:
    cost = _cost()
    cost["targets"]["latency_ms"]["selected_candidate_counts"] = {
        "extra_trees_log": 1.5
    }

    with pytest.raises(ValueError, match="positive integer"):
        build_stage4_closure_audit(
            cost,
            _completion(10, 8),
            _completion(13, 10),
            _feedback(),
            _holdout(),
            reproducibility_audit=_reproducibility(),
            merged_rows=_merged_rows(),
        )
