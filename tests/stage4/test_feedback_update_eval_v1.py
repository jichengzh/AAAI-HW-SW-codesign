"""Contract tests for the public, in-memory feedback-update evaluator."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
from typing import Any

import pytest

from framework.stage4.feedback_update_eval_v1 import build_feedback_update_evaluation


TARGETS = ("latency_ms", "energy_j", "ap70")
ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)
CANONICAL_HEADS = {
    "latency_ms": "extra_trees_log",
    "energy_j": "extra_trees_log",
    "ap70": "lgbm_huber_residual",
}


def _rows() -> list[dict[str, object]]:
    roles = (
        ("g0", "initial_coldstart", "train"),
        ("g1", "initial_coldstart", "train"),
        ("g2", "initial_coldstart", "locked_holdout"),
        ("g3", "initial_coldstart", "locked_holdout"),
        ("g4", "online_feedback", "online_feedback"),
        ("g5", "online_feedback", "online_feedback"),
        ("g6", "online_feedback", "online_feedback"),
        ("g7", "online_feedback", "online_feedback"),
    )
    return [
        {
            "manifest_job_id": f"{group_id}|{dispatch}|{precision}",
            "group_id": group_id,
            "training_source": source,
            "split": split,
            "dispatch_key": dispatch,
            "q_mode": precision,
            "latency_ms": 1.0 + arm_index,
            "energy_j": 2.0 + arm_index,
            "ap70": 0.5 + arm_index / 100,
        }
        for group_id, source, split in roles
        for arm_index, (dispatch, precision) in enumerate(ARMS)
    ]


def _predictions(
    rows: list[dict[str, object]], *, after: bool
) -> dict[str, list[dict[str, object]]]:
    feedback_rows = [
        row for row in rows if row["training_source"] == "online_feedback"
    ]
    delta = 0.0 if after else 0.1
    return {
        target: [
            {
                "manifest_job_id": row["manifest_job_id"],
                "group_id": row["group_id"],
                "prediction": float(row[target]) + delta,
                "lower": float(row[target]) - 0.25,
                "upper": float(row[target]) + 0.25,
            }
            for row in feedback_rows
        ]
        for target in TARGETS
    }


def _build(
    *,
    rows: list[dict[str, object]] | None = None,
    before: dict[str, list[dict[str, object]]] | None = None,
    after: dict[str, list[dict[str, object]]] | None = None,
    heads: dict[str, str] | None = None,
) -> dict[str, Any]:
    input_rows = rows or _rows()
    return build_feedback_update_evaluation(
        input_rows,
        before or _predictions(input_rows, after=False),
        after or _predictions(input_rows, after=True),
        heads or CANONICAL_HEADS,
    )


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


def _cost() -> dict[str, Any]:
    return {
        "targets": {
            target: {"selected_candidate_counts": {head: 1}}
            for target, head in CANONICAL_HEADS.items()
        }
    }


def _holdout() -> dict[str, Any]:
    return {
        "schema_version": "stage4_independent_holdout_v1",
        "status": "frozen_before_source_materialization_and_labels",
        "group_count": 4,
        "row_count_after_four_arm_measurement": 16,
        "required_arm_product": [list(arm) for arm in ARMS],
        "groups": [
            {
                "group_id": f"pyramid|new{index}",
                "model": "pyramid" if index < 2 else "codriving",
                "width": [1, 2, 3],
                "label_state_at_freeze": "unavailable",
            }
            for index in range(4)
        ],
        "freeze_audit": {
            "overlap_with_gold176_or_feedback16_groups": [],
            "pre_freeze_results_or_script_reference_count_by_width": {"1x2x3": 0},
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


def _reproducibility() -> dict[str, Any]:
    return {
        "schema_version": "stage4_p1_p3_reproducibility_audit_v1",
        "all_exact_match": True,
        "artifacts": {
            name: {
                "original_sha256": "a" * 64,
                "rerun_sha256": "a" * 64,
                "byte_exact": True,
            }
            for name in (
                "gold176_baseline_completion",
                "feedback_cost_model",
                "feedback_update_eval",
                "feedback_completion_ranking_uncertainty_pareto_replay",
            )
        },
    }


def test_builds_a_closure_compatible_whitelisted_report() -> None:
    rows = _rows()
    report = _build(rows=rows)

    assert set(report) == {
        "schema_version",
        "canonical_value_heads",
        "uncertainty_method",
        "folds",
        "before_feedback",
        "after_feedback",
    }
    assert report["schema_version"] == "stage4_feedback_update_eval_v1"
    assert report["canonical_value_heads"] == CANONICAL_HEADS
    assert report["uncertainty_method"] == "lgbm_quantile_plus_group_conformal"
    assert len(report["folds"]) == 4
    assert report["before_feedback"]["row_count"] == 16
    assert report["after_feedback"]["row_count"] == 16
    for phase in ("before_feedback", "after_feedback"):
        for target in TARGETS:
            target_report = report[phase]["targets"][target]
            assert set(target_report) == {"point_metrics", "interval"}
            assert target_report["point_metrics"]["mae"] >= 0
            assert target_report["interval"]["mean_width"] >= 0
            assert 0.0 <= target_report["interval"]["row_coverage"] <= 1.0
            assert all(
                set(record) == {"manifest_job_id", "group_id"}
                for record in report[phase]["records"][target]
            )
            assert all(
                str(record["group_id"]).startswith("g")
                and str(record["manifest_job_id"]).startswith(f"{record['group_id']}|")
                for record in report[phase]["records"][target]
            )

    from framework.stage4.closure_audit_v1 import build_stage4_closure_audit

    closure = build_stage4_closure_audit(
        _cost(),
        _completion(10, 8),
        _completion(12, 9),
        report,
        _holdout(),
        reproducibility_audit=_reproducibility(),
        merged_rows=rows,
    )
    assert closure["gates"]["feedback_update_reproducible"] is True


def test_rejects_duplicate_or_missing_feedback_measurement_groups() -> None:
    rows = _rows()
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["manifest_job_id"] = duplicate[-2]["manifest_job_id"]
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=duplicate)

    missing_group = [row for row in rows if row["group_id"] != "g7"]
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=missing_group)


def test_rejects_nonfinite_predictions_and_invalid_intervals() -> None:
    rows = _rows()
    invalid_prediction = _predictions(rows, after=False)
    invalid_prediction["ap70"][0]["prediction"] = math.nan
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=rows, before=invalid_prediction)

    invalid_interval = _predictions(rows, after=True)
    invalid_interval["latency_ms"][0]["lower"] = 3.0
    invalid_interval["latency_ms"][0]["prediction"] = 2.0
    invalid_interval["latency_ms"][0]["upper"] = 4.0
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=rows, after=invalid_interval)

    invalid_measurement = _rows()
    invalid_measurement[0]["energy_j"] = -1.0
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=invalid_measurement)

    invalid_ap = _rows()
    invalid_ap[0]["ap70"] = 1.1
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=invalid_ap)


def test_rejects_prediction_group_binding_and_before_after_set_drift() -> None:
    rows = _rows()
    wrong_group = _predictions(rows, after=False)
    wrong_group["energy_j"][0]["group_id"] = "g5"
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=rows, before=wrong_group)

    drifted_after = _predictions(rows, after=True)
    drifted_after["energy_j"][0]["manifest_job_id"] = "g7|drift"
    with pytest.raises(ValueError, match="invalid feedback input"):
        _build(rows=rows, after=drifted_after)


def test_fails_closed_without_echoing_untrusted_fields_or_values() -> None:
    private_marker = "private-user-root-cache-checkpoint-marker"
    rows = _rows()
    rows[0]["root"] = private_marker

    with pytest.raises(ValueError, match="invalid feedback input") as exc_info:
        _build(rows=rows)

    assert private_marker not in str(exc_info.value)


def test_rejects_non_anonymous_group_and_row_identifiers_without_echoing_them() -> None:
    private_marker = "untrusted-lab-identity"
    rows = _rows()
    for row in rows:
        if row["group_id"] == "g4":
            row["group_id"] = private_marker
            row["manifest_job_id"] = (
                f"{private_marker}|{row['dispatch_key']}|{row['q_mode']}"
            )
    before = _predictions(rows, after=False)
    after = _predictions(rows, after=True)

    with pytest.raises(ValueError, match="invalid feedback input") as exc_info:
        _build(rows=rows, before=before, after=after)

    assert private_marker not in str(exc_info.value)


def test_stage4_selection_accessor_remains_available_after_lightweight_import() -> None:
    from framework.stage4 import run_nested_selection

    assert callable(run_nested_selection)


def test_feedback_evaluator_import_does_not_load_numpy() -> None:
    script = (
        "import sys;"
        "import framework.stage4.feedback_update_eval_v1;"
        "print('numpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
