"""Behavioral and redaction tests for the public Stage2 outlier policy."""

from __future__ import annotations

from typing import Any

import pytest

from framework.stage2.latency_outlier_policy_v1 import (
    classify_latency_row,
    detect_latency_outliers,
)


def _row(**overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": "latency_lut_row_v1",
        "measurement_status": "measured",
        "model": "pyramid",
        "config_id": "width-16x32x64",
        "schedule_policy": "fixed_schedule",
        "row_id": "row-a",
        "run_id": "run-a",
        "candidate_id": "candidate-a",
        "latency_p50_us": 6_213.0,
        "latency_min_us": 6_210.0,
        "latency_max_us": 6_218.0,
    }
    return {**row, **overrides}


def test_classifies_stable_and_repeat_unstable_rows_without_echoing_identifiers() -> None:
    private_marker = "private-location-marker"
    report = detect_latency_outliers(
        [
            _row(row_id=private_marker, run_id=private_marker),
            _row(
                row_id="other-row",
                run_id="other-run",
                latency_p50_us=30_169.297,
                latency_min_us=21_621.723,
                latency_max_us=46_872.504,
            ),
        ]
    )

    assert report["total_rows"] == 2
    assert report["claimable_rows"] == 1
    assert report["unstable_rows"] == 1
    assert report["rows"][0]["row_index"] == 0
    assert report["rows"][0]["quality_flag"] == "stable"
    assert report["rows"][1]["quality_flag"] == "unstable_repeat"
    assert "repeat_variability_exceeds_threshold" in report["rows"][1]["reasons"]
    assert private_marker not in repr(report)


def test_non_measured_and_malformed_rows_fail_closed_without_echoing_values() -> None:
    private_marker = "private-location-marker"
    report = detect_latency_outliers(
        [
            _row(measurement_status=private_marker),
            _row(latency_min_us=private_marker),
            "not-a-row",
        ]
    )

    assert [row["quality_flag"] for row in report["rows"]] == [
        "not_measured",
        "invalid_latency",
        "invalid_row",
    ]
    assert report["claimable_rows"] == 0
    assert private_marker not in repr(report)


def test_paper_grade_rejects_multi_run_spread_and_anchor_drift() -> None:
    spread = detect_latency_outliers(
        [
            _row(latency_p50_us=6_000.0),
            _row(latency_p50_us=7_000.0),
        ],
        grade="paper",
    )
    drift = detect_latency_outliers(
        [_row(latency_p50_us=7_000.0)],
        grade="paper",
        historical_anchors_ms={"pyramid|width-16x32x64|fixed_schedule": 6.0},
    )

    assert all(row["quality_flag"] == "unstable_multi_run" for row in spread["rows"])
    assert drift["rows"][0]["quality_flag"] == "anchor_drift"
    assert drift["rows"][0]["claim_status"] == "no_claim"


def test_invalid_grade_and_anchor_are_rejected_or_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported grade"):
        detect_latency_outliers([_row()], grade="private-location-marker")

    report = detect_latency_outliers(
        [_row()],
        historical_anchors_ms={"pyramid|width-16x32x64|fixed_schedule": 0.0},
    )

    assert report["rows"][0]["quality_flag"] == "invalid_anchor"
    assert report["claimable_rows"] == 0


def test_input_rows_are_not_mutated_and_report_has_no_source_identity_fields() -> None:
    row = _row()
    original = dict(row)

    report = detect_latency_outliers([row])

    assert row == original
    assert set(report["rows"][0]).isdisjoint(
        {"row_id", "run_id", "config_id", "model", "candidate_id", "schedule_policy"}
    )


def test_public_function_boundaries_reject_malformed_direct_inputs() -> None:
    with pytest.raises(ValueError, match="row must be a mapping"):
        classify_latency_row("not-a-row")
    with pytest.raises(ValueError, match="row_index"):
        classify_latency_row(_row(), row_index="private-location-marker")
    with pytest.raises(ValueError, match="group_p50_ms"):
        classify_latency_row(_row(), group_p50_ms=["private-location-marker"])
    with pytest.raises(ValueError, match="must not be empty"):
        detect_latency_outliers([])
    with pytest.raises(ValueError, match="unsupported grade"):
        detect_latency_outliers([_row()], grade=[])


def test_missing_or_non_string_group_keys_cannot_be_claimable() -> None:
    private_marker = "private-location-marker"
    report = detect_latency_outliers(
        [
            _row(model=None, config_id=None, schedule_policy=None),
            _row(model=1, config_id=private_marker, schedule_policy=object()),
        ]
    )

    assert [row["quality_flag"] for row in report["rows"]] == [
        "invalid_group",
        "invalid_group",
    ]
    assert report["claimable_rows"] == 0
    assert private_marker not in repr(report)


def test_extreme_numeric_inputs_fail_closed_instead_of_raising() -> None:
    huge_number = 10**10_000
    malformed_latency = detect_latency_outliers([_row(latency_min_us=huge_number)])
    malformed_anchor = detect_latency_outliers(
        [_row()],
        historical_anchors_ms={"pyramid|width-16x32x64|fixed_schedule": huge_number},
    )

    assert malformed_latency["rows"][0]["quality_flag"] == "invalid_latency"
    assert malformed_anchor["rows"][0]["quality_flag"] == "invalid_anchor"
