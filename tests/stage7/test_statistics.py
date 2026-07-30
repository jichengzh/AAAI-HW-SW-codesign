"""Descriptive-only Stage7 statistics contracts."""

from __future__ import annotations

import pytest

from framework.stage7 import ablation_statistics_v2 as statistics


def _trajectories() -> list[dict[str, object]]:
    rows = []
    for variant_index, variant in enumerate(statistics.CORE_VARIANTS):
        for seed_index, seed in enumerate(statistics.SEEDS):
            rows.append({"variant": variant, "seed": seed, "delta_hv_auc": float(variant_index * 10 + seed_index)})
    return rows


def test_descriptive_paired_statistics_have_hand_checked_values_and_no_significance_test() -> None:
    """Catches population deviation, unpaired deltas, or prohibited significance output."""
    result = statistics.paired_statistics(_trajectories())
    a1 = result["variants"]["without_surrogate"]["delta_hv_auc"]

    assert a1["mean"] == 11.0
    assert a1["sample_std"] == pytest.approx(1.0)
    assert a1["median"] == 11.0
    assert a1["range"] == [10.0, 12.0]
    assert a1["paired_delta_vs_full"]["values"] == [10.0, 10.0, 10.0]
    assert a1["paired_delta_vs_full"]["direction_consistency"] == {"positive": 3, "equal": 0, "negative": 0}
    assert result["significance_tests_performed"] is False
    assert "p_value" not in str(result)


def test_summary_requires_complete_twelve_trajectory_192_terminal_event_contract() -> None:
    """Catches summarizing partial data or filling absent terminal evidence with estimates."""
    with pytest.raises(ValueError, match="12 trajectories"):
        statistics.validate_terminal_events([])

    events = []
    for variant in statistics.CORE_VARIANTS:
        for seed in statistics.SEEDS:
            for event_index in range(16):
                events.append({
                    "variant": variant, "seed": seed, "round_index": event_index // 4,
                    "event_index": event_index, "terminal_status": "completed", "delta_hv": float(event_index),
                })
    assert len(statistics.validate_terminal_events(events)) == 192
    summaries = statistics.trajectory_summaries(events)
    assert len(summaries) == 12
    assert all("delta_hv_auc" in row for row in summaries)
