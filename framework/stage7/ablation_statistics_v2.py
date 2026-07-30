"""Validated descriptive statistics for the Stage7 four-variant selection core."""

from __future__ import annotations

import copy
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from framework.stage7.online_component_ablation_v1 import CORE_VARIANTS, SEEDS


JSON = dict[str, Any]
ROUND_COUNT = 4
BATCH_SIZE = 4
TOTAL_TRAJECTORIES = 12
TOTAL_EVENTS = 192


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    try:
        return (str(row.get("variant") or ""), int(row.get("seed", -1)), int(row.get("round_index", -1)), int(row.get("event_index", -1)))
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal event identity is invalid") from exc


def validate_terminal_events(events: Sequence[Mapping[str, Any]]) -> list[JSON]:
    """Require the complete 4×3×4×4 terminal-event matrix; never estimate gaps."""
    if len(events) != TOTAL_EVENTS:
        raise ValueError("summary requires 12 trajectories and 192 terminal events")
    identities = [_identity(row) for row in events]
    expected = {(variant, seed, round_index, event_index) for variant in CORE_VARIANTS for seed in SEEDS for round_index in range(ROUND_COUNT) for event_index in range(round_index * BATCH_SIZE, (round_index + 1) * BATCH_SIZE)}
    if set(identities) != expected or len(set(identities)) != TOTAL_EVENTS:
        raise ValueError("summary requires complete unique terminal-event identities")
    copied = [copy.deepcopy(dict(row)) for row in events]
    if any(row.get("terminal_status") not in {"completed", "failed"} for row in copied):
        raise ValueError("summary requires explicit terminal feedback")
    if any(isinstance(row.get("delta_hv"), bool) or not isinstance(row.get("delta_hv"), (int, float)) for row in copied):
        raise ValueError("summary requires explicit numeric delta_hv feedback")
    return copied


def trajectory_summaries(events: Sequence[Mapping[str, Any]]) -> list[JSON]:
    """Compute four-checkpoint DeltaHV-AUC solely from explicit terminal feedback."""
    validated = validate_terminal_events(events)
    summaries: list[JSON] = []
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            trajectory = sorted((row for row in validated if row["variant"] == variant and row["seed"] == seed), key=lambda row: int(row["event_index"]))
            checkpoints = [0.0]
            for round_index in range(ROUND_COUNT):
                values = [float(row["delta_hv"]) for row in trajectory if int(row["round_index"]) <= round_index and row["terminal_status"] == "completed"]
                checkpoints.append(max(values, default=0.0))
            auc = sum((left + right) / 2.0 for left, right in zip(checkpoints, checkpoints[1:]))
            summaries.append({"variant": variant, "seed": seed, "delta_hv_auc": auc, "delta_hv_at_16": checkpoints[-1], "delta_hv_checkpoints": checkpoints})
    return summaries


def describe(values: Sequence[float]) -> JSON:
    if not values:
        raise ValueError("descriptive statistics require values")
    numeric = [float(value) for value in values]
    return {"values": numeric, "mean": statistics.fmean(numeric), "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0, "median": statistics.median(numeric), "range": [min(numeric), max(numeric)]}


def paired_statistics(trajectories: Sequence[Mapping[str, Any]]) -> JSON:
    """Describe paired DeltaHV-AUC only; significance tests are intentionally absent."""
    by_key = {(str(row.get("variant")), int(row.get("seed", -1))): row for row in trajectories}
    expected = {(variant, seed) for variant in CORE_VARIANTS for seed in SEEDS}
    if set(by_key) != expected:
        raise ValueError("statistics requires exactly 12 trajectories")
    variants: dict[str, JSON] = {}
    for variant in CORE_VARIANTS:
        values = [float(by_key[(variant, seed)]["delta_hv_auc"]) for seed in SEEDS]
        paired = [value - float(by_key[("full", seed)]["delta_hv_auc"]) for value, seed in zip(values, SEEDS)]
        variants[variant] = {"delta_hv_auc": {**describe(values), "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)}, "paired_delta_vs_full": {**describe(paired), "per_seed": {str(seed): value for seed, value in zip(SEEDS, paired)}, "direction_consistency": {"positive": sum(value > 0 for value in paired), "equal": sum(value == 0 for value in paired), "negative": sum(value < 0 for value in paired)}}}}
    return {"schema_version": "stage7_descriptive_paired_statistics_v2", "statistics_policy": "descriptive_paired_only", "seed_count": len(SEEDS), "significance_tests_performed": False, "variants": variants}


def paper_rows(stats: Mapping[str, Any]) -> list[JSON]:
    """Render compact descriptive rows without inferential claims."""
    rows: list[JSON] = []
    for variant in CORE_VARIANTS:
        metric = stats["variants"][variant]["delta_hv_auc"]
        delta = metric["paired_delta_vs_full"]
        rows.append({"Variant": variant, "DeltaHV-AUC ↑": f"{metric['mean']:.6g}±{metric['sample_std']:.3g}; median {metric['median']:.6g} [{metric['range'][0]:.6g},{metric['range'][1]:.6g}]", "Paired ΔHV-AUC vs Full": f"{delta['mean']:.6g}; +/0/-={delta['direction_consistency']['positive']}/{delta['direction_consistency']['equal']}/{delta['direction_consistency']['negative']}", "Status": "complete"})
    return rows
