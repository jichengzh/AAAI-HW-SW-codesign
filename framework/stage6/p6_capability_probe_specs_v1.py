"""Canonical structural probe partition and feature aggregation for P6 RTX."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any


RECORD_SCHEMA_VERSION = "p6_compiler_capability_probe_record_v1"
Q_MODES = ("fp16", "int8")
NEUTRAL_PROBE_IDS = ("P1", "P2", "P3", "P4", "P5", "P6")
PRUNING_PROBE_IDS = (
    "aligned_channels",
    "misaligned_channels",
    "small_channels",
    "group_packed",
    "group_unpacked",
    "off_diagonal_two_stage_boundary",
)
FEATURE_NAMES = (
    "s1q_probe_pair_coverage",
    "s1q_build_success_coverage",
    "s1q_fp16_build_success_coverage",
    "s1q_int8_build_success_coverage",
    "s1q_int8_propagation_ratio_mean",
    "s1q_int8_propagation_observed_coverage",
    "s1q_qdq_fold_ratio_mean",
    "s1q_qdq_fold_observed_coverage",
    "s1q_reformat_rate_mean",
    "s1q_reformat_observed_coverage",
    "s1q_fusion_coverage_mean",
    "s1q_fusion_observed_coverage",
    "s1p_probe_pair_coverage",
    "s1p_probe_group_count_observed",
    "s1p_build_success_coverage",
    "s1p_fp16_build_success_coverage",
    "s1p_int8_build_success_coverage",
    "s1p_fp16_group_build_rate_stddev",
    "s1p_int8_group_build_rate_stddev",
    "s1p_fp16_group_reformat_rate_stddev",
    "s1p_int8_group_reformat_rate_stddev",
    "s1p_int8_group_propagation_rate_stddev",
    "s1p_int8_group_qdq_fold_rate_stddev",
)
RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "probe_id",
        "q_mode",
        "onnx_sha256",
        "build_success",
        "observation_status",
        "compiler_ir_sha256",
        "int8_propagated_ops",
        "precision_eligible_ops",
        "qdq_folded_pairs",
        "qdq_pairs",
        "reformat_ops",
        "total_ops",
        "fused_ops",
        "fusible_ops",
    }
)
COUNT_FIELDS = (
    "int8_propagated_ops",
    "precision_eligible_ops",
    "qdq_folded_pairs",
    "qdq_pairs",
    "reformat_ops",
    "total_ops",
    "fused_ops",
    "fusible_ops",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _count(value: object, *, allow_none: bool) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("probe observation count is invalid")
    return value


def _validate_record(raw: object, *, expected_probe_ids: frozenset[str]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != RECORD_FIELDS:
        raise ValueError("probe observation record is invalid")
    record = dict(raw)
    if (
        record.get("schema_version") != RECORD_SCHEMA_VERSION
        or record.get("probe_id") not in expected_probe_ids
        or record.get("q_mode") not in Q_MODES
        or not _is_sha256(record.get("onnx_sha256"))
        or not isinstance(record.get("build_success"), bool)
    ):
        raise ValueError("probe observation record is invalid")
    success = record["build_success"]
    if record.get("observation_status") != (
        "observed_build_success" if success else "observed_build_failure"
    ):
        raise ValueError("probe observation is missing")
    ir_digest = record.get("compiler_ir_sha256")
    if success != _is_sha256(ir_digest) or not success and ir_digest is not None:
        raise ValueError("probe compiler IR evidence is invalid")
    for name in COUNT_FIELDS:
        record[name] = _count(
            record.get(name),
            allow_none=not success
            or record["q_mode"] == "fp16"
            and name
            in {"int8_propagated_ops", "precision_eligible_ops", "qdq_folded_pairs", "qdq_pairs"},
        )
    return record


def validate_probe_partition(
    records: object, *, probe_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Require exactly one observed FP16 and INT8 cell for every tracked probe."""
    if not isinstance(records, list):
        raise ValueError("probe observations must be an array")
    allowed_ids = frozenset(probe_ids)
    validated = tuple(
        _validate_record(record, expected_probe_ids=allowed_ids) for record in records
    )
    identities = [(row["probe_id"], row["q_mode"]) for row in validated]
    expected = {(probe_id, q_mode) for probe_id in probe_ids for q_mode in Q_MODES}
    if len(identities) != len(expected) or len(set(identities)) != len(identities):
        raise ValueError("probe observation partition is incomplete")
    if set(identities) != expected:
        raise ValueError("probe observation partition is incomplete")
    return tuple(
        sorted(
            validated,
            key=lambda row: (probe_ids.index(row["probe_id"]), Q_MODES.index(row["q_mode"])),
        )
    )


def _successful(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [record for record in records if bool(record["build_success"])]


def _for_mode(records: Sequence[Mapping[str, Any]], q_mode: str) -> list[Mapping[str, Any]]:
    return [record for record in records if record["q_mode"] == q_mode]


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pstdev(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _observed_ratios(
    records: Sequence[Mapping[str, Any]], numerator: str, denominator: str
) -> list[float]:
    return [
        ratio
        for record in _successful(records)
        if (ratio := _ratio(record[numerator], record[denominator])) is not None
    ]


def _rates_by_probe(
    records: Sequence[Mapping[str, Any]], numerator: str, denominator: str
) -> list[float]:
    rates = []
    for probe_id in PRUNING_PROBE_IDS:
        values = _observed_ratios(
            [record for record in records if record["probe_id"] == probe_id],
            numerator,
            denominator,
        )
        if values:
            rates.append(_mean(values))
    return rates


def _coverage(records: Sequence[Mapping[str, Any]]) -> float:
    return float(len(_successful(records)) / len(records))


def aggregate_neutral_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    fp16 = _for_mode(records, "fp16")
    int8 = _for_mode(records, "int8")
    successful = _successful(records)
    successful_int8 = _successful(int8)
    propagation = _observed_ratios(int8, "int8_propagated_ops", "precision_eligible_ops")
    folded = _observed_ratios(int8, "qdq_folded_pairs", "qdq_pairs")
    reformats = _observed_ratios(records, "reformat_ops", "total_ops")
    fusions = _observed_ratios(records, "fused_ops", "fusible_ops")
    return {
        "s1q_probe_pair_coverage": 1.0,
        "s1q_build_success_coverage": _coverage(records),
        "s1q_fp16_build_success_coverage": _coverage(fp16),
        "s1q_int8_build_success_coverage": _coverage(int8),
        "s1q_int8_propagation_ratio_mean": _mean(propagation),
        "s1q_int8_propagation_observed_coverage": float(len(propagation) / len(successful_int8))
        if successful_int8
        else 0.0,
        "s1q_qdq_fold_ratio_mean": _mean(folded),
        "s1q_qdq_fold_observed_coverage": float(len(folded) / len(successful_int8))
        if successful_int8
        else 0.0,
        "s1q_reformat_rate_mean": _mean(reformats),
        "s1q_reformat_observed_coverage": float(len(reformats) / len(successful))
        if successful
        else 0.0,
        "s1q_fusion_coverage_mean": _mean(fusions),
        "s1q_fusion_observed_coverage": float(len(fusions) / len(successful))
        if successful
        else 0.0,
    }


def aggregate_pruning_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    fp16 = _for_mode(records, "fp16")
    int8 = _for_mode(records, "int8")
    return {
        "s1p_probe_pair_coverage": 1.0,
        "s1p_probe_group_count_observed": len(PRUNING_PROBE_IDS),
        "s1p_build_success_coverage": _coverage(records),
        "s1p_fp16_build_success_coverage": _coverage(fp16),
        "s1p_int8_build_success_coverage": _coverage(int8),
        "s1p_fp16_group_build_rate_stddev": _pstdev([1.0 for row in fp16 if row["build_success"]]),
        "s1p_int8_group_build_rate_stddev": _pstdev([1.0 for row in int8 if row["build_success"]]),
        "s1p_fp16_group_reformat_rate_stddev": _pstdev(
            _rates_by_probe(fp16, "reformat_ops", "total_ops")
        ),
        "s1p_int8_group_reformat_rate_stddev": _pstdev(
            _rates_by_probe(int8, "reformat_ops", "total_ops")
        ),
        "s1p_int8_group_propagation_rate_stddev": _pstdev(
            _rates_by_probe(int8, "int8_propagated_ops", "precision_eligible_ops")
        ),
        "s1p_int8_group_qdq_fold_rate_stddev": _pstdev(
            _rates_by_probe(int8, "qdq_folded_pairs", "qdq_pairs")
        ),
    }


def aggregate_capability_features(
    neutral_records: object, pruning_records: object
) -> tuple[dict[str, float | int], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Validate the exact partition and derive all canonical numeric features."""
    neutral = validate_probe_partition(neutral_records, probe_ids=NEUTRAL_PROBE_IDS)
    pruning = validate_probe_partition(pruning_records, probe_ids=PRUNING_PROBE_IDS)
    features = {
        **aggregate_neutral_features(neutral),
        **aggregate_pruning_features(pruning),
    }
    if tuple(features) != FEATURE_NAMES or any(
        isinstance(value, bool) or not math.isfinite(float(value)) for value in features.values()
    ):
        raise ValueError("capability feature aggregation is invalid")
    return features, neutral, pruning


__all__ = [
    "FEATURE_NAMES",
    "NEUTRAL_PROBE_IDS",
    "PRUNING_PROBE_IDS",
    "Q_MODES",
    "RECORD_FIELDS",
    "RECORD_SCHEMA_VERSION",
    "aggregate_capability_features",
    "validate_probe_partition",
]
