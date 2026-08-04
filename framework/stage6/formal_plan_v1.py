"""Build a backend-symmetric, immutable Stage6 execution plan."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


AP_GRAPH_FEATURES = (
    "parameter_elements",
    "conv_flops",
    "conv_count",
    "group_conv_count",
    "stride2_conv_count",
    "arithmetic_intensity_proxy",
)
ALLOWED_Q_MODES = frozenset({"fp16", "int8", "fp32"})


def _invalid_genome() -> ValueError:
    return ValueError("invalid candidate genome")


def _invalid_scalar() -> ValueError:
    return ValueError("formal candidate contains invalid scalar")


def _validated_widths(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in value
        )
    ):
        raise _invalid_genome()
    return value[0], value[1], value[2]


def _validated_q_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_Q_MODES:
        raise _invalid_genome()
    return value


def _genome_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    genome = row.get("genome")
    if not isinstance(genome, (list, tuple)) or len(genome) != 4:
        raise _invalid_genome()
    widths = _validated_widths(genome[:3])
    return *widths, _validated_q_mode(genome[3])


def _finite(value: Any) -> float:
    number: float | None = None
    if not isinstance(value, bool):
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            number = None
    if number is None or not math.isfinite(number):
        raise _invalid_scalar()
    return number


def _normalize(values: Sequence[float], value: float) -> float:
    low, high = min(values), max(values)
    return 0.0 if high == low else (value - low) / (high - low)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    indexed: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for source in rows:
        row = deepcopy(dict(source))
        key = _genome_key(row)
        if key in indexed:
            raise ValueError(f"duplicate candidate genome: {key}")
        indexed[key] = row
    return indexed


def _ap_feature(
    *, model: Any, width: Any, q_mode: Any, graph: Any
) -> list[float]:
    if not isinstance(model, str):
        raise ValueError("invalid formal model")
    widths = _validated_widths(width)
    mode = _validated_q_mode(q_mode)
    if not isinstance(graph, Mapping):
        raise ValueError("invalid formal graph features")
    return [
        *[_finite(width) for width in widths],
        1.0 if mode == "int8" else 0.0,
        1.0 if model == "codriving" else 0.0,
        *[_finite(graph.get(name, 0.0)) for name in AP_GRAPH_FEATURES],
    ]


def fit_neutral_ap_surrogate(
    training_rows: Sequence[Mapping[str, Any]],
    graph_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260720,
) -> dict[tuple[int, int, int, str], float]:
    """Fit AP without backend, capability, latency, or energy inputs."""
    graph_by_group: dict[str, dict[str, Any]] = {}
    for row in graph_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid formal graph features")
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("invalid formal graph features")
        graph_by_group[group_id] = dict(row)
    x_train, y_train = [], []
    for row in training_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid formal training row")
        if row.get("terminal_status") != "measured_success_gold":
            continue
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("invalid formal training row")
        graph = graph_by_group.get(group_id)
        if graph is None:
            continue
        ap70 = _finite(row.get("ap70"))
        x_train.append(
            _ap_feature(
                model=row.get("model"),
                width=row.get("width"),
                q_mode=row.get("q_mode"),
                graph=graph,
            )
        )
        y_train.append(ap70)
    if len(x_train) < 16:
        raise ValueError("insufficient backend-neutral AP training rows")
    model = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float))
    ordered = []
    for row in candidates:
        if not isinstance(row, Mapping):
            raise _invalid_genome()
        ordered.append((_genome_key(row), row))
    ordered.sort(key=lambda item: item[0])
    x_predict = [
        _ap_feature(
            model=row.get("model", "pyramid"),
            width=row.get("width"),
            q_mode=row.get("q_mode"),
            graph=row.get("graph_features", {}),
        )
        for _, row in ordered
    ]
    predictions = model.predict(np.asarray(x_predict, dtype=float))
    return {key: _finite(value) for (key, _), value in zip(ordered, predictions)}


def build_formal_plan(
    tvm_rows: Sequence[Mapping[str, Any]],
    trt_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pool_size: int = 100,
    neutral_ap_by_genome: Mapping[tuple[int, int, int, str], float] | None = None,
) -> dict[str, Any]:
    """Freeze identical genome choices for the two independent backend tasks."""
    tvm, trt = _index(tvm_rows), _index(trt_rows)
    if set(tvm) != set(trt):
        raise ValueError("TVM/TRT candidate genome sets differ")
    if len(tvm) != expected_pool_size:
        raise ValueError(
            f"effective candidate pool mismatch: expected {expected_pool_size}, got {len(tvm)}"
        )

    neutral = []
    evidence_records = []
    for key in sorted(tvm):
        left, right = tvm[key], trt[key]
        left_evidence = left.get("source_evidence_sha256")
        right_evidence = right.get("source_evidence_sha256")
        if not _is_sha256(left_evidence) or not _is_sha256(right_evidence):
            raise ValueError("invalid source evidence binding")
        if left_evidence != right_evidence:
            raise ValueError(f"backend source evidence drift for genome {key}")
        if neutral_ap_by_genome is None:
            left_ap = _finite((left.get("predictions") or {}).get("ap70"))
            right_ap = _finite((right.get("predictions") or {}).get("ap70"))
            if not math.isclose(left_ap, right_ap, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"backend-dependent AP surrogate drift for genome {key}")
        else:
            if key not in neutral_ap_by_genome:
                raise ValueError(f"neutral AP surrogate missing genome {key}")
            left_ap = _finite(neutral_ap_by_genome[key])
        left_graph = left.get("graph_features") or {}
        right_graph = right.get("graph_features") or {}
        parameter_count = _finite(left_graph.get("parameter_elements"))
        flops = _finite(left_graph.get("conv_flops"))
        if not math.isclose(
            parameter_count,
            _finite(right_graph.get("parameter_elements")),
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            flops,
            _finite(right_graph.get("conv_flops")),
            rel_tol=0.0,
            abs_tol=1e-3,
        ):
            raise ValueError(f"backend candidate graph feature drift for genome {key}")
        neutral.append(
            {
                "genome": [*key[:3], key[3]],
                "candidate_id": f"pyramid|{'x'.join(map(str, key[:3]))}|q={key[3]}",
                "ap_surrogate": left_ap,
                "parameter_count": parameter_count,
                "flops": flops,
                "source_evidence_sha256": left_evidence,
            }
        )
        evidence_records.append(
            {
                "genome": [*key[:3], key[3]],
                "source_evidence_sha256": left_evidence,
            }
        )

    columns = {
        name: [row[name] for row in neutral]
        for name in ("ap_surrogate", "parameter_count", "flops")
    }
    ranked = []
    for row in neutral:
        score = (
            _normalize(columns["ap_surrogate"], row["ap_surrogate"])
            - 0.5 * _normalize(columns["parameter_count"], row["parameter_count"])
            - 0.5 * _normalize(columns["flops"], row["flops"])
        )
        ranked.append({**row, "hardware_blind_acquisition_score": score})
    ranked.sort(key=lambda row: (-row["hardware_blind_acquisition_score"], row["candidate_id"]))

    selected = [row["genome"] for row in ranked[:16]]
    screened = [row["genome"] for row in ranked[:12]]
    locked = [row["genome"] for row in ranked[:4]]
    backend_plans = {
        backend: {
            "compression_only": deepcopy(selected),
            "compress_then_tune": deepcopy(locked),
            "tune_then_compress": deepcopy(selected),
        }
        for backend in ("tvm", "trt")
    }
    source_evidence_binding = {
        "schema_version": "stage6_source_evidence_binding_v1",
        "records": evidence_records,
        "evidence_version_sha256": _digest(evidence_records),
    }
    payload = {
        "schema_version": "stage6_formal_execution_plan_v1",
        "passed": True,
        "effective_candidate_pool_size": len(neutral),
        "pool_policy": "registered_materializable_minus_frozen_gold176_rows",
        "hardware_blind_backend_labels_used": False,
        "ranked_candidates": ranked,
        "source_evidence_binding": source_evidence_binding,
        "arms": {
            "original_default": {"fixed_genome": [64, 128, 256, "fp32"]},
            "compression_only": {"selected_genomes": selected, "outer_budget": 16},
            "schedule_only": {"fixed_genome": [64, 128, 256, "fp32"]},
            "compress_then_tune": {
                "screened_genomes": screened,
                "locked_genomes": locked,
                "outer_budget": {"screen": 12, "locked": 4},
            },
            "tune_then_compress": {
                "attempt_genomes": selected,
                "outer_budget": {"base_tune": 1, "compressed_attempts": 16},
            },
            "joint_shcosearch": {"reuse_actual_feedback_v3_budget": 16},
        },
        "backend_plans": backend_plans,
    }
    return {**payload, "plan_sha256": _digest(payload)}
