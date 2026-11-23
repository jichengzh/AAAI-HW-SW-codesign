from __future__ import print_function

from collections import defaultdict

import numpy as np

from ditl_trace_schema import SCHEMA_VERSION_V2, validate_trace_sequence


EDGE_NONOVERLAP_STAGES = (
    "source_read",
    "preprocess",
    "h2d",
    "gpu_inference",
    "d2h",
    "postprocess",
    "pack",
    "hash",
    "compress",
    "protocol_serialize",
    "local_handoff",
    "ros_message",
    "ros_publish_call",
)


def _stats_ns(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array) / 1_000_000.0),
        "min_ms": float(np.min(array) / 1_000_000.0),
        "max_ms": float(np.max(array) / 1_000_000.0),
        "p50_ms": float(np.percentile(array, 50) / 1_000_000.0),
        "p90_ms": float(np.percentile(array, 90) / 1_000_000.0),
        "p95_ms": float(np.percentile(array, 95) / 1_000_000.0),
        "p99_ms": float(np.percentile(array, 99) / 1_000_000.0),
    }


def _stats_bytes(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "min": int(np.min(array)),
        "max": int(np.max(array)),
        "p50": int(round(float(np.percentile(array, 50)))),
        "p95": int(round(float(np.percentile(array, 95)))),
    }


def summarize_stage_records(records, warmup_count=0):
    if isinstance(warmup_count, bool) or not isinstance(warmup_count, int):
        raise ValueError("warmup_count must be an integer")
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative")
    normalized = validate_trace_sequence(records)
    v2_records = [record for record in normalized if record["schema_version"] == SCHEMA_VERSION_V2]
    if len(v2_records) != len(normalized):
        raise ValueError("stage summaries require ditl.trace.v2 records")
    if warmup_count >= len(v2_records):
        raise ValueError("warmup_count must leave at least one measured record")
    measured = v2_records[warmup_count:]

    stage_values = defaultdict(list)
    total_values = defaultdict(list)
    residuals = []
    raw_sizes = []
    wire_sizes = []
    for record in measured:
        for name, value in record["durations_ns"].items():
            stage_values[name].append(value)
        for name, value in record["derived_ns"].items():
            total_values[name].append(value)
        device_total = record["derived_ns"]["rsu_device_pipeline"]
        edge_sum = sum(
            record["durations_ns"].get(name, 0)
            for name in EDGE_NONOVERLAP_STAGES
        )
        residuals.append(device_total - edge_sum)
        raw_sizes.append(record["codec"]["raw_payload_bytes"])
        wire_sizes.append(record["codec"]["wire_payload_bytes"])

    stages = {
        name: _stats_ns(values) for name, values in sorted(stage_values.items())
    }
    totals = {
        name: _stats_ns(values) for name, values in sorted(total_values.items())
    }
    totals["stage_sum_residual_ms"] = _stats_ns(residuals)
    candidate_stages = {
        name: values["p50_ms"]
        for name, values in stages.items()
        if name in EDGE_NONOVERLAP_STAGES
    }
    bottleneck = max(candidate_stages, key=candidate_stages.get) if candidate_stages else None
    return {
        "schema_version": "ditl.stage-summary.v1",
        "samples": {
            "total": len(v2_records),
            "warmup": warmup_count,
            "measured": len(measured),
            "headline_eligible": sum(
                1 for record in measured if record["headline_eligible"]
            ),
        },
        "stages": stages,
        "totals": totals,
        "payload": {
            "raw_bytes": _stats_bytes(raw_sizes),
            "wire_bytes": _stats_bytes(wire_sizes),
            "codec": sorted(set(record["codec"]["name"] for record in measured)),
        },
        "largest_edge_compute_stage": bottleneck,
    }
