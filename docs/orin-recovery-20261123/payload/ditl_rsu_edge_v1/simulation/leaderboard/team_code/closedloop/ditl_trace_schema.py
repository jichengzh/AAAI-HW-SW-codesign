from __future__ import print_function

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SCHEMA_VERSION = "ditl.trace.v1"
OUTCOMES = frozenset(("consumed", "delivered", "dropped", "expired"))
CLOCK_DOMAINS = frozenset(("ptp", "monotonic_raw"))
SYNC_STATES = frozenset(("locked", "unlocked", "unknown"))
STAGE_PAIRS = (
    ("inference_start", "inference_end"),
    ("compress_start", "compress_end"),
    ("d2h_start", "d2h_end"),
    ("serialize_start", "serialize_end"),
    ("deserialize_start", "deserialize_end"),
    ("h2d_start", "h2d_end"),
    ("decompress_start", "decompress_end"),
)
TIMESTAMP_ORDER = (
    "source",
    "inference_start",
    "inference_end",
    "compress_start",
    "compress_end",
    "d2h_start",
    "d2h_end",
    "serialize_start",
    "serialize_end",
    "publish",
    "receive",
    "deserialize_start",
    "deserialize_end",
    "h2d_start",
    "h2d_end",
    "decompress_start",
    "decompress_end",
    "fusion_consume",
)
DITL_LATENCY_CSV_COLUMNS = (
    "call_idx",
    "tau_ms",
    "latency_ms_source",
    "trace_id",
    "run_id",
    "source_id",
    "target_id",
    "sequence_id",
    "frame_id",
    "payload_bytes",
    "transport_ms",
    "age_at_fusion_ms",
)


class TraceValidationError(ValueError):
    pass


def _error(path, message):
    raise TraceValidationError("%s: %s" % (path, message))


def _require_mapping(value, path):
    if not isinstance(value, dict):
        _error(path, "must be an object")
    return value


def _require_fields(value, path, fields):
    mapping = _require_mapping(value, path)
    for field in fields:
        if field not in mapping:
            _error("%s.%s" % (path, field), "is required")
    return mapping


def _require_string(value, path):
    if not isinstance(value, str) or not value.strip():
        _error(path, "must be a non-empty string")
    return value


def _require_int(value, path, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(path, "must be an integer >= %d" % minimum)
    return value


def _require_finite_number(value, path, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        _error(path, "must be a finite number")
    if minimum is not None and numeric < minimum:
        _error(path, "must be >= %s" % minimum)
    return numeric


def _validate_message(message):
    required = (
        "source_id",
        "target_id",
        "frame_id",
        "sequence_id",
        "topic",
        "message_type",
        "payload_bytes",
        "tensor_shape",
        "tensor_dtype",
        "compression_ratio",
    )
    message = _require_fields(message, "message", required)
    for field in ("source_id", "target_id", "topic", "message_type", "tensor_dtype"):
        _require_string(message[field], "message.%s" % field)
    _require_int(message["frame_id"], "message.frame_id")
    _require_int(message["sequence_id"], "message.sequence_id")
    _require_int(message["payload_bytes"], "message.payload_bytes")
    shape = message["tensor_shape"]
    if not isinstance(shape, list) or not shape:
        _error("message.tensor_shape", "must be a non-empty list")
    for index, dimension in enumerate(shape):
        _require_int(dimension, "message.tensor_shape[%d]" % index, minimum=1)
    _require_finite_number(
        message["compression_ratio"], "message.compression_ratio", minimum=0.0
    )


def _validate_clock_snapshot(snapshot, path, max_clock_error_ns):
    required = (
        "host_id",
        "domain",
        "sync_state",
        "offset_ns",
        "uncertainty_ns",
    )
    snapshot = _require_fields(snapshot, path, required)
    _require_string(snapshot["host_id"], "%s.host_id" % path)
    domain = _require_string(snapshot["domain"], "%s.domain" % path)
    if domain not in CLOCK_DOMAINS:
        _error("%s.domain" % path, "unsupported clock domain %s" % domain)
    sync_state = _require_string(snapshot["sync_state"], "%s.sync_state" % path)
    if sync_state not in SYNC_STATES:
        _error("%s.sync_state" % path, "unsupported sync state %s" % sync_state)
    offset_ns = _require_int_or_signed(snapshot["offset_ns"], "%s.offset_ns" % path)
    uncertainty_ns = _require_int(
        snapshot["uncertainty_ns"], "%s.uncertainty_ns" % path
    )
    if domain == "ptp":
        _require_string(snapshot.get("grandmaster_id"), "%s.grandmaster_id" % path)
        if sync_state != "locked":
            _error("%s.sync_state" % path, "PTP clock must be locked")
        if abs(offset_ns) > max_clock_error_ns or uncertainty_ns > max_clock_error_ns:
            _error(path, "clock error exceeds %d ns" % max_clock_error_ns)
    elif not snapshot.get("boot_id"):
        _error("%s.boot_id" % path, "is required for monotonic_raw")


def _require_int_or_signed(value, path):
    if isinstance(value, bool) or not isinstance(value, int):
        _error(path, "must be an integer")
    return value


def _validate_clock(clock, max_clock_error_ns):
    clock = _require_fields(clock, "clock", ("sender", "receiver"))
    sender = clock["sender"]
    receiver = clock["receiver"]
    _validate_clock_snapshot(sender, "clock.sender", max_clock_error_ns)
    _validate_clock_snapshot(receiver, "clock.receiver", max_clock_error_ns)
    cross_host = sender["host_id"] != receiver["host_id"]
    if cross_host:
        if sender["domain"] != "ptp" or receiver["domain"] != "ptp":
            _error("clock", "cross-host timestamps require PTP")
        if sender["grandmaster_id"] != receiver["grandmaster_id"]:
            _error("clock", "sender and receiver grandmaster must match")
    else:
        if sender["domain"] != receiver["domain"]:
            _error("clock", "same-host clock domains must match")
        if sender["domain"] == "monotonic_raw" and (
            sender.get("boot_id") != receiver.get("boot_id")
        ):
            _error("clock", "same-host monotonic_raw boot_id must match")


def _validate_status(status):
    status = _require_fields(
        status, "status", ("outcome", "duplicate", "out_of_order", "reason")
    )
    outcome = _require_string(status["outcome"], "status.outcome")
    if outcome not in OUTCOMES:
        _error("status.outcome", "unsupported outcome %s" % outcome)
    for field in ("duplicate", "out_of_order"):
        if not isinstance(status[field], bool):
            _error("status.%s" % field, "must be boolean")
    if status["reason"] is not None and not isinstance(status["reason"], str):
        _error("status.reason", "must be null or string")


def _validate_timestamps(timestamps, outcome):
    timestamps = _require_fields(
        timestamps, "timestamps_ns", ("source", "publish", "receive", "fusion_consume")
    )
    for name, value in timestamps.items():
        if value is not None:
            _require_int(value, "timestamps_ns.%s" % name)
    if timestamps["source"] is None:
        _error("timestamps_ns.source", "must not be null")
    if timestamps["publish"] is None:
        _error("timestamps_ns.publish", "must not be null")

    for start, end in STAGE_PAIRS:
        start_value = timestamps.get(start)
        end_value = timestamps.get(end)
        if (start_value is None) != (end_value is None):
            _error(
                "timestamps_ns.%s" % start,
                "and timestamps_ns.%s must both be present or null" % end,
            )
        if start_value is not None and start_value > end_value:
            _error(
                "timestamps_ns.%s" % start,
                "must be <= timestamps_ns.%s" % end,
            )

    previous_name = None
    previous_value = None
    for name in TIMESTAMP_ORDER:
        value = timestamps.get(name)
        if value is None:
            continue
        if previous_value is not None and previous_value > value:
            _error(
                "timestamps_ns.%s" % previous_name,
                "must be <= timestamps_ns.%s" % name,
            )
        previous_name = name
        previous_value = value

    receive = timestamps.get("receive")
    consume = timestamps.get("fusion_consume")
    if outcome == "dropped" and (receive is not None or consume is not None):
        _error("status.outcome", "dropped record must not have receive or fusion_consume")
    if outcome == "consumed" and (receive is None or consume is None):
        _error("status.outcome", "consumed record requires receive and fusion_consume")
    if outcome in ("delivered", "expired") and receive is None:
        _error("status.outcome", "%s record requires receive" % outcome)
    if outcome in ("delivered", "expired") and consume is not None:
        _error("status.outcome", "%s record must not have fusion_consume" % outcome)


def _validate_durations(durations):
    durations = _require_mapping(durations, "durations_ns")
    for name, value in durations.items():
        _require_int(value, "durations_ns.%s" % name)


def _validate_network(network):
    network = _require_fields(
        network, "network", ("transport", "qos", "sender_count", "receiver_count")
    )
    _require_string(network["transport"], "network.transport")
    _require_string(network["qos"], "network.qos")
    _require_int(network["sender_count"], "network.sender_count")
    _require_int(network["receiver_count"], "network.receiver_count")


def validate_trace_record(record, max_clock_error_ns=1_000_000):
    """Validate one DitL trace record and return an independent normalized copy."""
    _require_int(max_clock_error_ns, "max_clock_error_ns")
    required = (
        "schema_version",
        "trace_id",
        "run_id",
        "route_id",
        "scenario_id",
        "message",
        "clock",
        "status",
        "timestamps_ns",
        "network",
        "durations_ns",
    )
    normalized = copy.deepcopy(_require_fields(record, "record", required))
    if normalized["schema_version"] != SCHEMA_VERSION:
        _error("schema_version", "must equal %s" % SCHEMA_VERSION)
    for field in ("trace_id", "run_id", "route_id", "scenario_id"):
        _require_string(normalized[field], field)
    _validate_message(normalized["message"])
    _validate_clock(normalized["clock"], max_clock_error_ns)
    _validate_status(normalized["status"])
    _validate_timestamps(
        normalized["timestamps_ns"], normalized["status"]["outcome"]
    )
    _validate_network(normalized["network"])
    _validate_durations(normalized["durations_ns"])

    timestamps = normalized["timestamps_ns"]
    derived = {}
    if timestamps.get("receive") is not None:
        derived["transport"] = timestamps["receive"] - timestamps["publish"]
    if timestamps.get("fusion_consume") is not None:
        derived["age_at_fusion"] = (
            timestamps["fusion_consume"] - timestamps["source"]
        )
    normalized["derived_ns"] = derived
    return normalized


def validate_trace_sequence(records, max_clock_error_ns=1_000_000):
    """Validate records and enforce explicit duplicate/out-of-order marking."""
    normalized_records = [
        validate_trace_record(record, max_clock_error_ns=max_clock_error_ns)
        for record in records
    ]
    seen = defaultdict(set)
    maxima = {}
    for index, record in enumerate(normalized_records):
        message = record["message"]
        status = record["status"]
        stream = (
            record["run_id"],
            message["source_id"],
            message["target_id"],
            message["topic"],
        )
        sequence_id = message["sequence_id"]
        duplicate = sequence_id in seen[stream]
        out_of_order = stream in maxima and sequence_id < maxima[stream]
        if duplicate and not status["duplicate"]:
            _error("records[%d].status.duplicate" % index, "duplicate sequence_id")
        if status["duplicate"] and not duplicate:
            _error("records[%d].status.duplicate" % index, "sequence_id was not seen")
        if out_of_order and not status["out_of_order"]:
            _error("records[%d].status.out_of_order" % index, "out_of_order sequence_id")
        if status["out_of_order"] and not out_of_order:
            _error("records[%d].status.out_of_order" % index, "sequence_id is ordered")
        seen[stream].add(sequence_id)
        maxima[stream] = max(maxima.get(stream, sequence_id), sequence_id)
    return normalized_records


def write_trace_jsonl(path, records, max_clock_error_ns=1_000_000):
    normalized = validate_trace_sequence(
        records, max_clock_error_ns=max_clock_error_ns
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in normalized:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return len(normalized)


def read_trace_jsonl(path, max_clock_error_ns=1_000_000):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(
                    validate_trace_record(
                        value, max_clock_error_ns=max_clock_error_ns
                    )
                )
            except (json.JSONDecodeError, TraceValidationError) as error:
                raise TraceValidationError("line %d: %s" % (line_number, error))
    try:
        return validate_trace_sequence(
            records, max_clock_error_ns=max_clock_error_ns
        )
    except TraceValidationError as error:
        raise TraceValidationError("sequence: %s" % error)


def write_latency_csv(path, records, max_clock_error_ns=1_000_000):
    normalized = validate_trace_sequence(
        records, max_clock_error_ns=max_clock_error_ns
    )
    consumed = [
        record
        for record in normalized
        if record["status"]["outcome"] == "consumed"
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DITL_LATENCY_CSV_COLUMNS)
        writer.writeheader()
        for call_idx, record in enumerate(consumed):
            derived = record["derived_ns"]
            message = record["message"]
            writer.writerow(
                {
                    "call_idx": call_idx,
                    "tau_ms": "%.6f" % (derived["age_at_fusion"] / 1_000_000.0),
                    "latency_ms_source": "ditl_orin_transport",
                    "trace_id": record["trace_id"],
                    "run_id": record["run_id"],
                    "source_id": message["source_id"],
                    "target_id": message["target_id"],
                    "sequence_id": message["sequence_id"],
                    "frame_id": message["frame_id"],
                    "payload_bytes": message["payload_bytes"],
                    "transport_ms": "%.6f"
                    % (derived["transport"] / 1_000_000.0),
                    "age_at_fusion_ms": "%.6f"
                    % (derived["age_at_fusion"] / 1_000_000.0),
                }
            )
    return len(consumed)
