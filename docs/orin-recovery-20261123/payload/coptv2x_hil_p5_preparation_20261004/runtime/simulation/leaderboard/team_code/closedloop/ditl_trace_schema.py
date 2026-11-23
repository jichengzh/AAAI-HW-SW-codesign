from __future__ import print_function

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SCHEMA_VERSION = "ditl.trace.v1"
SCHEMA_VERSION_V2 = "ditl.trace.v2"
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
V2_SENDER_TIMESTAMP_ORDER = (
    "rsu_data_available",
    "source_read_start",
    "source_read_end",
    "preprocess_start",
    "preprocess_end",
    "h2d_start",
    "h2d_end",
    "inference_start",
    "inference_end",
    "d2h_start",
    "d2h_end",
    "postprocess_start",
    "postprocess_end",
    "pack_start",
    "pack_end",
    "hash_start",
    "hash_end",
    "compress_start",
    "compress_end",
    "protocol_serialize_start",
    "protocol_serialize_end",
    "ros_message_start",
    "ros_message_end",
    "ros_publish_start",
    "ros_publish_end",
)
V2_RECEIVER_TIMESTAMP_ORDER = (
    "ros_receive",
    "frame_parse_start",
    "frame_parse_end",
    "decompress_start",
    "decompress_end",
    "verify_start",
    "verify_end",
    "reconstruct_start",
    "reconstruct_end",
    "fusion_ready",
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


def _validate_trace_record_v1(record, max_clock_error_ns=1_000_000):
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


def _validate_clock_snapshot_v2(snapshot, path):
    snapshot = _require_fields(
        snapshot,
        path,
        ("host_id", "domain", "sync_state", "offset_ns", "uncertainty_ns"),
    )
    _require_string(snapshot["host_id"], "%s.host_id" % path)
    domain = _require_string(snapshot["domain"], "%s.domain" % path)
    if domain not in CLOCK_DOMAINS:
        _error("%s.domain" % path, "unsupported clock domain %s" % domain)
    state = _require_string(snapshot["sync_state"], "%s.sync_state" % path)
    if state not in SYNC_STATES:
        _error("%s.sync_state" % path, "unsupported sync state %s" % state)
    _require_int_or_signed(snapshot["offset_ns"], "%s.offset_ns" % path)
    _require_int(snapshot["uncertainty_ns"], "%s.uncertainty_ns" % path)
    if domain == "ptp":
        _require_string(snapshot.get("grandmaster_id"), "%s.grandmaster_id" % path)
    else:
        _require_string(snapshot.get("boot_id"), "%s.boot_id" % path)


def _v2_clocks_comparable(clock, max_clock_error_ns):
    sender = clock["sender"]
    receiver = clock["receiver"]
    same_host = sender["host_id"] == receiver["host_id"]
    if same_host:
        if sender["domain"] != receiver["domain"]:
            return False, "same_host_domain_mismatch"
        if sender["domain"] == "monotonic_raw":
            if sender.get("boot_id") != receiver.get("boot_id"):
                return False, "same_host_boot_mismatch"
            return True, "same_host_monotonic"
    if sender["domain"] != "ptp" or receiver["domain"] != "ptp":
        return False, "cross_host_requires_ptp"
    if sender["sync_state"] != "locked" or receiver["sync_state"] != "locked":
        return False, "ptp_not_locked"
    if sender.get("grandmaster_id") != receiver.get("grandmaster_id"):
        return False, "ptp_grandmaster_mismatch"
    for snapshot in (sender, receiver):
        if (
            abs(snapshot["offset_ns"]) > max_clock_error_ns
            or snapshot["uncertainty_ns"] > max_clock_error_ns
        ):
            return False, "ptp_error_exceeds_limit"
    return True, "ptp_locked"


def _validate_ordered_timestamps(timestamps, names):
    previous_name = None
    previous_value = None
    for name in names:
        value = timestamps[name]
        _require_int(value, "timestamps_ns.%s" % name)
        if previous_value is not None and previous_value > value:
            _error(
                "timestamps_ns.%s" % previous_name,
                "must be <= timestamps_ns.%s" % name,
            )
        previous_name = name
        previous_value = value


def validate_trace_record_v2(record, max_clock_error_ns=1_000_000):
    required = (
        "schema_version",
        "trace_id",
        "run_id",
        "route_id",
        "scenario_id",
        "measurement_scope",
        "headline_eligible",
        "headline_ineligible_reason",
        "source",
        "message",
        "clock",
        "clock_comparability",
        "status",
        "timestamps_ns",
        "network",
        "codec",
        "durations_ns",
        "testbed_overhead_ns",
        "model",
        "device",
    )
    normalized = copy.deepcopy(_require_fields(record, "record", required))
    if normalized["schema_version"] != SCHEMA_VERSION_V2:
        _error("schema_version", "must equal %s" % SCHEMA_VERSION_V2)
    for field in ("trace_id", "run_id", "route_id", "scenario_id"):
        _require_string(normalized[field], field)
    scope = _require_string(normalized["measurement_scope"], "measurement_scope")
    if scope not in ("rsu_to_fusion", "testbed_probe"):
        _error("measurement_scope", "must be rsu_to_fusion or testbed_probe")
    if not isinstance(normalized["headline_eligible"], bool):
        _error("headline_eligible", "must be boolean")
    reason = normalized["headline_ineligible_reason"]
    if normalized["headline_eligible"]:
        if reason is not None:
            _error("headline_ineligible_reason", "must be null when eligible")
    else:
        _require_string(reason, "headline_ineligible_reason")

    source = _require_fields(
        normalized["source"],
        "source",
        ("type", "payload_semantics", "postprocess"),
    )
    for field in ("type", "payload_semantics", "postprocess"):
        _require_string(source[field], "source.%s" % field)
    _validate_message(normalized["message"])
    _validate_status(normalized["status"])
    if normalized["status"]["outcome"] != "consumed":
        _error("status.outcome", "v2 formal result must be consumed")
    _validate_network(normalized["network"])
    _validate_durations(normalized["durations_ns"])
    overhead = _require_mapping(normalized["testbed_overhead_ns"], "testbed_overhead_ns")
    for name, value in overhead.items():
        _require_int(value, "testbed_overhead_ns.%s" % name)

    codec = _require_fields(
        normalized["codec"],
        "codec",
        ("name", "raw_payload_bytes", "wire_payload_bytes"),
    )
    if codec["name"] not in ("none", "zlib", "lz4"):
        _error("codec.name", "unsupported codec")
    _require_int(codec["raw_payload_bytes"], "codec.raw_payload_bytes")
    _require_int(codec["wire_payload_bytes"], "codec.wire_payload_bytes")

    clock = _require_fields(normalized["clock"], "clock", ("sender", "receiver"))
    _validate_clock_snapshot_v2(clock["sender"], "clock.sender")
    _validate_clock_snapshot_v2(clock["receiver"], "clock.receiver")
    actual_comparable, actual_reason = _v2_clocks_comparable(
        clock, max_clock_error_ns
    )
    declared = _require_fields(
        normalized["clock_comparability"],
        "clock_comparability",
        ("comparable", "reason"),
    )
    if not isinstance(declared["comparable"], bool):
        _error("clock_comparability.comparable", "must be boolean")
    _require_string(declared["reason"], "clock_comparability.reason")
    if declared["comparable"] != actual_comparable:
        _error(
            "clock_comparability.comparable",
            "does not match clock snapshots (%s)" % actual_reason,
        )
    if normalized["headline_eligible"] and not actual_comparable:
        _error("headline_eligible", "requires comparable sender and receiver clocks")
    if normalized["headline_eligible"] and scope != "rsu_to_fusion":
        _error("headline_eligible", "testbed probes cannot be headline eligible")

    timestamps = _require_fields(
        normalized["timestamps_ns"],
        "timestamps_ns",
        V2_SENDER_TIMESTAMP_ORDER + V2_RECEIVER_TIMESTAMP_ORDER,
    )
    _validate_ordered_timestamps(timestamps, V2_SENDER_TIMESTAMP_ORDER)
    _validate_ordered_timestamps(timestamps, V2_RECEIVER_TIMESTAMP_ORDER)
    if actual_comparable and timestamps["ros_publish_start"] > timestamps["ros_receive"]:
        _error("timestamps_ns.ros_publish_start", "must be <= timestamps_ns.ros_receive")

    derived = {
        "rsu_device_pipeline": (
            timestamps["ros_publish_start"] - timestamps["rsu_data_available"]
        ),
        "receiver_decode": timestamps["fusion_ready"] - timestamps["ros_receive"],
    }
    if actual_comparable:
        derived["ros_and_network"] = (
            timestamps["ros_receive"] - timestamps["ros_publish_start"]
        )
        derived["rsu_to_fusion"] = (
            timestamps["fusion_ready"] - timestamps["rsu_data_available"]
        )
    normalized["derived_ns"] = derived
    normalized["clock_comparability"]["reason"] = actual_reason
    return normalized


def validate_trace_record(record, max_clock_error_ns=1_000_000):
    """Validate one DitL trace record and return an independent normalized copy."""
    _require_int(max_clock_error_ns, "max_clock_error_ns")
    mapping = _require_mapping(record, "record")
    version = mapping.get("schema_version")
    if version == SCHEMA_VERSION:
        return _validate_trace_record_v1(mapping, max_clock_error_ns)
    if version == SCHEMA_VERSION_V2:
        return validate_trace_record_v2(mapping, max_clock_error_ns)
    _error("schema_version", "unsupported schema version %r" % version)


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
        and (
            record["schema_version"] == SCHEMA_VERSION
            or (
                record["headline_eligible"]
                and "rsu_to_fusion" in record["derived_ns"]
            )
        )
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DITL_LATENCY_CSV_COLUMNS)
        writer.writeheader()
        for call_idx, record in enumerate(consumed):
            derived = record["derived_ns"]
            message = record["message"]
            is_v2 = record["schema_version"] == SCHEMA_VERSION_V2
            total_name = "rsu_to_fusion" if is_v2 else "age_at_fusion"
            transport_name = "ros_and_network" if is_v2 else "transport"
            writer.writerow(
                {
                    "call_idx": call_idx,
                    "tau_ms": "%.6f" % (derived[total_name] / 1_000_000.0),
                    "latency_ms_source": (
                        "ditl_rsu_to_fusion_v2" if is_v2 else "ditl_orin_transport"
                    ),
                    "trace_id": record["trace_id"],
                    "run_id": record["run_id"],
                    "source_id": message["source_id"],
                    "target_id": message["target_id"],
                    "sequence_id": message["sequence_id"],
                    "frame_id": message["frame_id"],
                    "payload_bytes": message["payload_bytes"],
                    "transport_ms": "%.6f"
                    % (derived[transport_name] / 1_000_000.0),
                    "age_at_fusion_ms": "%.6f"
                    % (derived[total_name] / 1_000_000.0),
                }
            )
    return len(consumed)
