"""Strict replay samples derived from eligible Raw DitL trace records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from .ditl_trace_schema import TraceValidationError, validate_trace_sequence
except ImportError:  # Support direct imports used by the closedloop scripts.
    from ditl_trace_schema import TraceValidationError, validate_trace_sequence


REPLAY_SAMPLE_SCHEMA_VERSION = "ditl.replay-sample.v1"
NORMALIZER_NAME = "raw-rsu-to-fusion"
NORMALIZER_VERSION = "1.0.0"
STAGE_NAMES = (
    "acquisition_preprocess",
    "gpu_inference",
    "packing_postprocess",
    "transport_receive",
)

_REPLAY_FIELDS = frozenset(
    (
        "schema_version",
        "sample_id",
        "ordinal",
        "source_trace_id",
        "source_sequence_id",
        "base_delay_ns",
        "stage_ns",
        "raw_payload_bytes",
        "wire_payload_bytes",
        "capture_condition_id",
    )
)
_IDENTITY_PATHS = (
    ("run_id", ("run_id",)),
    ("model_id", ("model", "model_id")),
    ("engine_sha256", ("model", "engine_sha256")),
    ("message.source_id", ("message", "source_id")),
    ("source.type", ("source", "type")),
    ("payload_semantics", ("source", "payload_semantics")),
)
_ACQUISITION_DURATIONS = ("source_read", "preprocess", "h2d")
_PACKING_DURATIONS = (
    "d2h",
    "postprocess",
    "pack",
    "hash",
    "compress",
    "protocol_serialize",
)


class ReplaySchemaError(ValueError):
    """Raised when a source record or replay sample violates the contract."""


@dataclass(frozen=True)
class ReplaySample:
    sample_id: str
    ordinal: int
    source_trace_id: str
    source_sequence_id: int
    base_delay_ns: int
    stage_ns: Mapping[str, int]
    raw_payload_bytes: int
    wire_payload_bytes: int
    capture_condition_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage_ns", MappingProxyType(dict(self.stage_ns))
        )


@dataclass(frozen=True)
class NormalizationResult:
    samples: Tuple[ReplaySample, ...]
    total_count: int
    warmup_count: int
    rejected_count: int
    accepted_count: int
    identity: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(
            self, "identity", MappingProxyType(dict(self.identity))
        )


def _error(path: str, message: str) -> None:
    raise ReplaySchemaError("%s: %s" % (path, message))


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(path, "must be an object")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(path, "must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _error(path, "must be an integer >= 0")
    return value


def _require_replay_fields(record: Mapping[str, Any]) -> None:
    actual = set(record)
    missing = _REPLAY_FIELDS - actual
    unknown = actual - _REPLAY_FIELDS
    if missing:
        _error(sorted(missing)[0], "is required")
    if unknown:
        _error(sorted(unknown)[0], "is not allowed")


def parse_replay_sample(record: Mapping[str, Any]) -> ReplaySample:
    """Validate a replay sample mapping and return an immutable value."""
    value = _require_mapping(record, "record")
    _require_replay_fields(value)
    if value["schema_version"] != REPLAY_SAMPLE_SCHEMA_VERSION:
        _error("schema_version", "must equal %s" % REPLAY_SAMPLE_SCHEMA_VERSION)

    sample_id = _require_string(value["sample_id"], "sample_id")
    source_trace_id = _require_string(value["source_trace_id"], "source_trace_id")
    capture_condition_id = _require_string(
        value["capture_condition_id"], "capture_condition_id"
    )
    ordinal = _require_nonnegative_int(value["ordinal"], "ordinal")
    source_sequence_id = _require_nonnegative_int(
        value["source_sequence_id"], "source_sequence_id"
    )
    base_delay_ns = _require_nonnegative_int(value["base_delay_ns"], "base_delay_ns")
    raw_payload_bytes = _require_nonnegative_int(
        value["raw_payload_bytes"], "raw_payload_bytes"
    )
    wire_payload_bytes = _require_nonnegative_int(
        value["wire_payload_bytes"], "wire_payload_bytes"
    )

    raw_stages = _require_mapping(value["stage_ns"], "stage_ns")
    missing_stages = set(STAGE_NAMES) - set(raw_stages)
    extra_stages = set(raw_stages) - set(STAGE_NAMES)
    if missing_stages:
        _error("stage_ns.%s" % sorted(missing_stages)[0], "is required")
    if extra_stages:
        _error("stage_ns.%s" % sorted(extra_stages)[0], "is not allowed")
    stages = {
        name: _require_nonnegative_int(raw_stages[name], "stage_ns.%s" % name)
        for name in STAGE_NAMES
    }
    if sum(stages.values()) != base_delay_ns:
        _error("base_delay_ns", "must equal the sum of stage_ns")

    return ReplaySample(
        sample_id=sample_id,
        ordinal=ordinal,
        source_trace_id=source_trace_id,
        source_sequence_id=source_sequence_id,
        base_delay_ns=base_delay_ns,
        stage_ns=stages,
        raw_payload_bytes=raw_payload_bytes,
        wire_payload_bytes=wire_payload_bytes,
        capture_condition_id=capture_condition_id,
    )


def replay_sample_to_dict(sample: ReplaySample) -> Dict[str, Any]:
    """Return a JSON-compatible independent representation of ``sample``."""
    return {
        "schema_version": REPLAY_SAMPLE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "ordinal": sample.ordinal,
        "source_trace_id": sample.source_trace_id,
        "source_sequence_id": sample.source_sequence_id,
        "base_delay_ns": sample.base_delay_ns,
        "stage_ns": {name: sample.stage_ns[name] for name in STAGE_NAMES},
        "raw_payload_bytes": sample.raw_payload_bytes,
        "wire_payload_bytes": sample.wire_payload_bytes,
        "capture_condition_id": sample.capture_condition_id,
    }


def _nested_string(record: Mapping[str, Any], label: str, path: Tuple[str, ...]) -> str:
    value: Any = record
    for component in path:
        mapping = _require_mapping(value, label)
        if component not in mapping:
            _error(label, "is required")
        value = mapping[component]
    return _require_string(value, label)


def _profile_identity(record: Mapping[str, Any]) -> Dict[str, str]:
    return {
        label: _nested_string(record, label, path)
        for label, path in _IDENTITY_PATHS
    }


def _check_same_identity(
    expected: Mapping[str, str], actual: Mapping[str, str], index: int
) -> None:
    for label, _ in _IDENTITY_PATHS:
        if actual[label] != expected[label]:
            _error("records[%d].%s" % (index, label), "does not match profile identity")


def _duration(record: Mapping[str, Any], name: str, index: int) -> int:
    durations = _require_mapping(
        record.get("durations_ns"), "records[%d].durations_ns" % index
    )
    if name not in durations:
        _error("records[%d].durations_ns.%s" % (index, name), "is required")
    return _require_nonnegative_int(
        durations[name], "records[%d].durations_ns.%s" % (index, name)
    )


def _require_eligible(record: Mapping[str, Any], index: int) -> None:
    prefix = "records[%d]" % index
    if record.get("measurement_scope") != "rsu_to_fusion":
        _error(prefix + ".measurement_scope", "must equal rsu_to_fusion")
    comparability = _require_mapping(
        record.get("clock_comparability"), prefix + ".clock_comparability"
    )
    if comparability.get("comparable") is not True:
        _error(prefix + ".clock_comparability", "clocks must be comparable")
    if record.get("headline_eligible") is not True:
        _error(prefix + ".headline_eligible", "must be true")
    status = _require_mapping(record.get("status"), prefix + ".status")
    if status.get("outcome") != "consumed":
        _error(prefix + ".status.outcome", "must equal consumed")
    network = _require_mapping(record.get("network"), prefix + ".network")
    if "remote_input_tensor_bytes" not in network:
        _error(prefix + ".network.remote_input_tensor_bytes", "is required")
    if _require_nonnegative_int(
        network["remote_input_tensor_bytes"],
        prefix + ".network.remote_input_tensor_bytes",
    ) != 0:
        _error(prefix + ".network.remote_input_tensor_bytes", "must equal 0")
    codec = _require_mapping(record.get("codec"), prefix + ".codec")
    if codec.get("name") != "none":
        _error(prefix + ".codec.name", "must equal none")


def _sample_id(
    source_trace_id: str, sequence_id: int, capture_condition_id: str
) -> str:
    identity = {
        "capture_condition_id": capture_condition_id,
        "normalizer_version": NORMALIZER_VERSION,
        "source_sequence_id": sequence_id,
        "source_trace_id": source_trace_id,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_sample(
    record: Mapping[str, Any], ordinal: int, capture_condition_id: str, index: int
) -> ReplaySample:
    derived = _require_mapping(
        record.get("derived_ns"), "records[%d].derived_ns" % index
    )
    if "rsu_to_fusion" not in derived:
        _error("records[%d].derived_ns.rsu_to_fusion" % index, "is required")
    base_delay_ns = _require_nonnegative_int(
        derived["rsu_to_fusion"], "records[%d].derived_ns.rsu_to_fusion" % index
    )
    acquisition = sum(_duration(record, name, index) for name in _ACQUISITION_DURATIONS)
    inference = _duration(record, "gpu_inference", index)
    packing = sum(_duration(record, name, index) for name in _PACKING_DURATIONS)
    transport = base_delay_ns - acquisition - inference - packing
    if transport < 0:
        _error("records[%d].stage_ns.transport_receive" % index, "must be >= 0")

    message = _require_mapping(record.get("message"), "records[%d].message" % index)
    sequence_id = _require_nonnegative_int(
        message.get("sequence_id"), "records[%d].message.sequence_id" % index
    )
    trace_id = _require_string(record.get("trace_id"), "records[%d].trace_id" % index)
    codec = _require_mapping(record.get("codec"), "records[%d].codec" % index)
    return ReplaySample(
        sample_id=_sample_id(trace_id, sequence_id, capture_condition_id),
        ordinal=ordinal,
        source_trace_id=trace_id,
        source_sequence_id=sequence_id,
        base_delay_ns=base_delay_ns,
        stage_ns={
            "acquisition_preprocess": acquisition,
            "gpu_inference": inference,
            "packing_postprocess": packing,
            "transport_receive": transport,
        },
        raw_payload_bytes=_require_nonnegative_int(
            codec.get("raw_payload_bytes"),
            "records[%d].codec.raw_payload_bytes" % index,
        ),
        wire_payload_bytes=_require_nonnegative_int(
            codec.get("wire_payload_bytes"),
            "records[%d].codec.wire_payload_bytes" % index,
        ),
        capture_condition_id=capture_condition_id,
    )


def normalize_raw_trace(
    records: Iterable[Mapping[str, Any]], capture_condition_id: str
) -> NormalizationResult:
    """Normalize one strictly eligible Raw trace into ordered replay samples."""
    condition = _require_string(capture_condition_id, "capture_condition_id")
    source_records = list(records)
    if not source_records:
        _error("records", "must not be empty")
    try:
        normalized = validate_trace_sequence(source_records)
    except TraceValidationError as error:
        raise ReplaySchemaError(str(error))

    for index, record in enumerate(normalized):
        if record.get("schema_version") != "ditl.trace.v2":
            _error("records[%d].schema_version" % index, "must equal ditl.trace.v2")

    identity = _profile_identity(normalized[0])
    previous_sequence_id = None
    samples = []
    warmup_count = 0
    for index, record in enumerate(normalized):
        current_identity = _profile_identity(record)
        _check_same_identity(identity, current_identity, index)
        message = _require_mapping(record.get("message"), "records[%d].message" % index)
        sequence_id = _require_nonnegative_int(
            message.get("sequence_id"), "records[%d].message.sequence_id" % index
        )
        if previous_sequence_id is not None and sequence_id <= previous_sequence_id:
            _error(
                "records[%d].message.sequence_id" % index,
                "must be strictly increasing",
            )
        previous_sequence_id = sequence_id

        warmup = record.get("warmup", False)
        if not isinstance(warmup, bool):
            _error("records[%d].warmup" % index, "must be boolean")
        if warmup:
            warmup_count += 1
            continue
        _require_eligible(record, index)
        samples.append(_normalize_sample(record, len(samples), condition, index))

    return NormalizationResult(
        samples=samples,
        total_count=len(normalized),
        warmup_count=warmup_count,
        rejected_count=warmup_count,
        accepted_count=len(samples),
        identity=identity,
    )
