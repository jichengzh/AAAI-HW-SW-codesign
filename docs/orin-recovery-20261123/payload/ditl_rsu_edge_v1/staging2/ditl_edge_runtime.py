from __future__ import print_function

import copy
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from ditl_edge_protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    ProtocolLimits,
    decode_wire_frame,
    prepare_envelope,
    prepare_envelope_header,
    recv_wire_frame,
    send_wire_frame,
)


def monotonic_raw_ns():
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if clock is not None and hasattr(time, "clock_gettime_ns"):
        return time.clock_gettime_ns(clock)
    return time.monotonic_ns()


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown-boot"


def collect_device_snapshot():
    snapshot = {
        "hostname": socket.gethostname(),
        "loadavg_1m": float(os.getloadavg()[0]),
        "monotonic_raw_ns": monotonic_raw_ns(),
    }
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = int(value.strip().split()[0]) * 1024
        snapshot["memory_available_bytes"] = meminfo.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        snapshot["memory_available_bytes"] = None
    return snapshot


class ReplayBackend(object):
    """Deterministic backend for transport validation before model deployment."""

    def __init__(self, scale=1.0, model_id="replay-rsu-v1"):
        self.scale = float(scale)
        self.model_id = str(model_id)
        self.inference_count = 0

    def infer(self, arrays):
        start = monotonic_raw_ns()
        outputs = {
            name: np.ascontiguousarray(value * self.scale)
            for name, value in arrays.items()
        }
        end = monotonic_raw_ns()
        self.inference_count += 1
        return {
            "outputs": outputs,
            "model": {
                "backend": "replay",
                "model_id": self.model_id,
            },
            "durations_ns": {"gpu_inference": max(0, end - start)},
            "edge_timestamps_ns": {
                "inference_start": start,
                "inference_end": end,
            },
        }


def _error_header(request_id, code, message):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "error",
        "request_id": request_id,
        "error_code": code,
        "message": str(message)[:512],
    }


def handle_connection(
    connection,
    address,
    backend,
    response_compression="none",
    limits=ProtocolLimits(),
):
    receive_start = monotonic_raw_ns()
    try:
        header_bytes, payload, request_wire_bytes = recv_wire_frame(connection, limits)
        receive_end = monotonic_raw_ns()
        deserialize_start = monotonic_raw_ns()
        request, arrays = decode_wire_frame(header_bytes, payload, limits)
        deserialize_end = monotonic_raw_ns()
    except ProtocolError as error:
        response = _error_header("invalid-request", "invalid_frame", error)
        prepared_header, prepared_payload = prepare_envelope(response, {}, "none", limits)
        send_wire_frame(connection, prepared_header, prepared_payload)
        return response

    request_id = request["request_id"]
    if request.get("kind") != "request" or request.get("operation") != "infer":
        response = _error_header(
            request_id, "unsupported_operation", "only infer requests are accepted"
        )
        prepared_header, prepared_payload = prepare_envelope(response, {}, "none", limits)
        send_wire_frame(connection, prepared_header, prepared_payload)
        return response

    try:
        inference = backend.infer(arrays)
        serialize_start = monotonic_raw_ns()
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "response",
            "request_id": request_id,
            "status": "ok",
            "model": copy.deepcopy(inference["model"]),
            "durations_ns": copy.deepcopy(inference.get("durations_ns", {})),
            "edge_timestamps_ns": copy.deepcopy(
                inference.get("edge_timestamps_ns", {})
            ),
            "device": collect_device_snapshot(),
            "request_wire_bytes": request_wire_bytes,
            "peer": str(address[0]) if isinstance(address, tuple) else str(address),
        }
        response["edge_timestamps_ns"].update(
            {
                "request_receive_start": receive_start,
                "request_receive_end": receive_end,
                "deserialize_start": deserialize_start,
                "deserialize_end": deserialize_end,
            }
        )
        prepared_header, prepared_payload = prepare_envelope(
            response, inference["outputs"], response_compression, limits
        )
        response_tensor_bundle = json.loads(
            prepared_header.decode("utf-8")
        )["tensor_bundle"]
        serialize_end = monotonic_raw_ns()
        response["durations_ns"]["deserialize"] = max(
            0, deserialize_end - deserialize_start
        )
        response["durations_ns"]["serialize"] = max(
            0, serialize_end - serialize_start
        )
        response["edge_timestamps_ns"]["serialize_start"] = serialize_start
        response["edge_timestamps_ns"]["serialize_end"] = serialize_end
        prepared_header = prepare_envelope_header(
            response, response_tensor_bundle, limits
        )
        send_wire_frame(connection, prepared_header, prepared_payload)
        return response
    except Exception as error:
        response = _error_header(request_id, "backend_error", type(error).__name__)
        prepared_header, prepared_payload = prepare_envelope(response, {}, "none", limits)
        send_wire_frame(connection, prepared_header, prepared_payload)
        return response


def request_inference(
    host,
    port,
    request,
    arrays,
    timeout_s=30.0,
    request_compression="none",
    limits=ProtocolLimits(),
):
    start = monotonic_raw_ns()
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as connection:
        connection.settimeout(float(timeout_s))
        request_wire_bytes, request_tensor_bundle = _send_timed_request(
            connection, request, arrays, request_compression, limits
        )
        send_end = monotonic_raw_ns()
        response, outputs, response_wire_bytes = _receive_timed_response(
            connection, limits
        )
        receive_end = monotonic_raw_ns()
    if response["request_id"] != request["request_id"]:
        raise ProtocolError("response request_id does not match request")
    if response["kind"] == "error":
        raise ProtocolError(
            "edge service error %s" % response.get("error_code", "unknown")
        )
    if response["kind"] != "response" or response.get("status") != "ok":
        raise ProtocolError("invalid edge response")
    return response, outputs, {
        "request_send_start_ns": start,
        "request_send_end_ns": send_end,
        "response_receive_ns": receive_end,
        "round_trip_ns": receive_end - start,
        "request_wire_bytes": request_wire_bytes,
        "response_wire_bytes": response_wire_bytes,
        "request_tensor_bundle": request_tensor_bundle,
    }


def _send_timed_request(connection, request, arrays, compression, limits):
    header_bytes, payload = prepare_envelope(request, arrays, compression, limits)
    request_tensor_bundle = json.loads(header_bytes.decode("utf-8"))["tensor_bundle"]
    wire_bytes = send_wire_frame(connection, header_bytes, payload)
    return wire_bytes, request_tensor_bundle


def _receive_timed_response(connection, limits):
    header_bytes, payload, wire_bytes = recv_wire_frame(connection, limits)
    header, arrays = decode_wire_frame(header_bytes, payload, limits)
    return header, arrays, wire_bytes


def build_round_trip_trace(request, response, measurement, observer_host_id=None):
    """Build a same-clock RTT probe record; it is not a PTP one-way measurement."""
    observer = observer_host_id or socket.gethostname()
    boot = boot_id()
    request_bundle = measurement.get("request_tensor_bundle")
    if not isinstance(request_bundle, dict):
        raise ProtocolError("measurement.request_tensor_bundle is required")
    descriptors = request_bundle.get("arrays", [])
    first = descriptors[0] if descriptors else {
        "shape": [1],
        "dtype": "uint8",
    }
    source_ns = int(measurement["request_send_start_ns"])
    publish_ns = int(measurement["request_send_end_ns"])
    receive_ns = int(measurement["response_receive_ns"])
    durations = {
        str(name): int(value)
        for name, value in response.get("durations_ns", {}).items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }
    durations["sim_bridge_round_trip"] = int(measurement["round_trip_ns"])
    return {
        "schema_version": "ditl.trace.v1",
        "trace_id": request["request_id"],
        "run_id": request["run_id"],
        "route_id": request["route_id"],
        "scenario_id": request["scenario_id"],
        "message": {
            "source_id": request["source_id"],
            "target_id": request["target_id"],
            "frame_id": int(request["frame_id"]),
            "sequence_id": int(request["sequence_id"]),
            "topic": request["topic"],
            "message_type": request["message_type"],
            "payload_bytes": int(request_bundle["raw_payload_bytes"]),
            "tensor_shape": list(first["shape"]),
            "tensor_dtype": first["dtype"],
            "compression_ratio": float(request["compression_ratio"]),
        },
        "clock": {
            "sender": {
                "host_id": observer,
                "domain": "monotonic_raw",
                "boot_id": boot,
                "sync_state": "locked",
                "offset_ns": 0,
                "uncertainty_ns": 0,
            },
            "receiver": {
                "host_id": observer,
                "domain": "monotonic_raw",
                "boot_id": boot,
                "sync_state": "locked",
                "offset_ns": 0,
                "uncertainty_ns": 0,
            },
        },
        "status": {
            "outcome": "consumed",
            "duplicate": False,
            "out_of_order": False,
            "reason": None,
        },
        "timestamps_ns": {
            "source": source_ns,
            "publish": publish_ns,
            "receive": receive_ns,
            "fusion_consume": receive_ns,
        },
        "network": {
            "transport": "tcp_round_trip_probe",
            "qos": "reliable",
            "sender_count": int(request["sequence_id"]) + 1,
            "receiver_count": int(request["sequence_id"]) + 1,
            "measurement_mode": "same_clock_round_trip",
            "request_wire_bytes": int(measurement["request_wire_bytes"]),
            "response_wire_bytes": int(measurement["response_wire_bytes"]),
        },
        "durations_ns": durations,
        "edge_timestamps_ns": copy.deepcopy(
            response.get("edge_timestamps_ns", {})
        ),
        "model": copy.deepcopy(response.get("model", {})),
        "device": copy.deepcopy(response.get("device", {})),
    }
