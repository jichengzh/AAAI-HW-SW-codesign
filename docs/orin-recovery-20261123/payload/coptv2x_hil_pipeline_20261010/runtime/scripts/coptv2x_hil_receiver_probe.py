#!/usr/bin/env python3
"""Fixed-wire CPU diagnostic for the production CARLA-host HIL receiver.

Run on the actual receiver host. This measures neither Orin inference nor
CARLA, ROS, network, or complete HIL. Each file is ONE fixed frame repeated N
times, not N independent frames. Mandatory decoder/admission/prefusion checks
remain inside the measured path. Registry initialization, wire loading, header
inspection, fresh-buffer setup and diagnostic assertions are outside it.
"""

import argparse
import json
from pathlib import Path
import platform
import socket
import sys
import time
from collections.abc import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = ROOT / "simulation/leaderboard/team_code/closedloop"
sys.path.insert(0, str(CLOSEDLOOP))
import coptv2x_hil_backend as backend
from coptv2x_hil_artifacts import DEFAULT_CONFIG_DIR
from coptv2x_hil_client import _remote_rsu_only
from coptv2x_hil_proxy_transport import _decode_result_wire
from coptv2x_hil_result_buffer import CoptV2xHilResultBuffer, _immutable_bytes_owner
from coptv2x_hil_trigger import parse_hil_trigger
from ditl_edge_protocol import FRAME_PREFIX, MAGIC, ProtocolLimits, inspect_envelope_header

SCHEMA_VERSION = "coptv2x.fixed-wire-receiver-probe.v1"
DEFAULT_MAX_WIRE_BYTES = 64 * 1024 * 1024
OUTPUT_NAMES = ("res0", "res1", "res2")


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _host():
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not processor and cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                processor = line.split(":", 1)[1].strip()
                break
    return {"hostname": socket.gethostname(), "processor": processor,
            "machine": platform.machine(), "platform": platform.platform(),
            "python_version": platform.python_version(), "numpy_version": np.__version__}


def _inspect_wire(wire, maximum):
    """Inspect only metadata; do not warm the production tensor decoder."""
    if not wire or len(wire) > maximum or len(wire) < FRAME_PREFIX.size:
        raise ValueError("wire is empty, too short, or exceeds max_wire_bytes")
    magic, header_size, payload_size = FRAME_PREFIX.unpack(wire[:FRAME_PREFIX.size])
    if magic != MAGIC or len(wire) != FRAME_PREFIX.size + header_size + payload_size:
        raise ValueError("wire prefix or frame length is invalid")
    limits = ProtocolLimits(max_payload_bytes=maximum, max_raw_payload_bytes=maximum)
    if header_size > limits.max_header_bytes:
        raise ValueError("wire header exceeds its limit")
    payload_start = FRAME_PREFIX.size + header_size
    return inspect_envelope_header(wire[FRAME_PREFIX.size:payload_start], wire[payload_start:], limits)


def summarize_samples(rows):
    warm = rows[1:]
    statistics = {}
    for name in sorted({name for row in warm for name in row["stages_ns"]}):
        values = [row["stages_ns"][name] for row in warm]
        # NumPy's default is linear in both the Python 3.7 and current runtimes;
        # the explicit keyword changed from interpolation to method across them.
        statistics[name] = {"count": len(values), "mean_ns": float(np.mean(values)),
                            "p50_ns": float(np.percentile(values, 50)),
                            "p95_ns": float(np.percentile(values, 95)),
                            "min_ns": int(min(values)), "max_ns": int(max(values))}
    return {"cold_sample": rows[0] if rows else None, "warm_sample_count": len(warm),
            "warm_stages_ns": statistics, "percentile_method": "linear",
            "cold_sample_scope": "first_timed_repeat_per_wire_not_process_or_os_cold"}


def _new_store(trigger, validator, checksum_dedup):
    kwargs = {"contract_validator": validator, "timing_mode": "serial_fixed_comm"}
    if checksum_dedup:
        kwargs["checksum_dedup"] = True
    store = CoptV2xHilResultBuffer(trigger.run_id, trigger.route_id, trigger.scenario_id,
                                 trigger.arm_id, capacity=1, **kwargs)
    store.register_trigger(trigger, published_ns=0)
    return store


def _assert_provenance(decoded, admitted, header, request_id):
    for name in ("trigger_identity", "artifact_identity", "payload_semantics", "payload_sha256",
                 "remote_input_tensor_bytes"):
        expected = header[name]
        if _plain(getattr(decoded, name)) != expected or _plain(getattr(admitted, name)) != expected:
            raise ValueError("receiver provenance mismatch: %s" % name)
    for field, key in (("timestamps_ns", "edge_timestamps_ns"), ("durations_ns", "durations_ns"),
                       ("model", "model")):
        if _plain(getattr(admitted, field)) != header.get(key, {}):
            raise ValueError("receiver provenance mismatch: %s" % field)
    if request_id != header["request_id"]:
        raise ValueError("receiver request_id mismatch")


def _memory_proof(decoded, admitted, remote):
    return {name: {
        "decode_readonly": not decoded.outputs[name].flags.writeable,
        "admitted_readonly": not admitted.outputs[name].flags.writeable,
        "prefusion_readonly": not forwarded.flags.writeable,
        "decode_immutable_bytes_owner": _immutable_bytes_owner(decoded.outputs[name]) is not None,
        "decode_admission_shared": bool(np.shares_memory(decoded.outputs[name], admitted.outputs[name])),
        "admission_prefusion_shared": bool(np.shares_memory(admitted.outputs[name], forwarded)),
    } for name, forwarded in zip(OUTPUT_NAMES, remote)}


def _repeat(wire, header, trigger, validator, maximum, checksum_dedup, index, comparison_hash):
    setup_start = time.perf_counter_ns()
    store = _new_store(trigger, validator, checksum_dedup)
    setup_ns = time.perf_counter_ns() - setup_start
    start = time.perf_counter_ns()
    envelope, request_id = _decode_result_wire(wire, maximum, readonly_results=True)
    decoded_end = time.perf_counter_ns()
    decision = store.accept(envelope, arrival_ns=0)
    admission_end = time.perf_counter_ns()
    if not decision.accepted:
        raise ValueError("receiver admission rejected: %s" % decision.reason)
    admitted = store.get_exact_result(trigger)
    lookup_end = time.perf_counter_ns()
    remote = _remote_rsu_only(admitted.outputs, payload_semantics=admitted.payload_semantics)
    end = time.perf_counter_ns()
    _assert_provenance(envelope.feature_result, admitted, header, request_id)
    if store.size != 1 or len(remote) != 3:
        raise ValueError("receiver result count mismatch")
    if any(value.shape[0] != 1 for value in remote):
        raise ValueError("prefusion did not produce RSU-only tensors")
    comparison = None
    if comparison_hash:
        comparison = backend.hash_feature_payload(admitted.outputs)
        if comparison != admitted.payload_sha256:
            raise ValueError("outside-window independent comparison hash mismatch")
    return {"event": "receiver_sample", "repeat_index": index,
            "temperature": "first_timed_sample" if index == 0 else "warm",
            "accepted": True, "canonical_hash_validated": True,
            "buffer_setup_and_register_ns_outside_window": setup_ns,
            "independent_comparison_sha256_outside_window": comparison,
            "memory_sharing": _memory_proof(envelope.feature_result, admitted, remote),
            "stages_ns": {"decode": decoded_end - start, "admission": admission_end - decoded_end,
                          "exact_lookup": lookup_end - admission_end, "prefusion": end - lookup_end,
                          "receiver_elapsed": end - start}}


def probe_wires(paths, config_dir=DEFAULT_CONFIG_DIR, count=20, checksum_dedup=False,
                max_wire_bytes=DEFAULT_MAX_WIRE_BYTES, contract_validator=None,
                independent_comparison_hash=False):
    """Return JSON-safe evidence; validator injection is for small CPU fixtures."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(max_wire_bytes) is not int or max_wire_bytes <= 0:
        raise ValueError("max_wire_bytes must be a positive integer")
    if type(checksum_dedup) is not bool or type(independent_comparison_hash) is not bool:
        raise ValueError("checksum options must be boolean")
    if not paths:
        raise ValueError("at least one wire file is required")
    init_start = time.perf_counter_ns()
    # Load all fixed samples before the first measured receiver window.
    loaded = [(str(Path(path).resolve()), Path(path).read_bytes()) for path in paths]
    inspected = [(path, wire, _inspect_wire(wire, max_wire_bytes)) for path, wire in loaded]
    validator_start = time.perf_counter_ns()
    validator = contract_validator or backend.create_feature_payload_validator(config_dir)
    validator_init_ns = time.perf_counter_ns() - validator_start
    initialization_ns = time.perf_counter_ns() - init_start
    host = _host()
    common = {"schema_version": SCHEMA_VERSION, "measurement_scope": "fixed_wire_cpu_receiver_diagnostic",
              "is_hil": False, "carla_ros_network_measured": False, "orin_inference_measured": False,
              "host": host, "timing_clock": "perf_counter_ns", "readonly_results": True,
              "checksum_dedup": checksum_dedup, "repeated_fixed_frame": True,
              "unique_frames_per_wire": 1, "count_requested": count,
              "mandatory_identity_finite_checksum_checks_in_window": True}
    records = [dict(common, event="receiver_initialization", initialization_ns=initialization_ns,
                    cached_validator_init_ns=validator_init_ns, wire_count=len(inspected),
                    config_dir=str(config_dir))]
    for path, wire, header in inspected:
        trigger = parse_hil_trigger(header.get("trigger_identity"))
        bundle = header["tensor_bundle"]
        detail = dict(common, wire_path=path, wire_bytes=len(wire), protocol_version=header["protocol_version"],
                      protocol_checksum_scope=bundle.get("sha256_scope", "raw_tensor_bytes_legacy"),
                      protocol_payload_sha256=bundle["sha256"], payload_sha256=header["payload_sha256"],
                      payload_semantics=header["payload_semantics"], wire_compression=bundle["compression"],
                      tensor_bytes=bundle["raw_payload_bytes"], payload_wire_bytes=bundle["wire_payload_bytes"],
                      tensor_shapes={entry["name"]: entry["shape"] for entry in bundle["arrays"]},
                      trigger_identity=trigger.to_metadata(), artifact_identity=header["artifact_identity"],
                      request_id=header["request_id"], arm_id=trigger.arm_id,
                      wire_source_provenance={key: header[key] for key in
                          ("source_id", "source_type", "edge_clock", "source_batch_order", "input_provenance", "model")
                          if key in header})
        samples = [dict(detail, **_repeat(wire, header, trigger, validator, max_wire_bytes,
                                         checksum_dedup, index, independent_comparison_hash))
                   for index in range(count)]
        if len(samples) != count or sum(row["accepted"] for row in samples) != count:
            raise ValueError("receiver repetition/acceptance count mismatch")
        records.extend(samples)
        records.append(dict(detail, event="receiver_summary", accepted_count=count,
                            **summarize_samples(samples)))
    return records


def main(argv=None, output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", action="append", required=True, help="One fixed raw result frame; repeat per arm")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--count", type=int, default=20, help="Repetitions of each fixed wire (default: 20)")
    parser.add_argument("--checksum-dedup", action="store_true", help="Opt into proven protocol checksum reuse")
    parser.add_argument("--max-wire-bytes", type=int, default=DEFAULT_MAX_WIRE_BYTES)
    parser.add_argument("--independent-comparison-hash", action="store_true",
                        help="Additional diagnostic canonical hash outside every timed window")
    args = parser.parse_args(argv)
    rows = probe_wires(args.wire, config_dir=args.config_dir, count=args.count,
                       checksum_dedup=args.checksum_dedup, max_wire_bytes=args.max_wire_bytes,
                       independent_comparison_hash=args.independent_comparison_hash)
    for row in rows:
        print(json.dumps(row, sort_keys=True, allow_nan=False), file=output or sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
