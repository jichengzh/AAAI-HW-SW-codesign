#!/usr/bin/env python3
from __future__ import print_function

import argparse
from dataclasses import replace
import json
import socket
import sys
import time
from pathlib import Path
from types import MappingProxyType

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from ditl_edge_protocol import ProtocolLimits, prepare_envelope_header, send_wire_frame
from ditl_edge_runtime import boot_id, monotonic_raw_ns
from ditl_rsu_pipeline import RsuEdgePipeline
from ditl_rsu_source import LocalRsuFrameSource
from ditl_trt_backend import TensorRtBackend


class HilTriggerProcessor(object):
    """Explicit indexed-HIL entry point; the legacy replay CLI stays unchanged."""

    def __init__(self, admission, source, pipeline):
        self.admission = admission
        self.source = source
        self.pipeline = pipeline

    def process_trigger(self, payload):
        trigger = self.admission.reserve(payload)
        try:
            frame = self.source.frame_for_trigger(trigger)
            self.admission.admit_trigger(trigger)
        except Exception:
            self.admission.cancel(trigger)
            raise
        result = self.pipeline.process(frame)
        identity = trigger.to_metadata()
        header = dict(result.header)
        for field in identity:
            if field in header and header[field] != identity[field]:
                raise ValueError("pipeline %s does not match trigger identity" % field)
        aliases = {
            "sequence_id": identity["sequence_no"],
            "frame_id": identity["source_frame"],
        }
        for field, value in aliases.items():
            if field in header and header[field] != value:
                raise ValueError("pipeline %s does not match trigger identity" % field)
        if (
            "trigger_identity" in header
            and header["trigger_identity"] != identity
        ):
            raise ValueError("pipeline trigger_identity does not match trigger")
        if "remote_input_tensor_bytes" in header:
            remote_input_tensor_bytes = header["remote_input_tensor_bytes"]
            if type(remote_input_tensor_bytes) is not int or remote_input_tensor_bytes != 0:
                raise ValueError(
                    "pipeline remote_input_tensor_bytes must be integer zero"
                )
        header.update(identity)
        header["trigger_identity"] = dict(identity)
        header["remote_input_tensor_bytes"] = 0
        return replace(result, header=MappingProxyType(header))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run an Orin-local RSU source and send result frames to the local ROS publisher."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--result-host", default="127.0.0.1")
    parser.add_argument("--result-port", type=int, default=55021)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-id", default="tensor-replay")
    parser.add_argument("--scenario-id", default="edge-rsu")
    parser.add_argument("--source-id", default="rsu-orin-01")
    parser.add_argument("--target-id", default="ego-server-01")
    parser.add_argument("--host-id", default="orin-agx-01")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--compression", choices=("none", "lz4", "zlib"), default="none")
    parser.add_argument("--max-payload-mib", type=int, default=512)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--output-npz")
    return parser


def _load_manifest(path):
    source = Path(path)
    if not source.is_file():
        raise ValueError("manifest file does not exist")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("manifest must contain an input mapping")
    return value


def _connect_local_result_handoff(host, port, timeout_s):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("result handoff must stay on localhost")
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            connection = socket.create_connection((host, port), timeout=2.0)
            connection.settimeout(timeout_s)
            return connection
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError("local result publisher is unavailable: %s" % last_error)


def _prepare_result_frame(result, warmup):
    header = dict(result.header)
    tensor_bundle = dict(result.tensor_bundle)
    header["warmup"] = bool(warmup)
    timestamps = dict(result.timestamps_ns)
    durations = dict(result.durations_ns)
    serialize_start = monotonic_raw_ns()
    timestamps["protocol_serialize_start"] = serialize_start
    timestamps["protocol_serialize_end"] = serialize_start
    timestamps["local_handoff_start"] = serialize_start
    durations["protocol_serialize"] = 0
    header["edge_timestamps_ns"] = timestamps
    header["durations_ns"] = durations
    prepare_envelope_header(header, tensor_bundle)
    serialize_end = monotonic_raw_ns()
    timestamps["protocol_serialize_start"] = serialize_start
    timestamps["protocol_serialize_end"] = serialize_end
    timestamps["local_handoff_start"] = serialize_end
    durations["protocol_serialize"] = max(0, serialize_end - serialize_start)
    header["edge_timestamps_ns"] = timestamps
    header["durations_ns"] = durations
    return prepare_envelope_header(header, tensor_bundle), result.payload


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.count <= 0 or args.warmup_count < 0 or args.warmup_count >= args.count:
        raise SystemExit("count must be positive and warmup-count must be smaller than count")
    if not 1 <= args.result_port <= 65535:
        raise SystemExit("result-port must be between 1 and 65535")
    if args.max_payload_mib <= 0 or args.connect_timeout_s <= 0:
        raise SystemExit("payload limit and timeout must be positive")
    backend = TensorRtBackend(args.engine, model_id=args.model_id)
    source = LocalRsuFrameSource(
        args.data_root,
        _load_manifest(args.manifest),
        backend.input_contract,
        source_id=args.source_id,
        cycle=True,
    )
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )
    pipeline = RsuEdgePipeline(
        backend,
        run_id=args.run_id,
        compression=args.compression,
        limits=limits,
        source_id=args.source_id,
        target_id=args.target_id,
        route_id=args.route_id,
        scenario_id=args.scenario_id,
        host_id=args.host_id,
        host_boot_id=boot_id(),
    )
    last_outputs = None
    with _connect_local_result_handoff(
        args.result_host, args.result_port, args.connect_timeout_s
    ) as connection:
        for index in range(args.count):
            frame = source.next_frame()
            result = pipeline.process(frame)
            header_bytes, payload = _prepare_result_frame(
                result, warmup=index < args.warmup_count
            )
            send_wire_frame(connection, header_bytes, payload)
            last_outputs = result.outputs
    if args.output_npz and last_outputs is not None:
        target = Path(args.output_npz)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(target), **last_outputs)
    print(
        json.dumps(
            {
                "event": "local_rsu_run_complete",
                "run_id": args.run_id,
                "frames": args.count,
                "warmup": args.warmup_count,
                "compression": args.compression,
                "remote_input_tensor_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
