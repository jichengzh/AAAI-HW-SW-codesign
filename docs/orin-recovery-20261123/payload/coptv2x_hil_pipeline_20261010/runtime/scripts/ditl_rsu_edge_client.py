#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import socket
import sys
import uuid
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from ditl_edge_protocol import PROTOCOL_VERSION, ProtocolLimits
from ditl_edge_runtime import build_round_trip_trace, request_inference
from ditl_trace_schema import write_latency_csv, write_trace_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one RSU tensor replay against a DitL edge service."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=55020)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--latency-csv", required=True)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--run-id", default="ditl-edge-smoke")
    parser.add_argument("--route-id", default="tensor-replay")
    parser.add_argument("--scenario-id", default="edge-rsu")
    parser.add_argument("--source-id", default="rsu-orin-01")
    parser.add_argument("--target-id", default="ego-server-01")
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--sequence-id", type=int, default=0)
    parser.add_argument("--topic", default="/ditl/rsu/features")
    parser.add_argument("--message-type", default="intermediate_feature")
    parser.add_argument("--compression-ratio", type=float, default=1.0)
    parser.add_argument("--request-compression", choices=("none", "zlib"), default="none")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-payload-mib", type=int, default=512)
    parser.add_argument("--observer-host-id", default=socket.gethostname())
    return parser.parse_args(argv)


def load_npz(path):
    source = Path(path)
    if not source.is_file():
        raise ValueError("input NPZ not found: %s" % source)
    with np.load(str(source), allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("input NPZ contains no arrays")
        return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.frame_id < 0 or args.sequence_id < 0:
        raise SystemExit("frame and sequence IDs must be non-negative")
    if args.timeout_s <= 0 or args.max_payload_mib <= 0:
        raise SystemExit("timeout and payload limit must be positive")
    arrays = load_npz(args.input_npz)
    request_id = args.request_id or ("req-" + uuid.uuid4().hex)
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "request",
        "request_id": request_id,
        "operation": "infer",
        "run_id": args.run_id,
        "route_id": args.route_id,
        "scenario_id": args.scenario_id,
        "source_id": args.source_id,
        "target_id": args.target_id,
        "frame_id": args.frame_id,
        "sequence_id": args.sequence_id,
        "topic": args.topic,
        "message_type": args.message_type,
        "compression_ratio": args.compression_ratio,
    }
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )
    response, outputs, measurement = request_inference(
        args.host,
        args.port,
        request,
        arrays,
        timeout_s=args.timeout_s,
        request_compression=args.request_compression,
        limits=limits,
    )
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **outputs)
    trace = build_round_trip_trace(
        request,
        response,
        measurement,
        observer_host_id=args.observer_host_id,
    )
    write_trace_jsonl(args.trace_jsonl, [trace])
    write_latency_csv(args.latency_csv, [trace])
    summary = {
        "request_id": request_id,
        "model_id": response["model"]["model_id"],
        "backend": response["model"]["backend"],
        "round_trip_ms": measurement["round_trip_ns"] / 1_000_000.0,
        "request_wire_bytes": measurement["request_wire_bytes"],
        "response_wire_bytes": measurement["response_wire_bytes"],
        "output_names": sorted(outputs),
        "trace_jsonl": str(Path(args.trace_jsonl)),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
