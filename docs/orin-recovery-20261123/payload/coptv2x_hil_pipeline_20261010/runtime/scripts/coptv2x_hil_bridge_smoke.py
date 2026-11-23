#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_artifacts import HIL_ARM_IDS
from coptv2x_hil_backend import CoptV2xHilBackend
from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger
from ditl_edge_runtime import monotonic_raw_ns
from ditl_rsu_pipeline import CoptV2xHilPipeline
from ditl_rsu_source import LocalRsuFrame
from scripts.ditl_rsu_edge_publisher import (
    _connect_local_result_handoff,
    _prepare_result_frame,
)
from ditl_edge_protocol import send_wire_frame


EXPECTED_INPUT_SHA256 = "0897da44f61df0a05f75963abbcfe80a7d8c35a9670c9b0d69c962c296f99ddb"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result_handoff(host, port):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("CoptV2X result handoff must stay on loopback")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("CoptV2X result handoff port is invalid")
    return host, port


def _fixed_input(path):
    path = Path(path).expanduser().resolve()
    if _sha256(path) != EXPECTED_INPUT_SHA256:
        raise ValueError("bridge smoke input is not the frozen Stage6 tensor")
    value = np.load(str(path), allow_pickle=False)
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if tuple(value.shape) != (2, 64, 256, 512) or value.dtype != np.float32:
        raise ValueError("bridge smoke input must be [2,64,256,512] float32")
    return np.array(value, copy=True, order="C")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Send one Orin-local CoptV2X result through the ROS bridge"
    )
    parser.add_argument("--arm", choices=HIL_ARM_IDS, default="joint_shcosearch")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--native-repo-root", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--result-host", default="127.0.0.1")
    parser.add_argument("--result-port", type=int, default=55143)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-id", default="frozen-stage6-reference")
    parser.add_argument("--scenario-id", default="orin-ros-bridge-smoke")
    parser.add_argument("--sample-id", default="codriving-latency-input")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_result_handoff(args.result_host, args.result_port)
    backend = CoptV2xHilBackend(
        args.arm,
        artifact_root=args.artifact_root,
        config_dir=args.config_dir,
        native_repo_root=args.native_repo_root,
    )
    backend.load(args.arm)
    trigger = parse_hil_trigger(
        {
            "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
            "run_id": args.run_id,
            "route_id": args.route_id,
            "scenario_id": args.scenario_id,
            "sample_id": args.sample_id,
            "sequence_no": 0,
            "source_frame": 15,
            "source_sim_time_ns": 0,
            "arm_id": args.arm,
        }
    )
    source_start = monotonic_raw_ns()
    spatial_features = _fixed_input(args.input_npy)
    source_end = monotonic_raw_ns()
    frame = LocalRsuFrame(
        frame_id=trigger.source_frame,
        sequence_id=trigger.sequence_no,
        data_available_ns=source_start,
        arrays={"spatial_features": spatial_features},
        source_metadata={
            "source_type": "local_indexed_feature",
            "trigger_identity": trigger.to_metadata(),
            "batch_order": ("ego", "rsu"),
            "remote_input_tensor_bytes": 0,
            "timestamps_ns": {
                "source_read_start": source_start,
                "source_read_end": source_end,
            },
            "durations_ns": {"source_read": source_end - source_start},
        },
    )
    pipeline = CoptV2xHilPipeline(
        backend,
        run_id=trigger.run_id,
        route_id=trigger.route_id,
        scenario_id=trigger.scenario_id,
        compression="none",
    )
    result = pipeline.process(frame)
    with _connect_local_result_handoff(
        args.result_host, args.result_port, timeout_s=60.0
    ) as connection:
        header_bytes, payload = _prepare_result_frame(result, warmup=False)
        send_wire_frame(connection, header_bytes, payload)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.output_npz), **result.outputs)
    metadata = {
        "schema_version": "coptv2x.hil-ros-bridge-smoke.v1",
        "arm_id": args.arm,
        "trigger_identity": dict(result.header["trigger_identity"]),
        "artifact_identity": dict(result.header["artifact_identity"]),
        "payload_sha256": result.header["payload_sha256"],
        "payload_semantics": result.header["payload_semantics"],
        "remote_input_tensor_bytes": result.header["remote_input_tensor_bytes"],
        "result_handoff": "%s:%d" % (args.result_host, args.result_port),
    }
    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
