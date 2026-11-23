#!/usr/bin/env python3
from __future__ import print_function

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_artifacts import HIL_ARM_IDS
from coptv2x_hil_backend import (
    CODRIVING_BATCH_ORDER,
    COPTV2X_PAYLOAD_SEMANTICS,
    CoptV2xHilBackend,
    hash_feature_payload,
)
from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger
from ditl_edge_runtime import monotonic_raw_ns


EXPECTED_INPUT_SHA256 = "0897da44f61df0a05f75963abbcfe80a7d8c35a9670c9b0d69c962c296f99ddb"
WARMUP_COUNT = 20
MEASURED_COUNT = 100
STAGES = ("local_source", "preprocess", "h2d", "inference", "d2h", "pack", "hash")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_input(path):
    value = np.load(str(path), allow_pickle=False)
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError("fixed input file must contain exactly one batch")
        value = value[0]
    if tuple(value.shape) != (2, 64, 256, 512) or value.dtype != np.dtype("float32"):
        raise ValueError("fixed input must be [2,64,256,512] float32")
    return np.array(value, copy=True, order="C")


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary_ns(values):
    return {
        "count": len(values),
        "min_ns": int(min(values)),
        "max_ns": int(max(values)),
        "mean_ns": float(statistics.mean(values)),
        "p50_ns": _percentile(values, 50),
        "p90_ns": _percentile(values, 90),
        "p99_ns": _percentile(values, 99),
    }


def _software_snapshot():
    import torch
    try:
        import tensorrt as trt
        tensorrt_version = trt.__version__
    except ImportError:
        tensorrt_version = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tensorrt": tensorrt_version,
    }


def _existing_stage6_evidence(artifact_root, arm_id):
    directory = "joint_shcosearch_fp16" if arm_id == "joint_shcosearch" else arm_id
    candidates = (
        Path(artifact_root) / "codriving" / directory / "primary_latency_power.json",
        Path(artifact_root) / "codriving" / directory / "primary_latency_power_sudo.json",
    )
    for path in candidates:
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            return {
                "path": str(path),
                "sha256": _sha256_file(path),
                "schema_version": document.get("schema_version"),
                "scope": document.get("scope"),
                "runtime": document.get("runtime", "tensorrt"),
                "engine_sha256": document.get("engine_sha256"),
                "checkpoint_sha256": document.get("checkpoint_sha256"),
                "input_sha256": document.get("inputs_sha256", document.get("input_sha256")),
                "output_shapes": document.get("output_shapes"),
                "output_channel_signature": document.get("output_channel_signature"),
            }
    raise ValueError("frozen Stage6 latency evidence is missing for %s" % arm_id)


def run(args):
    if args.warmup != WARMUP_COUNT or args.measured != MEASURED_COUNT:
        raise ValueError("Gate P2 requires exactly 20 warmup and 100 measured frames")
    input_path = args.input_npy.expanduser().resolve()
    input_sha256 = _sha256_file(input_path)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "fixed input SHA-256 mismatch: expected %s, got %s"
            % (EXPECTED_INPUT_SHA256, input_sha256)
        )
    spatial_features = _select_input(input_path)
    backend = CoptV2xHilBackend(
        arm_id=args.arm,
        artifact_root=args.artifact_root,
        config_dir=args.config_dir,
        native_repo_root=args.native_repo_root,
    )
    backend.load(args.arm)
    expected_contract = {
        tensor.name: (tuple(tensor.shape), np.dtype(tensor.dtype))
        for tensor in backend.spec.output_contract
    }
    stage_samples = {stage: [] for stage in STAGES}
    payload_hashes = []
    measured_frames = []
    artifact_identity = None
    total = args.warmup + args.measured
    for sequence_no in range(total):
        trigger_payload = {
            "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
            "run_id": args.run_id,
            "route_id": args.route_id,
            "scenario_id": args.scenario_id,
            "sample_id": args.sample_id,
            "sequence_no": sequence_no,
            "source_frame": args.source_frame,
            "source_sim_time_ns": sequence_no * 50000000,
            "arm_id": args.arm,
        }
        trigger = parse_hil_trigger(trigger_payload)
        source_start = monotonic_raw_ns()
        local_input = np.array(spatial_features, copy=True, order="C")
        source_end = monotonic_raw_ns()
        result = backend.infer(
            local_input,
            trigger,
            batch_order=CODRIVING_BATCH_ORDER,
            local_source_timestamps_ns={
                "local_source_start": source_start,
                "local_source_end": source_end,
            },
            local_source_durations_ns={"local_source": source_end - source_start},
        )
        if set(result.outputs) != set(expected_contract):
            raise ValueError("output names violate the frozen artifact contract")
        for name, value in result.outputs.items():
            expected_shape, expected_dtype = expected_contract[name]
            if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
                raise ValueError("%s shape/dtype violates the artifact contract" % name)
            if not np.isfinite(value).all():
                raise ValueError("%s contains a non-finite value" % name)
        if result.payload_semantics != COPTV2X_PAYLOAD_SEMANTICS:
            raise ValueError("payload semantics drifted")
        if result.remote_input_tensor_bytes != 0:
            raise ValueError("remote input tensor bytes must be exactly zero")
        if dict(result.trigger_identity) != trigger_payload:
            raise ValueError("trigger identity was not echoed exactly")
        if result.artifact_identity["arm_id"] != args.arm:
            raise ValueError("artifact identity arm drifted")
        artifact_identity = {
            key: dict(value) if hasattr(value, "items") else list(value) if isinstance(value, tuple) else value
            for key, value in result.artifact_identity.items()
        }
        if result.payload_sha256 != hash_feature_payload(result.outputs):
            raise ValueError("payload hash did not verify")
        if sequence_no >= args.warmup:
            for stage in STAGES:
                stage_samples[stage].append(int(result.durations_ns[stage]))
            payload_hashes.append(result.payload_sha256)
            measured_frames.append(
                {
                    "sequence_no": sequence_no,
                    "trigger_identity": dict(result.trigger_identity),
                    "payload_sha256": result.payload_sha256,
                    "durations_ns": {
                        stage: int(result.durations_ns[stage]) for stage in STAGES
                    },
                }
            )
    if len(set(payload_hashes)) != 1:
        raise ValueError("same frozen input did not produce deterministic outputs")
    frozen_evidence = _existing_stage6_evidence(args.artifact_root, args.arm)
    if frozen_evidence["input_sha256"] != input_sha256:
        raise ValueError("existing Stage6 runner used a different frozen input")
    if artifact_identity is None:
        raise ValueError("Gate P2 produced no frames")
    report = {
        "schema_version": "coptv2x.hil-orin-gate-p2.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS",
        "arm_id": args.arm,
        "run_id": args.run_id,
        "protocol": {
            "warmup_frames": args.warmup,
            "measured_frames": args.measured,
            "batch_order": list(CODRIVING_BATCH_ORDER),
            "remote_input_tensor_bytes": 0,
            "input_origin": "Orin-local frozen Stage6 contract input",
        },
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "shape": list(spatial_features.shape),
            "dtype": spatial_features.dtype.name,
        },
        "artifact_identity": artifact_identity,
        "output_contract": {
            name: {"shape": list(shape), "dtype": dtype.name}
            for name, (shape, dtype) in expected_contract.items()
        },
        "payload_semantics": COPTV2X_PAYLOAD_SEMANTICS,
        "determinism": {
            "measured_unique_payload_hashes": len(set(payload_hashes)),
            "payload_sha256": payload_hashes[0],
            "reference_mode": (
                "same frozen input repeatability plus frozen Stage6 AP/latency "
                "runner contract; no per-element reference was present"
            ),
        },
        "stage_latency_ns": {
            stage: _summary_ns(values) for stage, values in stage_samples.items()
        },
        "existing_stage6_evidence": frozen_evidence,
        "software_device": _software_snapshot(),
        "frames": measured_frames,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "arm_id": args.arm,
        "verdict": report["verdict"],
        "warmup": args.warmup,
        "measured": args.measured,
        "payload_sha256": payload_hashes[0],
        "output_json": str(args.output_json),
    }, sort_keys=True))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Run one frozen CoptV2X Orin Gate P2 arm")
    parser.add_argument("--arm", choices=HIL_ARM_IDS, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--native-repo-root", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-id", default="frozen-stage6-reference")
    parser.add_argument("--scenario-id", default="orin-gate-p2")
    parser.add_argument("--sample-id", default="codriving-latency-input")
    parser.add_argument("--source-frame", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=WARMUP_COUNT)
    parser.add_argument("--measured", type=int, default=MEASURED_COUNT)
    return parser


def main():
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
