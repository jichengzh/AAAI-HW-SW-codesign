#!/usr/bin/env python3
"""Controlled local_compute_only measurement; NOT an HIL or CARLA run.

Use the same Orin Python environment, PYTHONPATH, inputs and artifacts as the HIL
worker. Replace only its local result socket with a byte-counting null sink.
Example (keep existing absolute runtime/input/artifact paths):

  HIL_TIMING_LOG=/diagnostic/local_timing.jsonl python3 /diagnostic/local_probe.py \
    --manifest /inputs/manifest.json --data-root /inputs \
    --artifact-root /artifacts --config-dir /runtime/frozen-configs \
    --native-repo-root /native --arm compression_only --count 20 \
    --run-id timing-local-compression-20261007

Manifest order is retained. Sequence numbers are 0..count-1. Simulation time is
synthetic source_frame * 50,000,000 ns (the declared 20 Hz host convention), NOT
an observed CARLA timestamp. The first call is cold and reported separately;
there is no extra warmup. The original worker's result_count remains 2225.
Wire compression is the unchanged original-worker default, 'none'. Original
send_wire_frame still assembles the exact full frame before the sink counts it.
"""
from __future__ import print_function

import argparse
import importlib
import json
import os
from pathlib import Path


class NullConnection(object):
    """No socket, payload retention, hashing, decode, or filesystem traffic."""
    def __init__(self):
        self.total_bytes = 0
        self.send_calls = 0
        self.closed = False

    def sendall(self, data):
        if self.closed:
            raise OSError("null connection is closed")
        self.total_bytes += memoryview(data).nbytes
        self.send_calls += 1

    def close(self):
        self.closed = True


def trigger_payloads(manifest, run_id, arm, count):
    samples = manifest.get("samples", [])
    if type(count) is not int or not 1 <= count <= len(samples):
        raise ValueError("count must select a nonempty existing manifest prefix")
    return [{"schema_version": "coptv2x.hil-trigger.v1", "run_id": run_id,
             "route_id": manifest["route_id"], "scenario_id": manifest["scenario_id"],
             "sample_id": sample["sample_id"], "sequence_no": index,
             "source_frame": sample["source_frame"],
             "source_sim_time_ns": sample["source_frame"] * 50000000,
             "arm_id": arm}
            for index, sample in enumerate(samples[:count])]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "artifact-root", "config-dir", "native-repo-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("compression_only",), default="compression_only")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--run-id", default="timing-local-compression-20261007")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.environ.get("HIL_TIMING_LOG"):
        raise SystemExit("HIL_TIMING_LOG is required for the local diagnostic")
    if not 1 <= args.count <= 2225:
        raise SystemExit("count must be between 1 and frozen result_count 2225")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    payloads = trigger_payloads(manifest, args.run_id, args.arm, args.count)
    original = importlib.import_module("scripts.coptv2x_hil_orin_worker")
    from worker_probe import install, monotonic_raw_ns
    triggers = [original.parse_worker_trigger_line((json.dumps(value) + "\n").encode("utf-8"))
                for value in payloads]
    worker_args = original.build_parser().parse_args([
        "--manifest", args.manifest, "--data-root", args.data_root,
        "--artifact-root", args.artifact_root, "--config-dir", args.config_dir,
        "--native-repo-root", args.native_repo_root, "--result-count", "2225"])
    install(original)
    connection = NullConnection()
    connect = original._connect_local_result_handoff
    original._connect_local_result_handoff = lambda *unused: connection
    try:
        worker = original.OrinHilWorker(worker_args)
    finally:
        original._connect_local_result_handoff = connect
    durations = []
    try:
        for index, trigger in enumerate(triggers):
            start = monotonic_raw_ns()
            request_id = worker.process_trigger(trigger)
            duration = monotonic_raw_ns() - start
            durations.append(duration)
            print(json.dumps({"event": "local_compute_only.result", "request_id": request_id,
                              "sequence_no": trigger.sequence_no, "cold": index == 0,
                              "sample_id": trigger.sample_id, "source_frame": trigger.source_frame,
                              "source_sim_time_ns": trigger.source_sim_time_ns,
                              "simulation_time_basis": "synthetic_source_frame_20hz",
                              "process_call_including_probe_flush_ns": duration,
                              "input_sha256": {name: entry["sha256"] for name, entry in
                                               manifest["samples"][index]["inputs"].items()}},
                             sort_keys=True), flush=True)
    finally:
        worker.close()
    print(json.dumps({"event": "local_compute_only.complete", "measurement": "local_compute_only",
                      "is_hil": False, "count": len(durations), "extra_warmup_count": 0,
                      "cold_first_ns": durations[0],
                      "remaining_mean_ns": sum(durations[1:]) // (len(durations) - 1)
                      if len(durations) > 1 else None,
                      "null_handoff_wire_bytes": connection.total_bytes,
                      "null_handoff_send_calls": connection.send_calls,
                      "wire_compression": worker_args.wire_compression}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
