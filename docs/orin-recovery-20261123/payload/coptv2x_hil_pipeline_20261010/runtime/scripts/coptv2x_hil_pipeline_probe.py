#!/usr/bin/env python3
"""Paired static replay: source to edge-ready, never network/HIL/CARLA timing.

Both modes use identical split stages. Receiver decode, validation, independent
hashing, sample printing and optional wire exports run after the ENTIRE window.
"""

import argparse
from collections import deque
from contextlib import redirect_stdout
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_paired_serial_helpers", str(Path(__file__).with_name("coptv2x_hil_serial_probe.py")))
_serial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serial)
HIL_ARM_IDS = _serial.HIL_ARM_IDS
SCHEMA_VERSION = "coptv2x.paired-pipeline-static-replay.v1"


def offered_schedule_ns(samples, cadence, sim_step_ms):
    if cadence not in ("source", "saturated") or not math.isfinite(sim_step_ms) or sim_step_ms <= 0:
        raise ValueError("invalid cadence or simulation step")
    frames = [sample["source_frame"] for sample in samples]
    if any(type(frame) is not int or frame < 0 for frame in frames):
        raise ValueError("source frames must be nonnegative integers")
    if any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError("source frames must increase strictly")
    step = int(round(sim_step_ms * 1000000))
    return [(frame - frames[0]) * step if cadence == "source" else 0 for frame in frames]


def verified_prefix(manifest, count):
    prefix = _serial.prefix_manifest(manifest, count)
    ids = [sample["sample_id"] for sample in prefix["samples"]]
    if len(set(ids)) != len(ids):
        raise ValueError("prefix must contain distinct sample identities")
    offered_schedule_ns(prefix["samples"], "saturated", 50)
    return prefix


def experimental_acceptance(samples, arms, modes, execution_success=None, pairs_verified=0):
    """Execution smoke is distinct from the five-arm paired evidence gate.

    Content identity ignores file paths and normalizes input names, file SHA,
    dtype and shape. The source verifies these declared file SHAs before a
    successful run can qualify; this is not a dynamic-world/HIL acceptance gate.
    """
    identities = set()
    for sample in samples:
        contract = {name: {"sha256": spec["sha256"].lower(), "shape": list(spec["shape"]),
                           "dtype": np.dtype(spec["dtype"]).name}
                    for name, spec in sample["inputs"].items()}
        identities.add(json.dumps(contract, sort_keys=True, separators=(",", ":")))
    distinct = len(identities)
    eligible = (len(samples) >= 20 and distinct >= 20
                and len(arms) == len(HIL_ARM_IDS) and set(arms) == set(HIL_ARM_IDS)
                and len(modes) == 2 and set(modes) == {"serial", "pipeline"})
    return {"distinct_input_content_count": distinct,
            "distinct_input_content_basis": "normalized_input_name_file_sha256_shape_dtype; source_verifies_sha_before_acceptance",
            "execution_success": execution_success, "success_scope": "execution_only_not_experimental_acceptance",
            "acceptance_scope": "paired_five_arm_static_replay_per_cadence",
            "acceptance_eligible": eligible,
            "acceptance_success": eligible and execution_success is True and pairs_verified == len(HIL_ARM_IDS) * len(samples)}


def load_runtime(runtime_root, need_executor=True):
    directory = Path(runtime_root).resolve() / "simulation/leaderboard/team_code/closedloop"
    if not directory.is_dir():
        raise ValueError("runtime-root has no closedloop module directory")
    runtime_names = {path.stem for path in directory.glob("*.py")
                     if path.stem.startswith(("coptv2x_", "ditl_"))}
    for name, module in tuple(sys.modules.items()):
        if name in runtime_names and getattr(module, "__file__", None):
            if Path(module.__file__).resolve().parent != directory:
                raise ValueError("runtime dependency already imported from another tree; use a fresh process")
    runtime = _serial.load_runtime(runtime_root)
    if need_executor:
        runtime.executor = importlib.import_module("coptv2x_hil_pipeline_executor")
    return runtime


def interval_union_ns(intervals):
    total, end = 0, None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("negative timestamp interval")
        total += right - left if end is None else max(0, right - max(left, end))
        end = right if end is None else max(end, right)
    return total


def _statistics(values):
    return {"count": len(values), "mean_ns": float(np.mean(values)),
            "p50_ns": float(np.percentile(values, 50)), "p95_ns": float(np.percentile(values, 95)),
            "min_ns": min(values), "max_ns": max(values)}


def summarize_window(rows, start, end, requested, peak):
    if end < start or requested < len(rows):
        raise ValueError("invalid window or requested count")
    completed = [row for row in rows if row["status"] == "completed"]
    sequences = [row.get("sequence_no", index) for index, row in enumerate(rows)]
    if any(right <= left for left, right in zip(sequences, sequences[1:])):
        raise ValueError("sample sequence order violates FIFO")
    completions = [row["timestamps_ns"]["deliver_end"] for row in completed]
    if any(right < left for left, right in zip(completions, completions[1:])):
        raise ValueError("completion order violates FIFO")
    warm = [row for row in completed if row.get("sequence_no", 0) > 0]
    names = sorted({key for row in warm for key in row["stages_ns"]})
    elapsed = end - start
    return {"submitted": len(rows), "completed": len(completed),
            "failed": len(rows) - len(completed), "unsubmitted": requested - len(rows),
            "elapsed_ns": elapsed, "window_start_ns": start, "window_end_ns": end,
            "inflight_peak": peak, "throughput_per_second": len(completed) * 1e9 / elapsed if elapsed else 0.,
            "completion_intervals_ns": [b - a for a, b in zip(completions, completions[1:])],
            "first_sample": completed[0] if completed else None,
            "warm_sample_count": len(warm),
            "warm_stages_ns": {name: _statistics([row["stages_ns"][name] for row in warm if name in row["stages_ns"]]) for name in names},
            "statistics_note": "first measured sample separate; warm N-1 retains pipeline fill/drain effects; no reliable P99 claimed"}


def compare_pair(serial, pipeline):
    keys = ("arm_id", "sample_id", "sequence_no", "source_frame", "source_sim_time_ns", "run_id", "route_id", "scenario_id",
            "input_provenance", "artifact_identity", "model", "output_contract", "payload_semantics",
            "decoded_feature_sha256", "wire_payload_sha256")
    for key in keys:
        if serial[key] != pipeline[key]:
            raise ValueError("serial/pipeline pair mismatch: " + key)
    return {"event": "pair_comparison", "arm_id": serial["arm_id"], "sample_id": serial["sample_id"],
            "sequence_no": serial["sequence_no"], "matched": True,
            "decoded_feature_sha256": serial["decoded_feature_sha256"],
            "wire_payload_sha256": serial["wire_payload_sha256"]}


def _timed_serial(item, prepare, infer, finish, deliver, clock):
    stamps = {"submitted": clock()}
    stamps["admitted"] = stamps["prepare_start"] = stamps["submitted"]
    stages = {"credit_wait": 0, "prepare_queue": 0, "infer_queue": 0, "finish_queue": 0, "deliver_queue": 0}
    value, error, previous_end = item, None, stamps["submitted"]
    for name, callback in (("prepare", prepare), ("infer", infer), ("finish", finish), ("deliver", deliver)):
        stamps[name + "_start"] = previous_end
        try:
            value = callback(value)
        except Exception as exception:
            error = str(exception)
        previous_end = stamps[name + "_end"] = clock()
        stages["prepare_service" if name == "prepare" else name] = previous_end - stamps[name + "_start"]
        if error is not None:
            break
    if error is None:
        stages["elapsed"] = stamps["deliver_end"] - stamps["submitted"]
    return stamps, stages, error


def _ready_durations(stamps, stages, offered):
    return dict(stages, source_to_ready=stamps["finish_end"] - stamps["prepare_start"],
                source_to_encoded_result=stamps["finish_end"] - stamps["prepare_start"],
                source_to_wire_ready=stamps["deliver_end"] - stamps["prepare_start"],
                ingress_to_edge_ready=stamps["finish_end"] - stamps["submitted"],
                ingress_to_wire_ready=stamps["deliver_end"] - stamps["submitted"],
                offered_to_wire_ready=stamps["deliver_end"] - offered)


def _window_row(index, stamps, stages, offered, actual, error=None):
    row = {"sequence_no": index, "status": "failed" if error is not None else "completed",
           "timestamps_ns": stamps, "stages_ns": stages, "offered_ns": offered,
           "actual_submit_ns": actual, "submit_lateness_ns": max(0, actual - offered)}
    if error is not None:
        return dict(row, error=str(error))
    return dict(row, stages_ns=_ready_durations(stamps, stages, offered),
                admission_lateness_ns=max(0, stamps["admitted"] - offered))


def _collect_future(index, future, offered, actual):
    error = None
    try:
        future.result()
    except Exception as exception:
        error = exception
    return _window_row(index, dict(future.pipeline_timestamps_ns), dict(future.pipeline_durations_ns), offered, actual, error)


def run_window(items, offsets, mode, prepare, infer, finish, deliver, clock, executor_factory=None):
    """Only source, split engine stages and lightweight delivery in this window."""
    if len(items) != len(offsets) or mode not in ("serial", "pipeline"):
        raise ValueError("invalid window inputs or mode")
    executor = executor_factory(prepare, infer, finish, deliver, capacity=2) if mode == "pipeline" else None
    rows, pending = [], deque()
    start, peak, stop_error = clock(), 0, None

    for index, item in enumerate(items):
        offered = start + offsets[index]
        while clock() < offered:
            time.sleep(max(0, min((offered - clock()) / 1e9, .05)))
        actual = clock()
        try:
            if executor is None:
                stamps, stages, error = _timed_serial(item, prepare, infer, finish, deliver, clock)
                rows.append(_window_row(index, stamps, stages, offered, actual, error))
                peak = 1
                if error is not None:
                    stop_error = error
                    break
            else:
                future = executor.submit(item)
                pending.append((index, future, offered, actual))
                while pending and pending[0][1].done():
                    rows.append(_collect_future(*pending.popleft()))
        except Exception as error:
            if executor is None:
                rows.append({"sequence_no": index, "status": "failed", "error": str(error)})
                peak = 1
            stop_error = str(error)
            break
    while pending:
        rows.append(_collect_future(*pending.popleft()))
    end = clock()
    snapshot = None
    if executor is not None:
        try:
            executor.close()
        except Exception as error:
            stop_error = str(error)
        snapshot = dict(executor.snapshot())
        peak = snapshot["inflight_peak"]
    summary = summarize_window(rows, start, end, len(items), peak)
    summary.update({"executor_snapshot": snapshot, "admission_error": stop_error,
                    "clock_domain": "caller_clock; production requires monotonic_raw"})
    return rows, summary


def _verify_result(args, sample, result, header_bytes, row, validator, limits, runtime):
    clock = runtime.clock.monotonic_raw_ns
    verification_start = clock()
    header, outputs, decode_stages = runtime.protocol.decode_wire_frame(
        header_bytes, result.payload, limits, return_stages=True, readonly=True)
    decoded = clock()
    validator(header["artifact_identity"], outputs, payload_semantics=header["payload_semantics"])
    canonical = _serial.hash_feature_payload(outputs)
    raw_hash = hashlib.sha256(result.payload).hexdigest()
    if canonical != header["payload_sha256"]:
        raise ValueError("independent decoded feature hash mismatch")
    expected = {"run_id": args.run_id, "route_id": row["route_id"], "scenario_id": row["scenario_id"], "sample_id": sample["sample_id"],
                "source_frame": sample["source_frame"], "sequence_no": row["sequence_no"],
                "arm_id": row["arm_id"],
                "source_sim_time_ns": sample["source_frame"] * int(round(args.sim_step_ms * 1000000))}
    for name, value in expected.items():
        if header.get(name) != value or header["trigger_identity"].get(name) != value:
            raise ValueError("result provenance mismatch: " + name)
    end = clock()
    stamps = dict(header["edge_timestamps_ns"])
    stages = dict(row["stages_ns"])
    stages.update({"engine_" + name: value for name, value in header["durations_ns"].items()})
    stages["actual_source_read_to_wire_ready"] = row["timestamps_ns"]["deliver_end"] - stamps["source_read_start"]
    completed = dict(row, event="sample", verified=True, run_id=args.run_id, sample_id=sample["sample_id"], source_frame=sample["source_frame"],
        source_sim_time_ns=expected["source_sim_time_ns"], input_provenance=copy.deepcopy(sample["inputs"]),
        artifact_identity=header["artifact_identity"], model=header["model"], payload_semantics=header["payload_semantics"],
        output_contract={name: {"shape": list(value.shape), "dtype": value.dtype.name} for name, value in outputs.items()},
        decoded_feature_sha256=canonical, wire_payload_sha256=raw_hash, stages_ns=stages,
        edge_timestamps_ns=stamps, payload_bytes=len(result.payload), header_bytes=len(header_bytes),
        verification_start_ns=verification_start, verification_end_ns=end,
        receiver_decode_diagnostic_ns=decoded - verification_start,
        receiver_validate_independent_hash_diagnostic_ns=end - decoded,
        receiver_decode_stages_ns=dict(decode_stages), receiver_diagnostic_scope="same_process_outside_entire_timed_window_not_remote_host",
        first_measured_sample=row["sequence_no"] == 0,
        ready_definition="prepare_start through envelope-serialized deliver_end; excludes all receiver work",
        clock_domain="monotonic_raw")
    if args.wire_dir:
        directory = Path(args.wire_dir)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = runtime.protocol.FRAME_PREFIX.pack(runtime.protocol.MAGIC, len(header_bytes), len(result.payload))
        (directory / ("%s_%s_%04d.wire" % (row["arm_id"], row["mode"], row["sequence_no"]))).write_bytes(prefix + header_bytes + result.payload)
    return completed


def _gpu_timeline_summary(rows, start, end):
    intervals, forward = [], []
    for row in rows:
        stamps = row.get("edge_timestamps_ns", {})
        for stage in ("h2d", "inference", "d2h"):
            if stage + "_start" in stamps and stage + "_end" in stamps:
                interval = (stamps[stage + "_start"], stamps[stage + "_end"])
                if interval[0] < start or interval[1] > end:
                    raise ValueError("GPU timeline is outside RAW measurement window")
                intervals.append(interval)
                if stage == "inference":
                    forward.append(interval)
    busy = interval_union_ns(intervals)
    return {"gpu_stage_timestamp_union_ns": busy, "forward_timestamp_union_ns": interval_union_ns(forward),
            "gpu_stage_idle_inferred_ns": end - start - busy,
            "gpu_idle_scope": "host-observed H2D/forward/D2H timestamp union; inferred gaps, NOT hardware-measured GPU utilization"}


def _initialize_mode(args, source, manifest, arm_id, mode, limits, runtime):
    clock = runtime.clock.monotonic_raw_ns
    initialization_start = clock()
    with redirect_stdout(sys.stderr):
        backend = runtime.backend.CoptV2xHilBackend(arm_id, artifact_root=args.artifact_root,
            config_dir=args.config_dir, native_repo_root=args.native_repo_root,
            rsu_only=True, trust_readonly_inputs=True, checksum_dedup=True, verified_input_source=source)
        engine = backend.load(arm_id)
        pipeline = runtime.pipeline.CoptV2xHilPipeline(backend, run_id=args.run_id,
            route_id=manifest["route_id"], scenario_id=manifest["scenario_id"], compression="none", limits=limits,
            checksum_dedup=True)
    initialization_end = clock()
    warmup_start = clock()
    if args.warmup_count:
        zero_inputs = {name: np.zeros(spec["shape"], dtype=spec["dtype"]) for name, spec in backend.input_contract.items()}
        zero_inputs = backend.validate_inputs(zero_inputs)
        execute = getattr(engine, "infer_validated_rsu", engine.infer_validated)
        with redirect_stdout(sys.stderr):
            for _ in range(args.warmup_count):
                execute(zero_inputs)
        del zero_inputs
    warmup_end = clock()
    validator_start = clock()
    validator = runtime.backend.create_feature_payload_validator(args.config_dir)
    validator_end = clock()
    _emit({"event": "initialization", "arm_id": arm_id, "mode": mode,
           "engine_initialization_ns": initialization_end - initialization_start,
           "warmup_ns": warmup_end - warmup_start, "warmup_count": args.warmup_count,
           "warmup_input": "synthetic_zero_runtime_only_no_source_request_consumed",
           "validator_initialization_ns": validator_end - validator_start,
           "widths": list(backend.spec.widths), "artifact_sha256": dict(backend.spec.artifact_sha256)})
    return pipeline, validator


def _run_mode(args, source, manifest, arm_id, mode, limits, runtime):
    clock = runtime.clock.monotonic_raw_ns
    pipeline, validator = _initialize_mode(args, source, manifest, arm_id, mode, limits, runtime)
    items = []
    for index, sample in enumerate(manifest["samples"]):
        items.append(runtime.trigger.parse_hil_trigger({
            "schema_version": runtime.trigger.HIL_TRIGGER_SCHEMA_VERSION, "run_id": args.run_id,
            "route_id": manifest["route_id"], "scenario_id": manifest["scenario_id"],
            "sample_id": sample["sample_id"], "source_frame": sample["source_frame"], "sequence_no": index,
            "source_sim_time_ns": sample["source_frame"] * int(round(args.sim_step_ms * 1000000)), "arm_id": arm_id}))
    held = {}

    def deliver(result):
        header_bytes = runtime.protocol.prepare_envelope_header(dict(result.header), dict(result.tensor_bundle), limits)
        # Retain wire data only: no second copy of all immutable output arrays.
        held[result.sequence_id] = (SimpleNamespace(payload=result.payload), header_bytes)

    offsets = offered_schedule_ns(manifest["samples"], args.cadence, args.sim_step_ms)
    factory = runtime.executor.BoundedHilPipeline if mode == "pipeline" else None
    with redirect_stdout(sys.stderr):
        rows, summary = run_window(items, offsets, mode, source.frame_for_trigger,
            pipeline.infer_frame, pipeline.encode_inferred_frame, deliver, clock, factory)
    # No decoder, validator, independent hash, sample printing or wire export
    # is reachable until every accepted job and the timed window have finished.
    return _verify_window(args, manifest, arm_id, mode, limits, runtime, rows, summary, held, validator, offsets)


def _verify_window(args, manifest, arm_id, mode, limits, runtime, rows, summary, held, validator, offsets):
    verified = []
    verification_failures = 0
    for row in rows:
        row = dict(row, arm_id=arm_id, mode=mode, cadence=args.cadence,
                   route_id=manifest["route_id"], scenario_id=manifest["scenario_id"])
        if row["status"] == "completed":
            try:
                result, header_bytes = held.pop(row["sequence_no"])
                row = _verify_result(args, manifest["samples"][row["sequence_no"]], result, header_bytes,
                                     row, validator, limits, runtime)
            except Exception as error:
                verification_failures += 1
                row = dict(row, event="sample", verification_error=str(error), verified=False)
        else:
            row = dict(row, event="sample", verified=False)
        verified.append(row)
        _emit(row)
    updated = summarize_window(verified, summary["window_start_ns"], summary["window_end_ns"], args.count, summary["inflight_peak"])
    summary.update(updated)
    summary.update(_gpu_timeline_summary(verified, summary["window_start_ns"], summary["window_end_ns"]))
    summary.update(event="summary", arm_id=arm_id, mode=mode, cadence=args.cadence,
                   verification_failures=verification_failures, verified=len(verified) - summary["failed"] - verification_failures,
                   clock_domain="monotonic_raw", receiver_work_excluded_from_window=True,
                   offered_schedule_offsets_ns=offsets, depth=args.depth)
    _emit(summary)
    return verified, summary


def run(args):
    runtime = load_runtime(args.runtime_root, need_executor=args.mode != "serial")
    if args.config_dir is None:
        args = argparse.Namespace(**dict(vars(args), config_dir=str(runtime.artifacts.DEFAULT_CONFIG_DIR)))
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = verified_prefix(json.loads(manifest_bytes.decode("utf-8")), args.count)
    offered_schedule_ns(manifest["samples"], args.cadence, args.sim_step_ms)
    arms = HIL_ARM_IDS if args.arm == "all" else (args.arm,)
    modes = ("serial", "pipeline") if args.mode == "both" else (args.mode,)
    registry = runtime.artifacts.load_hil_artifact_registry(args.config_dir)
    contract = {tensor.name: {"shape": tuple(tensor.shape), "dtype": tensor.dtype} for tensor in registry[arms[0]].input_contract}
    _emit({"event": "start", "runtime_root": str(Path(args.runtime_root).resolve()), "config_dir": args.config_dir,
           "runtime_module_directory": runtime.closedloop, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
           "arms": list(arms), "modes": list(modes), "count": args.count, "cadence": args.cadence,
           "sim_step_ms": args.sim_step_ms, "depth": args.depth, "warmup_count": args.warmup_count,
           "receiver_diagnostic_scope": "post-window_same-process_only_host_receiver_measured_separately",
           "no_ros": True, "no_network": True, "clock_domain": "monotonic_raw",
           **experimental_acceptance(manifest["samples"], arms, modes)})
    source_start = runtime.clock.monotonic_raw_ns()
    source = runtime.source.IndexedLocalRsuSource(args.data_root, manifest, contract, max_samples=args.count,
        max_total_tensor_bytes=_serial.MAX_PRELOADED_BYTES, **_serial.variant_source_kwargs("optimized"))
    _emit({"event": "source_initialization", "elapsed_ns": runtime.clock.monotonic_raw_ns() - source_start,
           "preloaded_tensor_bytes": source.preloaded_tensor_bytes, "max_preloaded_tensor_bytes": _serial.MAX_PRELOADED_BYTES,
           "sample_count": args.count, "shared_across_arms_and_modes": True})
    limits = runtime.protocol.ProtocolLimits(max_payload_bytes=512 * 1024 ** 2, max_raw_payload_bytes=512 * 1024 ** 2)
    pairs, total_completed, total_failed, verification_failures = 0, 0, 0, 0
    all_ok = True
    for arm_id in arms:
        results = {}
        for mode in modes:
            results[mode], summary = _run_mode(args, source, manifest, arm_id, mode, limits, runtime)
            total_completed += summary["completed"]
            total_failed += summary["failed"]
            verification_failures += summary["verification_failures"]
            all_ok = all_ok and summary["completed"] == args.count and not summary["verification_failures"] and not summary["admission_error"]
            gc.collect()
        if args.mode == "both" and all_ok:
            for serial, pipeline in zip(results["serial"], results["pipeline"]):
                _emit(compare_pair(serial, pipeline))
                pairs += 1
    _emit({"event": "complete", "completed": total_completed, "failed": total_failed,
           "verification_failures": verification_failures, "pairs_compared": pairs,
           "expected_pairs": len(arms) * args.count if args.mode == "both" else 0,
           "pair_hash_validation": "PASS" if args.mode == "both" and all_ok else "not_run_or_failed",
           "success": all_ok, **experimental_acceptance(manifest["samples"], arms, modes, all_ok, pairs)})
    return 0 if all_ok else 1


def _emit(record):
    common = {"schema_version": SCHEMA_VERSION, "is_hil": False, "is_carla": False,
              "measurement_scope": "orin_source_to_edge_ready_static_replay", "communication_measured": False,
              "modeled_communication_ns": 100000000, "wire_compression": "none", "checksum_dedup": True,
              "simulation_time_basis": "actual_manifest_source_frame_times_sim_step"}
    print(json.dumps(dict(common, **record), sort_keys=True, allow_nan=False), flush=True)


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _warmup_count(text):
    value = int(text)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError("warmup count must be between 0 and 100")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "manifest", "artifact-root", "native-repo-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--runtime-root", default=str(ROOT))
    parser.add_argument("--config-dir")
    parser.add_argument("--mode", choices=("serial", "pipeline", "both"), default="both")
    parser.add_argument("--cadence", choices=("saturated", "source"), default="saturated")
    parser.add_argument("--depth", type=int, choices=(2,), default=2)
    parser.add_argument("--arm", choices=HIL_ARM_IDS + ("all",), default="all")
    parser.add_argument("--count", type=_positive_integer, default=20)
    parser.add_argument("--warmup-count", type=_warmup_count, default=1)
    parser.add_argument("--sim-step-ms", type=float, default=50)
    parser.add_argument("--run-id", default="paired-pipeline-static-replay")
    parser.add_argument("--wire-dir", help="export verified raw wire frames after the entire timed window")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        _emit({"event": "error", "error_type": type(error).__name__, "error": str(error)})
        _emit({"event": "complete", "success": False, "completed": None, "failed": None,
               "counts_unknown_due_to_fatal_error": True, "pair_hash_validation": "FAILED",
               "execution_success": False, "success_scope": "execution_only_not_experimental_acceptance",
               "distinct_input_content_count": None, "acceptance_success": False,
               "acceptance_scope": "paired_five_arm_static_replay_per_cadence"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
