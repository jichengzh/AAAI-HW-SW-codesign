#!/usr/bin/env python3
"""Serial Orin static replay diagnostic: real engines, no ROS/CARLA/network.

The modeled 100 ms communication term is metadata only: no sleep, codec, or
network wait is measured. Local totals are elapsed intervals, never stage sums.
"""

import argparse
from contextlib import redirect_stdout
import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HIL_ARM_IDS = ("original_default", "compression_only", "schedule_only",
               "compress_then_tune", "joint_shcosearch")


SCHEMA_VERSION = "coptv2x.serial-static-replay-probe.v1"
MODELED_COMMUNICATION_NS = 100_000_000
MAX_PRELOADED_BYTES = 2 * 1024 ** 3


def load_runtime(runtime_root):
    """Import ONLY the selected runtime tree, after command-line parsing."""
    closedloop = Path(runtime_root).resolve() / "simulation/leaderboard/team_code/closedloop"
    if not closedloop.is_dir():
        raise ValueError("runtime-root has no closedloop module directory")
    module_names = ("coptv2x_hil_artifacts", "coptv2x_hil_backend", "coptv2x_hil_trigger",
                    "ditl_edge_protocol", "ditl_edge_runtime", "ditl_rsu_pipeline", "ditl_rsu_source")
    for name in module_names:
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve().parent != closedloop:
            raise ValueError("runtime module already imported from another tree; use a fresh process")
    sys.path.insert(0, str(closedloop))
    modules = {name: importlib.import_module(name) for name in module_names}
    return SimpleNamespace(
        artifacts=modules["coptv2x_hil_artifacts"], backend=modules["coptv2x_hil_backend"],
        trigger=modules["coptv2x_hil_trigger"], protocol=modules["ditl_edge_protocol"],
        clock=modules["ditl_edge_runtime"], pipeline=modules["ditl_rsu_pipeline"],
        source=modules["ditl_rsu_source"], closedloop=str(closedloop),
    )


def variant_backend_kwargs(variant):
    return {"rsu_only": True, "trust_readonly_inputs": True} if variant == "optimized" else {}


def variant_source_kwargs(variant):
    return {"preload_verified": True, "max_preloaded_tensor_bytes": MAX_PRELOADED_BYTES} \
        if variant == "optimized" else {"memory_map": True, "lazy_load": True}


def hash_feature_payload(outputs):
    """Canonical feature hash, compatible with both legacy and new runtimes."""
    if tuple(sorted(outputs)) != ("res0", "res1", "res2"):
        raise ValueError("feature payload must contain only res0/res1/res2")
    digest = hashlib.sha256()
    for name in ("res0", "res1", "res2"):
        value = np.ascontiguousarray(np.asarray(outputs[name]))
        descriptor = json.dumps({"dtype": value.dtype.name, "name": name, "shape": list(value.shape)},
                                sort_keys=True, separators=(",", ":")).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, byteorder="big"))
        digest.update(descriptor)
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def prefix_manifest(manifest, count):
    """Isolate exactly the requested prefix before any source initialization."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(manifest) is not dict or type(manifest.get("samples")) is not list:
        raise ValueError("manifest must contain a samples list")
    if count > len(manifest["samples"]):
        raise ValueError("count exceeds available manifest samples")
    return copy.deepcopy(dict(manifest, samples=manifest["samples"][:count]))


def summarize_samples(rows):
    warm = rows[1:]
    names = sorted({name for row in warm for name in row["stages_ns"]})
    statistics = {}
    for name in names:
        values = [row["stages_ns"][name] for row in warm if name in row["stages_ns"]]
        statistics[name] = {
            "count": len(values), "mean_ns": float(np.mean(values)),
            "p50_ns": float(np.percentile(values, 50)),
            "p95_ns": float(np.percentile(values, 95)),
            "min_ns": int(min(values)), "max_ns": int(max(values)),
        }
    return {"cold_sample": rows[0] if rows else None,
            "warm_sample_count": len(warm), "warm_stages_ns": statistics}


def rsu_projection_hash(outputs, variant):
    if variant not in ("baseline", "optimized"):
        raise ValueError("unknown probe variant")
    batch = 2 if variant == "baseline" else 1
    if any(np.asarray(value).ndim != 4 or value.shape[0] != batch for value in outputs.values()):
        raise ValueError("decoded output batch shape violates probe variant")
    projected = {name: value[1:2] if variant == "baseline" else value
                 for name, value in outputs.items()}
    return hash_feature_payload(projected)


def compare_pair(baseline, optimized):
    keys = ("arm_id", "sample_id", "sequence_no", "source_frame", "input_provenance",
            "artifact_identity", "rsu_projection_sha256")
    for name in keys:
        if baseline[name] != optimized[name]:
            raise ValueError("baseline/optimized pair mismatch: %s" % name)
    return {"event": "pair_comparison", "arm_id": baseline["arm_id"],
            "sample_id": baseline["sample_id"], "sequence_no": baseline["sequence_no"],
            "matched": True, "rsu_projection_sha256": baseline["rsu_projection_sha256"]}


def _emit(record):
    common = {"schema_version": SCHEMA_VERSION, "is_hil": False,
              "measurement_scope": "static_replay_same_host_process_cpu_diagnostic",
              "simulation_time_basis": "synthetic_sequence_20hz_not_carla",
              "modeled_communication_ns": MODELED_COMMUNICATION_NS,
              "communication_measured": False, "wire_compression": "none"}
    print(json.dumps(dict(common, **record), sort_keys=True, allow_nan=False), flush=True)


def _source(args, manifest, input_contract, variant, runtime):
    start = runtime.clock.monotonic_raw_ns()
    source = runtime.source.IndexedLocalRsuSource(
        args.data_root, manifest, input_contract,
        max_samples=len(manifest["samples"]), max_total_tensor_bytes=MAX_PRELOADED_BYTES,
        **variant_source_kwargs(variant),
    )
    elapsed = runtime.clock.monotonic_raw_ns() - start
    _emit({"event": "source_initialization", "variant": variant,
           "source_init_ns": elapsed, "preloaded_tensor_bytes": getattr(source, "preloaded_tensor_bytes", 0),
           "sample_count": len(manifest["samples"]), "shared_across_arms": True})
    return source


def _sample(args, source, pipeline, sample, sequence_no, arm_id, variant, limits, runtime, validator=None):
    clock = runtime.clock.monotonic_raw_ns
    trigger = runtime.trigger.parse_hil_trigger({
        "schema_version": runtime.trigger.HIL_TRIGGER_SCHEMA_VERSION, "run_id": args.run_id,
        "route_id": source.route_id, "scenario_id": source.scenario_id,
        "sample_id": sample["sample_id"], "source_frame": sample["source_frame"],
        "sequence_no": sequence_no, "source_sim_time_ns": sequence_no * 50_000_000,
        "arm_id": arm_id,
    })
    frame = source.frame_for_trigger(trigger)
    with redirect_stdout(sys.stderr):
        result = pipeline.process(frame)
    encoded_result_ready = clock()
    serialize_start = clock()
    header_bytes = runtime.protocol.prepare_envelope_header(dict(result.header), dict(result.tensor_bundle), limits)
    serialize_end = clock()
    decode_start = clock()
    decode_kwargs = {"readonly": True} if variant == "optimized" else {}
    header, outputs, decode_stages = runtime.protocol.decode_wire_frame(
        header_bytes, result.payload, limits, return_stages=True, **decode_kwargs,
    )
    decode_end = clock()
    validate_start = clock()
    validation_kwargs = {"payload_semantics": header["payload_semantics"]} if variant == "optimized" else {}
    if validator is None:
        runtime.backend.validate_feature_payload_contract(
            header["artifact_identity"], outputs, config_dir=args.config_dir,
            **validation_kwargs,
        )
    else:
        validator(header["artifact_identity"], outputs, **validation_kwargs)
    verified_hash = None
    if variant == "optimized" and getattr(args, "checksum_dedup", False):
        verified_hash = runtime.protocol.verified_feature_payload_sha256(outputs)
    decoded_hash = verified_hash if verified_hash is not None else hash_feature_payload(outputs)
    if decoded_hash != header["payload_sha256"]:
        raise ValueError("decoded feature payload hash does not match engine snapshot")
    validate_end = clock()
    stages = dict(header["durations_ns"])
    stages.update({"protocol_serialize_actual": serialize_end - serialize_start,
                   "source_through_encoded_result_actual": encoded_result_ready - frame.source_metadata["timestamps_ns"]["source_read_start"],
                   "decode_actual": decode_end - decode_start,
                   "feature_validate_and_hash_actual": validate_end - validate_start,
                   "local_total_actual": validate_end - frame.source_metadata["timestamps_ns"]["source_read_start"]})
    stages.update({"decode_" + name: int(value) for name, value in decode_stages.items()
                   if name.endswith("_ns") and type(value) is int})
    projection_start = clock()
    projected_hash = rsu_projection_hash(outputs, variant)
    projection_end = clock()
    # Optional raw ROS smoke artifact is written after all measured intervals.
    last_arm = HIL_ARM_IDS[-1] if args.arm == "all" else args.arm
    last_variant = "optimized" if args.variant == "both" else args.variant
    if args.last_wire and sequence_no == args.count - 1 and arm_id == last_arm and variant == last_variant:
        prefix = runtime.protocol.FRAME_PREFIX.pack(runtime.protocol.MAGIC, len(header_bytes), len(result.payload))
        Path(args.last_wire).write_bytes(prefix + header_bytes + result.payload)
    if getattr(args, "arm_wire_dir", None) and sequence_no == args.count - 1:
        export_dir = Path(args.arm_wire_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        prefix = runtime.protocol.FRAME_PREFIX.pack(runtime.protocol.MAGIC, len(header_bytes), len(result.payload))
        (export_dir / (arm_id + "_" + variant + ".wire")).write_bytes(prefix + header_bytes + result.payload)
    return {
        "event": "sample", "variant": variant, "arm_id": arm_id,
        "sample_id": sample["sample_id"], "source_frame": sample["source_frame"],
        "sequence_no": sequence_no, "cold_first_frame": sequence_no == 0,
        "input_provenance": copy.deepcopy(sample["inputs"]),
        "artifact_identity": header["artifact_identity"], "model": header["model"],
        "payload_semantics": header["payload_semantics"], "decoded_feature_sha256": decoded_hash,
        "verified_checksum_reused": verified_hash is not None,
        "tensor_bundle_sha256_scope": header["tensor_bundle"].get("sha256_scope", "raw_payload"),
        "rsu_projection_sha256": projected_hash, "wire_payload_sha256": hashlib.sha256(result.payload).hexdigest(),
        "output_contract": {name: {"shape": list(value.shape), "dtype": value.dtype.name,
                                   "readonly": not value.flags.writeable} for name, value in outputs.items()},
        "payload_bytes": len(result.payload), "header_bytes": len(header_bytes),
        "remote_input_tensor_bytes": header["remote_input_tensor_bytes"],
        "stages_ns": stages, "timestamps_ns": dict(header["edge_timestamps_ns"],
            decode_actual_start=decode_start, decode_actual_end=decode_end,
            validation_actual_end=validate_end),
        "comparison_hash_diagnostic_ns": projection_end - projection_start,
        "comparison_hash_excluded_from_local_total": True,
        "total_definition": "elapsed source_read_start through decode and feature validation/hash; stage durations overlap",
    }


def _run_variant(args, source, manifest, arm_id, variant, limits, runtime):
    start = runtime.clock.monotonic_raw_ns()
    backend_kwargs = variant_backend_kwargs(variant)
    if variant == "optimized":
        backend_kwargs = dict(backend_kwargs, verified_input_source=source)
        if getattr(args, "checksum_dedup", False):
            backend_kwargs["checksum_dedup"] = True
    with redirect_stdout(sys.stderr):
        backend = runtime.backend.CoptV2xHilBackend(
            arm_id, artifact_root=args.artifact_root, config_dir=args.config_dir,
            native_repo_root=args.native_repo_root, **backend_kwargs,
        )
        backend.load(arm_id)
    engine_init_ns = runtime.clock.monotonic_raw_ns() - start
    _emit({"event": "engine_initialization", "variant": variant, "arm_id": arm_id,
           "engine_init_ns": engine_init_ns, "widths": list(backend.spec.widths),
           "artifact_sha256": dict(backend.spec.artifact_sha256), "runtime": backend.spec.runtime})
    pipeline_options = {"checksum_dedup": True} if variant == "optimized" and getattr(args, "checksum_dedup", False) else {}
    pipeline = runtime.pipeline.CoptV2xHilPipeline(
        backend, run_id=args.run_id, route_id=manifest["route_id"],
        scenario_id=manifest["scenario_id"], compression="none", limits=limits, **pipeline_options,
    )
    validator = None
    if variant == "optimized":
        validation_init_start = runtime.clock.monotonic_raw_ns()
        validator = runtime.backend.create_feature_payload_validator(args.config_dir)
        _emit({"event": "validator_initialization", "variant": variant, "arm_id": arm_id,
               "validator_init_ns": runtime.clock.monotonic_raw_ns() - validation_init_start})
    rows = []
    for index, sample in enumerate(manifest["samples"]):
        row = _sample(args, source, pipeline, sample, index, arm_id, variant, limits, runtime, validator)
        _emit(row)
        rows.append(row)
    _emit(dict(summarize_samples(rows), event="variant_summary", arm_id=arm_id, variant=variant,
               cold_definition="first inference of newly loaded runtime; no cache flush"))
    return rows


def run(args):
    runtime = load_runtime(args.runtime_root)
    if args.config_dir is None:
        args = argparse.Namespace(**dict(vars(args), config_dir=str(runtime.artifacts.DEFAULT_CONFIG_DIR)))
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = prefix_manifest(json.loads(manifest_bytes.decode("utf-8")), args.count)
    registry = runtime.artifacts.load_hil_artifact_registry(args.config_dir)
    arms = HIL_ARM_IDS if args.arm == "all" else (args.arm,)
    variants = ("baseline", "optimized") if args.variant == "both" else (args.variant,)
    contract = {tensor.name: {"shape": tuple(tensor.shape), "dtype": tensor.dtype}
                for tensor in registry[arms[0]].input_contract}
    _emit({"event": "probe_start", "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
           "runtime_root": str(Path(args.runtime_root).resolve()), "runtime_module_directory": runtime.closedloop,
           "config_dir": args.config_dir,
           "comparison_scope": "same_runtime_local_flags_only" if args.variant == "both" else "separate_runtime_process",
           "sample_count": args.count, "arms": list(arms), "variants": list(variants),
           "hostname": platform.node(), "platform": platform.platform(),
           "python": platform.python_version(), "numpy": np.__version__,
           "no_ros": True, "no_carla": True, "no_network": True,
           "serial_execution": True, "max_preloaded_tensor_bytes": MAX_PRELOADED_BYTES})
    sources = {variant: _source(args, manifest, contract, variant, runtime) for variant in variants}
    limits = runtime.protocol.ProtocolLimits(max_payload_bytes=512 * 1024 ** 2, max_raw_payload_bytes=512 * 1024 ** 2)
    for arm_id in arms:
        rows = {}
        for variant in variants:
            rows[variant] = _run_variant(args, sources[variant], manifest, arm_id, variant, limits, runtime)
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.variant == "both":
            for baseline, optimized in zip(rows["baseline"], rows["optimized"]):
                _emit(compare_pair(baseline, optimized))
    _emit({"event": "probe_complete", "pair_hash_validation": "PASS" if args.variant == "both" else "not_run"})
    return 0


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--native-repo-root", required=True)
    parser.add_argument("--runtime-root", default=str(ROOT), help="module tree to import; legacy baseline must use old tree")
    parser.add_argument("--config-dir", help="default: selected runtime's frozen configs")
    parser.add_argument("--variant", choices=("baseline", "optimized", "both"), default="both")
    parser.add_argument("--arm", choices=tuple(HIL_ARM_IDS) + ("all",), default="all")
    parser.add_argument("--count", type=_positive_integer, default=20)
    parser.add_argument("--run-id", default="serial-static-replay-probe")
    parser.add_argument("--last-wire", help="save final sample raw frame after the measurement window")
    parser.add_argument("--arm-wire-dir", help="export each arm's final raw frame after its measurement window")
    parser.add_argument("--checksum-dedup", action="store_true",
                        help="optimized path uses one canonical feature checksum per endpoint; legacy default unchanged")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        _emit({"event": "probe_error", "error_type": type(error).__name__, "error": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
