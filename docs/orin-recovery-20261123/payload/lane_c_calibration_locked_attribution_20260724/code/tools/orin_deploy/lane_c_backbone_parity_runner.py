#!/usr/bin/env python3
"""Build and measure the Lane C Pyramid multiscale backbone on Jetson Orin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .lane_c_contract import parse_tegrastats
except ImportError:
    from lane_c_contract import parse_tegrastats


EXPECTED_INPUT_TAIL = (2, 64, 128, 256)
EXPECTED_CALIBRATION_FILE_COUNT = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fresh_build_outputs(paths: list[Path | None]) -> None:
    existing = [str(Path(path)) for path in paths if path is not None and Path(path).exists()]
    if existing:
        raise FileExistsError(
            "fresh build requires all output paths to be absent; existing: "
            + ", ".join(existing)
        )


def validate_calibration_manifest(
    *,
    manifest_path: Path,
    calibration_dir: Path,
    expected_shape: tuple[int, ...] | None = EXPECTED_INPUT_TAIL,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    calibration_dir = Path(calibration_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lane_c_shared_calibration_manifest_v1":
        raise ValueError("unsupported calibration manifest schema")
    calibration = manifest.get("calibration", {})
    rows = calibration.get("files", [])
    if calibration.get("file_count") != EXPECTED_CALIBRATION_FILE_COUNT:
        raise ValueError("calibration manifest must declare exactly 15 files")
    if len(rows) != EXPECTED_CALIBRATION_FILE_COUNT:
        raise ValueError("calibration manifest must contain exactly 15 file rows")
    if calibration.get("batch_size") != 2:
        raise ValueError("calibration manifest batch_size must be 2")

    expected_names = [f"batch2_{index:03d}.npy" for index in range(15)]
    declared_names = [str(row.get("filename", "")) for row in rows]
    actual_names = sorted(path.name for path in calibration_dir.glob("batch2_*.npy"))
    if declared_names != expected_names or actual_names != expected_names:
        raise ValueError(
            "calibration payload is not byte-identical: file set/order mismatch"
        )

    verified: list[dict[str, Any]] = []
    for row in rows:
        filename = str(row["filename"])
        if Path(filename).name != filename:
            raise ValueError("calibration manifest filenames must be relative basenames")
        path = calibration_dir / filename
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha = sha256_file(path) if path.is_file() else None
        if actual_size != int(row["bytes"]) or actual_sha != str(row["sha256"]):
            raise ValueError(
                f"calibration payload is not byte-identical: {filename}"
            )
        array = np.load(path, allow_pickle=False)
        if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
            raise ValueError(f"invalid calibration shape in {filename}: {list(array.shape)}")
        if array.dtype != np.dtype(np.float32):
            raise ValueError(f"invalid calibration dtype in {filename}: {array.dtype}")
        verified.append(
            {
                "filename": filename,
                "bytes": actual_size,
                "sha256": actual_sha,
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "c_contiguous": bool(array.flags.c_contiguous),
            }
        )
    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "calibration_dir": str(calibration_dir),
        "verified_file_count": len(verified),
        "files": verified,
    }


def parse_nvml_power_log(raw: str) -> dict[str, Any]:
    samples: list[float] = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            samples.append(float(fields[-1]))
        except ValueError:
            continue
    if not samples:
        raise ValueError("NVML power log contains no board power samples")
    return {
        "source": "H800 NVML board power",
        "physical_quantity": "NVML board power",
        "sample_count": len(samples),
        "board_power_w": samples,
        "board_power_mean_w": float(statistics.mean(samples)),
        "board_power_median_w": float(statistics.median(samples)),
        "rail_semantics": (
            "NVML accelerator board power; not physically equivalent to "
            "Jetson tegrastats rails"
        ),
    }


def builder_flags_for_precision(precision: str) -> list[str]:
    if precision == "fp16":
        return ["fp16"]
    if precision == "int8":
        return ["int8", "fp16_fallback"]
    raise ValueError(f"unsupported precision: {precision}")


def validate_heldout_inputs(inputs: np.ndarray) -> list[int]:
    shape = tuple(int(value) for value in np.asarray(inputs).shape)
    if len(shape) != 5 or shape[1:] != EXPECTED_INPUT_TAIL:
        raise ValueError(
            "expected held-out inputs [N,2,64,128,256], got "
            f"{list(shape)}"
        )
    if shape[0] <= 0:
        raise ValueError("held-out input set must contain at least one batch")
    if not np.isfinite(inputs).all():
        raise ValueError("held-out inputs contain non-finite values")
    return list(shape)


def summarize_latency_samples(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(float(value) for value in samples_ms)
    return {
        "median_ms": float(statistics.median(ordered)),
        "p90_ms": float(ordered[int(0.90 * len(ordered)) - 1]),
        "p99_ms": float(ordered[int(0.99 * len(ordered)) - 1]),
        "mean_ms": float(statistics.mean(ordered)),
    }


def validate_primary_protocol(*, warmup: int, iters: int, repeat: int) -> None:
    expected = {"warmup": 20, "iters": 300, "repeat": 5}
    actual = {"warmup": warmup, "iters": iters, "repeat": repeat}
    if actual != expected:
        raise ValueError(
            "primary protocol must be warmup=20, iters=300, repeat=5; "
            f"got warmup={warmup}, iters={iters}, repeat={repeat}"
        )


def classify_inspector_precisions(inspector: dict[str, Any]) -> dict[str, int]:
    counts = {"int8": 0, "fp16": 0, "fp32": 0, "other": 0}
    for layer in inspector.get("Layers", []):
        text = json.dumps(layer, sort_keys=True).lower()
        if "int8" in text or "i8i8" in text or "imma" in text:
            counts["int8"] += 1
        elif "half" in text or "fp16" in text or "f16f16" in text:
            counts["fp16"] += 1
        elif "float" in text or "fp32" in text or "f32f32" in text:
            counts["fp32"] += 1
        else:
            counts["other"] += 1
    return counts


def _runtime_modules():
    import tensorrt as trt
    import torch

    return trt, torch


def _make_calibrator(
    trt,
    torch,
    calibration_files: list[Path],
    cache_path: Path,
):
    class BackboneEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.files = [Path(path) for path in calibration_files]
            if len(self.files) != EXPECTED_CALIBRATION_FILE_COUNT:
                raise ValueError(
                    f"expected 15 batch2 calibration tensors, got {len(self.files)}"
                )
            self.index = 0
            self.device_tensor = None
            self.consumed_files: list[dict[str, Any]] = []

        def get_batch_size(self):
            return 2

        def get_batch(self, names):
            if self.index >= len(self.files):
                return None
            if len(names) != 1:
                raise ValueError(f"expected one calibration input, got {names}")
            array = np.load(self.files[self.index], allow_pickle=False)
            if tuple(array.shape) != EXPECTED_INPUT_TAIL:
                raise ValueError(
                    f"invalid calibration shape in {self.files[self.index]}: "
                    f"{list(array.shape)}"
                )
            contiguous = np.ascontiguousarray(array, dtype=np.float32)
            self.device_tensor = torch.from_numpy(contiguous).cuda()
            self.index += 1
            self.consumed_files.append(
                {
                    "index": self.index - 1,
                    "filename": self.files[self.index - 1].name,
                    "bytes": self.files[self.index - 1].stat().st_size,
                    "sha256": sha256_file(self.files[self.index - 1]),
                }
            )
            return [int(self.device_tensor.data_ptr())]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(cache)

    return BackboneEntropyCalibrator()


def _deserialize_engine(engine_path: Path):
    trt, _ = _runtime_modules()
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    return trt, engine


def _inspect_engine(trt, engine) -> tuple[dict[str, Any], dict[str, int]]:
    raw = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        parsed = {"Layers": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("unexpected TensorRT inspector output")
    return parsed, classify_inspector_precisions(parsed)


def build_engine(args: argparse.Namespace) -> int:
    build_outputs = [
        args.engine,
        args.inspector_json,
        args.artifact_json,
        args.calibration_cache if args.precision == "int8" else None,
    ]
    validate_fresh_build_outputs(build_outputs)
    calibration_audit = None
    calibration_files: list[Path] = []
    if args.precision == "int8":
        calibration_audit = validate_calibration_manifest(
            manifest_path=args.calibration_manifest,
            calibration_dir=args.calibration_dir,
        )
        calibration_files = [
            args.calibration_dir / row["filename"]
            for row in calibration_audit["files"]
        ]

    trt, torch = _runtime_modules()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + " | ".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 3:
        raise ValueError(
            f"expected one input and three outputs, got "
            f"{network.num_inputs}/{network.num_outputs}"
        )
    input_shape = tuple(int(value) for value in network.get_input(0).shape)
    if input_shape != EXPECTED_INPUT_TAIL:
        raise ValueError(f"unexpected ONNX input shape: {list(input_shape)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3
    if hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    flags = builder_flags_for_precision(args.precision)
    calibrator = None
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Orin TensorRT reports no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calibrator = _make_calibrator(
            trt, torch, calibration_files, args.calibration_cache
        )
        config.int8_calibrator = calibrator

    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.time() - started
    if serialized is None:
        raise RuntimeError("TensorRT build returned no serialized engine")
    if calibrator is not None and len(calibrator.consumed_files) != 15:
        raise RuntimeError(
            "fresh INT8 build did not consume all 15 calibration batches: "
            f"{len(calibrator.consumed_files)}"
        )
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    _, engine = _deserialize_engine(args.engine)
    inspector, precision_counts = _inspect_engine(trt, engine)
    args.inspector_json.parent.mkdir(parents=True, exist_ok=True)
    args.inspector_json.write_text(
        json.dumps(inspector, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "lane_c_backbone_calibration_locked_engine_build_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pyramid_get_multiscale_feature_16x32x64_only",
        "onnx_path": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "input_name": network.get_input(0).name,
        "input_shape": list(input_shape),
        "output_names": [
            network.get_output(index).name for index in range(network.num_outputs)
        ],
        "engine_path": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "precision": args.precision,
        "builder_flags": flags,
        "build_seconds": build_seconds,
        "tensorrt_version": trt.__version__,
        "host": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "cuda_device_name": torch.cuda.get_device_name(),
        },
        "fresh_build": {
            "preexisting_output_count": 0,
            "calibration_cache_read": False,
        },
        "calibration_manifest_sha256": (
            calibration_audit["manifest_sha256"] if calibration_audit else None
        ),
        "calibration_verified_file_count": (
            calibration_audit["verified_file_count"] if calibration_audit else 0
        ),
        "calibration_consumed_file_count": (
            len(calibrator.consumed_files) if calibrator is not None else 0
        ),
        "calibration_consumed_files": (
            calibrator.consumed_files if calibrator is not None else []
        ),
        "calibration_cache_path": (
            str(args.calibration_cache) if args.precision == "int8" else None
        ),
        "calibration_cache_sha256": (
            sha256_file(args.calibration_cache)
            if args.precision == "int8" and args.calibration_cache.is_file()
            else None
        ),
        "inspector_json_path": str(args.inspector_json),
        "inspector_json_sha256": sha256_file(args.inspector_json),
        "layer_precision_counts": precision_counts,
        "execution_provider": "tensorrt_engine",
        "external_fallback_count": 0,
    }
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _torch_dtype(trt, torch, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT tensor dtype: {dtype}")
    return mapping[dtype]


def _prepare_execution(engine, spatial: np.ndarray):
    trt, torch = _runtime_modules()
    context = engine.create_execution_context()
    tensors: dict[str, Any] = {}
    output_names: list[str] = []
    input_count = 0
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_count += 1
            array = np.ascontiguousarray(spatial, dtype=np.float32)
            context.set_input_shape(name, tuple(array.shape))
            tensor = torch.from_numpy(array).cuda()
        else:
            output_names.append(name)
            shape = tuple(int(value) for value in context.get_tensor_shape(name))
            tensor = torch.empty(
                shape,
                dtype=_torch_dtype(trt, torch, engine.get_tensor_dtype(name)),
                device="cuda",
            )
        tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))
    if input_count != 1 or len(output_names) != 3:
        raise ValueError(
            f"expected one engine input and three outputs, got "
            f"{input_count}/{len(output_names)}"
        )
    output_names.sort(
        key=lambda name: -int(np.prod(tuple(tensors[name].shape[-2:])))
    )
    stream = torch.cuda.Stream()

    def execute_once(timed: bool) -> float | None:
        start = torch.cuda.Event(enable_timing=True) if timed else None
        end = torch.cuda.Event(enable_timing=True) if timed else None
        with torch.cuda.stream(stream):
            if start is not None:
                start.record(stream)
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            if end is not None:
                end.record(stream)
        if end is not None:
            end.synchronize()
            return float(start.elapsed_time(end))
        return None

    return execute_once, stream, tensors, output_names


def run_numerical(args: argparse.Namespace) -> int:
    _, engine = _deserialize_engine(args.engine)
    inputs = np.load(args.inputs_npy, allow_pickle=False)
    input_shape = validate_heldout_inputs(inputs)
    outputs: dict[str, list[np.ndarray]] = {}
    for batch in inputs:
        execute, stream, tensors, output_names = _prepare_execution(engine, batch)
        execute(False)
        stream.synchronize()
        for name in output_names:
            outputs.setdefault(name, []).append(
                tensors[name].detach().cpu().float().numpy()
            )
    stacked = {name: np.stack(values) for name, values in outputs.items()}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **stacked)
    report = {
        "schema_version": "lane_c_backbone_orin_numerical_v1",
        "scope": "pyramid_get_multiscale_feature_16x32x64_only",
        "engine_sha256": sha256_file(args.engine),
        "inputs_sha256": sha256_file(args.inputs_npy),
        "input_shape": input_shape,
        "output_npz": str(args.output_npz),
        "output_sha256": sha256_file(args.output_npz),
        "output_shapes": {name: list(value.shape) for name, value in stacked.items()},
        "all_finite": all(bool(np.isfinite(value).all()) for value in stacked.values()),
        "execution_provider": "tensorrt_engine",
        "external_fallback_count": 0,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _scrubbed_child_env() -> dict[str, str]:
    child_env = dict(os.environ)
    child_env.pop("ORIN_SUDO_PW", None)
    return child_env


def _start_tegrastats(log_path: Path):
    password = os.environ.get("ORIN_SUDO_PW", "")
    command = ["tegrastats", "--interval", "50"]
    if password:
        command = ["sudo", "-S", "-p", ""] + command
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if password else subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=_scrubbed_child_env(),
    )
    if password and process.stdin is not None:
        process.stdin.write(password + "\n")
        process.stdin.flush()
    return process, handle, bool(password)


def _start_nvml(log_path: Path, gpu_index: int):
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=timestamp,power.draw",
        "--format=csv,noheader,nounits",
        "--loop-ms=50",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=_scrubbed_child_env(),
    )
    return process, handle


def _stop_process(process, handle) -> None:
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        handle.close()


def _stop_tegrastats(process, handle, *, privileged: bool) -> None:
    cleanup_error = None
    try:
        if privileged:
            password = os.environ.get("ORIN_SUDO_PW", "")
            kwargs = {
                "input": password + "\n",
                "text": True,
                "check": True,
                "timeout": 5,
                "env": _scrubbed_child_env(),
            }
            try:
                subprocess.run(
                    ["sudo", "-S", "-p", "", "tegrastats", "--stop"], **kwargs
                )
            except (OSError, subprocess.SubprocessError):
                try:
                    subprocess.run(
                        ["sudo", "-S", "-p", "", "pkill", "-x", "tegrastats"],
                        **kwargs,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    cleanup_error = error
        else:
            process.terminate()
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        handle.close()
    if cleanup_error is not None:
        raise RuntimeError("failed to stop privileged tegrastats") from cleanup_error


def measure_primary(args: argparse.Namespace) -> int:
    validate_primary_protocol(
        warmup=args.warmup, iters=args.iters, repeat=args.repeat
    )
    _, engine = _deserialize_engine(args.engine)
    inputs = np.load(args.inputs_npy, allow_pickle=False)
    validate_heldout_inputs(inputs)
    execute, stream, _, _ = _prepare_execution(engine, inputs[0])
    for _ in range(args.warmup):
        execute(False)
    stream.synchronize()

    if args.power_source == "tegrastats":
        power_process, power_handle, power_privileged = _start_tegrastats(
            args.power_log
        )
    else:
        power_process, power_handle = _start_nvml(
            args.power_log, args.nvml_gpu_index
        )
        power_privileged = False
    time.sleep(0.25)
    samples: list[float] = []
    per_repeat: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for repeat_index in range(args.repeat):
            repeat_samples = []
            for _ in range(args.iters):
                latency = execute(True)
                if latency is None:
                    raise RuntimeError("timed execution returned no latency")
                repeat_samples.append(latency)
                samples.append(latency)
            per_repeat.append(
                {
                    "repeat_index": repeat_index,
                    "sample_count": len(repeat_samples),
                    **summarize_latency_samples(repeat_samples),
                }
            )
    finally:
        elapsed = time.perf_counter() - started
        if args.power_source == "tegrastats":
            _stop_tegrastats(
                power_process, power_handle, privileged=power_privileged
            )
        else:
            _stop_process(power_process, power_handle)
    raw_power = args.power_log.read_text(encoding="utf-8")
    power = (
        parse_tegrastats(raw_power)
        if args.power_source == "tegrastats"
        else parse_nvml_power_log(raw_power)
    )
    summary = summarize_latency_samples(samples)
    report = {
        "schema_version": "lane_c_backbone_orin_h800_protocol_latency_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pyramid_get_multiscale_feature_engine_compute_no_data_transfer",
        "protocol": {
            "agent_batch": 2,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeat": args.repeat,
            "timing": "CUDA_event",
            "data_transfer_inside_timed_region": False,
        },
        "engine_sha256": sha256_file(args.engine),
        "inputs_sha256": sha256_file(args.inputs_npy),
        "sample_count": len(samples),
        "measurement_wall_seconds": elapsed,
        **summary,
        "per_repeat": per_repeat,
        "power_measurement": {
            "source": (
                "Orin tegrastats rail"
                if args.power_source == "tegrastats"
                else "H800 NVML board power"
            ),
            "rail_semantics": power.get(
                "rail_semantics",
                "Jetson rail telemetry; not physically equivalent to H800 "
                "NVML board power",
            ),
            **power,
        },
        "power_log": str(args.power_log),
        "power_log_sha256": sha256_file(args.power_log),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def validate_output_paths(args: argparse.Namespace) -> None:
    fields = {
        "build": (
            "engine",
            "calibration_cache",
            "inspector_json",
            "artifact_json",
        ),
        "numerical": ("output_npz", "report_json"),
        "measure-primary": ("power_log", "output_json"),
    }
    root = args.artifact_root.resolve()
    for field in fields[args.command]:
        path = getattr(args, field, None)
        if path is None:
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"{field} is outside artifact root {root}: {path}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--artifact-root", type=_path, required=True)
    build.add_argument("--onnx", type=_path, required=True)
    build.add_argument("--engine", type=_path, required=True)
    build.add_argument("--precision", choices=("fp16", "int8"), required=True)
    build.add_argument("--calibration-dir", type=_path)
    build.add_argument("--calibration-manifest", type=_path)
    build.add_argument("--calibration-cache", type=_path)
    build.add_argument("--inspector-json", type=_path, required=True)
    build.add_argument("--artifact-json", type=_path, required=True)
    build.set_defaults(handler=build_engine)

    numerical = commands.add_parser("numerical")
    numerical.add_argument("--artifact-root", type=_path, required=True)
    numerical.add_argument("--engine", type=_path, required=True)
    numerical.add_argument("--inputs-npy", type=_path, required=True)
    numerical.add_argument("--output-npz", type=_path, required=True)
    numerical.add_argument("--report-json", type=_path, required=True)
    numerical.set_defaults(handler=run_numerical)

    measure = commands.add_parser("measure-primary")
    measure.add_argument("--artifact-root", type=_path, required=True)
    measure.add_argument("--engine", type=_path, required=True)
    measure.add_argument("--inputs-npy", type=_path, required=True)
    measure.add_argument("--warmup", type=int, default=20)
    measure.add_argument("--iters", type=int, default=300)
    measure.add_argument("--repeat", type=int, default=5)
    measure.add_argument(
        "--power-log",
        "--tegrastats-log",
        dest="power_log",
        type=_path,
        required=True,
    )
    measure.add_argument(
        "--power-source",
        choices=("tegrastats", "nvml"),
        default="tegrastats",
    )
    measure.add_argument("--nvml-gpu-index", type=int, default=0)
    measure.add_argument("--output-json", type=_path, required=True)
    measure.set_defaults(handler=measure_primary)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_output_paths(args)
    if args.command == "build" and args.precision == "int8":
        if (
            args.calibration_dir is None
            or args.calibration_cache is None
            or args.calibration_manifest is None
        ):
            raise ValueError(
                "INT8 build requires calibration directory, manifest, and cache"
            )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
