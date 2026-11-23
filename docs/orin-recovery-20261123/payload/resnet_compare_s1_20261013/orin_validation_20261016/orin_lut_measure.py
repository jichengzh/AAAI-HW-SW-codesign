"""Measure only requested MDP RN50 Conv2d LUT keys on the actual Orin.

The pinned MDP checkout documents its LUT input format but ships no generator.
This task-local collector uses PyTorch/cuDNN FP32 batch-one convolution and
CUDA events; it does not infer any entry from a different GPU or precision.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import time


REMOTE_ROOT = Path("/home/jichengzhi/resnet_compare_s1_20261013")
LOCK = REMOTE_ROOT / ".orin-runtime.lock"


def parse_key(key):
    parts = key.split("_") if isinstance(key, str) else []
    if len(parts) != 6 or any(not part.isdecimal() for part in parts):
        raise ValueError(f"Invalid ordinary-convolution LUT key: {key!r}")
    batch, cin, cout, fmap, kernel, stride = map(int, parts)
    if batch != 1 or min(cin, cout, fmap) < 1 or kernel not in (1, 3, 7) or stride not in (1, 2):
        raise ValueError(f"Unsupported Orin RN50 LUT key: {key!r}")
    if (kernel == 7 and (cin != 3 or fmap != 224 or stride != 2)):
        raise ValueError(f"Unexpected RN50 stem shape: {key!r}")
    return batch, cin, cout, fmap, kernel, stride


def validate_request(request):
    if (not isinstance(request, dict) or request.get("batch_size") != 1
            or request.get("precision") != "float32" or request.get("latency_unit") != "seconds"):
        raise ValueError("FP32 batch-one seconds request required")
    keys = request.get("keys")
    if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
        raise ValueError("Nonempty string LUT key list required")
    if len(keys) != len(set(keys)):
        raise ValueError("LUT request has duplicate keys")
    if request.get("all_key_count") != len(keys):
        raise ValueError("LUT request count differs")
    for key in keys:
        parse_key(key)
    return keys


def select_keys(keys, limit):
    if type(limit) is not int or limit < 0:
        raise ValueError("Pilot key limit must be nonnegative")
    if limit == 0 or limit >= len(keys):
        return list(keys)
    if limit == 1:
        return [keys[0]]
    return [keys[round(i * (len(keys) - 1) / (limit - 1))] for i in range(limit)]


def summarize_seconds(times_ms):
    if not times_ms or any(not math.isfinite(value) or value <= 0 for value in times_ms):
        raise ValueError("CUDA timing samples must be positive finite milliseconds")
    return statistics.median(times_ms) / 1000.0


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measure_key(key, repeats, warmup, torch):
    import torch.nn.functional as functional

    batch, cin, cout, fmap, kernel, stride = parse_key(key)
    x = torch.zeros((batch, cin, fmap, fmap), dtype=torch.float32, device="cuda:0")
    weight = torch.zeros((cout, cin, kernel, kernel), dtype=torch.float32, device="cuda:0")
    padding = kernel // 2  # torchvision RN50 V1: 7x7/3x3 same, 1x1 no padding.
    with torch.no_grad():
        for _ in range(warmup):
            functional.conv2d(x, weight, stride=stride, padding=padding)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            start.record()
            functional.conv2d(x, weight, stride=stride, padding=padding)
            end.record()
        torch.cuda.synchronize()
    times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return summarize_seconds(times_ms), times_ms


def run(request_path, output_root, limit, repeats, warmup):
    request_path, output_root = Path(request_path).resolve(), Path(output_root).resolve()
    if REMOTE_ROOT not in request_path.parents or REMOTE_ROOT not in output_root.parents:
        raise ValueError("Remote request/output must stay under approved experiment root")
    if type(repeats) is not int or repeats < 3 or type(warmup) is not int or warmup < 1:
        raise ValueError("At least three repeats and one warmup required")
    request = json.loads(request_path.read_text())
    keys = select_keys(validate_request(request), limit)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if output_root.exists():
                raise FileExistsError("LUT evidence root already exists; preserving it")
            output_root.mkdir(parents=True)
            import torch

            platform = Path("/proc/device-tree/model").read_text().strip("\x00\n")
            capability = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            if capability != (8, 7) or "orin" not in platform.lower() or "orin" not in name.lower():
                raise ValueError("Actual Jetson Orin sm87 required")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            setup_done = time.perf_counter()
            table, failures = {}, []
            with (output_root / "timings.jsonl").open("x") as log:
                for index, key in enumerate(keys):
                    started = time.perf_counter()
                    try:
                        seconds, times_ms = measure_key(key, repeats, warmup, torch)
                        table[key] = seconds
                        row = {"index": index, "key": key, "latency_seconds": seconds,
                               "raw_cuda_event_ms": times_ms,
                               "wall_seconds": time.perf_counter() - started}
                    except (RuntimeError, ValueError) as exc:
                        row = {"index": index, "key": key, "error": repr(exc),
                               "wall_seconds": time.perf_counter() - started}
                        failures.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
            summary = {"status": "COMPLETE" if not failures else "PARTIAL",
                       "requested_key_count": len(keys), "measured_key_count": len(table),
                       "failed_key_count": len(failures), "failed_keys": [row["key"] for row in failures],
                       "request_sha256": file_sha(request_path), "target": platform,
                       "hardware_id": name, "capability": list(capability),
                       "torch_version": torch.__version__, "cudnn_version": torch.backends.cudnn.version(),
                       "batch_size": 1, "precision": "float32", "tf32_enabled": False,
                       "latency_unit": "seconds", "warmup_per_key": warmup,
                       "repeats_per_key": repeats,
                       "measurement_boundary": "single Conv2d forward, CUDA event start/end, synchronized; excludes tensor allocation and warmup",
                       "evidence_scope": "real_orin_layer_measurement",
                       "measurement_wall_seconds": time.perf_counter() - setup_done,
                       "timings_sha256": file_sha(output_root / "timings.jsonl")}
            if not failures:
                lut = output_root / "orin_fp32_batch1_lut.pkl"
                with lut.open("xb") as stream:
                    pickle.dump(table, stream, protocol=4)
                summary["lut_sha256"] = file_sha(lut)
            with (output_root / "manifest.json").open("x") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            return summary
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    options = parser.parse_args()
    print(json.dumps(run(options.request, options.output, options.limit, options.repeats, options.warmup),
                     allow_nan=False))
