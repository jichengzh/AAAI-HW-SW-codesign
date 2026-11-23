"""Completed-request ResNet S1 runtime measurement (CPU diagnostics or Orin)."""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .contracts import validate_result


def _platform_name():
    model = Path("/proc/device-tree/model")
    return model.read_text().strip("\x00\n") if model.exists() else platform.platform()


class TorchRuntime:
    """Synchronous logits adapter; initialization and model loading are excluded.

    No random model fallback. TF32 is disabled for D0 FP32 numerical comparison.
    CUDA event completion includes copies; L1 is forward only. CPU diagnostics
    never produce a CUDA-event claim or an Orin target identity.
    """
    backend = "pytorch"
    target_graph_verified = True

    def __init__(self, model, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.hardware = {"name": "cpu", "capability": None, "platform": _platform_name()}
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.hardware = {"name": torch.cuda.get_device_name(self.device),
                             "capability": list(torch.cuda.get_device_capability(self.device)),
                             "platform": _platform_name()}
        self.layer_precision = [{"name": "pytorch_parameters", "precision": str(next(
            iter(self.model.parameters()), torch.empty(0)).dtype).replace("torch.", "").upper()}]
        # Expose conventional labels rather than dtype spelling.
        self.layer_precision = [{**p, "precision": {"FLOAT32": "FP32", "FLOAT16": "FP16"}.get(
            p["precision"], p["precision"])} for p in self.layer_precision]

    def infer(self, tensor):
        start = time.perf_counter()
        stages = {}
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    events[0].record()
                    data = tensor.to(self.device, dtype=getattr(self, "input_dtype", tensor.dtype))
                    events[1].record()
                    output = self.model(data)
                    events[2].record()
                    logits = output.to("cpu")
                    events[3].record()
                    wait_start = time.perf_counter()
                    events[3].synchronize()
                    stages = {"h2d": events[0].elapsed_time(events[1]),
                              "forward": events[1].elapsed_time(events[2]),
                              "d2h": events[2].elapsed_time(events[3]),
                              "wait": (time.perf_counter() - wait_start) * 1000}
                    completion = "cuda_event_synchronized"
                    l1 = stages["forward"]
            else:
                forward_start = time.perf_counter()
                logits = self.model(tensor.to(self.device, dtype=getattr(self, "input_dtype", tensor.dtype))).detach().cpu()
                stages = {"h2d": None, "forward": (time.perf_counter() - forward_start) * 1000,
                          "d2h": None, "wait": None}
                completion, l1 = "cpu_return", None
        return {"logits": logits, "completion": completion, "stages_ms": stages,
                "l1_device_ms": l1, "source_to_done_ms": (time.perf_counter() - start) * 1000}


def _base(candidate):
    identities = ("run_id", "method_id", "seed", "candidate_id", "source_commit",
                  "initial_weight_sha256", "structure_digest", "precision_semantics",
                  "sample_manifest_sha256")
    return {"schema_version": "resnet_compare_result_v1",
            **{key: candidate.get(key) for key in identities}, "actual_backend": None,
            "hardware_id": None, "top1_pp": None, "top5_pp": None,
            "accuracy_sample_count": 0, "accuracy_phase": "S1_smoke_not_formal",
            "latency_ms": None, "energy_j": None,
            "energy_reason": "power unavailable; no calibrated synchronized energy integration",
            "measurement_boundary": None, "budget_used": {}, "status": "PARTIAL",
            "reason": "Not measured", "evidence_paths": [],
            "target_graph_verified": False, "numerical_verified": False}


def _check_target(adapter, target):
    hardware = adapter.hardware
    if target == "orin":
        if hardware.get("capability") != [8, 7] or "orin" not in hardware.get(
                "name", "").lower() or "orin" not in hardware.get("platform", "").lower():
            raise ValueError("Target identity must be actual Jetson Orin sm87, not source sm89/sm90")
    elif target != "cpu" or hardware.get("name") != "cpu":
        raise ValueError("Unsupported target; CPU diagnostics or actual Orin required")


def _numpy(logits):
    value = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    if value.shape != (1, 1000) or not np.isfinite(value).all():
        raise ValueError("Invalid logits: expected finite [1,1000], no NaN/shape mismatch")
    return value


def _observation(adapter, tensor, target):
    obs = adapter.infer(tensor)
    expected = "cuda_event_synchronized" if target == "orin" else "cpu_return"
    if obs.get("completion") != expected:
        raise ValueError("Missing real completion synchronization; enqueue is not latency")
    logits = _numpy(obs["logits"])
    l1 = obs.get("l1_device_ms")
    if target == "orin" and (l1 is None or not math.isfinite(l1) or l1 <= 0):
        raise ValueError("Invalid CUDA completion/l1_device_ms: finite positive time required")
    for stage, duration in obs.get("stages_ms", {}).items():
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            raise ValueError(f"Invalid stage {stage}: finite nonnegative duration or explicit null required")
    if not math.isfinite(obs["source_to_done_ms"]) or obs["source_to_done_ms"] <= 0:
        raise ValueError("Invalid source-to-done completion latency")
    return obs, logits


def _adapter(candidate, profile):
    if profile.get("adapter") is not None:
        return profile["adapter"]
    if profile.get("backend", "pytorch") == "pytorch":
        if "model" not in candidate:
            raise ValueError("Explicit loaded source/candidate model required; no random fallback")
        return TorchRuntime(candidate["model"], profile.get("device", "cuda:0"))
    if profile["backend"] == "tensorrt":
        from .trt_runtime import TensorRTRuntime
        return TensorRTRuntime(Path(candidate["engine_path"]))
    raise ValueError("Unsupported runtime backend; no silent substitution")


def _numeric(outputs, samples, precision):
    if any("source_logits" not in sample for sample in samples):
        return {"verified": False, "reason": "Source logits unavailable", "comparisons": []}
    tolerance = {"FP32": (1e-4, 1e-3), "FP16": (1e-2, 1e-2)}.get(precision)
    comparisons = []
    for output, sample in zip(outputs, samples):
        source = _numpy(sample["source_logits"])
        comparisons.append({"sample_id": sample["sample_id"],
                            "max_abs_error": float(np.max(np.abs(output - source))),
                            "prediction_agreement": bool(output.argmax() == source.argmax()),
                            "allclose": bool(np.allclose(output, source, atol=tolerance[0],
                                                         rtol=tolerance[1])) if tolerance else None})
    # INT8 is never accepted by a floating-point allclose heuristic.
    return {"verified": bool(tolerance) and all(c["allclose"] for c in comparisons),
            "tolerance": {"atol": tolerance[0], "rtol": tolerance[1]} if tolerance else None,
            "reason": "Fixed tolerance; never automatically relaxed" if tolerance else "INT8 needs separate accuracy acceptance",
            "comparisons": comparisons}


def _measure(adapter, samples, profile, timings):
    target = profile["target"]
    warmup, windows, requests = (profile.get(k, default) for k, default in (
        ("warmup", 50), ("windows", 3), ("requests_per_window", 200)))
    if any(type(n) is not int or n < 1 for n in (warmup, windows, requests)):
        raise ValueError("Positive integer warmup/windows/requests required")
    for index in range(warmup):
        _observation(adapter, samples[index % len(samples)]["input"], target)
    outputs = [_observation(adapter, s["input"], target)[1] for s in samples]
    for window in range(windows):
        for index in range(requests):
            sample = samples[index % len(samples)]
            obs, _ = _observation(adapter, sample["input"], target)
            timings.append({key: value for key, value in obs.items() if key != "logits"})
            timings[-1].update(window=window, request=index, sample_id=sample["sample_id"],
                               preprocess_ms=None, postprocess_ms=None,
                               excluded_stages_reason="Input is preprocessed; accuracy postprocessing is outside logits service")
    return outputs, {"warmup_requests": warmup, "windows": windows,
                     "requests_per_window": requests, "completed_requests": len(timings),
                     "numerical_requests": len(samples)}


def _finish(result, numeric, timings, layers, root):
    paths = [root / name for name in ("baseline_runtime.json", "timings.jsonl", "layer_precision.json", "numerical.json")]
    result = {**result, "evidence_paths": [str(p) for p in paths]}
    result = validate_result(result)
    paths[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    paths[1].write_text("".join(json.dumps(t, allow_nan=False) + "\n" for t in timings))
    paths[2].write_text(json.dumps(layers, indent=2, allow_nan=False) + "\n")
    paths[3].write_text(json.dumps(numeric, indent=2, allow_nan=False) + "\n")
    return result


def measure_candidate(candidate: dict, runtime_profile: dict, samples: Sequence[dict], output_root: Path) -> dict:
    """Measure sequential completed requests; write only to a fresh evidence dir.

    Samples contain sample_id, preprocessed CPU input [1,3,224,224], label and
    source_logits. Real sample manifest identity belongs to Task2. Synthetic
    structural probes are explicitly marked is_synthetic=True and never READY.
    No power/energy estimate is inferred from missing sensors or latency.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation also prevents two processes accidentally sharing a run.
    with (root / ".measurement-reserved").open("x"):
        pass
    result, timings, layers, numeric = _base(candidate), [], [], {}
    if not samples:
        result.update(precondition="NEEDS_DATA", reason="NEEDS_DATA: no real ImageNet samples supplied")
        return _finish(result, numeric, timings, layers, root)
    if runtime_profile.get("precision") == "INT8" and not runtime_profile.get("calibration_verified"):
        result.update(precondition="NEEDS_CALIBRATION", reason="D3 unexecuted: fixed 1024-image calibration manifest unavailable")
        return _finish(result, numeric, timings, layers, root)
    try:
        for sample in samples:
            if sample["input"].shape != (1, 3, 224, 224) or not torch.isfinite(sample["input"]).all():
                raise ValueError("Invalid sample input shape/nonfinite values")
        adapter = _adapter(candidate, runtime_profile)
        _check_target(adapter, runtime_profile["target"])
        layers = adapter.layer_precision
        precision = runtime_profile.get("precision", "FP32")
        actual = sorted({p["precision"] for p in layers if p.get("precision")})
        result.update(actual_backend=adapter.backend, hardware_id=adapter.hardware["name"],
                      hardware=adapter.hardware, requested_precision=precision, actual_precision=actual,
                      precision_fallback=any(p != precision for p in actual),
                      target_graph_verified=adapter.target_graph_verified,
                      tf32_enabled=False if adapter.backend == "pytorch" else None,
                      measurement_boundary="preprocessed-CPU-input/source-to-CPU-logits-done; CUDA L1=forward-only"
                      if runtime_profile["target"] == "orin" else "CPU-service/source-to-done; not CUDA L1")
        if precision == "INT8" and not any("INT8" in p.split("+") for p in actual):
            result.update(status="UNSUPPORTED", reason="INT8 requested but inspected target graph has no INT8 execution evidence")
            return _finish(result, numeric, timings, layers, root)
        outputs, budget = _measure(adapter, samples, runtime_profile, timings)
        numeric = _numeric(outputs, samples, precision)
        result.update(numerical_verified=numeric["verified"], budget_used=budget)
        if numeric["comparisons"] and precision != "INT8" and not numeric["verified"]:
            raise ValueError("Numerical acceptance failed at fixed tolerance; diagnose, do not relax")
        values = [row["source_to_done_ms"] for row in timings]
        summary = {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
                   "p95": float(np.percentile(values, 95))}
        result.update(latency_ms=summary["mean"], latency_summary_ms=summary,
                      reason="S1 smoke completed; not formal P99/energy/accuracy evidence")
        synthetic = any(s.get("is_synthetic", False) for s in samples)
        if synthetic:
            result.update(accuracy_phase="synthetic_structural_only", precondition="NEEDS_DATA",
                          reason="Synthetic structural probe ONLY, NOT ImageNet or formal accuracy; NEEDS_DATA")
        else:
            labels = [s.get("label") for s in samples]
            if any(type(label) is not int or not 0 <= label < 1000 for label in labels):
                raise ValueError("Real samples require integer class labels 0..999")
            result.update(top1_pp=100 * sum(int(o.argmax()) == y for o, y in zip(outputs, labels)) / len(labels),
                          top5_pp=100 * sum(y in o[0].argsort()[-5:] for o, y in zip(outputs, labels)) / len(labels),
                          accuracy_sample_count=len(labels))
            enough = budget["warmup_requests"] >= 50 and budget["windows"] >= 3 and budget["requests_per_window"] >= 200
            precision_known = bool(actual) and "UNKNOWN" not in actual
            if not precision_known:
                result["reason"] = "Actual layer precision is unknown; cannot accept target READY"
            if enough and runtime_profile["target"] == "orin" and numeric["verified"] and adapter.target_graph_verified and precision_known:
                result["status"] = "READY"
    except (ValueError, RuntimeError, KeyError, TypeError, ImportError, OSError) as exc:
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}", latency_ms=None,
                      top1_pp=None, top5_pp=None, accuracy_sample_count=0)
    return _finish(result, numeric, timings, layers, root)
