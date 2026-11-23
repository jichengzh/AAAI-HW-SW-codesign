"""Bounded same-ONNX/same-sample FP16 TensorRT diagnosis on Orin only.

The existing engine is read-only. One additional engine is built only under the
approved preparation directory. This is synthetic numerical diagnosis, not
ImageNet acceptance, calibration, or a precision policy change.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/jichengzhi/resnet_compare_s1_20261013")
JOB = ROOT / "s1_03_transport/849be9c2a3f050f1e0b9d31aceabb93429b46e7f6ac5d203711861e35203e5de"
DIAG = ROOT / "native_preparation_20261015/fp16_probe"
ONNX_SHA256 = "232f666524f07b9d3971381462365774222f0488f6912e144f73c50d36d2fc21"
SAMPLES_SHA256 = "96804831f6aaf4e0384dacd66db59fd5509bf0a62ca0a227c0646bdd6b7134d1"
EXISTING_ENGINE_SHA256 = "72010c4f41d2fdc46aaaac0dc86709e0bf8230d2d69893702351c2626e61cd86"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_repeats(outputs, reference):
    """Keep the frozen 0.01/0.01 limit and expose per-logit failures."""
    expected = np.asarray(reference, dtype=np.float32)
    if expected.shape != (1, expected.size) or not np.isfinite(expected).all() or not outputs:
        raise ValueError("Expected one finite reference logit row and at least one output")
    values = [np.asarray(output, dtype=np.float32) for output in outputs]
    if any(value.shape != expected.shape or not np.isfinite(value).all() for value in values):
        raise ValueError("FP16 output shape or finiteness differs from reference")
    failures = []
    for value in values:
        delta = np.abs(value - expected)
        failures.append(np.flatnonzero(delta > 0.01 + 0.01 * np.abs(expected)).tolist())
    baseline = values[0]
    return {
        "tolerance": {"atol": 0.01, "rtol": 0.01},
        "allclose_each_run": [bool(np.allclose(value, expected, atol=0.01, rtol=0.01))
                              for value in values],
        "failure_indices_each_run": failures,
        "max_abs_error_each_run": [float(np.max(np.abs(value - expected))) for value in values],
        "top1_agreement_each_run": [bool(np.argmax(value) == np.argmax(expected)) for value in values],
        "run_to_run_max_abs": float(max(np.max(np.abs(value - baseline)) for value in values)),
    }


def classify_build_variation(engines):
    stable = all(engine["run_to_run_max_abs"] == 0.0 for engine in engines)
    validity = [all(engine["allclose_each_run"]) for engine in engines]
    if stable and len(set(validity)) > 1:
        return "build_sensitive_not_runtime_noise"
    if not stable:
        return "runtime_variation_observed"
    return "same_fixed_tolerance_outcome_across_builds"


def _infer_repeats(engine_path, input_tensor, reference):
    from framework.resnet_compare.trt_runtime import TensorRTRuntime

    runtime = TensorRTRuntime(engine_path)
    outputs = [runtime.infer(input_tensor)["logits"].numpy() for _ in range(5)]
    return {**summarize_repeats(outputs, reference),
            "engine_sha256": _sha(engine_path),
            "layer_precisions": sorted({row["precision"] for row in runtime.layer_precision}),
            "target_graph_verified": runtime.target_graph_verified}


def main():
    import torch
    from framework.resnet_compare.trt_runtime import build_engine

    onnx, samples, old_engine = JOB / "candidate.onnx", JOB / "samples.npz", JOB / "candidate.engine"
    if (_sha(onnx), _sha(samples), _sha(old_engine)) != (
            ONNX_SHA256, SAMPLES_SHA256, EXISTING_ENGINE_SHA256):
        raise ValueError("Fixed ONNX/sample/existing-engine identity changed")
    with np.load(samples, allow_pickle=False) as pack:
        input_tensor = torch.from_numpy(pack["inputs"][0:1].copy())
        reference = pack["source_logits"][0:1].copy()
    DIAG.mkdir(parents=True, exist_ok=True)
    second_engine = DIAG / "second_fp16.engine"
    old = _infer_repeats(old_engine, input_tensor, reference)
    build = build_engine(onnx, second_engine, "FP16")
    second = _infer_repeats(second_engine, input_tensor, reference)
    receipt = {"scope": "same_ONNX_sample_reference_synthetic_FP16_diagnosis_not_ImageNet",
               "onnx_sha256": ONNX_SHA256, "samples_sha256": SAMPLES_SHA256,
               "existing_engine": old, "second_build": build, "second_engine": second,
               "engine_builds_total_including_existing": 2,
               "precision_policy_changed": False,
               "diagnosis": classify_build_variation([old, second])}
    (DIAG / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def run_third():
    """One final bounded rebuild; preserve the earlier two-build receipt."""
    import torch
    from framework.resnet_compare.trt_runtime import build_engine

    receipt = json.loads((DIAG / "receipt.json").read_text())
    if receipt["onnx_sha256"] != ONNX_SHA256 or receipt["samples_sha256"] != SAMPLES_SHA256:
        raise ValueError("Earlier fixed diagnosis identity differs")
    onnx, samples = JOB / "candidate.onnx", JOB / "samples.npz"
    if _sha(onnx) != ONNX_SHA256 or _sha(samples) != SAMPLES_SHA256:
        raise ValueError("ONNX/sample changed before third build")
    with np.load(samples, allow_pickle=False) as pack:
        input_tensor = torch.from_numpy(pack["inputs"][0:1].copy())
        reference = pack["source_logits"][0:1].copy()
    third_engine = DIAG / "third_fp16.engine"
    build = build_engine(onnx, third_engine, "FP16")
    third = _infer_repeats(third_engine, input_tensor, reference)
    summary = {"scope": receipt["scope"], "onnx_sha256": ONNX_SHA256,
               "samples_sha256": SAMPLES_SHA256,
               "prior_receipt_sha256": _sha(DIAG / "receipt.json"),
               "third_build": build, "third_engine": third,
               "engine_builds_total_including_existing": 3,
               "precision_policy_changed": False,
               "diagnosis": classify_build_variation([
                   receipt["existing_engine"], receipt["second_engine"], third])}
    (DIAG / "third_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == []:
        main()
    elif sys.argv[1:] == ["third"]:
        run_third()
    else:
        raise SystemExit("Usage: resnet_compare_orin_fp16_probe.py [third]")
