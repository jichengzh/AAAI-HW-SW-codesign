"""Bounded real-input diagnosis of a preserved, failed dense FP16 result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CHECKPOINT_SHA = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
APPROVED_ROOT = Path("/home/jichengzhi/resnet_compare_s1_20261013")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference, first, second):
    reference = np.asarray(reference, dtype=np.float32)
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if reference.shape != first.shape or first.shape != second.shape or not all(
            np.isfinite(array).all() for array in (reference, first, second)):
        raise ValueError("Diagnostic logits must have equal finite shape")
    margin = 0.01 + 0.01 * np.abs(reference)
    delta = np.abs(first - reference)
    return {"bitwise_equal": bool(np.array_equal(first, second)),
            "repeat_max_abs": float(np.max(np.abs(first - second))),
            "first_max_abs": float(np.max(delta)),
            "first_mean_abs": float(np.mean(delta)),
            "first_violation_count": int(np.count_nonzero(delta > margin)),
            "first_allclose_fp16_fixed": bool(np.allclose(first, reference, atol=0.01, rtol=0.01))}


def run(fp16_root, fp32_root, checkpoint, output, indices):
    import torch
    from torchvision.models import resnet50
    from framework.resnet_compare.target_entry import serial_target, unpack_samples, verify_bundle
    from framework.resnet_compare.trt_runtime import TensorRTRuntime

    fp16_root, fp32_root = Path(fp16_root).resolve(), Path(fp32_root).resolve()
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if any(APPROVED_ROOT not in path.parents for path in (fp16_root, fp32_root, checkpoint, output)):
        raise ValueError("Diagnostic files must stay within approved Orin scope")
    if output.exists():
        raise FileExistsError("Diagnostic evidence exists; preserve original")
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Official checkpoint SHA differs")
    fp16_manifest = verify_bundle(fp16_root)
    fp32_manifest = verify_bundle(fp32_root)
    if (fp16_manifest["precision"] != "FP16" or fp32_manifest["precision"] != "FP32"
            or fp16_manifest["sample_pack_sha256"] != fp32_manifest["sample_pack_sha256"]
            or fp16_manifest["export_sha256"] != fp32_manifest["export_sha256"]):
        raise ValueError("Diagnostic must compare same ONNX/samples across two precisions")
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < 256 for i in indices):
        raise ValueError("Distinct frozen sample indices 0..255 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with serial_target(APPROVED_ROOT):
        samples = unpack_samples(fp16_root, fp16_manifest)
        state = torch.load(checkpoint, map_location="cpu")
        model = resnet50(weights=None, num_classes=1000)
        model.load_state_dict(state, strict=True)
        fp32_model = model.eval().cuda()
        fp16_model = resnet50(weights=None, num_classes=1000)
        fp16_model.load_state_dict(state, strict=True)
        fp16_model = fp16_model.eval().half().cuda()
        fp16_runtime = TensorRTRuntime(fp16_root / "candidate.engine")
        fp32_runtime = TensorRTRuntime(fp32_root / "candidate.engine")
        cases = []
        with torch.no_grad():
            for index in indices:
                sample = samples[index]
                tensor = sample["input"]
                source = sample["source_logits"].numpy()
                trt16_first = fp16_runtime.infer(tensor)["logits"].numpy()
                trt16_second = fp16_runtime.infer(tensor)["logits"].numpy()
                trt32 = fp32_runtime.infer(tensor)["logits"].numpy()
                torch32 = fp32_model(tensor.cuda()).float().cpu().numpy()
                torch16 = fp16_model(tensor.half().cuda()).float().cpu().numpy()
                cases.append({"index": index, "sample_id": sample["sample_id"],
                              "trt16_vs_source": compare(source, trt16_first, trt16_second),
                              "trt16_vs_native16": compare(torch16, trt16_first, trt16_second),
                              "native16_vs_source": compare(source, torch16, torch16),
                              "trt32_vs_source": compare(source, trt32, trt32),
                              "torch32_vs_source": compare(source, torch32, torch32),
                              "argmax": {"source": int(source.argmax()), "trt16": int(trt16_first.argmax()),
                                         "native16": int(torch16.argmax()), "trt32": int(trt32.argmax())}})
        output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {"fp16_transport_id": fp16_manifest["transport_id"],
                    "fp32_transport_id": fp32_manifest["transport_id"],
                    "same_sample_pack_sha256": fp16_manifest["sample_pack_sha256"],
                    "same_onnx_sha256": fp16_manifest["export_sha256"],
                    "fp16_engine_sha256": sha(fp16_root / "candidate.engine"),
                    "fp32_engine_sha256": sha(fp32_root / "candidate.engine"),
                    "fp16_binding_dtype": str(fp16_runtime.buffers[fp16_runtime.output_index].dtype),
                    "fp32_binding_dtype": str(fp32_runtime.buffers[fp32_runtime.output_index].dtype),
                    "tf32_enabled": False, "cases": cases}
        with output.open("x") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-root", required=True)
    parser.add_argument("--fp32-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.fp16_root, args.fp32_root, args.checkpoint, args.output,
                         args.indices), allow_nan=False))
