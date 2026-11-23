"""Run D0 original PyTorch RN50 using the authenticated FP32 transfer pack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHECKPOINT_SHA = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
APPROVED_ROOT = Path("/home/jichengzhi/resnet_compare_s1_20261013")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_job(manifest):
    candidate = manifest.get("candidate", {})
    if (candidate.get("method_id") != "dense_baseline"
            or candidate.get("precision_semantics") != "FP32"
            or manifest.get("precision") != "FP32"):
        raise ValueError("D0 requires original dense FP32 source bundle")
    if candidate.get("initial_weight_sha256") != CHECKPOINT_SHA:
        raise ValueError("D0 common checkpoint SHA differs")
    if len(manifest.get("samples", [])) != 256:
        raise ValueError("D0 requires exactly 256 frozen smoke samples")


def run(job_root, checkpoint):
    import torch
    from torchvision.models import resnet50
    from framework.resnet_compare.runtime import TorchRuntime, measure_candidate
    from framework.resnet_compare.target_entry import serial_target, unpack_samples, verify_bundle

    job_root, checkpoint = Path(job_root).resolve(), Path(checkpoint).resolve()
    if APPROVED_ROOT not in job_root.parents or APPROVED_ROOT not in checkpoint.parents:
        raise ValueError("D0 target files must stay under approved experiment root")
    manifest = verify_bundle(job_root)
    validate_job(manifest)
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("D0 transferred checkpoint bytes differ from common SHA")
    output = job_root / "d0_result"
    with serial_target(APPROVED_ROOT):
        if output.exists():
            raise FileExistsError("D0 result already exists; preserving original evidence")
        torch.set_num_threads(4)
        state = torch.load(checkpoint, map_location="cpu")  # SHA-authenticated official checkpoint.
        model = resnet50(weights=None, num_classes=1000)
        model.load_state_dict(state, strict=True)
        samples = unpack_samples(job_root, manifest)
        runtime = TorchRuntime(model.eval(), "cuda:0")
        result = measure_candidate(manifest["candidate"], {
            "target": "orin", "precision": "FP32", "adapter": runtime,
            "warmup": 50, "windows": 3, "requests_per_window": 200,
        }, samples, output)
        receipt = {"checkpoint_sha256": CHECKPOINT_SHA,
                   "transfer_manifest_sha256": sha(job_root / "transfer_manifest.json"),
                   "samples_sha256": sha(job_root / "samples.npz"),
                   "result_sha256": sha(output / "baseline_runtime.json"),
                   "requested_precision": "FP32", "actual_backend": "pytorch",
                   "result_status": result["status"]}
        with (output / "d0_receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.job_root, args.checkpoint), allow_nan=False))
