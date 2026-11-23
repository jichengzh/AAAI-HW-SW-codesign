#!/usr/bin/env python3
"""Evaluate F-Cooper AP with TRT replacing its post-scatter dense encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorrt as trt
import torch
from torch import nn
from torch.utils.data import DataLoader


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TorchTensorRTRunner:
    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inputs = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError("F-Cooper dense engine must have one input and one output")
        self.input_shape = tuple(self.engine.get_tensor_shape(self.inputs[0]))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.outputs[0]))
        if any(value <= 0 for value in (*self.input_shape, *self.output_shape)):
            raise ValueError("F-Cooper AP bridge requires static engine shapes")
        self.output_dtype = torch.from_numpy(
            np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.outputs[0])))
        ).dtype
        self.output_buffer = torch.empty(
            self.output_shape, device="cuda", dtype=self.output_dtype
        )
        self.call_count = 0

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"engine input shape drift: {tuple(source.shape)}")
        source = source.contiguous()
        self.context.set_tensor_address(self.inputs[0], source.data_ptr())
        self.context.set_tensor_address(self.outputs[0], self.output_buffer.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self.call_count += 1
        return self.output_buffer


class EngineBackbone(nn.Module):
    def __init__(self, runner: TorchTensorRTRunner):
        super().__init__()
        self.runner = runner

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source = data["spatial_features"]
        agents = int(source.shape[0])
        engine_batch = int(self.runner.input_shape[0])
        if agents > engine_batch:
            raise ValueError(f"record has {agents} agents, engine supports {engine_batch}")
        padded = torch.zeros(
            self.runner.input_shape, device=source.device, dtype=source.dtype
        )
        padded[:agents] = source
        encoded = self.runner(padded)
        return {"spatial_features_2d": encoded[:agents].float()}


class IdentityShrinker(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit checkpoint for formal recovered-source evaluation.",
    )
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    hypes = yaml_utils.load_yaml(str(args.config), SimpleNamespace(model_dir=None))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    model = train_utils.create_model(hypes)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        _, model = train_utils.load_saved_model(str(args.checkpoint_dir), model)
    model.cuda().eval()
    runner = TorchTensorRTRunner(args.engine)
    model.backbone_m1 = EngineBackbone(runner)
    model.shrinker_m1 = IdentityShrinker()
    stats = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    processed = failed = 0
    started = time.time()
    for batch in loader:
        if batch is None:
            continue
        try:
            batch = train_utils.to_device(batch, torch.device("cuda"))
            with torch.no_grad():
                result = inference_utils.inference_intermediate_fusion(
                    batch, model, dataset
                )
            for threshold in stats:
                eval_utils.caluclate_tp_fp(
                    result["pred_box_tensor"],
                    result["pred_score"],
                    result["gt_box_tensor"],
                    stats,
                    threshold,
                )
            processed += 1
            if args.max_samples is not None and processed >= args.max_samples:
                break
        except Exception:
            failed += 1
            raise
    ap30, ap50, ap70 = (
        eval_utils.calculate_ap(stats, threshold)[0]
        for threshold in (0.3, 0.5, 0.7)
    )
    expected_samples = (
        min(len(dataset), args.max_samples)
        if args.max_samples is not None
        else len(dataset)
    )
    report = {
        "schema_version": "fcooper_trt_ap_report_v1",
        "status": (
            "success_full"
            if failed == 0 and processed == len(dataset)
            else "success_sanity"
            if failed == 0 and processed == expected_samples
            else "failure"
        ),
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": len(dataset),
        "processed_samples": processed,
        "failed_samples": failed,
        "engine_samples": processed,
        "fallback_samples": 0,
        "engine_calls": runner.call_count,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_seconds": time.time() - started,
        "engine_path": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "checkpoint_sha256": sha256_file(
            args.checkpoint
            if args.checkpoint is not None
            else args.checkpoint_dir / "net_epoch_bestval_at23.pth"
        ),
        "config_sha256": sha256_file(args.config),
        "numerical_contract": {
            "full_dataset_engine_execution": processed == len(dataset),
            "requested_samples": expected_samples,
            "silent_fallback_forbidden": True,
            "fallback_samples": 0,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
