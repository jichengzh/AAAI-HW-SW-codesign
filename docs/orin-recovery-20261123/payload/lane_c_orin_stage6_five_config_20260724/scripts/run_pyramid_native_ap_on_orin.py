#!/usr/bin/env python3
"""Run Pyramid native FP32 AP directly, bypassing H800-only GPU preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/home/jichengzhi/lane_c_orin_stage6_five_config_20260724")
sys.path.insert(0, str(ROOT / "code"))

from scripts.stage2_h800_true_fp32_ap_eval import run_true_fp32_eval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, choices=(16, 1789), required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    runtime_args = argparse.Namespace(
        heal_root=str(args.heal_root),
        ckpt_dir=str(args.model_dir),
        checkpoint_path=str(args.checkpoint),
        raw_dir=str(args.raw_dir),
        fusion_method="intermediate",
        eval_range="102.4,51.2",
        num_samples=args.num_samples,
        dataset="DAIR-V2X",
        eval_split="val",
        label="pyramid_original_default_orin",
    )
    report = run_true_fp32_eval(runtime_args, args.report_json)
    if int(report.get("num_samples") or 0) != args.num_samples:
        raise RuntimeError("Pyramid native AP sample contract mismatch")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
