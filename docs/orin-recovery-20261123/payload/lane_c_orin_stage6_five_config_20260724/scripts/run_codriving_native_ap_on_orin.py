#!/usr/bin/env python3
"""Run the CoDriving native full-1789 AP function without H800 NVML steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/home/jichengzhi/lane_c_orin_stage6_five_config_20260724")
sys.path.insert(0, str(ROOT / "code"))

from scripts.stage6_codriving_native_fp32_baseline_v1 import run_ap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ap-samples", type=int, choices=(16, 1789), required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ap-out", type=Path, required=True)
    args = parser.parse_args()
    report = run_ap(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
