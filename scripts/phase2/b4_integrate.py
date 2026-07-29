"""Build B4 direct-grid LUT and consolidated AP inputs without mutating inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from framework.stage2.contracts import (
    resolve_safe_output_root,
    safe_output_path,
    write_json_idempotent,
)


ROOT = Path(__file__).resolve().parents[2]
LABEL_W = {
    "s0_16": [16, 128, 256],
    "s0_32": [32, 128, 256],
    "s1_32": [64, 32, 256],
    "s1_64": [64, 64, 256],
    "s2_64": [64, 128, 64],
    "s2_128": [64, 128, 128],
    "mix_a": [32, 96, 192],
    "mix_b": [48, 64, 256],
    "mix_c": [16, 128, 128],
    "mix_d": [48, 128, 128],
    "mix_e": [64, 96, 128],
    "mix_f": [32, 64, 64],
}


def _find_ap(expansion: dict, tag: str) -> float | None:
    for item in expansion.get("results", []):
        if item.get("tag") == tag:
            return round(float(item["ap70"]), 4)
    return None


def main(output_root: str | Path | None = None, input_root: str | Path | None = None) -> None:
    if output_root is None:
        raise ValueError("output_root is required")
    output_root = resolve_safe_output_root(output_root, ROOT)
    inputs = Path(input_root) if input_root is not None else ROOT / "results"
    gap1 = json.loads((inputs / "gap1_grid_corrected.json").read_text(encoding="utf-8"))
    widths = {
        tuple(int(value) for value in row["num_filters"]): {
            "num_filters": [int(value) for value in row["num_filters"]],
            "label": row["label"],
            "default_us": float(row["default_us"]),
            "tuned_us": float(row["tuned_us"]),
        }
        for row in gap1["grid"]
    }
    for row in csv.DictReader(
        (inputs / "lut_results_grid.csv").read_text(encoding="utf-8").splitlines()
    ):
        label, tuned = row["label"], float(row["tuned_us"])
        if label in LABEL_W and tuned > 0:
            widths[tuple(LABEL_W[label])] = {
                "num_filters": LABEL_W[label],
                "label": label,
                "default_us": float(row["default_us"]),
                "tuned_us": tuned,
            }
    lut = {
        "_format": "direct grid (per-width real H800 tuned/default us)",
        "_source": "gap1_grid_corrected + lut_results_grid (relaunch); -1 rows excluded",
        "widths": sorted(widths.values(), key=lambda row: row["num_filters"]),
    }
    expansion = json.loads((inputs / "ap70_depgraph_expansion.json").read_text(encoding="utf-8"))
    ap_mixb, ap_mixd = _find_ap(expansion, "mix_b"), _find_ap(expansion, "mix_d")
    apm = json.loads((inputs / "ap70_model_pyramid.json").read_text(encoding="utf-8"))
    apm["table"] = [
        {"num_filters": [64, 128, 256], "ap70": 0.6309, "src": "stage_a"},
        {"num_filters": [32, 64, 128], "ap70": 0.5641, "src": "stage_a"},
        {"num_filters": [16, 32, 64], "ap70": 0.5300, "src": "stage_a"},
        {"num_filters": [48, 96, 192], "ap70": 0.5905, "src": "stage_a trap25 W_g(pair1)"},
        {"num_filters": [64, 96, 192], "ap70": 0.5905, "src": "pad64 P_g(pair1) weight-identity"},
        {"num_filters": [48, 64, 256], "ap70": ap_mixb, "src": "expansion mix_b W_g(pair2)"},
        {"num_filters": [64, 64, 256], "ap70": ap_mixb, "src": "s1_64 P_g(pair2) weight-identity"},
        {"num_filters": [48, 128, 128], "ap70": ap_mixd, "src": "expansion mix_d W_g(pair3)"},
        {
            "num_filters": [64, 128, 128],
            "ap70": ap_mixd,
            "src": "s2_128 P_g(pair3) weight-identity",
        },
    ]
    lut_out = safe_output_path(output_root, "results/latency_lut_pyramid.json")
    ap_out = safe_output_path(output_root, "results/ap70_model_pyramid.json")
    write_json_idempotent(lut_out, lut)
    write_json_idempotent(ap_out, apm)
    print(f"[LUT] {len(widths)} priceable widths -> {lut_out}")
    print(f"[AP] consolidated table = {len(apm['table'])} real-AP widths -> {ap_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-root", default=None)
    arguments = parser.parse_args()
    main(arguments.output_root, arguments.input_root)
