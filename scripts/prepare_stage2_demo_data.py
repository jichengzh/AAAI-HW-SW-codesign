#!/usr/bin/env python3
"""Create self-contained Stage2 demo inputs.

The open-source package intentionally does not commit experimental JSON/YAML
artifacts.  This script writes a minimal local demo under results/ and
framework/partitions/ so the Stage2 search drivers can run immediately after a
fresh clone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("pyyaml is required: pip install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]
DEMO_MARKER = ".stage2_demo_v1.json"


PYRAMID_GRID = [
    ("p75", [16, 32, 64], 12689.07, 483.63, 0.5300),
    ("p50", [32, 64, 128], 26241.67, 3010.51, 0.5641),
    ("mix_b", [48, 64, 256], 41460.19, 19404.95, 0.6362),
    ("trap25", [48, 96, 192], 42355.27, 21614.80, 0.5905),
    ("mix_d", [48, 128, 128], 42447.41, 21071.43, 0.6369),
    ("s1_64", [64, 64, 256], 46445.27, 5894.56, 0.6362),
    ("pad64", [64, 96, 192], 47407.26, 6152.04, 0.5905),
    ("s2_128", [64, 128, 128], 47435.27, 5506.57, 0.6369),
    ("base", [64, 128, 256], 56321.21, 6319.98, 0.6309),
]


CODRIVING_GRID = [
    ("p75", [16, 32, 64], 795.0, 361.0, 0.4049),
    ("p50", [32, 64, 128], 3643.15, 1609.10, 0.3845),
    ("p25", [48, 96, 192], 8874.0, 4033.0, 0.3661),
    ("base", [64, 128, 256], 16680.47, 8057.79, 0.4063),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        help="New or previously-created Stage2 demo directory; never the repository root.",
    )
    return parser.parse_args()


def _safe_output_root(value: str) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    repository_root = ROOT.resolve()
    if lexical != resolved:
        raise ValueError("output-root must not traverse a symlink or parent escape")
    if resolved == Path("/"):
        raise ValueError("output-root must not be the filesystem root")
    if resolved == repository_root:
        raise ValueError("output-root must not be the repository root")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("output-root must be a directory")
        marker = resolved / DEMO_MARKER
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("refusing existing non-demo directory")
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("refusing existing non-demo directory") from exc
        if marker_data != {"schema": "stage2_demo_root_v1"}:
            raise ValueError("refusing existing non-demo directory")
    else:
        resolved.mkdir(parents=True, exist_ok=False)
        (resolved / DEMO_MARKER).write_text(
            json.dumps({"schema": "stage2_demo_root_v1"}, indent=2) + "\n",
            encoding="utf-8",
        )
    return resolved


def _output_path(output_root: Path, relative_path: str) -> Path:
    candidate = output_root / relative_path
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(output_root):
        raise ValueError("demo output escapes output-root")
    if candidate.is_symlink():
        raise ValueError("refusing to write through a symlink")
    return candidate


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"refusing to overwrite non-file output: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite existing output: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _seed_grid(rows: list[tuple[str, list[int], float, float, float]]) -> dict[str, Any]:
    return {
        "schema": "stage2_demo_seed_grid_v1",
        "scope": "demo-only H800 TVM latency/AP anchors",
        "grid": [
            {
                "label": label,
                "num_filters": width,
                "default_us": default_us,
                "tuned_us": tuned_us,
                "ratio": round(default_us / tuned_us, 6),
                "ap70": ap70,
            }
            for label, width, default_us, tuned_us, ap70 in rows
        ],
    }


def _latency_lut(
    model: str, rows: list[tuple[str, list[int], float, float, float]]
) -> dict[str, Any]:
    return {
        "_format": "direct_grid",
        "_model": model,
        "_scope": "demo-only; replace with measured per-width TVM/MetaSchedule LUT for real use",
        "widths": [
            {
                "label": label,
                "num_filters": width,
                "default_us": default_us,
                "tuned_us": tuned_us,
            }
            for label, width, default_us, tuned_us, _ap70 in rows
        ],
    }


def _ap_model(model: str, rows: list[tuple[str, list[int], float, float, float]]) -> dict[str, Any]:
    return {
        "schema": "stage2_demo_ap_model_v1",
        "model": model,
        "metric": "AP70",
        "scope": "demo-only AP anchors; replace with validation/finetune results for real use",
        "table": [
            {
                "label": label,
                "num_filters": width,
                "ap70": ap70,
            }
            for label, width, _default_us, _tuned_us, ap70 in rows
        ],
    }


def _pyramid_manifest() -> dict[str, Any]:
    return {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "ckpt_status": "demo_synthetic",
        "hw_capability": {
            "name": "h800_tvm_demo",
            "int8_align": 32,
            "fp16_align": 8,
            "int8_pack_factor": 4,
            "alignment_enforcement": "hard",
            "legal_bits": ["INT8", "FP16"],
            "legal_granularity_w": ["per_tensor", "per_channel"],
        },
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_stage0_grouped",
                "bucket": "pyramid_backbone_stage0",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": 64,
                "max_rate": 0.75,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["stage0"],
                "feature": {
                    "min_ic_bn": 1,
                    "max_groups": 32,
                    "op_types": ["Conv2d"],
                    "fanout_buckets": ["pyramid_backbone_stage1"],
                },
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone_dense",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "legal_granularity_w": ["per_tensor", "per_channel"],
                "member_groups": ["pyramid_stage0_grouped"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _codriving_manifest() -> dict[str, Any]:
    return {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "codriving",
        "scan_status": "ok",
        "ckpt_status": "demo_synthetic",
        "hw_capability": {
            "name": "h800_tvm_demo",
            "int8_align": 32,
            "fp16_align": 8,
            "int8_pack_factor": 4,
            "alignment_enforcement": "hard",
            "legal_bits": ["INT8", "FP16"],
            "legal_granularity_w": ["per_tensor", "per_channel"],
        },
        "view_b1_search_groups": [
            {
                "search_group_id": "codriving_resnet_dense",
                "bucket": "codriving_resnet_backbone",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": 4,
                "max_rate": 0.75,
                "grouped_conv": False,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["resnet"],
                "feature": {
                    "min_ic_bn": 16,
                    "max_groups": 1,
                    "op_types": ["Conv2d"],
                    "fanout_buckets": ["codriving_neck"],
                },
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "codriving_resnet_dense",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "legal_granularity_w": ["per_tensor", "per_channel"],
                "member_groups": ["codriving_resnet_dense"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _classification_report() -> dict[str, Any]:
    return {
        "schema": "stage1_model_classification_v1",
        "generated_on": "demo",
        "backend_policy": {
            "default_backend": "h800_tvm",
            "default_new_measurement_backend": "h800_tvm",
            "allowed_new_measurement_backends": ["h800_tvm"],
            "historical_evidence_backends": ["trt"],
            "historical_trt_evidence_only": True,
        },
        "acceleration_class_labels": {
            "CO_ACCELERATION_REQUIRED": "needs co-optimized acceleration",
            "SEPARABLE_ACCELERATION": "separable acceleration in scoped envelope",
            "SCAN_FAILED": "scan failed",
        },
        "source_manifests": [
            "framework/partitions/pyramid_lidar_partition.yaml",
            "framework/partitions/codriving_partition.yaml",
        ],
        "models": [
            {
                "model": "pyramid_lidar",
                "manifest": "framework/partitions/pyramid_lidar_partition.yaml",
                "ckpt_status": "demo_synthetic",
                "acceleration_class": "CO_ACCELERATION_REQUIRED",
                "classification": "DEMO_P_IC_BN_HUB_CONTEXT",
                "scope": "demo_rsu_dense_core",
                "evidence_level": "demo_only_not_measured",
                "unsupported_conclusions": [
                    "demo data as paper measurement",
                    "dense_core_speedup_as_full_model_speedup",
                ],
                "blockers": ["full_model_claim_requires_real_e2e_evidence"],
                "no_overpromotion": True,
            },
            {
                "model": "codriving",
                "manifest": "framework/partitions/codriving_partition.yaml",
                "ckpt_status": "demo_synthetic",
                "acceleration_class": "SEPARABLE_ACCELERATION",
                "classification": "DEMO_STANDARD_CONV_DENSE_ENVELOPE",
                "scope": "demo_rsu_dense_core",
                "evidence_level": "demo_only_not_measured",
                "unsupported_conclusions": [
                    "groups=1 proves model-level separability",
                    "cross-model extrapolation from CoDriving to other models",
                    "dense_core_speedup_as_full_model_speedup",
                ],
                "blockers": ["full_model_claim_requires_real_e2e_evidence"],
                "no_overpromotion": True,
            },
        ],
        "no_overpromotion": True,
    }


def main() -> None:
    output_root = _safe_output_root(_parse_args().output_root)
    outputs = {
        "gap": _output_path(output_root, "results/gap1_grid_corrected.json"),
        "pyramid_lut": _output_path(output_root, "results/latency_lut_pyramid.json"),
        "pyramid_ap": _output_path(output_root, "results/ap70_model_pyramid.json"),
        "codriving_lut": _output_path(output_root, "results/latency_lut_codriving.json"),
        "codriving_ap": _output_path(output_root, "results/ap70_model_codriving.json"),
        "pyramid_manifest": _output_path(
            output_root, "framework/partitions/pyramid_lidar_partition.yaml"
        ),
        "codriving_manifest": _output_path(
            output_root, "framework/partitions/codriving_partition.yaml"
        ),
        "classification": _output_path(output_root, "results/model_classifier.json"),
    }

    _write_json(outputs["gap"], _seed_grid(PYRAMID_GRID))
    _write_json(outputs["pyramid_lut"], _latency_lut("pyramid_lidar", PYRAMID_GRID))
    _write_json(outputs["pyramid_ap"], _ap_model("pyramid_lidar", PYRAMID_GRID))
    _write_json(outputs["codriving_lut"], _latency_lut("codriving", CODRIVING_GRID))
    _write_json(outputs["codriving_ap"], _ap_model("codriving", CODRIVING_GRID))
    _write_yaml(outputs["pyramid_manifest"], _pyramid_manifest())
    _write_yaml(outputs["codriving_manifest"], _codriving_manifest())
    _write_json(outputs["classification"], _classification_report())

    print("stage2_demo_data_ready")
    for path in outputs.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
