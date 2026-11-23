"""Validated, cwd-independent S1 protocol and honest classification records."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


METHOD_IDS = ("copt", "mdp", "geta", "modelopt", "pruner")
RESULT_METHOD_IDS = (*METHOD_IDS, "dense_baseline")
RESULT_STATUSES = ("PLANNED", "RUNNING", "READY", "PARTIAL", "UNSUPPORTED", "FAILED")
RESULT_SCHEMA_VERSION = "resnet_compare_result_v1"
REFERENCE_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{name} must be a mapping")
    return value


def _path(value: Any, base: Path, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be a path string")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _validate_evaluation(protocol: Mapping[str, Any]) -> None:
    preprocessing = _mapping(protocol.get("preprocessing"), "preprocessing")
    expected = {"resize": 256, "center_crop": 224, "interpolation": "bilinear",
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    _require(all(preprocessing.get(k) == v for k, v in expected.items()), "V1 preprocessing changed")
    metrics = _mapping(protocol.get("metrics"), "metrics")
    _require(metrics.get("accuracy_unit") == "percentage_points", "Accuracy must use percentage points")
    _require(metrics.get("objectives") == ["latency_ms", "energy_j", "top1_pp"], "Classification objectives changed")
    _require(metrics.get("directions") == {"latency_ms": "min", "energy_j": "min", "top1_pp": "max"}, "Classification objective directions changed")
    budgets = _mapping(protocol.get("budgets"), "budgets")
    _require(budgets.get("copt_feedback_rounds") == 4, "Copt smoke requires four real feedback rounds")
    _require(budgets.get("formal_training") is False, "S1 does not authorize formal training")
    for key, maximum in (("smoke_samples", 256), ("smoke_training_steps", 32), ("pruner_smoke_measurements", 32)):
        value = budgets.get(key)
        _require(type(value) is int and 1 <= value <= maximum, f"{key} must be an integer from 1 to {maximum}")
    _require(protocol.get("phase") == "S1", "Only S1 execution is authorized")
    resources = _mapping(protocol.get("resources"), "resources")
    _require(resources.get("target") == "orin" and resources.get("target_serial") is True, "Orin target execution must be serial")


def load_protocol(config_path: Path) -> dict:
    """Validate fixed RN50-V1/S1 identity; resolve paths against the config file.

    A null/unavailable data_root is valid configuration, but preflight reports
    NEEDS_DATA. Loading never creates paths, downloads data, or allocates GPUs.
    """
    path = Path(config_path).expanduser().resolve()
    protocol = copy.deepcopy(_mapping(yaml.safe_load(path.read_text()), "protocol"))
    _require(protocol.get("schema_version") == "resnet_compare_protocol_v1", "Unknown protocol schema")
    _require(isinstance(protocol.get("run_id"), str) and bool(protocol["run_id"].strip()), "run_id is required")
    methods = protocol.get("methods")
    _require(isinstance(methods, list) and len(methods) == 5 and all(isinstance(m, str) for m in methods) and set(methods) == set(METHOD_IDS), "Exactly five named methods required")
    _require(type(protocol.get("seed")) is int and protocol["seed"] == 17, "S1 seed must be 17")
    model = _mapping(protocol.get("model"), "model")
    _require(model.get("model_id") == "resnet50_imagenet_v1", "RN50-V1 model required")
    _require(model.get("weights") == "IMAGENET1K_V1", "Explicit IMAGENET1K_V1 required")
    _require(model.get("input_size") == [3, 224, 224] and model.get("num_classes") == 1000, "Expected RGB 224 and 1000 logits")
    _require(model.get("initial_weight_sha256") == REFERENCE_SHA256, "Common weight digest differs")
    data = _mapping(protocol.get("data"), "data")
    for key, value in (("search_count", 10000), ("calibration_count", 1024), ("validation_count", 50000)):
        _require(type(data.get(key)) is int and data[key] == value, f"Invalid {key}")
    _require(data.get("final_validation_for_selection") is False, "Final val cannot select candidates")
    resources = _mapping(protocol.get("resources"), "resources")
    from .resources import validate_gpu_request
    validate_gpu_request(resources.get("gpu_pool"), resources.get("explicit_ids"))
    _validate_evaluation(protocol)
    protocol["run_root"] = _path(protocol.get("run_root"), path.parent, "run_root")
    _require(Path(protocol["run_root"]).name == protocol["run_id"], "run_root basename must match run_id; do not mix runs")
    model["checkpoint"] = _path(model.get("checkpoint"), path.parent, "checkpoint")
    if data.get("data_root") is not None:
        data["data_root"] = _path(data["data_root"], path.parent, "data_root")
    return protocol


def preflight_protocol(protocol: Mapping[str, Any]) -> dict:
    """Check dataset location only, not its ImageNet identity or class mapping.

    Manifest construction must still verify sample counts and class identity.
    READY here means the two directory prerequisites exist, not result READY.
    """
    root = protocol["data"].get("data_root")
    missing = []
    if root is None or not Path(root).is_dir():
        missing.append("data_root")
    else:
        missing.extend(name for name in ("train", "val") if not (Path(root) / name).is_dir())
    return {
        "status": "NEEDS_DATA" if missing else "READY",
        "missing": missing,
        "reason": "ImageNet root/train/val missing; class mapping and counts unverified" if missing else "Directories exist; manifest identity/count audit still required",
        "dataset_verified": False,
    }


def validate_result(result: Mapping[str, Any]) -> dict:
    """Validate a result without converting missing measurements into zeros."""
    row = copy.deepcopy(dict(result))
    required = (
        "run_id", "method_id", "seed", "candidate_id", "source_commit",
        "initial_weight_sha256", "structure_digest", "precision_semantics",
        "actual_backend", "hardware_id", "sample_manifest_sha256", "top1_pp",
        "top5_pp", "accuracy_sample_count", "accuracy_phase", "latency_ms",
        "energy_j", "measurement_boundary", "budget_used", "status", "reason", "evidence_paths",
    )
    _require(row.get("schema_version") == RESULT_SCHEMA_VERSION, "Unknown result schema")
    _require(all(key in row for key in required), "Result fields missing")
    _require(row["method_id"] in RESULT_METHOD_IDS, "Unknown method")
    _require(row["status"] in RESULT_STATUSES, "Unknown result status")
    _require(isinstance(row["reason"], str) and bool(row["reason"].strip()), "Reason required")
    for key in ("top1_pp", "top5_pp", "latency_ms", "energy_j"):
        value = row[key]
        if value is not None:
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid {key}")
            if key.startswith("top"):
                _require(value <= 100, f"{key} must be percentage points")
    if row["top1_pp"] is not None and row["top5_pp"] is not None:
        _require(row["top5_pp"] >= row["top1_pp"], "top5 cannot be below top1")
    _require(type(row["accuracy_sample_count"]) is int and row["accuracy_sample_count"] >= 0, "Invalid sample count")
    if row["top1_pp"] is not None or row["top5_pp"] is not None:
        _require(row["accuracy_sample_count"] > 0, "Measured accuracy requires sample count")
    if row["status"] == "READY":
        _require(type(row["seed"]) is int and row["seed"] == 17, "S1 READY seed must be 17")
        _require(row["initial_weight_sha256"] == REFERENCE_SHA256, "READY common weight digest differs")
        identity = ("source_commit", "initial_weight_sha256", "structure_digest", "actual_backend", "hardware_id", "sample_manifest_sha256", "measurement_boundary", "evidence_paths")
        _require(all(row[key] for key in identity), "READY requires measured identity/evidence")
        _require(row.get("target_graph_verified") is True and row.get("numerical_verified") is True, "READY requires target graph and numerical acceptance")
        _require(row["top1_pp"] is not None and row["top5_pp"] is not None and row["latency_ms"] is not None, "READY requires target measurements")
    return row
