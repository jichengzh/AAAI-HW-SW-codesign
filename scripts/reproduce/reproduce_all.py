#!/usr/bin/env python3
"""Run the public, selection-only AAAI27 reproduction boundary.

Smoke mode exercises demo-derived contracts only.  Verified mode audits the
checked-in verified-artifact manifest and refuses to substitute demo records
for unavailable paper-evidence stages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEMO_MANIFEST = ROOT / "data" / "demo" / "manifest.json"
VERIFIED_MANIFEST = ROOT / "artifacts" / "verified" / "manifest.json"
SCHEMA_VERSION = "aaai27_reproduction_run_manifest_v1"
OPTIONAL_REPRO_DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
FROZEN_SEEDS = {"stage4": 20260716, "stage5": 20260717, "stage7": 20260718}
VERIFIED_ARTIFACT_CONTRACTS = {
    "stage4-cost-model-selection-report": (
        "stage4_cost_model_selection_v1",
        "verified_sanitized",
    ),
    "stage4-nested-cv-folds": ("stage4_nested_cv_folds_v1", "verified_sanitized"),
}


class PublicReproductionError(ValueError):
    """A public failure that intentionally does not expose local paths."""


@dataclass(frozen=True)
class OutputRootClaim:
    """Explicit ownership state required before this invocation may write."""

    writable: bool
    existing_manifest: dict[str, Any] | None = None


class ReproductionUnavailable(PublicReproductionError):
    """A fail-closed result that preserves completed public audit state."""

    def __init__(
        self,
        message: str,
        *,
        stages: Sequence[Mapping[str, Any]],
        inputs: Sequence[Mapping[str, str]],
    ) -> None:
        super().__init__(message)
        self.stages = [dict(stage) for stage in stages]
        self.inputs = [dict(item) for item in inputs]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "verified"))
    parser.add_argument("--output-root", required=True, help="Explicit caller-owned output directory.")
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _render_json(payload: Mapping[str, Any] | list[Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _render_jsonl_record(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicReproductionError(f"{label} is unavailable") from exc


def _safe_output_root(value: str) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved or resolved in {Path("/"), ROOT.resolve()}:
        raise PublicReproductionError("output-root must be a safe explicit path")
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_dir()):
        raise PublicReproductionError("output-root must be a directory")
    return resolved


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PublicReproductionError("internal artifact boundary violation") from exc


def _public_path(path: Path, output_root: Path) -> str:
    """Render a portable identity without revealing a caller's local output root."""
    try:
        return _relative(path)
    except PublicReproductionError:
        try:
            return path.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError as exc:
            raise PublicReproductionError("internal artifact boundary violation") from exc


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PublicReproductionError("refusing non-file output")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".reproduce-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PublicReproductionError("refusing non-file output")
    if path.exists():
        if path.read_bytes() != data:
            raise PublicReproductionError("refusing to overwrite different output")
        return
    _atomic_write(path, data)


def _claim_output_root(root: Path, mode: str) -> OutputRootClaim:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return OutputRootClaim(writable=True)
    entries = list(root.iterdir())
    if not entries:
        return OutputRootClaim(writable=True)
    manifest_path = root / "run_manifest.json"
    if len(entries) == 1 and manifest_path.is_file() and not manifest_path.is_symlink():
        existing = _read_json(manifest_path, label="existing run manifest")
        if isinstance(existing, Mapping) and existing.get("mode") == mode:
            _validate_reusable_manifest(root, existing)
            return OutputRootClaim(writable=False, existing_manifest=dict(existing))
    if manifest_path.is_file() and not manifest_path.is_symlink():
        existing = _read_json(manifest_path, label="existing run manifest")
        if isinstance(existing, Mapping) and existing.get("mode") == mode:
            _validate_reusable_manifest(root, existing)
            return OutputRootClaim(writable=False, existing_manifest=dict(existing))
    raise PublicReproductionError("output-root already contains different content")


def _validate_reusable_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """Only a byte-for-byte matching completed run can be reused idempotently."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PublicReproductionError("output-root already contains different content")
    if manifest.get("status") == "unavailable":
        if any(path.name != "run_manifest.json" for path in root.iterdir()):
            raise PublicReproductionError("output-root already contains different content")
        return
    outputs = manifest.get("outputs")
    if manifest.get("status") != "completed" or not isinstance(outputs, list):
        raise PublicReproductionError("output-root already contains different content")
    expected: set[str] = set()
    for item in outputs:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise PublicReproductionError("output-root already contains different content")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.name == "run_manifest.json":
            raise PublicReproductionError("output-root already contains different content")
        path = root / relative
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != item["sha256"]:
            raise PublicReproductionError("output-root already contains different content")
        expected.add(relative.as_posix())
    expected_directories: set[str] = set()
    for relative_path in expected:
        parent = Path(relative_path).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PublicReproductionError("output-root already contains different content")
        if path.is_dir() and relative_path not in expected_directories:
            raise PublicReproductionError("output-root already contains different content")
        if not path.is_dir() and not path.is_file():
            raise PublicReproductionError("output-root already contains different content")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "run_manifest.json"
    }
    if actual != expected:
        raise PublicReproductionError("output-root already contains different content")


def _versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for dependency in OPTIONAL_REPRO_DEPENDENCIES:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = "not-installed"
    return versions


def _component_sources() -> list[dict[str, str]]:
    paths = (
        ROOT / "scripts" / "prepare_stage2_demo_data.py",
        ROOT / "scripts" / "reproduce" / "cost_model_selection.py",
        ROOT / "scripts" / "reproduce" / "stage5_selection.py",
        ROOT / "scripts" / "reproduce" / "stage6_table.py",
        ROOT / "scripts" / "reproduce" / "stage7_selection.py",
        ROOT / "framework" / "stage4" / "cost_model_selection_v1.py",
        ROOT / "framework" / "stage5" / "production_search_v1.py",
        ROOT / "framework" / "stage6" / "paper_table_v1.py",
        ROOT / "framework" / "stage7" / "search_policy_v1.py",
    )
    return [{"identity": _relative(path), "sha256": _sha256_file(path)} for path in paths]


def _required_repro_dependencies() -> None:
    missing = [name for name in OPTIONAL_REPRO_DEPENDENCIES if _versions()[name] == "not-installed"]
    if missing:
        raise PublicReproductionError(
            "optional reproduction dependencies unavailable: " + ", ".join(missing)
        )


def _run_script(stage: str, script: Path, arguments: Sequence[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_PROXY": "*"},
    )
    if result.returncode != 0:
        raise PublicReproductionError(f"{stage} unavailable")


def _demo_inputs() -> dict[str, Any]:
    manifest = _read_json(DEMO_MANIFEST, label="demo manifest")
    if not isinstance(manifest, Mapping) or manifest.get("paper_evidence") is not False:
        raise PublicReproductionError("demo input contract is unavailable")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PublicReproductionError("demo input contract is unavailable")
    by_id: dict[str, Path] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or item.get("paper_evidence") is not False:
            raise PublicReproductionError("demo input contract is unavailable")
        artifact_id, relative_path = item.get("artifact_id"), item.get("path")
        if not isinstance(artifact_id, str) or not isinstance(relative_path, str):
            raise PublicReproductionError("demo input contract is unavailable")
        path = (ROOT / relative_path).resolve()
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(ROOT):
            raise PublicReproductionError("demo input contract is unavailable")
        by_id[artifact_id] = path
    try:
        profiles_document = _read_json(by_id["demo-capability-profiles"], label="demo profiles")
        profiles = profiles_document["profiles"]
        measurements = [
            json.loads(line)
            for line in by_id["demo-coldstart-measurements"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        graphs = [
            json.loads(line)
            for line in by_id["demo-coldstart-graph-features"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        candidates = [
            json.loads(line)
            for line in by_id["demo-candidate-pool"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise PublicReproductionError("demo input contract is unavailable") from exc
    values = [*profiles, *measurements, *graphs, *candidates]
    if not values or any(not isinstance(item, Mapping) or item.get("paper_evidence") is not False for item in values):
        raise PublicReproductionError("demo input contract is unavailable")
    return {
        "manifest": manifest,
        "profiles": list(profiles),
        "measurements": measurements,
        "graphs": graphs,
        "candidates": candidates,
        "input_hashes": [
            {"path": _relative(path), "sha256": _sha256_file(path)}
            for path in (DEMO_MANIFEST, *sorted(by_id.values()))
        ],
    }


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode())


def _write_contracts(root: Path, demo: Mapping[str, Any]) -> dict[str, Path]:
    """Materialize deterministic, explicitly non-evidence inputs for existing selectors."""
    from framework.stage2.canonical_search_v3 import build_capability_profile
    from framework.stage5.single_target_search_v2 import SearchTask, build_task_candidate_manifest

    contract_root = root / "demo_contract"
    contract_root.mkdir(exist_ok=True)
    profiles = [
        build_capability_profile(
            capability_profile_id="h800-tvm",
            hardware_target="h800",
            compiler_fingerprint=_sha256_bytes(b"demo-h800-tvm"),
            dispatch_key="tvm_auto",
            features={"int8_propagation": 0.0, "qdq_fold": 0.0},
        ),
        build_capability_profile(
            capability_profile_id="h800-trt",
            hardware_target="h800",
            compiler_fingerprint=_sha256_bytes(b"demo-h800-trt"),
            dispatch_key="trt_engine",
            features={"int8_propagation": 1.0, "qdq_fold": 0.5},
        ),
    ]
    demo_measurements = list(demo["measurements"])
    demo_graphs = list(demo["graphs"])
    demo_candidates = list(demo["candidates"])
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(6):
        measurement = demo_measurements[index % len(demo_measurements)]
        graph = demo_graphs[index % len(demo_graphs)]
        # Consume the public candidate records so the generated rows remain
        # demo-derived, while retaining Stage5's canonical structure identity.
        _ = str(demo_candidates[index % len(demo_candidates)]["family"])
        width = [16 + index * 8, 32 + index * 8, 64 + index * 16]
        model = "pyramid" if index % 2 == 0 else "codriving"
        group_id = f"{model}|{'x'.join(map(str, width))}"
        graphs.append(
            {
                "group_id": group_id,
                "model": model,
                "width": width,
                "node_count": int(graph["node_count"]) + index,
                "conv_count": int(graph["conv_count"]) + index,
                "parameter_elements": int(graph["parameter_elements"]) + index * 32,
                "conv_macs": float(width[0] * width[1] * width[2]),
            }
        )
        for profile_index, profile in enumerate(profiles):
            for q_mode in ("fp16", "int8"):
                factor = 1.0 + index * 0.05 + profile_index * 0.1 + (0.03 if q_mode == "int8" else 0.0)
                latency = float(measurement["latency_ms"]) * factor
                rows.append(
                    {
                        "manifest_job_id": f"{group_id}|{profile['capability_profile_id']}|{q_mode}",
                        "group_id": group_id,
                        "model": model,
                        "width": width,
                        "capability_profile_id": profile["capability_profile_id"],
                        "dispatch_key": profile["dispatch_key"],
                        "q_mode": q_mode,
                        "latency_ms": latency,
                        "energy_j": float(measurement["energy_j"]) * factor,
                        "ap30": 0.88 - index * 0.005,
                        "ap50": 0.78 - index * 0.005,
                        "ap70": 0.68 - index * 0.01 - (0.005 if q_mode == "int8" else 0.0),
                        "terminal_status": "measured_success_gold",
                        "training_source": "initial_coldstart" if index < 4 else "online_feedback",
                    }
                )
    closure = {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "ranker_policy": "rejected_use_value_heads_only",
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 16, "online_feedback": 8},
        "frozen_holdout": {"groups": []},
        "paper_evidence": False,
    }
    registry_groups: list[dict[str, Any]] = []
    for index in range(6, 10):
        width = [16 + index * 8, 32 + index * 8, 64 + index * 16]
        # Stage5/7 bind candidate identities to the published pyramid width schema.
        group_id = "pyramid|" + "x".join(map(str, width))
        contract = {
            "schema_version": "stage5_source_contract_v1",
            "group_id": group_id,
            "model": "pyramid",
            "width": width,
            "artifact_id": f"demo-candidate-{index}",
            "source_status": "ready",
            "source_evidence_sha256": _sha256_bytes(f"demo:{group_id}".encode()),
            "materialization_scope": "public_demo_only",
        }
        registry_groups.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "source_status": "ready",
                "source_evidence_sha256": contract["source_evidence_sha256"],
                "source_contract": contract,
                "source_contract_sha256": _canonical_sha(contract),
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": width,
                    "conv_count": 10 + index,
                    "conv_macs": float(width[0] * width[1] * width[2]),
                },
            }
        )
    stage5_task = {
        "task_id": "S5-PYR-TVM",
        "target_model": "pyramid",
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm",
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "round_index": 0,
        "paper_evidence": False,
    }
    stage6_measurements = [
        {
            "evidence_id": f"demo-stage6-{index}",
            "model": "demo-pyramid",
            "backend": "demo-selection-only",
            "method": "original_default" if index == 0 else "co_design",
            "AP70": 80.0 - index * 0.02,
            "latency_ms": 20.0 - index * 3.0,
            "energy_j": 8.0 - index,
            "terminal_status": "measured_success_gold",
            "independent_validation_passed": True,
            "evidence_sha_verified": True,
            "paper_evidence": False,
            "intended_use": "smoke_test_only",
        }
        for index in range(3)
    ]
    stage6_evidence = [
        {
            "evidence_id": row["evidence_id"],
            "sha256_status": "verified",
            "paper_evidence": False,
            "intended_use": "smoke_test_only",
        }
        for row in stage6_measurements
    ]
    stage6_cells = [
        {
            "model": "demo-pyramid",
            "backend": "demo-selection-only",
            "method": "co_design",
            "status": "complete",
            "paper_evidence": False,
            "intended_use": "smoke_test_only",
        }
    ]
    stage7_task = {
        "task_id": "S7-PYR-TVM",
        "target_model": "pyramid",
        "hardware_id": "h800",
        "capability_profile": profiles[0],
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
    }
    stage7_search_task = SearchTask(
        "S7-PYR-TVM", "pyramid", "h800", profiles[0], sample_budget=16, batch_size=4, round_count=4
    )
    stage7_registry = {"schema_version": "stage5_candidate_source_registry_v1", "groups": registry_groups}
    stage7_rows = build_task_candidate_manifest(stage7_registry, task=stage7_search_task, measured_row_ids=set())["rows"]
    for index, row in enumerate(stage7_rows):
        latency = 1.0 + index / 10
        row["predictions"] = {"latency_ms": latency, "energy_j": latency / 4, "ap70": 0.8 - index / 100}
        row["prediction_intervals"] = {
            key: {"lower": value - 0.1, "median": value, "upper": value + 0.1}
            for key, value in row["predictions"].items()
        }
    payloads: dict[str, Mapping[str, Any] | list[Any]] = {
        "measurements.jsonl": rows,
        "graph_features.jsonl": graphs,
        "capability_profiles.jsonl": profiles,
        "stage4_closure.json": closure,
        "candidate_registry.json": {"schema_version": "stage5_candidate_source_registry_v1", "groups": registry_groups},
        "stage5_task.json": stage5_task,
        "stage6_measurements.json": stage6_measurements,
        "stage6_evidence.json": stage6_evidence,
        "stage6_cells.json": stage6_cells,
        "stage7_task.json": stage7_task,
        "stage7_candidates.json": stage7_rows,
        "stage7_selected_ids.json": [],
        "stage7_measured_rows.json": [],
        "stage7_measured_graphs.json": [],
    }
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = contract_root / name
        data = (
            b"".join(_render_jsonl_record(item) for item in payload)
            if name.endswith(".jsonl")
            else _render_json(payload)
        )
        _write_new(path, data)
        paths[name] = path
    return paths


def _stage_record(
    name: str, inputs: Sequence[Path], outputs: Sequence[Path], *, output_root: Path
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "completed",
        "inputs": [{"path": _public_path(path, output_root), "sha256": _sha256_file(path)} for path in inputs],
        "outputs": [{"path": _public_path(path, output_root), "sha256": _sha256_file(path)} for path in outputs],
    }


def _all_output_hashes(root: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "run_manifest.json"
    ]


def _smoke(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    _required_repro_dependencies()
    demo = _demo_inputs()
    stage2_root = root / "stage2_demo"
    _run_script("stage2 demo generation", ROOT / "scripts" / "prepare_stage2_demo_data.py", ["--output-root", str(stage2_root)])
    contracts = _write_contracts(root, demo)
    stages = [
        _stage_record(
            "stage2_demo_generation",
            [DEMO_MANIFEST],
            [path for path in sorted(stage2_root.rglob("*")) if path.is_file() and path.name != ".stage2_demo_v1.json"],
            output_root=root,
        ),
        _stage_record("demo_contract", [DEMO_MANIFEST], list(contracts.values()), output_root=root),
    ]
    stage4_root = root / "stage4"
    _run_script(
        "stage4 selection",
        ROOT / "scripts" / "reproduce" / "cost_model_selection.py",
        [
            "--measurements", str(contracts["measurements.jsonl"]), "--graph-features", str(contracts["graph_features.jsonl"]),
            "--capability-profiles", str(contracts["capability_profiles.jsonl"]),
            "--measurements-provenance", "demo-measurements", "--graph-features-provenance", "demo-graphs",
            "--capability-profiles-provenance", "demo-profiles", "--output-root", str(stage4_root),
        ],
    )
    stages.append(_stage_record("stage4_selection", [contracts[name] for name in ("measurements.jsonl", "graph_features.jsonl", "capability_profiles.jsonl")], [stage4_root / "report.json", stage4_root / "folds.csv", stage4_root / "manifest.json"], output_root=root))
    stage5_root = root / "stage5"
    _run_script(
        "stage5 selection",
        ROOT / "scripts" / "reproduce" / "stage5_selection.py",
        [
            "--measurements", str(contracts["measurements.jsonl"]), "--measurements-provenance", "demo-measurements",
            "--graph-features", str(contracts["graph_features.jsonl"]), "--graph-features-provenance", "demo-graphs",
            "--capability-profiles", str(contracts["capability_profiles.jsonl"]), "--capability-profiles-provenance", "demo-profiles",
            "--stage4-closure", str(contracts["stage4_closure.json"]), "--stage4-closure-provenance", "demo-closure",
            "--candidate-registry", str(contracts["candidate_registry.json"]), "--candidate-registry-provenance", "demo-registry",
            "--search-task", str(contracts["stage5_task.json"]), "--search-task-provenance", "demo-task",
            "--seed", str(FROZEN_SEEDS["stage5"]), "--output-root", str(stage5_root),
        ],
    )
    stages.append(_stage_record("stage5_selection", [contracts[name] for name in ("measurements.jsonl", "graph_features.jsonl", "capability_profiles.jsonl", "stage4_closure.json", "candidate_registry.json", "stage5_task.json")], [stage5_root / "selected_ids.json", stage5_root / "measurement_request.json", stage5_root / "manifest.json"], output_root=root))
    stage6_root = root / "stage6"
    _run_script(
        "stage6 representative selection", ROOT / "scripts" / "reproduce" / "stage6_table.py",
        [
            "--measurements", str(contracts["stage6_measurements.json"]), "--measurements-provenance", "demo-selection",
            "--evidence", str(contracts["stage6_evidence.json"]), "--evidence-provenance", "demo-evidence",
            "--cells", str(contracts["stage6_cells.json"]), "--cells-provenance", "demo-cells", "--output-root", str(stage6_root),
        ],
    )
    stages.append(_stage_record("stage6_representative_selection", [contracts[name] for name in ("stage6_measurements.json", "stage6_evidence.json", "stage6_cells.json")], [stage6_root / "paper_table.raw.json", stage6_root / "paper_table.csv", stage6_root / "paper_table.md", stage6_root / "manifest.json"], output_root=root))
    stage7_output = root / "stage7" / "selection.json"
    _run_script(
        "stage7 selection", ROOT / "scripts" / "reproduce" / "stage7_selection.py",
        [
            "select", "--variant", "full", "--seed", str(FROZEN_SEEDS["stage7"]), "--round", "0",
            "--candidates", str(contracts["stage7_candidates.json"]), "--selected-ids", str(contracts["stage7_selected_ids.json"]),
            "--search-task", str(contracts["stage7_task.json"]), "--measured-rows", str(contracts["stage7_measured_rows.json"]),
            "--measured-graph-features", str(contracts["stage7_measured_graphs.json"]), "--output-json", str(stage7_output),
        ],
    )
    stages.append(_stage_record("stage7_selection_only", [contracts[name] for name in ("stage7_candidates.json", "stage7_selected_ids.json", "stage7_task.json", "stage7_measured_rows.json", "stage7_measured_graphs.json")], [stage7_output], output_root=root))
    return stages, demo["input_hashes"]


def _verified(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    manifest = _read_json(VERIFIED_MANIFEST, label="verified artifact manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "aaai27_verified_artifact_manifest_v1":
        raise PublicReproductionError("verified artifact manifest is unavailable")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise PublicReproductionError("verified artifact manifest is unavailable")
    inputs = [
        {
            "artifact_id": "verified-artifact-manifest",
            "path": _relative(VERIFIED_MANIFEST),
            "sha256": _sha256_file(VERIFIED_MANIFEST),
            "schema": str(manifest["schema"]),
            "verification_status": "verified_manifest",
        }
    ]
    stage4_inputs: list[dict[str, str]] = []
    artifact_ids: set[str] = set()
    for entry in entries:
        required = {"artifact_id", "path", "sha256", "schema", "verification_status"}
        if not isinstance(entry, Mapping) or not required <= set(entry):
            raise PublicReproductionError("verified artifact manifest is unavailable")
        artifact_id, relative_path, digest = entry["artifact_id"], entry["path"], entry["sha256"]
        schema, status = entry["schema"], entry["verification_status"]
        if (
            not isinstance(artifact_id, str)
            or artifact_id in artifact_ids
            or not isinstance(relative_path, str)
            or not isinstance(schema, str)
            or not schema
            or not isinstance(status, str)
            or not status
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or VERIFIED_ARTIFACT_CONTRACTS.get(artifact_id) != (schema, status)
        ):
            raise PublicReproductionError("verified artifact manifest is unavailable")
        artifact_ids.add(artifact_id)
        path = (ROOT / relative_path).resolve()
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(ROOT) or _sha256_file(path) != digest:
            raise PublicReproductionError("verified artifact manifest is unavailable")
        metadata = {
            "artifact_id": artifact_id,
            "path": _relative(path),
            "sha256": digest,
            "schema": schema,
            "verification_status": status,
        }
        inputs.append(metadata)
        if artifact_id.startswith("stage4-"):
            stage4_inputs.append(metadata)
    stages = [
        {
            "name": "verified_artifact_audit",
            "status": "completed",
            "inputs": inputs,
            "outputs": [],
        },
        {
            "name": "stage4_verified_analysis",
            "status": "completed" if stage4_inputs else "unavailable",
            "inputs": stage4_inputs,
            "outputs": [],
        },
        {"name": "stage6_representative_selection", "status": "unavailable", "inputs": [], "outputs": []},
        {"name": "stage7_formal_aggregate", "status": "unavailable", "inputs": [], "outputs": []},
    ]
    if not any(artifact_id.startswith("stage6-") for artifact_id in artifact_ids) or not any(artifact_id.startswith("stage7-") for artifact_id in artifact_ids):
        raise ReproductionUnavailable(
            "verified Stage6 evidence or Stage7 formal aggregate unavailable",
            stages=stages,
            inputs=inputs,
        )
    return stages, inputs


def _manifest(*, mode: str, status: str, started_at: str, stages: Sequence[Mapping[str, Any]], inputs: Sequence[Mapping[str, str]], root: Path, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "paper_evidence": mode == "verified",
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "component_versions": _versions(),
        "component_sources": _component_sources(),
        "frozen_seeds": FROZEN_SEEDS,
        "execution": {
            "network": False,
            "gpu": False,
            "hardware_execution": False,
            "cache_execution": False,
            "tvm_or_ap_execution": False,
        },
        "inputs": list(inputs),
        "stages": list(stages),
        "outputs": _all_output_hashes(root),
        "idempotency": "A matching completed manifest is reused; different existing content is refused.",
    }
    if error is not None:
        payload["error"] = error
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started_at = _utc_now()
    root: Path | None = None
    claim: OutputRootClaim | None = None
    try:
        root = _safe_output_root(args.output_root)
        claim = _claim_output_root(root, args.mode)
        if claim.existing_manifest is not None:
            status = claim.existing_manifest.get("status")
            if status == "completed":
                print(f"reproduction completed: {args.mode}")
                return 0
            print(
                f"error: {claim.existing_manifest.get('error', 'reproduction unavailable')}",
                file=sys.stderr,
            )
            return 2
        stages, inputs = _smoke(root) if args.mode == "smoke" else _verified(root)
        manifest = _manifest(mode=args.mode, status="completed", started_at=started_at, stages=stages, inputs=inputs, root=root)
        _atomic_write(root / "run_manifest.json", _render_json(manifest))
        print(f"reproduction completed: {args.mode}")
        return 0
    except ReproductionUnavailable as exc:
        if root is not None and claim is not None and claim.writable:
            failure = _manifest(
                mode=args.mode,
                status="unavailable",
                started_at=started_at,
                stages=exc.stages,
                inputs=exc.inputs,
                root=root,
                error=str(exc),
            )
            _atomic_write(root / "run_manifest.json", _render_json(failure))
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PublicReproductionError as exc:
        if root is not None and claim is not None and claim.writable:
            failure = _manifest(mode=args.mode, status="unavailable", started_at=started_at, stages=[], inputs=[], root=root, error=str(exc))
            _atomic_write(root / "run_manifest.json", _render_json(failure))
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, OSError, TypeError, ValueError):
        if root is not None and claim is not None and claim.writable:
            failure = _manifest(mode=args.mode, status="unavailable", started_at=started_at, stages=[], inputs=[], root=root, error="reproduction unavailable")
            _atomic_write(root / "run_manifest.json", _render_json(failure))
        print("error: reproduction unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
