#!/usr/bin/env python3
"""Run the preregistered Stage4 nested grouped cost-model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage4 import cost_model_selection_v1 as selection  # noqa: E402


MIGRATION_LINEAGE_MODULE_SHA256 = (
    "e93b42b0d659f1ecaa7ecbf6ba7e847c21da37922eec15e312f07074c414c784"
)
EXPECTED_SOURCE_SHA256 = "c8b32769cf1b2fd9b75c7fdb30c810237b071ae7d3c187dcbb887667dd8c3af9"
OUTPUT_NAMES = ("report.json", "folds.csv", "manifest.json")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, help="Measurement JSONL input.")
    parser.add_argument("--graph-features", required=True, help="Graph-feature JSONL input.")
    parser.add_argument(
        "--capability-profiles",
        required=True,
        help="Capability-profile JSONL input; never inferred from measurements.",
    )
    parser.add_argument(
        "--measurements-provenance",
        required=True,
        help="Non-path provenance label for the measurement input.",
    )
    parser.add_argument(
        "--graph-features-provenance",
        required=True,
        help="Non-path provenance label for the graph-feature input.",
    )
    parser.add_argument(
        "--capability-profiles-provenance",
        required=True,
        help="Non-path provenance label for the capability-profile input.",
    )
    parser.add_argument("--output-root", required=True, help="Explicit artifact directory.")
    parser.add_argument("--seed", type=int, default=20260716, help="Deterministic CV seed.")
    return parser.parse_args(argv)


def _safe_path(value: str, *, label: str, require_file: bool) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved:
        raise ValueError(f"{label} must not traverse a symlink or parent escape")
    if resolved in {Path("/"), ROOT.resolve()}:
        raise ValueError(f"{label} must not be the filesystem or repository root")
    if require_file:
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(f"{label} must be an existing regular file")
    elif resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise ValueError(f"{label} must be a directory")
    return resolved


def _provenance(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized or "/" in normalized or "\\" in normalized:
        raise ValueError(f"{label} must be a non-path provenance label")
    return normalized


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read {label}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {label} line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"{label} must contain at least one JSON object")
    return records


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _render_folds(report: Mapping[str, Any]) -> bytes:
    fields = (
        "target",
        "outer_fold",
        "train_groups",
        "test_groups",
        "train_rows",
        "test_rows",
        "selected_candidate",
        "inner_candidate_scores",
        "mae",
        "mape",
        "spearman",
    )
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for target in sorted(report["targets"]):
        for fold in report["targets"][target]["outer_folds"]:
            metrics = fold["metrics"]
            writer.writerow(
                {
                    "target": target,
                    "outer_fold": fold["outer_fold"],
                    "train_groups": ";".join(fold["train_groups"]),
                    "test_groups": ";".join(fold["test_groups"]),
                    "train_rows": fold["train_rows"],
                    "test_rows": fold["test_rows"],
                    "selected_candidate": fold["selected_candidate"],
                    "inner_candidate_scores": json.dumps(
                        fold["inner_candidate_scores"], sort_keys=True, separators=(",", ":")
                    ),
                    "mae": metrics["mae"],
                    "mape": metrics["mape"],
                    "spearman": metrics["spearman"],
                }
            )
    return handle.getvalue().encode("utf-8")


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
    return {name: importlib.metadata.version(name) for name in names}


def _verify_source_identity() -> str:
    current = _sha256(Path(selection.__file__).read_bytes())
    if current != EXPECTED_SOURCE_SHA256:
        raise ValueError("source identity verification failed")
    return current


def _safe_output_root(value: str) -> Path:
    root = _safe_path(value, label="output-root", require_file=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_artifact(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"refusing to overwrite non-file output: {path.name}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different output: {path.name}")
        return
    path.write_bytes(data)


def _preflight_artifacts(output_root: Path, artifacts: Mapping[str, bytes]) -> None:
    for name, data in artifacts.items():
        path = output_root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"refusing to overwrite non-file output: {name}")
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different output: {name}")


def _manifest(
    *,
    report: Mapping[str, Any],
    input_metadata: Mapping[str, Mapping[str, str]],
    report_sha256: str,
    folds_sha256: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "stage4_cost_model_selection_manifest_v1",
        "inputs": input_metadata,
        "seed": report["seed"],
        "row_count": report["row_count"],
        "group_count": report["group_count"],
        "dependency_versions": _dependency_versions(),
        "source": {
            "expected_module_sha256": EXPECTED_SOURCE_SHA256,
            "current_module_sha256": current_source_sha256,
            "migration_lineage_module_sha256": MIGRATION_LINEAGE_MODULE_SHA256,
            "migration_source": "framework/stage4/cost_model_selection_v1.py",
        },
        "outputs": {
            "report.json": report_sha256,
            "folds.csv": folds_sha256,
        },
    }
    # A file cannot contain a SHA256 of its own final bytes. This explicit
    # digest covers the complete manifest body before the self-digest field.
    payload["outputs"]["manifest.json"] = {
        "sha256": _sha256(_render_json(payload)),
        "sha256_scope": "manifest_body_before_self_digest",
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        current_source_sha256 = _verify_source_identity()
        measurements_path = _safe_path(args.measurements, label="measurements", require_file=True)
        graph_path = _safe_path(args.graph_features, label="graph-features", require_file=True)
        profiles_path = _safe_path(
            args.capability_profiles, label="capability-profiles", require_file=True
        )
        output_root = _safe_output_root(args.output_root)
        inputs = {
            "measurements": {
                "sha256": _sha256(measurements_path.read_bytes()),
                "provenance": _provenance(args.measurements_provenance, label="measurements provenance"),
            },
            "graph_features": {
                "sha256": _sha256(graph_path.read_bytes()),
                "provenance": _provenance(args.graph_features_provenance, label="graph-features provenance"),
            },
            "capability_profiles": {
                "sha256": _sha256(profiles_path.read_bytes()),
                "provenance": _provenance(
                    args.capability_profiles_provenance, label="capability-profiles provenance"
                ),
            },
        }
        report = selection.run_nested_selection(
            _load_jsonl(measurements_path, label="measurements"),
            _load_jsonl(graph_path, label="graph-features"),
            _load_jsonl(profiles_path, label="capability-profiles"),
            seed=args.seed,
        )
        report_bytes = _render_json(report)
        folds_bytes = _render_folds(report)
        manifest_bytes = _render_json(
            _manifest(
                report=report,
                input_metadata=inputs,
                report_sha256=_sha256(report_bytes),
                folds_sha256=_sha256(folds_bytes),
                current_source_sha256=current_source_sha256,
            )
        )
        artifacts = dict(zip(OUTPUT_NAMES, (report_bytes, folds_bytes, manifest_bytes)))
        _preflight_artifacts(output_root, artifacts)
        for name, data in artifacts.items():
            _write_artifact(output_root / name, data)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"stage4_cost_model_selection_ok rows={report['row_count']} groups={report['group_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
