#!/usr/bin/env python3
"""Build evidence-conservative Stage6 paper-table artifacts from explicit inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6 import paper_table_v1 as selection  # noqa: E402


FROZEN_SOURCE_SHA256 = "a929e6bf58637f5ee74a7232f2a923474b9ed64c8623d0f53a61bd5d967e7b58"
OUTPUT_NAMES = ("paper_table.raw.json", "paper_table.csv", "paper_table.md", "manifest.json")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("measurements", "evidence", "cells"):
        option = "--" + name
        parser.add_argument(option, required=True, help=f"Explicit {name} JSON or JSONL input.")
        parser.add_argument(
            f"{option}-provenance",
            required=True,
            help=f"Non-path public provenance label for {name}.",
        )
    parser.add_argument("--output-root", required=True, help="Explicit caller-owned artifact directory.")
    return parser.parse_args(argv)


def _safe_path(value: str, *, label: str, require_file: bool) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved:
        raise selection.PublicInputError(f"{label} must not traverse a symlink or parent escape")
    if resolved in {Path("/"), ROOT.resolve()}:
        raise selection.PublicInputError(f"{label} must not be the filesystem or repository root")
    if require_file:
        if not resolved.is_file() or resolved.is_symlink():
            raise selection.PublicInputError(f"{label} must be an existing regular file")
    elif resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise selection.PublicInputError(f"{label} must be a directory")
    return resolved


def _safe_output_root(value: str) -> Path:
    root = _safe_path(value, label="output-root", require_file=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _provenance(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized or "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
        raise selection.PublicInputError(f"{label} must be a non-path provenance label")
    return normalized


def _load_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise selection.PublicInputError(f"unable to read {label}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        records = value
    elif value is not None:
        raise selection.PublicInputError(f"{label} must be a JSON array or JSONL objects")
    else:
        records = []
        for line in text.splitlines():
            if not line.strip():
                raise selection.PublicInputError(f"{label} must contain JSON objects")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise selection.PublicInputError(f"{label} must contain valid JSON") from exc
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise selection.PublicInputError(f"{label} must contain JSON objects")
    return [dict(record) for record in records]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _input_metadata(path: Path, provenance: str) -> dict[str, str]:
    return {"sha256": _sha256(path.read_bytes()), "provenance": provenance}


def _evidence_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        identity = record.get("evidence_id")
        status = record.get("sha256_status")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(status, str):
            raise selection.PublicInputError("evidence has missing required fields")
        if identity in result:
            raise selection.PublicInputError("duplicate evidence_id in evidence")
        result[identity] = dict(record)
    return result


def _bind_evidence(
    measurements: Sequence[Mapping[str, Any]], evidence: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    for row in measurements:
        identity = row.get("evidence_id")
        evidence_row = evidence.get(identity) if isinstance(identity, str) else None
        if evidence_row is None:
            raise selection.PublicInputError("measurements reference undeclared evidence")
        copied = dict(row)
        copied["evidence_sha_verified"] = (
            copied.get("evidence_sha_verified") is True
            and evidence_row.get("sha256_status") == "verified"
        )
        bound.append(copied)
    return bound


def _render_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _rows(report: Mapping[str, Any], measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in report["cells"]:
        cell = result["cell"]
        matching_ids = sorted(
            str(row["evidence_id"])
            for row in measurements
            if row.get("model") == cell["model"] and row.get("backend") == cell["backend"]
            and row.get("method") in {"original_default", cell["method"]}
        )
        representative = result["representative"]
        records.append(
            {
                "model": cell["model"],
                "backend": cell["backend"],
                "method": cell["method"],
                "status": result["status"],
                "outcome": result["outcome"],
                "delta_ap_max": result.get("delta_ap_max", selection.PUBLIC_DELTA_AP70),
                "ap70_floor": result.get("ap70_floor"),
                "AP70": representative.get("AP70") if representative else None,
                "latency_ms": representative.get("latency_ms") if representative else None,
                "energy_j": representative.get("energy_j") if representative else None,
                "evidence_id": representative.get("evidence_id") if representative else None,
                "input_evidence_ids": matching_ids,
            }
        )
    return records


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "model", "backend", "method", "status", "outcome", "delta_ap_max", "ap70_floor",
        "AP70", "latency_ms", "energy_j", "evidence_id", "input_evidence_ids",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = dict(row)
        rendered["input_evidence_ids"] = ";".join(row["input_evidence_ids"])
        writer.writerow(rendered)
    return handle.getvalue().encode("utf-8")


def _markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_markdown(rows: Sequence[Mapping[str, Any]]) -> bytes:
    columns = ("model", "backend", "method", "status", "AP70", "latency_ms", "energy_j", "evidence_id")
    lines = [
        "| Model | Backend | Method | Status | AP70 | Latency (ms) | Energy (J) | Evidence ID |",
        "|:--|:--|:--|:--|--:|--:|--:|:--|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row[column]) for column in columns) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _write_artifact(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise selection.PublicInputError(f"refusing to overwrite non-file output: {path.name}")
    if path.exists():
        if path.read_bytes() != data:
            raise selection.PublicInputError(f"refusing to overwrite different output: {path.name}")
        return
    path.write_bytes(data)


def _preflight_artifacts(output_root: Path, artifacts: Mapping[str, bytes]) -> None:
    for name, data in artifacts.items():
        path = output_root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise selection.PublicInputError(f"refusing to overwrite non-file output: {name}")
        if path.exists() and path.read_bytes() != data:
            raise selection.PublicInputError(f"refusing to overwrite different output: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    measurement_path = _safe_path(args.measurements, label="measurements", require_file=True)
    evidence_path = _safe_path(args.evidence, label="evidence", require_file=True)
    cell_path = _safe_path(args.cells, label="cells", require_file=True)
    output_root = _safe_output_root(args.output_root)
    provenances = {
        "measurements": _provenance(args.measurements_provenance, label="measurements-provenance"),
        "evidence": _provenance(args.evidence_provenance, label="evidence-provenance"),
        "cells": _provenance(args.cells_provenance, label="cells-provenance"),
    }
    measurements = _load_records(measurement_path, label="measurements")
    evidence = _evidence_by_id(_load_records(evidence_path, label="evidence"))
    cells = _load_records(cell_path, label="cells")
    report = selection.select_representative_points(_bind_evidence(measurements, evidence), cells)
    output_rows = _rows(report, measurements)
    raw = _render_json(report)
    rendered_csv = _render_csv(output_rows)
    markdown = _render_markdown(output_rows)
    manifest = {
        "schema_version": "stage6_paper_table_manifest_v1",
        "inputs": {
            "measurements": _input_metadata(measurement_path, provenances["measurements"]),
            "evidence": _input_metadata(evidence_path, provenances["evidence"]),
            "cells": _input_metadata(cell_path, provenances["cells"]),
        },
        "source": {
            "frozen_source_sha256": FROZEN_SOURCE_SHA256,
            "adapted_module_sha256": _sha256(Path(selection.__file__).read_bytes()),
            "adaptation": "validated public adapter, stable identity ordering, and reproducible renderers",
        },
        "dependency_versions": _dependency_versions(),
        "output_rows": [
            {"evidence_id": row["evidence_id"], "input_evidence_ids": row["input_evidence_ids"], "status": row["status"]}
            for row in output_rows
        ],
        "outputs": {
            "paper_table.raw.json": {"sha256": _sha256(raw)},
            "paper_table.csv": {"sha256": _sha256(rendered_csv)},
            "paper_table.md": {"sha256": _sha256(markdown)},
        },
    }
    manifest["manifest_body_sha256"] = _sha256(_render_json(manifest))
    artifacts = {
        "paper_table.raw.json": raw,
        "paper_table.csv": rendered_csv,
        "paper_table.md": markdown,
        "manifest.json": _render_json(manifest),
    }
    _preflight_artifacts(output_root, artifacts)
    for name in OUTPUT_NAMES:
        _write_artifact(output_root / name, artifacts[name])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    try:
        args = _parse_args(argv)
        run(args)
    except selection.PublicInputError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
