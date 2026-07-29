#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate and archive a Stage2 evidence delta for Stage1 refresh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.contracts import (
    STAGE2_EVIDENCE_DELTA_SCHEMA,
    Stage2EvidenceDelta,
    Stage2EvidenceRecord,
)  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True, help="Stage2 evidence delta JSON.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Archive directory for validated Stage2 evidence deltas.",
    )
    return parser.parse_args()


def _load_and_validate(path: Path) -> Stage2EvidenceDelta:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence delta JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("evidence delta JSON must decode to an object")
    if data.get("schema") != STAGE2_EVIDENCE_DELTA_SCHEMA:
        raise ValueError(f"unexpected schema: {data.get('schema')}")
    records = [
        Stage2EvidenceRecord(
            backend=record["backend"],
            hardware=record["hardware"],
            scope=record["scope"],
            evidence_kind=record["evidence_kind"],
            provenance=record["provenance"],
            candidate_config=record["candidate_config"],
            metric=record["metric"],
        )
        for record in data.get("records", [])
    ]
    return Stage2EvidenceDelta(model=str(data.get("model", "unknown")), records=records)


def _safe_archive_path(out_dir: str | Path, filename: str) -> Path:
    requested = Path(out_dir).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved:
        raise ValueError("out-dir must not traverse a symlink or parent escape")
    if resolved == Path("/"):
        raise ValueError("out-dir must not be the filesystem root")
    resolved.mkdir(parents=True, exist_ok=True)
    output = resolved / filename
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError(f"refusing to write non-file archive output: {output}")
    return output


def main() -> int:
    args = _parse_args()
    delta_path = Path(args.delta).expanduser()
    delta = _load_and_validate(delta_path)
    out_path = _safe_archive_path(args.out_dir, f"{delta.model}_stage2_evidence_delta_v1.json")
    rendered = json.dumps(delta.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if out_path.exists():
        if out_path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"refusing to overwrite existing archive output: {out_path}")
    else:
        out_path.write_text(rendered, encoding="utf-8")
    print(
        f"stage2_update_evidence_ok model={delta.model} records={len(delta.records)} out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
