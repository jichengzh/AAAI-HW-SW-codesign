#!/usr/bin/env python3
"""Extract a local-only P4 candidate draft from a restricted P3 review."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


P4_DISPOSITION = "external_contract_p4"
PRIVATE_DRAFT_FORMAT = "aaai27_p4_private_candidate_draft_v1"

ASSET_TERMS = {
    "dataset": ("dataset", "data set", "数据集"),
    "model": ("model", "模型"),
    "checkpoint": ("checkpoint", "权重"),
    "onnx": ("onnx",),
    "engine": ("engine", "tensorrt", "tvm"),
    "measurement_bundle": ("latency", "energy", "telemetry", "measurement", "测量"),
    "evidence_bundle": ("evidence", "ap report", "证据"),
    "regenerable_large_file": ("lut", "cache", "artifact", "制品"),
}

_TEXT_FIELDS = ("actual_role", "inputs", "outputs", "evidence_artifact", "reason")


class CandidateDraftError(ValueError):
    """Raised when a local P3 review cannot produce a private draft."""


@dataclass(frozen=True)
class CandidateDraft:
    """A private candidate recommendation retained in the requested local output."""

    origin: str
    path: str
    suggested_decision: str
    suggested_asset_kinds: tuple[str, ...]
    matched_asset_terms: tuple[str, ...]


def _required_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise CandidateDraftError("invalid local P3 review")
    return value


def load_p4_records(path: Path) -> tuple[Mapping[str, object], ...]:
    """Load only P4-disposition records from a caller-supplied local review."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateDraftError("unable to load local P3 review") from error
    if not isinstance(document, Mapping) or not isinstance(document.get("records"), list):
        raise CandidateDraftError("invalid local P3 review")

    records: list[Mapping[str, object]] = []
    for record in document["records"]:
        if not isinstance(record, Mapping):
            raise CandidateDraftError("invalid local P3 review")
        if record.get("disposition") == P4_DISPOSITION:
            records.append(record)
    return tuple(records)


def suggest_candidate(record: Mapping[str, object]) -> CandidateDraft:
    """Classify one P4 record conservatively from its five textual fields."""
    origin = _required_string(record, "origin")
    path = _required_string(record, "path")
    text = " ".join(_required_string(record, field) for field in _TEXT_FIELDS).lower()
    suggested_asset_kinds = tuple(
        asset_kind
        for asset_kind, terms in ASSET_TERMS.items()
        if any(term in text for term in terms)
    )
    matched_asset_terms = tuple(
        term
        for terms in ASSET_TERMS.values()
        for term in terms
        if term in text
    )
    suggested_decision = (
        "document"
        if not suggested_asset_kinds
        else "asset"
        if len(suggested_asset_kinds) == 1
        else "ambiguous"
    )
    return CandidateDraft(
        origin=origin,
        path=path,
        suggested_decision=suggested_decision,
        suggested_asset_kinds=suggested_asset_kinds,
        matched_asset_terms=matched_asset_terms,
    )


def extract_candidate_draft(path: Path) -> tuple[CandidateDraft, ...]:
    """Return private P4 candidate recommendations for one local review."""
    return tuple(suggest_candidate(record) for record in load_p4_records(path))


def render_private_draft(draft: Sequence[CandidateDraft]) -> dict[str, object]:
    """Render a private local draft; its paths must never be copied to terminal output."""
    return {
        "format": PRIVATE_DRAFT_FORMAT,
        "candidates": [
            {
                "origin": item.origin,
                "path": item.path,
                "suggested_decision": item.suggested_decision,
                "suggested_asset_kinds": list(item.suggested_asset_kinds),
                "matched_asset_terms": list(item.matched_asset_terms),
            }
            for item in draft
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _write_private_draft(path: Path, draft: Sequence[CandidateDraft]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(render_private_draft(draft), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        draft = extract_candidate_draft(args.p3_review)
        _write_private_draft(args.output, draft)
    except CandidateDraftError:
        print("p4 candidate draft could not be created", file=sys.stderr)
        return 2

    counts = Counter(item.suggested_decision for item in draft)
    print(
        "p4 candidate draft created: "
        f"records={len(draft)} asset={counts['asset']} "
        f"document={counts['document']} ambiguous={counts['ambiguous']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
