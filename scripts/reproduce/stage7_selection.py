#!/usr/bin/env python3
"""Publish Stage7 selection requests and descriptive summaries from explicit files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage7 import ablation_statistics_v2 as statistics  # noqa: E402
from framework.stage7 import search_policy_v1 as policy  # noqa: E402


class PublicInputError(ValueError):
    """A validation failure whose text is safe to display from the CLI."""


def _safe_path(value: str, *, label: str, require_file: bool) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved or resolved in {Path("/"), ROOT.resolve()}:
        raise PublicInputError(f"{label} must be a safe explicit path")
    if require_file:
        if not resolved.is_file() or resolved.is_symlink():
            raise PublicInputError(f"{label} must be an existing regular file")
    elif resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise PublicInputError(f"{label} must be a directory")
    return resolved


def _safe_output_path(value: str) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved or resolved in {Path("/"), ROOT.resolve()}:
        raise PublicInputError("output-json must be a safe explicit path")
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_file()):
        raise PublicInputError("output-json must be a regular file path")
    return resolved


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicInputError(f"{label} must contain valid JSON") from exc


def _load_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicInputError(f"unable to read {label}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        values = parsed
    elif parsed is None:
        values = []
        for line in raw.splitlines():
            if not line.strip():
                raise PublicInputError(f"{label} must contain JSON objects")
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PublicInputError(f"{label} must contain valid JSON") from exc
    else:
        raise PublicInputError(f"{label} must be a JSON array or JSONL objects")
    if not isinstance(values, list) or any(not isinstance(value, Mapping) for value in values):
        raise PublicInputError(f"{label} must contain JSON objects")
    return [dict(value) for value in values]


def _render(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_idempotent(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PublicInputError("refusing to overwrite non-file output")
    if path.exists():
        if path.read_bytes() != data:
            raise PublicInputError("refusing to overwrite different output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _stage7_task(path: Path) -> Any:
    payload = _load_json(path, label="search-task")
    required = {
        "task_id", "target_model", "hardware_id", "capability_profile",
        "sample_budget", "batch_size", "round_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PublicInputError("search-task must contain exactly the Stage7 task fields")
    if (
        payload.get("task_id") != "S7-PYR-TVM"
        or payload.get("target_model") != "pyramid"
        or payload.get("hardware_id") != "h800"
        or (payload.get("sample_budget"), payload.get("batch_size"), payload.get("round_count")) != (16, 4, 4)
        or not isinstance(payload.get("capability_profile"), Mapping)
    ):
        raise PublicInputError("search-task is not the frozen Stage7 task")
    try:
        return policy.build_stage7_task(dict(payload["capability_profile"]))
    except (TypeError, ValueError) as exc:
        raise PublicInputError("search-task capability profile is invalid") from exc


def _select(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _load_rows(_safe_path(args.candidates, label="candidates", require_file=True), label="candidates")
    task = _stage7_task(_safe_path(args.search_task, label="search-task", require_file=True))
    measured_rows = _load_rows(_safe_path(args.measured_rows, label="measured-rows", require_file=True), label="measured-rows")
    measured_graph_features = _load_rows(_safe_path(args.measured_graph_features, label="measured-graph-features", require_file=True), label="measured-graph-features")
    selected_payload = _load_json(_safe_path(args.selected_ids, label="selected-ids", require_file=True), label="selected-ids")
    if not isinstance(selected_payload, list) or any(not isinstance(value, str) for value in selected_payload):
        raise PublicInputError("selected-ids must be a JSON array of identities")
    feedback: list[dict[str, Any]] = []
    if args.feedback is not None:
        feedback = _load_rows(_safe_path(args.feedback, label="feedback", require_file=True), label="feedback")
    previous_request: Mapping[str, Any] | None = None
    if args.previous_request is not None:
        previous_payload = _load_json(_safe_path(args.previous_request, label="previous-request", require_file=True), label="previous-request")
        if not isinstance(previous_payload, Mapping):
            raise PublicInputError("previous-request must be a JSON object")
        candidate_request = previous_payload.get("measurement_request", previous_payload)
        if not isinstance(candidate_request, Mapping):
            raise PublicInputError("previous-request must contain a measurement request")
        previous_request = dict(candidate_request)
    frozen: Mapping[str, Any] | None = None
    if args.a2_frozen is not None:
        frozen_payload = _load_json(_safe_path(args.a2_frozen, label="a2-frozen", require_file=True), label="a2-frozen")
        if not isinstance(frozen_payload, Mapping):
            raise PublicInputError("a2-frozen must be a JSON object")
        frozen = dict(frozen_payload)
    blind_bundle: Mapping[str, Any] | None = None
    if args.blind_prediction_bundle is not None:
        blind_payload = _load_json(_safe_path(args.blind_prediction_bundle, label="blind-prediction-bundle", require_file=True), label="blind-prediction-bundle")
        if not isinstance(blind_payload, Mapping):
            raise PublicInputError("blind-prediction-bundle must be a JSON object")
        blind_bundle = dict(blind_payload)
    if args.round > 0 and previous_request is None:
        raise PublicInputError("later rounds require explicit previous-request")
    if args.round > 0 and args.variant == "without_measured_feedback" and (frozen is None or args.a2_frozen_sha256 is None):
        raise PublicInputError("A2 later rounds require explicit frozen bundle and SHA identity")
    if args.variant == "backend_blind" and (blind_bundle is None or args.blind_prediction_bundle_sha256 is None):
        raise PublicInputError("backend-blind requires explicit blind prediction bundle and SHA identity")
    return policy.select_stage7_round(variant=args.variant, seed=args.seed, round_index=args.round, task=task, candidate_pool=candidates, measured_rows=measured_rows, measured_graph_features=measured_graph_features, selected_ids=set(selected_payload), feedback_rows=feedback, previous_measurement_request=previous_request, a2_frozen=frozen, expected_a2_frozen_sha256=args.a2_frozen_sha256, blind_prediction_bundle=blind_bundle, expected_blind_prediction_bundle_sha256=args.blind_prediction_bundle_sha256)


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = _safe_path(args.trajectory_root, label="trajectory-root", require_file=False)
    events_path = root / "events.jsonl"
    if not events_path.is_file() or events_path.is_symlink():
        raise PublicInputError("trajectory-root must contain explicit events.jsonl")
    trajectories = statistics.trajectory_summaries(_load_rows(events_path, label="events"))
    stats = statistics.paired_statistics(trajectories)
    return {"schema_version": "stage7_selection_summary_v1", "trajectory_count": len(trajectories), "statistics": stats, "paper_rows": statistics.paper_rows(stats)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Create one deterministic selection request.")
    select.add_argument("--variant", required=True, choices=statistics.CORE_VARIANTS)
    select.add_argument("--seed", required=True, type=int, choices=policy.SEEDS)
    select.add_argument("--round", required=True, type=int, choices=range(4))
    select.add_argument("--candidates", required=True)
    select.add_argument("--selected-ids", required=True)
    select.add_argument("--search-task", required=True)
    select.add_argument("--measured-rows", required=True)
    select.add_argument("--measured-graph-features", required=True)
    select.add_argument("--feedback")
    select.add_argument("--previous-request")
    select.add_argument("--a2-frozen")
    select.add_argument("--a2-frozen-sha256")
    select.add_argument("--blind-prediction-bundle")
    select.add_argument("--blind-prediction-bundle-sha256")
    select.add_argument("--output-json", required=True)
    summarize = commands.add_parser("summarize", help="Summarize a complete terminal-event matrix.")
    summarize.add_argument("--trajectory-root", required=True)
    summarize.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = _safe_output_path(args.output_json)
    report = _select(args) if args.command == "select" else _summarize(args)
    _write_idempotent(output, _render(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    try:
        run(_parse_args(argv))
    except (PublicInputError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
