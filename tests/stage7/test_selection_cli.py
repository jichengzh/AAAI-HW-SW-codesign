"""CLI boundary tests for published Stage7 selection/statistics only."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "reproduce" / "stage7_selection.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage7_selection_cli", CLI)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load Stage7 CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT, text=True, capture_output=True, check=False)


def test_select_requires_explicit_files_and_is_idempotent(tmp_path: Path) -> None:
    """Catches implicit input discovery or overwriting a prior different artifact."""
    candidates = tmp_path / "candidates.json"
    selected = tmp_path / "selected.json"
    output = tmp_path / "request.json"
    candidates.write_text(json.dumps([
        {"row_id": f"candidate-{index}", "score": index, "width": [16, 32, 64], "q_mode": "fp16"}
        for index in range(5)
    ]), encoding="utf-8")
    selected.write_text("[]", encoding="utf-8")
    command = ("select", "--variant", "full", "--seed", "20260718", "--round", "0", "--candidates", str(candidates), "--selected-ids", str(selected), "--output-json", str(output))

    first = _run(*command)
    second = _run(*command)

    assert first.returncode == second.returncode == 0, first.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["measurement_request"]["request_identity"]
    assert len(payload["measurement_request"]["rows"]) == 4


def test_summarize_fails_closed_when_terminal_matrix_is_incomplete(tmp_path: Path) -> None:
    """Catches a summary that substitutes missing trajectories or terminal evidence."""
    root = tmp_path / "incomplete"
    root.mkdir()
    (root / "events.jsonl").write_text(json.dumps({"variant": "full"}) + "\n", encoding="utf-8")
    output = tmp_path / "summary.json"

    result = _run("summarize", "--trajectory-root", str(root), "--output-json", str(output))

    assert result.returncode != 0
    assert not output.exists()


def test_cli_module_runs_selection_and_rejects_unsafe_output(tmp_path: Path) -> None:
    """Catches a CLI adapter bypassing its path or JSON validation when imported."""
    cli = _module()
    candidates = tmp_path / "candidates.json"
    selected = tmp_path / "selected.json"
    output = tmp_path / "direct.json"
    candidates.write_text(json.dumps([
        {"row_id": f"row-{index}", "score": index, "width": [16, 32, 64], "q_mode": "int8"}
        for index in range(4)
    ]), encoding="utf-8")
    selected.write_text("[]", encoding="utf-8")

    args = cli._parse_args([
        "select", "--variant", "without_surrogate", "--seed", "20260719", "--round", "0",
        "--candidates", str(candidates), "--selected-ids", str(selected), "--output-json", str(output),
    ])
    result = cli.run(args)

    assert result["acquisition"]["policy"] == "uniform_random_without_replacement"
    with pytest.raises(cli.PublicInputError, match="safe explicit"):
        cli._safe_output_path(str(ROOT))


def test_cli_module_summarizes_complete_jsonl_matrix(tmp_path: Path) -> None:
    """Catches the imported CLI skipping JSONL parsing or closed-matrix validation."""
    cli = _module()
    trajectory_root = tmp_path / "trajectory"
    trajectory_root.mkdir()
    events = []
    for variant in ("full", "without_surrogate", "without_measured_feedback", "backend_blind"):
        for seed in (20260718, 20260719, 20260720):
            for event_index in range(16):
                events.append({
                    "variant": variant, "seed": seed, "round_index": event_index // 4,
                    "event_index": event_index, "terminal_status": "completed", "delta_hv": event_index,
                })
    (trajectory_root / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    output = tmp_path / "summary.json"

    report = cli.run(cli._parse_args([
        "summarize", "--trajectory-root", str(trajectory_root), "--output-json", str(output),
    ]))

    assert report["trajectory_count"] == 12
    assert json.loads(output.read_text(encoding="utf-8"))["statistics"]["seed_count"] == 3


def test_cli_module_accepts_explicit_a2_feedback_and_frozen_bundle(tmp_path: Path) -> None:
    """Catches later A2 CLI rounds dropping an explicit frozen or feedback input."""
    cli = _module()
    candidates = tmp_path / "candidates.json"
    selected = tmp_path / "selected.json"
    initial_output = tmp_path / "initial.json"
    candidates.write_text(json.dumps([
        {"row_id": f"row-{index}", "score": index, "width": [16, 32, 64], "q_mode": "fp16"}
        for index in range(8)
    ]), encoding="utf-8")
    selected.write_text("[]", encoding="utf-8")
    initial = cli.run(cli._parse_args([
        "select", "--variant", "without_measured_feedback", "--seed", "20260718", "--round", "0",
        "--candidates", str(candidates), "--selected-ids", str(selected), "--output-json", str(initial_output),
    ]))
    selected.write_text(json.dumps(initial["acquisition"]["selected_row_ids"]), encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    feedback = tmp_path / "feedback.json"
    frozen.write_text(json.dumps(initial["a2_frozen"]), encoding="utf-8")
    feedback.write_text(json.dumps([
        {"row_id": row_id, "request_identity": initial["measurement_request"]["request_identity"], "terminal_status": "completed"}
        for row_id in initial["acquisition"]["selected_row_ids"]
    ]), encoding="utf-8")

    later = cli.run(cli._parse_args([
        "select", "--variant", "without_measured_feedback", "--seed", "20260718", "--round", "1",
        "--candidates", str(candidates), "--selected-ids", str(selected), "--feedback", str(feedback),
        "--previous-request", str(initial_output), "--a2-frozen", str(frozen),
        "--a2-frozen-sha256", initial["a2_frozen"]["frozen_payload_sha256"], "--output-json", str(tmp_path / "later.json"),
    ]))

    assert later["a2_frozen"] == initial["a2_frozen"]
    feedback.write_text(json.dumps([
        {"row_id": row_id, "request_identity": "0" * 64, "terminal_status": "completed"}
        for row_id in initial["acquisition"]["selected_row_ids"]
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        cli.run(cli._parse_args([
            "select", "--variant", "without_measured_feedback", "--seed", "20260718", "--round", "1",
            "--candidates", str(candidates), "--selected-ids", str(selected), "--feedback", str(feedback),
            "--previous-request", str(initial_output), "--a2-frozen", str(frozen),
            "--a2-frozen-sha256", initial["a2_frozen"]["frozen_payload_sha256"], "--output-json", str(tmp_path / "bad.json"),
        ]))
    cli._write_idempotent(initial_output, initial_output.read_bytes())
    with pytest.raises(cli.PublicInputError, match="different output"):
        cli._write_idempotent(initial_output, b"different")
    with pytest.raises(SystemExit) as exited:
        cli.main(["summarize", "--trajectory-root", str(tmp_path / "missing"), "--output-json", str(tmp_path / "never.json")])
    assert exited.value.code == 2
