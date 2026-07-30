"""CLI boundary tests for published Stage7 selection/statistics only."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from .test_search_policy import _candidates, _profile


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


def _selection_inputs(tmp_path: Path) -> tuple[str, ...]:
    candidates = tmp_path / "candidates.json"
    selected = tmp_path / "selected.json"
    task = tmp_path / "search-task.json"
    measured_rows = tmp_path / "measured-rows.json"
    measured_graphs = tmp_path / "measured-graphs.json"
    candidates.write_text(json.dumps(_candidates()), encoding="utf-8")
    selected.write_text("[]", encoding="utf-8")
    task.write_text(json.dumps({
        "task_id": "S7-PYR-TVM", "target_model": "pyramid", "hardware_id": "h800",
        "capability_profile": _profile(), "sample_budget": 16, "batch_size": 4, "round_count": 4,
    }), encoding="utf-8")
    measured_rows.write_text("[]", encoding="utf-8")
    measured_graphs.write_text("[]", encoding="utf-8")
    return (
        "--candidates", str(candidates), "--selected-ids", str(selected), "--search-task", str(task),
        "--measured-rows", str(measured_rows), "--measured-graph-features", str(measured_graphs),
    )


def test_select_requires_explicit_files_and_is_idempotent(tmp_path: Path) -> None:
    """Catches implicit input discovery or overwriting a prior different artifact."""
    output = tmp_path / "request.json"
    command = ("select", "--variant", "full", "--seed", "20260718", "--round", "0", *_selection_inputs(tmp_path), "--output-json", str(output))

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
    output = tmp_path / "direct.json"

    args = cli._parse_args([
        "select", "--variant", "without_surrogate", "--seed", "20260719", "--round", "0",
        *_selection_inputs(tmp_path), "--output-json", str(output),
    ])
    result = cli.run(args)

    assert result["acquisition"]["policy"] == "uniform_random_without_replacement"
    with pytest.raises(cli.PublicInputError, match="safe explicit"):
        cli._safe_output_path(str(ROOT))


def test_cli_backend_blind_requires_independent_bound_bundle_and_redacts_tampering(tmp_path: Path) -> None:
    """Catches backend-blind CLI selection trusting caller predictions or bundle rehashes."""
    cli = _module()
    inputs = _selection_inputs(tmp_path)
    task_path = tmp_path / "search-task.json"
    bundle = cli.policy.build_backend_blind_prediction_bundle(
        task=cli._stage7_task(task_path), candidate_pool=_candidates(), model_bundle_sha256="a" * 64,
    )
    bundle_path = tmp_path / "blind-bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    output = tmp_path / "blind.json"
    common = (
        "select", "--variant", "backend_blind", "--seed", "20260718", "--round", "0", *inputs,
    )

    missing = _run(*common, "--output-json", str(output))
    assert missing.returncode != 0
    assert "explicit blind prediction bundle" in missing.stderr

    first = _run(
        *common, "--blind-prediction-bundle", str(bundle_path),
        "--blind-prediction-bundle-sha256", bundle["bundle_sha256"], "--output-json", str(output),
    )
    assert first.returncode == 0, first.stderr
    expected_ids = json.loads(output.read_text(encoding="utf-8"))["acquisition"]["selected_row_ids"]

    altered_candidates = _candidates()
    for row in altered_candidates:
        row["predictions"]["latency_ms"] *= 1000
    (tmp_path / "candidates.json").write_text(json.dumps(altered_candidates), encoding="utf-8")
    altered_output = tmp_path / "blind-altered.json"
    altered = _run(
        *common, "--blind-prediction-bundle", str(bundle_path),
        "--blind-prediction-bundle-sha256", bundle["bundle_sha256"], "--output-json", str(altered_output),
    )
    assert altered.returncode == 0, altered.stderr
    assert json.loads(altered_output.read_text(encoding="utf-8"))["acquisition"]["selected_row_ids"] == expected_ids

    tampered = dict(bundle)
    tampered["predictions"] = [dict(row) for row in bundle["predictions"]]
    tampered["predictions"][0]["predictions"] = dict(tampered["predictions"][0]["predictions"])
    tampered["predictions"][0]["predictions"]["latency_ms"] *= 2
    unsigned = {key: value for key, value in tampered.items() if key != "bundle_sha256"}
    tampered["bundle_sha256"] = cli.policy.canonical_sha256(unsigned)
    bundle_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = _run(
        *common, "--blind-prediction-bundle", str(bundle_path),
        "--blind-prediction-bundle-sha256", bundle["bundle_sha256"], "--output-json", str(tmp_path / "tampered.json"),
    )
    assert rejected.returncode != 0
    assert "blind prediction bundle identity drift" in rejected.stderr
    assert "Traceback" not in rejected.stderr
    assert str(bundle_path) not in rejected.stderr


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
    initial_output = tmp_path / "initial.json"
    inputs = _selection_inputs(tmp_path)
    selected = tmp_path / "selected.json"
    initial = cli.run(cli._parse_args([
        "select", "--variant", "without_measured_feedback", "--seed", "20260718", "--round", "0",
        *inputs, "--output-json", str(initial_output),
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
        *inputs, "--feedback", str(feedback),
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
            *inputs, "--feedback", str(feedback),
            "--previous-request", str(initial_output), "--a2-frozen", str(frozen),
            "--a2-frozen-sha256", initial["a2_frozen"]["frozen_payload_sha256"], "--output-json", str(tmp_path / "bad.json"),
        ]))
    forged = tmp_path / "forged-request.json"
    forged_payload = json.loads(initial_output.read_text(encoding="utf-8"))
    forged_payload["measurement_request"]["rows"].reverse()
    forged.write_text(json.dumps(forged_payload), encoding="utf-8")
    rejected = _run(
        "select", "--variant", "without_measured_feedback", "--seed", "20260718", "--round", "1",
        *inputs, "--feedback", str(feedback), "--previous-request", str(forged), "--a2-frozen", str(frozen),
        "--a2-frozen-sha256", initial["a2_frozen"]["frozen_payload_sha256"], "--output-json", str(tmp_path / "forged-output.json"),
    )
    assert rejected.returncode != 0
    assert "Traceback" not in rejected.stderr
    assert str(forged) not in rejected.stderr
    assert "canonical" in rejected.stderr
    cli._write_idempotent(initial_output, initial_output.read_bytes())
    with pytest.raises(cli.PublicInputError, match="different output"):
        cli._write_idempotent(initial_output, b"different")
    with pytest.raises(SystemExit) as exited:
        cli.main(["summarize", "--trajectory-root", str(tmp_path / "missing"), "--output-json", str(tmp_path / "never.json")])
    assert exited.value.code == 2
