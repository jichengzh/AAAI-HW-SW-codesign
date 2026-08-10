from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/extract_p4_resource_candidates.py"


def _module():
    spec = importlib.util.spec_from_file_location("p4_resource_candidates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(path: str, disposition: str, text: str) -> dict[str, object]:
    return {
        "origin": "tracked",
        "path": path,
        "disposition": disposition,
        "actual_role": text,
        "inputs": text,
        "outputs": text,
        "evidence_artifact": text,
        "reason": "reviewed",
    }


def test_extract_p4_candidates_classifies_assets_documents_and_ambiguities(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"records": [
        _record("private/checkpoint.md", "external_contract_p4", "checkpoint input"),
        _record("private/paper.md", "external_contract_p4", "paper method discussion"),
        _record("private/mixed.md", "external_contract_p4", "dataset and ONNX export"),
        _record("private/p6.md", "execution_contract_p6", "checkpoint input"),
    ]}), encoding="utf-8")

    draft = _module().extract_candidate_draft(review)

    assert [item.suggested_decision for item in draft] == ["asset", "document", "ambiguous"]
    assert draft[0].suggested_asset_kinds == ("checkpoint",)
    assert draft[2].suggested_asset_kinds == ("dataset", "onnx")


def test_cli_writes_private_draft_without_echoing_candidate_path(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    draft = tmp_path / "draft.json"
    private_path = "private/checkpoint.md"
    review.write_text(
        json.dumps({"records": [_record(private_path, "external_contract_p4", "checkpoint input")]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--p3-review", str(review), "--output", str(draft)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "p4 candidate draft created" in result.stdout
    assert private_path not in result.stdout
    rendered = draft.read_text(encoding="utf-8")
    assert private_path in rendered
    assert json.loads(rendered)["format"] == "aaai27_p4_private_candidate_draft_v1"
