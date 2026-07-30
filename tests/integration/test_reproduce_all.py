"""Public demo/verified artifact boundary checks."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_MANIFEST_PATH = REPOSITORY_ROOT / "data/demo/manifest.json"
VERIFIED_MANIFEST_PATH = REPOSITORY_ROOT / "artifacts/verified/manifest.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PUBLIC_PATTERNS = (
    re.compile(r"/home/[^\s\"']+", re.IGNORECASE),
    re.compile(r"/Users/[^\s\"']+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
    re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
    re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,})\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    return []


def _assert_demo_manifest_is_smoke_only(manifest: dict[str, object]) -> None:
    assert manifest.get("paper_evidence") is False
    items = manifest.get("artifacts")
    assert isinstance(items, list) and items
    for item in items:
        assert isinstance(item, dict)
        assert item.get("paper_evidence") is False
        assert item.get("intended_use") == "smoke_test_only"


def _assert_verified_manifest_is_valid(manifest: dict[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list) and artifacts
    artifact_ids: set[str] = set()
    for item in artifacts:
        assert isinstance(item, dict)
        required = {
            "artifact_id",
            "path",
            "sha256",
            "schema",
            "provenance",
            "paper_mapping",
            "verification_status",
        }
        assert required <= item.keys()
        artifact_id = item["artifact_id"]
        assert isinstance(artifact_id, str) and artifact_id
        assert artifact_id not in artifact_ids
        artifact_ids.add(artifact_id)
        relative_path = item["path"]
        assert isinstance(relative_path, str)
        assert not Path(relative_path).is_absolute()
        assert SHA256_PATTERN.fullmatch(str(item["sha256"]))
        assert isinstance(item["schema"], str) and item["schema"]
        assert isinstance(item["provenance"], dict)
        assert isinstance(item["paper_mapping"], dict)
        assert isinstance(item["verification_status"], str) and item["verification_status"]
        assert not any(
            pattern.search(text)
            for text in _walk_strings(item["provenance"])
            for pattern in FORBIDDEN_PUBLIC_PATTERNS[:3]
        )


def test_demo_manifest_is_explicitly_non_evidence_smoke_data() -> None:
    """Demo inputs must never be represented as paper evidence."""
    _assert_demo_manifest_is_smoke_only(_read_json(DEMO_MANIFEST_PATH))


def test_demo_manifest_rejects_records_marked_as_paper_evidence() -> None:
    """A single evidence-marked demo record invalidates the demo boundary."""
    manifest = _read_json(DEMO_MANIFEST_PATH)
    invalid_manifest = copy.deepcopy(manifest)
    invalid_manifest["artifacts"][0]["paper_evidence"] = True

    with pytest.raises(AssertionError):
        _assert_demo_manifest_is_smoke_only(invalid_manifest)


def test_verified_manifest_has_complete_portable_entries_and_matching_hashes() -> None:
    """Every verified artifact has the release metadata needed to audit its bytes."""
    manifest = _read_json(VERIFIED_MANIFEST_PATH)
    _assert_verified_manifest_is_valid(manifest)
    for artifact in manifest["artifacts"]:
        artifact_path = REPOSITORY_ROOT / artifact["path"]
        assert artifact_path.is_file()
        assert _file_sha256(artifact_path) == artifact["sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["artifacts"][0].pop("sha256"),
        lambda manifest: manifest["artifacts"][0]["provenance"].update(
            {"source_path": "/home/example/private-input.json"}
        ),
        lambda manifest: manifest["artifacts"].append(copy.deepcopy(manifest["artifacts"][0])),
    ],
    ids=["missing-sha256", "absolute-source-path", "duplicate-artifact-id"],
)
def test_verified_manifest_rejects_incomplete_or_nonportable_metadata(mutation: object) -> None:
    """Missing checksums, private paths, and duplicate IDs are release blockers."""
    manifest = copy.deepcopy(_read_json(VERIFIED_MANIFEST_PATH))
    mutation(manifest)

    with pytest.raises(AssertionError):
        _assert_verified_manifest_is_valid(manifest)


def test_verified_stage4_preserves_audited_cv_contract_without_private_inputs() -> None:
    """Sanitization retains Stage4 evidence values while removing local input locations."""
    manifest = _read_json(VERIFIED_MANIFEST_PATH)
    report_path = REPOSITORY_ROOT / next(
        item["path"]
        for item in manifest["artifacts"]
        if item["artifact_id"] == "stage4-cost-model-selection-report"
    )
    report = _read_json(report_path)
    assert report["seed"] == 20260716
    assert report["row_count"] == 176
    assert report["group_count"] == 44
    assert report["outer_splits"] == 5
    assert report["inner_splits"] == 3
    assert "inputs" not in report
    assert report["source_data_content_hashes"]
    assert all(
        not pattern.search(text)
        for text in _walk_strings(report)
        for pattern in FORBIDDEN_PUBLIC_PATTERNS[:3]
    )


def test_verified_stage4_fold_table_is_parseable_and_stage7_is_pending() -> None:
    """Stage4 folds remain audit-ready; Stage7 is excluded until formal closure."""
    manifest = _read_json(VERIFIED_MANIFEST_PATH)
    artifact_ids = {item["artifact_id"] for item in manifest["artifacts"]}
    assert not any("stage7" in artifact_id.lower() for artifact_id in artifact_ids)
    readme = (REPOSITORY_ROOT / "artifacts/README.md").read_text(encoding="utf-8").lower()
    assert "stage7" in readme and "pending formal closure" in readme
    csv_path = REPOSITORY_ROOT / next(
        item["path"]
        for item in manifest["artifacts"]
        if item["artifact_id"] == "stage4-nested-cv-folds"
    )
    assert b"\r" not in csv_path.read_bytes()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    assert len(rows) == 15
    assert {row["outer_fold"] for row in rows} == {"0", "1", "2", "3", "4"}


def test_public_data_and_artifact_files_contain_no_private_identity_or_secret_patterns() -> None:
    """All newly public JSON, JSONL, CSV, and Markdown files stay portable."""
    public_files = [
        path
        for root in (REPOSITORY_ROOT / "data", REPOSITORY_ROOT / "artifacts")
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md"}
    ]
    assert public_files
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            assert not pattern.search(text), f"forbidden public pattern in {path}: {pattern.pattern}"


def test_ci_runs_only_the_current_manifest_integration_scope() -> None:
    """The Task7 CI stage executes these boundary checks without later release gates."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pytest -q tests/integration/test_reproduce_all.py" in workflow
    assert "tests/integration --cov" not in workflow
