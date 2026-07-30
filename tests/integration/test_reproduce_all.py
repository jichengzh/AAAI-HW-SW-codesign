"""Public demo/verified artifact boundary checks."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
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
    re.compile(r"\bprivate_user_\d+\b", re.IGNORECASE),
    re.compile(r"\bgithub\.com/[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", re.IGNORECASE),
)
DEMO_JSON_RECORD_CONTAINERS = {"demo-capability-profiles": "profiles"}


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


def _load_demo_records(manifest: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Load real demo records while respecting JSON metadata and JSONL row boundaries."""
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list) and artifacts
    records_by_artifact: dict[str, list[dict[str, object]]] = {}
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        artifact_id = artifact.get("artifact_id")
        relative_path = artifact.get("path")
        assert isinstance(artifact_id, str) and artifact_id
        assert isinstance(relative_path, str) and not Path(relative_path).is_absolute()
        path = REPOSITORY_ROOT / relative_path
        if path.suffix == ".json":
            document = _read_json(path)
            assert document.get("paper_evidence") is False
            assert document.get("intended_use") == "smoke_test_only"
            container = DEMO_JSON_RECORD_CONTAINERS.get(artifact_id)
            assert container is not None
            records = document.get(container)
            assert isinstance(records, list) and records
        else:
            assert path.suffix == ".jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            assert records
        assert all(isinstance(record, dict) for record in records)
        records_by_artifact[artifact_id] = records
    return records_by_artifact


def _assert_demo_records_are_smoke_only(records_by_artifact: dict[str, list[dict[str, object]]]) -> None:
    assert records_by_artifact
    for records in records_by_artifact.values():
        assert records
        for record in records:
            assert record.get("paper_evidence") is False


def _assert_no_forbidden_public_patterns(text: str) -> None:
    for pattern in FORBIDDEN_PUBLIC_PATTERNS:
        assert not pattern.search(text), "forbidden public pattern detected"


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


def test_every_actual_demo_record_is_explicitly_non_evidence() -> None:
    """Every JSON/JSONL demo record, not just the manifest, is smoke-only."""
    records_by_artifact = _load_demo_records(_read_json(DEMO_MANIFEST_PATH))

    _assert_demo_records_are_smoke_only(records_by_artifact)


@pytest.mark.parametrize(
    "artifact_id",
    [
        "demo-capability-profiles",
        "demo-coldstart-graph-features",
        "demo-coldstart-measurements",
        "demo-candidate-pool",
    ],
)
def test_demo_record_validator_rejects_paper_evidence_in_each_source(artifact_id: str) -> None:
    """Changing any source's actual first record to evidence must fail validation."""
    records_by_artifact = _load_demo_records(_read_json(DEMO_MANIFEST_PATH))
    invalid_records = copy.deepcopy(records_by_artifact)
    invalid_records[artifact_id][0]["paper_evidence"] = True

    with pytest.raises(AssertionError):
        _assert_demo_records_are_smoke_only(invalid_records)


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
        _assert_no_forbidden_public_patterns(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_text",
    [
        "created by private_user_123",
        "https://github.com/private-owner/private-repository",
    ],
    ids=["bare-private-identity", "github-owner-repository"],
)
def test_public_leak_scanner_rejects_identity_and_github_repository_shapes(fixture_text: str) -> None:
    """Scanner rejects synthetic identity and repository fixtures without echoing them."""
    with pytest.raises(AssertionError, match="forbidden public pattern"):
        _assert_no_forbidden_public_patterns(fixture_text)


def test_ci_runs_task8_reproduction_integration_without_later_release_gates() -> None:
    """CI exercises the Task8 entrypoint without enabling archive or global coverage work."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pytest -q tests/integration/test_reproduce_all.py" in workflow
    assert "scripts/reproduce/reproduce_all.py" in workflow
    assert "anonymous archive" not in workflow.lower()
    assert "tests/integration --cov" not in workflow


def _run_reproduction(*args: str) -> subprocess.CompletedProcess[str]:
    """Execute the public entrypoint as an external user would."""
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/reproduce/reproduce_all.py"),
            *args,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _reproduction_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "reproduce_all_integration_module",
        REPOSITORY_ROOT / "scripts/reproduce/reproduce_all.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _content_identity(output_root: Path) -> dict[str, object]:
    """Return deterministic reproduction identity, excluding run timestamps."""
    manifest = _read_json(output_root / "run_manifest.json")
    return {
        "schema_version": manifest["schema_version"],
        "mode": manifest["mode"],
        "status": manifest["status"],
        "stages": manifest["stages"],
        "outputs": manifest["outputs"],
        "stage5_selected": _read_json(output_root / "stage5/selected_ids.json"),
        "stage7_selected": _read_json(output_root / "stage7/selection.json"),
    }


def test_smoke_reproduction_is_cpu_only_demo_only_and_content_identical_across_roots(
    tmp_path: Path,
) -> None:
    """Run every public selection stage twice without paper evidence or hardware execution."""
    first_root = tmp_path / "smoke-first"
    second_root = tmp_path / "smoke-second"

    first = _run_reproduction("--mode", "smoke", "--output-root", str(first_root))
    second = _run_reproduction("--mode", "smoke", "--output-root", str(second_root))
    repeat = _run_reproduction("--mode", "smoke", "--output-root", str(first_root))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert repeat.returncode == 0, repeat.stderr
    first_identity = _content_identity(first_root)
    second_identity = _content_identity(second_root)
    assert first_identity == second_identity
    assert first_identity["status"] == "completed"
    assert first_identity["stage5_selected"]["ordered_selected_ids"]
    assert first_identity["stage7_selected"]["acquisition"]["selected_row_ids"]
    manifest = _read_json(first_root / "run_manifest.json")
    reproduce_all = _reproduction_module()
    assert reproduce_all._claim_output_root(first_root, "smoke")["status"] == "completed"
    with pytest.raises(reproduce_all.PublicReproductionError, match="safe explicit"):
        reproduce_all._safe_output_root(str(REPOSITORY_ROOT))
    assert manifest["paper_evidence"] is False
    assert manifest["execution"] == {
        "network": False,
        "gpu": False,
        "hardware_execution": False,
        "cache_execution": False,
        "tvm_or_ap_execution": False,
    }
    assert all(stage["status"] == "completed" for stage in manifest["stages"])
    for item in manifest["outputs"]:
        assert SHA256_PATTERN.fullmatch(item["sha256"])
        assert _file_sha256(first_root / item["path"]) == item["sha256"]
    assert all(item["path"] != "run_manifest.json" for item in manifest["outputs"])


def test_verified_reproduction_fails_closed_without_required_evidence(
    tmp_path: Path,
) -> None:
    """Verified mode never substitutes demo data when Stage6/Stage7 evidence is absent."""
    output_root = tmp_path / "verified"

    result = _run_reproduction("--mode", "verified", "--output-root", str(output_root))

    assert result.returncode != 0
    assert "unavailable" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert str(output_root) not in result.stderr
    manifest = _read_json(output_root / "run_manifest.json")
    assert manifest["mode"] == "verified"
    assert manifest["status"] == "unavailable"
    assert manifest["paper_evidence"] is True
    assert not (output_root / "stage5").exists()
    assert not (output_root / "stage6").exists()
