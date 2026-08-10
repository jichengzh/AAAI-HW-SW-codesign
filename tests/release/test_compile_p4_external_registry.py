from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/compile_p4_external_registry.py"


def _module():
    spec = importlib.util.spec_from_file_location("p4_external_registry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _review_record(path: str, disposition: str = "external_contract_p4") -> dict[str, str]:
    return {"origin": "synthetic", "path": path, "disposition": disposition}


def _resource(input_id: str = "stage6-terminal-evidence") -> dict[str, object]:
    return {
        "input_id": input_id,
        "asset_kind": "evidence_bundle",
        "source": "not-publicly-available",
        "license": {"status": "unconfirmed", "reference": "provider-terms-required"},
        "version": "unreleased",
        "relative_path": "stage6/terminal-evidence.jsonl",
        "intended_use": "stage6_representative_selection",
        "consumer_ids": ["stage6_representative_selection"],
        "availability": "unavailable",
        "unavailable_reason": "bundle_not_published",
    }


def _resolution() -> dict[str, object]:
    return {
        "format": "aaai27_p4_private_resource_resolution_v1",
        "decisions": [
            {
                "origin": "synthetic",
                "path": "private/one.md",
                "decision": "asset",
                "resource_id": "stage6-terminal-evidence",
            },
            {
                "origin": "synthetic",
                "path": "private/two.md",
                "decision": "asset",
                "resource_id": "stage6-terminal-evidence",
            },
            {
                "origin": "synthetic",
                "path": "private/document.md",
                "decision": "document",
            },
        ],
        "resources": [_resource()],
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_review(path: Path) -> None:
    _write_json(
        path,
        {
            "records": [
                _review_record("private/one.md"),
                _review_record("private/two.md"),
                _review_record("private/document.md"),
                _review_record("private/not-p4.md", "execution_contract_p6"),
            ]
        },
    )


def test_compile_registry_deduplicates_assets_and_omits_private_candidate_data(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    _write_review(review_path)
    _write_json(resolution_path, _resolution())

    registry, coverage = _module().compile_registry(review_path, resolution_path)

    assert [item["input_id"] for item in registry["inputs"]] == [
        "stage6-terminal-evidence"
    ]
    assert coverage == {
        "format": "aaai27_p4_external_resource_coverage_v1",
        "p3_p4_candidate_count": 3,
        "document_candidate_count": 1,
        "asset_candidate_count": 2,
        "distinct_resource_count": 1,
        "asset_kind_counts": [{"asset_kind": "evidence_bundle", "count": 1}],
        "availability_counts": [{"availability": "unavailable", "count": 1}],
    }
    assert "private/one.md" not in json.dumps(registry)
    assert "private/one.md" not in json.dumps(coverage)
    assert "candidate_id" not in json.dumps(registry)
    assert "private/one.md" not in registry["inputs"][0]["consumer_ids"]


def test_compile_registry_rejects_missing_p4_decision(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    resolution = _resolution()
    resolution["decisions"] = resolution["decisions"][:-1]
    _write_review(review_path)
    _write_json(resolution_path, resolution)

    module = _module()
    with pytest.raises(module.CompilationError, match="cover every P4 candidate"):
        module.compile_registry(review_path, resolution_path)


def test_compile_registry_rejects_resource_without_asset_decision(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    resolution = _resolution()
    resolution["resources"] = [_resource(), _resource("unused-resource")]
    _write_review(review_path)
    _write_json(resolution_path, resolution)

    module = _module()
    with pytest.raises(module.CompilationError, match="every resource must be used"):
        module.compile_registry(review_path, resolution_path)


def test_compile_registry_rejects_private_candidate_path_in_consumer_ids(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    resolution = _resolution()
    resolution["resources"][0]["consumer_ids"] = ["derived-from-private/one.md"]
    _write_review(review_path)
    _write_json(resolution_path, resolution)

    module = _module()
    with pytest.raises(module.CompilationError, match="consumer_ids"):
        module.compile_registry(review_path, resolution_path)


def test_cli_writes_both_public_outputs_only_after_success(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    registry_output = tmp_path / "registry.json"
    coverage_output = tmp_path / "coverage.json"
    _write_review(review_path)
    resolution = _resolution()
    resolution["decisions"] = resolution["decisions"][:-1]
    _write_json(resolution_path, resolution)

    failed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--p3-review",
            str(review_path),
            "--resolution",
            str(resolution_path),
            "--registry-output",
            str(registry_output),
            "--coverage-output",
            str(coverage_output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert failed.returncode == 2
    assert not registry_output.exists()
    assert not coverage_output.exists()

    _write_json(resolution_path, _resolution())
    succeeded = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--p3-review",
            str(review_path),
            "--resolution",
            str(resolution_path),
            "--registry-output",
            str(registry_output),
            "--coverage-output",
            str(coverage_output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert succeeded.returncode == 0, succeeded.stderr
    assert "p3_p4_candidates=3" in succeeded.stdout
    assert "private/one.md" not in succeeded.stdout
    assert registry_output.is_file()
    assert coverage_output.is_file()


def test_cli_does_not_leave_registry_when_coverage_parent_cannot_be_created(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    registry_output = tmp_path / "registry.json"
    blocked_parent = tmp_path / "blocked-parent"
    coverage_output = blocked_parent / "coverage.json"
    _write_review(review_path)
    _write_json(resolution_path, _resolution())
    blocked_parent.write_text("not a directory", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--p3-review",
            str(review_path),
            "--resolution",
            str(resolution_path),
            "--registry-output",
            str(registry_output),
            "--coverage-output",
            str(coverage_output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert not registry_output.exists()
    assert not coverage_output.exists()


def test_cli_preserves_existing_registry_when_coverage_output_is_not_a_file(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.json"
    resolution_path = tmp_path / "resolution.json"
    registry_output = tmp_path / "registry.json"
    coverage_output = tmp_path / "coverage-output"
    _write_review(review_path)
    _write_json(resolution_path, _resolution())
    registry_output.write_text("existing registry", encoding="utf-8")
    coverage_output.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--p3-review",
            str(review_path),
            "--resolution",
            str(resolution_path),
            "--registry-output",
            str(registry_output),
            "--coverage-output",
            str(coverage_output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert registry_output.read_text(encoding="utf-8") == "existing registry"
    assert coverage_output.is_dir()
