from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/validate_external_inputs.py"


def _record(**overrides: object) -> dict[str, object]:
    base = {
        "input_id": "stage7-formal-aggregate",
        "asset_kind": "evidence_bundle",
        "source": "publication-pending",
        "license": {"status": "unconfirmed", "reference": "provider-terms-required"},
        "version": "unreleased",
        "relative_path": "stage7/formal-aggregate.json",
        "intended_use": "stage7_formal_aggregate",
        "consumer_ids": [],
        "availability": "available_for_verification",
    }
    return {**base, **overrides}


def _write_registry(path: Path, record: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "aaai27_external_input_registry_v1",
                "registry_version": 1,
                "inputs": [record],
            }
        ),
        encoding="utf-8",
    )


def _run(registry: Path, asset_root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry),
            "--asset-root",
            str(asset_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_reports_verified_file_without_checksum(tmp_path: Path) -> None:
    registry, assets, output = (
        tmp_path / "registry.json",
        tmp_path / "assets",
        tmp_path / "result.json",
    )
    target = assets / "stage7/formal-aggregate.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"paper_evidence": false}\n', encoding="utf-8")
    _write_registry(registry, _record())

    result = _run(registry, assets, output)

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "verified"


def test_cli_marks_missing_file_unavailable(tmp_path: Path) -> None:
    registry, assets, output = (
        tmp_path / "registry.json",
        tmp_path / "assets",
        tmp_path / "result.json",
    )
    assets.mkdir()
    _write_registry(registry, _record())

    result = _run(registry, assets, output)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode != 0
    assert manifest["inputs"][0]["reason"] == "asset_missing"


def test_cli_compares_checksum_only_when_given(tmp_path: Path) -> None:
    registry, assets, output = (
        tmp_path / "registry.json",
        tmp_path / "assets",
        tmp_path / "result.json",
    )
    target = assets / "stage7/formal-aggregate.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bundle")
    _write_registry(
        registry, _record(sha256=hashlib.sha256(b"different").hexdigest())
    )

    result = _run(registry, assets, output)

    assert result.returncode != 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["inputs"][0]["reason"]
        == "asset_sha256_mismatch"
    )
