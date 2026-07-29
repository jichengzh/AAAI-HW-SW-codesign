"""Fail-closed schema tests for the Stage2 evidence archive CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UPDATE = REPOSITORY_ROOT / "scripts" / "stage2_update_evidence.py"


def _run_update(tmp_path: Path, payload: object) -> subprocess.CompletedProcess[str]:
    delta = tmp_path / "delta.json"
    delta.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(UPDATE),
            "--delta",
            str(delta),
            "--out-dir",
            str(tmp_path / "archive"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "stage2_evidence_delta_v1", "records": []},
        {"schema": "stage2_evidence_delta_v1", "model": "pyramid_lidar"},
        {"schema": "stage2_evidence_delta_v1", "model": "pyramid_lidar", "records": [{}]},
        {
            "schema": "stage2_evidence_delta_v1",
            "model": "pyramid_lidar",
            "records": [
                {
                    "backend": 7,
                    "hardware": "H800",
                    "scope": "dense",
                    "evidence_kind": "demo",
                    "provenance": "test",
                    "candidate_config": {},
                    "metric": {},
                }
            ],
        },
    ],
)
def test_evidence_archive_rejects_missing_or_invalid_required_fields(
    tmp_path: Path, payload: object
) -> None:
    result = _run_update(tmp_path, payload)

    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert not (tmp_path / "archive").exists()
