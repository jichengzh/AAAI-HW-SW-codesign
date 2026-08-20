from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from tests.stage6.test_p6_history_normalization import valid_private_source_map


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NORMALIZER = REPOSITORY_ROOT / "tools/release/normalize_p6_history_root.py"


def _write_source_map(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(NORMALIZER), *args],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_normalizes_absolute_private_source_map_without_private_echo(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    map_path = _write_source_map(tmp_path / "private-source-map.json", source_map)
    private_dir = tmp_path / "normalized-private"

    result = _run_cli(
        "--source-map",
        str(map_path),
        "--history-root",
        source_map["history_root"],
        "--private-dir",
        str(private_dir),
    )

    assert result.returncode == 0
    assert result.stdout == "p6_history_root_normalized\n"
    assert result.stderr == ""
    legacy = yaml.safe_load((private_dir / "legacy.local.yaml").read_text())
    assert legacy["schema_version"] == "p6_h800_coptv2x_local_v2"


def test_cli_rejects_relative_source_map_without_private_echo(tmp_path: Path) -> None:
    private_dir = tmp_path / "normalized-private"

    result = _run_cli(
        "--source-map",
        "private-source-map.json",
        "--history-root",
        str(tmp_path / "history-root"),
        "--private-dir",
        str(private_dir),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not private_dir.exists()


def test_cli_redacts_invalid_source_map_and_writes_nothing(tmp_path: Path) -> None:
    source_map = valid_private_source_map(tmp_path)
    map_path = _write_source_map(tmp_path / "private-source-map.json", source_map)
    private_dir = tmp_path / "normalized-private"
    source_map["input_sources"]["closure"] = str(tmp_path / "missing.json")
    map_path.write_text(json.dumps(source_map), encoding="utf-8")

    result = _run_cli(
        "--source-map",
        str(map_path),
        "--history-root",
        source_map["history_root"],
        "--private-dir",
        str(private_dir),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_normalization_invalid\n"
    assert not private_dir.exists()


def test_cli_rejects_symlink_parent_source_map_without_private_echo(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    real_parent = tmp_path / "real-private-map-parent"
    map_path = _write_source_map(real_parent / "private-source-map.json", source_map)
    symlink_parent = tmp_path / "linked-private-map-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    private_dir = tmp_path / "normalized-private"

    result = _run_cli(
        "--source-map",
        str(symlink_parent / map_path.name),
        "--history-root",
        source_map["history_root"],
        "--private-dir",
        str(private_dir),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_normalization_invalid\n"
    assert str(symlink_parent) not in result.stderr
    assert not private_dir.exists()


def test_cli_rejects_unignored_repository_source_map_without_private_echo(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    map_path = REPOSITORY_ROOT / f".p6-unignored-source-map-{tmp_path.name}.json"
    private_dir = tmp_path / "normalized-private"
    try:
        _write_source_map(map_path, source_map)

        result = _run_cli(
            "--source-map",
            str(map_path),
            "--history-root",
            source_map["history_root"],
            "--private-dir",
            str(private_dir),
        )
    finally:
        map_path.unlink(missing_ok=True)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_normalization_invalid\n"
    assert not private_dir.exists()


def test_cli_allows_ignored_repository_source_map_without_private_echo(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    map_path = (
        REPOSITORY_ROOT
        / ".superpowers"
        / "tmp"
        / f"p6-ignored-source-map-{tmp_path.name}.json"
    )
    private_dir = tmp_path / "normalized-private"
    try:
        _write_source_map(map_path, source_map)

        result = _run_cli(
            "--source-map",
            str(map_path),
            "--history-root",
            source_map["history_root"],
            "--private-dir",
            str(private_dir),
        )
    finally:
        map_path.unlink(missing_ok=True)
        try:
            map_path.parent.rmdir()
        except OSError:
            pass

    assert result.returncode == 0
    assert result.stdout == "p6_history_root_normalized\n"
    assert result.stderr == ""
    assert (private_dir / "legacy.local.yaml").exists()
