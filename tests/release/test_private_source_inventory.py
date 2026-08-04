"""Black-box tests for the safe P2 private-source inventory command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "tools/release/inventory_private_source.py"


def _write(path: Path, content: str | bytes = "safe\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _private_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "private-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _write(root / "framework/safe.py", "VALUE = 1\n")
    _write(root / "docs/design.md", "# Design\n")
    _write(root / "framework/tests/fixtures/tiny.json", "{}\n")
    _write(root / "project/data/labels.json", "{\"private\": true}\n")
    _write(root / "project/datasets/train.yaml", "split: private\n")
    _write(root / "project/test/case.json", "{\"private\": true}\n")
    _write(root / "project/val/list.txt", "private-example\n")
    _write(root / "scripts/runner.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write(root / "checkpoints/model.pt", b"not scanned as a model artifact")
    _write(root / "results/generated.py", "print('generated')\n")
    _write(root / "docs/private.md", "/" + "home" + "/operator/private\n")
    _write(root / "docs/untracked.md", "# Untracked design note\n")
    _write(tmp_path / "outside.json", "{\"outside\": true}\n")
    (root / "docs" / "outside-link.json").symlink_to(tmp_path / "outside.json")
    subprocess.run(
        [
            "git",
            "add",
            "framework",
            "docs/design.md",
            "docs/private.md",
            "docs/outside-link.json",
            "scripts",
            "checkpoints",
            "results",
            "project",
        ],
        cwd=root,
        check=True,
    )
    return root


def test_inventory_emits_only_safe_migration_paths_and_opaque_sensitive_entries(
    tmp_path: Path,
) -> None:
    source_root = _private_fixture(tmp_path)
    output = tmp_path / "inventory.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--source-label",
            "private-fixture",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    inventory = json.loads(output.read_text(encoding="utf-8"))
    candidate_paths = {entry["path"] for entry in inventory["candidates"]}
    assert inventory["source"]["label"] == "private-fixture"
    assert str(source_root) not in output.read_text(encoding="utf-8")
    assert {
        "framework/safe.py",
        "docs/design.md",
        "framework/tests/fixtures/tiny.json",
        "docs/untracked.md",
    } <= candidate_paths
    assert "scripts/runner.sh" not in candidate_paths
    assert "checkpoints/model.pt" not in candidate_paths
    assert "results/generated.py" not in candidate_paths
    assert "docs/private.md" not in candidate_paths
    assert "docs/outside-link.json" not in candidate_paths
    assert not any(path.startswith("project/") for path in candidate_paths)
    assert {entry["classification"] for entry in inventory["opaque_entries"]} >= {
        "external_input",
        "exclude_generated",
        "review_sensitive",
        "review_shell",
    }
    assert all("path" not in entry for entry in inventory["opaque_entries"])
    assert "/" + "home" + "/operator/private" not in output.read_text(encoding="utf-8")


def test_inventory_can_exclude_untracked_files_and_refuses_an_in_tree_output(tmp_path: Path) -> None:
    source_root = _private_fixture(tmp_path)
    output = tmp_path / "without-untracked.json"

    no_untracked = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--source-label",
            "private-fixture",
            "--output",
            str(output),
            "--no-untracked",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert no_untracked.returncode == 0, no_untracked.stderr
    candidates = json.loads(output.read_text(encoding="utf-8"))["candidates"]
    assert "docs/untracked.md" not in {entry["path"] for entry in candidates}

    in_tree = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--source-label",
            "private-fixture",
            "--output",
            str(source_root / "inventory.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert in_tree.returncode == 2
    assert "output" in in_tree.stderr.lower()

    existing_output = tmp_path / "existing.json"
    _write(existing_output, "preserve this report\n")
    existing = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--source-label",
            "private-fixture",
            "--output",
            str(existing_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert existing.returncode == 2
    assert existing_output.read_text(encoding="utf-8") == "preserve this report\n"


def test_inventory_never_echoes_a_private_source_path_in_a_validation_error(tmp_path: Path) -> None:
    missing_root = tmp_path / "private-missing"
    output = tmp_path / "inventory.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(missing_root),
            "--source-label",
            "private-fixture",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "source root" in result.stderr.lower()
    assert str(missing_root) not in result.stderr
