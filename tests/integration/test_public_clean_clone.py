"""Public release entrypoint checks for a newcomer with no local project state."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts/reproduce/smoke_clean_clone.sh"


def _is_anonymous_reviewer_archive() -> bool:
    root_readme = REPOSITORY_ROOT / "README.md"
    return (
        not (REPOSITORY_ROOT / "README.anonymous.md").exists()
        and root_readme.is_file()
        and root_readme.read_text(encoding="utf-8").startswith("# Anonymous AAAI Submission")
    )


def test_public_readme_exposes_anonymous_safe_clean_clone_workflow() -> None:
    if _is_anonymous_reviewer_archive():
        pytest.skip("the mapped anonymous README intentionally omits public-clone instructions")
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "GEAR_REPOSITORY_URL" in readme
    assert "--depth 1" in readme
    assert "--filter=blob:none" in readme
    assert "smoke_clean_clone.sh" in readme
    assert "does not download datasets, checkpoints" in readme


def test_cpu_smoke_requirements_pin_the_runtime_and_test_surface() -> None:
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "download.pytorch.org/whl/cpu" in requirements
    assert "torch==" in requirements
    assert "numpy==" in requirements
    assert "PyYAML==" in requirements
    assert "pytest==" in requirements
    assert "scikit-learn==" in requirements
    assert "lightgbm==" in requirements


def test_clean_clone_script_has_safe_controls_and_never_fetches_experiment_assets() -> None:
    syntax = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0, help_result.stderr
    for option in ("--repo-url", "--ref", "--work-dir", "--python", "--skip-install"):
        assert option in help_result.stdout
    assert "dataset" in help_result.stdout.lower()
    assert "checkpoint" in help_result.stdout.lower()

    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "git clone --depth 1 --filter=blob:none --single-branch" in source
    assert "scripts/reproduce/reproduce_all.py" in source
    assert "wget" not in source
    assert "curl" not in source
    assert "gdown" not in source


def test_clean_clone_help_does_not_require_a_git_checkout(tmp_path: Path) -> None:
    """An extracted anonymous ZIP can display usage before it is placed in a Git repository."""
    isolated_script = tmp_path / "scripts" / "reproduce" / SCRIPT.name
    isolated_script.parent.mkdir(parents=True)
    isolated_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    isolated_script.chmod(0o755)

    help_result = subprocess.run(
        ["bash", str(isolated_script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert help_result.returncode == 0, help_result.stderr
    assert "Usage:" in help_result.stdout


def test_ci_installs_the_public_cpu_requirements_and_exercises_clean_clone() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "public-smoke" in workflow
    assert "pip install -r requirements.txt" in workflow
    assert "smoke_clean_clone.sh" in workflow
    assert "file:" + "//" not in workflow


def test_public_smoke_outputs_are_not_accidentally_staged() -> None:
    ignore_rules = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "repro-smoke-output/" in ignore_rules
    assert "repro-verified-output/" in ignore_rules
