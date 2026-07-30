"""Release-profile dependency contracts declared by ``pyproject.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _extra_requirements(name: str) -> set[str]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    return set(project["project"]["optional-dependencies"][name])


def _requirement_names(requirements: set[str]) -> set[str]:
    return {requirement.split(">=", maxsplit=1)[0] for requirement in requirements}


def test_dev_extra_contains_all_dependencies_needed_by_stage1_tests() -> None:
    """The documented release test install can import real Stage1 scan modules."""
    requirements = _extra_requirements("dev")

    assert {"torch>=2.0", "torch-pruning>=1.4"} <= requirements


def test_repro_extra_remains_independent_of_optional_scan_runtime() -> None:
    """CPU smoke reproduction stays light and does not pull model-scan packages."""
    requirement_names = _requirement_names(_extra_requirements("repro"))

    assert "torch" not in requirement_names
    assert "torch-pruning" not in requirement_names
