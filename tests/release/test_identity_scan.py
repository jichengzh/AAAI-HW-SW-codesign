"""Release identity checks for the public package metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job.
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _public_metadata() -> dict[str, object]:
    return tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_public_package_identity_is_portable_and_apache_licensed() -> None:
    """The released metadata identifies a portable Python 3.10+ Apache package."""
    metadata = _public_metadata()

    assert metadata["name"] == "gear_codesign"
    assert metadata["requires-python"] == ">=3.10"
    assert "Apache License" in (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")

    serialized_metadata = json.dumps(metadata, sort_keys=True)
    assert str(REPOSITORY_ROOT) not in serialized_metadata
    assert "/home/" not in serialized_metadata
    assert "file://" not in serialized_metadata


@pytest.mark.parametrize(
    ("extra", "expected_dependencies"),
    [
        ("scan", ["torch>=2.0", "torch-pruning>=1.4"]),
        (
            "repro",
            [
                "pandas>=2.0,<3",
                "scipy>=1.10,<2",
                "scikit-learn>=1.5,<2",
                "lightgbm>=4.0,<5",
            ],
        ),
    ],
)
def test_release_extras_are_declared_as_public_dependencies(
    extra: str, expected_dependencies: list[str]
) -> None:
    """Optional release capabilities resolve from package metadata, never local paths."""
    metadata = _public_metadata()

    assert metadata["optional-dependencies"][extra] == expected_dependencies
