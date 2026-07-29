"""Release identity checks for the public package metadata."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job.
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PATH_PATTERNS = (
    re.compile(r"/home/[^\s\"']+"),
    re.compile(r"/Users/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
    re.compile(r"file://", re.IGNORECASE),
)
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
SECRET_PATTERN = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*[^\s\"']+"
    r"|\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[A-Z0-9]{16})\b)"
)


def _public_metadata() -> dict[str, object]:
    return tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _assert_no_release_identity_leaks(text: str) -> None:
    for pattern in (*LOCAL_PATH_PATTERNS, EMAIL_PATTERN, SECRET_PATTERN):
        assert not pattern.search(text), f"release identity leak matched {pattern.pattern!r}"


def _metadata_headers(text: str) -> dict[str, str]:
    message = Parser().parsestr(text)
    return {name.lower(): value for name, value in message.items()}


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build the release artifacts, instead of relying on editable-install imports."""
    artifact_dir = tmp_path_factory.mktemp("release-artifacts")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(artifact_dir),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    return (
        next(artifact_dir.glob("*.whl")),
        next(artifact_dir.glob("*.tar.gz")),
    )


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


def test_public_release_metadata_has_no_identity_or_secret_leaks(
    built_artifacts: tuple[Path, Path],
) -> None:
    """Published metadata files and artifact headers remain portable and anonymous."""
    wheel_path, sdist_path = built_artifacts
    source_metadata = [
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8"),
    ]
    for text in source_metadata:
        _assert_no_release_identity_leaks(text)

    citation = yaml.safe_load(source_metadata[1])
    assert citation["authors"] == [{"name": "The GEAR Authors"}]

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        wheel_headers = _metadata_headers(wheel.read(wheel_metadata).decode("utf-8"))
    with tarfile.open(sdist_path) as sdist:
        pkg_info = next(member for member in sdist.getmembers() if member.name.endswith("/PKG-INFO"))
        extracted = sdist.extractfile(pkg_info)
        assert extracted is not None
        sdist_headers = _metadata_headers(extracted.read().decode("utf-8"))

    for headers in (wheel_headers, sdist_headers):
        _assert_no_release_identity_leaks("\n".join(f"{key}: {value}" for key, value in headers.items()))
        assert "author" not in headers
        assert "author-email" not in headers


def test_real_artifacts_include_existing_framework_stage2_package(
    built_artifacts: tuple[Path, Path],
) -> None:
    """Package discovery includes every present ``framework.*`` import package."""
    source_paths = {
        REPOSITORY_ROOT / "framework/stage2/__init__.py",
        REPOSITORY_ROOT / "framework/stage2/contracts.py",
    }
    if not all(path.is_file() for path in source_paths):
        pytest.skip("framework.stage2 has not been migrated into this checkout yet")

    wheel_path, sdist_path = built_artifacts
    expected_wheel_paths = {
        "framework/stage2/__init__.py",
        "framework/stage2/contracts.py",
    }

    with zipfile.ZipFile(wheel_path) as wheel:
        assert expected_wheel_paths <= set(wheel.namelist())
    with tarfile.open(sdist_path) as sdist:
        sdist_paths = {member.name.split("/", 1)[1] for member in sdist.getmembers() if "/" in member.name}
        assert expected_wheel_paths <= sdist_paths


def test_ci_lints_only_the_task_one_test_scope() -> None:
    """The staged CI lint check does not claim to lint historical source directories."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ruff check tests" in workflow
    assert "ruff check framework scripts tests tools" not in workflow


def test_gitignore_protects_local_environment_secret_files() -> None:
    """Developer-specific environment files cannot be committed accidentally."""
    ignored = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored


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
