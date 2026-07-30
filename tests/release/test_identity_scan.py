"""Release identity checks for the public package metadata."""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
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
    re.compile(r"/" + r"home/[^\s\"']+"),
    re.compile(r"/" + r"Users/[^\s\"']+"),
    re.compile(r"[A-Za-z]:" + r"\\[^\s\"']+"),
    re.compile("file:" + "//", re.IGNORECASE),
)
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
SECRET_PATTERN = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*[^\s\"']+"
    r"|\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[A-Z0-9]{16})\b)"
)
DOCUMENTED_PYTHON_SCRIPT_PATTERN = re.compile(r"\bpython(?:3)?\s+([A-Za-z0-9_./-]+\.py)\b")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")
SHELL_FENCE_PATTERN = re.compile(
    r"^```(?:bash|sh|shell)\s*$\n(.*?)^```\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL
)
PUBLIC_DOCUMENTATION = (
    "README.md",
    "README.zh-CN.md",
    "REPRODUCIBILITY.md",
    "ARTIFACTS.md",
    "README.anonymous.md",
)
ANONYMOUS_ARCHIVE_DOCUMENTATION = ("README.md", "REPRODUCIBILITY.md", "ARTIFACTS.md")


def _is_anonymous_reviewer_archive() -> bool:
    """Distinguish the mapped anonymous ZIP root from the public source tree."""
    root_readme = REPOSITORY_ROOT / "README.md"
    return (
        (REPOSITORY_ROOT / "tools/release/anonymous_allowlist.txt").is_file()
        and not (REPOSITORY_ROOT / "README.anonymous.md").exists()
        and root_readme.is_file()
        and root_readme.read_text(encoding="utf-8").startswith("# Anonymous AAAI Submission")
    )


def _release_documentation() -> tuple[str, ...]:
    if _is_anonymous_reviewer_archive():
        return ANONYMOUS_ARCHIVE_DOCUMENTATION
    return PUBLIC_DOCUMENTATION


def _anonymous_readme_path() -> Path:
    return REPOSITORY_ROOT / ("README.md" if _is_anonymous_reviewer_archive() else "README.anonymous.md")


def _public_metadata() -> dict[str, object]:
    return tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def _assert_no_release_identity_leaks(text: str) -> None:
    for pattern in (*LOCAL_PATH_PATTERNS, EMAIL_PATTERN, SECRET_PATTERN):
        assert not pattern.search(text), f"release identity leak matched {pattern.pattern!r}"


def _metadata_headers(text: str) -> dict[str, str]:
    message = Parser().parsestr(text)
    return {name.lower(): value for name, value in message.items()}


def _shell_command_argvs(document: str) -> list[list[str]]:
    """Parse every documented shell command, joining normal backslash continuations."""
    commands: list[list[str]] = []
    for block in SHELL_FENCE_PATTERN.findall(document):
        pending = ""
        for source_line in block.splitlines():
            line = source_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                pending += line[:-1] + " "
                continue
            command_line = pending + line
            pending = ""
            lexer = shlex.shlex(command_line, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            current: list[str] = []
            for token in lexer:
                if token in {";", "&&", "||", "|"}:
                    if current:
                        commands.append(current)
                        current = []
                else:
                    current.append(token)
            if current:
                commands.append(current)
        assert not pending, "documented shell block ends with an unfinished continuation"
    return commands


def _is_documented_repository_path(token: str) -> bool:
    normalized = token.removeprefix("./")
    return (
        normalized
        in {
            "README.md",
            "README.zh-CN.md",
            "README.anonymous.md",
            "REPRODUCIBILITY.md",
            "ARTIFACTS.md",
        }
        or normalized in {"docs", "framework", "scripts", "tests", "tools"}
        or normalized.startswith(("docs/", "framework/", "scripts/", "tests/", "tools/"))
    )


def _assert_documented_repository_path(document_name: str, token: str) -> None:
    if not _is_documented_repository_path(token):
        return
    path = REPOSITORY_ROOT / token.removeprefix("./")
    assert path.exists(), f"{document_name} has a missing repository path: {token}"


def _module_is_executable(module: str) -> bool:
    """Return whether ``python -m module`` can execute a Python implementation."""
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
        return False
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None:
        if spec.submodule_search_locations is not None:
            return any(
                (Path(location) / "__main__.py").is_file()
                for location in spec.submodule_search_locations
            )
        origin = spec.origin
        return isinstance(origin, str) and origin.endswith(".py") and Path(origin).is_file()
    module_path = REPOSITORY_ROOT.joinpath(*module.split("."))
    return module_path.with_suffix(".py").is_file() or (module_path / "__main__.py").is_file()


def _command_start(command: list[str]) -> int:
    index = 0
    while index < len(command) and re.fullmatch(r"[A-Za-z_]\w*=.*", command[index]):
        index += 1
    return index


def _assert_pytest_paths(document_name: str, arguments: list[str]) -> None:
    for argument in arguments:
        if not argument.startswith("-"):
            _assert_documented_repository_path(document_name, argument)


def _assert_relative_python_script(document_name: str, token: str) -> None:
    path = Path(token)
    if path.is_absolute():
        return
    assert (REPOSITORY_ROOT / path).is_file(), (
        f"{document_name} has a missing repository path: {token}"
    )


def _assert_documented_command_paths(document_name: str, document: str) -> None:
    """Ensure examples name resolvable Python, pytest, tool, test, and document paths."""
    for command in _shell_command_argvs(document):
        for token in command:
            _assert_documented_repository_path(document_name, token)
        start = _command_start(command)
        if start == len(command):
            continue
        executable = command[start]
        arguments = command[start + 1 :]
        if executable in {"pytest", "py.test"}:
            _assert_pytest_paths(document_name, arguments)
            continue
        if executable not in {"python", "python3"}:
            continue
        if len(arguments) >= 2 and arguments[0] == "-m":
            module = arguments[1]
            assert _module_is_executable(module), (
                f"{document_name} has a Python module that is not executable: {module}"
            )
            if module in {"pytest", "py.test"}:
                _assert_pytest_paths(document_name, arguments[2:])
            continue
        for argument in arguments:
            if argument.endswith(".py"):
                _assert_relative_python_script(document_name, argument)


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
    assert "file:" + "//" not in serialized_metadata


def test_anonymous_readme_has_no_public_identity_or_network_locations() -> None:
    """The reviewer-facing README must stand alone without public-project identity."""
    anonymous_readme = _anonymous_readme_path().read_text(encoding="utf-8")

    _assert_no_release_identity_leaks(anonymous_readme)
    assert "github.com" not in anonymous_readme.lower()
    assert "http://" not in anonymous_readme.lower()
    assert "https://" not in anonymous_readme.lower()


def test_release_documentation_commands_and_relative_links_resolve_locally() -> None:
    """Each artifact's documented quick-start commands resolve in that artifact."""
    documents: dict[str, str] = {}
    for relative_name in _release_documentation():
        path = REPOSITORY_ROOT / relative_name
        assert path.is_file(), f"missing release documentation: {relative_name}"
        documents[relative_name] = path.read_text(encoding="utf-8")

    for relative_name, document in documents.items():
        _assert_documented_command_paths(relative_name, document)
        for script_name in DOCUMENTED_PYTHON_SCRIPT_PATTERN.findall(document):
            script_path = REPOSITORY_ROOT / script_name
            assert script_path.is_file(), f"{relative_name} documents missing script {script_name}"
        if not _is_anonymous_reviewer_archive():
            for target in MARKDOWN_LINK_PATTERN.findall(document):
                local_target = target.split("#", maxsplit=1)[0]
                if not local_target or "://" in local_target or local_target.startswith("mailto:"):
                    continue
                assert not Path(local_target).is_absolute(), (
                    f"{relative_name} links outside the checkout"
                )
                assert (REPOSITORY_ROOT / local_target).exists(), (
                    f"{relative_name} has a broken relative link: {target}"
                )


def test_release_readme_contract_is_unambiguous_for_its_artifact_mode() -> None:
    """The reviewer ZIP must never carry a public README alongside its anonymous root."""
    root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    anonymous_path = REPOSITORY_ROOT / "README.anonymous.md"

    if _is_anonymous_reviewer_archive():
        assert root_readme.startswith("# Anonymous AAAI Submission")
        assert not anonymous_path.exists()
        assert "# GEAR Co-design reproducibility package" not in root_readme
    else:
        assert root_readme.startswith("# GEAR Co-design reproducibility package")
        assert anonymous_path.is_file()
        assert anonymous_path.read_text(encoding="utf-8").startswith("# Anonymous AAAI Submission")


def test_documented_command_validator_rejects_a_missing_pytest_target() -> None:
    """A stale test path in a shell example must block the public documentation."""
    document = "```bash\npytest tests/does-not-exist.py -q\n```\n"

    with pytest.raises(AssertionError, match="missing repository path"):
        _assert_documented_command_paths("fixture.md", document)


def test_documented_command_validator_rejects_a_missing_anonymous_script() -> None:
    """The anonymous README receives the same command-path validation as public docs."""
    document = "```shell\npython scripts/reproduce/does-not-exist.py --mode smoke\n```\n"

    with pytest.raises(AssertionError, match="missing repository path"):
        _assert_documented_command_paths("README.anonymous.md", document)


def test_documented_command_validator_rejects_an_importable_nonexecutable_package() -> None:
    """An importable package without ``__main__.py`` cannot be documented with ``python -m``."""
    document = "```bash\npython -m framework.stage4 --help\n```\n"

    with pytest.raises(AssertionError, match="not executable"):
        _assert_documented_command_paths("fixture.md", document)


def test_documented_command_validator_handles_continuations_modules_and_tool_paths() -> None:
    """Shell fences accept valid continued Python/module, pytest, and tool examples."""
    document = """```sh
python -m compileall --help framework
pytest \\
  tests/release -q
python tools/release/verify_archive.py \\
  --help
```
"""

    _assert_documented_command_paths("fixture.md", document)
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "--help", "framework"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_public_reproducibility_matrix_states_evidence_boundaries() -> None:
    """Removing an evidence field would make paper readiness look stronger than it is."""
    reproducibility = (REPOSITORY_ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    artifacts_path = REPOSITORY_ROOT / "ARTIFACTS.md"
    assert artifacts_path.is_file(), "artifact classes need a public index"
    artifacts = artifacts_path.read_text(encoding="utf-8")

    for field in ("Code", "Input artifact", "Seed", "Algorithm runs", "Evidence status"):
        assert field in reproducibility
    for evidence_class in ("demo", "verified", "external", "unavailable"):
        assert evidence_class in reproducibility.lower()
        assert evidence_class in artifacts.lower()
    for required_fact in (
        "176 measurements",
        "44 groups",
        "5 outer",
        "3 inner",
        "20260716",
        "20260718",
        "20260719",
        "20260720",
        "12 trajectories",
        "192 selected events",
    ):
        assert required_fact in reproducibility


def test_public_release_metadata_has_no_identity_or_secret_leaks(
    built_artifacts: tuple[Path, Path],
) -> None:
    """Published metadata files and complete artifact metadata remain portable."""
    wheel_path, sdist_path = built_artifacts
    source_metadata = [
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8"),
        (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8"),
    ]
    if not _is_anonymous_reviewer_archive():
        source_metadata.extend(
            [
                (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
                (REPOSITORY_ROOT / "README.zh-CN.md").read_text(encoding="utf-8"),
            ]
        )
    for text in source_metadata:
        _assert_no_release_identity_leaks(text)

    if not _is_anonymous_reviewer_archive():
        citation = yaml.safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        assert citation["authors"] == [{"name": "GEAR Co-design Collective"}]

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_metadata = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata_text = wheel.read(wheel_metadata).decode("utf-8")
    with tarfile.open(sdist_path) as sdist:
        pkg_info = next(
            member for member in sdist.getmembers() if member.name.endswith("/PKG-INFO")
        )
        extracted = sdist.extractfile(pkg_info)
        assert extracted is not None
        sdist_metadata_text = extracted.read().decode("utf-8")

    readme_title = (
        "# Anonymous AAAI Submission: Code and Data"
        if _is_anonymous_reviewer_archive()
        else "# GEAR Co-design reproducibility package"
    )
    for metadata_text in (wheel_metadata_text, sdist_metadata_text):
        assert readme_title in metadata_text
        _assert_no_release_identity_leaks(metadata_text)

        headers = _metadata_headers(metadata_text)
        assert "author" not in headers
        assert "author-email" not in headers


@pytest.mark.parametrize(
    "description_body",
    [
        "Local checkout: /" + "home/alice/private-release",
        "Local checkout: /" + "Users/alice/private-release",
        "Local checkout: " + "C:" + "\\Users\\alice\\private-release",
        "Credential: " + "github_pat_" + "abcdefghijklmnopqrstuvwxyz123456",
        "Contact: " + "alice" + "@" + "example.com",
    ],
)
def test_release_leak_scanner_rejects_leaks_in_description_body(description_body: str) -> None:
    """Leaks in the non-header metadata body are rejected, not merely parsed away."""
    metadata_text = f"Metadata-Version: 2.4\nName: gear_codesign\n\n{description_body}"

    assert _metadata_headers(metadata_text)["name"] == "gear_codesign"
    with pytest.raises(AssertionError):
        _assert_no_release_identity_leaks(metadata_text)


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
        sdist_paths = {
            member.name.split("/", 1)[1] for member in sdist.getmembers() if "/" in member.name
        }
        assert expected_wheel_paths <= sdist_paths


def test_ci_enforces_release_source_quality_and_global_coverage() -> None:
    """The release workflow checks every shipped source directory with one global gate."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ruff check framework scripts tests tools" in workflow
    assert "python -m compileall -q framework scripts tools" in workflow
    assert re.search(
        r"pytest -q --cov=framework --cov=scripts/reproduce\s+"
        r"--cov-report=term-missing --cov-report=xml --cov-fail-under=80",
        workflow,
    )


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
                "scikit-learn>=1.6,<2",
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
