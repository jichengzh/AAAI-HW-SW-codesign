"""Black-box security tests for the anonymous AAAI archive tools."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPOSITORY_ROOT / "tools/release/build_anonymous_archive.py"
ARCHIVE_NAME = "aaai27_code_data_anonymous.zip"
sys.path.insert(0, str(BUILDER.parent))
from build_anonymous_archive import ArchiveSafetyError, build_archive  # noqa: E402
import build_anonymous_archive as archive_builder  # noqa: E402
from verify_archive import verify_archive  # noqa: E402


def _write(path: Path, content: str = "safe content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def anonymous_repo(tmp_path: Path) -> Path:
    """A complete minimal input tree, built at runtime without private values."""
    root = tmp_path / "input"
    _write(root / "LICENSE", "Apache License\n")
    _write(root / "README.anonymous.md", "# Anonymous submission\n")
    _write(root / "REPRODUCIBILITY.md", "# Reproduce\n")
    _write(root / "pyproject.toml", "[project]\nname = 'anonymous'\nversion = '0'\n")
    _write(root / "framework/__init__.py")
    _write(root / "scripts/reproduce/run.py")
    _write(root / "scripts/prepare_stage2_demo_data.py")
    _write(root / "scripts/stage1_classify_models.py")
    _write(root / "scripts/stage2_optimize_model.py")
    _write(root / "data/demo.json", "{}\n")
    _write(root / "artifacts/verified.json", "{}\n")
    _write(root / "tests/test_smoke.py")
    return root


def _run_builder(root: Path, output_dir: Path) -> SimpleNamespace:
    try:
        build_archive(root, output_dir)
    except ArchiveSafetyError as error:
        category = "unsafe output" if "output" in str(error) else "unsafe input"
        return SimpleNamespace(returncode=2, stdout="", stderr=category)
    return SimpleNamespace(returncode=0, stdout="archive created", stderr="")


def _run_verifier(archive: Path) -> SimpleNamespace:
    try:
        verify_archive(archive)
    except ArchiveSafetyError:
        return SimpleNamespace(returncode=2, stdout="", stderr="unsafe archive")
    return SimpleNamespace(returncode=0, stdout="archive verified", stderr="")


def _refresh_integrity_sidecars(output: Path) -> None:
    """Bind a deliberately altered ZIP so preflight checks, not stale metadata, reject it."""
    archive = output / ARCHIVE_NAME
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path = output / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["zip_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (output / f"{ARCHIVE_NAME}.sha256").write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="ascii")


def test_builder_emits_only_anonymous_allowlisted_deterministic_files(
    anonymous_repo: Path, tmp_path: Path
) -> None:
    """Removing allowlist filtering or README renaming breaks this archive contract."""
    _write(anonymous_repo / "README.md", "public material\n")
    _write(anonymous_repo / ".github/workflows/ci.yml")
    _write(anonymous_repo / "cache/ignored.txt")
    first = tmp_path / "one"
    second = tmp_path / "two"

    first_result = _run_builder(anonymous_repo, first)
    second_result = _run_builder(anonymous_repo, second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    first_archive = first / ARCHIVE_NAME
    second_archive = second / ARCHIVE_NAME
    assert hashlib.sha256(first_archive.read_bytes()).hexdigest() == hashlib.sha256(
        second_archive.read_bytes()
    ).hexdigest()
    assert (first / f"{ARCHIVE_NAME}.sha256").read_text(encoding="ascii").strip() == (
        f"{hashlib.sha256(first_archive.read_bytes()).hexdigest()}  {ARCHIVE_NAME}"
    )
    manifest = json.loads((first / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["zip_sha256"] == hashlib.sha256(first_archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(first_archive) as archive:
        names = archive.namelist()
    assert "README.md" in names
    assert "README.anonymous.md" not in names
    assert "README.md" == names[0] or "README.md" in names
    assert not {name for name in names if name.startswith((".github/", "cache/"))}
    assert all(not name.startswith((".git/", "dist/", "results/")) for name in names)
    assert sorted(names) == names
    assert {entry["path"] for entry in manifest["members"]} == set(names)


@pytest.mark.parametrize("kind", ["local_path", "owner_url", "email", "ipv4", "token", "filename", "large"])
def test_builder_rejects_identity_and_size_leaks_without_echoing_them(
    anonymous_repo: Path, tmp_path: Path, kind: str
) -> None:
    """Disabling any source scanner branch must reject a selected unsafe input."""
    target = anonymous_repo / "framework/unsafe.py"
    unsafe_value = ""
    if kind == "local_path":
        unsafe_value = "/" + "home" + "/person/private"
        _write(target, unsafe_value)
    elif kind == "owner_url":
        unsafe_value = "https://" + "github.com" + "/private-owner/private-repo"
        _write(target, unsafe_value)
    elif kind == "email":
        unsafe_value = "person" + "@" + "example.invalid"
        _write(target, unsafe_value)
    elif kind == "ipv4":
        unsafe_value = "203" + ".0.113.42"
        _write(target, unsafe_value)
    elif kind == "token":
        unsafe_value = "ghp_" + "a" * 36
        _write(target, unsafe_value)
    elif kind == "filename":
        unsafe_value = "person" + "@" + "example.invalid"
        _write(anonymous_repo / "framework" / f"{unsafe_value}.py")
    else:
        unsafe_value = "oversize-content"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * (20 * 1024 * 1024 + 1))

    result = _run_builder(anonymous_repo, tmp_path / "output")

    assert result.returncode != 0
    assert "unsafe input" in result.stderr.lower()
    assert unsafe_value not in result.stdout
    assert unsafe_value not in result.stderr


def test_builder_rejects_repository_symlink_even_when_allowlisted(
    anonymous_repo: Path, tmp_path: Path
) -> None:
    """Following a symlink would permit source-tree escapes and is forbidden."""
    outside = tmp_path / "outside.py"
    _write(outside)
    link = anonymous_repo / "framework/escaped.py"
    os.symlink(outside, link)

    result = _run_builder(anonymous_repo, tmp_path / "output")

    assert result.returncode != 0
    assert "unsafe input" in result.stderr.lower()
    assert str(outside) not in result.stderr


def test_builder_refuses_symlinked_or_prepopulated_output_directory(
    anonymous_repo: Path, tmp_path: Path
) -> None:
    """Output validation prevents writes through links and silent overwrites."""
    destination = tmp_path / "destination"
    destination.mkdir()
    link = tmp_path / "linked-output"
    os.symlink(destination, link)

    linked_result = _run_builder(anonymous_repo, link)
    assert linked_result.returncode != 0
    assert "output" in linked_result.stderr.lower()

    output = tmp_path / "output"
    output.mkdir()
    _write(output / ARCHIVE_NAME, "different contents")
    existing_result = _run_builder(anonymous_repo, output)
    assert existing_result.returncode != 0
    assert "output" in existing_result.stderr.lower()


def test_verifier_rechecks_real_archive_and_sidecars(anonymous_repo: Path, tmp_path: Path) -> None:
    """A verifier that trusts the builder rather than the ZIP/manifest fails this check."""
    output = tmp_path / "output"
    assert _run_builder(anonymous_repo, output).returncode == 0

    result = _run_verifier(output / ARCHIVE_NAME)

    assert result.returncode == 0, result.stderr
    assert "verified" in result.stdout.lower()


def test_builder_excludes_cache_products_and_rejects_unknown_binary(anonymous_repo: Path, tmp_path: Path) -> None:
    """Broad allowlist globs must never admit local caches or opaque binaries."""
    cached = anonymous_repo / "framework/__pycache__/module.pyc"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"\0cache")
    output = tmp_path / "output"
    assert _run_builder(anonymous_repo, output).returncode == 0
    with zipfile.ZipFile(output / ARCHIVE_NAME) as archive:
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in archive.namelist())

    (anonymous_repo / "framework/opaque.bin").write_bytes(b"\xff\xfe\x00")
    rejected = _run_builder(anonymous_repo, tmp_path / "binary-output")
    assert rejected.returncode != 0
    assert "unsafe input" in rejected.stderr


def test_verifier_requires_complete_sidecar_and_manifest(anonymous_repo: Path, tmp_path: Path) -> None:
    """A standalone ZIP cannot establish its expected member set or integrity."""
    output = tmp_path / "output"
    assert _run_builder(anonymous_repo, output).returncode == 0
    (output / f"{ARCHIVE_NAME}.sha256").unlink()
    assert _run_verifier(output / ARCHIVE_NAME).returncode != 0

    clean_output = tmp_path / "clean-output"
    assert _run_builder(anonymous_repo, clean_output).returncode == 0
    (clean_output / f"{ARCHIVE_NAME}.sha256").write_text("malformed\n", encoding="ascii")
    assert _run_verifier(clean_output / ARCHIVE_NAME).returncode != 0


@pytest.mark.parametrize("first,second", [("README.md", "readme.md"), ("framework/caf\u00e9.py", "framework/cafe\u0301.py")])
def test_verifier_rejects_platform_normalized_member_collisions(
    anonymous_repo: Path, tmp_path: Path, first: str, second: str
) -> None:
    """Case-insensitive and Unicode-normalizing extractors must not be ambiguous."""
    output = tmp_path / "output"
    assert _run_builder(anonymous_repo, output).returncode == 0
    archive_path = output / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr(first, "x")
            archive.writestr(second, "y")
    _refresh_integrity_sidecars(output)
    assert _run_verifier(archive_path).returncode != 0


def test_builder_reads_selected_source_once_before_writing(
    anonymous_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second source-path read after validation would reintroduce a swap race."""
    source = anonymous_repo / "framework/__init__.py"
    original_read_bytes = Path.read_bytes

    def deny_source_reread(path: Path) -> bytes:
        if path == source:
            raise AssertionError("selected source was reopened by pathname")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", deny_source_reread)
    assert _run_builder(anonymous_repo, tmp_path / "output").returncode == 0


def test_builder_preserves_competing_output_created_during_publish(
    anonymous_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-validation competitor must cause cleanup, never an overwrite."""
    output = tmp_path / "output"
    output.mkdir()
    competitor = output / "competitor.txt"
    original_listdir = archive_builder.os.listdir
    injected = False

    def inject_competitor(path: object) -> list[str]:
        nonlocal injected
        if Path(path) == output and not injected:
            injected = True
            competitor.write_text("retain", encoding="utf-8")
            return []
        return original_listdir(path)

    monkeypatch.setattr(archive_builder.os, "listdir", inject_competitor)
    result = _run_builder(anonymous_repo, output)
    assert result.returncode != 0
    assert competitor.read_text(encoding="utf-8") == "retain"
    assert not (output / ARCHIVE_NAME).exists()


def _write_bad_archive(path: Path, kind: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if kind == "absolute":
            archive.writestr("/absolute.py", "x")
        elif kind == "traversal":
            archive.writestr("framework/../../escape.py", "x")
        elif kind == "backslash":
            archive.writestr(r"framework\\..\\escape.py", "x")
        elif kind == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("framework/same.py", "one")
                archive.writestr("framework/same.py", "two")
        elif kind == "symlink":
            member = zipfile.ZipInfo("framework/link.py")
            member.create_system = 3
            member.external_attr = 0o120777 << 16
            archive.writestr(member, "outside")
        elif kind == "members":
            for index in range(10_001):
                archive.writestr(f"framework/{index}.py", "x")
        elif kind == "total_size":
            archive.writestr("framework/large.py", b"x" * (64 * 1024 * 1024 + 1))
        elif kind == "ratio":
            archive.writestr("framework/compressed.py", b"0" * (2 * 1024 * 1024))
        else:  # pragma: no cover - test construction guard
            raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind", ["absolute", "traversal", "backslash", "duplicate", "symlink", "members", "total_size", "ratio"]
)
def test_verifier_fails_closed_before_extracting_malicious_members(tmp_path: Path, kind: str) -> None:
    """Archive path, link, duplicate, and bomb checks must precede extraction."""
    archive = tmp_path / f"{kind}.zip"
    _write_bad_archive(archive, kind)

    result = _run_verifier(archive)

    assert result.returncode != 0
    assert "unsafe archive" in result.stderr.lower()
    assert "traceback" not in result.stderr.lower()
