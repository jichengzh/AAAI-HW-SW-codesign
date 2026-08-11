#!/usr/bin/env python3
"""Build a deterministic, security-scanned anonymous AAAI ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOOL_VERSION = "1.0.0"
ARCHIVE_NAME = "aaai27_code_data_anonymous.zip"
MAX_FILE_SIZE = 20 * 1024 * 1024
OPTIONAL_ALLOWLIST_ENTRIES = frozenset({"ARTIFACTS.md"})
EXECUTABLE_SUFFIXES = frozenset({".py", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1"})
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class ArchiveSafetyError(Exception):
    """A safe-to-display archive validation failure."""


@dataclass(frozen=True)
class Member:
    archive_path: str
    size: int
    digest: str
    payload: bytes


def _tool_directory() -> Path:
    return Path(__file__).resolve().parent


def _read_allowlist() -> tuple[str, ...]:
    lines = (_tool_directory() / "anonymous_allowlist.txt").read_text(encoding="utf-8").splitlines()
    return tuple(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))


def load_forbidden_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Load categorized patterns without allowing arbitrary executable configuration."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    pattern_file = _tool_directory() / "forbidden_patterns.txt"
    for line in pattern_file.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        category, expression = line.split("\t", maxsplit=1)
        patterns.append((category, re.compile(expression, re.IGNORECASE)))
    return tuple(patterns)


def _resolved_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ArchiveSafetyError("unsafe input: path outside repository") from error


def _matches_forbidden(text: str, patterns: Iterable[tuple[str, re.Pattern[str]]]) -> tuple[str, ...]:
    return tuple(category for category, pattern in patterns if pattern.search(text))


def _diagnostic_path(relative_path: str, patterns: Iterable[tuple[str, re.Pattern[str]]]) -> str:
    return "<redacted-path>" if _matches_forbidden(relative_path, patterns) else relative_path


def _scan_payload(payload: bytes, relative_path: str, patterns: Iterable[tuple[str, re.Pattern[str]]]) -> tuple[str, ...]:
    """Scan exactly the bytes that will become an archive member."""
    issues = list(_matches_forbidden(relative_path, patterns))
    if len(payload) > MAX_FILE_SIZE:
        issues.append("file-size")
        return tuple(sorted(set(issues)))
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        issues.append("unknown-binary")
    else:
        issues.extend(_matches_forbidden(text, patterns))
    return tuple(sorted(set(issues)))


def scan_file(path: Path, root: Path, patterns: Iterable[tuple[str, re.Pattern[str]]]) -> tuple[str, ...]:
    """Scan a stable local file; source collection uses the stronger no-follow reader."""
    return _scan_payload(path.read_bytes(), _safe_relative_path(path, root), patterns)


DENIED_PATH_PARTS = frozenset(
    {".git", ".github", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist", "results", "cache", "checkpoint", "checkpoints", "engine", "engines", "model", "models"}
)
DENIED_SUFFIXES = frozenset({".pyc", ".pyo", ".so", ".dll", ".dylib", ".onnx", ".engine", ".pt", ".pth", ".ckpt", ".zip", ".tar", ".gz", ".whl"})
ANONYMOUS_HIDDEN_ALLOWLIST = frozenset({".github/workflows/ci.yml", ".gitignore"})
DENIED_SOURCE_PREFIXES = ("configs/local/",)


def _is_denied_source_path(relative_path: str) -> bool:
    if relative_path in ANONYMOUS_HIDDEN_ALLOWLIST:
        return False
    if relative_path.startswith(DENIED_SOURCE_PREFIXES):
        return True
    parts = Path(relative_path).parts
    filename = parts[-1].lower()
    return (
        any(part.lower() in DENIED_PATH_PARTS or part.startswith(".") for part in parts)
        or filename.startswith((".", "#"))
        or filename.endswith(("~", ".coverage"))
        or Path(filename).suffix in DENIED_SUFFIXES
    )


def _selected_source_paths(root: Path) -> tuple[Path, ...]:
    selected: set[Path] = set()
    missing: list[str] = []
    for entry in _read_allowlist():
        matches = tuple(
            path
            for path in root.glob(entry)
            if (path.is_file() or path.is_symlink()) and not _is_denied_source_path(_safe_relative_path(path, root))
        )
        if not matches and entry not in OPTIONAL_ALLOWLIST_ENTRIES:
            missing.append(entry)
        selected.update(matches)
    if missing:
        raise ArchiveSafetyError("unsafe input: required allowlist entries are missing")
    return tuple(sorted(selected, key=lambda path: path.relative_to(root).as_posix()))


def _read_source_once(source: Path, root: Path, patterns: Iterable[tuple[str, re.Pattern[str]]]) -> Member:
    relative_path = _safe_relative_path(source, root)
    if source.is_symlink() or not _resolved_within(source, root):
        raise ArchiveSafetyError("unsafe input: symlink")
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ArchiveSafetyError("unsafe input: unreadable selected file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_SIZE:
            raise ArchiveSafetyError("unsafe input: non-regular-file")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise ArchiveSafetyError("unsafe input: file-size")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ArchiveSafetyError("unsafe input: source changed")
    payload = b"".join(chunks)
    issues = _scan_payload(payload, relative_path, patterns)
    if issues:
        raise ArchiveSafetyError("unsafe input: " + ", ".join(issues))
    archive_path = "README.md" if relative_path == "README.anonymous.md" else relative_path
    return Member(archive_path, len(payload), hashlib.sha256(payload).hexdigest(), payload)


def collect_members(repo_root: Path) -> tuple[Member, ...]:
    """Validate the allowlisted source tree and return immutable archive member metadata."""
    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise ArchiveSafetyError("unsafe input: repository root is not a directory")
    patterns = load_forbidden_patterns()
    members: list[Member] = []
    issues: list[str] = []
    for source in _selected_source_paths(root):
        try:
            members.append(_read_source_once(source, root, patterns))
        except ArchiveSafetyError as error:
            issues.append(str(error).rsplit(": ", maxsplit=1)[-1])
    archive_names = [member.archive_path for member in members]
    if len(archive_names) != len(set(archive_names)):
        issues.append("duplicate-member")
    if issues:
        categories = ", ".join(sorted(set(issues)))
        raise ArchiveSafetyError(f"unsafe input: {categories}")
    return tuple(sorted(members, key=lambda member: member.archive_path))


def _validate_output_directory(output_dir: Path, repo_root: Path) -> Path:
    raw_output = output_dir.expanduser()
    absolute_output = raw_output if raw_output.is_absolute() else (Path.cwd() / raw_output)
    lexical_part = Path(absolute_output.anchor)
    for component in absolute_output.parts[1:]:
        lexical_part /= component
        if lexical_part.is_symlink():
            raise ArchiveSafetyError("unsafe output directory")
    existing = absolute_output
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if existing.is_symlink() or absolute_output.is_symlink():
        raise ArchiveSafetyError("unsafe output directory")
    output = absolute_output.resolve(strict=False)
    root = repo_root.resolve(strict=True)
    if output == output.parent or output == root:
        raise ArchiveSafetyError("unsafe output directory")
    if output.exists() and not output.is_dir():
        raise ArchiveSafetyError("unsafe output directory")
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or any(output.iterdir()):
        raise ArchiveSafetyError("unsafe output directory")
    return output


def _write_zip(path: Path, members: Iterable[Member]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for member in members:
            info = zipfile.ZipInfo(member.archive_path, date_time=FIXED_TIMESTAMP)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _manifest(members: Iterable[Member], zip_digest: str) -> dict[str, object]:
    return {
        "format": 1,
        "tool": f"anonymous-archive-builder/{TOOL_VERSION}",
        "zip_sha256": zip_digest,
        "members": [
            {"path": member.archive_path, "sha256": member.digest, "size": member.size}
            for member in members
        ],
    }


def _publish_no_clobber(output: Path, artifacts: tuple[tuple[str, bytes], ...]) -> None:
    """Publish only into an empty directory; never replace a competing file."""
    created: list[str] = []
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        for name, payload in artifacts:
            if set(os.listdir(output)) != set(created):
                raise ArchiveSafetyError("unsafe output directory")
            try:
                target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=descriptor)
            except FileExistsError as error:
                raise ArchiveSafetyError("unsafe output directory") from error
            try:
                with os.fdopen(target, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                try:
                    os.unlink(name, dir_fd=descriptor)
                except OSError:
                    pass
                raise
            created.append(name)
    except Exception:
        for name in created:
            try:
                os.unlink(name, dir_fd=descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(descriptor)


def build_archive(repo_root: Path, output_dir: Path) -> Path:
    """Build the ZIP and its sidecars atomically after all source validation succeeds."""
    members = collect_members(repo_root)
    output = _validate_output_directory(output_dir, repo_root)
    temporary_dir = Path(tempfile.mkdtemp(prefix="anonymous-archive-", dir=output.parent))
    try:
        temporary_zip = temporary_dir / ARCHIVE_NAME
        _write_zip(temporary_zip, members)
        zip_digest = hashlib.sha256(temporary_zip.read_bytes()).hexdigest()
        temporary_sha = f"{zip_digest}  {ARCHIVE_NAME}\n".encode("ascii")
        temporary_manifest = (json.dumps(_manifest(members, zip_digest), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        _publish_no_clobber(output, (
            (ARCHIVE_NAME, temporary_zip.read_bytes()),
            (f"{ARCHIVE_NAME}.sha256", temporary_sha),
            ("archive_manifest.json", temporary_manifest),
        ))
    except OSError as error:
        raise ArchiveSafetyError("unsafe output directory") from error
    finally:
        for temporary_path in temporary_dir.iterdir() if temporary_dir.exists() else ():
            temporary_path.unlink()
        if temporary_dir.exists():
            temporary_dir.rmdir()
    return output / ARCHIVE_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        archive = build_archive(arguments.repo_root, arguments.output_dir)
    except (ArchiveSafetyError, OSError, ValueError):
        print("anonymous archive build failed: unsafe input or output", file=sys.stderr)
        return 2
    print(f"anonymous archive created: {archive.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
