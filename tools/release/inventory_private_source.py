#!/usr/bin/env python3
"""Create a sanitized, read-only P2 inventory from a private Git worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from build_anonymous_archive import load_forbidden_patterns


FORMAT_VERSION = 1
TOOL_VERSION = "1.1.0"
MAX_TEXT_BYTES_DEFAULT = 1024 * 1024
SAFE_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

GENERATED_DIRECTORY_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "cache",
        "caches",
        "calibration",
        "checkpoint",
        "checkpoints",
        "ckpts",
        "engine",
        "engines",
        "logs",
        "model",
        "models",
        "onnx",
        "output",
        "outputs",
        "result",
        "results",
        "trt_engines",
        "tvm_offline_cp310",
        "tvm_venv310",
        "work_dirs",
    }
)
EXTERNAL_DATA_DIRECTORY_PARTS = frozenset(
    {
        "data",
        "dataset",
        "datasets",
        "input",
        "inputs",
        "test",
        "testing",
        "val",
        "valid",
        "validation",
    }
)
GENERATED_SUFFIXES = frozenset(
    {
        ".avi",
        ".bin",
        ".ckpt",
        ".dll",
        ".dylib",
        ".engine",
        ".gz",
        ".onnx",
        ".pdf",
        ".plan",
        ".pt",
        ".pth",
        ".pkl",
        ".pyc",
        ".so",
        ".tar",
        ".trt",
        ".whl",
        ".zip",
    }
)
CODE_SUFFIXES = frozenset({".py"})
DOCUMENT_SUFFIXES = frozenset({".md", ".rst", ".txt"})
CONFIG_SUFFIXES = frozenset({".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"})
PAPER_SOURCE_SUFFIXES = frozenset({".bib", ".bst", ".cls", ".sty", ".tex"})


class InventorySafetyError(Exception):
    """A safe-to-display P2 inventory failure."""


def _git_output(root: Path, *arguments: str) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InventorySafetyError("source root is not a readable Git worktree") from error
    return tuple(
        path.decode("utf-8", "surrogateescape")
        for path in completed.stdout.split(b"\0")
        if path
    )


def _validate_source_root(raw_root: Path) -> Path:
    try:
        root = raw_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise InventorySafetyError("source root is not a readable Git worktree") from error
    if not root.is_dir():
        raise InventorySafetyError("source root is not a directory")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InventorySafetyError("source root is not a readable Git worktree") from error
    if Path(completed.stdout.strip()).resolve(strict=True) != root:
        raise InventorySafetyError("source root must be the Git worktree root")
    return root


def _validate_output(raw_output: Path, source_root: Path) -> Path:
    output = raw_output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).absolute()
    if output.exists() or output.is_symlink():
        raise InventorySafetyError("output path must not already exist")
    parent = output.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise InventorySafetyError("output parent is not a safe directory")
    resolved_parent = parent.resolve(strict=True)
    resolved_output = resolved_parent / output.name
    try:
        resolved_output.relative_to(source_root)
    except ValueError:
        return resolved_output
    raise InventorySafetyError("output path must be outside the source worktree")


def _path_id(origin: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{origin}\0{relative_path}".encode("utf-8", "surrogateescape"))
    return digest.hexdigest()[:20]


def _has_surrogate(value: str) -> bool:
    return any("\ud800" <= character <= "\udfff" for character in value)


def _classify_path(relative_path: str) -> tuple[str, tuple[str, ...]]:
    path = Path(relative_path)
    lower_parts = tuple(part.casefold() for part in path.parts)
    suffix = path.suffix.casefold()
    if any(part in GENERATED_DIRECTORY_PARTS for part in lower_parts):
        return "exclude_generated", ("generated-directory",)
    if any(part in EXTERNAL_DATA_DIRECTORY_PARTS for part in lower_parts):
        return "external_input", ("data-boundary",)
    if suffix in GENERATED_SUFFIXES:
        return "exclude_generated", ("generated-suffix",)
    if suffix == ".sh":
        return "review_shell", ("shell-script",)
    if "fixtures" in lower_parts and suffix in CONFIG_SUFFIXES:
        return "fixture_candidate", ("small-fixture",)
    if suffix in CODE_SUFFIXES:
        return "migrate_code", ("source-code",)
    if suffix in DOCUMENT_SUFFIXES:
        return "migrate_document", ("documentation",)
    if suffix in CONFIG_SUFFIXES:
        return "migrate_config", ("configuration",)
    if suffix in PAPER_SOURCE_SUFFIXES:
        return "review_paper_source", ("paper-source",)
    return "review_other", ("unsupported-suffix",)


def _is_migration_candidate(classification: str) -> bool:
    return classification in {
        "fixture_candidate",
        "migrate_code",
        "migrate_config",
        "migrate_document",
    }


def _opaque_entry(
    origin: str, relative_path: str, classification: str, reasons: Iterable[str], size: int
) -> dict[str, object]:
    return {
        "id": _path_id(origin, relative_path),
        "origin": origin,
        "classification": classification,
        "reasons": sorted(set(reasons)),
        "bytes": size,
    }


def _nested_repositories(source_root: Path) -> tuple[str, ...]:
    """Find only shallow nested repositories without walking generated directories."""
    discovered: list[str] = []
    for current, directories, _ in os.walk(source_root, topdown=True):
        relative = Path(current).relative_to(source_root)
        depth = len(relative.parts)
        if depth >= 3:
            directories.clear()
            continue
        directories[:] = [
            name
            for name in directories
            if name.casefold() not in GENERATED_DIRECTORY_PARTS and not name.startswith(".")
        ]
        if relative == Path("."):
            continue
        git_marker = Path(current) / ".git"
        if git_marker.exists():
            candidate = relative.as_posix()
            if not _has_surrogate(candidate):
                discovered.append(candidate)
            directories.clear()
    return tuple(sorted(set(discovered)))


def _open_regular_file_beneath_root(source_root: Path, relative_path: str) -> int:
    """Open a regular file without following directory or final-component symlinks."""
    parts = Path(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InventorySafetyError("source entry has an unsafe path")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(source_root, directory_flags)
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(parts[-1], file_flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise InventorySafetyError("source entry is not safely readable") from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _is_single_link_regular_file(metadata: os.stat_result) -> bool:
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _stable_metadata(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_payload(descriptor: int, max_text_bytes: int) -> tuple[int, bytes]:
    """Read a pinned regular file and reject changes while it is scanned."""
    try:
        before = os.fstat(descriptor)
        if not _is_single_link_regular_file(before):
            raise InventorySafetyError("source entry is not a regular file")
        if before.st_size > max_text_bytes:
            after = os.fstat(descriptor)
            if _stable_metadata(before) != _stable_metadata(after):
                raise InventorySafetyError("source entry changed during scanning")
            return before.st_size, b""

        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            if total > max_text_bytes:
                raise InventorySafetyError("source entry exceeded the text size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise InventorySafetyError("source entry is not safely readable") from error
    if _stable_metadata(before) != _stable_metadata(after):
        raise InventorySafetyError("source entry changed during scanning")
    return before.st_size, b"".join(chunks)


def _scan_payload(
    payload: bytes, relative_path: str, patterns: Iterable[tuple[str, re.Pattern[str]]]
) -> tuple[str, ...]:
    issues = [category for category, pattern in patterns if pattern.search(relative_path)]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        issues.append("unknown-binary")
    else:
        issues.extend(category for category, pattern in patterns if pattern.search(text))
    return tuple(sorted(set(issues)))


def build_inventory(
    source_root: Path,
    source_label: str,
    include_untracked: bool = True,
    max_text_bytes: int = MAX_TEXT_BYTES_DEFAULT,
) -> dict[str, object]:
    """Return a deterministic sanitized P2 classification for one Git worktree."""
    if not SAFE_LABEL_PATTERN.fullmatch(source_label):
        raise InventorySafetyError("source label has unsupported characters")
    if max_text_bytes <= 0:
        raise InventorySafetyError("max text bytes must be positive")

    tracked = _git_output(source_root, "ls-files", "-z")
    untracked = _git_output(source_root, "ls-files", "--others", "--exclude-standard", "-z")
    entries = [("tracked", path) for path in tracked]
    if include_untracked:
        entries.extend(("untracked", path) for path in untracked)

    patterns = load_forbidden_patterns()
    candidates: list[dict[str, object]] = []
    opaque_entries: list[dict[str, object]] = []
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})

    for origin, relative_path in sorted(entries, key=lambda item: (item[1], item[0])):
        if _has_surrogate(relative_path):
            classification = "review_sensitive"
            reasons = ("path-encoding",)
            opaque_entries.append(_opaque_entry(origin, relative_path, classification, reasons, 0))
            summary[classification]["count"] += 1
            continue
        try:
            descriptor = _open_regular_file_beneath_root(source_root, relative_path)
            try:
                metadata = os.fstat(descriptor)
                if not _is_single_link_regular_file(metadata):
                    raise InventorySafetyError("source entry is not a regular file")
                size = metadata.st_size
                classification, reasons = _classify_path(relative_path)
                payload = b""
                if _is_migration_candidate(classification):
                    size, payload = _read_stable_payload(descriptor, max_text_bytes)
            finally:
                os.close(descriptor)
        except (InventorySafetyError, OSError):
            classification = "review_sensitive"
            reasons = ("unreadable",)
            opaque_entries.append(_opaque_entry(origin, relative_path, classification, reasons, 0))
            summary[classification]["count"] += 1
            continue
        if _is_migration_candidate(classification):
            if size > max_text_bytes:
                classification = "review_large_text"
                reasons = ("text-size-limit",)
            else:
                issues = _scan_payload(payload, relative_path, patterns)
                if issues:
                    classification = "review_sensitive"
                    reasons = tuple(issues)
        summary[classification]["count"] += 1
        summary[classification]["bytes"] += size
        if _is_migration_candidate(classification):
            candidates.append(
                {
                    "path": relative_path,
                    "origin": origin,
                    "classification": classification,
                    "bytes": size,
                }
            )
        else:
            opaque_entries.append(_opaque_entry(origin, relative_path, classification, reasons, size))

    return {
        "format": FORMAT_VERSION,
        "tool": f"private-source-inventory/{TOOL_VERSION}",
        "source": {
            "label": source_label,
            "tracked_files": len(tracked),
            "untracked_files_considered": len(untracked) if include_untracked else 0,
            "nested_repositories": list(_nested_repositories(source_root)),
        },
        "policy": {
            "max_text_bytes": max_text_bytes,
            "candidate_classes": [
                "migrate_code",
                "migrate_document",
                "migrate_config",
                "fixture_candidate",
            ],
            "opaque_entries_exclude_private_paths": True,
        },
        "summary": [
            {"classification": classification, **values}
            for classification, values in sorted(summary.items())
        ],
        "candidates": sorted(candidates, key=lambda entry: (str(entry["path"]), str(entry["origin"]))),
        "opaque_entries": sorted(
            opaque_entries, key=lambda entry: (str(entry["classification"]), str(entry["id"]))
        ),
    }


def _write_output(output: Path, inventory: dict[str, object]) -> None:
    payload = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    temporary_name = f".{output.name}.{secrets.token_hex(16)}.tmp"
    directory_descriptor: int | None = None
    temporary_created = False
    try:
        directory_descriptor = os.open(output.parent, directory_flags)
        try:
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            temporary_created = True
            with os.fdopen(temporary_descriptor, "wb") as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, 0o644, dir_fd=directory_descriptor, follow_symlinks=False)
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            os.fsync(directory_descriptor)
        except FileExistsError as error:
            raise InventorySafetyError("output path must not already exist") from error
        except OSError as error:
            raise InventorySafetyError("unable to write inventory") from error
    except OSError as error:
        raise InventorySafetyError("unable to write inventory") from error
    finally:
        if temporary_created and directory_descriptor is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-text-bytes", type=int, default=MAX_TEXT_BYTES_DEFAULT)
    parser.add_argument("--no-untracked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        source_root = _validate_source_root(arguments.source_root)
        output = _validate_output(arguments.output, source_root)
        inventory = build_inventory(
            source_root,
            arguments.source_label,
            include_untracked=not arguments.no_untracked,
            max_text_bytes=arguments.max_text_bytes,
        )
        _write_output(output, inventory)
    except InventorySafetyError as error:
        print(f"private inventory failed: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("private inventory failed: unable to complete inventory", file=sys.stderr)
        return 2
    print("private inventory created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
