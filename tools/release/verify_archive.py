#!/usr/bin/env python3
"""Verify an anonymous archive without trusting its ZIP metadata or sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from build_anonymous_archive import MAX_FILE_SIZE, ArchiveSafetyError, load_forbidden_patterns, scan_file


MAX_MEMBERS = 10_000
MAX_TOTAL_UNCOMPRESSED = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100


def _archive_error() -> ArchiveSafetyError:
    return ArchiveSafetyError("unsafe archive")


def _safe_member_name(name: str) -> str:
    if not name or "\\" in name or name.startswith(("/", "\\")):
        raise _archive_error()
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _archive_error()
    if ":" in path.parts[0]:
        raise _archive_error()
    return path.as_posix()


def _preflight_members(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    infos = tuple(archive.infolist())
    if not infos or len(infos) > MAX_MEMBERS:
        raise _archive_error()
    names: set[str] = set()
    total_size = 0
    for info in infos:
        name = _safe_member_name(info.filename)
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode):
            raise _archive_error()
        if name in names or info.file_size > MAX_FILE_SIZE:
            raise _archive_error()
        if info.file_size and (not info.compress_size or info.file_size > info.compress_size * MAX_COMPRESSION_RATIO):
            raise _archive_error()
        total_size += info.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED:
            raise _archive_error()
        names.add(name)
    return infos


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sidecars(archive_path: Path, digest: str) -> dict[str, object] | None:
    sidecar = archive_path.with_name(f"{archive_path.name}.sha256")
    if sidecar.exists():
        if sidecar.is_symlink() or sidecar.read_text(encoding="ascii").strip() != f"{digest}  {archive_path.name}":
            raise _archive_error()
    manifest_path = archive_path.with_name("archive_manifest.json")
    if not manifest_path.exists():
        return None
    if manifest_path.is_symlink():
        raise _archive_error()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _archive_error() from error
    if not isinstance(manifest, dict) or manifest.get("zip_sha256") != digest:
        raise _archive_error()
    return manifest


def _extract_safely(archive: zipfile.ZipFile, infos: tuple[zipfile.ZipInfo, ...], destination: Path) -> dict[str, dict[str, object]]:
    extracted: dict[str, dict[str, object]] = {}
    destination_root = destination.resolve(strict=True)
    for info in infos:
        name = _safe_member_name(info.filename)
        target = destination / name
        if not _is_within(target.resolve(strict=False), destination_root):
            raise _archive_error()
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        try:
            with archive.open(info, "r") as source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > MAX_FILE_SIZE:
                        raise _archive_error()
                    digest.update(block)
                    output.write(block)
        except (OSError, zipfile.BadZipFile) as error:
            raise _archive_error() from error
        if count != info.file_size:
            raise _archive_error()
        extracted[name] = {"sha256": digest.hexdigest(), "size": count}
    return extracted


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_manifest(manifest: dict[str, object] | None, extracted: dict[str, dict[str, object]]) -> None:
    if manifest is None:
        return
    members = manifest.get("members")
    if not isinstance(members, list):
        raise _archive_error()
    expected: dict[str, dict[str, object]] = {}
    for entry in members:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise _archive_error()
        path = _safe_member_name(entry["path"])
        if path in expected or not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("size"), int):
            raise _archive_error()
        expected[path] = {"sha256": entry["sha256"], "size": entry["size"]}
    if expected != extracted:
        raise _archive_error()


def verify_archive(archive_path: Path) -> None:
    """Preflight, safely extract, bind sidecars, then independently rescan archive content."""
    if archive_path.suffix != ".zip" or archive_path.is_symlink() or not archive_path.is_file():
        raise _archive_error()
    digest = _sha256(archive_path)
    manifest = _load_sidecars(archive_path, digest)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _preflight_members(archive)
            if archive.testzip() is not None:
                raise _archive_error()
            temporary_root = Path(tempfile.mkdtemp(prefix="anonymous-verify-"))
            try:
                extracted = _extract_safely(archive, infos, temporary_root)
                patterns = load_forbidden_patterns()
                for path in sorted(temporary_root.rglob("*")):
                    if path.is_symlink() or (path.is_file() and scan_file(path, temporary_root, patterns)):
                        raise _archive_error()
                _verify_manifest(manifest, extracted)
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)
    except (OSError, zipfile.BadZipFile) as error:
        raise _archive_error() from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args(argv)
    try:
        verify_archive(arguments.archive)
    except (ArchiveSafetyError, OSError, ValueError):
        print("anonymous archive verification failed: unsafe archive", file=sys.stderr)
        return 2
    print("anonymous archive verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
