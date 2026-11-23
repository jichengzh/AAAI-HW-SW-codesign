"""Reuse an identical cached inputs.npy body without reserializing sample packs.

Only this project's two-member, unencrypted np.savez ZIP_STORED format is
supported. No compression, network access, or fallback to different input data.
Hashes detect corruption, not authenticity: transfer splice.json securely.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import zipfile

import numpy as np


CHUNK = 1024 * 1024


def _copy(source, destination=None, length=None, full_hash=None):
    digest = hashlib.sha256()
    remaining = length
    while remaining is None or remaining:
        block = source.read(CHUNK if remaining is None else min(CHUNK, remaining))
        if not block:
            if remaining:
                raise ValueError("Truncated input body")
            break
        digest.update(block)
        if full_hash is not None:
            full_hash.update(block)
        if destination is not None:
            destination.write(block)
        if remaining is not None:
            remaining -= len(block)
    return digest.hexdigest()


def _inspect(path):
    """Validate shapes/finiteness in bounded chunks; return stored body extent."""
    try:
        with zipfile.ZipFile(path) as archive, open(path, "rb") as raw:
            members = archive.infolist()
            if len(members) != 2 or {m.filename for m in members} != {"inputs.npy", "source_logits.npy"}:
                raise ValueError("Expected only inputs.npy and source_logits.npy")
            batch = None
            extent = None
            for name, tail in [("inputs.npy", (3, 224, 224)), ("source_logits.npy", (1000,))]:
                member = archive.getinfo(name)
                if member.compress_type != zipfile.ZIP_STORED or member.flag_bits != 0:
                    raise ValueError("Only unencrypted ZIP_STORED without data descriptors is supported")
                raw.seek(member.header_offset)
                header = raw.read(30)
                if len(header) != 30:
                    raise ValueError("Truncated ZIP local header")
                signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
                if signature != b"PK\x03\x04" or flags != member.flag_bits or method != zipfile.ZIP_STORED:
                    raise ValueError("Invalid ZIP local header")
                if raw.read(name_len) != name.encode("ascii"):
                    raise ValueError("ZIP local filename mismatch")
                start = member.header_offset + 30 + name_len + extra_len
                if start + member.file_size > archive.start_dir:
                    raise ValueError("Invalid ZIP body extent")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version != (1, 0):
                        raise ValueError("Only np.savez NPY version 1.0 is supported")
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    if dtype != np.dtype("float32") or len(shape) != len(tail) + 1 or shape[1:] != tail:
                        raise ValueError(f"Invalid {name}: expected float32[N,{tail}]")
                    if not 1 <= shape[0] <= 256 or (batch is not None and batch != shape[0]):
                        raise ValueError("Invalid or mismatched batch N; expected 1..256")
                    batch = shape[0]
                    remaining = int(np.prod(shape)) * dtype.itemsize
                    if stream.tell() + remaining != member.file_size:
                        raise ValueError("NPY body length mismatch")
                    while remaining:
                        block = stream.read(min(CHUNK, remaining))
                        if not block or len(block) % 4 or not np.isfinite(np.frombuffer(block, dtype=dtype)).all():
                            raise ValueError(f"Invalid or nonfinite {name} data")
                        remaining -= len(block)
                    if stream.read(1):
                        raise ValueError("Trailing NPY data")
                if name == "inputs.npy":
                    extent = (start, member.file_size)
            return extent
    except (zipfile.BadZipFile, EOFError, struct.error) as error:
        raise ValueError(f"Invalid sample pack: {error}") from error


def make(source_pack, output_dir):
    """Write original prefix/suffix and hashes to a new patch directory."""
    source_pack, output_dir = Path(source_pack), Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    start, length = _inspect(source_pack)
    output_dir.mkdir()
    full_hash = hashlib.sha256()
    with source_pack.open("rb") as source:
        with (output_dir / "prefix.bin").open("xb") as prefix:
            prefix_hash = _copy(source, prefix, start, full_hash)
        body_hash = _copy(source, length=length, full_hash=full_hash)
        with (output_dir / "suffix.bin").open("xb") as suffix:
            suffix_hash = _copy(source, suffix, full_hash=full_hash)
    metadata = {"schema_version": 1, "source_pack_sha256": full_hash.hexdigest(),
                "inputs_body_sha256": body_hash, "inputs_body_length": length,
                "prefix_sha256": prefix_hash, "suffix_sha256": suffix_hash}
    with (output_dir / "splice.json").open("x", encoding="utf-8") as manifest:
        json.dump(metadata, manifest, indent=2, sort_keys=True)
        manifest.write("\n")
    return metadata


def restore(cache_pack, patch_dir, output_pack):
    """Rebuild byte-exactly, verify every hash, then publish without overwrite."""
    cache_pack, patch_dir, output_pack = Path(cache_pack), Path(patch_dir), Path(output_pack)
    if os.path.lexists(output_pack):
        raise FileExistsError(f"Output pack already exists: {output_pack}")
    try:
        metadata = json.loads((patch_dir / "splice.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Unsupported splice manifest schema")
        for key in ("source_pack_sha256", "inputs_body_sha256", "prefix_sha256", "suffix_sha256"):
            value = metadata.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid SHA256 field: {key}")
        if type(metadata.get("inputs_body_length")) is not int or metadata["inputs_body_length"] <= 0:
            raise ValueError("Invalid inputs body length")
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid splice manifest: {error}") from error
    start, length = _inspect(cache_pack)
    if length != metadata["inputs_body_length"]:
        raise ValueError("Cached inputs body length mismatch")
    temporary = None
    try:
        full_hash = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=output_pack.parent, prefix=".sample-splice-", delete=False) as output:
            temporary = Path(output.name)
            with (patch_dir / "prefix.bin").open("rb") as prefix:
                if _copy(prefix, output, full_hash=full_hash) != metadata["prefix_sha256"]:
                    raise ValueError("Prefix SHA256 mismatch")
            with cache_pack.open("rb") as cache:
                cache.seek(start)
                if _copy(cache, output, length, full_hash) != metadata["inputs_body_sha256"]:
                    raise ValueError("Cached inputs body SHA256 mismatch")
            with (patch_dir / "suffix.bin").open("rb") as suffix:
                if _copy(suffix, output, full_hash=full_hash) != metadata["suffix_sha256"]:
                    raise ValueError("Suffix SHA256 mismatch")
            if full_hash.hexdigest() != metadata["source_pack_sha256"]:
                raise ValueError("Restored source pack SHA256 mismatch")
        _inspect(temporary)
        os.link(temporary, output_pack)  # atomic exclusive creation, even if a destination races us
    finally:
        if temporary is not None:
            temporary.unlink()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("make")
    create.add_argument("--source-pack", required=True)
    create.add_argument("--output-dir", required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--cache-pack", required=True)
    recover.add_argument("--patch-dir", required=True)
    recover.add_argument("--output-pack", required=True)
    args = parser.parse_args()
    try:
        result = make(args.source_pack, args.output_dir) if args.command == "make" else restore(args.cache_pack, args.patch_dir, args.output_pack)
    except (OSError, ValueError) as error:
        parser.exit(1, f"sample splice: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
