from __future__ import print_function

import copy
import ctypes
import ctypes.util
import hashlib
import json
import re
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


PROTOCOL_VERSION = "ditl.edge.v1"
MAGIC = b"DITL1"
FRAME_PREFIX = struct.Struct("!5sIQ")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
ARRAY_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
ALLOWED_KINDS = frozenset(("request", "response", "result", "error"))
ALLOWED_COMPRESSIONS = frozenset(("none", "zlib", "lz4"))
LZ4_ACCELERATION = 2
_LZ4_LIBRARY = None
FEATURE_CHECKSUM_SCOPE = "codriving_feature_payload.v1"
_FEATURE_NAMES = ("res0", "res1", "res2")
_VERIFICATION_TOKEN = object()
ALLOWED_DTYPES = {
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "bool": np.dtype("bool"),
}


class ProtocolError(ValueError):
    pass


def _immutable_array_signature(value):
    if (type(value) is not np.ndarray or value.flags.writeable
            or not value.flags.c_contiguous):
        return None
    owner = value
    while isinstance(owner, (np.ndarray, memoryview)):
        if isinstance(owner, memoryview):
            if not owner.readonly:
                return None
            owner = owner.obj
        else:
            if owner.flags.writeable:
                return None
            owner = owner.base
    if type(owner) is not bytes:
        return None
    return (id(value), value.shape, value.dtype.str, value.strides,
            value.__array_interface__["data"][0], value.nbytes)


class _VerifiedFeatureArrays(Mapping):
    """In-process proof only; never accepted from wire metadata or dict flags."""
    __slots__ = ("_arrays", "_digest", "_signatures")

    def __init__(self, arrays, digest, token):
        if token is not _VERIFICATION_TOKEN:
            raise TypeError("feature verification proof is internal")
        signatures = tuple((name, _immutable_array_signature(arrays[name]))
                           for name in _FEATURE_NAMES)
        if any(signature is None for _, signature in signatures):
            raise ProtocolError("feature proof requires immutable bytes storage")
        object.__setattr__(self, "_arrays", MappingProxyType(dict(arrays)))
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_signatures", signatures)

    def __setattr__(self, name, value):
        raise AttributeError("feature verification proof is immutable")

    def __getitem__(self, name):
        return self._arrays[name]

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self):
        return len(self._arrays)


def _verified_feature_arrays(arrays, digest):
    # Internal callers must have just computed/verified the canonical digest.
    return _VerifiedFeatureArrays(arrays, digest, _VERIFICATION_TOKEN)


def verified_feature_payload_sha256(outputs):
    """Return a previously checked canonical digest only for its exact arrays.

    Array descriptors are mutable even with immutable backing bytes. Recheck
    object identity, shape, dtype, strides and storage before reusing the proof.
    Plain dictionaries, copies and views deliberately carry no proof.
    """
    if type(outputs) is not _VerifiedFeatureArrays:
        return None
    signatures = tuple((name, _immutable_array_signature(outputs[name]))
                       for name in _FEATURE_NAMES)
    return outputs._digest if signatures == outputs._signatures else None


def canonical_feature_payload_sha256(outputs):
    """Canonical CoDriving identity/bytes checksum (existing payload_sha256)."""
    if not isinstance(outputs, Mapping) or tuple(sorted(outputs)) != _FEATURE_NAMES:
        raise ProtocolError("feature payload must contain only res0/res1/res2")
    digest = hashlib.sha256()
    for name in _FEATURE_NAMES:
        value = np.ascontiguousarray(np.asarray(outputs[name]))
        descriptor = json.dumps(
            {"dtype": value.dtype.name, "name": name, "shape": list(value.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, byteorder="big"))
        digest.update(descriptor)
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _checksum_scope(metadata):
    scope = metadata.get("sha256_scope")
    if "sha256_scope" in metadata and scope != FEATURE_CHECKSUM_SCOPE:
        raise ProtocolError("unsupported tensor sha256 scope")
    return scope


def _clock_ns():
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if clock is not None and hasattr(time, "clock_gettime_ns"):
        return time.clock_gettime_ns(clock)
    return time.monotonic_ns()


@dataclass(frozen=True)
class ProtocolLimits:
    max_header_bytes: int = 1024 * 1024
    max_payload_bytes: int = 512 * 1024 * 1024
    max_raw_payload_bytes: int = 512 * 1024 * 1024
    max_tensor_count: int = 32
    max_tensor_rank: int = 8
    max_dimension: int = 16384


def _validate_limits(limits):
    if not isinstance(limits, ProtocolLimits):
        raise ProtocolError("limits must be ProtocolLimits")
    for name, value in vars(limits).items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProtocolError("limits.%s must be a positive integer" % name)


def _validate_header_identity(header):
    if not isinstance(header, dict):
        raise ProtocolError("header must be an object")
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("protocol_version must equal %s" % PROTOCOL_VERSION)
    if header.get("kind") not in ALLOWED_KINDS:
        raise ProtocolError("kind must be request, response, result or error")
    request_id = header.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.match(request_id):
        raise ProtocolError("request_id contains invalid characters")


def _canonical_array(value, name, limits):
    if not isinstance(name, str) or not ARRAY_NAME_RE.match(name):
        raise ProtocolError("invalid tensor name %r" % name)
    array = np.asarray(value)
    dtype_name = array.dtype.name
    if dtype_name not in ALLOWED_DTYPES:
        raise ProtocolError("tensor %s has unsupported dtype %s" % (name, dtype_name))
    if array.ndim > limits.max_tensor_rank:
        raise ProtocolError("tensor %s rank exceeds limit" % name)
    for dimension in array.shape:
        if dimension < 0 or dimension > limits.max_dimension:
            raise ProtocolError("tensor %s shape exceeds limit" % name)
    return np.ascontiguousarray(array)


def _load_lz4():
    global _LZ4_LIBRARY
    if _LZ4_LIBRARY is not None:
        return _LZ4_LIBRARY
    library_name = ctypes.util.find_library("lz4")
    if not library_name:
        raise ProtocolError("LZ4 codec is unavailable")
    try:
        library = ctypes.CDLL(library_name)
    except OSError as error:
        raise ProtocolError("LZ4 codec is unavailable: %s" % error)
    library.LZ4_compressBound.argtypes = [ctypes.c_int]
    library.LZ4_compressBound.restype = ctypes.c_int
    library.LZ4_compress_fast.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.LZ4_compress_fast.restype = ctypes.c_int
    library.LZ4_decompress_safe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.LZ4_decompress_safe.restype = ctypes.c_int
    _LZ4_LIBRARY = library
    return _LZ4_LIBRARY


def _lz4_compress(raw):
    if not raw:
        return b""
    if len(raw) > 2 ** 31 - 1:
        raise ProtocolError("raw payload exceeds LZ4 size limit")
    library = _load_lz4()
    capacity = int(library.LZ4_compressBound(len(raw)))
    destination = ctypes.create_string_buffer(capacity)
    source_address = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    compressed_size = int(
        library.LZ4_compress_fast(
            source_address,
            ctypes.addressof(destination),
            len(raw),
            capacity,
            LZ4_ACCELERATION,
        )
    )
    if compressed_size <= 0:
        raise ProtocolError("LZ4 compression failed")
    return destination.raw[:compressed_size]


def _lz4_decompress(payload, raw_size):
    if raw_size == 0:
        if payload:
            raise ProtocolError("LZ4 payload must be empty for an empty bundle")
        return b""
    if not payload or raw_size > 2 ** 31 - 1 or len(payload) > 2 ** 31 - 1:
        raise ProtocolError("invalid LZ4 payload size")
    library = _load_lz4()
    source = ctypes.create_string_buffer(bytes(payload), len(payload))
    destination = ctypes.create_string_buffer(raw_size)
    decoded_size = int(
        library.LZ4_decompress_safe(
            ctypes.addressof(source),
            ctypes.addressof(destination),
            len(payload),
            raw_size,
        )
    )
    if decoded_size != raw_size:
        raise ProtocolError("LZ4 decompression failed")
    return destination.raw[:decoded_size]


def encode_tensor_bundle(
    arrays,
    compression="none",
    limits=ProtocolLimits(),
    zlib_level=1,
    feature_checksum=False,
):
    """Encode named ndarrays without pickle or executable metadata."""
    _validate_limits(limits)
    if not isinstance(arrays, dict) and not (feature_checksum and isinstance(arrays, Mapping)):
        raise ProtocolError("arrays must be an object")
    if len(arrays) > limits.max_tensor_count:
        raise ProtocolError("tensor count exceeds limit")
    if compression not in ALLOWED_COMPRESSIONS:
        raise ProtocolError("compression must be none, zlib or lz4")
    if feature_checksum and compression != "none":
        raise ProtocolError("feature checksum optimization requires compression none")
    if (
        isinstance(zlib_level, bool)
        or not isinstance(zlib_level, int)
        or not 0 <= zlib_level <= 9
    ):
        raise ProtocolError("zlib_level must be an integer between 0 and 9")

    pack_start = _clock_ns()
    canonical_arrays = []
    descriptors = []
    offset = 0
    for name in sorted(arrays):
        array = _canonical_array(arrays[name], name, limits)
        raw_size = int(array.nbytes)
        descriptors.append(
            {
                "name": name,
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": raw_size,
            }
        )
        canonical_arrays.append(array)
        offset += raw_size
        if offset > limits.max_raw_payload_bytes:
            raise ProtocolError("raw payload exceeds limit")
    if compression == "none":
        # Build the final immutable payload once, without bytearray -> bytes copying.
        raw_buffer = b"".join(memoryview(array).cast("B") for array in canonical_arrays)
    else:
        raw_buffer = bytearray(offset)
        raw_view = memoryview(raw_buffer)
        for descriptor, array in zip(descriptors, canonical_arrays):
            start = descriptor["offset"]
            end = start + descriptor["nbytes"]
            raw_view[start:end] = memoryview(array).cast("B")
    pack_end = _clock_ns()

    hash_start = _clock_ns()
    if feature_checksum:
        digest = verified_feature_payload_sha256(arrays)
        if digest is None:
            # Hash the final packed snapshot, not mutable caller-owned memory.
            packed = {
                descriptor["name"]: np.frombuffer(
                    raw_buffer, dtype=ALLOWED_DTYPES[descriptor["dtype"]],
                    count=descriptor["nbytes"] // ALLOWED_DTYPES[descriptor["dtype"]].itemsize,
                    offset=descriptor["offset"],
                ).reshape(descriptor["shape"])
                for descriptor in descriptors
            }
            digest = canonical_feature_payload_sha256(packed)
    else:
        digest = hashlib.sha256(raw_buffer).hexdigest()
    hash_end = _clock_ns()

    compress_start = _clock_ns()
    if compression == "zlib":
        payload = zlib.compress(raw_buffer, level=zlib_level)
    elif compression == "lz4":
        payload = _lz4_compress(raw_buffer)
    else:
        payload = raw_buffer
    compress_end = _clock_ns()
    if len(payload) > limits.max_payload_bytes:
        raise ProtocolError("payload exceeds limit")
    metadata = {
        "compression": compression,
        "raw_payload_bytes": len(raw_buffer),
        "wire_payload_bytes": len(payload),
        "sha256": digest,
        "arrays": descriptors,
        "codec_stages_ns": {
            "pack_ns": max(0, pack_end - pack_start),
            "hash_ns": max(0, hash_end - hash_start),
            "compress_ns": (
                max(0, compress_end - compress_start)
                if compression != "none"
                else 0
            ),
        },
        "codec_timestamps_ns": {
            "pack_start": pack_start,
            "pack_end": pack_end,
            "hash_start": hash_start,
            "hash_end": hash_end,
            "compress_start": compress_start,
            "compress_end": compress_end,
        },
    }
    if compression == "zlib":
        metadata["zlib_level"] = zlib_level
    elif compression == "lz4":
        metadata["lz4_acceleration"] = LZ4_ACCELERATION
    if feature_checksum:
        metadata["sha256_scope"] = FEATURE_CHECKSUM_SCOPE
    return metadata, payload


def _validate_descriptor(descriptor, seen_names, raw_size, limits):
    if not isinstance(descriptor, dict):
        raise ProtocolError("tensor descriptor must be an object")
    required = ("name", "dtype", "shape", "offset", "nbytes")
    for field in required:
        if field not in descriptor:
            raise ProtocolError("tensor descriptor.%s is required" % field)
    name = descriptor["name"]
    if not isinstance(name, str) or not ARRAY_NAME_RE.match(name):
        raise ProtocolError("invalid tensor name")
    if name in seen_names:
        raise ProtocolError("duplicate tensor name %s" % name)
    seen_names.add(name)
    dtype_name = descriptor["dtype"]
    if dtype_name not in ALLOWED_DTYPES:
        raise ProtocolError("tensor %s has unsupported dtype" % name)
    shape = descriptor["shape"]
    if not isinstance(shape, list) or len(shape) > limits.max_tensor_rank:
        raise ProtocolError("tensor %s has invalid rank" % name)
    item_count = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ProtocolError("tensor %s shape must contain integers" % name)
        if dimension < 0 or dimension > limits.max_dimension:
            raise ProtocolError("tensor %s shape exceeds limit" % name)
        item_count *= dimension
    offset = descriptor["offset"]
    nbytes = descriptor["nbytes"]
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or nbytes < 0
    ):
        raise ProtocolError("tensor %s offset/nbytes are invalid" % name)
    expected_nbytes = item_count * ALLOWED_DTYPES[dtype_name].itemsize
    if nbytes != expected_nbytes:
        raise ProtocolError("tensor %s nbytes does not match shape and dtype" % name)
    if offset + nbytes > raw_size:
        raise ProtocolError("tensor %s exceeds payload bounds" % name)
    return offset, offset + nbytes


def _decompress_zlib_bounded(payload, max_raw_payload_bytes):
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(
            bytes(payload), max_raw_payload_bytes + 1
        )
        if len(raw) > max_raw_payload_bytes or decompressor.unconsumed_tail:
            raise ProtocolError("raw payload exceeds limit")
        raw += decompressor.flush(max_raw_payload_bytes + 1 - len(raw))
    except zlib.error as error:
        raise ProtocolError("invalid zlib payload: %s" % error)
    if len(raw) > max_raw_payload_bytes:
        raise ProtocolError("raw payload exceeds limit")
    if not decompressor.eof or decompressor.unused_data:
        raise ProtocolError("invalid zlib payload framing")
    return raw


def decode_tensor_bundle(
    metadata,
    payload,
    limits=ProtocolLimits(),
    return_stages=False,
    readonly=False,
):
    _validate_limits(limits)
    if not isinstance(metadata, dict):
        raise ProtocolError("tensor_bundle must be an object")
    scope = _checksum_scope(metadata)
    if not isinstance(payload, (bytes, bytearray)):
        raise ProtocolError("payload must be bytes")
    if len(payload) > limits.max_payload_bytes:
        raise ProtocolError("payload exceeds limit")
    compression = metadata.get("compression")
    if compression not in ALLOWED_COMPRESSIONS:
        raise ProtocolError("unsupported tensor compression")
    raw_size = metadata.get("raw_payload_bytes")
    if (
        isinstance(raw_size, bool)
        or not isinstance(raw_size, int)
        or raw_size < 0
        or raw_size > limits.max_raw_payload_bytes
    ):
        raise ProtocolError("raw payload exceeds limit")
    decompress_start = _clock_ns()
    if compression == "zlib":
        raw = _decompress_zlib_bounded(payload, limits.max_raw_payload_bytes)
    elif compression == "lz4":
        raw = _lz4_decompress(payload, raw_size)
    else:
        raw = bytes(payload)
    decompress_end = _clock_ns()
    if len(raw) > limits.max_raw_payload_bytes:
        raise ProtocolError("raw payload exceeds limit")
    if raw_size != len(raw):
        raise ProtocolError("raw payload length mismatch")
    if metadata.get("wire_payload_bytes") != len(payload):
        raise ProtocolError("wire payload length mismatch")
    verify_start = _clock_ns()
    expected_hash = metadata.get("sha256")
    actual_hash = hashlib.sha256(raw).hexdigest() if scope is None else None
    if scope is None and (not isinstance(expected_hash, str) or expected_hash != actual_hash):
        prefix = "LZ4 " if compression == "lz4" else ""
        raise ProtocolError(prefix + "tensor payload sha256 mismatch")
    verify_end = _clock_ns()
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) > limits.max_tensor_count:
        raise ProtocolError("invalid tensor descriptor list")
    reconstruct_start = _clock_ns()
    seen_names = set()
    ranges = []
    output = {}
    for descriptor in descriptors:
        start, end = _validate_descriptor(descriptor, seen_names, len(raw), limits)
        for other_start, other_end in ranges:
            if start < other_end and other_start < end:
                raise ProtocolError("tensor payload ranges overlap")
        ranges.append((start, end))
        dtype = ALLOWED_DTYPES[descriptor["dtype"]]
        array = np.frombuffer(
            raw, dtype=dtype, count=(end - start) // dtype.itemsize, offset=start,
        ).reshape(descriptor["shape"])
        # bytes pins immutable storage for the lifetime of every returned view.
        # Default callers retain independently owned writable ndarray semantics.
        output[descriptor["name"]] = array if readonly else array.copy()
    covered = sum(end - start for start, end in ranges)
    if covered != len(raw):
        raise ProtocolError("tensor descriptors do not cover payload exactly")
    reconstruct_end = _clock_ns()
    if scope == FEATURE_CHECKSUM_SCOPE:
        verify_start = _clock_ns()
        actual_hash = canonical_feature_payload_sha256(output)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise ProtocolError("canonical feature payload sha256 mismatch")
        verify_end = _clock_ns()
        if readonly:
            output = _verified_feature_arrays(output, actual_hash)
    if not return_stages:
        return output
    return output, {
        "decompress_ns": (
            max(0, decompress_end - decompress_start)
            if compression != "none"
            else 0
        ),
        "verify_ns": max(0, verify_end - verify_start),
        "reconstruct_ns": max(0, reconstruct_end - reconstruct_start),
        "timestamps_ns": {
            "decompress_start": decompress_start,
            "decompress_end": decompress_end,
            "verify_start": verify_start,
            "verify_end": verify_end,
            "reconstruct_start": reconstruct_start,
            "reconstruct_end": reconstruct_end,
        },
    }


def prepare_envelope(header, arrays, compression="none", limits=ProtocolLimits()):
    _validate_limits(limits)
    _validate_header_identity(header)
    metadata, payload = encode_tensor_bundle(arrays, compression, limits)
    return prepare_envelope_header(header, metadata, limits), payload


def prepare_envelope_header(header, tensor_bundle, limits=ProtocolLimits()):
    """Serialize a header around metadata produced by encode_tensor_bundle."""
    _validate_limits(limits)
    _validate_header_identity(header)
    if not isinstance(tensor_bundle, dict):
        raise ProtocolError("tensor_bundle must be an object")
    wire_header = copy.deepcopy(header)
    wire_header["tensor_bundle"] = copy.deepcopy(tensor_bundle)
    try:
        header_bytes = json.dumps(
            wire_header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError("header is not JSON serializable: %s" % error)
    if len(header_bytes) > limits.max_header_bytes:
        raise ProtocolError("header exceeds limit")
    return header_bytes


def send_wire_frame(sock, header_bytes, payload, chunk_bytes=None):
    wire = _join_wire_frame(header_bytes, payload)
    if chunk_bytes is None:
        sock.sendall(wire)
    else:
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ProtocolError("chunk_bytes must be a positive integer")
        for offset in range(0, len(wire), chunk_bytes):
            sock.sendall(wire[offset:offset + chunk_bytes])
    return len(wire)


def _join_wire_frame(header_bytes, payload):
    prefix = FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload))
    return prefix + header_bytes + payload


def build_wire_bytes(header, arrays, compression="none", limits=ProtocolLimits()):
    """Build one complete frame for an in-memory message-bus transport."""
    header_bytes, payload = prepare_envelope(header, arrays, compression, limits)
    return _join_wire_frame(header_bytes, payload)


def send_envelope(
    sock,
    header,
    arrays,
    compression="none",
    limits=ProtocolLimits(),
    chunk_bytes=None,
):
    header_bytes, payload = prepare_envelope(header, arrays, compression, limits)
    return send_wire_frame(sock, header_bytes, payload, chunk_bytes=chunk_bytes)


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ProtocolError("connection closed before frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_wire_frame(sock, limits=ProtocolLimits()):
    _validate_limits(limits)
    prefix = _recv_exact(sock, FRAME_PREFIX.size)
    magic, header_size, payload_size = FRAME_PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("invalid frame magic")
    if header_size > limits.max_header_bytes:
        raise ProtocolError("header exceeds limit")
    if payload_size > limits.max_payload_bytes:
        raise ProtocolError("payload exceeds limit")
    header_bytes = _recv_exact(sock, header_size)
    payload = _recv_exact(sock, payload_size)
    return header_bytes, payload, FRAME_PREFIX.size + header_size + payload_size


def decode_wire_frame(
    header_bytes,
    payload,
    limits=ProtocolLimits(),
    return_stages=False,
    readonly=False,
):
    frame_parse_start = _clock_ns()
    header = inspect_envelope_header(header_bytes, payload, limits)
    frame_parse_end = _clock_ns()
    if return_stages:
        arrays, stages = decode_tensor_bundle(
            header["tensor_bundle"], payload, limits, return_stages=True, readonly=readonly,
        )
        stages["frame_parse_ns"] = max(0, frame_parse_end - frame_parse_start)
        stages["timestamps_ns"].update(
            {
                "frame_parse_start": frame_parse_start,
                "frame_parse_end": frame_parse_end,
            }
        )
        return header, arrays, stages
    arrays = decode_tensor_bundle(header["tensor_bundle"], payload, limits, readonly=readonly)
    return header, arrays


def inspect_envelope_header(header_bytes, payload, limits=ProtocolLimits()):
    """Validate envelope metadata without decompressing its tensor payload."""
    _validate_limits(limits)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid JSON header: %s" % error)
    _validate_header_identity(header)
    if "tensor_bundle" not in header:
        raise ProtocolError("tensor_bundle is required")
    bundle = _require_tensor_bundle_metadata(header["tensor_bundle"], payload, limits)
    descriptors = bundle.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) > limits.max_tensor_count:
        raise ProtocolError("invalid tensor descriptor list")
    seen_names = set()
    ranges = []
    for descriptor in descriptors:
        start, end = _validate_descriptor(
            descriptor, seen_names, bundle["raw_payload_bytes"], limits
        )
        for other_start, other_end in ranges:
            if start < other_end and other_start < end:
                raise ProtocolError("tensor payload ranges overlap")
        ranges.append((start, end))
    if sum(end - start for start, end in ranges) != bundle["raw_payload_bytes"]:
        raise ProtocolError("tensor descriptors do not cover payload exactly")
    return header


def _require_tensor_bundle_metadata(bundle, payload, limits):
    if not isinstance(bundle, dict):
        raise ProtocolError("tensor_bundle must be an object")
    _checksum_scope(bundle)
    compression = bundle.get("compression")
    if compression not in ALLOWED_COMPRESSIONS:
        raise ProtocolError("unsupported tensor compression")
    raw_size = bundle.get("raw_payload_bytes")
    wire_size = bundle.get("wire_payload_bytes")
    for name, value, maximum in (
        ("raw payload", raw_size, limits.max_raw_payload_bytes),
        ("wire payload", wire_size, limits.max_payload_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
            raise ProtocolError("%s exceeds limit" % name)
    if wire_size != len(payload):
        raise ProtocolError("wire payload length mismatch")
    digest = bundle.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ProtocolError("tensor payload sha256 is invalid")
    return bundle


def decode_wire_bytes(wire, limits=ProtocolLimits(), return_stages=False, readonly=False):
    """Decode an in-memory frame received through ROS 2 or another message bus."""
    _validate_limits(limits)
    if not isinstance(wire, (bytes, bytearray)):
        raise ProtocolError("wire frame must be bytes")
    if len(wire) < FRAME_PREFIX.size:
        raise ProtocolError("wire frame is shorter than prefix")
    magic, header_size, payload_size = FRAME_PREFIX.unpack(
        bytes(wire[:FRAME_PREFIX.size])
    )
    if magic != MAGIC:
        raise ProtocolError("invalid frame magic")
    if header_size > limits.max_header_bytes:
        raise ProtocolError("header exceeds limit")
    if payload_size > limits.max_payload_bytes:
        raise ProtocolError("payload exceeds limit")
    expected_size = FRAME_PREFIX.size + header_size + payload_size
    if len(wire) != expected_size:
        raise ProtocolError("wire frame length mismatch")
    header_start = FRAME_PREFIX.size
    payload_start = header_start + header_size
    decoded = decode_wire_frame(
        bytes(wire[header_start:payload_start]),
        bytes(wire[payload_start:]),
        limits,
        return_stages=return_stages,
        readonly=readonly,
    )
    if return_stages:
        header, arrays, stages = decoded
        return header, arrays, expected_size, stages
    header, arrays = decoded
    return header, arrays, expected_size


def inspect_wire_bytes_header(wire, limits=ProtocolLimits()):
    """Inspect an in-memory frame identity without decoding tensor data."""
    _validate_limits(limits)
    if not isinstance(wire, (bytes, bytearray)) or len(wire) < FRAME_PREFIX.size:
        raise ProtocolError("wire frame is shorter than prefix")
    magic, header_size, payload_size = FRAME_PREFIX.unpack(bytes(wire[:FRAME_PREFIX.size]))
    if magic != MAGIC:
        raise ProtocolError("invalid frame magic")
    if header_size > limits.max_header_bytes or payload_size > limits.max_payload_bytes:
        raise ProtocolError("wire frame exceeds limit")
    expected_size = FRAME_PREFIX.size + header_size + payload_size
    if len(wire) != expected_size:
        raise ProtocolError("wire frame length mismatch")
    header_start = FRAME_PREFIX.size
    payload_start = header_start + header_size
    header = inspect_envelope_header(
        bytes(wire[header_start:payload_start]), bytes(wire[payload_start:]), limits
    )
    return header, expected_size


def recv_envelope(sock, limits=ProtocolLimits()):
    header_bytes, payload, wire_bytes = recv_wire_frame(sock, limits)
    header, arrays = decode_wire_frame(header_bytes, payload, limits)
    return header, arrays, wire_bytes
