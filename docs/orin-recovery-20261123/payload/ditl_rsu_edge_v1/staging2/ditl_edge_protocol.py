from __future__ import print_function

import copy
import hashlib
import json
import re
import socket
import struct
import zlib
from dataclasses import dataclass

import numpy as np


PROTOCOL_VERSION = "ditl.edge.v1"
MAGIC = b"DITL1"
FRAME_PREFIX = struct.Struct("!5sIQ")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
ARRAY_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
ALLOWED_KINDS = frozenset(("request", "response", "error"))
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
        raise ProtocolError("kind must be request, response or error")
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


def encode_tensor_bundle(arrays, compression="none", limits=ProtocolLimits()):
    """Encode named ndarrays without pickle or executable metadata."""
    _validate_limits(limits)
    if not isinstance(arrays, dict):
        raise ProtocolError("arrays must be an object")
    if len(arrays) > limits.max_tensor_count:
        raise ProtocolError("tensor count exceeds limit")
    if compression not in ("none", "zlib"):
        raise ProtocolError("compression must be none or zlib")

    parts = []
    descriptors = []
    offset = 0
    for name in sorted(arrays):
        array = _canonical_array(arrays[name], name, limits)
        raw = array.tobytes(order="C")
        descriptors.append(
            {
                "name": name,
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        parts.append(raw)
        offset += len(raw)
        if offset > limits.max_raw_payload_bytes:
            raise ProtocolError("raw payload exceeds limit")
    raw_payload = b"".join(parts)
    payload = zlib.compress(raw_payload) if compression == "zlib" else raw_payload
    if len(payload) > limits.max_payload_bytes:
        raise ProtocolError("payload exceeds limit")
    metadata = {
        "compression": compression,
        "raw_payload_bytes": len(raw_payload),
        "wire_payload_bytes": len(payload),
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "arrays": descriptors,
    }
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


def decode_tensor_bundle(metadata, payload, limits=ProtocolLimits()):
    _validate_limits(limits)
    if not isinstance(metadata, dict):
        raise ProtocolError("tensor_bundle must be an object")
    if not isinstance(payload, (bytes, bytearray)):
        raise ProtocolError("payload must be bytes")
    if len(payload) > limits.max_payload_bytes:
        raise ProtocolError("payload exceeds limit")
    compression = metadata.get("compression")
    if compression not in ("none", "zlib"):
        raise ProtocolError("unsupported tensor compression")
    raw = (
        _decompress_zlib_bounded(payload, limits.max_raw_payload_bytes)
        if compression == "zlib"
        else bytes(payload)
    )
    if len(raw) > limits.max_raw_payload_bytes:
        raise ProtocolError("raw payload exceeds limit")
    if metadata.get("raw_payload_bytes") != len(raw):
        raise ProtocolError("raw payload length mismatch")
    if metadata.get("wire_payload_bytes") != len(payload):
        raise ProtocolError("wire payload length mismatch")
    expected_hash = metadata.get("sha256")
    actual_hash = hashlib.sha256(raw).hexdigest()
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ProtocolError("tensor payload sha256 mismatch")
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) > limits.max_tensor_count:
        raise ProtocolError("invalid tensor descriptor list")
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
        array = np.frombuffer(raw[start:end], dtype=dtype).reshape(descriptor["shape"])
        output[descriptor["name"]] = array.copy()
    covered = sum(end - start for start, end in ranges)
    if covered != len(raw):
        raise ProtocolError("tensor descriptors do not cover payload exactly")
    return output


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


def decode_wire_frame(header_bytes, payload, limits=ProtocolLimits()):
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid JSON header: %s" % error)
    _validate_header_identity(header)
    if "tensor_bundle" not in header:
        raise ProtocolError("tensor_bundle is required")
    arrays = decode_tensor_bundle(header["tensor_bundle"], payload, limits)
    return header, arrays


def decode_wire_bytes(wire, limits=ProtocolLimits()):
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
    header, arrays = decode_wire_frame(
        bytes(wire[header_start:payload_start]), bytes(wire[payload_start:]), limits
    )
    return header, arrays, expected_size


def recv_envelope(sock, limits=ProtocolLimits()):
    header_bytes, payload, wire_bytes = recv_wire_frame(sock, limits)
    header, arrays = decode_wire_frame(header_bytes, payload, limits)
    return header, arrays, wire_bytes
