from __future__ import print_function

import hashlib
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ditl_edge_runtime import monotonic_raw_ns


LOCAL_INPUT_MANIFEST_SCHEMA_VERSION = "coptv2x.local-input-manifest.v1"
_LOCAL_MANIFEST_FIELDS = {
    "schema_version", "route_id", "scenario_id", "batch_order", "samples",
}
_LOCAL_SAMPLE_FIELDS = {"sample_id", "source_frame", "inputs"}
_LOCAL_INPUT_FIELDS = {"path", "sha256", "shape", "dtype"}
_LOCAL_INPUT_OPTIONAL_FIELDS = {"member"}


@dataclass(frozen=True)
class LocalRsuFrame:
    frame_id: int
    sequence_id: int
    data_available_ns: int
    arrays: Mapping[str, np.ndarray]
    source_metadata: Mapping[str, object]


def _resolve_under_root(data_root, relative_path):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("manifest path must be a non-empty string")
    candidate = (data_root / relative_path).resolve()
    try:
        inside_root = os.path.commonpath((str(data_root), str(candidate))) == str(data_root)
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError("manifest path escapes data root")
    if not candidate.is_file():
        raise ValueError("manifest input does not exist under data root: %s" % relative_path)
    return candidate


def _load_manifest_array(path, member=None):
    try:
        if path.suffix == ".npy":
            if member is not None:
                raise ValueError("NPY manifest entries cannot select a member")
            value = np.load(str(path), allow_pickle=False, mmap_mode="r")
        elif path.suffix == ".npz":
            if not isinstance(member, str) or not member:
                raise ValueError("NPZ manifest entries require a member")
            with np.load(str(path), allow_pickle=False) as archive:
                if member not in archive.files:
                    raise ValueError("NPZ member %s is missing" % member)
                value = np.array(archive[member], copy=True, order="C")
        else:
            raise ValueError("local RSU inputs must be NPY or NPZ")
    except (OSError, ValueError) as error:
        raise ValueError("local input rejected because pickle/object data is unsafe: %s" % error)
    if np.asarray(value).dtype.hasobject:
        raise ValueError("object arrays requiring pickle are forbidden")
    return value


def _load_verified_indexed_npy(path, expected_hash, maximum_file_bytes):
    if path.suffix != ".npy":
        raise ValueError("indexed HIL inputs must be NPY files")
    descriptor = None
    try:
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("indexed HIL input must be a safe regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(maximum_file_bytes + 1)
    except OSError as error:
        raise ValueError("indexed HIL input must be a safe regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > maximum_file_bytes:
        raise ValueError("indexed HIL input exceeds its tensor byte bound")
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("indexed HIL input hash mismatch")
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("indexed HIL input is not a safe NPY array") from error
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError("indexed HIL input must be a non-object NPY array")
    return value


def _load_verified_indexed_npy_mmap(path, expected_hash, maximum_file_bytes):
    """Verify a regular NPY file and retain a read-only file-backed view."""

    descriptor = None
    try:
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("indexed HIL input must be a safe regular file")
        if file_stat.st_size > maximum_file_bytes:
            raise ValueError("indexed HIL input exceeds its tensor byte bound")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError("indexed HIL input hash mismatch")
        mapped_path = "/proc/self/fd/%d" % descriptor
        array = np.load(mapped_path, allow_pickle=False, mmap_mode="r")
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValueError("indexed HIL input must be a safe NPY array")
        if array.nbytes > maximum_file_bytes:
            raise ValueError("indexed HIL tensor exceeds its byte bound")
        if not np.isfinite(array).all():
            raise ValueError("indexed HIL tensor contains non-finite values")
        return array
    except OSError as error:
        raise ValueError("indexed HIL input must be a safe regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_local_identity(value, label):
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or any(character in value for character in ("/", "\\", "\x00"))
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("%s must be a bounded non-path identity string" % label)
    return value


def _require_local_counter(value, label):
    if type(value) is not int or value < 0 or value > 2 ** 63 - 1:
        raise ValueError("%s must be a non-negative signed 64-bit integer" % label)
    return value


def _require_positive_limit(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return value


class IndexedLocalRsuSource(object):
    """Index a route manifest and resolve triggers by exact composite key."""

    def __init__(
        self,
        data_root,
        manifest,
        input_contract,
        max_samples,
        max_total_tensor_bytes,
        source_id="rsu-orin-01",
        memory_map=False,
        lazy_load=False,
    ):
        self.max_samples = _require_positive_limit(max_samples, "max_samples")
        self.max_total_tensor_bytes = _require_positive_limit(
            max_total_tensor_bytes, "max_total_tensor_bytes"
        )
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise ValueError("data root must be an existing directory")
        if type(manifest) is not dict or set(manifest) != _LOCAL_MANIFEST_FIELDS:
            raise ValueError("local manifest fields do not match the fixed contract")
        if manifest["schema_version"] != LOCAL_INPUT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("local manifest schema_version is unsupported")
        self.route_id = _require_local_identity(manifest["route_id"], "route_id")
        self.scenario_id = _require_local_identity(
            manifest["scenario_id"], "scenario_id"
        )
        self.source_id = _require_local_identity(source_id, "source_id")
        if type(memory_map) is not bool:
            raise ValueError("memory_map must be a boolean")
        if type(lazy_load) is not bool:
            raise ValueError("lazy_load must be a boolean")
        self.memory_map = memory_map
        self.lazy_load = lazy_load
        batch_order = manifest["batch_order"]
        if type(batch_order) is not list or batch_order != ["ego", "rsu"]:
            raise ValueError("local manifest batch order must be [ego, rsu]")
        self.batch_order = ("ego", "rsu")
        if type(input_contract) is not dict or not input_contract:
            raise ValueError("input contract must be a non-empty mapping")
        if any(
            type(specification) is not dict
            or set(specification) != {"shape", "dtype"}
            for specification in input_contract.values()
        ):
            raise ValueError("input contract tensors require shape and dtype")
        self.input_contract = dict(input_contract)
        samples = manifest["samples"]
        if type(samples) is not list or not samples:
            raise ValueError("local manifest samples must be a non-empty list")
        if len(samples) > self.max_samples:
            raise ValueError("local manifest exceeds max_samples")
        bytes_per_sample = 0
        for specification in self.input_contract.values():
            shape = specification["shape"]
            if (
                not isinstance(shape, (list, tuple))
                or not shape
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            ):
                raise ValueError("input contract shape is invalid")
            try:
                dtype = np.dtype(specification["dtype"])
            except (TypeError, ValueError) as error:
                raise ValueError("input contract dtype is invalid") from error
            tensor_bytes = dtype.itemsize
            for dimension in shape:
                tensor_bytes *= dimension
            bytes_per_sample += tensor_bytes
        projected_tensor_bytes = bytes_per_sample * len(samples)
        if projected_tensor_bytes > self.max_total_tensor_bytes:
            raise ValueError("local manifest exceeds max_total_tensor_bytes")
        self._samples = {}
        for sample_index, sample in enumerate(samples):
            if type(sample) is not dict or set(sample) != _LOCAL_SAMPLE_FIELDS:
                raise ValueError(
                    "sample %d fields do not match the contract" % sample_index
                )
            sample_id = _require_local_identity(
                sample["sample_id"], "samples[%d].sample_id" % sample_index
            )
            source_frame = _require_local_counter(
                sample["source_frame"], "samples[%d].source_frame" % sample_index
            )
            key = (self.route_id, self.scenario_id, sample_id, source_frame)
            if key in self._samples:
                raise ValueError("duplicate local sample composite key")
            inputs = sample["inputs"]
            if type(inputs) is not dict or set(inputs) != set(self.input_contract):
                raise ValueError("sample input names must exactly match input contract")
            arrays = {}
            for name in sorted(inputs):
                entry = inputs[name]
                allowed_fields = _LOCAL_INPUT_FIELDS | _LOCAL_INPUT_OPTIONAL_FIELDS
                if (
                    type(entry) is not dict
                    or not _LOCAL_INPUT_FIELDS.issubset(entry)
                    or not set(entry).issubset(allowed_fields)
                ):
                    raise ValueError("input %s fields do not match the contract" % name)
                path = _resolve_under_root(self.data_root, entry["path"])
                if "member" in entry:
                    raise ValueError("indexed HIL inputs must be NPY files")
                expected_hash = entry["sha256"]
                if (
                    type(expected_hash) is not str
                    or len(expected_hash) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in expected_hash
                    )
                ):
                    raise ValueError("input %s hash must be lowercase SHA-256" % name)
                declared_shape = entry["shape"]
                if (
                    type(declared_shape) is not list
                    or any(
                        type(dimension) is not int or dimension <= 0
                        for dimension in declared_shape
                    )
                ):
                    raise ValueError("input %s shape declaration is invalid" % name)
                contract_shape = tuple(self.input_contract[name]["shape"])
                if tuple(declared_shape) != contract_shape:
                    raise ValueError("input %s shape mismatch" % name)
                declared_dtype = entry["dtype"]
                try:
                    contract_dtype = np.dtype(self.input_contract[name]["dtype"])
                    manifest_dtype = np.dtype(declared_dtype)
                except (TypeError, ValueError) as error:
                    raise ValueError("input %s dtype mismatch" % name) from error
                if (
                    type(declared_dtype) is not str
                    or manifest_dtype != contract_dtype
                ):
                    raise ValueError("input %s dtype mismatch" % name)
                expected_bytes = int(np.prod(contract_shape)) * contract_dtype.itemsize
                loader = (
                    _load_verified_indexed_npy_mmap
                    if self.memory_map
                    else _load_verified_indexed_npy
                )
                if self.lazy_load:
                    arrays[name] = {
                        "path": path,
                        "sha256": expected_hash,
                        "maximum_file_bytes": expected_bytes + 1024 * 1024,
                        "shape": contract_shape,
                        "dtype": contract_dtype,
                    }
                else:
                    array = loader(
                        path, expected_hash,
                        maximum_file_bytes=expected_bytes + 1024 * 1024,
                    )
                    if tuple(array.shape) != contract_shape:
                        raise ValueError("input %s shape mismatch" % name)
                    if np.dtype(array.dtype) != contract_dtype:
                        raise ValueError("input %s dtype mismatch" % name)
                    if self.memory_map:
                        arrays[name] = array
                    else:
                        copied = np.array(array, copy=True, order="C")
                        copied.setflags(write=False)
                        arrays[name] = copied
            self._samples[key] = MappingProxyType(arrays)

    def frame_for_trigger(self, trigger):
        from coptv2x_hil_trigger import HilTrigger

        if type(trigger) is not HilTrigger:
            raise ValueError("trigger must be a validated HilTrigger")
        trigger_metadata = trigger.to_metadata()
        key = (
            trigger.route_id,
            trigger.scenario_id,
            trigger.sample_id,
            trigger.source_frame,
        )
        try:
            preloaded = self._samples[key]
        except KeyError as error:
            raise ValueError(
                "trigger route/scenario/sample composite key is not preloaded"
            ) from error
        data_available_ns = monotonic_raw_ns()
        source_read_start = monotonic_raw_ns()
        arrays = {}
        for name, value in preloaded.items():
            if self.lazy_load:
                loader = (
                    _load_verified_indexed_npy_mmap
                    if self.memory_map
                    else _load_verified_indexed_npy
                )
                value = loader(
                    value["path"],
                    value["sha256"],
                    maximum_file_bytes=value["maximum_file_bytes"],
                )
                if tuple(value.shape) != preloaded[name]["shape"]:
                    raise ValueError("input %s shape mismatch" % name)
                if np.dtype(value.dtype) != preloaded[name]["dtype"]:
                    raise ValueError("input %s dtype mismatch" % name)
            copied = np.array(value, copy=True, order="C")
            copied.setflags(write=False)
            arrays[name] = copied
            mapped_file = getattr(value, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()
        source_read_end = monotonic_raw_ns()
        metadata = {
            "source_type": "local_indexed_feature",
            "source_id": self.source_id,
            "trigger_identity": MappingProxyType(trigger_metadata),
            "batch_order": self.batch_order,
            "remote_input_tensor_bytes": 0,
            "timestamps_ns": {
                "source_read_start": source_read_start,
                "source_read_end": source_read_end,
            },
            "durations_ns": {
                "source_read": max(0, source_read_end - source_read_start),
            },
        }
        return LocalRsuFrame(
            frame_id=trigger.source_frame,
            sequence_id=trigger.sequence_no,
            data_available_ns=data_available_ns,
            arrays=MappingProxyType(arrays),
            source_metadata=MappingProxyType(metadata),
        )


class LocalRsuFrameSource(object):
    """Reads model inputs from an allowlisted Orin-local manifest."""

    def __init__(
        self,
        data_root,
        manifest,
        input_contract,
        source_id="rsu-orin-01",
        cycle=True,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise ValueError("data root must be an existing directory")
        if not isinstance(manifest, dict) or set(manifest) != set(input_contract):
            raise ValueError("manifest names must exactly match the input contract")
        self.input_contract = dict(input_contract)
        self.source_id = str(source_id)
        self.cycle = bool(cycle)
        self._sequence_id = 0
        self._arrays = {}
        self._frame_axis = {}
        self._fixed_index = {}
        frame_counts = set()
        fixed_modes = set()
        fixed_values = set()
        for name in sorted(manifest):
            entry = manifest[name]
            if isinstance(entry, str):
                entry = {"path": entry, "frame_axis": 0}
            if not isinstance(entry, dict):
                raise ValueError("manifest entry %s must be an object" % name)
            path = _resolve_under_root(self.data_root, entry.get("path"))
            frame_axis = entry.get("frame_axis", 0)
            if isinstance(frame_axis, bool) or not isinstance(frame_axis, int):
                raise ValueError("frame_axis must be an integer")
            array = _load_manifest_array(path, entry.get("member"))
            if frame_axis < 0 or frame_axis >= array.ndim:
                raise ValueError("frame_axis is outside tensor rank")
            frame_index = entry.get("frame_index")
            if frame_index is not None:
                if (
                    isinstance(frame_index, bool)
                    or not isinstance(frame_index, int)
                    or frame_index < 0
                    or frame_index >= array.shape[frame_axis]
                ):
                    raise ValueError("frame_index is outside the local tensor")
                fixed_values.add(frame_index)
            fixed_modes.add(frame_index is not None)
            frame_shape = tuple(
                dimension for index, dimension in enumerate(array.shape)
                if index != frame_axis
            )
            specification = input_contract[name]
            if frame_shape != tuple(specification["shape"]):
                raise ValueError(
                    "input %s frame shape %s does not match %s"
                    % (name, frame_shape, tuple(specification["shape"]))
                )
            if np.dtype(array.dtype) != np.dtype(specification["dtype"]):
                raise ValueError(
                    "input %s dtype %s does not match %s"
                    % (name, array.dtype.name, specification["dtype"])
                )
            frame_counts.add(1 if frame_index is not None else int(array.shape[frame_axis]))
            self._arrays[name] = array
            self._frame_axis[name] = frame_axis
            self._fixed_index[name] = frame_index
        if len(frame_counts) != 1:
            raise ValueError("all local input tensors must have the same frame count")
        if len(fixed_modes) != 1 or len(fixed_values) > 1:
            raise ValueError("all local input tensors must select the same frame_index")
        self.frame_count = frame_counts.pop()
        if self.frame_count <= 0:
            raise ValueError("local input tensors contain no frames")

    def next_frame(self):
        if not self.cycle and self._sequence_id >= self.frame_count:
            raise StopIteration("local RSU frame source is exhausted")
        sequence_id = self._sequence_id
        frame_id = sequence_id % self.frame_count
        data_available_ns = monotonic_raw_ns()
        source_read_start = monotonic_raw_ns()
        arrays = {}
        selected_indices = set()
        for name in sorted(self._arrays):
            selected_index = self._fixed_index[name]
            if selected_index is None:
                selected_index = frame_id
            selected_indices.add(selected_index)
            value = np.take(
                self._arrays[name], selected_index, axis=self._frame_axis[name]
            )
            copied = np.array(value, copy=True, order="C")
            copied.setflags(write=False)
            arrays[name] = copied
        source_read_end = monotonic_raw_ns()
        self._sequence_id += 1
        metadata = {
            "source_type": "local_feature_replay",
            "source_id": self.source_id,
            "frame_count": self.frame_count,
            "cycle_index": sequence_id // self.frame_count,
            "source_frame_index": (
                next(iter(selected_indices)) if len(selected_indices) == 1 else frame_id
            ),
            "timestamps_ns": {
                "source_read_start": source_read_start,
                "source_read_end": source_read_end,
            },
            "durations_ns": {
                "source_read": max(0, source_read_end - source_read_start),
            },
        }
        return LocalRsuFrame(
            frame_id=frame_id,
            sequence_id=sequence_id,
            data_available_ns=data_available_ns,
            arrays=MappingProxyType(arrays),
            source_metadata=MappingProxyType(metadata),
        )
