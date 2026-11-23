from __future__ import print_function

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ditl_edge_runtime import monotonic_raw_ns


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
