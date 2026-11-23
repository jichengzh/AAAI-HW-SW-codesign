"""Opt-in capture of exact CoDriving backbone inputs for HIL integration smoke.

Captured inputs are integration fixtures only. They are not live-inference data
and must not be used to claim closed-loop driving performance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np


ROUTE_CAPTURE_SCHEMA_VERSION = "coptv2x.hil-route-input-capture.v1"
ROUTE_CAPTURE_MANIFEST_NAME = "route_input_capture.json"
EXPECTED_SPATIAL_FEATURES_SHAPE = (2, 64, 256, 512)
EXPECTED_SPATIAL_FEATURES_DTYPE = "float32"
_SKIPPED_SINGLE_AGENT_FRAME = object()


def _identity(value, label):
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or any(ord(character) < 32 for character in value)
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("%s must be a bounded non-path identity string" % label)
    return value


def _tensor_array(value):
    """Convert NumPy or torch-like tensors without importing torch here."""
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise ValueError("spatial_features must be a NumPy or tensor value")
    value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if not callable(to_numpy):
        raise ValueError("spatial_features tensor cannot be converted to NumPy")
    try:
        array = to_numpy()
    except (TypeError, RuntimeError) as error:
        raise ValueError("spatial_features tensor cannot be converted to NumPy") from error
    if not isinstance(array, np.ndarray):
        raise ValueError("spatial_features conversion did not return an array")
    return array


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create_bytes(target, payload):
    """Create target atomically without replacing any pre-existing file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(target))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_save_npy(target, array):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(target))
        return _sha256_file(target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class RouteInputCapture:
    """Save one exact `[ego, rsu]` backbone tensor per CARLA source frame.

    Construct this only for an explicitly requested input-capture/prepass run.
    Call :meth:`capture_frame` at the backbone pre-hook with the live
    ``batch_dict``; call :meth:`finalize` after the route has completed.
    """

    def __init__(self, output_dir, route_id, scenario_id):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.route_id = _identity(route_id, "route_id")
        self.scenario_id = _identity(scenario_id, "scenario_id")
        self._frames = []
        self._source_frames = set()
        self._finalized = False

    @property
    def frames(self):
        return tuple(dict(frame) for frame in self._frames)

    def capture_frame(self, batch_dict, source_frame, sample_id=None):
        if self._finalized:
            raise RuntimeError("route input capture is finalized")
        if type(source_frame) is not int or source_frame < 0:
            raise ValueError("source_frame must be a non-negative integer")
        expected_sample_id = "carla-frame-%d" % source_frame
        if sample_id is None:
            sample_id = expected_sample_id
        if sample_id != expected_sample_id:
            raise ValueError("sample_id must match source_frame identity")
        if source_frame in self._source_frames:
            raise ValueError("duplicate source_frame in route input capture")
        if type(batch_dict) is not dict or "spatial_features" not in batch_dict:
            raise ValueError("batch_dict must contain spatial_features")

        tensor = batch_dict["spatial_features"]
        shape = tuple(getattr(tensor, "shape", ()))
        dtype_name = str(getattr(tensor, "dtype", "")).replace("torch.", "")
        if shape != EXPECTED_SPATIAL_FEATURES_SHAPE:
            raise ValueError(
                "spatial_features shape must be %s, got %s" %
                (EXPECTED_SPATIAL_FEATURES_SHAPE, shape)
            )
        if dtype_name != EXPECTED_SPATIAL_FEATURES_DTYPE:
            raise ValueError("spatial_features dtype must be float32, got %s" % dtype_name)
        array = _tensor_array(tensor)
        if tuple(array.shape) != EXPECTED_SPATIAL_FEATURES_SHAPE:
            raise ValueError("converted spatial_features shape changed")
        if str(array.dtype) != EXPECTED_SPATIAL_FEATURES_DTYPE:
            raise ValueError("converted spatial_features dtype changed")

        feature_path = self.output_dir / ("spatial_features_%08d.npy" % source_frame)
        if feature_path.exists():
            raise FileExistsError("route input frame file already exists: %s" % feature_path)
        file_hash = _atomic_save_npy(feature_path, array)
        record = {
            "source_frame": source_frame,
            "sample_id": expected_sample_id,
            "spatial_features_path": str(feature_path),
            "spatial_features_sha256": file_hash,
            "shape": list(EXPECTED_SPATIAL_FEATURES_SHAPE),
            "dtype": EXPECTED_SPATIAL_FEATURES_DTYPE,
        }
        self._frames.append(record)
        self._source_frames.add(source_frame)
        return dict(record)

    def finalize(self):
        if self._finalized:
            raise RuntimeError("route input capture is finalized")
        if not self._frames:
            raise ValueError("route input capture requires at least one frame")
        manifest_path = self.output_dir / ROUTE_CAPTURE_MANIFEST_NAME
        if manifest_path.exists():
            raise FileExistsError("route input capture manifest already exists")
        manifest = {
            "schema_version": ROUTE_CAPTURE_SCHEMA_VERSION,
            "purpose": "integration-input-only",
            "route_id": self.route_id,
            "scenario_id": self.scenario_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": len(self._frames),
            "frames": [dict(frame) for frame in self._frames],
        }
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _atomic_create_bytes(manifest_path, payload)
        self._finalized = True
        return manifest_path


class RouteInputCaptureHook:
    """Bind one active CARLA inference frame to a model backbone pre-hook."""

    def __init__(self, capture, perception_model):
        self.capture = capture
        model = getattr(perception_model, "module", perception_model)
        backbone = getattr(model, "backbone", None)
        register_hook = getattr(backbone, "register_forward_pre_hook", None)
        if not callable(register_hook):
            raise ValueError("perception model must expose a hookable backbone")
        self._source_frame = None
        self._record = None
        self._handle = register_hook(self._before_backbone)

    def begin_frame(self, source_frame):
        if self._source_frame is not None:
            raise RuntimeError("a route capture frame is already active")
        if type(source_frame) is not int or source_frame < 0:
            raise ValueError("source_frame must be a non-negative integer")
        self._source_frame = source_frame
        self._record = None

    def _before_backbone(self, _module, inputs):
        if self._source_frame is None:
            return
        if self._record is not None:
            return
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise ValueError("backbone pre-hook did not receive a batch dictionary")
        batch_dict = inputs[0]
        if type(batch_dict) is not dict or "spatial_features" not in batch_dict:
            raise ValueError("backbone input must contain spatial_features")
        shape = tuple(getattr(batch_dict["spatial_features"], "shape", ()))
        if len(shape) == 4 and shape[0] == 1:
            self._record = _SKIPPED_SINGLE_AGENT_FRAME
            return
        self._record = self.capture.capture_frame(batch_dict, self._source_frame)

    def finish_frame(self):
        if self._source_frame is None:
            raise RuntimeError("no route capture frame is active")
        source_frame = self._source_frame
        record = self._record
        self._source_frame = None
        self._record = None
        if record is _SKIPPED_SINGLE_AGENT_FRAME:
            return None
        if record is None:
            raise RuntimeError(
                "no backbone input was captured for route frame %d" % source_frame
            )
        return dict(record)

    def abort_frame(self):
        self._source_frame = None
        self._record = None

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
