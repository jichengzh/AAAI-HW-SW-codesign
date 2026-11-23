#!/usr/bin/env python3
"""Verify route capture tensors and emit host/Orin pilot input indexes."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


HOST_INDEX_SCHEMA = "coptv2x.route-input-host-index.v1"
LOCAL_INDEX_SCHEMA = "coptv2x.local-input-manifest.v1"
PILOT_CELL = "Town05:_1:r2"
EXPECTED_SHAPE = [2, 64, 256, 512]
EXPECTED_DTYPE = "float32"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create_json(path, payload):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(target))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _verify_capture(capture_manifest_path, data_root, cohort_manifest):
    data_root = Path(data_root).expanduser().resolve()
    capture_path = Path(capture_manifest_path).expanduser().resolve()
    try:
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("capture manifest is unreadable: %s" % error) from error
    if capture.get("schema_version") != "coptv2x.hil-route-input-capture.v1":
        raise ValueError("unsupported route capture manifest schema")
    if capture.get("purpose") != "integration-input-only":
        raise ValueError("route capture must be marked integration-input-only")
    if cohort_manifest.get("schema") != "coptv2x.hil-cohort.v1":
        raise ValueError("cohort manifest schema is not the frozen CoptV2X HIL schema")
    if cohort_manifest.get("route_scenarios") != [PILOT_CELL]:
        raise ValueError("cohort manifest must select only %s" % PILOT_CELL)
    records = cohort_manifest.get("route_scenario_records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("pilot cohort must contain exactly one route record")
    record = records[0]
    if record.get("cell_id") != PILOT_CELL:
        raise ValueError("pilot route record is not %s" % PILOT_CELL)
    expected_route_id = "Town05-route-%d" % record.get("route")
    expected_scenario_id = record.get("scenario_id")
    if (
        capture.get("route_id") != expected_route_id
        or capture.get("scenario_id") != expected_scenario_id
    ):
        raise ValueError("capture route/scenario identity does not match pilot cohort")
    frames = capture.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("route capture contains no frames")
    if capture.get("frame_count") != len(frames):
        raise ValueError("capture frame_count does not match its frame list")

    host_frames = []
    local_samples = []
    seen = set()
    for frame in frames:
        required = (
            "source_frame", "sample_id", "spatial_features_path",
            "spatial_features_sha256", "shape", "dtype",
        )
        if not isinstance(frame, dict) or not all(key in frame for key in required):
            raise ValueError("route capture frame mapping is incomplete")
        source_frame = frame["source_frame"]
        sample_id = frame["sample_id"]
        if type(source_frame) is not int or source_frame < 0:
            raise ValueError("route capture source_frame is invalid")
        if sample_id != "carla-frame-%d" % source_frame:
            raise ValueError("route capture sample identity does not match source_frame")
        if source_frame in seen:
            raise ValueError("route capture repeats a source_frame")
        seen.add(source_frame)
        if frame["shape"] != EXPECTED_SHAPE or frame["dtype"] != EXPECTED_DTYPE:
            raise ValueError("route capture shape/dtype violates the frozen contract")
        feature = Path(frame["spatial_features_path"]).expanduser().resolve()
        try:
            feature.relative_to(data_root)
        except ValueError as error:
            raise ValueError("route capture feature escapes its data root") from error
        if not feature.is_file() or _sha256_file(feature) != frame["spatial_features_sha256"]:
            raise ValueError("route capture feature file is missing or has wrong SHA-256")
        try:
            array = np.load(str(feature), allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ValueError("route capture feature is not a safe NPY") from error
        if array.dtype.hasobject or list(array.shape) != EXPECTED_SHAPE or str(array.dtype) != EXPECTED_DTYPE:
            raise ValueError("route capture NPY shape/dtype does not match its index")
        if not np.isfinite(array).all():
            raise ValueError("route capture NPY contains non-finite spatial features")
        relative = feature.relative_to(data_root).as_posix()
        host_frames.append({
            "source_frame": source_frame,
            "sample_id": sample_id,
            "spatial_features_path": str(feature),
            "spatial_features_sha256": frame["spatial_features_sha256"],
            "shape": EXPECTED_SHAPE,
            "dtype": EXPECTED_DTYPE,
        })
        local_samples.append({
            "sample_id": sample_id,
            "source_frame": source_frame,
            "inputs": {
                "spatial_features": {
                    "path": relative,
                    "sha256": frame["spatial_features_sha256"],
                    "shape": EXPECTED_SHAPE,
                    "dtype": EXPECTED_DTYPE,
                }
            },
        })
    host_frames.sort(key=lambda item: item["source_frame"])
    local_samples.sort(key=lambda item: item["source_frame"])
    return capture, host_frames, local_samples


def build_route_indexes(capture_manifest_path, cohort_manifest, data_root, host_index_path):
    """Create runner-visible absolute index and Orin-local relative manifest."""
    capture, host_frames, local_samples = _verify_capture(
        capture_manifest_path, data_root, cohort_manifest
    )
    host_index = {
        "schema": HOST_INDEX_SCHEMA,
        "version": HOST_INDEX_SCHEMA,
        "status": "HOST_INDEX_VALID",
        "purpose": "integration-input-only",
        "pilot_cell": PILOT_CELL,
        "route_scenarios_sha256": cohort_manifest["route_scenarios_sha256"],
        "route_scenarios": {PILOT_CELL: host_frames},
        "capture_manifest_sha256": _sha256_file(capture_manifest_path),
    }
    local_index = {
        "schema_version": LOCAL_INDEX_SCHEMA,
        "route_id": capture["route_id"],
        "scenario_id": capture["scenario_id"],
        "batch_order": ["ego", "rsu"],
        "samples": local_samples,
    }
    root = Path(data_root).expanduser().resolve()
    local_path = _atomic_create_json(root / "local_input_manifest.json", local_index)
    host_path = _atomic_create_json(host_index_path, host_index)
    return local_path, host_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--host-index-output", required=True, type=Path)
    args = parser.parse_args(argv)
    cohort = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    local_path, host_path = build_route_indexes(
        args.capture_manifest, cohort, args.data_root, args.host_index_output
    )
    print(json.dumps({
        "status": "HOST_INDEX_VALID",
        "pilot_cell": PILOT_CELL,
        "frames": len(json.loads(args.capture_manifest.read_text(encoding="utf-8"))["frames"]),
        "host_index": str(host_path),
        "local_manifest": str(local_path),
        "host_index_sha256": _sha256_file(host_path),
        "capture_manifest_sha256": _sha256_file(args.capture_manifest),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
