#!/usr/bin/env python3
"""Throwaway Orin worker timing launcher (Python 3.8; standard library only).

Deployment: copy this file to the Orin diagnostic directory. Keep PYTHONPATH
pointing at the existing fresh runtime, then run:

  HIL_TIMING_LOG=/absolute/diagnostic/worker_timing.jsonl \
    python3 /absolute/diagnostic/worker_probe.py [original worker arguments]

All arguments go unchanged to scripts.coptv2x_hil_orin_worker.main(). No new CLI
flags, model replacement, CUDA calls, warmup, tensor copies, or protocol edits.
JSONL is buffered during each trigger and appended after the original call ends.
probe.flush reports serialization plus append cost (its own final write excluded).
send_wire_frame is ONLY the local publisher socket handoff, not remote delivery.
Only compare RAW timestamps within the same host/boot; wall time is annotation.
"""
from __future__ import print_function

from collections.abc import Mapping
import functools
import importlib
import json
import os
import sys
import threading
import time


def monotonic_raw_ns():
    try:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    except (AttributeError, OSError):
        return time.monotonic_ns()


def _snapshot(value):
    """Copy only JSON metadata; never touch tensor data or mutable inputs."""
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _identity(value):
    names = ("run_id", "route_id", "scenario_id", "arm_id", "sample_id",
             "sequence_no", "source_frame")
    metadata = {name: getattr(value, name, None) for name in names}
    sequence = metadata["sequence_no"]
    if sequence is None:
        sequence = getattr(value, "sequence_id", None)
    request = None
    if metadata["run_id"] is not None and sequence is not None:
        request = "%s:%s" % (metadata["run_id"], sequence)
    return {"request_id": request, "sequence_no": sequence}, metadata


def _result_metadata(result):
    header = getattr(result, "header", {})
    bundle = getattr(result, "tensor_bundle", {})
    return {
        "edge_timestamps_ns": header.get("edge_timestamps_ns", {}),
        "durations_ns": header.get("durations_ns", {}),
        "result_timestamps_ns": getattr(result, "timestamps_ns", {}),
        "result_durations_ns": getattr(result, "durations_ns", {}),
        "raw_payload_bytes": bundle.get("raw_payload_bytes"),
        "wire_payload_bytes": len(result.payload),
        "output_tensor_bytes": sum(int(value.nbytes) for value in result.outputs.values()),
        "payload_sha256": header.get("payload_sha256"),
    }


class _Recorder(object):
    def __init__(self, path):
        self.path = path
        self.local = threading.local()
        self.warned = False

    def warn(self, error):
        # Diagnostics must never replace a successful return or original exception.
        if not self.warned:
            self.warned = True
            try:
                sys.stderr.write("worker_probe logging failed: %s\n" % type(error).__name__)
            except Exception:
                pass

    def event(self, event, stamp, duration, metadata):
        try:
            context = self.local.context
            row = dict(context["identity"])
            row.update({"event": event, "monotonic_raw_ns": stamp,
                        "wall_time_ns": time.time_ns(), "duration_ns": duration,
                        "metadata": _snapshot(metadata)})
            context["rows"].append(row)
        except Exception as error:
            self.warn(error)

    def flush(self, context):
        start = monotonic_raw_ns()
        try:
            content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                              for row in context["rows"])
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(content)
            end = monotonic_raw_ns()
            row = dict(context["identity"])
            row.update({"event": "probe.flush", "monotonic_raw_ns": end,
                        "wall_time_ns": time.time_ns(), "duration_ns": end - start,
                        "metadata": {"events": len(context["rows"]),
                                     "final_probe_flush_record_write_excluded": True}})
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as error:
            self.warn(error)

    def wrap(self, original, stage, before=None, after=None):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            outer = getattr(self.local, "context", None) is None
            if outer:
                value = args[1] if len(args) > 1 else None
                identity, _ = _identity(value)
                self.local.context = {"identity": identity, "rows": []}
            metadata = {}
            try:
                if before is not None:
                    metadata = before(args, kwargs)
            except Exception as error:
                self.warn(error)
            start = monotonic_raw_ns()
            self.event(stage + ".start", start, 0, metadata)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                end = monotonic_raw_ns()
                details = dict(metadata)
                details.update({"error_type": type(error).__name__,
                                "error_message": str(error)[:500]})
                self.event(stage + ".error", end, end - start, details)
                raise
            else:
                end = monotonic_raw_ns()
                try:
                    if after is not None:
                        metadata = dict(metadata, **after(result))
                except Exception as error:
                    self.warn(error)
                self.event(stage + ".end", end, end - start, metadata)
                return result
            finally:
                if outer:
                    context = self.local.context
                    self.local.context = None
                    self.flush(context)
        return measured


def _trigger_metadata(args, kwargs):
    worker, trigger = args[:2]
    _, identity = _identity(trigger)
    key = (identity["run_id"], identity["route_id"], identity["scenario_id"], identity["arm_id"])
    return {"trigger_identity": identity,
            "arm_change": getattr(worker, "active_key", None) != key,
            "processed_before": getattr(worker, "processed", None)}


def _source_metadata(frame):
    metadata = getattr(frame, "source_metadata", {})
    return {"source_timestamps_ns": metadata.get("timestamps_ns", {}),
            "source_durations_ns": metadata.get("durations_ns", {}),
            "input_tensor_bytes": sum(int(value.nbytes) for value in
                                      getattr(frame, "arrays", {}).values())}


def install(module):
    """Instrument the imported worker, once; no environment log means no changes."""
    path = os.environ.get("HIL_TIMING_LOG")
    if not path or getattr(module, "_worker_probe_installed", False):
        return module
    recorder = _Recorder(path)
    worker = module.OrinHilWorker
    worker.process_trigger = recorder.wrap(worker.process_trigger, "process_trigger",
                                           before=_trigger_metadata)
    backend = module.CoptV2xHilBackend
    backend.load = recorder.wrap(backend.load, "backend.load", before=lambda a, k: {
        "runtime_already_loaded": getattr(a[0], "_runtime", None) is not None})
    source = module.IndexedLocalRsuSource
    source.frame_for_trigger = recorder.wrap(source.frame_for_trigger,
                                             "source.frame_for_trigger", after=_source_metadata)
    pipeline = module.CoptV2xHilPipeline
    pipeline.process = recorder.wrap(pipeline.process, "pipeline.process", after=_result_metadata)
    module._prepare_result_frame = recorder.wrap(
        module._prepare_result_frame, "_prepare_result_frame",
        after=lambda result: {"header_bytes": len(result[0]), "wire_payload_bytes": len(result[1])})
    module.send_wire_frame = recorder.wrap(
        module.send_wire_frame, "send_wire_frame", before=lambda a, k: {
            "header_bytes": len(a[1]), "wire_payload_bytes": len(a[2]),
            "scope": "local_result_publisher_handoff_only"})
    module._worker_probe_installed = True
    return module


def main():
    worker = importlib.import_module("scripts.coptv2x_hil_orin_worker")
    install(worker)
    return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
