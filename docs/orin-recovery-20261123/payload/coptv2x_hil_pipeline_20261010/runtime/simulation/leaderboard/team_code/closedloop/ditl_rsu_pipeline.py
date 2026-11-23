from __future__ import print_function

import copy
import socket
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ditl_edge_protocol import PROTOCOL_VERSION, ProtocolLimits, encode_tensor_bundle
from ditl_edge_runtime import boot_id, collect_device_snapshot, monotonic_raw_ns
from ditl_trt_backend import validate_input_arrays
from coptv2x_hil_backend import _freeze_metadata, _plain_metadata, _frozen_numpy_snapshot


@dataclass(frozen=True)
class EncodedRsuResult:
    sequence_id: int
    header: Mapping[str, object]
    payload: bytes
    tensor_bundle: Mapping[str, object]
    timestamps_ns: Mapping[str, int]
    durations_ns: Mapping[str, int]
    outputs: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class InferredRsuFrame:
    """Controlled in-process handoff, not an encoded or validated wire result."""

    frame: object
    inference: Mapping[str, object]
    timestamps_ns: Mapping[str, int]
    durations_ns: Mapping[str, int]
    source_batch_order: tuple
    hil_raw: object
    hil_feature_result: object
    _producer: object


class RsuEdgePipeline(object):
    """Runs one local RSU frame without receiving model tensors from a client."""

    def __init__(
        self,
        backend,
        run_id,
        compression="none",
        limits=ProtocolLimits(),
        source_id="rsu-orin-01",
        target_id="ego-server-01",
        route_id="tensor-replay",
        scenario_id="edge-rsu",
        host_id=None,
        host_boot_id=None,
        payload_semantics="dense_head_debug",
        checksum_dedup=False,
    ):
        self.backend = backend
        self.run_id = str(run_id)
        self.compression = str(compression)
        self.limits = limits
        self.source_id = str(source_id)
        self.target_id = str(target_id)
        self.route_id = str(route_id)
        self.scenario_id = str(scenario_id)
        self.host_id = str(host_id or socket.gethostname())
        self.host_boot_id = str(host_boot_id or boot_id())
        self.payload_semantics = str(payload_semantics)
        self.checksum_dedup = bool(checksum_dedup)
        if self.checksum_dedup and (
            self.compression != "none"
            or self.payload_semantics != "codriving_rsu_multiscale_features.v1"
            or not getattr(backend, "checksum_dedup", False)
        ):
            raise ValueError("checksum_dedup requires an optimized RSU HIL backend and compression none")
        if hasattr(self.backend, "infer_hil_validated"):
            from coptv2x_hil_backend import (
                COPTV2X_PAYLOAD_SEMANTICS, COPTV2X_RSU_PAYLOAD_SEMANTICS,
            )
            if self.payload_semantics not in (
                COPTV2X_PAYLOAD_SEMANTICS, COPTV2X_RSU_PAYLOAD_SEMANTICS,
            ):
                raise ValueError(
                    "CoptV2X HIL backend requires the explicit CoptV2xHilPipeline"
                )
        self._last_sequence_id = -1
        # Separate writers: infer owns reservations, finish owns encoded sequence.
        self._last_inferred_sequence_id = -1
        self._inferred_producer = object()

    def process(self, frame):
        # Keep the established infer_hil_validated override/instrumentation hook
        # for serial callers; both modes use the same stage implementations.
        previous_sequence = self._last_inferred_sequence_id
        inferred = self._infer_frame(frame, split_hil=False)
        try:
            return self.encode_inferred_frame(inferred)
        except Exception:
            # Legacy serial failures did not consume the frame sequence. Only
            # this serial inference caller owns the reservation being restored.
            self._last_inferred_sequence_id = previous_sequence
            raise

    def infer_frame(self, frame):
        """Run on one inference thread; host snapshots are ready before return."""
        return self._infer_frame(frame, split_hil=True)

    def _infer_frame(self, frame, split_hil):
        if frame.sequence_id <= self._last_inferred_sequence_id:
            raise ValueError("frame sequence must increase strictly")
        # Source objects can reuse their metadata; capture provenance before handoff.
        frame = replace(frame, source_metadata=_freeze_metadata(frame.source_metadata))

        timestamps = {
            "rsu_data_available": int(frame.data_available_ns),
        }
        source_timestamps = frame.source_metadata.get("timestamps_ns", {})
        timestamps.update(
            (str(name), int(value)) for name, value in source_timestamps.items()
        )
        durations = {
            str(name): int(value)
            for name, value in frame.source_metadata.get("durations_ns", {}).items()
        }

        timestamps["preprocess_start"] = monotonic_raw_ns()
        if hasattr(self.backend, "validate_inputs"):
            inputs = self.backend.validate_inputs(frame.arrays)
        else:
            inputs = validate_input_arrays(self.backend.input_contract, dict(frame.arrays))
        timestamps["preprocess_end"] = monotonic_raw_ns()
        durations["preprocess"] = max(
            0, timestamps["preprocess_end"] - timestamps["preprocess_start"]
        )

        hil_feature_result = None
        hil_raw = None
        source_batch_order = ()
        inference = {}
        if hasattr(self.backend, "infer_hil_validated"):
            from coptv2x_hil_backend import (
                CODRIVING_BATCH_ORDER, CODRIVING_RSU_BATCH_ORDER,
                COPTV2X_RSU_PAYLOAD_SEMANTICS,
            )
            from coptv2x_hil_trigger import parse_hil_trigger

            trigger_identity = frame.source_metadata.get("trigger_identity")
            trigger = parse_hil_trigger(dict(trigger_identity or {}))
            source_batch_order = frame.source_metadata.get("batch_order")
            if (
                not isinstance(source_batch_order, (tuple, list))
                or tuple(source_batch_order) != CODRIVING_BATCH_ORDER
            ):
                raise ValueError(
                    "CoptV2X source batch order must be exact [ego, rsu]"
                )
            source_read_start = timestamps.get(
                "source_read_start", int(frame.data_available_ns)
            )
            source_read_end = timestamps.get("source_read_end", source_read_start)
            split_capability = getattr(self.backend, "infer_hil_raw_validated", None)
            infer_hil = (
                split_capability if split_hil and callable(split_capability)
                else self.backend.infer_hil_validated
            )
            hil_result = infer_hil(
                inputs,
                trigger,
                batch_order=tuple(source_batch_order),
                preprocess_start=timestamps["preprocess_start"],
                preprocess_end=timestamps["preprocess_end"],
                local_source_timestamps_ns={
                    "local_source_start": source_read_start,
                    "local_source_end": source_read_end,
                    "source_read_start": source_read_start,
                    "source_read_end": source_read_end,
                },
                local_source_durations_ns={
                    "local_source": max(0, source_read_end - source_read_start),
                    "source_read": max(0, source_read_end - source_read_start),
                },
            )
            if split_hil and callable(split_capability):
                hil_raw = hil_result
            else:
                hil_feature_result = hil_result
        elif hasattr(self.backend, "infer_validated"):
            inference = self.backend.infer_validated(
                inputs,
                input_validate_ns=durations["preprocess"],
                input_validate_start=timestamps["preprocess_start"],
                input_validate_end=timestamps["preprocess_end"],
            )
        else:
            inference = self.backend.infer(inputs)
        if hil_raw is None and hil_feature_result is None:
            # The generic path may expose mutable runtime buffers too.
            inference = dict(inference, outputs=MappingProxyType({
                name: _frozen_numpy_snapshot(value)
                for name, value in inference["outputs"].items()
            }))
        self._last_inferred_sequence_id = frame.sequence_id
        return InferredRsuFrame(
            frame=frame,
            inference=_freeze_metadata(inference),
            timestamps_ns=MappingProxyType(timestamps),
            durations_ns=MappingProxyType(durations),
            source_batch_order=tuple(source_batch_order),
            hil_raw=hil_raw,
            hil_feature_result=hil_feature_result,
            _producer=self._inferred_producer,
        )

    def encode_inferred_frame(self, inferred):
        """Run on one finish thread, possibly overlapping the next infer_frame."""
        if type(inferred) is not InferredRsuFrame or inferred._producer is not self._inferred_producer:
            raise ValueError("inferred frame belongs to a different pipeline producer")
        frame = inferred.frame
        if frame.sequence_id <= self._last_sequence_id:
            raise ValueError("encoded frame sequence must increase strictly")
        timestamps = dict(inferred.timestamps_ns)
        durations = dict(inferred.durations_ns)
        source_batch_order = inferred.source_batch_order
        hil_feature_result = inferred.hil_feature_result
        inference = _plain_metadata(inferred.inference)
        if inferred.hil_raw is not None:
            hil_feature_result = self.backend.finish_hil_inference(inferred.hil_raw)
        if hil_feature_result is not None:
            from coptv2x_hil_backend import (
                CODRIVING_RSU_BATCH_ORDER, COPTV2X_RSU_PAYLOAD_SEMANTICS,
            )
            inference = {
                "outputs": hil_feature_result.outputs,
                "model": dict(hil_feature_result.model),
                "durations_ns": dict(hil_feature_result.durations_ns),
                "edge_timestamps_ns": dict(hil_feature_result.timestamps_ns),
            }
        timestamps.update(
            (str(name), int(value))
            for name, value in inference.get("edge_timestamps_ns", {}).items()
        )
        durations.update(
            (str(name), int(value))
            for name, value in inference.get("durations_ns", {}).items()
        )
        if hil_feature_result is not None:
            # Snapshot/hash work and wire codec work are distinct intervals.
            # Preserve both rather than overwriting the backend measurement.
            durations["feature_pack"] = int(inference["durations_ns"].get("pack", 0))
            durations["feature_hash"] = int(inference["durations_ns"].get("hash", 0))

        timestamps["postprocess_start"] = monotonic_raw_ns()
        outputs = {
            name: np.ascontiguousarray(value)
            for name, value in inference["outputs"].items()
        }
        if self.checksum_dedup:
            # Preserve the backend's exact-array proof across identity postprocess.
            outputs = hil_feature_result.outputs
        timestamps["postprocess_end"] = monotonic_raw_ns()
        durations["postprocess"] = max(
            0, timestamps["postprocess_end"] - timestamps["postprocess_start"]
        )

        timestamps["encode_start"] = monotonic_raw_ns()
        codec_options = {"feature_checksum": True} if self.checksum_dedup else {}
        tensor_bundle, payload = encode_tensor_bundle(
            outputs, compression=self.compression, limits=self.limits, **codec_options
        )
        timestamps["encode_end"] = monotonic_raw_ns()
        durations["encode"] = max(
            0, timestamps["encode_end"] - timestamps["encode_start"]
        )
        codec_stages = tensor_bundle.get("codec_stages_ns", {})
        timestamps.update(
            (str(name), int(value))
            for name, value in tensor_bundle.get("codec_timestamps_ns", {}).items()
        )
        durations["pack"] = int(codec_stages.get("pack_ns", 0))
        durations["hash"] = int(codec_stages.get("hash_ns", 0))
        durations["compress"] = int(codec_stages.get("compress_ns", 0))

        header = {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "result",
            "request_id": "%s:%d" % (self.run_id, frame.sequence_id),
            "status": "ok",
            "run_id": self.run_id,
            "route_id": self.route_id,
            "scenario_id": self.scenario_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "frame_id": int(frame.frame_id),
            "sequence_id": int(frame.sequence_id),
            "source_type": frame.source_metadata.get(
                "source_type", "local_feature_replay"
            ),
            "payload_semantics": self.payload_semantics,
            "postprocess": "identity",
            "remote_input_tensor_bytes": 0,
            "edge_clock": {
                "host_id": self.host_id,
                "domain": "monotonic_raw",
                "boot_id": self.host_boot_id,
                "sync_state": "locked",
                "offset_ns": 0,
                "uncertainty_ns": 0,
            },
            "model": copy.deepcopy(inference["model"]),
            "device": collect_device_snapshot(),
            "edge_timestamps_ns": dict(timestamps),
            "durations_ns": dict(durations),
        }
        if hil_feature_result is not None:
            if hil_feature_result.payload_semantics != self.payload_semantics:
                raise ValueError("HIL feature payload semantics drifted during inference")
            if hil_feature_result.remote_input_tensor_bytes != 0:
                raise ValueError("HIL result remote_input_tensor_bytes must be zero")
            trigger_identity = dict(hil_feature_result.trigger_identity)
            if trigger_identity != dict(frame.source_metadata["trigger_identity"]):
                raise ValueError("HIL trigger identity drifted during inference")
            header.update(trigger_identity)
            header.update(
                {
                    "trigger_identity": trigger_identity,
                    "artifact_identity": {
                        key: (
                            dict(value)
                            if isinstance(value, Mapping)
                            else list(value)
                            if isinstance(value, tuple)
                            else value
                        )
                        for key, value in hil_feature_result.artifact_identity.items()
                    },
                    "payload_sha256": hil_feature_result.payload_sha256,
                    "batch_order": list(
                        CODRIVING_RSU_BATCH_ORDER
                        if self.payload_semantics == COPTV2X_RSU_PAYLOAD_SEMANTICS
                        else source_batch_order
                    ),
                }
            )
            if self.payload_semantics == COPTV2X_RSU_PAYLOAD_SEMANTICS:
                header["source_batch_order"] = list(source_batch_order)
        self._last_sequence_id = frame.sequence_id
        if hil_feature_result is not None:
            # Elapsed Orin-local interval, including metadata work, never stage sums.
            timestamps["local_processing_start"] = timestamps.get(
                "source_read_start", int(frame.data_available_ns)
            )
            timestamps["local_processing_end"] = monotonic_raw_ns()
            durations["local_processing"] = max(
                0, timestamps["local_processing_end"] - timestamps["local_processing_start"]
            )
            header["edge_timestamps_ns"] = dict(timestamps)
            header["durations_ns"] = dict(durations)
        return EncodedRsuResult(
            sequence_id=frame.sequence_id,
            header=MappingProxyType(header),
            payload=payload,
            tensor_bundle=MappingProxyType(tensor_bundle),
            timestamps_ns=MappingProxyType(timestamps),
            durations_ns=MappingProxyType(durations),
            outputs=MappingProxyType(outputs),
        )


class CoptV2xHilPipeline(RsuEdgePipeline):
    """Explicit CoDriving multiscale entry; legacy result semantics stay unchanged."""

    def __init__(self, backend, *args, **kwargs):
        from coptv2x_hil_backend import COPTV2X_PAYLOAD_SEMANTICS, COPTV2X_RSU_PAYLOAD_SEMANTICS

        if getattr(backend, "payload_semantics", None) not in (
            COPTV2X_PAYLOAD_SEMANTICS, COPTV2X_RSU_PAYLOAD_SEMANTICS,
        ):
            raise ValueError("CoptV2X pipeline requires the multiscale HIL backend")
        if "payload_semantics" in kwargs:
            raise ValueError("CoptV2X payload semantics are fixed")
        super(CoptV2xHilPipeline, self).__init__(
            backend,
            *args,
            payload_semantics=backend.payload_semantics,
            **kwargs
        )
