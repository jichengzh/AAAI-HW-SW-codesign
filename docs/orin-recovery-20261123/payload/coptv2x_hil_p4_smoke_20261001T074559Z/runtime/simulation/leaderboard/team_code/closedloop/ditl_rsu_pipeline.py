from __future__ import print_function

import copy
import socket
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ditl_edge_protocol import PROTOCOL_VERSION, ProtocolLimits, encode_tensor_bundle
from ditl_edge_runtime import boot_id, collect_device_snapshot, monotonic_raw_ns
from ditl_trt_backend import validate_input_arrays


@dataclass(frozen=True)
class EncodedRsuResult:
    sequence_id: int
    header: Mapping[str, object]
    payload: bytes
    tensor_bundle: Mapping[str, object]
    timestamps_ns: Mapping[str, int]
    durations_ns: Mapping[str, int]
    outputs: Mapping[str, np.ndarray]


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
        if (
            hasattr(self.backend, "infer_hil_validated")
            and self.payload_semantics != "codriving_multiscale_features"
        ):
            raise ValueError(
                "CoptV2X HIL backend requires the explicit CoptV2xHilPipeline"
            )
        self._last_sequence_id = -1

    def process(self, frame):
        if frame.sequence_id <= self._last_sequence_id:
            raise ValueError("frame sequence must increase strictly")

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
        if hasattr(self.backend, "infer_hil_validated"):
            from coptv2x_hil_backend import CODRIVING_BATCH_ORDER
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
            hil_feature_result = self.backend.infer_hil_validated(
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
            inference = {
                "outputs": hil_feature_result.outputs,
                "model": dict(hil_feature_result.model),
                "durations_ns": dict(hil_feature_result.durations_ns),
                "edge_timestamps_ns": dict(hil_feature_result.timestamps_ns),
            }
        elif hasattr(self.backend, "infer_validated"):
            inference = self.backend.infer_validated(
                inputs,
                input_validate_ns=durations["preprocess"],
                input_validate_start=timestamps["preprocess_start"],
                input_validate_end=timestamps["preprocess_end"],
            )
        else:
            inference = self.backend.infer(inputs)
        timestamps.update(
            (str(name), int(value))
            for name, value in inference.get("edge_timestamps_ns", {}).items()
        )
        durations.update(
            (str(name), int(value))
            for name, value in inference.get("durations_ns", {}).items()
        )

        timestamps["postprocess_start"] = monotonic_raw_ns()
        outputs = {
            name: np.ascontiguousarray(value)
            for name, value in inference["outputs"].items()
        }
        timestamps["postprocess_end"] = monotonic_raw_ns()
        durations["postprocess"] = max(
            0, timestamps["postprocess_end"] - timestamps["postprocess_start"]
        )

        timestamps["encode_start"] = monotonic_raw_ns()
        tensor_bundle, payload = encode_tensor_bundle(
            outputs,
            compression=self.compression,
            limits=self.limits,
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
                    "batch_order": list(source_batch_order),
                }
            )
        self._last_sequence_id = frame.sequence_id
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
        from coptv2x_hil_backend import COPTV2X_PAYLOAD_SEMANTICS

        if getattr(backend, "payload_semantics", None) != COPTV2X_PAYLOAD_SEMANTICS:
            raise ValueError("CoptV2X pipeline requires the multiscale HIL backend")
        if "payload_semantics" in kwargs:
            raise ValueError("CoptV2X payload semantics are fixed")
        super(CoptV2xHilPipeline, self).__init__(
            backend,
            *args,
            payload_semantics=COPTV2X_PAYLOAD_SEMANTICS,
            **kwargs
        )
