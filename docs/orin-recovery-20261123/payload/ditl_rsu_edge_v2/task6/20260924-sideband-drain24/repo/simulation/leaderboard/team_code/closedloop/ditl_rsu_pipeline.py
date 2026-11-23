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

        if hasattr(self.backend, "infer_validated"):
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
            "payload_semantics": "dense_head_debug",
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
