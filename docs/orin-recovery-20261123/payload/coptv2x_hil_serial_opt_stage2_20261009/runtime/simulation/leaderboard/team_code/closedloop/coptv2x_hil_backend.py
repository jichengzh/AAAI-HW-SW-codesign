from __future__ import print_function

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping

import numpy as np

from coptv2x_hil_artifacts import (
    DEFAULT_CONFIG_DIR,
    HilArtifactSpec,
    load_hil_artifact_registry,
)
from coptv2x_hil_trigger import HilTrigger
from ditl_edge_runtime import monotonic_raw_ns
from ditl_trt_backend import TensorRtBackend, validate_input_arrays
from ditl_edge_protocol import canonical_feature_payload_sha256, _verified_feature_arrays


COPTV2X_PAYLOAD_SEMANTICS = "codriving_multiscale_features"
COPTV2X_RSU_PAYLOAD_SEMANTICS = "codriving_rsu_multiscale_features.v1"
CODRIVING_BATCH_ORDER = ("ego", "rsu")
CODRIVING_RSU_BATCH_ORDER = ("rsu",)
_OUTPUT_NAMES = ("res0", "res1", "res2")
_DEFAULT_NATIVE_REPO_ROOT = Path(
    "/home/jichengzhi/lane_c_orin_stage6_five_config_20260724/code/V2Xverse"
)


class HilBackendError(ValueError):
    """Raised when an online HIL backend cannot prove its frozen contract."""


@dataclass(frozen=True)
class HilFeatureResult:
    outputs: Mapping[str, np.ndarray]
    artifact_identity: Mapping[str, object]
    trigger_identity: Mapping[str, object]
    payload_semantics: str
    payload_sha256: str
    remote_input_tensor_bytes: int
    timestamps_ns: Mapping[str, int]
    durations_ns: Mapping[str, int]
    model: Mapping[str, object]


def _contract_mapping(contract):
    return {
        tensor.name: {"shape": tuple(tensor.shape), "dtype": tensor.dtype}
        for tensor in contract
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root, relative_path, label):
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise HilBackendError("%s escapes the artifact root" % label) from error
    if not path.is_file():
        raise HilBackendError("%s is missing: %s" % (label, path))
    return path


def _verify_artifact_files(spec, artifact_root):
    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise HilBackendError("artifact root is missing: %s" % root)
    relative_paths = {"config": spec.config, "checkpoint": spec.checkpoint}
    if spec.engine is not None:
        relative_paths["engine"] = spec.engine
    verified = {}
    for name in sorted(relative_paths):
        path = _artifact_path(root, relative_paths[name], "%s %s" % (spec.arm_id, name))
        actual = _sha256_file(path)
        expected = spec.artifact_sha256[name]
        if actual != expected:
            raise HilBackendError(
                "%s %s SHA-256 mismatch: expected %s, got %s"
                % (spec.arm_id, name, expected, actual)
            )
        verified[name] = path
    return MappingProxyType(verified)


def _all_finite(value):
    """Scan C-order bounded chunks without a tensor-sized boolean temporary."""
    array = np.asarray(value)
    chunk_elements = 262144
    if array.flags.c_contiguous:
        flat = array.reshape(-1)
        chunks = (
            flat[start:start + chunk_elements]
            for start in range(0, flat.size, chunk_elements)
        )
    else:
        # Buffered iteration bounds both the copy and finite mask for strided views.
        chunks = np.nditer(
            array, flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"], order="C", buffersize=chunk_elements,
        )
    return all(np.isfinite(chunk).all() for chunk in chunks)


def hash_feature_payload(outputs):
    """Hash tensor identity and canonical bytes without relying on wire encoding."""

    if not isinstance(outputs, Mapping) or tuple(sorted(outputs)) != _OUTPUT_NAMES:
        raise HilBackendError("feature payload must contain only res0/res1/res2")
    return canonical_feature_payload_sha256(outputs)


def validate_feature_payload_contract(
    artifact_identity, outputs, config_dir=DEFAULT_CONFIG_DIR,
    payload_semantics=COPTV2X_PAYLOAD_SEMANTICS,
    _registry=None,
):
    """Validate a decoded feature result against Task1 registry authority."""

    if payload_semantics not in (COPTV2X_PAYLOAD_SEMANTICS, COPTV2X_RSU_PAYLOAD_SEMANTICS):
        raise HilBackendError("unknown feature payload semantics")
    if not isinstance(artifact_identity, dict):
        raise HilBackendError("artifact identity must be an object")
    arm_id = artifact_identity.get("arm_id")
    registry = load_hil_artifact_registry(config_dir) if _registry is None else _registry
    try:
        spec = registry[arm_id]
    except KeyError as error:
        raise HilBackendError("artifact identity arm is not frozen") from error
    if artifact_identity != _plain_artifact_identity(spec):
        raise HilBackendError("artifact identity does not match Task1 registry")
    if not isinstance(outputs, Mapping) or set(outputs) != set(_OUTPUT_NAMES):
        raise HilBackendError("feature output contract requires res0/res1/res2")
    expected_contract = _contract_mapping(spec.output_contract)
    for name in _OUTPUT_NAMES:
        value = np.asarray(outputs[name])
        expected = expected_contract[name]
        expected_shape = tuple(expected["shape"])
        if payload_semantics == COPTV2X_RSU_PAYLOAD_SEMANTICS:
            expected_shape = (1,) + expected_shape[1:]
        if (
            tuple(value.shape) != expected_shape
            or value.dtype != np.dtype(expected["dtype"])
        ):
            raise HilBackendError("%s shape/dtype violates artifact contract" % name)
        if not _all_finite(value):
            raise HilBackendError("%s finite-value contract failed" % name)
    return spec


def create_feature_payload_validator(config_dir=DEFAULT_CONFIG_DIR):
    """Freeze validated registry authority once for an immutable experiment run.

    This does not cache payload acceptance: identity, shape, dtype and finite
    values are rechecked on every call. Start a new validator after deployment
    configuration changes; live reconfiguration is not this run's contract.
    """
    registry = MappingProxyType(dict(load_hil_artifact_registry(config_dir)))

    def validate(artifact_identity, outputs, payload_semantics=COPTV2X_PAYLOAD_SEMANTICS):
        return validate_feature_payload_contract(
            artifact_identity, outputs, config_dir,
            payload_semantics=payload_semantics, _registry=registry)

    return validate


def _immutable_bytes_owner(value):
    """Readonly views alone are unsafe: a writable base may be reused later."""
    owner = value
    while isinstance(owner, (np.ndarray, memoryview)):
        if isinstance(owner, np.ndarray):
            if owner.flags.writeable:
                return False
            owner = owner.base
        else:
            if not owner.readonly:
                return False
            owner = owner.obj
    return isinstance(owner, bytes)


def _frozen_numpy_snapshot(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _public_artifact_identity(spec):
    identity = {
        "arm_id": spec.arm_id,
        "model": spec.model,
        "widths": tuple(spec.widths),
        "precision": spec.precision,
        "runtime": spec.runtime,
        "config": str(spec.config),
        "checkpoint": str(spec.checkpoint),
        "engine": None if spec.engine is None else str(spec.engine),
        "artifact_sha256": MappingProxyType(dict(spec.artifact_sha256)),
    }
    return MappingProxyType(identity)


def _plain_artifact_identity(spec):
    return {
        "arm_id": spec.arm_id,
        "model": spec.model,
        "widths": list(spec.widths),
        "precision": spec.precision,
        "runtime": spec.runtime,
        "config": str(spec.config),
        "checkpoint": str(spec.checkpoint),
        "engine": None if spec.engine is None else str(spec.engine),
        "artifact_sha256": dict(spec.artifact_sha256),
    }


class NativePyTorchBackend(object):
    """Exact frozen CoDriving ResNet multiscale callable for the default arm."""

    def __init__(
        self,
        config_path,
        checkpoint_path,
        repo_root,
        model_id,
        checkpoint_sha256,
        input_contract,
        output_contract,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.is_dir():
            raise RuntimeError("native CoDriving repo root is missing: %s" % self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        try:
            import torch
            from opencood.hypes_yaml.yaml_utils import load_yaml
            from opencood.tools import train_utils
        except ImportError as error:
            raise RuntimeError("native CoDriving backend dependencies are unavailable") from error
        if not torch.cuda.is_available():
            raise RuntimeError("native CoDriving backend requires an available CUDA device")
        hypes = load_yaml(str(config_path))
        model = train_utils.create_model(hypes)
        state = torch.load(str(checkpoint_path), map_location="cpu")
        train_utils.check_missing_key(model.state_dict(), state)
        model.load_state_dict(state, strict=False)
        self._torch = torch
        self._model = model.cuda().eval()
        self._callable = self._model.backbone.resnet
        self.model_id = str(model_id)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.input_contract = copy.deepcopy(input_contract)
        self.output_contract = copy.deepcopy(output_contract)
        self.inference_count = 0

    def public_model_metadata(self):
        return {
            "backend": "native_pytorch",
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "input_contract": copy.deepcopy(self.input_contract),
            "output_contract": copy.deepcopy(self.output_contract),
        }

    def infer_validated(self, inputs, **unused):
        return self._infer_validated(inputs)

    def infer_validated_rsu(self, inputs):
        """Compute the frozen batch-2 model, transferring only its RSU slice."""
        if any(
            not specification["shape"] or specification["shape"][0] != 2
            for specification in self.output_contract.values()
        ):
            raise ValueError("RSU-only inference requires batch 2 output bindings")
        return self._infer_validated(inputs, rsu_only=True)

    def _infer_validated(self, inputs, rsu_only=False):
        torch = self._torch
        timestamps = {"h2d_start": monotonic_raw_ns()}
        tensor = torch.from_numpy(inputs["spatial_features"]).to(
            device="cuda", dtype=torch.float32, non_blocking=False
        ).contiguous()
        torch.cuda.synchronize()
        timestamps["h2d_end"] = monotonic_raw_ns()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timestamps["inference_start"] = monotonic_raw_ns()
        with torch.inference_mode():
            start_event.record()
            raw_outputs = tuple(self._callable(tensor))
            end_event.record()
            end_event.synchronize()
        timestamps["inference_end"] = monotonic_raw_ns()
        inference_ns = int(round(start_event.elapsed_time(end_event) * 1000000.0))

        if rsu_only:
            if len(raw_outputs) != len(_OUTPUT_NAMES):
                raise RuntimeError("native output keys violate frozen batch 2 contract")
            for name, value in zip(_OUTPUT_NAMES, raw_outputs):
                if tuple(value.shape) != tuple(self.output_contract[name]["shape"]):
                    raise RuntimeError("native output shape violates frozen batch 2 contract")

        timestamps["d2h_start"] = monotonic_raw_ns()
        outputs = {
            name: _frozen_numpy_snapshot(
                (value[1:2] if rsu_only else value).detach().cpu().float().numpy()
            )
            for name, value in zip(_OUTPUT_NAMES, raw_outputs)
        }
        torch.cuda.synchronize()
        timestamps["d2h_end"] = monotonic_raw_ns()
        self.inference_count += 1
        return {
            "outputs": outputs,
            "model": self.public_model_metadata(),
            "durations_ns": {
                "h2d": timestamps["h2d_end"] - timestamps["h2d_start"],
                "gpu_inference": inference_ns,
                "d2h": timestamps["d2h_end"] - timestamps["d2h_start"],
            },
            "edge_timestamps_ns": timestamps,
        }


def _default_runtime_factory(
    spec, paths, input_contract, output_contract, native_repo_root
):
    if spec.runtime == "tensorrt":
        return TensorRtBackend(
            paths["engine"],
            model_id="%s:%s" % (spec.model, spec.arm_id),
            expected_engine_sha256=spec.artifact_sha256["engine"],
            expected_input_contract=input_contract,
            expected_output_contract=output_contract,
        )
    if spec.runtime == "native_pytorch":
        return NativePyTorchBackend(
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            repo_root=native_repo_root,
            model_id="%s:%s" % (spec.model, spec.arm_id),
            checkpoint_sha256=spec.artifact_sha256["checkpoint"],
            input_contract=input_contract,
            output_contract=output_contract,
        )
    raise HilBackendError("unsupported frozen runtime %r" % spec.runtime)


class CoptV2xHilBackend(object):
    """One-arm-per-instance online backend for frozen CoDriving features."""

    payload_semantics = COPTV2X_PAYLOAD_SEMANTICS
    batch_order = CODRIVING_BATCH_ORDER

    def __init__(
        self,
        arm_id,
        artifact_root,
        config_dir=DEFAULT_CONFIG_DIR,
        runtime_factory=None,
        native_repo_root=_DEFAULT_NATIVE_REPO_ROOT,
        rsu_only=False,
        trust_readonly_inputs=False,
        verified_input_source=None,
        checksum_dedup=False,
    ):
        if verified_input_source is not None:
            from ditl_rsu_source import IndexedLocalRsuSource

            if (
                type(verified_input_source) is not IndexedLocalRsuSource
                or not verified_input_source.preload_verified
            ):
                raise HilBackendError(
                    "verified_input_source must be a preload_verified IndexedLocalRsuSource"
                )
        self._verified_input_source = verified_input_source
        self.rsu_only = bool(rsu_only)
        if checksum_dedup and not self.rsu_only:
            raise HilBackendError("checksum_dedup requires rsu_only")
        self.checksum_dedup = bool(checksum_dedup)
        self.trust_readonly_inputs = bool(trust_readonly_inputs)
        self.payload_semantics = (
            COPTV2X_RSU_PAYLOAD_SEMANTICS if self.rsu_only else COPTV2X_PAYLOAD_SEMANTICS
        )
        self.batch_order = CODRIVING_RSU_BATCH_ORDER if self.rsu_only else CODRIVING_BATCH_ORDER
        registry = load_hil_artifact_registry(config_dir)
        try:
            self.spec = registry[arm_id]
        except KeyError as error:
            raise HilBackendError("unknown frozen HIL arm %r" % arm_id) from error
        if not isinstance(self.spec, HilArtifactSpec):
            raise HilBackendError("registry did not return HilArtifactSpec")
        self.arm_id = self.spec.arm_id
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.native_repo_root = Path(native_repo_root).expanduser().resolve()
        self.input_contract = _contract_mapping(self.spec.input_contract)
        self.output_contract = _contract_mapping(self.spec.output_contract)
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._runtime = None
        self._artifact_paths = None
        self.inference_count = 0

    def load(self, arm_id=None):
        requested = self.arm_id if arm_id is None else arm_id
        if requested != self.arm_id:
            raise HilBackendError(
                "requested arm %s does not match fixed run arm %s"
                % (requested, self.arm_id)
            )
        if self._runtime is not None:
            return self._runtime
        paths = _verify_artifact_files(self.spec, self.artifact_root)
        try:
            runtime = self._runtime_factory(
                self.spec,
                paths,
                copy.deepcopy(self.input_contract),
                copy.deepcopy(self.output_contract),
                self.native_repo_root,
            )
        except HilBackendError:
            raise
        except Exception as error:
            raise HilBackendError(
                "%s runtime load failed: %s" % (self.arm_id, error)
            ) from error
        if getattr(runtime, "input_contract", None) != self.input_contract:
            raise HilBackendError("runtime input binding contract mismatch")
        if getattr(runtime, "output_contract", None) != self.output_contract:
            raise HilBackendError("runtime output binding contract mismatch")
        self._artifact_paths = paths
        self._runtime = runtime
        return runtime

    def validate_inputs(self, arrays):
        try:
            if self.trust_readonly_inputs:
                values = dict(arrays)
                if set(values) != set(self.input_contract):
                    raise ValueError("input tensor names must exactly match frozen contract")
                validated = {}
                for name, expected in self.input_contract.items():
                    value = np.asarray(values[name])
                    if tuple(value.shape) != tuple(expected["shape"]) or value.dtype != np.dtype(expected["dtype"]):
                        raise ValueError("%s shape/dtype violates input contract" % name)
                    verified_snapshot = (
                        self._verified_input_source is not None
                        and self._verified_input_source.owns_verified_array(name, value)
                    )
                    if not verified_snapshot and not _all_finite(value):
                        raise ValueError("%s finite-value contract failed" % name)
                    # Only a controlled worker may promise never to thaw/reuse this owner.
                    reusable = (value.flags.owndata or _immutable_bytes_owner(value))
                    if reusable and value.flags.c_contiguous and not value.flags.writeable:
                        validated[name] = value
                    else:
                        validated[name] = np.array(value, copy=True, order="C")
                return validated
            return validate_input_arrays(self.input_contract, dict(arrays))
        except (TypeError, ValueError) as error:
            raise HilBackendError("CoDriving input contract rejected: %s" % error) from error

    def infer(
        self,
        spatial_features,
        trigger,
        batch_order,
        local_source_timestamps_ns=None,
        local_source_durations_ns=None,
    ):
        preprocess_start = monotonic_raw_ns()
        inputs = self.validate_inputs({"spatial_features": spatial_features})
        preprocess_end = monotonic_raw_ns()
        return self.infer_hil_validated(
            inputs,
            trigger,
            batch_order=batch_order,
            preprocess_start=preprocess_start,
            preprocess_end=preprocess_end,
            local_source_timestamps_ns=local_source_timestamps_ns,
            local_source_durations_ns=local_source_durations_ns,
        )

    def infer_hil_validated(
        self,
        inputs,
        trigger,
        batch_order,
        preprocess_start=None,
        preprocess_end=None,
        local_source_timestamps_ns=None,
        local_source_durations_ns=None,
    ):
        if type(trigger) is not HilTrigger:
            raise HilBackendError("trigger must be a validated HilTrigger")
        if trigger.arm_id != self.arm_id:
            raise HilBackendError("trigger arm does not match fixed run arm")
        if tuple(batch_order) != CODRIVING_BATCH_ORDER:
            raise HilBackendError(
                "CoDriving batch order must be ego then rsu, got %r" % (batch_order,)
            )
        if set(inputs) != {"spatial_features"}:
            raise HilBackendError("CoDriving input must contain only spatial_features")
        value = inputs["spatial_features"]
        expected = self.input_contract["spatial_features"]
        if tuple(value.shape) != tuple(expected["shape"]) or value.dtype != np.dtype(
            expected["dtype"]
        ):
            raise HilBackendError("validated CoDriving input contract drifted")
        runtime = self.load()
        if preprocess_start is None:
            preprocess_start = monotonic_raw_ns()
        if preprocess_end is None:
            preprocess_end = preprocess_start
        rsu_capability = getattr(runtime, "infer_validated_rsu", None)
        projected_runtime = self.rsu_only and callable(rsu_capability)
        inference = (
            rsu_capability(inputs) if projected_runtime else runtime.infer_validated(inputs)
        )

        timestamps = {
            str(name): int(timestamp)
            for name, timestamp in (local_source_timestamps_ns or {}).items()
        }
        timestamps.update(
            {
                "preprocess_start": int(preprocess_start),
                "preprocess_end": int(preprocess_end),
            }
        )
        timestamps.update(
            (str(name), int(timestamp))
            for name, timestamp in inference.get("edge_timestamps_ns", {}).items()
        )
        durations = {
            str(name): int(duration)
            for name, duration in (local_source_durations_ns or {}).items()
        }
        durations["preprocess"] = max(0, preprocess_end - preprocess_start)
        runtime_durations = inference.get("durations_ns", {})
        durations["h2d"] = int(runtime_durations.get("h2d", 0))
        durations["inference"] = int(runtime_durations.get("gpu_inference", 0))
        durations["d2h"] = int(runtime_durations.get("d2h", 0))
        if "rsu_buffer_setup" in runtime_durations:
            durations["rsu_buffer_setup"] = int(runtime_durations["rsu_buffer_setup"])

        timestamps["pack_start"] = monotonic_raw_ns()
        outputs = {}
        raw_outputs = inference.get("outputs")
        if not isinstance(raw_outputs, dict) or set(raw_outputs) != set(_OUTPUT_NAMES):
            raise HilBackendError("runtime output keys must be exactly res0/res1/res2")
        for name in _OUTPUT_NAMES:
            expected = self.output_contract[name]
            value = np.asarray(raw_outputs[name])
            expected_shape = tuple(expected["shape"])
            if projected_runtime:
                expected_shape = (1,) + expected_shape[1:]
            if tuple(value.shape) != expected_shape:
                raise HilBackendError("%s output shape violates artifact contract" % name)
            if value.dtype != np.dtype(expected["dtype"]):
                raise HilBackendError("%s output dtype violates artifact contract" % name)
            if not _all_finite(value):
                raise HilBackendError("%s output must contain only finite values" % name)
            projected = value[1:2] if self.rsu_only and not projected_runtime else value
            if projected.flags.c_contiguous and _immutable_bytes_owner(projected):
                outputs[name] = projected
            elif self.checksum_dedup:
                # The checksum proof must outlive runtime buffer reuse safely.
                outputs[name] = np.frombuffer(
                    projected.tobytes(order="C"), dtype=projected.dtype,
                ).reshape(projected.shape)
            else:
                copied = np.array(projected, copy=True, order="C")
                copied.setflags(write=False)
                outputs[name] = copied
        timestamps["pack_end"] = monotonic_raw_ns()
        durations["pack"] = max(0, timestamps["pack_end"] - timestamps["pack_start"])
        timestamps["hash_start"] = monotonic_raw_ns()
        payload_sha256 = hash_feature_payload(outputs)
        timestamps["hash_end"] = monotonic_raw_ns()
        durations["hash"] = max(0, timestamps["hash_end"] - timestamps["hash_start"])
        verified_outputs = (
            _verified_feature_arrays(outputs, payload_sha256)
            if self.checksum_dedup else MappingProxyType(outputs)
        )
        if "local_source" not in durations:
            durations["local_source"] = max(
                0,
                int(timestamps.get("local_source_end", 0))
                - int(timestamps.get("local_source_start", 0)),
            )
        self.inference_count += 1
        model = copy.deepcopy(inference.get("model", {}))
        model["artifact_identity"] = _plain_artifact_identity(self.spec)
        return HilFeatureResult(
            outputs=verified_outputs,
            artifact_identity=_public_artifact_identity(self.spec),
            trigger_identity=MappingProxyType(trigger.to_metadata()),
            payload_semantics=self.payload_semantics,
            payload_sha256=payload_sha256,
            remote_input_tensor_bytes=0,
            timestamps_ns=MappingProxyType(timestamps),
            durations_ns=MappingProxyType(durations),
            model=MappingProxyType(model),
        )


__all__ = (
    "CODRIVING_BATCH_ORDER",
    "CODRIVING_RSU_BATCH_ORDER",
    "COPTV2X_PAYLOAD_SEMANTICS",
    "COPTV2X_RSU_PAYLOAD_SEMANTICS",
    "CoptV2xHilBackend",
    "HilBackendError",
    "HilFeatureResult",
    "NativePyTorchBackend",
    "hash_feature_payload",
    "validate_feature_payload_contract",
)
