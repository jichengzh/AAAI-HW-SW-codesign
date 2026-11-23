from __future__ import print_function

import hashlib
import sys
import copy
from pathlib import Path

import numpy as np

from ditl_edge_runtime import monotonic_raw_ns


def validate_input_arrays(contract, arrays):
    if not isinstance(contract, dict) or not contract:
        raise ValueError("input contract must be a non-empty object")
    if not isinstance(arrays, dict) or set(arrays) != set(contract):
        raise ValueError(
            "input tensor names must exactly match %s" % sorted(contract)
        )
    validated = {}
    for name in sorted(contract):
        specification = contract[name]
        expected_shape = tuple(specification["shape"])
        expected_dtype = np.dtype(specification["dtype"])
        value = np.asarray(arrays[name])
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "input %s shape %s does not match %s"
                % (name, tuple(value.shape), expected_shape)
            )
        if value.dtype != expected_dtype:
            raise ValueError(
                "input %s dtype %s does not match %s"
                % (name, value.dtype.name, expected_dtype.name)
            )
        validated[name] = np.array(value, copy=True, order="C")
    return validated


def _load_gpu_modules():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("TensorRtBackend requires CUDA-enabled torch") from error
    try:
        import tensorrt as trt
    except ImportError:
        system_packages = "/usr/lib/python3.8/dist-packages"
        if system_packages not in sys.path:
            sys.path.append(system_packages)
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRtBackend requires TensorRT Python") from error
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRtBackend requires an available CUDA device")
    return torch, trt


def _torch_dtype(torch, numpy_dtype):
    mapping = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool"): torch.bool,
    }
    try:
        return mapping[np.dtype(numpy_dtype)]
    except KeyError as error:
        raise RuntimeError("unsupported TensorRT binding dtype %s" % numpy_dtype) from error


class TensorRtBackend(object):
    """Static-shape TensorRT backend using torch CUDA buffers and CUDA events."""

    def __init__(self, engine_path, model_id=None):
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError("TensorRT engine not found: %s" % self.engine_path)
        self._torch, self._trt = _load_gpu_modules()
        self._logger = self._trt.Logger(self._trt.Logger.ERROR)
        engine_bytes = self.engine_path.read_bytes()
        self.engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
        self._runtime = self._trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self.model_id = model_id or self.engine_path.stem
        self.inference_count = 0
        self.input_contract = {}
        self.output_contract = {}
        self._binding_indices = {}
        for index in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(index)
            shape = tuple(int(v) for v in self._engine.get_binding_shape(index))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError("dynamic TensorRT bindings are not supported")
            dtype = np.dtype(self._trt.nptype(self._engine.get_binding_dtype(index)))
            specification = {"shape": shape, "dtype": dtype.name}
            self._binding_indices[name] = index
            if self._engine.binding_is_input(index):
                self.input_contract[name] = specification
            else:
                self.output_contract[name] = specification
        if not self.input_contract or not self.output_contract:
            raise RuntimeError("TensorRT engine requires input and output bindings")

    def public_model_metadata(self):
        return {
            "backend": "tensorrt",
            "model_id": self.model_id,
            "engine_sha256": self.engine_sha256,
            "input_contract": copy.deepcopy(self.input_contract),
            "output_contract": copy.deepcopy(self.output_contract),
        }

    def infer(self, arrays):
        inputs = validate_input_arrays(self.input_contract, arrays)
        torch = self._torch
        timestamps = {}

        timestamps["h2d_start"] = monotonic_raw_ns()
        input_tensors = {}
        for name, value in inputs.items():
            dtype = _torch_dtype(torch, value.dtype)
            input_tensors[name] = torch.from_numpy(value).to(
                device="cuda", dtype=dtype, non_blocking=False
            ).contiguous()
        torch.cuda.synchronize()
        timestamps["h2d_end"] = monotonic_raw_ns()

        output_tensors = {}
        for name, specification in self.output_contract.items():
            dtype = _torch_dtype(torch, np.dtype(specification["dtype"]))
            output_tensors[name] = torch.empty(
                specification["shape"], device="cuda", dtype=dtype
            )

        bindings = [0] * self._engine.num_bindings
        for name, tensor in input_tensors.items():
            bindings[self._binding_indices[name]] = int(tensor.data_ptr())
        for name, tensor in output_tensors.items():
            bindings[self._binding_indices[name]] = int(tensor.data_ptr())

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timestamps["inference_start"] = monotonic_raw_ns()
        start_event.record()
        if not self._context.execute_v2(bindings):
            raise RuntimeError("TensorRT execute_v2 returned false")
        end_event.record()
        end_event.synchronize()
        timestamps["inference_end"] = monotonic_raw_ns()
        inference_ns = int(round(start_event.elapsed_time(end_event) * 1_000_000.0))

        timestamps["d2h_start"] = monotonic_raw_ns()
        outputs = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in output_tensors.items()
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
