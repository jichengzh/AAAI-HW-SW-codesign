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
    """Static-shape TensorRT backend with reusable buffers for serial inference."""

    def __init__(
        self,
        engine_path,
        model_id=None,
        expected_engine_sha256=None,
        expected_input_contract=None,
        expected_output_contract=None,
    ):
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError("TensorRT engine not found: %s" % self.engine_path)
        self._torch, self._trt = _load_gpu_modules()
        self._logger = self._trt.Logger(self._trt.Logger.ERROR)
        engine_bytes = self.engine_path.read_bytes()
        self.engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
        if (
            expected_engine_sha256 is not None
            and self.engine_sha256 != str(expected_engine_sha256).lower()
        ):
            raise RuntimeError(
                "TensorRT engine SHA-256 mismatch: expected %s, got %s"
                % (expected_engine_sha256, self.engine_sha256)
            )
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
        if (
            expected_input_contract is not None
            and self.input_contract != expected_input_contract
        ):
            raise RuntimeError(
                "TensorRT input binding contract mismatch: expected %r, got %r"
                % (expected_input_contract, self.input_contract)
            )
        if (
            expected_output_contract is not None
            and self.output_contract != expected_output_contract
        ):
            raise RuntimeError(
                "TensorRT output binding contract mismatch: expected %r, got %r"
                % (expected_output_contract, self.output_contract)
            )
        self._prepare_buffers()

    def _prepare_buffers(self):
        # Charge allocation/event initialization to setup, not steady-state stages.
        setup_start = monotonic_raw_ns()
        torch = self._torch
        self._input_tensors = {}
        self._output_tensors = {}
        self._bindings = [0] * self._engine.num_bindings
        for contract, tensors in (
            (self.input_contract, self._input_tensors),
            (self.output_contract, self._output_tensors),
        ):
            for name, specification in contract.items():
                tensor = torch.empty(
                    specification["shape"],
                    device="cuda",
                    dtype=_torch_dtype(torch, np.dtype(specification["dtype"])),
                )
                tensors[name] = tensor
                self._bindings[self._binding_indices[name]] = int(tensor.data_ptr())
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        # CUDA events initialize lazily on their first record.
        self._start_event.record()
        self._end_event.record()
        self._end_event.synchronize()
        torch.cuda.synchronize()
        self.buffer_setup_ns = max(0, monotonic_raw_ns() - setup_start)

    def public_model_metadata(self):
        return {
            "backend": "tensorrt",
            "model_id": self.model_id,
            "engine_sha256": self.engine_sha256,
            "input_contract": copy.deepcopy(self.input_contract),
            "output_contract": copy.deepcopy(self.output_contract),
        }

    def _prepare_rsu_buffers(self):
        if hasattr(self, "_rsu_host_outputs"):
            return 0
        if any(
            not specification["shape"] or specification["shape"][0] != 2
            for specification in self.output_contract.values()
        ):
            raise ValueError("RSU-only inference requires batch 2 output bindings")
        setup_start = monotonic_raw_ns()
        torch = self._torch
        self._rsu_host_outputs = {
            name: torch.empty(
                (1,) + tuple(specification["shape"])[1:],
                device="cpu", pin_memory=True,
                dtype=_torch_dtype(torch, np.dtype(specification["dtype"])),
            )
            for name, specification in self.output_contract.items()
        }
        self._rsu_host_output_arrays = {
            name: tensor.numpy() for name, tensor in self._rsu_host_outputs.items()
        }
        torch.cuda.synchronize()
        self.rsu_buffer_setup_ns = max(0, monotonic_raw_ns() - setup_start)
        return self.rsu_buffer_setup_ns

    def validate_inputs(self, arrays):
        return validate_input_arrays(self.input_contract, dict(arrays))

    def infer(self, arrays):
        validation_start = monotonic_raw_ns()
        inputs = self.validate_inputs(arrays)
        validation_end = monotonic_raw_ns()
        return self.infer_validated(
            inputs,
            input_validate_ns=max(0, validation_end - validation_start),
            input_validate_start=validation_start,
            input_validate_end=validation_end,
        )

    def infer_validated(
        self,
        inputs,
        input_validate_ns=0,
        input_validate_start=None,
        input_validate_end=None,
    ):
        return self._infer_validated(
            inputs, input_validate_ns, input_validate_start, input_validate_end
        )

    def infer_validated_rsu(self, inputs):
        """Explicit opt-in: full batch-2 compute, pinned RSU-only host transfer."""
        setup_ns = self._prepare_rsu_buffers()
        result = self._infer_validated(inputs, rsu_only=True)
        return dict(
            result,
            durations_ns=dict(result["durations_ns"], rsu_buffer_setup=setup_ns),
        )

    def _infer_validated(
        self, inputs, input_validate_ns=0, input_validate_start=None,
        input_validate_end=None, rsu_only=False,
    ):
        torch = self._torch
        timestamps = {}
        if input_validate_start is not None:
            timestamps["input_validate_start"] = int(input_validate_start)
        if input_validate_end is not None:
            timestamps["input_validate_end"] = int(input_validate_end)

        timestamps["h2d_start"] = monotonic_raw_ns()
        for name, value in inputs.items():
            self._input_tensors[name].copy_(torch.from_numpy(value), non_blocking=False)
        torch.cuda.synchronize()
        timestamps["h2d_end"] = monotonic_raw_ns()

        timestamps["inference_start"] = monotonic_raw_ns()
        self._start_event.record()
        if not self._context.execute_v2(self._bindings):
            raise RuntimeError("TensorRT execute_v2 returned false")
        self._end_event.record()
        self._end_event.synchronize()
        timestamps["inference_end"] = monotonic_raw_ns()
        inference_ns = int(
            round(self._start_event.elapsed_time(self._end_event) * 1_000_000.0)
        )

        timestamps["d2h_start"] = monotonic_raw_ns()
        if rsu_only:
            for name, tensor in self._output_tensors.items():
                self._rsu_host_outputs[name].copy_(tensor[1:2].detach(), non_blocking=True)
            torch.cuda.synchronize()
            # Exactly one host snapshot copy: pinned array -> immutable bytes.
            outputs = {
                name: np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
                for name, value in self._rsu_host_output_arrays.items()
            }
        else:
            outputs = {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in self._output_tensors.items()
            }
        torch.cuda.synchronize()
        timestamps["d2h_end"] = monotonic_raw_ns()
        self.inference_count += 1

        return {
            "outputs": outputs,
            "model": self.public_model_metadata(),
            "durations_ns": {
                "input_validate": int(input_validate_ns),
                "h2d": timestamps["h2d_end"] - timestamps["h2d_start"],
                "gpu_inference": inference_ns,
                "d2h": timestamps["d2h_end"] - timestamps["d2h_start"],
            },
            "edge_timestamps_ns": timestamps,
        }
