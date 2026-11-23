from __future__ import print_function

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ditl_trt_backend import TensorRtBackend, validate_input_arrays

try:
    import torch
except ImportError:
    torch = None


class _TrackedCuda:
    def __init__(self):
        self.events = []

    def Event(self, **kwargs):
        event = torch.cuda.Event(**kwargs)
        self.events.append(event)
        return event

    def synchronize(self):
        torch.cuda.synchronize()


class _TrackedTorch:
    """Keep real torch allocations/kernels; only observe CUDA resource reuse."""

    def __init__(self):
        self.cuda = _TrackedCuda()
        self.tensors = {}
        self.allocations = []
        self._tensor_to = torch.Tensor.to

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        tensor = torch.empty(*args, **kwargs)
        self.allocations.append(tensor)
        self.tensors[tensor.data_ptr()] = tensor
        return tensor

    def track_to(self, tensor, *args, **kwargs):
        result = self._tensor_to(tensor, *args, **kwargs)
        self.tensors[result.data_ptr()] = result
        return result


class _TensorRtContext:
    """Stand in for unavailable TensorRT, computing two real CUDA outputs."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.binding_history = []
        self.succeed = True

    def execute_v2(self, bindings):
        self.binding_history.append(tuple(bindings))
        if not self.succeed:
            return False
        total, left, difference, right = (
            self.runtime.tensors[pointer] for pointer in bindings
        )
        total.copy_(left + right)
        difference.copy_(left - right)
        return True


class _TensorRtEngine:
    # Interleave input/output bindings to exercise real pointer binding logic.
    names = ("sum", "left", "difference", "right")
    num_bindings = 4

    def __init__(self, context):
        self.context = context

    def get_binding_name(self, index):
        return self.names[index]

    def get_binding_shape(self, index):
        return (2, 3)

    def get_binding_dtype(self, index):
        return np.float32

    def binding_is_input(self, index):
        return index in (1, 3)

    def create_execution_context(self):
        return self.context


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA torch")
class DitlTensorRtBackendCudaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.engine_path = os.path.join(self.temporary.name, "static.engine")
        from pathlib import Path

        Path(self.engine_path).write_bytes(b"test engine; not a real TensorRT artifact")
        self.runtime = _TrackedTorch()
        self.context = _TensorRtContext(self.runtime)
        self.engine = _TensorRtEngine(self.context)
        logger = type("Logger", (), {"ERROR": 1, "__init__": lambda self, level: None})
        self.trt = SimpleNamespace(
            Logger=logger,
            Runtime=lambda logger: SimpleNamespace(
                deserialize_cuda_engine=lambda data: self.engine
            ),
            nptype=lambda dtype: dtype,
        )
        loader = mock.patch(
            "ditl_trt_backend._load_gpu_modules", return_value=(self.runtime, self.trt)
        )
        loader.start()
        self.addCleanup(loader.stop)
        tracker = self.runtime.track_to
        tensor_to = mock.patch.object(
            torch.Tensor, "to", lambda tensor, *a, **kw: tracker(tensor, *a, **kw)
        )
        tensor_to.start()
        self.addCleanup(tensor_to.stop)
        self.backend = TensorRtBackend(self.engine_path)

    def inputs(self, left=1.0, right=4.0):
        return {
            "left": np.full((2, 3), left, dtype=np.float32),
            "right": np.full((2, 3), right, dtype=np.float32),
        }

    def test_reuses_static_cuda_binding_addresses_across_serial_calls(self):
        # Allocating fresh buffers would change pointers and grow allocations.
        self.backend.infer(self.inputs())
        allocations_after_first = len(self.runtime.allocations)
        self.backend.infer(self.inputs(left=9.0))

        self.assertEqual(self.context.binding_history[0], self.context.binding_history[1])
        self.assertEqual(4, len(set(self.context.binding_history[0])))
        self.assertEqual(allocations_after_first, len(self.runtime.allocations))
        self.assertEqual(2, self.backend.inference_count)

    def test_reuses_cuda_timing_events_across_serial_calls(self):
        # Recreating CUDA events in infer would grow the event list.
        self.backend.infer(self.inputs())
        self.backend.infer(self.inputs())

        self.assertEqual(2, len(self.runtime.cuda.events))

    def test_distinct_inputs_produce_independent_output_snapshots(self):
        # Skipping input copies or returning reused output views breaks these values.
        arrays = self.inputs()
        arrays["left"] = np.arange(6, dtype=np.float32).reshape(3, 2).T
        originals = {name: value.copy() for name, value in arrays.items()}
        first = self.backend.infer(arrays)
        second = self.backend.infer(self.inputs(left=9.0, right=2.0))

        np.testing.assert_array_equal(first["outputs"]["sum"], [[4, 6, 8], [5, 7, 9]])
        np.testing.assert_array_equal(first["outputs"]["difference"], [[-4, -2, 0], [-3, -1, 1]])
        np.testing.assert_array_equal(second["outputs"]["sum"], np.full((2, 3), 11))
        np.testing.assert_array_equal(second["outputs"]["difference"], np.full((2, 3), 7))
        for name, original in originals.items():
            np.testing.assert_array_equal(arrays[name], original)
        self.assertFalse(np.shares_memory(first["outputs"]["sum"], second["outputs"]["sum"]))
        first["outputs"]["sum"].fill(123)
        np.testing.assert_array_equal(second["outputs"]["sum"], np.full((2, 3), 11))

    def test_stage_end_timestamps_follow_completed_cuda_work(self):
        # Ending wall-clock stages before synchronization observes pending CUDA work.
        from ditl_edge_runtime import monotonic_raw_ns

        completion = []

        def clock():
            completion.append(torch.cuda.current_stream().query())
            return monotonic_raw_ns()

        with mock.patch("ditl_trt_backend.monotonic_raw_ns", side_effect=clock):
            result = self.backend.infer(self.inputs())

        self.assertTrue(all(completion))
        timestamps = result["edge_timestamps_ns"]
        for stage in ("input_validate", "h2d", "inference", "d2h"):
            self.assertLessEqual(timestamps[stage + "_start"], timestamps[stage + "_end"])
        for stage in ("h2d", "d2h"):
            self.assertEqual(
                timestamps[stage + "_end"] - timestamps[stage + "_start"],
                result["durations_ns"][stage],
            )
        self.assertGreaterEqual(result["durations_ns"]["gpu_inference"], 0)

    def test_failed_execution_does_not_count_successful_inference(self):
        self.context.succeed = False
        with self.assertRaisesRegex(RuntimeError, "execute_v2"):
            self.backend.infer(self.inputs())
        self.assertEqual(0, self.backend.inference_count)


class DitlTensorRtBackendContractTests(unittest.TestCase):
    def test_input_contract_accepts_exact_names_shapes_and_dtypes(self):
        contract = {
            "spatial_features": {"shape": (2, 64, 128, 256), "dtype": "float32"},
            "t_ego": {"shape": (2, 2, 3), "dtype": "float32"},
        }
        arrays = {
            "spatial_features": np.zeros((2, 64, 128, 256), dtype=np.float32),
            "t_ego": np.zeros((2, 2, 3), dtype=np.float32),
        }

        validated = validate_input_arrays(contract, arrays)

        self.assertEqual(set(contract), set(validated))
        self.assertTrue(validated["spatial_features"].flags.c_contiguous)
        self.assertIsNot(arrays["spatial_features"], validated["spatial_features"])

    def test_input_contract_rejects_extra_name_wrong_shape_and_wrong_dtype(self):
        contract = {"feature": {"shape": (2, 3), "dtype": "float32"}}
        with self.assertRaisesRegex(ValueError, "names"):
            validate_input_arrays(
                contract,
                {
                    "feature": np.zeros((2, 3), dtype=np.float32),
                    "engine_path": np.zeros((1,), dtype=np.float32),
                },
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_input_arrays(
                contract, {"feature": np.zeros((3, 2), dtype=np.float32)}
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_input_arrays(
                contract, {"feature": np.zeros((2, 3), dtype=np.float16)}
            )

    def test_backend_rejects_missing_engine_before_loading_gpu_modules(self):
        path = os.path.join(tempfile.gettempdir(), "missing-ditl.engine")
        if os.path.exists(path):
            os.unlink(path)
        with self.assertRaisesRegex(FileNotFoundError, "TensorRT engine"):
            TensorRtBackend(path)

    def test_public_model_metadata_does_not_expose_engine_path(self):
        backend = object.__new__(TensorRtBackend)
        backend.model_id = "rsu-edge"
        backend.engine_sha256 = "a" * 64
        backend.input_contract = {"spatial_features": {"shape": (1,), "dtype": "float32"}}
        backend.output_contract = {"cls_preds": {"shape": (1,), "dtype": "float32"}}

        metadata = backend.public_model_metadata()

        self.assertEqual("tensorrt", metadata["backend"])
        self.assertEqual("rsu-edge", metadata["model_id"])
        self.assertEqual("a" * 64, metadata["engine_sha256"])
        self.assertNotIn("engine_path", metadata)


if __name__ == "__main__":
    unittest.main()
