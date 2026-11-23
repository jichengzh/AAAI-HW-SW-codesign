import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import coptv2x_hil_backend as hil
import test_coptv2x_hil_backend as hil_fixtures
import test_ditl_trt_backend as trt_fixtures

torch = trt_fixtures.torch


class ImmutableMemoryviewOwnerTests(unittest.TestCase):
    def test_readonly_memoryview_of_bytes_is_an_immutable_snapshot_owner(self):
        # NumPy can terminate its base chain at memoryview rather than bytes.
        payload = np.arange(6, dtype=np.float32).tobytes()
        snapshot = np.frombuffer(memoryview(payload), dtype=np.float32).reshape(2, 3)
        self.assertFalse(snapshot.flags.writeable)
        self.assertTrue(hil._immutable_bytes_owner(snapshot))
        self.assertTrue(hil._immutable_bytes_owner(memoryview(snapshot)))
        self.assertTrue(hil._immutable_bytes_owner(memoryview(payload)))
        with self.assertRaises(ValueError):
            snapshot.setflags(write=True)

    def test_readonly_memoryview_of_mutable_owner_is_not_reusable(self):
        def readonly_view(value):
            view = memoryview(value)
            if hasattr(view, "toreadonly"):
                return view.toreadonly()
            # Python 3.7 lacks toreadonly; preserve the real mutable base chain.
            array = np.frombuffer(view, dtype=np.uint8)
            array.setflags(write=False)
            return memoryview(array)

        payload = np.arange(6, dtype=np.float32).tobytes()
        mutable_bytes = bytearray(payload)
        readonly_bytes = readonly_view(mutable_bytes)
        self.assertTrue(readonly_bytes.readonly)
        snapshot = np.frombuffer(readonly_bytes, dtype=np.float32).reshape(2, 3)
        self.assertFalse(hil._immutable_bytes_owner(readonly_bytes))
        self.assertFalse(hil._immutable_bytes_owner(snapshot))
        mutable_array = np.arange(6, dtype=np.float32)
        readonly_array = readonly_view(mutable_array)
        self.assertTrue(readonly_array.readonly)
        self.assertFalse(hil._immutable_bytes_owner(readonly_array))
        self.assertFalse(hil._immutable_bytes_owner(np.frombuffer(readonly_array, dtype=np.float32)))
        mutable_array.setflags(write=False)
        # Even a readonly owning ndarray can be thawed; it is not immutable bytes.
        self.assertFalse(hil._immutable_bytes_owner(memoryview(mutable_array)))
        self.assertFalse(hil._immutable_bytes_owner(memoryview(bytearray(payload))))


class RsuTensorRtCudaTests(trt_fixtures.DitlTensorRtBackendCudaTests):
    def test_rsu_capability_copies_only_batch_one_and_freezes_snapshots(self):
        # Full-output D2H, wrong slice, or reused host views must fail.
        capability = getattr(self.backend, "infer_validated_rsu", None)
        self.assertTrue(callable(capability), "missing RSU-only runtime capability")
        transfers = []
        h2d_transfers = []
        original_copy = torch.Tensor.copy_

        def copy(destination, source, **kwargs):
            if destination.device.type == "cuda" and source.device.type == "cpu":
                h2d_transfers.append(tuple(source.shape))
                self.assertFalse(source.is_pinned())
                self.assertFalse(kwargs.get("non_blocking"))
            if destination.device.type == "cpu" and source.device.type == "cuda":
                transfers.append(tuple(source.shape))
                self.assertTrue(destination.is_pinned())
            return original_copy(destination, source, **kwargs)

        inputs = {
            "left": np.asarray([[1, 2, 3], [9, 10, 11]], dtype=np.float32),
            "right": np.asarray([[4, 5, 6], [2, 3, 4]], dtype=np.float32),
        }
        with mock.patch.object(torch.Tensor, "copy_", copy):
            first = capability(inputs)
            allocations = len(self.runtime.allocations)
            pointers = tuple(t.data_ptr() for t in self.runtime.allocations)
            second = capability(self.inputs(left=20, right=1))
        self.assertEqual([(1, 3)] * 4, transfers)
        self.assertEqual([(2, 3)] * 4, h2d_transfers)
        self.assertEqual(allocations, len(self.runtime.allocations))
        self.assertEqual(pointers, tuple(t.data_ptr() for t in self.runtime.allocations))
        np.testing.assert_array_equal(first["outputs"]["sum"], [[11, 13, 15]])
        np.testing.assert_array_equal(first["outputs"]["difference"], [[7, 7, 7]])
        np.testing.assert_array_equal(second["outputs"]["sum"], [[21, 21, 21]])
        self.assertEqual((2, 3), self.backend.output_contract["sum"]["shape"])
        self.assertGreaterEqual(self.backend.rsu_buffer_setup_ns, 0)
        self.assertEqual(self.backend.rsu_buffer_setup_ns, first["durations_ns"].get("rsu_buffer_setup"))
        self.assertEqual(0, second["durations_ns"].get("rsu_buffer_setup"))
        for output in first["outputs"].values():
            self.assertEqual(np.dtype("float32"), output.dtype)
            self.assertTrue(hil._immutable_bytes_owner(output))
            with self.assertRaises(ValueError):
                output.setflags(write=True)
        pinned = [tensor for tensor in self.runtime.allocations if tensor.device.type == "cpu"]
        self.assertEqual(2, len(pinned))
        self.assertTrue(all(tensor.is_pinned() for tensor in pinned))
        np.testing.assert_array_equal(inputs["left"], [[1, 2, 3], [9, 10, 11]])
        legacy = self.backend.infer_validated(self.inputs())
        self.assertEqual((2, 3), legacy["outputs"]["sum"].shape)

    def test_rsu_capability_rejects_non_batch_two_binding_authority(self):
        capability = getattr(self.backend, "infer_validated_rsu", None)
        self.assertTrue(callable(capability), "missing RSU-only runtime capability")
        self.backend.output_contract["sum"] = {"shape": (3, 3), "dtype": "float32"}
        with self.assertRaisesRegex(ValueError, "batch.*2"):
            capability(self.inputs())


class BoundedFiniteTests(unittest.TestCase):
    def checker(self):
        checker = getattr(hil, "_all_finite", None)
        self.assertTrue(callable(checker), "missing bounded finite checker")
        return checker

    def test_finite_scan_bounds_ufunc_temporary_size(self):
        # A whole-array boolean temporary violates the bounded-memory contract.
        checker = self.checker()
        value = np.ones(2 * 262144 + 3, dtype=np.float32)
        original = np.isfinite
        scanned = []

        def finite(chunk):
            scanned.append(chunk.size)
            return original(chunk)

        with mock.patch.object(hil.np, "isfinite", finite):
            self.assertTrue(checker(value))
        self.assertEqual(value.size, sum(scanned))
        self.assertLessEqual(max(scanned), 262144)
        self.assertGreater(len(scanned), 1)

    def test_nan_and_infinity_are_rejected_at_chunk_boundaries_and_tail(self):
        # Missing the next chunk or final partial chunk must be observable.
        checker = self.checker()
        for index in (262143, 262144, 524287, 524288, 524290):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(index=index, invalid=invalid):
                    value = np.ones(2 * 262144 + 3, dtype=np.float32)
                    value[index] = invalid
                    self.assertFalse(checker(value))

    def test_noncontiguous_empty_scalar_and_integer_match_numpy_reference(self):
        checker = self.checker()
        value = np.arange(1048584, dtype=np.float32).reshape(2, 524292)
        noncontiguous = value.T[::-1, ::-1]
        self.assertFalse(noncontiguous.flags.c_contiguous)
        fixtures = [noncontiguous, np.empty((0, 3)), np.asarray(1.0), np.arange(7, dtype=np.int64)]
        broken = noncontiguous.copy(order="F")
        broken[-1, -1] = np.nan
        fixtures.append(broken)
        for fixture in fixtures:
            with self.subTest(shape=fixture.shape):
                self.assertEqual(bool(np.isfinite(fixture).all()), checker(fixture))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA torch")
class NativeRsuCudaTests(unittest.TestCase):
    def test_native_rsu_slices_before_host_transfer_with_batch_two_computation(self):
        runtime = object.__new__(hil.NativePyTorchBackend)
        runtime._torch = torch
        runtime.model_id = "frozen-native"
        runtime.checkpoint_sha256 = "a" * 64
        runtime.input_contract = {"spatial_features": {"shape": (2, 3), "dtype": "float32"}}
        runtime.output_contract = {name: {"shape": (2, 3), "dtype": "float32"} for name in ("res0", "res1", "res2")}
        runtime.inference_count = 0
        computed = []

        def model(value):
            computed.append(tuple(value.shape))
            return value + 1, value + 2, value + 3

        runtime._callable = model
        capability = getattr(runtime, "infer_validated_rsu", None)
        self.assertTrue(callable(capability), "missing native RSU-only capability")
        transferred = []
        original_cpu = torch.Tensor.cpu

        def cpu(value, *args, **kwargs):
            transferred.append(tuple(value.shape))
            return original_cpu(value, *args, **kwargs)

        inputs = {"spatial_features": np.asarray([[1, 2, 3], [9, 10, 11]], dtype=np.float32)}
        with mock.patch.object(torch.Tensor, "cpu", cpu):
            first = capability(inputs)
            second = capability({"spatial_features": np.full((2, 3), 20, dtype=np.float32)})
        self.assertEqual([(2, 3), (2, 3)], computed)
        self.assertEqual([(1, 3)] * 6, transferred)
        np.testing.assert_array_equal(first["outputs"]["res0"], [[10, 11, 12]])
        np.testing.assert_array_equal(second["outputs"]["res0"], [[21, 21, 21]])
        self.assertTrue(hil._immutable_bytes_owner(first["outputs"]["res0"]))


class HilRsuCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.tree = hil_fixtures.ArtifactTree()
        self.addCleanup(self.tree.close)

    def backend(self, batch=1, capability=True, rsu_only=True):
        def factory(spec, paths, input_contract, output_contract, native_repo_root):
            runtime = hil_fixtures.FakeRuntime(spec, input_contract, output_contract)
            if capability:
                def rsu(inputs):
                    runtime.last_outputs = {
                        name: hil._frozen_numpy_snapshot(np.full((batch,) + tuple(contract["shape"])[1:], 7, dtype=np.float32))
                        for name, contract in output_contract.items()
                    }
                    return {"outputs": runtime.last_outputs, "model": {"backend": spec.runtime}, "durations_ns": {"rsu_buffer_setup": 17}}
                runtime.infer_validated_rsu = rsu
            return runtime
        return hil.CoptV2xHilBackend(
            "joint_shcosearch", self.tree.artifact_root,
            config_dir=self.tree.config_dir, runtime_factory=factory,
            native_repo_root=self.tree.root, rsu_only=rsu_only,
        )

    def infer(self, backend):
        inputs = np.zeros(backend.input_contract["spatial_features"]["shape"], dtype=np.float32)
        return backend.infer_hil_validated({"spatial_features": inputs}, hil_fixtures._trigger("joint_shcosearch"), hil.CODRIVING_BATCH_ORDER)

    def test_hil_rsu_uses_capability_without_reprojecting_or_copying_snapshot(self):
        backend = self.backend()
        result = self.infer(backend)
        self.assertEqual(hil.COPTV2X_RSU_PAYLOAD_SEMANTICS, result.payload_semantics)
        self.assertEqual(17, result.durations_ns.get("rsu_buffer_setup"))
        for name, value in result.outputs.items():
            self.assertEqual(1, value.shape[0])
            self.assertEqual(7, float(value.flat[0]))
            self.assertIs(value, backend._runtime.last_outputs[name])
        hil.validate_feature_payload_contract(dict(result.artifact_identity, widths=list(result.artifact_identity["widths"]), artifact_sha256=dict(result.artifact_identity["artifact_sha256"])), result.outputs, self.tree.config_dir, result.payload_semantics)

    def test_hil_rejects_capability_returning_full_batch_two(self):
        backend = self.backend(batch=2)
        with self.assertRaisesRegex(hil.HilBackendError, "shape"):
            self.infer(backend)

    def test_hil_legacy_rsu_fallback_and_default_batch_two_are_preserved(self):
        legacy = self.backend(capability=False)
        self.assertEqual(1, self.infer(legacy).outputs["res0"].shape[0])
        default = self.backend(rsu_only=False)
        self.assertEqual(2, self.infer(default).outputs["res0"].shape[0])


if __name__ == "__main__":
    unittest.main()
