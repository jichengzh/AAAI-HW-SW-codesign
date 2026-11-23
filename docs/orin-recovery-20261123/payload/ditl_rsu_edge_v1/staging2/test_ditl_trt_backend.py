from __future__ import print_function

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ditl_trt_backend import TensorRtBackend, validate_input_arrays


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
