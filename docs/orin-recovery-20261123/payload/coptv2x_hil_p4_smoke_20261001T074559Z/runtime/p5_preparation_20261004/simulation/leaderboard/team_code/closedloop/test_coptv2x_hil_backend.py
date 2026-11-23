from __future__ import print_function

import copy
import gc
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml


sys.path.insert(0, os.path.dirname(__file__))

from coptv2x_hil_artifacts import (
    DEFAULT_CONFIG_DIR,
    HIL_ARM_IDS,
    load_hil_artifact_registry,
)
from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger
from coptv2x_hil_backend import (
    CODRIVING_BATCH_ORDER,
    COPTV2X_PAYLOAD_SEMANTICS,
    CoptV2xHilBackend,
    HilBackendError,
    hash_feature_payload,
)
import coptv2x_hil_backend as backend_module
from ditl_rsu_pipeline import CoptV2xHilPipeline, RsuEdgePipeline
from ditl_rsu_source import LocalRsuFrame


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trigger(arm_id, sequence_no=7):
    return parse_hil_trigger(
        {
            "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
            "run_id": "run-task3",
            "route_id": "Town05-route-01",
            "scenario_id": "scenario-01",
            "sample_id": "sample-000",
            "sequence_no": sequence_no,
            "source_frame": 42,
            "source_sim_time_ns": 123456789,
            "arm_id": arm_id,
        }
    )


class ArtifactTree(object):
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_dir = self.root / "config"
        self.artifact_root = self.root / "artifacts"
        self.config_dir.mkdir()
        self.artifact_root.mkdir()
        for arm_id in HIL_ARM_IDS:
            source = DEFAULT_CONFIG_DIR / ("artifact_%s.yaml" % arm_id)
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
            for artifact_name in ("config", "checkpoint", "engine"):
                relative = document[artifact_name]
                if relative is None:
                    continue
                path = self.artifact_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(("%s\n" % relative).encode("ascii"))
                document["artifact_sha256"][artifact_name] = _sha256(path)
            target = self.config_dir / source.name
            target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def close(self):
        self.temporary.cleanup()


class FakeRuntime(object):
    def __init__(self, spec, input_contract, output_contract, finite=True):
        self.spec = spec
        self.input_contract = copy.deepcopy(input_contract)
        self.output_contract = copy.deepcopy(output_contract)
        self.finite = finite
        self.inference_count = 0

    def infer_validated(self, inputs, **unused):
        from ditl_edge_runtime import monotonic_raw_ns

        h2d_start = monotonic_raw_ns()
        h2d_end = monotonic_raw_ns()
        inference_start = monotonic_raw_ns()
        inference_end = monotonic_raw_ns()
        d2h_start = monotonic_raw_ns()
        outputs = {
            name: np.zeros(tuple(contract["shape"]), dtype=np.dtype(contract["dtype"]))
            for name, contract in self.output_contract.items()
        }
        if not self.finite:
            outputs["res0"].flat[0] = np.nan
        d2h_end = monotonic_raw_ns()
        self.inference_count += 1
        return {
            "outputs": outputs,
            "model": {"backend": self.spec.runtime, "model_id": self.spec.arm_id},
            "durations_ns": {
                "h2d": h2d_end - h2d_start,
                "gpu_inference": inference_end - inference_start,
                "d2h": d2h_end - d2h_start,
            },
            "edge_timestamps_ns": {
                "h2d_start": h2d_start,
                "h2d_end": h2d_end,
                "inference_start": inference_start,
                "inference_end": inference_end,
                "d2h_start": d2h_start,
                "d2h_end": d2h_end,
            },
        }


class RuntimeFactory(object):
    def __init__(self, finite=True, alter_contract=None):
        self.finite = finite
        self.alter_contract = alter_contract
        self.calls = []

    def __call__(self, spec, paths, input_contract, output_contract, native_repo_root):
        self.calls.append((spec.arm_id, dict(paths)))
        runtime = FakeRuntime(spec, input_contract, output_contract, finite=self.finite)
        if self.alter_contract is not None:
            runtime.output_contract = self.alter_contract(runtime.output_contract)
        return runtime


class FakeTorchTensor(object):
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, **unused):
        return self

    def contiguous(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self.value = self.value.astype(np.float32)
        return self

    def numpy(self):
        return self.value


class FakeTorch(object):
    float32 = "float32"

    class Event(object):
        def __init__(self, enable_timing):
            self.enable_timing = enable_timing

        def record(self):
            return None

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            return 1.25

    class Cuda(object):
        Event = None

        @staticmethod
        def synchronize():
            return None

    class InferenceMode(object):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    cuda = Cuda()
    cuda.Event = Event

    @staticmethod
    def from_numpy(value):
        return FakeTorchTensor(value)

    @staticmethod
    def inference_mode():
        return FakeTorch.InferenceMode()


class CoptV2xHilBackendTests(unittest.TestCase):
    def setUp(self):
        self.tree = ArtifactTree()

    def tearDown(self):
        self.tree.close()

    def backend(self, arm_id, factory=None):
        return CoptV2xHilBackend(
            arm_id=arm_id,
            artifact_root=self.tree.artifact_root,
            config_dir=self.tree.config_dir,
            runtime_factory=factory or RuntimeFactory(),
            native_repo_root=self.root_or_none(),
        )

    def root_or_none(self):
        return self.tree.root

    def test_all_five_arms_load_once_and_emit_exact_finite_multiscale_contract(self):
        spatial = np.zeros((2, 64, 256, 512), dtype=np.float32)
        for arm_id in HIL_ARM_IDS:
            with self.subTest(arm_id=arm_id):
                factory = RuntimeFactory()
                backend = self.backend(arm_id, factory)

                first_runtime = backend.load(arm_id)
                self.assertIs(first_runtime, backend.load(arm_id))
                self.assertEqual(1, len(factory.calls))
                result = backend.infer(
                    spatial,
                    _trigger(arm_id),
                    batch_order=("ego", "rsu"),
                    local_source_timestamps_ns={
                        "local_source_start": 10,
                        "local_source_end": 11,
                    },
                    local_source_durations_ns={"local_source": 1},
                )

                expected = {
                    tensor.name: (tensor.shape, tensor.dtype)
                    for tensor in backend.spec.output_contract
                }
                self.assertEqual({"res0", "res1", "res2"}, set(result.outputs))
                for name, value in result.outputs.items():
                    self.assertEqual(expected[name][0], value.shape)
                    self.assertEqual(np.dtype(expected[name][1]), value.dtype)
                    self.assertTrue(np.isfinite(value).all())
                    self.assertFalse(value.flags.writeable)
                self.assertEqual(COPTV2X_PAYLOAD_SEMANTICS, result.payload_semantics)
                self.assertEqual(0, result.remote_input_tensor_bytes)
                self.assertEqual(_trigger(arm_id).to_metadata(), dict(result.trigger_identity))
                self.assertEqual(arm_id, result.artifact_identity["arm_id"])
                self.assertEqual(
                    backend.spec.artifact_sha256,
                    result.artifact_identity["artifact_sha256"],
                )
                self.assertEqual(hash_feature_payload(result.outputs), result.payload_sha256)
                for stage in ("local_source", "preprocess", "h2d", "inference", "d2h", "pack"):
                    self.assertIn(stage, result.durations_ns)
                    self.assertGreaterEqual(result.durations_ns[stage], 0)
                del result, backend, first_runtime
                gc.collect()

    def test_input_contract_and_batch_identity_are_fail_closed(self):
        backend = self.backend("joint_shcosearch")
        exact = np.zeros((2, 64, 256, 512), dtype=np.float32)
        with self.assertRaisesRegex(HilBackendError, "batch order"):
            backend.infer(exact, _trigger("joint_shcosearch"), batch_order=("rsu", "ego"))
        for invalid in (
            np.zeros((1, 64, 256, 512), dtype=np.float32),
            np.zeros((2, 64, 256, 512), dtype=np.float16),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(HilBackendError):
                    backend.infer(
                        invalid,
                        _trigger("joint_shcosearch"),
                        batch_order=CODRIVING_BATCH_ORDER,
                    )

    def test_input_contract_accepts_immutable_indexed_source_mapping(self):
        backend = self.backend("joint_shcosearch")
        exact = np.zeros((2, 64, 256, 512), dtype=np.float32)
        validated = backend.validate_inputs(
            MappingProxyType({"spatial_features": exact})
        )
        self.assertEqual({"spatial_features"}, set(validated))
        self.assertEqual(exact.shape, validated["spatial_features"].shape)
        self.assertTrue(np.array_equal(exact, validated["spatial_features"]))

    def test_wrong_engine_hash_cross_arm_and_binding_drift_are_rejected(self):
        engine = self.tree.artifact_root / "codriving/compression_only/backbone.engine"
        engine.write_bytes(b"tampered")
        factory = RuntimeFactory()
        backend = self.backend("compression_only", factory)
        with self.assertRaisesRegex(HilBackendError, "SHA-256"):
            backend.load()
        self.assertEqual([], factory.calls)

        fixed = ArtifactTree()
        try:
            backend = CoptV2xHilBackend(
                arm_id="compression_only",
                artifact_root=fixed.artifact_root,
                config_dir=fixed.config_dir,
                runtime_factory=RuntimeFactory(),
                native_repo_root=fixed.root,
            )
            with self.assertRaisesRegex(HilBackendError, "run arm"):
                backend.load("schedule_only")

            def remove_res2(contract):
                return {name: value for name, value in contract.items() if name != "res2"}

            drift = CoptV2xHilBackend(
                arm_id="schedule_only",
                artifact_root=fixed.artifact_root,
                config_dir=fixed.config_dir,
                runtime_factory=RuntimeFactory(alter_contract=remove_res2),
                native_repo_root=fixed.root,
            )
            with self.assertRaisesRegex(HilBackendError, "output binding contract"):
                drift.load()
        finally:
            fixed.close()

    def test_trigger_arm_nonfinite_output_and_payload_tampering_are_rejected(self):
        spatial = np.zeros((2, 64, 256, 512), dtype=np.float32)
        backend = self.backend("schedule_only")
        with self.assertRaisesRegex(HilBackendError, "trigger arm"):
            backend.infer(
                spatial,
                _trigger("compression_only"),
                batch_order=CODRIVING_BATCH_ORDER,
            )

        backend = self.backend("schedule_only", RuntimeFactory(finite=False))
        with self.assertRaisesRegex(HilBackendError, "finite"):
            backend.infer(
                spatial,
                _trigger("schedule_only"),
                batch_order=CODRIVING_BATCH_ORDER,
            )

        outputs = {
            "res0": np.zeros((1,), dtype=np.float32),
            "res1": np.zeros((1,), dtype=np.float32),
            "res2": np.zeros((1,), dtype=np.float32),
        }
        digest = hash_feature_payload(outputs)
        outputs["res1"][0] = 1.0
        self.assertNotEqual(digest, hash_feature_payload(outputs))

    def test_default_factory_selects_only_the_frozen_runtime(self):
        from coptv2x_hil_artifacts import load_hil_artifact_registry

        registry = load_hil_artifact_registry(self.tree.config_dir)
        input_contract = {"spatial_features": {"shape": (2, 64, 256, 512), "dtype": "float32"}}
        trt_spec = registry["compression_only"]
        trt_outputs = {
            tensor.name: {"shape": tensor.shape, "dtype": tensor.dtype}
            for tensor in trt_spec.output_contract
        }
        paths = {"engine": self.tree.artifact_root / trt_spec.engine}
        sentinel = object()
        with mock.patch.object(backend_module, "TensorRtBackend", return_value=sentinel) as constructor:
            actual = backend_module._default_runtime_factory(
                trt_spec, paths, input_contract, trt_outputs, self.tree.root
            )
        self.assertIs(sentinel, actual)
        self.assertEqual(trt_spec.artifact_sha256["engine"], constructor.call_args.kwargs["expected_engine_sha256"])

        native_spec = registry["original_default"]
        native_outputs = {
            tensor.name: {"shape": tensor.shape, "dtype": tensor.dtype}
            for tensor in native_spec.output_contract
        }
        native_paths = {
            "config": self.tree.artifact_root / native_spec.config,
            "checkpoint": self.tree.artifact_root / native_spec.checkpoint,
        }
        with mock.patch.object(backend_module, "NativePyTorchBackend", return_value=sentinel) as constructor:
            actual = backend_module._default_runtime_factory(
                native_spec, native_paths, input_contract, native_outputs, self.tree.root
            )
        self.assertIs(sentinel, actual)
        self.assertEqual(self.tree.root, constructor.call_args.kwargs["repo_root"])

        unsupported = copy.copy(native_spec)
        object.__setattr__(unsupported, "runtime", "other")
        with self.assertRaisesRegex(HilBackendError, "unsupported"):
            backend_module._default_runtime_factory(
                unsupported, native_paths, input_contract, native_outputs, self.tree.root
            )

    def test_unknown_arm_missing_artifact_and_invalid_payload_names_fail_closed(self):
        with self.assertRaisesRegex(HilBackendError, "unknown"):
            self.backend("unknown-arm")
        missing = self.backend("joint_shcosearch")
        (self.tree.artifact_root / missing.spec.engine).unlink()
        with self.assertRaisesRegex(HilBackendError, "missing"):
            missing.load()
        with self.assertRaisesRegex(HilBackendError, "res0/res1/res2"):
            hash_feature_payload({"res0": np.zeros((1,), dtype=np.float32)})

    def test_native_runtime_reports_separate_h2d_inference_and_d2h(self):
        runtime = object.__new__(backend_module.NativePyTorchBackend)
        runtime._torch = FakeTorch
        runtime._callable = lambda tensor: (
            FakeTorchTensor(np.zeros((2, 1, 2, 2), dtype=np.float32)),
            FakeTorchTensor(np.zeros((2, 2, 1, 1), dtype=np.float32)),
            FakeTorchTensor(np.zeros((2, 4, 1, 1), dtype=np.float32)),
        )
        runtime.model_id = "codriving:original_default"
        runtime.checkpoint_sha256 = "a" * 64
        runtime.input_contract = {
            "spatial_features": {"shape": (2, 64, 256, 512), "dtype": "float32"}
        }
        runtime.output_contract = {
            "res0": {"shape": (2, 1, 2, 2), "dtype": "float32"},
            "res1": {"shape": (2, 2, 1, 1), "dtype": "float32"},
            "res2": {"shape": (2, 4, 1, 1), "dtype": "float32"},
        }
        runtime.inference_count = 0

        result = runtime.infer_validated(
            {"spatial_features": np.zeros((2, 64, 256, 512), dtype=np.float32)}
        )

        self.assertEqual({"res0", "res1", "res2"}, set(result["outputs"]))
        self.assertEqual(1250000, result["durations_ns"]["gpu_inference"])
        self.assertEqual(1, runtime.inference_count)
        self.assertEqual("native_pytorch", result["model"]["backend"])


class CoptV2xPipelineContractTests(unittest.TestCase):
    def test_legacy_pipeline_keeps_dense_head_semantics(self):
        backend = object()
        pipeline = RsuEdgePipeline(backend, run_id="legacy")
        self.assertEqual("dense_head_debug", pipeline.payload_semantics)

    def test_copt_pipeline_is_an_explicit_semantics_entry(self):
        backend = type("Backend", (), {"payload_semantics": COPTV2X_PAYLOAD_SEMANTICS})()
        pipeline = CoptV2xHilPipeline(backend, run_id="hil")
        self.assertEqual(COPTV2X_PAYLOAD_SEMANTICS, pipeline.payload_semantics)

    def test_base_pipeline_rejects_a_hil_backend_under_legacy_semantics(self):
        backend = type(
            "HilBackend",
            (),
            {
                "payload_semantics": COPTV2X_PAYLOAD_SEMANTICS,
                "infer_hil_validated": lambda *args, **kwargs: None,
            },
        )()
        with self.assertRaisesRegex(ValueError, "CoptV2X"):
            RsuEdgePipeline(backend, run_id="wrong-entry")

    def test_copt_pipeline_carries_artifact_trigger_and_payload_identity(self):
        from ditl_edge_runtime import monotonic_raw_ns

        tree = ArtifactTree()
        try:
            backend = CoptV2xHilBackend(
                arm_id="joint_shcosearch",
                artifact_root=tree.artifact_root,
                config_dir=tree.config_dir,
                runtime_factory=RuntimeFactory(),
                native_repo_root=tree.root,
            )
            pipeline = CoptV2xHilPipeline(
                backend,
                run_id="run-task3",
                route_id="Town05-route-01",
                scenario_id="scenario-01",
            )
            trigger = _trigger("joint_shcosearch")
            available = monotonic_raw_ns()
            read_start = monotonic_raw_ns()
            read_end = monotonic_raw_ns()
            frame = LocalRsuFrame(
                frame_id=trigger.source_frame,
                sequence_id=trigger.sequence_no,
                data_available_ns=available,
                arrays={
                    "spatial_features": np.zeros(
                        (2, 64, 256, 512), dtype=np.float32
                    )
                },
                source_metadata={
                    "source_type": "local_indexed_feature",
                    "trigger_identity": trigger.to_metadata(),
                    "batch_order": ("ego", "rsu"),
                    "timestamps_ns": {
                        "source_read_start": read_start,
                        "source_read_end": read_end,
                    },
                    "durations_ns": {"source_read": read_end - read_start},
                },
            )

            result = pipeline.process(frame)

            self.assertEqual(COPTV2X_PAYLOAD_SEMANTICS, result.header["payload_semantics"])
            self.assertEqual(trigger.to_metadata(), result.header["trigger_identity"])
            self.assertEqual("joint_shcosearch", result.header["artifact_identity"]["arm_id"])
            self.assertEqual(hash_feature_payload(result.outputs), result.header["payload_sha256"])
            self.assertEqual(0, result.header["remote_input_tensor_bytes"])
            self.assertEqual(["ego", "rsu"], result.header["batch_order"])
            self.assertEqual({"res0", "res1", "res2"}, set(result.outputs))
        finally:
            tree.close()


class CoptV2xRosBridgeTests(unittest.TestCase):
    def test_task3_bridge_smoke_restricts_result_handoff_to_loopback(self):
        from scripts.coptv2x_hil_bridge_smoke import validate_result_handoff

        self.assertEqual(("127.0.0.1", 55143), validate_result_handoff("127.0.0.1", 55143))
        for host, port in (("172.16.62.222", 55143), ("localhost", 0), ("::1", 70000)):
            with self.subTest(host=host, port=port):
                with self.assertRaises(ValueError):
                    validate_result_handoff(host, port)

    def _wire_fixture(self, payload_sha256=None, outputs=None, remote_input_tensor_bytes=0):
        from ditl_edge_protocol import ProtocolLimits, build_wire_bytes

        if outputs is None:
            outputs = {
                "res0": np.zeros((2, 16, 128, 256), dtype=np.float32),
                "res1": np.zeros((2, 32, 64, 128), dtype=np.float32),
                "res2": np.zeros((2, 64, 32, 64), dtype=np.float32),
            }
        trigger = _trigger("joint_shcosearch").to_metadata()
        sender_names = (
            "rsu_data_available", "source_read_start", "source_read_end",
            "preprocess_start", "preprocess_end", "h2d_start", "h2d_end",
            "inference_start", "inference_end", "d2h_start", "d2h_end",
            "postprocess_start", "postprocess_end", "pack_start", "pack_end",
            "hash_start", "hash_end", "compress_start", "compress_end",
            "protocol_serialize_start", "protocol_serialize_end",
        )
        timestamps = {name: index + 1 for index, name in enumerate(sender_names)}
        timestamps["local_handoff_start"] = 25
        spec = load_hil_artifact_registry()["joint_shcosearch"]
        artifact = {
            "arm_id": spec.arm_id,
            "model": spec.model,
            "widths": list(spec.widths),
            "runtime": spec.runtime,
            "precision": spec.precision,
            "config": str(spec.config),
            "checkpoint": str(spec.checkpoint),
            "engine": str(spec.engine),
            "artifact_sha256": dict(spec.artifact_sha256),
        }
        header = {
            "protocol_version": "ditl.edge.v1",
            "kind": "result",
            "request_id": "run-task3:7",
            "status": "ok",
            "run_id": trigger["run_id"],
            "route_id": trigger["route_id"],
            "scenario_id": trigger["scenario_id"],
            "source_id": "rsu-orin-01",
            "target_id": "ego-server-01",
            "frame_id": trigger["source_frame"],
            "sequence_id": trigger["sequence_no"],
            "source_type": "local_indexed_feature",
            "payload_semantics": COPTV2X_PAYLOAD_SEMANTICS,
            "postprocess": "identity",
            "remote_input_tensor_bytes": remote_input_tensor_bytes,
            "edge_clock": {
                "host_id": "edge-host",
                "domain": "monotonic_raw",
                "boot_id": "edge-boot",
                "sync_state": "locked",
                "offset_ns": 0,
                "uncertainty_ns": 0,
            },
            "model": {"backend": "tensorrt", "model_id": "codriving:joint_shcosearch"},
            "device": {"hostname": "orin"},
            "edge_timestamps_ns": timestamps,
            "durations_ns": {"inference": 1},
            "trigger_identity": trigger,
            "artifact_identity": artifact,
            "payload_sha256": payload_sha256 or hash_feature_payload(outputs),
            "batch_order": ["ego", "rsu"],
            "warmup": False,
        }
        header.update(trigger)
        return build_wire_bytes(header, outputs, limits=ProtocolLimits()), outputs

    def test_bridge_records_real_ros_receive_decode_and_preserves_hil_identity(self):
        from scripts.ditl_ros2_edge_bridge import _finalize_result_wire
        from ditl_edge_protocol import ProtocolLimits

        wire, expected = self._wire_fixture()
        sideband = {
            "request_id": "run-task3:7",
            "local_handoff_end": 50,
            "ros_message_start": 23,
            "ros_message_end": 24,
            "ros_publish_start": 25,
            "ros_publish_end": 26,
        }
        arrays, record = _finalize_result_wire(
            wire,
            sideband,
            ros_receive_ns=27,
            receiver_host_id="receiver-host",
            receiver_boot_id="receiver-boot",
            limits=ProtocolLimits(),
            result_topic="/coptv2x/task4/result",
        )

        self.assertEqual(set(expected), set(arrays))
        self.assertEqual(COPTV2X_PAYLOAD_SEMANTICS, record["source"]["payload_semantics"])
        self.assertEqual(27, record["timestamps_ns"]["ros_receive"])
        self.assertLessEqual(
            record["timestamps_ns"]["ros_decode_start"],
            record["timestamps_ns"]["ros_decode_end"],
        )
        self.assertGreaterEqual(record["durations_ns"]["ros_decode"], 0)
        self.assertEqual(25, record["durations_ns"]["local_handoff"])
        self.assertEqual(0, record["network"]["remote_input_tensor_bytes"])
        self.assertEqual("joint_shcosearch", record["model"]["artifact_identity"]["arm_id"])
        self.assertEqual(["ego", "rsu"], record["model"]["batch_order"])
        self.assertEqual("/coptv2x/task4/result", record["message"]["topic"])

    def test_copt_pipeline_rejects_missing_or_swapped_source_batch_order(self):
        from ditl_edge_runtime import monotonic_raw_ns

        tree = ArtifactTree()
        try:
            for batch_order in (None, ("rsu", "ego")):
                with self.subTest(batch_order=batch_order):
                    backend = CoptV2xHilBackend(
                        arm_id="joint_shcosearch",
                        artifact_root=tree.artifact_root,
                        config_dir=tree.config_dir,
                        runtime_factory=RuntimeFactory(),
                        native_repo_root=tree.root,
                    )
                    pipeline = CoptV2xHilPipeline(
                        backend,
                        run_id="run-task3",
                        route_id="Town05-route-01",
                        scenario_id="scenario-01",
                    )
                    trigger = _trigger("joint_shcosearch")
                    metadata = {
                        "source_type": "local_indexed_feature",
                        "trigger_identity": trigger.to_metadata(),
                        "timestamps_ns": {},
                        "durations_ns": {},
                    }
                    if batch_order is not None:
                        metadata["batch_order"] = batch_order
                    frame = LocalRsuFrame(
                        frame_id=trigger.source_frame,
                        sequence_id=trigger.sequence_no,
                        data_available_ns=monotonic_raw_ns(),
                        arrays={
                            "spatial_features": np.zeros(
                                (2, 64, 256, 512), dtype=np.float32
                            )
                        },
                        source_metadata=metadata,
                    )
                    with self.assertRaisesRegex(ValueError, "batch order"):
                        pipeline.process(frame)
                    self.assertEqual(0, backend.inference_count)
        finally:
            tree.close()

    def test_bridge_rejects_payload_hash_conflict(self):
        from scripts.ditl_ros2_edge_bridge import _finalize_result_wire
        from ditl_edge_protocol import ProtocolError, ProtocolLimits

        wire, _ = self._wire_fixture(payload_sha256="0" * 64)
        sideband = {
            "request_id": "run-task3:7",
            "local_handoff_end": 22,
            "ros_message_start": 23,
            "ros_message_end": 24,
            "ros_publish_start": 25,
            "ros_publish_end": 26,
        }
        with self.assertRaisesRegex(ProtocolError, "payload hash"):
            _finalize_result_wire(
                wire,
                sideband,
                ros_receive_ns=27,
                receiver_host_id="receiver-host",
                receiver_boot_id="receiver-boot",
                limits=ProtocolLimits(),
            )

    def test_bridge_requires_remote_input_tensor_bytes_to_be_integer_zero(self):
        from scripts.ditl_ros2_edge_bridge import _finalize_result_wire
        from ditl_edge_protocol import ProtocolError, ProtocolLimits

        wire, _ = self._wire_fixture(remote_input_tensor_bytes=False)
        sideband = {
            "request_id": "run-task3:7",
            "local_handoff_end": 22,
            "ros_message_start": 23,
            "ros_message_end": 24,
            "ros_publish_start": 25,
            "ros_publish_end": 26,
        }
        with self.assertRaisesRegex(ProtocolError, "remote input"):
            _finalize_result_wire(
                wire,
                sideband,
                ros_receive_ns=27,
                receiver_host_id="receiver-host",
                receiver_boot_id="receiver-boot",
                limits=ProtocolLimits(),
            )

    def test_bridge_rejects_wrong_contract_or_nonfinite_self_hashed_payload(self):
        from scripts.ditl_ros2_edge_bridge import _finalize_result_wire
        from ditl_edge_protocol import ProtocolError, ProtocolLimits

        invalid_payloads = (
            {
                "res0": np.zeros((2, 1, 2, 2), dtype=np.float32),
                "res1": np.zeros((2, 2, 1, 1), dtype=np.float32),
                "res2": np.zeros((2, 4, 1, 1), dtype=np.float32),
            },
            {
                "res0": np.full((2, 16, 128, 256), np.nan, dtype=np.float32),
                "res1": np.zeros((2, 32, 64, 128), dtype=np.float32),
                "res2": np.zeros((2, 64, 32, 64), dtype=np.float32),
            },
        )
        sideband = {
            "request_id": "run-task3:7",
            "local_handoff_end": 22,
            "ros_message_start": 23,
            "ros_message_end": 24,
            "ros_publish_start": 25,
            "ros_publish_end": 26,
        }
        for outputs in invalid_payloads:
            wire, _ = self._wire_fixture(outputs=outputs)
            with self.subTest(shapes={name: value.shape for name, value in outputs.items()}):
                with self.assertRaisesRegex(ProtocolError, "contract"):
                    _finalize_result_wire(
                        wire,
                        sideband,
                        ros_receive_ns=27,
                        receiver_host_id="receiver-host",
                        receiver_boot_id="receiver-boot",
                        limits=ProtocolLimits(),
                    )


if __name__ == "__main__":
    unittest.main()
