"""Resident model lifetime must not preserve per-episode results or admission."""

from argparse import Namespace
from dataclasses import replace
import contextlib
import hashlib
import json
import io
from pathlib import Path
import socket
import select
import sys
import unittest
from unittest.mock import patch
import weakref

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation/leaderboard/team_code/closedloop"))

from scripts import coptv2x_hil_orin_worker as worker
import coptv2x_hil_backend as backend_module
from coptv2x_hil_artifacts import TensorContract, load_hil_artifact_registry
from ditl_edge_protocol import decode_wire_frame, recv_wire_frame
from ditl_rsu_source import LOCAL_INPUT_MANIFEST_SCHEMA_VERSION
from test_coptv2x_hil_backend import ArtifactTree


class SmallRuntime:
    """Replace GPU execution only; preserve real worker/backend/pipeline I/O."""

    def __init__(self, spec, input_contract, output_contract):
        self.arm_id = spec.arm_id
        self.input_contract = input_contract
        self.output_contract = output_contract
        self.inference_count = 0
        self.input_values = []

    def infer_validated(self, inputs):
        self.inference_count += 1
        value = inputs["spatial_features"].flat[0]
        self.input_values.append(float(value))
        return {
            "outputs": {
                name: np.full(contract["shape"], value, dtype=contract["dtype"])
                for name, contract in self.output_contract.items()
            },
            "model": {"model_id": self.arm_id},
        }


class ResidentModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tree = ArtifactTree()
        self.addCleanup(self.tree.close)
        registry = {
            arm: replace(
                spec,
                input_contract=(TensorContract("spatial_features", (2, 1, 1, 1), "float32"),),
                output_contract=tuple(
                    TensorContract(name, (2, 1, 1, 1), "float32")
                    for name in ("res0", "res1", "res2")
                ),
            )
            for arm, spec in load_hil_artifact_registry(self.tree.config_dir).items()
        }
        self.runtime_arms = []
        self.runtime_refs = []
        for target, name, options in (
            (worker, "load_hil_artifact_registry", {"return_value": registry}),
            (backend_module, "load_hil_artifact_registry", {"return_value": registry}),
            (backend_module, "_default_runtime_factory", {"side_effect": self.runtime_factory}),
        ):
            mocked = patch.object(target, name, **options)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.sender, self.receiver = socket.socketpair()
        self.receiver.settimeout(1)
        self.addCleanup(self.receiver.close)
        self.addCleanup(self.sender.close)
        handoff = patch.object(worker, "_connect_local_result_handoff", return_value=self.sender)
        handoff.start()
        self.addCleanup(handoff.stop)
        self.args = Namespace(
            config_dir=str(self.tree.config_dir), artifact_root=str(self.tree.artifact_root),
            native_repo_root=str(self.tree.root), data_root=str(self.tree.root),
            manifest=str(self.tree.root / "manifest.json"), max_samples=4,
            max_total_tensor_mib=1, max_payload_mib=1, result_count=20,
            result_host="127.0.0.1", result_port=55021, connect_timeout_s=1,
            serial_optimized=True, wire_compression="none",
        )
        self.write_manifest("route-1", "scenario-1")
        self.instance = worker.OrinHilWorker(self.args)
        self.addCleanup(self.instance.close)

    def runtime_factory(self, spec, paths, input_contract, output_contract, native_repo_root):
        runtime = SmallRuntime(spec, input_contract, output_contract)
        self.runtime_arms.append(spec.arm_id)
        self.runtime_refs.append(weakref.ref(runtime))
        return runtime

    def write_manifest(self, route_id, scenario_id):
        samples = []
        for source_frame in (1, 2):
            path = self.tree.root / ("input-%d.npy" % source_frame)
            np.save(str(path), np.full((2, 1, 1, 1), source_frame, dtype=np.float32))
            samples.append({
                "sample_id": "sample-%d" % source_frame, "source_frame": source_frame,
                "inputs": {"spatial_features": {
                    "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "shape": [2, 1, 1, 1], "dtype": "float32",
                }},
            })
        document = {
            "schema_version": LOCAL_INPUT_MANIFEST_SCHEMA_VERSION,
            "route_id": route_id, "scenario_id": scenario_id,
            "batch_order": ["ego", "rsu"], "samples": samples,
        }
        Path(self.args.manifest).write_text(json.dumps(document), encoding="utf-8")
        return document

    def install_source(self, route_id, scenario_id):
        # Route input replacement belongs to the surrounding service, not the
        # model: exercise lifecycle with a real exact-identity indexed source.
        manifest = self.write_manifest(route_id, scenario_id)
        self.instance.source = worker.IndexedLocalRsuSource(
            self.args.data_root, manifest,
            {"spatial_features": {"shape": (2, 1, 1, 1), "dtype": "float32"}},
            max_samples=4, max_total_tensor_bytes=1024, memory_map=True, lazy_load=True,
        )

    def trigger(self, sequence_no=0, source_frame=1, **identity):
        metadata = {
            "schema_version": "coptv2x.hil-trigger.v1", "run_id": "run-1",
            "route_id": "route-1", "scenario_id": "scenario-1",
            "arm_id": "joint_shcosearch", "sequence_no": sequence_no,
            "sample_id": "sample-%d" % source_frame, "source_frame": source_frame,
            "source_sim_time_ns": source_frame * 50000000,
        }
        metadata.update(identity)
        return worker.parse_hil_trigger(metadata)

    def process(self, trigger):
        request_id = self.instance.process_trigger(trigger)
        header_bytes, payload, _ = recv_wire_frame(self.receiver)
        header, arrays = decode_wire_frame(header_bytes, payload)
        self.assertEqual(header["request_id"], request_id)
        self.assertEqual(trigger.to_metadata(), header["trigger_identity"])
        self.assertFalse(header["warmup"])
        if self.args.serial_optimized:
            self.assertEqual("codriving_feature_payload.v1", header["tensor_bundle"]["sha256_scope"])
        else:
            self.assertNotIn("sha256_scope", header["tensor_bundle"])
        for array in arrays.values():
            np.testing.assert_array_equal(array, np.full(array.shape, trigger.source_frame))
        return header

    def test_serial_episode_changes_reuse_model_but_reset_pipeline_and_admission(self):
        self.process(self.trigger(sequence_no=7))
        backend = self.instance.backend
        previous_pipeline = self.instance.pipeline
        previous_admission = self.instance.admission
        for identity in (
            {"route_id": "route-2", "scenario_id": "scenario-1", "run_id": "run-1"},
            {"route_id": "route-2", "scenario_id": "scenario-2", "run_id": "run-1"},
            {"route_id": "route-2", "scenario_id": "scenario-2", "run_id": "run-2"},
        ):
            with self.subTest(identity=identity):
                self.install_source(identity["route_id"], identity["scenario_id"])
                trigger = self.trigger(source_frame=2, **identity)
                header = self.process(trigger)
                self.assertIs(backend, self.instance.backend)
                self.assertEqual(["joint_shcosearch"], self.runtime_arms)
                self.assertIsNot(previous_pipeline, self.instance.pipeline)
                self.assertIsNot(previous_admission, self.instance.admission)
                self.assertEqual(0, header["sequence_id"])
                with self.assertRaisesRegex(ValueError, "increase strictly"):
                    self.instance.process_trigger(trigger)
                self.assertEqual(1, self.instance.pipeline._last_sequence_id + 1)
                previous_pipeline = self.instance.pipeline
                previous_admission = self.instance.admission
        self.assertEqual(4, backend.inference_count)
        self.assertEqual(4, self.instance.processed)

    def test_new_episode_admission_rejects_previous_route_identity(self):
        old_trigger = self.trigger(sequence_no=7)
        self.process(old_trigger)
        self.install_source("route-2", "scenario-1")
        self.process(self.trigger(route_id="route-2"))
        with self.assertRaisesRegex(ValueError, "route_id does not match"):
            self.instance.admission.reserve(old_trigger.to_metadata())

    def test_same_episode_processes_each_requested_observation_without_reusing_results(self):
        self.process(self.trigger())
        self.process(self.trigger(sequence_no=1, source_frame=2))
        self.assertEqual([1.0, 2.0], self.runtime_refs[0]().input_values)
        self.assertEqual(2, self.instance.backend.inference_count)
        self.assertEqual(["joint_shcosearch"], self.runtime_arms)

    def test_serial_arm_change_evicts_the_only_resident_model(self):
        self.process(self.trigger())
        previous_backend = weakref.ref(self.instance.backend)
        self.process(self.trigger(arm_id="schedule_only"))
        self.assertEqual(["joint_shcosearch", "schedule_only"], self.runtime_arms)
        self.assertIsNone(previous_backend())
        self.assertIsNone(self.runtime_refs[0]())
        self.process(self.trigger(arm_id="joint_shcosearch"))
        self.assertEqual(["joint_shcosearch", "schedule_only", "joint_shcosearch"], self.runtime_arms)
        self.assertIsNone(self.runtime_refs[1]())

    def test_legacy_episode_change_still_reloads_model(self):
        self.args.serial_optimized = False
        self.process(self.trigger(sequence_no=7))
        previous_backend = weakref.ref(self.instance.backend)
        self.process(self.trigger(run_id="run-2"))
        self.assertEqual(["joint_shcosearch", "joint_shcosearch"], self.runtime_arms)
        self.assertIsNone(previous_backend())

    def test_close_releases_resident_model_and_episode_state(self):
        self.process(self.trigger())
        backend = weakref.ref(self.instance.backend)
        runtime = self.runtime_refs[0]
        self.instance.close()
        self.assertIsNone(backend())
        self.assertIsNone(runtime())
        self.assertIsNone(self.instance.pipeline)
        self.assertIsNone(self.instance.admission)
        self.assertIsNone(self.instance.active_key)
        self.assertIsNone(self.instance.connection)
        self.assertEqual(-1, self.sender.fileno())

    def test_startup_warmup_discards_zero_input_outputs_without_consuming_results(self):
        self.args.startup_arm_id = "joint_shcosearch"
        self.args.runtime_warmup_count = 2
        self.args.result_count = 1
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            self.instance = worker.OrinHilWorker(self.args)
            self.addCleanup(self.instance.close)
        self.assertIsNotNone(self.instance.backend)
        runtime = self.runtime_refs[0]()
        self.assertEqual([0.0, 0.0], runtime.input_values)
        self.assertEqual(2, runtime.inference_count)
        self.assertEqual(0, self.instance.backend.inference_count)
        self.assertEqual(0, self.instance.processed)
        self.assertIsNone(self.instance.pipeline)
        self.assertIsNone(self.instance.admission)
        self.assertIsNone(self.instance.active_key)
        self.assertEqual([], select.select([self.receiver], [], [], 0)[0])
        event = json.loads(log.getvalue())
        self.assertEqual("orin_model_initialized", event["event"])
        self.assertEqual("joint_shcosearch", event["arm_id"])
        self.assertEqual(2, event["runtime_warmup_count"])
        self.assertGreaterEqual(event["engine_load_ns"], 0)
        self.assertGreaterEqual(event["runtime_warmup_ns"], 0)
        with contextlib.redirect_stderr(log):
            self.process(self.trigger())
        self.assertEqual([0.0, 0.0, 1.0], runtime.input_values)
        self.assertEqual(1, self.instance.backend.inference_count)
        self.assertEqual(1, self.instance.processed)
        self.assertEqual(1, len(log.getvalue().splitlines()))
        self.assertEqual(["joint_shcosearch"], self.runtime_arms)
        with self.assertRaisesRegex(ValueError, "result_count"):
            self.instance.process_trigger(self.trigger(sequence_no=1, source_frame=2))

    def test_warmup_startup_options_are_serial_only_and_bounded(self):
        for options in (
            {"serial_optimized": False, "startup_arm_id": "joint_shcosearch"},
            {"serial_optimized": False, "runtime_warmup_count": 1},
            {"runtime_warmup_count": 1},
            {"startup_arm_id": "unknown-arm"},
            {"runtime_warmup_count": -1},
            {"runtime_warmup_count": 101},
            {"runtime_warmup_count": True},
        ):
            with self.subTest(options=options):
                args = Namespace(**dict(vars(self.args), **options))
                with self.assertRaises(ValueError):
                    worker.OrinHilWorker(args)

    def test_startup_connection_failure_releases_initialized_model(self):
        self.args.startup_arm_id = "joint_shcosearch"
        self.args.runtime_warmup_count = 1
        instance = worker.OrinHilWorker.__new__(worker.OrinHilWorker)
        with patch.object(worker, "_connect_local_result_handoff", side_effect=OSError("handoff failed")):
            with self.assertRaisesRegex(OSError, "handoff failed"):
                instance.__init__(self.args)
        self.assertIsNone(instance.backend)
        self.assertIsNone(instance.engine_key)
        self.assertIsNone(instance.connection)


if __name__ == "__main__":
    unittest.main()
