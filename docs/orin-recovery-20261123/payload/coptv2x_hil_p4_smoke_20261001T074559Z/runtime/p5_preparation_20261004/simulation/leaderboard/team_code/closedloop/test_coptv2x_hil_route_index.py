from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "scripts"
CWD_CLOSEDLOOP = Path(__file__).resolve().parent
if str(CWD_CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CWD_CLOSEDLOOP))

import importlib.util
import coptv2x_hil_route_capture as capture_module
from ditl_rsu_source import IndexedLocalRsuSource
from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger

SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "coptv2x_hil_route_index_script", SCRIPTS / "coptv2x_hil_route_capture.py"
)
index_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(index_script)
build_route_indexes = index_script.build_route_indexes
HOST_INDEX_SCHEMA = index_script.HOST_INDEX_SCHEMA
LOCAL_INDEX_SCHEMA = index_script.LOCAL_INDEX_SCHEMA
PILOT_CELL = index_script.PILOT_CELL
RouteInputCapture = capture_module.RouteInputCapture


class RouteInputIndexTests(unittest.TestCase):
    def test_builds_host_and_orin_indexes_for_exact_capture_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "route-cache"
            capture = RouteInputCapture(
                data_root, route_id="Town05-route-2", scenario_id="scenario1"
            )
            features = np.zeros((2, 64, 256, 512), dtype=np.float32)
            capture.capture_frame({"spatial_features": features}, 12)
            capture.capture_frame({"spatial_features": features}, 14)
            capture_manifest = capture.finalize()
            cohort = {
                "schema": "coptv2x.hil-cohort.v1",
                "route_scenarios_sha256": "a" * 64,
                "route_scenarios": [PILOT_CELL],
                "route_scenario_records": [{
                    "cell_id": PILOT_CELL,
                    "route": 2,
                    "scenario": "_1",
                }],
                "tasks": [{
                    "route_scenario_id": PILOT_CELL,
                    "scenario_id": "scenario1",
                }],
            }
            host_index_path = root / "host-index.json"

            local_path, returned_host_path = build_route_indexes(
                capture_manifest, cohort, data_root, host_index_path
            )

            host_index = json.loads(host_index_path.read_text(encoding="utf-8"))
            local_index = json.loads(local_path.read_text(encoding="utf-8"))
            self.assertEqual(HOST_INDEX_SCHEMA, host_index["schema"])
            self.assertEqual(host_index_path, returned_host_path)
            self.assertEqual("HOST_INDEX_VALID", host_index["status"])
            self.assertEqual("a" * 64, host_index["route_scenarios_sha256"])
            self.assertEqual([PILOT_CELL], list(host_index["route_scenarios"]))
            self.assertEqual([12, 14], [x["source_frame"] for x in
                                         host_index["route_scenarios"][PILOT_CELL]])
            self.assertEqual(LOCAL_INDEX_SCHEMA, local_index["schema_version"])
            self.assertEqual(["ego", "rsu"], local_index["batch_order"])
            self.assertEqual([12, 14], [x["source_frame"] for x in local_index["samples"]])
            self.assertEqual(
                [2, 64, 256, 512],
                local_index["samples"][0]["inputs"]["spatial_features"]["shape"],
            )
            source = IndexedLocalRsuSource(
                data_root,
                local_index,
                {"spatial_features": {"shape": [2, 64, 256, 512], "dtype": "float32"}},
                max_samples=10,
                max_total_tensor_bytes=1024 * 1024 * 1024,
            )
            trigger = parse_hil_trigger({
                "schema_version": HIL_TRIGGER_SCHEMA_VERSION,
                "run_id": "smoke",
                "route_id": "Town05-route-2",
                "scenario_id": "scenario1",
                "sample_id": "carla-frame-14",
                "sequence_no": 1,
                "source_frame": 14,
                "source_sim_time_ns": 700000000,
                "arm_id": "original_default",
            })
            selected = source.frame_for_trigger(trigger)
            self.assertEqual(14, selected.frame_id)
            self.assertEqual(1, selected.sequence_id)
            self.assertEqual((2, 64, 256, 512),
                             selected.arrays["spatial_features"].shape)

    def test_indexes_capture_files_relocated_to_verified_spill_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture_root = root / "capture"
            spill_root = root / "spill"
            capture_root.mkdir()
            spill_root.mkdir()
            capture = RouteInputCapture(
                capture_root, route_id="Town05-route-2", scenario_id="scenario1"
            )
            features = np.zeros((2, 64, 256, 512), dtype=np.float32)
            record = capture.capture_frame({"spatial_features": features}, 12)
            capture.capture_frame({"spatial_features": features}, 14)
            capture_manifest = capture.finalize()
            original = Path(record["spatial_features_path"])
            relocated = spill_root / original.name
            shutil.move(str(original), str(relocated))
            original.symlink_to(relocated)
            cohort = {
                "schema": "coptv2x.hil-cohort.v1",
                "route_scenarios_sha256": "c" * 64,
                "route_scenarios": [PILOT_CELL],
                "route_scenario_records": [{
                    "cell_id": PILOT_CELL,
                    "route": 2,
                    "scenario": "_1",
                }],
                "tasks": [{
                    "route_scenario_id": PILOT_CELL,
                    "scenario_id": "scenario1",
                }],
            }

            local_path, host_path = build_route_indexes(
                capture_manifest, cohort, spill_root, root / "host.json"
            )

            self.assertEqual(spill_root / "local_input_manifest.json", local_path)
            local_index = json.loads(local_path.read_text(encoding="utf-8"))
            self.assertEqual(
                original.name,
                local_index["samples"][0]["inputs"]["spatial_features"]["path"],
            )
            host_index = json.loads(host_path.read_text(encoding="utf-8"))
            self.assertEqual(
                str(relocated),
                host_index["route_scenarios"][PILOT_CELL][0]["spatial_features_path"],
            )

    def test_rejects_capture_for_wrong_route_or_scenario(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = RouteInputCapture(
                root / "cache", route_id="Town05-route-8", scenario_id="scenario4"
            )
            capture.capture_frame(
                {"spatial_features": np.zeros((2, 64, 256, 512), dtype=np.float32)},
                3,
            )
            cohort = {
                "schema": "coptv2x.hil-cohort.v1",
                "route_scenarios_sha256": "b" * 64,
                "route_scenarios": [PILOT_CELL],
                "route_scenario_records": [{
                    "cell_id": PILOT_CELL,
                    "route": 2,
                    "scenario": "_1",
                }],
                "tasks": [{
                    "route_scenario_id": PILOT_CELL,
                    "scenario_id": "scenario1",
                }],
            }
            with self.assertRaisesRegex(ValueError, "route/scenario"):
                build_route_indexes(
                    capture.finalize(), cohort, root / "cache", root / "host.json"
                )


if __name__ == "__main__":
    unittest.main()
