#!/usr/bin/env python3
"""Freeze the official Town05 85-cell x five-arm CoptV2X HIL cohort."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from action_fifo_full_manifest import SUBSETS
from coptv2x_hil_artifacts import HIL_ARM_IDS, load_hil_artifact_registry


SCHEMA = "coptv2x.hil-cohort.v1"
ARM_IDS = HIL_ARM_IDS
SOURCE_AUTHORITY = "scripts/batch_eval_driving_e2e.sh@f63e92b"
DEFAULT_ARTIFACT_CONFIG_DIR = (
    REPO_ROOT / "simulation/leaderboard/team_code/agent_config/coptv2x_hil"
)
RUNTIME_CONFIG_DIR = Path("simulation/leaderboard/team_code/agent_config/coptv2x_hil")
HIL_ARMS = tuple(
    {"arm_id": arm, "artifact_id": "joint_shcosearch_fp16" if arm == "joint_shcosearch" else arm,
     "runtime_config": str(RUNTIME_CONFIG_DIR / (arm + ".yaml"))}
    for arm in HIL_ARM_IDS
)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _route_scenarios():
    cells = []
    for subset, scenario, profile, routes in SUBSETS:
        for route in routes:
            cells.append({
                "cell_id": "Town05:%s:r%d" % (scenario, route),
                "town": "Town05", "subset": subset, "scenario": scenario,
                "route": route, "profile": "5_10",
                "route_xml": "simulation/leaderboard/data/evaluation_routes/town05_short_r%d.xml" % route,
                "scenarios_json": "simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json",
            })
    if len(cells) != 85 or len({item["cell_id"] for item in cells}) != 85:
        raise ValueError("frozen action FIFO source authority must produce exactly 85 unique cells")
    for cell in cells:
        if not (REPO_ROOT / cell["route_xml"]).is_file():
            raise ValueError("official route file is missing: %s" % cell["route_xml"])
    return cells


def build_manifest(seed=2000, repeat=0, cadence=2, profile="5_10",
                   artifact_config_dir=DEFAULT_ARTIFACT_CONFIG_DIR):
    if seed != 2000 or repeat != 0 or cadence != 2 or profile != "5_10":
        raise ValueError("Task5 cohort is frozen to seed=2000 repeat=0 cadence=2 profile=5_10")
    artifacts = load_hil_artifact_registry(artifact_config_dir)
    cells = _route_scenarios()
    route_payload = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode("utf-8")
    authority_path = REPO_ROOT / "scripts/action_fifo_full_manifest.py"
    artifact_files = {
        arm: _sha256_file(Path(artifact_config_dir) / ("artifact_%s.yaml" % arm))
        for arm in HIL_ARM_IDS
    }
    runtime_files = {
        arm["arm_id"]: _sha256_file(REPO_ROOT / arm["runtime_config"])
        for arm in HIL_ARMS
    }
    inputs = {
        "route_authority": SOURCE_AUTHORITY,
        "route_manifest_version": "town05-action-fifo-subsets.v1",
        "route_authority_sha256": _sha256_file(authority_path),
        "route_scenarios_sha256": _sha256_bytes(route_payload),
        "artifact_config_sha256": artifact_files,
        "runtime_config_sha256": runtime_files,
        "artifact_registry_version": "coptv2x.hil-artifact-registry.v1",
    }
    input_hash = _sha256_bytes(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    tasks = []
    arms = []
    for item in HIL_ARMS:
        spec = artifacts[item["arm_id"]]
        arms.append(dict(item))
    for cell in cells:
        cell_id = cell["cell_id"]
        for arm in arms:
            arm_id = arm["arm_id"]
            tasks.append({
                "task_id": "hil_%s_%s_seed2000_rep0" % (cell_id.replace(":", "_"), arm_id),
                "route_scenario_id": cell_id,
                "town": "Town05", "subset": cell["subset"],
                "scenario": cell["scenario"],
                "scenario_id": "scenario%s" % cell["scenario"].lstrip("_"),
                "route": cell["route"],
                "route_xml": cell["route_xml"], "scenarios_json": cell["scenarios_json"],
                "profile": "5_10", "arm_id": arm_id,
                "artifact_id": arm["artifact_id"],
                "runtime_config": arm["runtime_config"],
                "artifact_sha256": dict(artifacts[arm_id].artifact_sha256),
                "seed": 2000, "traffic_seed": 2000, "repeat": 0,
                "dt_ms": 50, "frequency_hz": 20, "cadence": 2,
            })
    if len(tasks) != 425 or len({task["task_id"] for task in tasks}) != 425:
        raise ValueError("cohort task IDs must be the exact 85 x 5 matrix")
    return {
        "schema": SCHEMA, "manifest_version": "coptv2x-live-hil-v1",
        "input_manifest_version": inputs["route_manifest_version"],
        "input_manifest_sha256": input_hash,
        "inputs": inputs,
        "route_scenarios": [item["cell_id"] for item in cells],
        "route_scenario_records": cells,
        "route_scenarios_sha256": inputs["route_scenarios_sha256"],
        "arms": arms, "tasks": tasks,
        "protocol": {"seed": 2000, "traffic_seed": 2000, "repeat": 0,
                     "town": "Town05", "frequency_hz": 20, "dt_ms": 50,
                     "cadence": 2, "profile": "5_10",
                     "remote_input_tensor_bytes": 0},
    }


def select_pilot(manifest, route_scenarios):
    allowed = set(manifest["route_scenarios"])
    selected = tuple(route_scenarios)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= allowed:
        raise ValueError("pilot cells must be unique members of the frozen route-scenario list")
    selected_set = set(selected)
    pilot_tasks = [task for task in manifest["tasks"] if task["route_scenario_id"] in selected_set]
    expected = len(selected_set) * len(HIL_ARM_IDS)
    if len(pilot_tasks) != expected:
        raise ValueError("pilot manifest lost paired arm tasks")
    pilot = {key: value for key, value in manifest.items()
             if key not in ("tasks", "route_scenarios", "route_scenario_records")}
    pilot.update({
        "pilot_route_scenarios": list(selected),
        "route_scenarios": list(selected),
        "route_scenario_records": [item for item in manifest["route_scenario_records"]
                                   if item["cell_id"] in selected_set],
        "tasks": pilot_tasks,
    })
    return pilot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = build_manifest()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
