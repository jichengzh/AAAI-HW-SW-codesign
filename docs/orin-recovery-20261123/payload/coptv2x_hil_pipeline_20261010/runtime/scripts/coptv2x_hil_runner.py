#!/usr/bin/env python3
"""Run/resume the frozen CoptV2X HIL manifest one isolated CARLA episode at a time."""

from __future__ import print_function

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import yaml

from coptv2x_hil_manifest import REPO_ROOT, build_manifest, select_pilot
from coptv2x_hil_artifacts import DEFAULT_ARTIFACT_ROOT, load_hil_artifact_registry
from coptv2x_hil_result import (atomic_json, classify_driving_result, file_sha256,
                                paired_row, validate_episode_audit, validate_task_artifacts,
                                write_paired_csv)


class DuplicateTaskResultError(ValueError):
    pass


class ManifestMismatchError(ValueError):
    pass


TERMINAL = frozenset(("COMPLETED", "DRIVING_TERMINAL", "INFRA_INVALID"))
INFRA_STATUSES = frozenset(("ros_disconnected", "orin_engine_crash", "identity_mismatch",
                            "result_corrupt", "carla_crash", "agent_crash",
                            "infrastructure_error", "missing_result"))
BASE_AGENT_CONFIG = REPO_ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_5_10.yaml"
DEFAULT_PLANNER_CHECKPOINT = (
    Path(os.environ.get("CARLA_ROOT", REPO_ROOT / "external_paths/carla_root"))
    .expanduser().resolve().parent
    / "checkpoints/codriving/planner/codriving_planner.ckpt"
)
DEFAULT_SIMULATION_PERCEPTION_DIR = DEFAULT_PLANNER_CHECKPOINT.parent.parent / "perception"
DEFAULT_SIMULATION_PERCEPTION_CONFIG = DEFAULT_SIMULATION_PERCEPTION_DIR / "config.yaml"
DEFAULT_PLANNING_FEATURE_ADAPTER_CHECKPOINT = (
    DEFAULT_SIMULATION_PERCEPTION_DIR / "net_epoch_bestval_at16.pth"
)


def _task_ids_digest(task_ids):
    encoded = json.dumps(list(task_ids), separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunState(object):
    """Atomic resumable task journal; attempts and terminal results are separate."""

    def __init__(self, path, task_ids, manifest_sha256=None, local_input_manifest=None):
        self.path = Path(path).expanduser().resolve()
        self.task_ids = tuple(task_ids)
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("manifest task IDs must be unique")
        self.digest = _task_ids_digest(self.task_ids)
        self.manifest_sha256 = manifest_sha256
        self.local_input_manifest = local_input_manifest
        self.data = {"schema": "coptv2x.hil-run-state.v1", "manifest_task_ids_sha256": self.digest,
                     "manifest_sha256": manifest_sha256,
                     "host_input_index": local_input_manifest,
                     "task_ids": list(self.task_ids), "attempts": {}, "results": {}}
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError("run state is unreadable: %s" % error) from error
            if (saved.get("manifest_task_ids_sha256") != self.digest
                    or saved.get("task_ids") != list(self.task_ids)
                    or saved.get("manifest_sha256") != manifest_sha256
                    or saved.get("host_input_index") != local_input_manifest):
                raise ManifestMismatchError("run state belongs to a different manifest")
            if not isinstance(saved.get("attempts"), dict) or not isinstance(saved.get("results"), dict):
                raise ValueError("run state schema is corrupt")
            self.data = saved
        else:
            self._save()

    def _save(self):
        atomic_json(self.path, self.data)

    def pending_ids(self):
        return [task_id for task_id in self.task_ids if task_id not in self.data["results"]]

    def record_attempt(self, task_id, entry):
        if task_id not in self.data["task_ids"]:
            raise KeyError("task ID is not in this manifest")
        if task_id in self.data["results"]:
            raise DuplicateTaskResultError("task already has a terminal result: %s" % task_id)
        updated = dict(self.data["attempts"])
        entries = list(updated.get(task_id, []))
        entries.append(dict(entry))
        updated[task_id] = entries
        self.data = dict(self.data, attempts=updated)
        self._save()

    def record(self, task_id, result):
        if task_id not in self.data["task_ids"]:
            raise KeyError("task ID is not in this manifest")
        if task_id in self.data["results"]:
            raise DuplicateTaskResultError("task already has a terminal result: %s" % task_id)
        classification = result.get("classification")
        if classification is None and result.get("status") == "Completed":
            result = dict(result, classification="COMPLETED")
            classification = "COMPLETED"
        if classification not in TERMINAL:
            raise ValueError("only a terminal classification may close a task")
        updated = dict(self.data["results"])
        updated[task_id] = dict(result)
        self.data = dict(self.data, results=updated)
        self._save()


def classify_episode(status, return_code=0):
    normalized = str(status or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ("completed", "complete", "success") and return_code == 0:
        return "COMPLETED"
    if normalized in ("collision", "blocked", "route_timeout", "timeout",
                      "route_not_completed", "not_completed", "failed_driving"):
        return "DRIVING_TERMINAL"
    return "INFRA_RETRY"


def _runtime_config_path(task, run_dir):
    return Path(run_dir).expanduser().resolve() / "agent_config.yaml"


def prepare_task_config(task, run_dir, cohort_id, ros_domain_id, proxy_port=56610):
    runtime_path = (REPO_ROOT / task["runtime_config"]).resolve()
    overlay = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    base_path = (REPO_ROOT / overlay["base_config"]).resolve()
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    artifacts = load_hil_artifact_registry()
    arm_id = task["arm_id"]
    try:
        artifact = artifacts[arm_id]
    except KeyError as error:
        raise ValueError("task arm is not in the frozen HIL artifact registry") from error
    if not DEFAULT_SIMULATION_PERCEPTION_CONFIG.is_file():
        raise RuntimeError("simulation CoDriving hypes are missing: %s" % DEFAULT_SIMULATION_PERCEPTION_CONFIG)
    frozen_hypes = yaml.safe_load((DEFAULT_ARTIFACT_ROOT / artifact.config).read_text(encoding="utf-8"))
    simulation_hypes = yaml.load(
        DEFAULT_SIMULATION_PERCEPTION_CONFIG.read_text(encoding="utf-8"),
        Loader=yaml.Loader,
    )
    if simulation_hypes.get("fusion", {}).get("dataset") != "v2xverse":
        raise RuntimeError("CARLA runtime hypes must use the V2XVerse online dataset adapter")
    simulation_model_args = simulation_hypes.get("model", {}).get("args", {})
    frozen_model_args = frozen_hypes.setdefault("model", {}).setdefault("args", {})
    # CoDriving online inference requires the simulation's multiclass output
    # contract, but architecture-changing flags must remain those of the
    # frozen arm: copying shrink_header changes checkpoint tensor shapes.
    for simulation_key in ("multi_class",):
        if simulation_key in simulation_model_args:
            frozen_model_args[simulation_key] = simulation_model_args[simulation_key]
    simulation_hypes["model"] = frozen_hypes["model"]
    simulation_hypes["preprocess"] = frozen_hypes["preprocess"]
    perception_dir = Path(run_dir).resolve() / "runtime-perception" / arm_id
    perception_dir.mkdir(parents=True, exist_ok=True)
    (perception_dir / "config.yaml").write_text(
        yaml.dump(simulation_hypes, sort_keys=False), encoding="utf-8"
    )
    checkpoint = (DEFAULT_ARTIFACT_ROOT / artifact.checkpoint).resolve()
    if not checkpoint.is_file():
        raise RuntimeError("frozen perception checkpoint is missing: %s" % checkpoint)
    runtime_checkpoint = perception_dir / artifact.checkpoint.name
    if not runtime_checkpoint.exists():
        runtime_checkpoint.symlink_to(checkpoint)
    elif runtime_checkpoint.resolve() != checkpoint:
        raise RuntimeError("runtime perception checkpoint path already points elsewhere")
    config.setdefault("perception", {})["perception_model_dir"] = str(perception_dir)
    if not DEFAULT_PLANNER_CHECKPOINT.is_file():
        raise RuntimeError("frozen CoDriving planner checkpoint is missing: %s" % DEFAULT_PLANNER_CHECKPOINT)
    config.setdefault("planning", {})["planner_model_checkpoint"] = str(DEFAULT_PLANNER_CHECKPOINT)
    if not DEFAULT_PLANNING_FEATURE_ADAPTER_CHECKPOINT.is_file():
        raise RuntimeError(
            "pretrained planner feature adapter is missing: %s"
            % DEFAULT_PLANNING_FEATURE_ADAPTER_CHECKPOINT
        )
    config["planning"]["feature_adapter_checkpoint"] = str(
        DEFAULT_PLANNING_FEATURE_ADAPTER_CHECKPOINT
    )
    config["planning"]["feature_adapter_hypes"] = str(
        DEFAULT_SIMULATION_PERCEPTION_CONFIG
    )
    config["planning"]["feature_adapter_source_widths"] = list(
        simulation_model_args["base_bev_backbone"]["num_upsample_filter"]
    )
    config["planning"]["feature_adapter_retained_widths"] = list(
        frozen_model_args["base_bev_backbone"]["num_upsample_filter"]
    )
    simulation = dict(config.get("simulation", {}))
    simulation["skip_frames"] = task["cadence"]
    audit_path = Path(run_dir).resolve() / "evidence" / "fusion_audit.jsonl"
    hil = dict(overlay.get("simulation", {}).get("coptv2x_hil", {}))
    hil.update({"enabled": True, "run_id": cohort_id,
                "route_id": "Town05-route-%d" % task["route"],
                "scenario_id": task["scenario_id"], "arm_id": task["arm_id"],
                "audit_jsonl": str(audit_path),
                "transport": "reverse_ssh_ros_proxy",
                "proxy_listen_host": "127.0.0.1",
                "proxy_listen_port": int(proxy_port),
                "trigger_topic": "/coptv2x/hil/trigger",
                "result_topic": "/ditl/rsu/perception_result",
                "ack_topic": "/ditl/rsu/perception_result_ack"})
    simulation["coptv2x_hil"] = hil
    config["simulation"] = simulation
    path = _runtime_config_path(task, run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(str(temporary), str(path))
    return path


def build_episode_command(task, run_dir, host="127.0.0.1", port=46000,
                          traffic_manager_port=None, cohort_id="coptv2x-hil"):
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint = run_dir / "carla" / "results.json"
    config_path = _runtime_config_path(task, run_dir)
    route = (REPO_ROOT / task["route_xml"]).resolve()
    scenarios = (REPO_ROOT / task["scenarios_json"]).resolve()
    scenario_parameter = REPO_ROOT / (
        "simulation/leaderboard/leaderboard/scenarios/scenario_parameter%s.yaml"
        % task["scenario"])
    tm_port = traffic_manager_port if traffic_manager_port is not None else port + 5
    return [
        sys.executable,
        str((REPO_ROOT / "simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py").resolve()),
        "--host", str(host), "--port", str(port), "--trafficManagerPort", str(tm_port),
        "--trafficManagerSeed", "2000", "--carlaProviderSeed", "2000",
        "--routes", str(route), "--scenarios", str(scenarios),
        "--scenario_parameter", str(scenario_parameter), "--repetitions", "1",
        "--agent", str((REPO_ROOT / "simulation/leaderboard/team_code/pnp_agent_e2e.py").resolve()),
        "--agent-config", str(config_path), "--track", "SENSORS", "--resume", "0",
        "--checkpoint", str(checkpoint), "--ego-num", "1", "--skip_existed", "0",
        "--timeout", "600.0",
    ]


def runner_environment(run_dir, ros_domain_id, port, route_path=None):
    run_dir = Path(run_dir).expanduser().resolve()
    env = dict(os.environ)
    env.update({
        "ROS_DOMAIN_ID": str(ros_domain_id),
        "CARLA_ROOT": str(REPO_ROOT / "external_paths/carla_root"),
        "LEADERBOARD_ROOT": str(REPO_ROOT / "simulation/leaderboard"),
        "SCENARIO_RUNNER_ROOT": str(REPO_ROOT / "simulation/scenario_runner"),
        "DATA_ROOT": str(REPO_ROOT / "external_paths/data_root"),
        "SAVE_PATH": str(run_dir / "images"),
        "RESULT_ROOT": str(run_dir / "results"),
        "CHECKPOINT_ENDPOINT": str(run_dir / "carla/results.json"),
        "PORT": str(port), "TM_PORT": str(port + 5),
        "CARLA_SEED": "2000", "TRAFFIC_SEED": "2000",
        "REPETITIONS": "1", "CHALLENGE_TRACK_CODENAME": "SENSORS",
    })
    if route_path is not None:
        resolved_route = Path(route_path).expanduser().resolve()
        if not resolved_route.is_file():
            raise RuntimeError("frozen CARLA route is missing: %s" % resolved_route)
        env["ROUTES"] = str(resolved_route)
    carla_python_api = REPO_ROOT / "external_paths/carla_root/PythonAPI/carla"
    python_abi_tag = "py%d.%d-" % sys.version_info[:2]
    carla_eggs = sorted(
        egg for egg in glob.glob(str(carla_python_api / "dist" / "carla-*.egg"))
        if python_abi_tag in Path(egg).name
    )
    if not carla_eggs:
        raise RuntimeError(
            "CARLA PythonAPI has no egg matching the active interpreter %s"
            % python_abi_tag[:-1]
        )
    python_path = [str(REPO_ROOT), env.get("PYTHONPATH", ""),
                   str(REPO_ROOT / "simulation/leaderboard"),
                   str(REPO_ROOT / "simulation/leaderboard/team_code"),
                   str(REPO_ROOT / "simulation/scenario_runner"),
                   str(REPO_ROOT / "external_paths/carla_root/PythonAPI"),
                   str(carla_python_api)] + carla_eggs
    env["PYTHONPATH"] = os.pathsep.join(part for part in python_path if part)
    return env


def _read_checkpoint(checkpoint):
    path = Path(checkpoint)
    candidates = (path, path.parent / "ego_vehicle_0" / path.name)
    found_file = False
    corrupt_file = False
    for candidate in candidates:
        if not candidate.is_file():
            continue
        found_file = True
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            checkpoint_data = payload.get("_checkpoint", payload)
            global_record = checkpoint_data.get("global_record", {})
            route_records = checkpoint_data.get("records", [])
            status = global_record.get("status")
            if status == "Completed":
                return "completed", global_record
            if status == "Failed" and route_records:
                route_record = route_records[0]
                infractions = route_record.get("infractions", {})
                if infractions.get("route_timeout"):
                    return "route_timeout", global_record
                return route_record.get("status"), global_record
        except (OSError, ValueError, TypeError, AttributeError):
            corrupt_file = True
    if found_file and corrupt_file:
        return "result_corrupt", None
    return "missing_result", None


def validate_carla_environment(host, port):
    """Read-only TCP liveness probe; it never starts/stops a CARLA server."""
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return {"reachable": True, "host": host, "port": int(port)}
    except OSError as error:
        return {"reachable": False, "host": host, "port": int(port), "error": str(error)}


def _atomic_episode_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)


def _episode_environment(env, task, run_dir, cohort_id, attempt_no):
    return dict(env, COPT_HIL_RUN_DIR=str(Path(run_dir).resolve()),
                COPT_HIL_TASK_ID=task["task_id"], COPT_HIL_ARM_ID=task["arm_id"],
                COPT_HIL_COHORT_ID=cohort_id, COPT_HIL_ATTEMPT=str(attempt_no))


def _run_episode_hook(command, phase, run_dir, env, timeout):
    """Run a scoped argv hook without shell expansion and retain its evidence."""
    if command is None:
        return None
    log_path = Path(run_dir) / "evidence" / ("episode-%s.log" % phase)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"log_path": str(log_path), "started_unix_ns": time.time_ns()}
    with log_path.open("w", encoding="utf-8") as output:
        try:
            argv = shlex.split(command)
            if not argv:
                raise ValueError("episode hook command must not be empty")
            report["command"] = argv
            process = subprocess.run(argv, cwd=str(REPO_ROOT), env=env,
                                     stdout=output, stderr=subprocess.STDOUT,
                                     timeout=timeout)
            report["return_code"] = process.returncode
        except subprocess.TimeoutExpired as error:
            report.update(return_code=124, error=str(error))
        except (OSError, ValueError) as error:
            report.update(return_code=127, error=str(error))
        if report.get("error"):
            output.write(report["error"] + "\n")
    return dict(report, finished_unix_ns=time.time_ns())


def run_cohort(manifest, output_dir, args):
    local_input_manifest = None
    if args.run:
        readiness_path = getattr(args, "local_input_manifest", None)
        local_input_manifest = validate_local_input_manifest(manifest, readiness_path)
        carla = validate_carla_environment(args.host, args.port)
        if not carla["reachable"]:
            raise RuntimeError("CARLA server is not reachable at %s:%s; refusing to launch episodes" %
                               (args.host, args.port))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    run_manifest = dict(manifest)
    if local_input_manifest is not None:
        run_manifest["host_input_index"] = dict(local_input_manifest)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_sha256(existing) != _manifest_sha256(run_manifest):
            raise ManifestMismatchError("output directory already contains a different manifest")
    else:
        _atomic_episode_json(manifest_path, run_manifest)
    run_manifest_sha256 = _manifest_sha256(run_manifest)
    state = RunState(output_dir / "state.json", [task["task_id"] for task in manifest["tasks"]],
                     manifest_sha256=run_manifest_sha256,
                     local_input_manifest=local_input_manifest)
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    for task_id in state.pending_ids():
        task = tasks[task_id]
        while task_id not in state.data["results"]:
            attempt_no = len(state.data["attempts"].get(task_id, [])) + 1
            run_dir = output_dir / "runs" / task_id / ("attempt-%02d" % attempt_no)
            run_dir.mkdir(parents=True, exist_ok=True)
            attempt = {"attempt": attempt_no, "started_unix_ns": time.time_ns(),
                       "run_dir": str(run_dir), "arm_id": task["arm_id"]}
            audit_record = None
            status, code, command = "missing_result", 127, []
            if args.run:
                config_path = prepare_task_config(
                    task, run_dir, output_dir.name, args.ros_domain_id,
                    proxy_port=args.port + 10,
                )
                command = build_episode_command(task, run_dir, args.host, args.port,
                                                cohort_id=output_dir.name)
                command[command.index("--agent-config") + 1] = str(config_path)
                env = runner_environment(
                    run_dir, args.ros_domain_id, args.port,
                    route_path=REPO_ROOT / task["route_xml"],
                )
                env = _episode_environment(env, task, run_dir, output_dir.name, attempt_no)
                Path(env["RESULT_ROOT"]).mkdir(parents=True, exist_ok=True)
                Path(env["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)
                stdout_path = run_dir / "carla" / "evaluator.log"
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                hook_timeout = getattr(args, "episode_hook_timeout", 120)
                try:
                    start = _run_episode_hook(
                        getattr(args, "episode_start_command", None), "start",
                        run_dir, env, hook_timeout,
                    )
                    if start is not None:
                        attempt["episode_start"] = start
                    if start is not None and start["return_code"] != 0:
                        status, code = "episode_start_failed", start["return_code"]
                    else:
                        with stdout_path.open("w", encoding="utf-8") as output:
                            try:
                                process = subprocess.run(command, cwd=str(REPO_ROOT), env=env,
                                                         stdout=output, stderr=subprocess.STDOUT,
                                                         timeout=args.episode_timeout)
                                code = process.returncode
                            except subprocess.TimeoutExpired:
                                status, code = "carla_crash", 124
                            except OSError:
                                status, code = "carla_crash", 127
                finally:
                    stop = _run_episode_hook(
                        getattr(args, "episode_stop_command", None), "stop",
                        run_dir, env, hook_timeout,
                    )
                    if stop is not None:
                        attempt["episode_stop"] = stop
                    if stop is not None and stop["return_code"] != 0:
                        status, code = "episode_stop_failed", stop["return_code"]
                if status == "missing_result":
                    status, carla_summary = _read_checkpoint(run_dir / "carla/results.json")
                else:
                    carla_summary = None
                fusion_audit = run_dir / "evidence/fusion_audit.jsonl"
                if fusion_audit.is_file():
                    try:
                        audit_rows = [json.loads(line) for line in fusion_audit.read_text(encoding="utf-8").splitlines() if line]
                        selected = [row for row in audit_rows
                                    if row.get("selected_result_sequence_no") is not None]
                        if selected:
                            row = selected[-1]
                            audit_record = {
                                "batch_order": row.get("batch_order"),
                                "trigger_identity": row.get("trigger_identity"),
                                "result_identity": row.get("selected_result_trigger_identity"),
                                "artifact_sha256": row.get("artifact_sha256"),
                                "aoi_frames": row.get("rsu_aoi_frames"),
                                "remote_input_tensor_bytes": row.get("remote_input_tensor_bytes", 0),
                            }
                            audit_record = validate_episode_audit(audit_record)
                            validate_task_artifacts(task, audit_record)
                    except (ValueError, TypeError, KeyError):
                        audit_record = None
                status_class = classify_episode(status, code)
                if status_class in ("COMPLETED", "DRIVING_TERMINAL") and audit_record is None:
                    status, status_class = "result_corrupt", "INFRA_RETRY"
                attempt.update({"finished_unix_ns": time.time_ns(), "return_code": code,
                               "status": status, "classification": status_class,
                               "command": command, "carla_result": carla_summary})
            else:
                status_class = "INFRA_RETRY"
                attempt.update({"status": "dry_run", "classification": status_class,
                                "command": build_episode_command(task, run_dir, args.host, args.port)})
            state.record_attempt(task_id, attempt)
            if status_class == "INFRA_RETRY" and attempt_no < args.max_infra_attempts:
                continue
            terminal_class = "INFRA_INVALID" if status_class == "INFRA_RETRY" else status_class
            terminal = dict(attempt, classification=terminal_class,
                            hil_audit=audit_record,
                            artifact_sha256=task["artifact_sha256"],
                            carla_result_path=str(run_dir / "carla/results.json"),
                            fusion_audit_path=str(run_dir / "evidence/fusion_audit.jsonl"))
            state.record(task_id, terminal)
    rows = []
    for task in manifest["tasks"]:
        terminal = state.data["results"].get(task["task_id"])
        if terminal:
            rows.append(paired_row(task["task_id"], task["arm_id"],
                                   terminal["classification"], terminal.get("carla_result")))
    write_paired_csv(output_dir / "paired_results.csv", rows)
    return {"output_dir": str(output_dir), "planned_tasks": len(manifest["tasks"]),
            "terminal_tasks": len(state.data["results"]), "pending_tasks": len(state.pending_ids()),
            "manifest_sha256": file_sha256(manifest_path)}


def _manifest_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_local_input_manifest(cohort_manifest, input_path):
    """Validate a host-visible route/frame NPY index, not Orin deployment readiness."""
    if input_path is None:
        raise RuntimeError("--run requires --host-input-index for host-visible route/frame data")
    path = Path(input_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("local input manifest is unreadable: %s" % error) from error
    if payload.get("schema") != "coptv2x.route-input-host-index.v1" or \
            payload.get("status") != "HOST_INDEX_VALID":
        raise RuntimeError("input index must declare coptv2x.route-input-host-index.v1 status HOST_INDEX_VALID")
    if payload.get("route_scenarios_sha256") != cohort_manifest["route_scenarios_sha256"]:
        raise RuntimeError("local input manifest route authority hash does not match cohort")
    entries = payload.get("route_scenarios")
    if not isinstance(entries, dict) or set(entries) != set(cohort_manifest["route_scenarios"]):
        raise RuntimeError("local input manifest must index all 85 exact route-scenario cells")
    import numpy as np
    verified_files = {}
    for cell, frames in entries.items():
        if not isinstance(frames, list) or not frames:
            raise RuntimeError("local input manifest has no frames for %s" % cell)
        identities = set()
        for frame in frames:
            required = ("source_frame", "sample_id", "spatial_features_path",
                        "spatial_features_sha256", "shape", "dtype")
            if not isinstance(frame, dict) or not all(key in frame for key in required):
                raise RuntimeError("local input frame mapping is incomplete for %s" % cell)
            if type(frame["source_frame"]) is not int or not frame["sample_id"]:
                raise RuntimeError("local input frame identity is invalid for %s" % cell)
            if frame["sample_id"] != "carla-frame-%d" % frame["source_frame"]:
                raise RuntimeError("local input sample_id must match the live CARLA source frame for %s" % cell)
            resolved = Path(frame["spatial_features_path"]).expanduser()
            if not resolved.is_absolute() or not resolved.is_file():
                raise RuntimeError("local input feature path is not a deployed absolute file: %s" % resolved)
            expected_shape = [2, 64, 256, 512]
            if frame["shape"] != expected_shape or frame["dtype"] != "float32":
                raise RuntimeError("local input tensor contract is invalid for %s" % cell)
            expected_hash = frame["spatial_features_sha256"]
            actual_hash = verified_files.get(str(resolved))
            if actual_hash is None:
                actual_hash = file_sha256(resolved)
                verified_files[str(resolved)] = actual_hash
            if actual_hash != expected_hash:
                raise RuntimeError("local input tensor SHA-256 mismatch for %s" % cell)
            try:
                array = np.load(str(resolved), allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as error:
                raise RuntimeError("local input tensor is not a safe NPY for %s" % cell) from error
            if array.dtype.hasobject or list(array.shape) != expected_shape or str(array.dtype) != "float32":
                raise RuntimeError("local input tensor file shape/dtype mismatch for %s" % cell)
            key = (frame["source_frame"], frame["sample_id"])
            if key in identities:
                raise RuntimeError("local input manifest repeats frame identity for %s" % cell)
            identities.add(key)
    return {"path": str(path), "sha256": file_sha256(path),
            "version": payload.get("version", "coptv2x.route-input-host-index.v1"),
            "route_scenarios_sha256": payload["route_scenarios_sha256"],
            "verified_feature_files": len(verified_files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true",
                        help="Launch episodes after host index and CARLA checks; does not verify Orin/ROS readiness")
    parser.add_argument("--pilot-cell", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=46000)
    parser.add_argument("--ros-domain-id", type=int, default=71)
    parser.add_argument("--episode-timeout", type=int, default=1800)
    parser.add_argument("--max-infra-attempts", type=int, default=2)
    parser.add_argument("--episode-start-command", help="argv command to start isolated services before each attempt (no shell)")
    parser.add_argument("--episode-stop-command", help="argv cleanup command, always run after each attempt (no shell)")
    parser.add_argument("--episode-hook-timeout", type=float, default=120,
                        help="timeout in seconds for each episode service hook")
    parser.add_argument("--host-input-index", "--local-input-manifest", dest="local_input_manifest", type=Path,
                        help="host-visible route/frame NPY index (does not prove Orin/ROS deployment)")
    args = parser.parse_args(argv)
    if args.run and args.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    if args.episode_hook_timeout <= 0 or args.episode_timeout <= 0 or args.max_infra_attempts < 1:
        parser.error("episode timeouts and max infrastructure attempts must be positive")
    if args.ros_domain_id < 0 or args.ros_domain_id > 232 or args.port < 1024 or args.port + 5 > 65535:
        parser.error("ROS domain/port is invalid")
    try:
        payload = build_manifest()
        if args.pilot_cell:
            payload = select_pilot(payload, args.pilot_cell)
        if args.dry_run:
            output_dir = args.output_dir.expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_episode_json(output_dir / "manifest.json", payload)
            report = {"output_dir": str(output_dir), "planned_tasks": len(payload["tasks"]),
                      "manifest_sha256": file_sha256(output_dir / "manifest.json"),
                      "dry_run": True}
        elif args.run:
            report = run_cohort(payload, args.output_dir, args)
        else:
            parser.error("select either --dry-run or --run")
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print("CoptV2X HIL runner failed: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
