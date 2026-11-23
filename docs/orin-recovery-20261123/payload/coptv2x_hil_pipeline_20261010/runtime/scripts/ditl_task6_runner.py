#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate and sequentially run the isolated Task 6 closed-loop cohort."""
from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_ROOT = Path("/var/tmp/codriving-ditl")
SHARED_RUNTIME_ROOT = Path("/home/jichengzhi/V2Xverse")
BASELINE = ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_dualfifo_baseline_l1.yaml"
ROUTE_FILE = ROOT / "simulation/leaderboard/data/evaluation_routes/town05_short_r146.xml"
EVAL_ONE = ROOT / "scripts/ditl_task6_eval_one.sh"
TRACE_MODULE_DIR = ROOT / "simulation/leaderboard/team_code/closedloop"
PROTECTED_WORKTREE = Path("/home/jichengzhi/V2X/.worktrees/V2Xverse-exact-age-c2-run")
PROTECTED_SUPERVISOR_PID = 2088364
PROFILE_IDS = (
    "raw_samehost_idle",
    "raw_samehost_gpu_low",
    "raw_samehost_gpu_medium",
    "raw_samehost_gpu_high",
)
LEGACY_LATENCY_FIELDS = (
    "rsu_fifo_delay_ms", "comm_latency", "latency_inject_ms", "latency_trace",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
FORMAL_RENDER_GPU = 6
FORMAL_INFER_GPU = 7
FORMAL_WORLD_PORT = 46000
FORMAL_TRAFFIC_PORT = 46005
MAX_IDLE_UTILIZATION = 5
MAX_IDLE_MEMORY_MIB = 50
PORT_RELEASE_TIMEOUT_S = 120.0
PORT_RELEASE_POLL_S = 0.25


class SignalCleanupGuard(object):
    """Convert INT/TERM to unwinding, deferring only across identity capture."""

    def __init__(self):
        self.previous = {}
        self.deferred = False
        self.pending = None

    def _handle(self, signum, frame):
        if self.deferred:
            self.pending = signum
            return
        for caught in (signal.SIGINT, signal.SIGTERM):
            signal.signal(caught, signal.SIG_IGN)
        raise RuntimeError("Task 6 runner interrupted by signal %s" % signum)

    def __enter__(self):
        for caught in (signal.SIGINT, signal.SIGTERM):
            self.previous[caught] = signal.signal(caught, self._handle)
        return self

    def begin_launch(self):
        self.deferred = True

    def finish_launch(self):
        self.deferred = False
        if self.pending is not None:
            signum, self.pending = self.pending, None
            self._handle(signum, None)

    def __exit__(self, exception_type, exception, traceback):
        for caught, previous in self.previous.items():
            signal.signal(caught, previous)
        return False


def _load_manifest_module():
    path = ROOT / "scripts/ditl_task6_manifest.py"
    spec = importlib.util.spec_from_file_location("ditl_task6_manifest_for_runner", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_status_once(path, value):
    """Publish a terminal arm status exactly once."""
    _write_json_new(path, value)


def _require_regular(path, label):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("%s must be absolute" % label)
    if path.is_symlink() or not path.is_file():
        raise ValueError("%s must be a non-symlink regular file" % label)
    return path


def _reject_symlink_components(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("cohort path contains a symlink component: %s" % current)
        if not current.exists():
            break


def _is_within(path, root):
    return path == root or root in path.parents


def _resolve_runtime_asset(root, value, label, allowed_roots, kind="file", executable=False):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("%s must be a confined relative path" % label)
    candidate = root / value
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("%s is missing or unreadable: %s" % (label, error))
    if not any(_is_within(resolved, allowed) for allowed in allowed_roots):
        raise ValueError("%s resolves outside the allowed runtime roots" % label)
    if kind == "file" and not resolved.is_file():
        raise ValueError("%s must resolve to a regular file" % label)
    if kind == "directory" and not resolved.is_dir():
        raise ValueError("%s must resolve to a directory" % label)
    if executable and not os.access(str(resolved), os.X_OK):
        raise ValueError("%s is not executable" % label)
    return resolved


def validate_runtime_assets(baseline_config, root=ROOT, allowed_roots=None):
    """Resolve every runtime dependency before a formal cohort can be created."""
    try:
        root = Path(root).resolve(strict=True)
        configured_roots = ((root, SHARED_RUNTIME_ROOT) if allowed_roots is None
                            else tuple(Path(path) for path in allowed_roots))
        trusted = tuple(Path(path).resolve(strict=True) for path in configured_roots)
    except OSError as error:
        raise ValueError("allowed runtime root is missing or unreadable: %s" % error)
    if not trusted or any(not path.is_dir() for path in trusted):
        raise ValueError("allowed runtime roots must be existing directories")

    baseline_path = Path(baseline_config)
    try:
        baseline_resolved = baseline_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("baseline config is missing or unreadable: %s" % error)
    if not baseline_resolved.is_file() or not any(
            _is_within(baseline_resolved, allowed) for allowed in trusted):
        raise ValueError("baseline config must be a regular file in an allowed runtime root")
    config = yaml.safe_load(baseline_resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("baseline config must contain a mapping")
    perception = config.get("perception")
    planning = config.get("planning")
    if not isinstance(perception, dict) or not isinstance(planning, dict):
        raise ValueError("baseline config must contain perception and planning mappings")

    launcher = _resolve_runtime_asset(
        root, "external_paths/carla_root/CarlaUE4.sh", "CARLA launcher",
        trusted, executable=True)
    data_root = _resolve_runtime_asset(
        root, "external_paths/data_root", "data root", trusted, kind="directory")
    perception_dir = _resolve_runtime_asset(
        root, perception.get("perception_model_dir"), "perception model directory",
        trusted, kind="directory")
    perception_config = _resolve_runtime_asset(
        root, str(Path(perception.get("perception_model_dir")) / "config.yaml"),
        "perception config", trusted)
    checkpoint_candidates = sorted(perception_dir.glob("net_epoch_bestval_at*.pth"))
    if len(checkpoint_candidates) != 1:
        raise ValueError("perception checkpoint must contain exactly one net_epoch_bestval_at*.pth file")
    perception_checkpoint = _resolve_runtime_asset(
        root, str(Path(perception.get("perception_model_dir")) / checkpoint_candidates[0].name),
        "perception checkpoint", trusted)
    planner_config = _resolve_runtime_asset(
        root, planning.get("planner_config"), "planner config", trusted)
    planner_checkpoint = _resolve_runtime_asset(
        root, planning.get("planner_model_checkpoint"), "planner checkpoint", trusted)
    native_extension = _resolve_runtime_asset(
        root, "opencood/utils/box_overlaps.cpython-37m-x86_64-linux-gnu.so",
        "box_overlaps Python 3.7 extension", trusted)
    return {
        "carla_launcher": launcher,
        "data_root": data_root,
        "perception_model_dir": perception_dir,
        "perception_config": perception_config,
        "perception_checkpoints": [perception_checkpoint],
        "planner_config": planner_config,
        "planner_checkpoint": planner_checkpoint,
        "native_extension": native_extension,
    }


def validate_new_cohort_path(cohort):
    """Accept only a new, direct Task 6 cohort below the authorized root."""
    cohort = Path(cohort)
    if not cohort.is_absolute():
        raise ValueError("cohort path must be absolute")
    if ".." in cohort.parts:
        raise ValueError("cohort path must not contain parent traversal")
    _reject_symlink_components(cohort)
    if cohort.parent != AUTHORIZED_ROOT or not cohort.name.startswith("task6-"):
        raise ValueError("cohort must be a direct task6-* child of %s" % AUTHORIZED_ROOT)
    if cohort.exists() or cohort.is_symlink():
        raise FileExistsError("cohort already exists: %s" % cohort)
    return cohort


def _default_verifier(bundle_path, library_root, catalog_path=None):
    sys.path.insert(0, str(TRACE_MODULE_DIR))
    try:
        from ditl_trace_library import verify_feature_bundle
        return verify_feature_bundle(bundle_path, library_root, catalog_path=catalog_path)
    finally:
        try:
            sys.path.remove(str(TRACE_MODULE_DIR))
        except ValueError:
            pass


def _derive_catalog(report):
    rows = report.get("profile_bundles", {})
    roots = set()
    for row in rows.values():
        try:
            bundle = Path(row["bundle_path"])
            roots.add(bundle.parents[2])
        except (KeyError, IndexError, TypeError):
            raise ValueError("invalid explicit bundle_path in capture report")
    if len(roots) != 1:
        raise ValueError("all explicit bundle paths must share one library root")
    return next(iter(roots)) / "catalog.json"


def resolve_bundle_mapping(report_path, catalog_path=None, verifier=None):
    """Verify the report's exact profile-to-feature mapping without catalog search."""
    report_path = _require_regular(Path(report_path), "capture report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "verified":
        raise ValueError("capture report status must be verified")
    rows = report.get("profile_bundles")
    if not isinstance(rows, dict) or set(rows) != set(PROFILE_IDS):
        raise ValueError("capture report must contain the exact four profile bundle mappings")
    catalog = Path(catalog_path) if catalog_path is not None else _derive_catalog(report)
    catalog = _require_regular(catalog, "catalog")
    library_root = catalog.parent
    if catalog != library_root / "catalog.json":
        raise ValueError("catalog must be library_root/catalog.json")
    verify = verifier or _default_verifier
    result = {}
    for profile in PROFILE_IDS:
        row = rows[profile]
        if not isinstance(row, dict) or set(row) != {
                "feature_id", "bundle_path", "manifest_sha256"}:
            raise ValueError("invalid explicit mapping for %s" % profile)
        feature_id = row["feature_id"]
        manifest_sha = row["manifest_sha256"]
        if not isinstance(feature_id, str) or SHA256.fullmatch(feature_id) is None:
            raise ValueError("invalid feature_id for %s" % profile)
        if not isinstance(manifest_sha, str) or SHA256.fullmatch(manifest_sha) is None:
            raise ValueError("invalid manifest_sha256 for %s" % profile)
        bundle = Path(row["bundle_path"])
        expected = library_root / "v1" / "bundles" / feature_id
        if not bundle.is_absolute() or bundle != expected:
            raise ValueError("bundle_path does not exactly match feature_id for %s" % profile)
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError("bundle_path is missing or symlinked for %s" % profile)
        verified = verify(bundle, library_root, catalog_path=catalog)
        if (verified.feature_id != feature_id or Path(verified.bundle_path) != bundle or
                verified.profile_id != profile):
            raise ValueError("verified feature_id/path/profile differs for %s" % profile)
        manifest_path = bundle / "manifest.json"
        if manifest_path.is_symlink() or sha_file(manifest_path) != manifest_sha:
            raise ValueError("manifest hash differs for %s" % profile)
        accepted = verified.manifest.get("accepted_count")
        if isinstance(accepted, bool) or not isinstance(accepted, int) or accepted < 100:
            raise ValueError("verified bundle has fewer than 100 accepted samples")
        result[profile] = {
            "feature_id": feature_id,
            "bundle_path": str(bundle),
            "manifest_sha256": manifest_sha,
            "library_root": str(library_root),
            "catalog_path": str(catalog),
        }
    return result


def _replay_config(bundle, drop_rate):
    return {
        "library_root": bundle["library_root"],
        "catalog": bundle["catalog_path"],
        "bundle": bundle["bundle_path"],
        "exhaustion": "cycle",
        "seed": 2000,
        "ttl_ms": None,
        "deadline_ms": None,
        "max_queue_events": 4096,
        "impairment": {
            "extra_delay_ms": 0,
            "jitter_ms": 0,
            "drop_rate": drop_rate,
            "burst": {"enabled": False, "enter_rate": 0, "exit_rate": 0},
            "reorder_rate": 0,
            "reorder_extra_ms": 0,
            "duplicate_rate": 0,
        },
    }


def _command_argv(config_path, result_root, arm_id, render_gpu, infer_gpu,
                  world_port, traffic_port):
    cohort = Path(result_root).parents[1]
    return [
        str(EVAL_ONE),
        "--config", str(config_path),
        "--result-root", str(result_root),
        "--route", "146",
        "--scenario", "_1",
        "--repeat", "0",
        "--carla-seed", "2000",
        "--traffic-seed", "2000",
        "--render-gpu", str(render_gpu),
        "--infer-gpu", str(infer_gpu),
        "--world-port", str(world_port),
        "--traffic-port", str(traffic_port),
        "--process-dir", str(cohort / "processes" / arm_id),
        "--log-dir", str(cohort / "logs" / arm_id),
    ]


def _model_paths(config):
    perception = config.get("perception", {})
    planning = config.get("planning", {})
    return {
        "perception_model_dir": perception.get("perception_model_dir"),
        "planner_config": planning.get("planner_config"),
        "planner_model_checkpoint": planning.get("planner_model_checkpoint"),
    }


def generate_cohort(cohort, baseline_config, bundles, catalog_path, route_file,
                    source_commit, render_gpu, infer_gpu, world_port,
                    protected_baseline):
    """Create all immutable inputs and empty output roots in a new cohort."""
    cohort = validate_new_cohort_path(cohort)
    baseline_path = _require_regular(Path(baseline_config), "baseline config")
    route_path = _require_regular(Path(route_file), "route file")
    catalog_path = _require_regular(Path(catalog_path), "catalog")
    if not isinstance(source_commit, str) or COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase commit ID")
    if set(bundles) != set(PROFILE_IDS):
        raise ValueError("bundles must contain the exact four profile mappings")
    validate_resource_coordinates(render_gpu, infer_gpu, world_port, world_port + 5)
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(baseline.get("simulation"), dict):
        raise ValueError("baseline YAML must contain a simulation mapping")
    task6 = _load_manifest_module()
    arms = task6.validate_task6_arms(task6.build_task6_arms())
    route_sha = sha_file(route_path)
    os.mkdir(str(cohort), 0o700)
    for name in ("configs", "results", "logs", "processes", "status", "snapshots"):
        (cohort / name).mkdir(mode=0o700)
    records = []
    for index, arm in enumerate(arms):
        arm_config = copy.deepcopy(baseline)
        simulation = arm_config["simulation"]
        for field in LEGACY_LATENCY_FIELDS:
            simulation.pop(field, None)
        bundle = bundles[arm.trace_profile_id]
        simulation["ditl_replay"] = _replay_config(bundle, arm.drop_rate)
        if arm.family == "spatial":
            simulation["ditl_spatial_offset"] = {
                "dx_m": arm.spatial_dx_m,
                "dy_m": arm.spatial_dy_m,
                "dyaw_deg": arm.spatial_dyaw_deg,
            }
        else:
            simulation.pop("ditl_spatial_offset", None)
        config_path = cohort / "configs" / ("%02d_%s.yaml" % (index, arm.arm_id))
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(arm_config, stream, default_flow_style=False, sort_keys=False)
        result_root = cohort / "results" / arm.arm_id
        result_root.mkdir(mode=0o700)
        (cohort / "logs" / arm.arm_id).mkdir(mode=0o700)
        (cohort / "processes" / arm.arm_id).mkdir(mode=0o700)
        command = _command_argv(
            config_path, result_root, arm.arm_id, render_gpu, infer_gpu,
            world_port, world_port + 5,
        )
        records.append({
            "sequence": index,
            "arm_id": arm.arm_id,
            "contract": dict(vars(arm)),
            "config_path": str(config_path),
            "config_sha256": sha_file(config_path),
            "bundle_feature_id": bundle["feature_id"],
            "bundle_path": bundle["bundle_path"],
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "route_file": str(route_path),
            "route_file_sha256": route_sha,
            "model_paths": _model_paths(arm_config),
            "seeds": {"carla": 2000, "traffic": 2000, "replay": 2000},
            "gpus": {"render": render_gpu, "inference": infer_gpu},
            "ports": {"world": world_port, "traffic": world_port + 5},
            "result_root": str(result_root),
            "command_argv": command,
        })
        config_path.chmod(0o400)
    manifest = {
        "schema": "ditl.task6-cohort.v1",
        "source_commit": source_commit,
        "source_worktree": str(ROOT),
        "baseline_config": {"path": str(baseline_path), "sha256": sha_file(baseline_path)},
        "catalog": {"path": str(catalog_path), "sha256": sha_file(catalog_path)},
        "route": {"id": 146, "scenario_suffix": "_1", "repeat": 0,
                  "path": str(route_path), "sha256": route_sha},
        "protected_baseline": copy.deepcopy(protected_baseline),
        "arms": records,
    }
    manifest_path = cohort / "manifest.json"
    _write_json_new(manifest_path, manifest)
    manifest_path.chmod(0o400)
    return manifest


def validate_resource_coordinates(render_gpu, infer_gpu, world_port, traffic_port):
    if render_gpu == infer_gpu:
        raise ValueError("render and inference GPUs must be different")
    if (render_gpu, infer_gpu) != (FORMAL_RENDER_GPU, FORMAL_INFER_GPU):
        raise ValueError("only formal GPUs 6 and 7 are reserved for Task 6")
    if (world_port, traffic_port) != (FORMAL_WORLD_PORT, FORMAL_TRAFFIC_PORT):
        raise ValueError("requested ports are reserved/protected; use 46000 and 46005")


def validate_resource_snapshot(snapshot, render_gpu, infer_gpu, world_port, traffic_port):
    validate_resource_coordinates(render_gpu, infer_gpu, world_port, traffic_port)
    if not isinstance(snapshot, dict):
        raise ValueError("resource snapshot is missing")
    listening = snapshot.get("listening_ports")
    if not isinstance(listening, list):
        raise ValueError("listening port snapshot is invalid")
    required_ports = (world_port, world_port + 1, world_port + 2, traffic_port)
    if any(port in listening for port in required_ports):
        raise ValueError("requested port is occupied")
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, dict):
        raise ValueError("GPU snapshot is missing")
    for gpu in (render_gpu, infer_gpu):
        row = gpus.get(str(gpu))
        if not isinstance(row, dict):
            raise ValueError("GPU %s snapshot is missing" % gpu)
        util = row.get("utilization_percent")
        memory = row.get("memory_used_mib")
        pids = row.get("compute_pids")
        if (isinstance(util, bool) or not isinstance(util, (int, float)) or
                isinstance(memory, bool) or not isinstance(memory, (int, float)) or
                not isinstance(pids, list)):
            raise ValueError("GPU %s snapshot is invalid" % gpu)
        if util > MAX_IDLE_UTILIZATION or memory > MAX_IDLE_MEMORY_MIB or pids:
            raise ValueError("GPU %s is busy" % gpu)


def bind_guard_ports(world_port, traffic_port, socket_factory=socket.socket):
    """Prove CARLA world/streaming and traffic ports bindable."""
    probes = []
    try:
        for port in (world_port, world_port + 1, world_port + 2, traffic_port):
            for family, address in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
                probe = socket_factory(family, socket.SOCK_STREAM)
                probes.append(probe)
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind((address, port))
    except OSError as error:
        raise RuntimeError("formal Task 6 port bind guard failed: %s" % error)
    finally:
        for probe in probes:
            probe.close()


def wait_for_ports_bindable(world_port, traffic_port,
                            timeout_s=PORT_RELEASE_TIMEOUT_S,
                            poll_s=PORT_RELEASE_POLL_S, bind_guard=None,
                            listening_ports=None, monotonic=time.monotonic,
                            pause=time.sleep):
    """Poll the exact CARLA port set until release, without taking ownership."""
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("port release timeout and poll interval must be positive")
    check = bind_guard or bind_guard_ports
    listen = listening_ports or _listening_ports
    ports = (world_port, world_port + 1, world_port + 2, traffic_port)
    deadline = monotonic() + timeout_s
    last_error = None
    while True:
        try:
            check(world_port, traffic_port)
            return
        except RuntimeError as error:
            last_error = error
        occupied = sorted(set(ports).intersection(listen()))
        if occupied:
            raise RuntimeError(
                "formal Task 6 ports %s have foreign or unowned listeners on %s" %
                (",".join(str(port) for port in ports),
                 ",".join(str(port) for port in occupied)))
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "formal Task 6 ports %s did not become bindable: %s" %
                (",".join(str(port) for port in ports), last_error))
        pause(min(poll_s, remaining))


def _load_dual_fifo_result():
    path = ROOT / "scripts/dual_fifo_result.py"
    spec = importlib.util.spec_from_file_location("ditl_task6_runner_result", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_arm_result(row):
    contract = row.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Task 6 arm contract is missing")
    route_id = contract.get("route_id")
    repeat = contract.get("repeat")
    if type(route_id) is not int or route_id < 0 or type(repeat) is not int or repeat < 0:
        raise RuntimeError("Task 6 arm route/repeat identity is invalid")
    result_root = Path(row.get("result_root", ""))
    if result_root.is_symlink() or not result_root.is_dir():
        raise RuntimeError("Task 6 arm result root is invalid")
    relative = Path(
        "v2x_final/town05_short_collab/r%d_repeat%d/ego_vehicle_0/results.json" %
        (route_id, repeat))
    return result_root, relative, "RouteScenario_%d" % route_id


def require_empty_arm_result_root(row):
    """Reject output planted before the owned evaluator starts."""
    result_root = _expected_arm_result(row)[0]
    if any(result_root.iterdir()):
        raise RuntimeError("pre-existing result output for Task 6 arm %s" % row.get("arm_id"))


def validate_arm_result(row):
    """Accept only this arm's canonical, countable Leaderboard result."""
    result_root, relative, expected_route = _expected_arm_result(row)
    canonical = result_root / relative
    matches = tuple(result_root.glob("v2x_final/**/ego_vehicle_*/results.json"))
    if len(matches) != 1 or matches[0] != canonical or canonical.is_symlink() or \
            not canonical.is_file():
        raise RuntimeError("canonical result is missing, duplicated, or has wrong identity")
    try:
        checkpoint = json.loads(canonical.read_text(encoding="utf-8"))["_checkpoint"]
        global_record = checkpoint["global_record"]
        records = checkpoint["records"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError("canonical result checkpoint is malformed: %s" % error)
    if not isinstance(global_record, dict) or type(global_record.get("route_id")) is not int or \
            global_record.get("route_id") != -1:
        raise RuntimeError("canonical result global route identity mismatch")
    if not isinstance(records, list) or len(records) != 1 or \
            not isinstance(records[0], dict) or records[0].get("route_id") != expected_route:
        raise RuntimeError("canonical result route identity mismatch")
    metadata = global_record.get("meta")
    if not isinstance(metadata, dict):
        raise RuntimeError("canonical result metadata is malformed")
    exceptions = metadata.get("exceptions", [])
    if not isinstance(exceptions, list) or any(
            not isinstance(item, (list, tuple)) or len(item) != 3 or
            item[0] != expected_route for item in exceptions):
        raise RuntimeError("canonical result exception route identity mismatch")
    parsed = _load_dual_fifo_result().parse_route_result(str(canonical))
    if not parsed.get("countable"):
        raise RuntimeError("canonical result is non-countable: %s" % parsed.get("error"))
    return canonical


def _git_snapshot(worktree):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(worktree),
        stderr=subprocess.STDOUT, universal_newlines=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=str(worktree), stderr=subprocess.STDOUT, universal_newlines=True,
    )
    return {"head": head, "status": status}


def inspect_process(pid):
    raw = (Path("/proc") / str(int(pid)) / "stat").read_text(encoding="utf-8")
    end = raw.rfind(")")
    if end < 0:
        raise RuntimeError("invalid process stat")
    fields = raw[end + 2:].split()
    return {
        "pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]),
        "sid": int(fields[3]), "state": fields[0], "start_token": fields[19],
    }


def _supervisor_snapshot(pid):
    try:
        row = inspect_process(pid)
        return {"pid": int(pid), "active": row["state"] != "Z",
                "start_token": row["start_token"]}
    except (OSError, ValueError, RuntimeError):
        return {"pid": int(pid), "active": False, "start_token": None}


def _listening_ports():
    result = set()
    for table in ("tcp", "tcp6"):
        for line in (Path("/proc/net") / table).read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) > 3 and fields[3] == "0A":
                result.add(int(fields[1].rsplit(":", 1)[1], 16))
    return sorted(result)


def _gpu_snapshot(gpu):
    summary = subprocess.check_output([
        "nvidia-smi", "--id=%s" % gpu,
        "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
    ], stderr=subprocess.STDOUT, universal_newlines=True).strip().splitlines()
    if len(summary) != 1:
        raise RuntimeError("unreadable nvidia-smi summary for GPU %s" % gpu)
    fields = [item.strip() for item in summary[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError("unreadable nvidia-smi summary for GPU %s" % gpu)
    apps = subprocess.check_output([
        "nvidia-smi", "--id=%s" % gpu, "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], stderr=subprocess.STDOUT, universal_newlines=True).strip()
    pids = [] if not apps or apps == "No running processes found" else [
        int(line.strip()) for line in apps.splitlines() if line.strip().isdigit()
    ]
    return {"utilization_percent": int(fields[0]), "memory_used_mib": int(fields[1]),
            "compute_pids": pids}


def capture_isolation_snapshot(render_gpu, infer_gpu):
    protected = _git_snapshot(PROTECTED_WORKTREE)
    protected["supervisor"] = _supervisor_snapshot(PROTECTED_SUPERVISOR_PID)
    process_tree = subprocess.check_output([
        "ps", "-eo", "pid=,ppid=,pgid=,sid=,stat=,args="
    ], stderr=subprocess.STDOUT, universal_newlines=True).splitlines()
    return {
        "protected": protected,
        "source": _git_snapshot(ROOT),
        "process_tree": process_tree,
        "listening_ports": _listening_ports(),
        "gpus": {str(gpu): _gpu_snapshot(gpu) for gpu in (render_gpu, infer_gpu)},
    }


def assert_protected_unchanged(expected, actual):
    for field in ("head", "status", "supervisor"):
        if expected.get(field) != actual.get(field):
            raise RuntimeError("protected worktree/supervisor state changed: %s" % field)


def _identity_matches(record, current):
    return all(current.get(field) == record.get(field)
               for field in ("pid", "pgid", "sid", "start_token")) and (
                   record.get("pid") == record.get("pgid") == record.get("sid"))


def terminate_recorded_groups(records, inspect=inspect_process, signal_group=os.killpg,
                              pause=time.sleep):
    """Signal only identity-matching, session-led groups after validating them all."""
    validated = []
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("invalid recorded process identity")
        try:
            current = inspect(record.get("pid"))
        except (OSError, ProcessLookupError, FileNotFoundError):
            continue
        if current.get("state") == "Z":
            continue
        if not _identity_matches(record, current):
            raise RuntimeError("recorded process identity is stale or foreign")
        validated.append(record)
    for sig, delay in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 0.0)):
        for record in validated:
            try:
                current = inspect(record["pid"])
            except (OSError, ProcessLookupError, FileNotFoundError):
                continue
            if not _identity_matches(record, current):
                if sig == signal.SIGTERM:
                    raise RuntimeError("recorded process identity changed before signal")
                continue
            try:
                signal_group(record["pgid"], sig)
            except ProcessLookupError:
                pass
        if delay:
            pause(delay)


def _read_identity_records(directory):
    records = []
    for path in sorted(Path(directory).glob("*.identity.json")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("invalid process identity file")
        record = json.loads(path.read_text(encoding="utf-8"))
        if "pgrp" in record or "session" in record:
            record = dict(record, pgid=record.get("pgrp"), sid=record.get("session"),
                          owner_pid=record.get("ppid"))
        records.append(record)
    return records


def _owned_process_record(process):
    current = inspect_process(process.pid)
    if current["pid"] != current["pgid"] or current["pid"] != current["sid"]:
        raise RuntimeError("runner child is not an isolated process-group leader")
    return current


def _fixture_isolation_snapshot(path):
    path = _require_regular(Path(path), "isolation snapshot fixture")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
            field in value for field in ("protected", "source", "process_tree",
                                         "listening_ports", "gpus")):
        raise ValueError("isolation snapshot fixture is incomplete")
    return value


def _discover_capture_report():
    matches = sorted(AUTHORIZED_ROOT.glob("task6-orin-*/orin_capture_report.json"))
    if len(matches) != 1:
        raise ValueError("specify --capture-report; expected exactly one verified Orin report")
    return matches[0]


def execute(args, emit=print):
    traffic_port = args.world_port + 5
    validate_resource_coordinates(args.render_gpu, args.infer_gpu, args.world_port, traffic_port)
    validate_runtime_assets(args.baseline)
    report = args.capture_report if args.capture_report is not None else _discover_capture_report()
    bundles = resolve_bundle_mapping(report, args.catalog)
    catalog = Path(next(iter(bundles.values()))["catalog_path"])
    snapshot_fixture = getattr(args, "isolation_snapshot", None)
    if snapshot_fixture is not None and not args.dry_run:
        raise ValueError("--isolation-snapshot is permitted only with --dry-run")
    initial = (_fixture_isolation_snapshot(snapshot_fixture) if snapshot_fixture is not None
               else capture_isolation_snapshot(args.render_gpu, args.infer_gpu))
    validate_resource_snapshot(initial, args.render_gpu, args.infer_gpu,
                               args.world_port, traffic_port)
    source = initial["source"]
    manifest = generate_cohort(
        args.cohort, args.baseline, bundles, catalog, args.route_file,
        source_commit=source["head"], render_gpu=args.render_gpu,
        infer_gpu=args.infer_gpu, world_port=args.world_port,
        protected_baseline=initial["protected"],
    )
    if args.dry_run:
        for row in manifest["arms"]:
            emit(json.dumps({"sequence": row["sequence"], "arm_id": row["arm_id"],
                             "argv": row["command_argv"]}, sort_keys=True))
        return 0
    cohort = Path(args.cohort)
    with SignalCleanupGuard() as signals:
        for row in manifest["arms"]:
            before = capture_isolation_snapshot(args.render_gpu, args.infer_gpu)
            assert_protected_unchanged(initial["protected"], before["protected"])
            validate_resource_snapshot(before, args.render_gpu, args.infer_gpu,
                                       args.world_port, traffic_port)
            bind_guard_ports(args.world_port, traffic_port)
            require_empty_arm_result_root(row)
            _write_json_new(cohort / "snapshots" / (row["arm_id"] + ".before.json"), before)
            process_dir = cohort / "processes" / row["arm_id"]
            process = None
            owner_record = None
            returncode = 1
            cleanup_error = None
            execution_error = None
            try:
                signals.begin_launch()
                process = subprocess.Popen(row["command_argv"], cwd=str(ROOT), start_new_session=True)
                owner_record = _owned_process_record(process)
                signals.finish_launch()
                returncode = process.wait()
            except BaseException as error:
                execution_error = error
            finally:
                signals.deferred = False
                try:
                    records = _read_identity_records(process_dir)
                    if process is not None and process.poll() is None:
                        if owner_record is None:
                            raise RuntimeError("runner child identity was not recorded")
                        records = [owner_record] + records
                    terminate_recorded_groups(records)
                except RuntimeError as error:
                    cleanup_error = str(error)
            result_validation_error = None
            validated_result = None
            if returncode == 0 and cleanup_error is None and execution_error is None:
                try:
                    validated_result = str(validate_arm_result(row))
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                    result_validation_error = str(error)
            postflight_error = None
            try:
                if process is not None:
                    wait_for_ports_bindable(args.world_port, traffic_port)
                after = capture_isolation_snapshot(args.render_gpu, args.infer_gpu)
                _write_json_new(cohort / "snapshots" / (row["arm_id"] + ".after.json"), after)
                assert_protected_unchanged(initial["protected"], after["protected"])
                validate_resource_snapshot(after, args.render_gpu, args.infer_gpu,
                                           args.world_port, traffic_port)
            except BaseException as error:
                postflight_error = error
            status = ("completed" if returncode == 0 and cleanup_error is None and
                      execution_error is None and result_validation_error is None and
                      postflight_error is None else "failed")
            write_status_once(cohort / "status" / (row["arm_id"] + ".json"), {
                "schema": "ditl.task6-arm-status.v1", "arm_id": row["arm_id"],
                "status": status, "returncode": returncode, "cleanup_error": cleanup_error,
                "execution_error": None if execution_error is None else str(execution_error),
                "result_validation_error": result_validation_error,
                "validated_result": validated_result,
                "postflight_error": None if postflight_error is None else str(postflight_error),
            })
            if status != "completed":
                raise RuntimeError("Task 6 arm failed and will not be replaced: %s" % row["arm_id"])
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--capture-report", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--route-file", type=Path, default=ROUTE_FILE)
    parser.add_argument("--render-gpu", type=int, required=True)
    parser.add_argument("--infer-gpu", type=int, required=True)
    parser.add_argument("--world-port", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isolation-snapshot", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return execute(parse_args(argv))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, yaml.YAMLError) as error:
        print("[ABORT] %s" % error, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
