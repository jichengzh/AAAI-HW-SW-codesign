#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the frozen Deployment-in-the-Loop packet-loss stage-one cohort."""
from __future__ import print_function

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ditl_packet_loss_manifest import (build_batches, build_partition,
    build_stage1_manifest, canonical_json_bytes, validate_stage1_manifest)
from scripts.ditl_packet_loss_result import (classify_task_evidence,
    finalize_task_results, generate_artifacts, plan_retry_task_keys)

AUTHORIZED_ROOT = Path("/var/tmp/codriving-ditl")
DATA_OUTPUT_ROOT = Path("/data/jichengzhi_v2x/ditl")
AUTHORIZED_OUTPUT_ROOTS = frozenset((AUTHORIZED_ROOT, DATA_OUTPUT_ROOT))
PROTECTED_WORKTREE = Path("/home/jichengzhi/V2X/.worktrees/V2Xverse-exact-age-c2-run")
PROTECTED_SUPERVISOR_PID = 2088364
DEFAULT_CAPTURE_REPORT = AUTHORIZED_ROOT / "task6-orin-20260924-gpu-m2048-ack1/orin_capture_report.json"
EXPECTED_CAPTURE_REPORT_SHA256 = "e3da5b1e9295bdc02ce5580502cf5276893ba2b77f20235952e6ff90031a7782"
EXPECTED_IDLE_FEATURE_ID = "2825d39faf158f8ca420be7a0288b83ad9f731749372b8d51f237dd94b4ac973"
EXPECTED_IDLE_MANIFEST_SHA256 = "a293c8c145a02e846444f9e44150bead760e00a9a54f6c34fd67d66cc93e6507"
COHORT_RE = re.compile(r"^packet-loss-stage1-[0-9]{8}T[0-9]{6}Z-[a-z0-9][a-z0-9-]{0,31}$")
IDLE_PROFILE = "raw_samehost_idle"
EVAL_BATCH = ROOT / "scripts/ditl_packet_loss_eval_batch.sh"
PROFILE_CONFIGS = {
    "8_10": ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_8_10_fifo0_l1.yaml",
    "5_10": ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_5_10_fifo0_l1.yaml",
    "5_50": ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_5_50_fifo0_l1.yaml",
}
FIRST_ASSIGNMENTS = (
    {"id": "worker6", "gpu": 6, "ports": (62000, 62001, 62002, 62005), "round": "first"},
    {"id": "worker7", "gpu": 7, "ports": (62100, 62101, 62102, 62105), "round": "first"},
)
RETRY_ASSIGNMENTS = (
    {"id": "worker6", "gpu": 6, "ports": (63000, 63001, 63002, 63005), "round": "retry"},
    {"id": "worker7", "gpu": 7, "ports": (63100, 63101, 63102, 63105), "round": "retry"},
)
ALL_PORTS = tuple(p for row in FIRST_ASSIGNMENTS + RETRY_ASSIGNMENTS for p in row["ports"])
PORT_RELEASE_TIMEOUT_S = 120.0
PORT_RELEASE_POLL_S = 0.25
PROTECTED_PIDS = frozenset((PROTECTED_SUPERVISOR_PID,))
_STOP = threading.Event()


class RunInterrupted(RuntimeError):
    """Abort the two-round workflow after an owned signal cleanup."""

    def __init__(self, signum):
        super(RunInterrupted, self).__init__("run interrupted by signal %d" % signum)
        self.signum = int(signum)


class RoundCancelled(RuntimeError):
    """Raised when a peer failure closes the shared round launch gate."""


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value):
    return value.replace(":", "__").replace("/", "_")


def _canonical_write_new(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o400)


def _atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(tmp), str(path))


def _append_event(path, value):
    payload = canonical_json_bytes(value)
    with Path(path).open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _reject_symlink_components(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("cohort path contains a symlink component: %s" % current)
        if not current.exists():
            break


def validate_new_cohort_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("cohort path must be absolute without traversal")
    _reject_symlink_components(path)
    if (path.parent not in AUTHORIZED_OUTPUT_ROOTS or
            COHORT_RE.fullmatch(path.name) is None):
        raise ValueError(
            "cohort must be a correctly named direct child of an authorized output root")
    if path.exists() or path.is_symlink():
        raise FileExistsError("cohort already exists: %s" % path)
    return path


def _load_yaml(source):
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as stream:
            source = yaml.safe_load(stream)
    if not isinstance(source, dict):
        raise ValueError("profile config must be a mapping")
    return copy.deepcopy(source)


def build_task_config(profile_config, idle_bundle, drop_rate):
    config = _load_yaml(profile_config)
    simulation = config.setdefault("simulation", {})
    for field in ("comm_latency", "rsu_fifo_delay_ms", "latency_inject_ms", "latency_trace"):
        simulation.pop(field, None)
    simulation["skip_frames"] = 2
    simulation["action_fifo_delay_ms"] = 0
    simulation.pop("ditl_spatial_offset", None)
    simulation["ditl_replay"] = {
        "library_root": idle_bundle["library_root"],
        "catalog": idle_bundle["catalog_path"],
        "bundle": idle_bundle["bundle_path"],
        "exhaustion": "cycle", "seed": 2000, "ttl_ms": None,
        "deadline_ms": None, "max_queue_events": 4096,
        "impairment": {"extra_delay_ms": 0, "jitter_ms": 0,
                       "drop_rate": float(drop_rate),
                       "burst": {"enabled": False, "enter_rate": 0, "exit_rate": 0},
                       "reorder_rate": 0, "reorder_extra_ms": 0,
                       "duplicate_rate": 0},
    }
    return config


def build_config_payloads(manifest, idle_bundle):
    payloads = {}
    for task in manifest["tasks"]:
        config = build_task_config(PROFILE_CONFIGS[task["profile"]], idle_bundle,
                                   task["drop_rate"])
        payloads[task["key"]] = yaml.safe_dump(
            config, default_flow_style=False, sort_keys=False).encode("utf-8")
    return payloads


def inspect_process(pid):
    raw = (Path("/proc") / str(int(pid)) / "stat").read_text(encoding="utf-8")
    end = raw.rfind(")")
    if end < 0: raise RuntimeError("invalid process stat")
    fields = raw[end + 2:].split()
    return {"pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]),
            "sid": int(fields[3]), "state": fields[0], "start_token": fields[19]}


def _supervisor_snapshot(pid):
    try:
        row = inspect_process(pid)
        return {"pid": int(pid), "active": row["state"] != "Z",
                "start_token": row["start_token"]}
    except (OSError, ValueError, RuntimeError):
        return {"pid": int(pid), "active": False, "start_token": None}


def _git_snapshot(worktree):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(worktree),
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1",
                                      "--untracked-files=all"], cwd=str(worktree),
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
    return {"head": head, "status": status}


def _listening_ports():
    ports = set()
    for table in ("tcp", "tcp6"):
        lines = (Path("/proc/net") / table).read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            fields = line.split()
            if len(fields) > 3 and fields[3] == "0A":
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def _gpu_snapshot(gpu):
    summary = subprocess.check_output([
        "nvidia-smi", "--id=%d" % gpu,
        "--query-gpu=index,uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"], stderr=subprocess.STDOUT,
        universal_newlines=True).strip().splitlines()
    if len(summary) != 1: raise RuntimeError("unreadable GPU summary")
    values = [item.strip() for item in summary[0].split(",")]
    if len(values) != 4 or int(values[0]) != gpu:
        raise RuntimeError("GPU coordinate mismatch")
    apps = subprocess.check_output([
        "nvidia-smi", "--id=%d" % gpu,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], stderr=subprocess.STDOUT,
        universal_newlines=True).strip()
    rows = []
    if apps and apps != "No running processes found":
        for line in apps.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 3 or not parts[0].isdigit() or not parts[2].isdigit():
                raise RuntimeError("unreadable GPU compute process")
            rows.append({"pid": int(parts[0]), "process_name": parts[1],
                         "used_memory_mib": int(parts[2])})
    return {"index": int(values[0]), "uuid": values[1],
            "utilization_percent": int(values[2]), "memory_used_mib": int(values[3]),
            "compute_pids": [row["pid"] for row in rows], "compute_processes": rows}


def capture_resource_snapshot(gpus, ports):
    protected = _git_snapshot(PROTECTED_WORKTREE)
    protected["supervisor"] = _supervisor_snapshot(PROTECTED_SUPERVISOR_PID)
    return {"schema": "ditl.packet-loss-stage1-resource-snapshot.v1",
            "captured_at_unix_ns": int(time.time() * 1000000000),
            "source": _git_snapshot(ROOT), "protected": protected,
            "gpus": {str(gpu): _gpu_snapshot(gpu) for gpu in gpus},
            "listening_ports": _listening_ports(), "checked_ports": list(ports)}


def validate_resource_snapshot(snapshot, gpus, ports):
    if not isinstance(snapshot, dict): raise ValueError("resource snapshot is missing")
    protected = snapshot.get("protected")
    supervisor = protected.get("supervisor") if isinstance(protected, dict) else None
    supervisor_active = (isinstance(supervisor, dict) and
                         supervisor.get("active") is True)
    supervisor_inactive = (isinstance(supervisor, dict) and
                           supervisor.get("active") is False)
    supervisor_identity_valid = (
        (supervisor_active and
         isinstance(supervisor.get("start_token"), str) and
         bool(supervisor["start_token"])) or
        (supervisor_inactive and supervisor.get("start_token") is None))
    if (not isinstance(protected, dict) or not isinstance(protected.get("head"), str) or
            not isinstance(protected.get("status"), str) or
            not isinstance(supervisor, dict) or
            supervisor.get("pid") != PROTECTED_SUPERVISOR_PID or
            not supervisor_identity_valid):
        raise ValueError("protected exact-age snapshot is incomplete or inconsistent")
    rows = snapshot.get("gpus")
    if not isinstance(rows, dict): raise ValueError("GPU snapshot is missing")
    required = {"index", "uuid", "utilization_percent", "memory_used_mib",
                "compute_pids", "compute_processes"}
    for gpu in gpus:
        row = rows.get(str(gpu))
        if not isinstance(row, dict) or not required.issubset(row) or row["index"] != gpu:
            raise ValueError("GPU %d snapshot is incomplete" % gpu)
        if not isinstance(row["compute_pids"], list) or not isinstance(row["compute_processes"], list):
            raise ValueError("GPU %d compute process query is unreliable" % gpu)
    listening = snapshot.get("listening_ports")
    if not isinstance(listening, list): raise ValueError("listening port snapshot is invalid")
    occupied = sorted(set(ports).intersection(listening))
    if occupied: raise ValueError("packet-loss ports are occupied: %s" % occupied)
    return snapshot


def bind_guard_ports(ports, socket_factory=socket.socket):
    probes = []
    try:
        for port in ports:
            for family, address in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
                probe = socket_factory(family, socket.SOCK_STREAM); probes.append(probe)
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind((address, int(port)))
    except OSError as error:
        raise RuntimeError("packet-loss port bind guard failed: %s" % error) from error
    finally:
        for probe in probes: probe.close()


def wait_for_ports_bindable(ports, timeout_s=PORT_RELEASE_TIMEOUT_S,
                            poll_s=PORT_RELEASE_POLL_S,
                            socket_factory=socket.socket,
                            monotonic=time.monotonic, pause=time.sleep,
                            stop_event=_STOP):
    """Wait for kernel port release without masking external ownership."""
    ports = tuple(ports)
    if (not ports or any(isinstance(port, bool) or not isinstance(port, int) or
                         port <= 0 or port > 65535 for port in ports)):
        raise ValueError("packet-loss ports must be valid TCP port integers")
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("port release timeout and poll interval must be positive")
    deadline = monotonic() + timeout_s
    while True:
        if stop_event is not None and stop_event.is_set():
            raise RoundCancelled("packet-loss round cancelled during port wait")
        try:
            bind_guard_ports(ports, socket_factory=socket_factory)
            return
        except RuntimeError as error:
            cause = error.__cause__
            if not isinstance(cause, OSError) or cause.errno != errno.EADDRINUSE:
                raise
            last_error = error
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "packet-loss ports %s did not become bindable within %gs: %s" %
                (",".join(str(port) for port in ports), timeout_s, last_error))
        pause(min(poll_s, remaining))
        if stop_event is not None and stop_event.is_set():
            raise RoundCancelled("packet-loss round cancelled during port wait")


def validate_launch_coordinates(round_name, worker_id, gpu, ports):
    assignments = FIRST_ASSIGNMENTS if round_name == "first" else (
        RETRY_ASSIGNMENTS if round_name == "retry" else ())
    expected = next((row for row in assignments if row["id"] == worker_id), None)
    if expected is None or expected["gpu"] != gpu or tuple(expected["ports"]) != tuple(ports):
        raise ValueError("invalid fixed worker/GPU/port coordinates")


def resolve_idle_bundle(report_path, catalog_path=None):
    from scripts.ditl_task6_runner import resolve_bundle_mapping
    report_path = Path(report_path)
    if sha_file(report_path) != EXPECTED_CAPTURE_REPORT_SHA256:
        raise ValueError("capture report differs from the frozen stage-one report")
    bundle = resolve_bundle_mapping(report_path, catalog_path)[IDLE_PROFILE]
    if (bundle["feature_id"] != EXPECTED_IDLE_FEATURE_ID or
            bundle["manifest_sha256"] != EXPECTED_IDLE_MANIFEST_SHA256):
        raise ValueError("idle bundle differs from the frozen stage-one bundle")
    return bundle


def validate_frozen_inputs(manifest, capture_report, idle_bundle):
    paths = [Path(capture_report), EVAL_BATCH, Path(__file__),
             ROOT / "scripts/ditl_packet_loss_manifest.py",
             ROOT / "scripts/ditl_packet_loss_result.py",
             ROOT / "scripts/ditl_packet_loss_reuse.py"] + list(PROFILE_CONFIGS.values())
    for task in manifest["tasks"]:
        paths.append(ROOT / "simulation/leaderboard/data/evaluation_routes" /
                     ("town05_short_r%d.xml" % task["route"]))
        paths.append(ROOT / "simulation/leaderboard/leaderboard/scenarios" /
                     ("scenario_parameter%s.yaml" % task["scenario"]))
    for field in ("catalog_path", "bundle_path"):
        paths.append(Path(idle_bundle[field]))
    paths.append(Path(idle_bundle["bundle_path"]) / "manifest.json")
    for config_path in PROFILE_CONFIGS.values():
        config = _load_yaml(config_path)
        perception = ROOT / config["perception"]["perception_model_dir"]
        planner = ROOT / config["planning"]["planner_model_checkpoint"]
        paths.extend(path for path in perception.rglob("*") if path.is_file())
        paths.append(planner)
    for path in sorted(set(paths), key=str):
        if path.is_symlink() or not path.exists():
            raise ValueError("frozen input is missing or symbolic: %s" % path)
    source = _git_snapshot(ROOT)
    if source["status"]:
        raise RuntimeError("source worktree must be clean before launch")
    return {str(path): sha_file(path) if path.is_file() else None for path in paths}


def build_metadata(manifest, resource_snapshot, manifest_sha256,
                   reused_task_count=0, scheduled_task_count=None):
    if scheduled_task_count is None:
        scheduled_task_count = len(manifest.get("tasks", []))
    return {"schema": "ditl.packet-loss-stage1-metadata.v1",
            "manifest_sha256": manifest_sha256,
            "task_count": len(manifest.get("tasks", [])),
            "batch_count": len(manifest.get("batches", [])),
            "reused_task_count": int(reused_task_count),
            "scheduled_task_count": int(scheduled_task_count),
            "resource_preflight": copy.deepcopy(resource_snapshot),
            "shared_gpu_scope": "driving-and-AoI-only; no latency-throughput-energy claims"}


def reuse_cohort_counts(reuse_plan):
    if reuse_plan is None:
        return {}
    counts = {}
    selected = reuse_plan.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("reuse plan selection is malformed")
    for row in selected.values():
        source = row.get("source_cohort") if isinstance(row, dict) else None
        if not isinstance(source, str) or not source:
            raise ValueError("reuse plan selected source cohort is missing")
        name = Path(source).name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _config_path(cohort, key):
    return Path(cohort) / "configs" / (_safe_name(key) + ".yaml")


def build_batch_command(batch, assignment, cohort, attempt, task_map=None):
    round_name = "first" if attempt == 1 else "retry"
    validate_launch_coordinates(round_name, assignment["id"], assignment["gpu"],
                                assignment["ports"])
    if task_map is None:
        manifest = json.loads((Path(cohort) / "manifest.json").read_text(encoding="utf-8"))
        task_map = {task["key"]: task for task in validate_stage1_manifest(manifest)["tasks"]}
    command = [str(EVAL_BATCH), "--cohort", str(Path(cohort).resolve()),
               "--batch-id", batch["id"], "--attempt", str(attempt),
               "--worker", assignment["id"], "--gpu", str(assignment["gpu"])]
    labels = ("--world-port", "--streaming-port", "--secondary-port", "--traffic-port")
    for label, port in zip(labels, assignment["ports"]): command.extend((label, str(port)))
    for key in batch["task_keys"]:
        task = task_map[key]
        command.extend(("--task", key, str(task["route"]), task["scenario"],
                        task["profile"], str(_config_path(cohort, key))))
    return command


def verify_cohort_frozen(cohort, batch):
    """Fail closed if a frozen document, input, or selected config drifted."""
    cohort = Path(cohort)
    documents = {}
    for name in ("manifest.json", "partition.json", "metadata.json", "provenance.json"):
        path = cohort / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("frozen cohort document is missing: %s" % name)
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    manifest_bytes = canonical_json_bytes(validate_stage1_manifest(documents["manifest.json"]))
    if hashlib.sha256(manifest_bytes).hexdigest() != documents["metadata.json"].get("manifest_sha256"):
        raise RuntimeError("manifest hash drift")
    if documents["partition.json"] != build_partition():
        raise RuntimeError("partition drift")
    for raw_path, expected in documents["provenance.json"].get("frozen_input_sha256", {}).items():
        path = Path(raw_path)
        if expected is not None and (path.is_symlink() or not path.is_file() or
                                     sha_file(path) != expected):
            raise RuntimeError("frozen input drift: %s" % path)
    reuse_sha = documents["provenance.json"].get("reuse_plan_sha256")
    if reuse_sha is not None:
        reuse_path = cohort / "reuse_plan.json"
        if (not isinstance(reuse_sha, str) or len(reuse_sha) != 64 or
                reuse_path.is_symlink() or not reuse_path.is_file() or
                sha_file(reuse_path) != reuse_sha):
            raise RuntimeError("reuse plan drift")
    config_hashes = documents["provenance.json"].get("config_sha256", {})
    for key in batch["task_keys"]:
        path = _config_path(cohort, key)
        if path.is_symlink() or not path.is_file() or sha_file(path) != config_hashes.get(key):
            raise RuntimeError("task config drift: %s" % key)
    return True


def validate_round_resources(assignments):
    ports = tuple(port for assignment in assignments for port in assignment["ports"])
    snapshot = capture_resource_snapshot((6, 7), ports)
    validate_resource_snapshot(snapshot, (6, 7), ())
    wait_for_ports_bindable(ports)
    return snapshot


def run_round(batches, assignments, attempt, launch):
    pending = queue.Queue()
    for batch in batches: pending.put(batch)
    def worker(assignment):
        rows = []
        while not _STOP.is_set():
            try: batch = pending.get_nowait()
            except queue.Empty: break
            try:
                rows.append(launch(batch, assignment, attempt))
            except BaseException as error:
                stop_and_cancel = getattr(launch, "stop_and_cancel", None)
                try:
                    if stop_and_cancel is not None:
                        stop_and_cancel()
                    else:
                        _STOP.set()
                        cancel_owned = getattr(launch, "cancel_owned", None)
                        if cancel_owned is not None:
                            cancel_owned()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise
            finally: pending.task_done()
        return rows
    output = []
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, assignment) for assignment in assignments]
        for future in as_completed(futures):
            try:
                output.extend(future.result())
            except BaseException as error:
                errors.append(error)
    if errors:
        selected = next((error for error in errors
                         if not isinstance(error, RoundCancelled)), errors[0])
        raise selected.with_traceback(selected.__traceback__)
    return output


def classify_attempt(manifest, cohort, attempt, task_keys=None):
    selected = set(task_keys) if task_keys is not None else None
    rows = {}
    for task in manifest["tasks"]:
        if selected is None or task["key"] in selected:
            directory = Path(cohort) / "attempts" / ("attempt-%d" % attempt) / _safe_name(task["key"])
            rows[task["key"]] = classify_task_evidence(task, directory)
    return rows


def execute_two_rounds(manifest, cohort, launch, first_task_keys=None):
    first_batches = build_batches(manifest, first_task_keys)
    if first_batches:
        run_round(first_batches, FIRST_ASSIGNMENTS, 1, launch)
    first = classify_attempt(manifest, cohort, 1)
    retry_keys = plan_retry_task_keys(manifest, first)
    retry = {}
    if retry_keys:
        validate_round_resources(RETRY_ASSIGNMENTS)
        retry_batches = build_batches(manifest, retry_keys)
        run_round(retry_batches, RETRY_ASSIGNMENTS, 2, launch)
        retry = classify_attempt(manifest, cohort, 2, retry_keys)
    finalized = finalize_task_results(manifest, first, retry)
    expected_keys = {task["key"] for task in manifest["tasks"]}
    if set(finalized) != expected_keys:
        raise RuntimeError("finalized task set does not match the canonical manifest")
    artifacts = generate_artifacts(cohort)
    return {"first_attempt": first, "retry_attempt": retry,
            "finalized": finalized, "artifacts": artifacts}


def _identity_matches(record, current, cohort):
    return (record.get("cohort") == str(Path(cohort).resolve()) and
            record.get("ppid") == os.getpid() and
            record.get("pid") not in PROTECTED_PIDS and
            record.get("pid") == record.get("pgid") == record.get("sid") and
            all(record.get(field) == current.get(field)
                for field in ("pid", "ppid", "pgid", "sid", "start_token")))


def terminate_owned_processes(records, cohort, inspect=inspect_process,
                              signal_group=os.killpg, pause=time.sleep):
    validated = []
    mismatches = []
    for record in records:
        if not isinstance(record, dict) or record.get("pid") in PROTECTED_PIDS:
            mismatches.append("invalid or protected process identity")
            continue
        try: current = inspect(record.get("pid"))
        except (OSError, ProcessLookupError, FileNotFoundError): continue
        if not isinstance(current, dict) or not _identity_matches(
                record, current, cohort):
            mismatches.append("recorded process identity is stale or foreign")
            continue
        validated.append(record)
    for sig, delay in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 0.0)):
        for record in validated:
            try: current = inspect(record["pid"])
            except (OSError, ProcessLookupError, FileNotFoundError): continue
            if not isinstance(current, dict) or not _identity_matches(
                    record, current, cohort):
                mismatches.append("process identity changed")
                continue
            try: signal_group(record["pgid"], sig)
            except ProcessLookupError: pass
        if delay: pause(delay)
    if mismatches:
        raise RuntimeError("; ".join(mismatches))


def _create_cohort(cohort, partition, manifest, idle_bundle, preflight, hashes, args,
                   first_task_keys=None, reuse_plan=None):
    os.mkdir(str(cohort), 0o700)
    for name in ("configs", "attempts", "logs", "processes", "snapshots"):
        (cohort / name).mkdir(mode=0o700)
    (cohort / "attempts" / "attempt-1").mkdir(mode=0o700)
    (cohort / "attempts" / "attempt-2").mkdir(mode=0o700)
    (cohort / "events.jsonl").touch(mode=0o600)
    config_payloads = build_config_payloads(manifest, idle_bundle)
    for task in manifest["tasks"]:
        path = _config_path(cohort, task["key"])
        with path.open("xb") as stream:
            stream.write(config_payloads[task["key"]])
        path.chmod(0o400)
    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    config_hashes = {task["key"]: sha_file(_config_path(cohort, task["key"]))
                     for task in manifest["tasks"]}
    task_map = {task["key"]: task for task in manifest["tasks"]}
    selected_first_keys = ([task["key"] for task in manifest["tasks"]]
                           if first_task_keys is None else list(first_task_keys))
    reused_keys = []
    if reuse_plan is not None:
        from scripts.ditl_packet_loss_reuse import materialize_reuse_plan
        materialized = materialize_reuse_plan(manifest, reuse_plan, cohort)
        reused_keys = list(materialized["selected_task_keys"])
        expected_reused = set(task_map).difference(selected_first_keys)
        if set(reused_keys) != expected_reused:
            raise RuntimeError("materialized reuse set does not match scheduled complement")
    first_commands = [build_batch_command(
        batch, FIRST_ASSIGNMENTS[index % 2], cohort, 1, task_map)
        for index, batch in enumerate(build_batches(manifest, selected_first_keys))]
    reuse_sha = None
    if reuse_plan is not None:
        reuse_sha = hashlib.sha256(canonical_json_bytes(reuse_plan)).hexdigest()
    documents = {
        "partition.json": partition, "manifest.json": manifest,
        "metadata.json": build_metadata(
            manifest, preflight, manifest_sha,
            reused_task_count=len(reused_keys),
            scheduled_task_count=len(selected_first_keys)),
        "provenance.json": {"schema": "ditl.packet-loss-stage1-provenance.v1",
                            "capture_report": str(args.capture_report),
                            "idle_bundle": idle_bundle, "frozen_input_sha256": hashes,
                            "config_sha256": config_hashes,
                            "reuse_plan_sha256": reuse_sha,
                            "source": preflight.get("source"),
                            "protected": preflight.get("protected")},
        "resource_preflight.json": preflight,
        "commands.json": {"schema": "ditl.packet-loss-stage1-commands.v1",
                          "argv": list(getattr(args, "argv", sys.argv)),
                          "first_pass": first_commands,
                          "retry_ports": [port for assignment in RETRY_ASSIGNMENTS
                                          for port in assignment["ports"]]},
    }
    if reuse_plan is not None:
        documents["reuse_plan.json"] = reuse_plan
    for name, value in documents.items(): _canonical_write_new(cohort / name, value)


def _launch_factory(cohort, manifest, baseline_protected):
    task_map = {task["key"]: task for task in manifest["tasks"]}
    owned, lock = [], threading.Lock()
    launch_gate = threading.Lock()
    cancellation_lock = threading.Lock()
    cancellation_started = [False]

    def stop_and_cancel():
        with launch_gate:
            _STOP.set()
            with cancellation_lock:
                if cancellation_started[0]:
                    return
                cancellation_started[0] = True
            with lock:
                records = list(owned)
        terminate_owned_processes(records, cohort)

    def launch(batch, assignment, attempt):
        round_name = "first" if attempt == 1 else "retry"
        ports = assignment["ports"]
        snapshot = capture_resource_snapshot((6, 7), ports)
        validate_resource_snapshot(snapshot, (6, 7), ())
        verify_cohort_frozen(cohort, batch)
        if snapshot.get("protected") != baseline_protected:
            raise RuntimeError("protected exact-age identity changed")
        wait_for_ports_bindable(ports)
        if _STOP.is_set():
            raise RoundCancelled("packet-loss round cancelled before batch launch")
        _atomic_json(cohort / "snapshots" / ("attempt-%d-%s.json" %
                    (attempt, _safe_name(batch["id"]))), snapshot)
        command = build_batch_command(batch, assignment, cohort, attempt, task_map)
        with launch_gate:
            if _STOP.is_set():
                raise RoundCancelled("packet-loss round cancelled before batch launch")
            process = subprocess.Popen(command, cwd=str(ROOT), start_new_session=True)
            identity = inspect_process(process.pid)
            identity.update({"cohort": str(cohort.resolve()), "batch_id": batch["id"],
                             "attempt": attempt, "worker": assignment["id"]})
            with lock:
                owned.append(identity)
                _append_event(cohort / "events.jsonl", {
                    "event": "batch_started", "batch_id": batch["id"],
                    "attempt": attempt, "worker": assignment["id"],
                    "gpu": assignment["gpu"], "ports": list(ports),
                    "pid": process.pid,
                    "at_unix_ns": int(time.time() * 1000000000)})
        identity_path = cohort / "processes" / ("attempt-%d-%s.json" %
                                                (attempt, _safe_name(batch["id"])))
        _atomic_json(identity_path, identity)
        try:
            returncode = process.wait()
        finally:
            with lock:
                if identity in owned:
                    owned.remove(identity)
        with lock:
            _append_event(cohort / "events.jsonl", {
                "event": "batch_finished", "batch_id": batch["id"],
                "attempt": attempt, "worker": assignment["id"],
                "returncode": returncode,
                "at_unix_ns": int(time.time() * 1000000000)})
        return {"batch_id": batch["id"], "attempt": attempt,
                "worker": assignment["id"], "returncode": returncode,
                "command_argv": command}
    launch.stop_and_cancel = stop_and_cancel
    launch.cancel_owned = stop_and_cancel
    return launch, owned


def execute(args, emit=print):
    _STOP.clear()
    partition = build_partition()
    manifest = validate_stage1_manifest(build_stage1_manifest(partition))
    cohort = validate_new_cohort_path(args.output_root / args.cohort_id)
    idle_bundle = resolve_idle_bundle(args.capture_report, args.catalog)
    reuse_plan = None
    first_task_keys = None
    if args.reuse_cohort:
        from scripts.ditl_packet_loss_reuse import build_reuse_plan
        config_payloads = build_config_payloads(manifest, idle_bundle)
        expected_config_sha256 = {
            key: hashlib.sha256(payload).hexdigest()
            for key, payload in config_payloads.items()
        }
        reuse_plan = build_reuse_plan(
            manifest, args.reuse_cohort, expected_config_sha256,
            AUTHORIZED_OUTPUT_ROOTS)
        first_task_keys = list(reuse_plan["missing_task_keys"])
    hashes = validate_frozen_inputs(manifest, args.capture_report, idle_bundle)
    preflight = capture_resource_snapshot((6, 7), ALL_PORTS)
    validate_resource_snapshot(preflight, (6, 7), ())
    wait_for_ports_bindable(ALL_PORTS)
    first_batches = build_batches(manifest, first_task_keys)
    task_map = {task["key"]: task for task in manifest["tasks"]}
    dry = {"schema": "ditl.packet-loss-stage1-dry-run.v1", "cells": 45,
           "arms": 4, "tasks": len(manifest["tasks"]),
           "reused_tasks": 0 if reuse_plan is None else reuse_plan["selected_count"],
           "scheduled_tasks": (len(manifest["tasks"]) if first_task_keys is None
                               else len(first_task_keys)),
           "reuse_source_counts": ({} if reuse_plan is None
                                   else reuse_plan["source_counts"]),
           "reuse_cohort_counts": reuse_cohort_counts(reuse_plan),
           "first_pass_batches": len(first_batches), "gpus": [6, 7],
           "all_ports": list(ALL_PORTS), "cohort": str(cohort),
           "resource_preflight": copy.deepcopy(preflight),
           "argv": list(getattr(args, "argv", sys.argv)),
           "retry_ports": [port for assignment in RETRY_ASSIGNMENTS
                           for port in assignment["ports"]],
           "first_pass_commands": [build_batch_command(
               batch, FIRST_ASSIGNMENTS[index % 2], cohort, 1,
               task_map)
               for index, batch in enumerate(first_batches)]}
    if args.dry_run:
        emit(json.dumps(dry, sort_keys=True)); return 0
    _create_cohort(cohort, partition, manifest, idle_bundle, preflight, hashes, args,
                   first_task_keys=first_task_keys, reuse_plan=reuse_plan)
    launch, owned = _launch_factory(cohort, manifest, preflight.get("protected"))
    old_handlers = {}
    def interrupted(signum, _frame):
        launch.stop_and_cancel()
        _atomic_json(cohort / "interrupted.json", {"signal": signum,
                                                   "at_unix": time.time()})
        raise RunInterrupted(signum)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        try:
            if first_task_keys is None:
                result = execute_two_rounds(manifest, cohort, launch)
            else:
                result = execute_two_rounds(
                    manifest, cohort, launch, first_task_keys=first_task_keys)
        except RunInterrupted as interruption:
            return 128 + interruption.signum
    finally:
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)
    emit(json.dumps({"schema": "ditl.packet-loss-stage1-finished.v1",
                     "cohort": str(cohort),
                     "finalized": len(result["finalized"]),
                     "retry_count": len(result["retry_attempt"]),
                     "artifacts": result["artifacts"]}, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--output-root", type=Path, default=AUTHORIZED_ROOT)
    parser.add_argument("--capture-report", type=Path, default=DEFAULT_CAPTURE_REPORT)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--reuse-cohort", action="append", type=Path, default=[])
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv); args.argv = list(argv if argv is not None else sys.argv[1:])
    return args


def main(argv=None):
    try: return execute(parse_args(argv))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError,
            yaml.YAMLError, subprocess.CalledProcessError) as error:
        print("packet-loss stage-one runner failed: %s" % error, file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
