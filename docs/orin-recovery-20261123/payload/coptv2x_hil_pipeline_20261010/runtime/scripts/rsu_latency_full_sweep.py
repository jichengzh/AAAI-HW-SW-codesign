#!/usr/bin/env python
"""Schedule the official 340-episode, RSU-only latency validation sweep."""

from __future__ import print_function

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rsu_latency_full_manifest import build_manifest  # noqa: E402
from dual_fifo_result import TERMINAL_DRIVING_STATUSES  # noqa: E402


DELAYS_MS = (0, 200, 500, 800)
BATCH_PORT_STRIDE = 200
ROUTE_PORT_STRIDE = 10
COHORT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RUN_DIR_RE = re.compile(r"\brun=(\S+)")
RUN_IDENTITY_RE = re.compile(
    r"\[worker\] GPU=(?P<gpu>\d+) port=(?P<port>\d+) "
    r"cohort=(?P<cohort>\S+) arm=(?P<arm>\S+) "
    r"scenario=(?P<scenario>_[1-5]) carla_seed=(?P<carla_seed>\d+) "
    r"traffic_seed=(?P<traffic_seed>\d+) routes=\{(?P<routes>[0-9 ]+)\} "
    r"repeat=(?P<repeat>\d+) run=(?P<run_dir>\S+)"
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_gpu_pool(value):
    if value is None or not re.match(r"^[1-8]$", value):
        raise ValueError("GPU_POOL must be an integer count from 1 through 8")
    return int(value)


def parse_idle_gpus(output):
    idle = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise ValueError("unreadable nvidia-smi row: %s" % raw_line)
        index, utilization, memory = map(int, fields)
        if utilization <= 5 and memory <= 50:
            idle.append(index)
    return sorted(idle)


def select_idle_gpus(output, count):
    idle = parse_idle_gpus(output)
    if len(idle) < count:
        raise RuntimeError(
            "need %d idle GPUs but found %d (%s)"
            % (count, len(idle), ",".join(map(str, idle)) or "none")
        )
    return idle[:count]


def query_idle_gpus(count):
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command, universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed: %s" % result.stdout.strip())
    return select_idle_gpus(result.stdout, count)


def build_batches(tasks):
    grouped = collections.OrderedDict()
    for task in tasks:
        identity = (
            task["scenario"], task["rsu_delay_ms"], task["carla_seed"],
            task["traffic_seed"], task["repeat"],
        )
        grouped.setdefault(identity, []).append(task)
    batches = []
    for index, (identity, members) in enumerate(grouped.items()):
        scenario, delay, carla_seed, traffic_seed, repeat = identity
        routes = [member["route"] for member in members]
        batches.append({
            "index": index,
            "key": "sc%s_d%d_cseed%d_tseed%d_rep%d" % (
                scenario.lstrip("_"), delay, carla_seed, traffic_seed, repeat
            ),
            "scenario": scenario,
            "rsu_delay_ms": delay,
            "ego_delay_ms": 0,
            "carla_seed": carla_seed,
            "traffic_seed": traffic_seed,
            "repeat": repeat,
            "routes": routes,
            "task_keys": [member["key"] for member in members],
        })
    return batches


def runner_command(batch, gpu, port, cohort, max_attempts=None):
    command = [
        "bash", "scripts/dual_fifo_multiroute_sc.sh",
        "--scenario", batch["scenario"],
        "--carla-seed", str(batch["carla_seed"]),
        "--traffic-seed", str(batch["traffic_seed"]),
        "--cohort", cohort,
    ]
    if max_attempts is not None:
        command.extend(("--max-attempts", str(max_attempts)))
    return command + [
        str(gpu), str(port), str(batch["rsu_delay_ms"]), "0",
        " ".join(str(route) for route in batch["routes"]),
        str(batch["repeat"]),
    ]


def build_preview(batches, worker_count, base_port, cohort):
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if (base_port < 1 or any(
            len(batch["routes"]) * ROUTE_PORT_STRIDE > BATCH_PORT_STRIDE or
            base_port + batch["index"] * BATCH_PORT_STRIDE +
            (len(batch["routes"]) - 1) * ROUTE_PORT_STRIDE + 5 > 65535
            for batch in batches)):
        raise ValueError("base port range is invalid")
    preview = []
    for index, batch in enumerate(batches):
        slot = index % worker_count
        port = base_port + batch["index"] * BATCH_PORT_STRIDE
        item = dict(batch)
        item.update({
            "slot": slot,
            "port": port,
            "route_ports": [port + route_index * ROUTE_PORT_STRIDE
                            for route_index in range(len(batch["routes"]))],
            "command": runner_command(
                batch, "SLOT%d" % slot, port, cohort
            ),
        })
        preview.append(item)
    return preview


def create_cohort(output_root, cohort=None):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if cohort is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cohort = "rsu-full-%s-p%d" % (timestamp, os.getpid())
    if not COHORT_RE.match(cohort):
        raise ValueError("unsafe cohort label: %r" % cohort)
    path = output_root / cohort
    path.mkdir()
    (path / "logs").mkdir()
    return path


def write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def git_provenance():
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        universal_newlines=True,
    ).strip())
    return {"git_commit": commit, "git_dirty": dirty}


class SweepExecutor(object):
    def __init__(self, batches, gpus, base_port, cohort_dir, cohort,
                 round_name="first"):
        self.batches = list(batches)
        self.gpus = list(gpus)
        self.base_port = base_port
        self.cohort_dir = Path(cohort_dir)
        self.cohort = cohort
        self.round_name = round_name
        self.pending = queue.Queue()
        for batch in self.batches:
            self.pending.put(batch)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.active = {}
        self.failures = []
        self.completed = []

    def append_event(self, event, **fields):
        payload = dict(fields)
        payload.update({
            "event": event,
            "time_utc": utc_now(),
        })
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self.lock:
            with (self.cohort_dir / "events.jsonl").open("a") as stream:
                stream.write(line)

    def terminate_active(self):
        self.stopping.set()
        with self.lock:
            processes = list(self.active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def worker(self, slot, gpu):
        while not self.stopping.is_set():
            try:
                batch = self.pending.get_nowait()
            except queue.Empty:
                return
            port = self.base_port + batch["index"] * BATCH_PORT_STRIDE
            command = runner_command(batch, gpu, port, self.cohort, max_attempts=1)
            log_path = self.cohort_dir / "logs" / (
                batch["key"] + "." + self.round_name + ".log"
            )
            self.append_event(
                "batch_started", batch_key=batch["key"], gpu=gpu,
                port=port, command=command, round=self.round_name,
            )
            with log_path.open("w") as log:
                try:
                    process = subprocess.Popen(
                        command, cwd=str(REPO), stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    with self.lock:
                        self.active[slot] = process
                    return_code = process.wait()
                except OSError as error:
                    log.write("[ABORT] runner could not start: %s\n" % error)
                    return_code = 127
                finally:
                    with self.lock:
                        self.active.pop(slot, None)
            run_dir = None
            contents = log_path.read_text(errors="replace")
            match = RUN_DIR_RE.search(contents)
            if match:
                run_dir = match.group(1)
            self.append_event(
                "batch_finished", batch_key=batch["key"], gpu=gpu,
                port=port, return_code=return_code, run_dir=run_dir,
                log=str(log_path), round=self.round_name,
            )
            with self.lock:
                target = self.completed if return_code == 0 else self.failures
                target.append(batch["key"])
            self.pending.task_done()

    def run(self):
        threads = []
        for slot, gpu in enumerate(self.gpus):
            thread = threading.Thread(target=self.worker, args=(slot, gpu))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        return 0 if not self.failures and len(self.completed) == len(self.batches) else 1


def plan_pending_batches(tasks, accepted_keys):
    """Preserve stable group/port identities while omitting measured tasks."""
    accepted_keys = set(accepted_keys)
    planned = []
    for batch in build_batches(tasks):
        members = [(route, key) for route, key in
                   zip(batch["routes"], batch["task_keys"])
                   if key not in accepted_keys]
        if members:
            planned.append(dict(batch, routes=[route for route, _ in members],
                                task_keys=[key for _, key in members]))
    return planned


def manifest_hash(tasks):
    encoded = json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_infraction(value):
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    if isinstance(value, bool) or value is None:
        return bool(value)
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(value)


def _verified_row(row, task, commit):
    """Accept one runner row only with matching task identity and evidence."""
    if row["countable"] != "True":
        raise ValueError("runner marked route INFRA_INVALID")
    if int(row["route"]) != task["route"] or int(row["repeat"]) != task["repeat"]:
        raise ValueError("route/repeat mismatch")
    if (int(row["rsu_fifo_delay_ms"]) != task["rsu_delay_ms"] or
            int(row["ego_output_fifo_delay_ms"]) != 0 or
            Path(row["agent_config"]).name != "pnp_config_%s.yaml" % task["agent"]):
        raise ValueError("delay/agent mismatch")
    if row["git_commit"] != commit:
        raise ValueError("git provenance mismatch")
    for field in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv"):
        if not Path(row[field]).is_file():
            raise ValueError("missing %s evidence" % field)
    checkpoint = json.loads(Path(row["ego_json"]).read_text())["_checkpoint"]
    record = checkpoint["global_record"]
    if row["status"] not in TERMINAL_DRIVING_STATUSES or record["status"] != row["status"]:
        raise ValueError("driving status mismatch")
    ds, rc = float(row["DS"]), float(row["RC"])
    if not math.isfinite(ds) or not math.isfinite(rc):
        raise ValueError("non-finite driving score")
    infractions = record.get("infractions") or {}
    timed_out = (_positive_infraction(infractions.get("route_timeout")) or
                 row["status"] == "Failed - Agent timed out")
    return {
        "task_key": task["key"], "delay": task["rsu_delay_ms"],
        "status": row["status"], "DS": 0.0 if timed_out else ds,
        "RC": rc, "route_timeout": timed_out,
        "vehicle_blocked": _positive_infraction(infractions.get("vehicle_blocked")),
        "collision": any(_positive_infraction(value)
                         for name, value in infractions.items()
                         if name.startswith("collisions_")),
    }


def collect_verified_results(cohort_dir, batch_file="batches.json", round_name=None):
    """Return per-task valid evidence even when its enclosing batch exited nonzero."""
    cohort_dir = Path(cohort_dir)
    tasks = json.loads((cohort_dir / "manifest.json").read_text())
    batches = json.loads((cohort_dir / batch_file).read_text())
    metadata = json.loads((cohort_dir / "metadata.json").read_text())
    events_path = cohort_dir / "events.jsonl"
    events = (json.loads(line) for line in events_path.read_text().splitlines()
              if line.strip()) if events_path.is_file() else ()
    finished = collections.defaultdict(list)
    for event in events:
        if event.get("event") == "batch_finished" and (
                round_name is None or event.get("round") == round_name):
            finished[event.get("batch_key")].append(event)
    task_by_key = {task["key"]: task for task in tasks}
    accepted, invalid = {}, {}
    for batch in batches:
        keys = batch["task_keys"]
        invalid.update({key: "batch did not finish" for key in keys})
        batch_events = finished.get(batch["key"], [])
        if len(batch_events) != 1 or not batch_events[0].get("run_dir"):
            continue
        event = batch_events[0]
        log_path = Path(event.get("log") or "")
        if not log_path.is_file():
            continue
        identity = RUN_IDENTITY_RE.search(log_path.read_text(errors="replace"))
        arm = "baseline" if batch["rsu_delay_ms"] == 0 else "rsu%d" % batch["rsu_delay_ms"]
        expected = {
            "cohort": metadata["cohort"], "arm": arm,
            "scenario": batch["scenario"], "carla_seed": str(batch["carla_seed"]),
            "traffic_seed": str(batch["traffic_seed"]),
            "routes": " ".join(map(str, batch["routes"])),
            "repeat": str(batch["repeat"]), "run_dir": event["run_dir"],
        }
        if identity is None or any(identity.group(name) != value
                                   for name, value in expected.items()):
            invalid.update({key: "runner identity mismatch" for key in keys})
            continue
        summary_path = Path(event["run_dir"]) / "summary.csv"
        if not summary_path.is_file():
            invalid.update({key: "summary.csv missing" for key in keys})
            continue
        with summary_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        route_keys = dict(zip(batch["routes"], keys))
        seen = set()
        for row in rows:
            try:
                route = int(row["route"])
                if route not in route_keys or route in seen:
                    raise ValueError("duplicate or unexpected route")
                key = route_keys[route]
                seen.add(route)
                task = task_by_key[key]
                result = _verified_row(row, task, metadata["git_commit"])
                accepted[key] = result
                invalid.pop(key, None)
            except (KeyError, TypeError, ValueError, OSError,
                    json.JSONDecodeError) as error:
                if row.get("route", "").isdigit() and int(row["route"]) in route_keys:
                    key = route_keys[int(row["route"])]
                    accepted.pop(key, None)
                    invalid[key] = str(error)
    return accepted, invalid


def summarize_recovery_tasks(tasks, reused, first, retry):
    """Account for every manifest task; unresolved twice-run tasks contribute DS=0."""
    rows = []
    for task in tasks:
        key = task["key"]
        result = reused.get(key) or first.get(key) or retry.get(key)
        rows.append((task["rsu_delay_ms"], result, key not in reused and key not in first))
    by_delay = {}
    for delay in sorted({task["rsu_delay_ms"] for task in tasks}):
        arm = [(result, first_failed) for arm_delay, result, first_failed in rows
               if arm_delay == delay]
        valid = [result for result, _ in arm if result is not None]
        zeroed = len(arm) - len(valid)
        by_delay[str(delay)] = {
            "planned": len(arm), "countable": len(valid),
            "zeroed_after_retry": zeroed,
            "Completed": sum(row["status"] == "Completed" for row in valid),
            "Failed": sum(row["status"].startswith("Failed") for row in valid),
            "mean_DS": sum(row["DS"] for row in valid) / len(arm) if arm else None,
            "mean_RC": sum(row["RC"] for row in valid) / len(valid) if valid else None,
            "first_pass_failures": sum(failed for _, failed in arm),
            "first_pass_failure_rate": sum(failed for _, failed in arm) / len(arm) if arm else None,
            "final_failures": zeroed,
            "final_failure_rate": zeroed / len(arm) if arm else None,
            "collision_episodes": sum(row["collision"] for row in valid),
            "vehicle_blocked": sum(row["vehicle_blocked"] for row in valid),
            "route_timeout": sum(row["route_timeout"] for row in valid),
        }
    return {
        "planned": len(tasks), "countable": sum(result is not None for _, result, _ in rows),
        "zeroed_after_retry": sum(result is None for _, result, _ in rows),
        "zeroed_task_keys": [task["key"] for task, (_, result, _) in zip(tasks, rows)
                             if result is None],
        "reused": len(reused), "by_delay_ms": by_delay,
    }


def _finished_batch_keys(cohort_dir, round_name):
    path = Path(cohort_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [event.get("batch_key") for event in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())
            if event.get("event") == "batch_finished" and event.get("round") == round_name]


def summarize_recovery_cohort(cohort_dir):
    cohort_dir = Path(cohort_dir)
    metadata = json.loads((cohort_dir / "metadata.json").read_text())
    tasks = json.loads((cohort_dir / "manifest.json").read_text())
    source = Path(metadata["reuse_cohort"])
    reused, _ = collect_verified_results(source)
    first, first_invalid = collect_verified_results(cohort_dir, round_name="first")
    retry_batches_path = cohort_dir / "retry_batches.json"
    retry = {}
    if retry_batches_path.is_file():
        retry, _ = collect_verified_results(
            cohort_dir, batch_file="retry_batches.json", round_name="retry"
        )
    result = summarize_recovery_tasks(tasks, reused, first, retry)
    first_batches = json.loads((cohort_dir / "batches.json").read_text())
    retry_batches = json.loads(retry_batches_path.read_text()) if retry_batches_path.is_file() else []
    first_done = _finished_batch_keys(cohort_dir, "first")
    retry_done = _finished_batch_keys(cohort_dir, "retry")
    expected_retry = {key for batch in first_batches for key in batch["task_keys"]} - set(first)
    planned_retry = {key for batch in retry_batches for key in batch["task_keys"]}
    result.update({
        "cohort_dir": str(cohort_dir),
        "reused_source": str(source),
        "first_pass_failures": len(first_invalid),
        "first_pass_finished_batches": len(first_done),
        "retry_finished_batches": len(retry_done),
        "complete": (
            len(reused) == metadata["reused_tasks"] and
            len(first_done) == len(first_batches) and
            len(retry_done) == len(retry_batches) and
            expected_retry == planned_retry and
            result["countable"] + result["zeroed_after_retry"] == len(tasks)
        ),
    })
    return result


def summarize_cohort(cohort_dir):
    """Join runner summaries back to manifest tasks and aggregate final metrics."""
    cohort_dir = Path(cohort_dir)
    tasks = json.loads((cohort_dir / "manifest.json").read_text())
    batches = json.loads((cohort_dir / "batches.json").read_text())
    metadata_path = cohort_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    events = []
    event_path = cohort_dir / "events.jsonl"
    if event_path.is_file():
        for number, line in enumerate(event_path.read_text().splitlines(), 1):
            try:
                events.append(json.loads(line))
            except ValueError:
                events.append({"event": "invalid", "line": number})

    issues = []
    if any(event.get("event") == "invalid" for event in events):
        issues.append("invalid events.jsonl record")
    task_by_key = {task["key"]: task for task in tasks}
    if len(task_by_key) != len(tasks):
        issues.append("duplicate task key in manifest")
    expected_keys = set(task_by_key)
    finished = collections.defaultdict(list)
    for event in events:
        if event.get("event") == "batch_finished":
            finished[event.get("batch_key")].append(event)

    scientific = []
    observed = []
    infra_invalid = 0
    for batch in batches:
        batch_key = batch["key"]
        expected_batch_keys = set(batch["task_keys"])
        unknown = expected_batch_keys - expected_keys
        if unknown:
            issues.append("batch %s references unknown tasks" % batch_key)
        batch_events = finished.get(batch_key, [])
        if len(batch_events) != 1:
            issues.append(
                "batch %s has %d finished events" % (batch_key, len(batch_events))
            )
            infra_invalid += len(expected_batch_keys)
            continue
        event = batch_events[0]
        if event.get("return_code") != 0 or not event.get("run_dir"):
            issues.append("batch %s did not finish cleanly" % batch_key)
            infra_invalid += len(expected_batch_keys)
            continue
        log_path = Path(event.get("log", ""))
        identity_match = None
        if log_path.is_file():
            identity_match = RUN_IDENTITY_RE.search(log_path.read_text(errors="replace"))
        expected_arm = (
            "baseline" if batch["rsu_delay_ms"] == 0
            else "rsu%d" % batch["rsu_delay_ms"]
        )
        expected_identity = {
            "cohort": metadata.get("cohort", cohort_dir.name),
            "arm": expected_arm,
            "scenario": batch["scenario"],
            "carla_seed": str(batch["carla_seed"]),
            "traffic_seed": str(batch["traffic_seed"]),
            "routes": " ".join(str(route) for route in batch["routes"]),
            "repeat": str(batch["repeat"]),
            "run_dir": event["run_dir"],
        }
        if identity_match is None or any(
                identity_match.group(field) != expected
                for field, expected in expected_identity.items()):
            issues.append("batch %s runner identity mismatch" % batch_key)
            infra_invalid += len(expected_batch_keys)
            continue
        run_dir = Path(event["run_dir"])
        summary_path = run_dir / "summary.csv"
        if not summary_path.is_file():
            issues.append("batch %s summary.csv is missing" % batch_key)
            infra_invalid += len(expected_batch_keys)
            continue
        with summary_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        seen_routes = set()
        for row in rows:
            try:
                route = int(row["route"])
                identity = (batch["scenario"], route, batch["rsu_delay_ms"])
                matches = [
                    task for task in tasks
                    if (task["scenario"], task["route"], task["rsu_delay_ms"])
                    == identity
                ]
                if len(matches) != 1:
                    raise ValueError("no unique manifest task")
                task = matches[0]
                observed.append(task["key"])
                if route in seen_routes:
                    issues.append("duplicate route %d in batch %s" % (route, batch_key))
                seen_routes.add(route)
                expected_agent = "pnp_config_%s.yaml" % task["agent"]
                if Path(row["agent_config"]).name != expected_agent:
                    raise ValueError("agent config mismatch")
                if int(row["repeat"]) != task["repeat"]:
                    raise ValueError("repeat mismatch")
                if int(row["rsu_fifo_delay_ms"]) != task["rsu_delay_ms"]:
                    raise ValueError("RSU delay mismatch")
                if int(row["ego_output_fifo_delay_ms"]) != 0:
                    raise ValueError("Ego delay is not zero")
                if metadata.get("git_commit") and row["git_commit"] != metadata["git_commit"]:
                    raise ValueError("git commit mismatch")
                required_files = (
                    row["ego_json"], row["rsu_audit_csv"], row["ego_output_audit_csv"]
                )
                if not all(Path(path).is_file() for path in required_files):
                    raise ValueError("result or audit evidence is missing")
                if row["countable"] != "True":
                    infra_invalid += 1
                    continue
                if row["status"] not in ("Completed", "Failed"):
                    raise ValueError("unrecognized driving status")
                payload = json.loads(Path(row["ego_json"]).read_text())
                record = payload["_checkpoint"]["global_record"]
                infractions = record.get("infractions") or {}
                timed_out = _positive_infraction(infractions.get("route_timeout"))
                blocked = _positive_infraction(infractions.get("vehicle_blocked"))
                collision = any(
                    _positive_infraction(value) for name, value in infractions.items()
                    if name.startswith("collisions_")
                )
                ds = float(row["DS"])
                scientific.append({
                    "task_key": task["key"], "delay": task["rsu_delay_ms"],
                    "status": row["status"], "DS": 0.0 if timed_out else ds,
                    "RC": float(row["RC"]), "route_timeout": timed_out,
                    "vehicle_blocked": blocked, "collision": collision,
                })
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                infra_invalid += 1
                issues.append("batch %s row invalid: %s" % (batch_key, error))
        missing_routes = set(batch["routes"]) - seen_routes
        if missing_routes:
            issues.append("batch %s missing routes %s" % (
                batch_key, sorted(missing_routes)
            ))
            infra_invalid += len(missing_routes)

    duplicates = len(observed) - len(set(observed))
    missing_keys = expected_keys - set(observed)
    if duplicates:
        issues.append("duplicate task results: %d" % duplicates)
    if missing_keys:
        issues.append("missing task results: %d" % len(missing_keys))

    by_delay = {}
    for delay in sorted({task["rsu_delay_ms"] for task in tasks}):
        rows = [row for row in scientific if row["delay"] == delay]
        planned = sum(task["rsu_delay_ms"] == delay for task in tasks)
        by_delay[str(delay)] = {
            "planned": planned,
            "countable": len(rows),
            "Completed": sum(row["status"] == "Completed" for row in rows),
            "Failed": sum(row["status"] == "Failed" for row in rows),
            "mean_DS": (sum(row["DS"] for row in rows) / len(rows)) if rows else None,
            "mean_RC": (sum(row["RC"] for row in rows) / len(rows)) if rows else None,
            "collision_episodes": sum(row["collision"] for row in rows),
            "vehicle_blocked": sum(row["vehicle_blocked"] for row in rows),
            "route_timeout": sum(row["route_timeout"] for row in rows),
        }
    complete = (
        not issues and infra_invalid == 0 and len(scientific) == len(tasks)
        and len(set(observed)) == len(tasks)
    )
    return {
        "cohort_dir": str(cohort_dir), "complete": complete,
        "planned": len(tasks), "observed": len(observed),
        "countable": len(scientific), "infra_invalid": infra_invalid,
        "issues": issues, "by_delay_ms": by_delay,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check-results")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--carla-seed", type=int)
    parser.add_argument("--traffic-seed", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=45000)
    parser.add_argument("--output-root", default=str(REPO / "results" / "rsu_latency_full"))
    parser.add_argument("--cohort")
    parser.add_argument("--reuse-cohort", help="prior cohort with verified task evidence")
    parser.add_argument("--expected-reused", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check_results:
        try:
            metadata = json.loads((Path(args.check_results) / "metadata.json").read_text())
            summary = (summarize_recovery_cohort(args.check_results)
                       if metadata.get("result_policy") == "two_pass_zero_ds"
                       else summarize_cohort(args.check_results))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print("[ABORT] %s" % error, file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["complete"] else 1
    try:
        worker_count = parse_gpu_pool(os.environ.get("GPU_POOL"))
        if args.seed < 0 or args.repeat < 0:
            raise ValueError("seed and repeat must be non-negative")
        carla_seed = args.seed if args.carla_seed is None else args.carla_seed
        traffic_seed = args.seed if args.traffic_seed is None else args.traffic_seed
        if carla_seed < 0 or traffic_seed < 0:
            raise ValueError("seeds must be non-negative")
        tasks = build_manifest(carla_seed, traffic_seed, args.repeat)
        reused = {}
        source = None
        if args.reuse_cohort:
            source = Path(args.reuse_cohort).resolve()
            source_tasks = json.loads((source / "manifest.json").read_text())
            if manifest_hash(source_tasks) != manifest_hash(tasks):
                raise ValueError("reuse cohort manifest does not match requested sweep")
            reused, _ = collect_verified_results(source)
            if args.expected_reused is not None and len(reused) != args.expected_reused:
                raise ValueError("expected %d reused tasks, verified %d" %
                                 (args.expected_reused, len(reused)))
        batches = plan_pending_batches(tasks, reused) if source else build_batches(tasks)
        gpus = None if args.dry_run else query_idle_gpus(worker_count)
        cohort_dir = create_cohort(args.output_root, args.cohort)
        cohort = cohort_dir.name
        preview = build_preview(batches, worker_count, args.base_port, cohort)
        if source:
            build_preview(build_batches(tasks), worker_count,
                          args.base_port + 5000, cohort)
        write_json(cohort_dir / "manifest.json", tasks)
        write_json(cohort_dir / "batches.json", preview)
        metadata = {
            "cohort": cohort,
            "created_utc": utc_now(),
            "manifest_sha256": manifest_hash(tasks),
            "planned_tasks": len(tasks),
            "planned_batches": len(batches),
            "worker_count": worker_count,
            "base_port": args.base_port,
            "code_sha256": {
                name: file_sha256(REPO / "scripts" / name)
                for name in (
                    "rsu_latency_full_manifest.py",
                    "rsu_latency_full_sweep.py",
                    "dual_fifo_multiroute_sc.sh",
                    "action_fifo_eval_one.sh",
                    "dual_fifo_result.py",
                )
            },
        }
        metadata.update(git_provenance())
        if source:
            metadata.update({
                "result_policy": "two_pass_zero_ds",
                "reuse_cohort": str(source), "reused_tasks": len(reused),
                "reuse_source_metadata": json.loads((source / "metadata.json").read_text()),
                "retry_base_port": args.base_port + 5000,
            })
            write_json(cohort_dir / "reused_task_keys.json", sorted(reused))
        write_json(cohort_dir / "metadata.json", metadata)
        if args.dry_run:
            print(json.dumps({
                "mode": "dry-run", "cohort_dir": str(cohort_dir),
                "planned_tasks": len(tasks), "planned_batches": len(batches),
                "manifest_sha256": metadata["manifest_sha256"],
            }, sort_keys=True))
            return 0
        metadata["selected_gpus"] = gpus
        write_json(cohort_dir / "metadata.json", metadata)
        executor = SweepExecutor(
            batches, gpus, args.base_port, cohort_dir, cohort
        )
        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: executor.terminate_active())
        print(json.dumps({
            "mode": "run", "cohort_dir": str(cohort_dir),
            "planned_tasks": len(tasks), "planned_batches": len(batches),
            "selected_gpus": gpus,
        }, sort_keys=True), flush=True)
        first_exit = executor.run()
        if not source:
            return first_exit
        first, first_invalid = collect_verified_results(cohort_dir, round_name="first")
        write_json(cohort_dir / "first_pass_failures.json", first_invalid)
        retry_batches = plan_pending_batches(tasks, set(reused) | set(first))
        retry_preview = build_preview(
            retry_batches, worker_count, args.base_port + 5000, cohort
        )
        write_json(cohort_dir / "retry_batches.json", retry_preview)
        if retry_batches and not executor.stopping.is_set():
            retry_executor = SweepExecutor(
                retry_batches, gpus, args.base_port + 5000,
                cohort_dir, cohort, round_name="retry",
            )
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, lambda _signum, _frame: retry_executor.terminate_active())
            retry_executor.run()
        summary = summarize_recovery_cohort(cohort_dir)
        write_json(cohort_dir / "final_summary.json", summary)
        print(json.dumps({
            "cohort_dir": str(cohort_dir), "complete": summary["complete"],
            "reused": summary["reused"], "countable": summary["countable"],
            "zeroed_after_retry": summary["zeroed_after_retry"],
        }, sort_keys=True), flush=True)
        return 0 if summary["complete"] else 1
    except (OSError, RuntimeError, ValueError) as error:
        print("[ABORT] %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
