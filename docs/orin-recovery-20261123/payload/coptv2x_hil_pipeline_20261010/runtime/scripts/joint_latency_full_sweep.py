#!/usr/bin/env python
"""Run the 850-episode joint RSU/action sweep with two-pass evidence recovery."""

import argparse
import collections
import csv
import datetime
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

from joint_latency_full_manifest import build_manifest
from rsu_latency_full_sweep import (RUN_DIR_RE, RUN_IDENTITY_RE, create_cohort,
                                    file_sha256, git_provenance, manifest_hash,
                                    parse_idle_gpus, utc_now, write_json)
from dual_fifo_result import TERMINAL_DRIVING_STATUSES


REPO = Path(__file__).resolve().parent.parent
BATCH_PORT_STRIDE = 200
ROUTE_PORT_STRIDE = 10
RETRY_PORT_OFFSET = 10000
DEFAULT_OUTPUT_ROOT = Path("/var/tmp/codriving-joint-latency")
STRICT_EFFECTIVE_ARMS = ((200, 300), (200, 400), (200, 500), (400, 500))
GPU_HANDOFF_POLL_SECONDS = 2


def select_tasks_by_arms(tasks, arms):
    """Return manifest tasks belonging to an exact, explicitly ordered arm set."""
    arms = tuple(arms)
    selected = set(arms)
    if len(selected) != len(arms):
        raise ValueError("latency arms must be unique")
    result = [task for task in tasks
              if (task["rsu_delay_ms"], task["ego_delay_ms"]) in selected]
    observed = {(task["rsu_delay_ms"], task["ego_delay_ms"]) for task in result}
    if observed != selected:
        raise ValueError("requested latency arms are not present in the manifest")
    return result


def create_joint_cohort(output_root, cohort=None):
    if cohort is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cohort = "joint-full-%s-p%d" % (timestamp, os.getpid())
    return create_cohort(output_root, cohort)


def parse_gpu_pool(value):
    if value is None or not re.match(r"^[1-8]$", value):
        raise ValueError("GPU_POOL must be an integer count from 1 through 8")
    return int(value)


def select_idle_gpus(output, count=4):
    idle = parse_idle_gpus(output)
    if len(idle) < count:
        raise RuntimeError("need %d idle GPUs but found %d" % (count, len(idle)))
    return idle[:count]


def query_idle_gpus(count):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: %s" % result.stdout.strip())
    return select_idle_gpus(result.stdout, count)


def query_gpu_state(gpu):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("GPU %s query failed: %s" % (gpu, result.stdout.strip()))
    match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", result.stdout)
    if not match:
        raise RuntimeError("GPU %s state unreadable: %s" % (gpu, result.stdout.strip()))
    return int(match.group(1)), int(match.group(2))


def wait_for_gpu_memory_release(gpu, stopping):
    """Wait interruptibly until the assigned GPU passes the joint memory gate."""
    while not stopping.is_set():
        try:
            _utilization, memory = query_gpu_state(gpu)
            if memory <= 50:
                return True
        except RuntimeError:
            pass
        if stopping.wait(GPU_HANDOFF_POLL_SECONDS):
            break
    return False


def build_batches(tasks):
    grouped = collections.OrderedDict()
    for task in tasks:
        identity = (task["scenario"], task["rsu_delay_ms"], task["ego_delay_ms"],
                    task["carla_seed"], task["traffic_seed"], task["repeat"])
        grouped.setdefault(identity, []).append(task)
    batches = []
    for index, (identity, members) in enumerate(grouped.items()):
        scenario, rsu, ego, carla_seed, traffic_seed, repeat = identity
        batches.append({
            "index": index,
            "key": "sc%s_rsu%d_ego%d_cseed%d_tseed%d_rep%d" % (
                scenario.lstrip("_"), rsu, ego, carla_seed, traffic_seed, repeat),
            "scenario": scenario, "rsu_delay_ms": rsu, "ego_delay_ms": ego,
            "carla_seed": carla_seed, "traffic_seed": traffic_seed,
            "repeat": repeat, "routes": [task["route"] for task in members],
            "task_keys": [task["key"] for task in members],
            "cadence": 2 if all(task["key"].startswith("jointc2_")
                                for task in members) else 4,
        })
    return batches


def plan_retry_batches(batches, first_results):
    """Retry only routes without verified first-pass evidence."""
    accepted = set(first_results)
    retry = []
    for batch in batches:
        missing = [(route, key) for route, key in
                   zip(batch["routes"], batch["task_keys"]) if key not in accepted]
        if missing:
            retry.append(dict(batch, routes=[route for route, _ in missing],
                              task_keys=[key for _, key in missing]))
    return retry


def runner_command(batch, gpu, port, cohort):
    return [
        "bash", "scripts/dual_fifo_multiroute_sc.sh",
        "--joint-c2" if batch.get("cadence") == 2 else "--joint",
        "--scenario", batch["scenario"],
        "--carla-seed", str(batch["carla_seed"]),
        "--traffic-seed", str(batch["traffic_seed"]),
        "--cohort", cohort, "--max-attempts", "1",
        str(gpu), str(port), str(batch["rsu_delay_ms"]),
        str(batch["ego_delay_ms"]), " ".join(map(str, batch["routes"])),
        str(batch["repeat"]),
    ]


def runner_environment(run_root):
    return dict(os.environ, DUAL_FIFO_RUN_ROOT=str(Path(run_root).resolve()))


def build_preview(batches, worker_count, base_port, cohort):
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    preview = []
    for batch in batches:
        port = base_port + batch["index"] * BATCH_PORT_STRIDE
        route_ports = [port + index * ROUTE_PORT_STRIDE
                       for index in range(len(batch["routes"]))]
        if (base_port < 1 or len(batch["routes"]) > 19 or
                not route_ports or route_ports[-1] + 5 > 65535):
            raise ValueError("base port range is invalid")
        item = dict(batch, slot=batch["index"] % worker_count,
                    port=port, route_ports=route_ports,
                    command=runner_command(batch, "SLOT%d" % (batch["index"] % worker_count),
                                           port, cohort))
        preview.append(item)
    return preview


class SweepExecutor:
    def __init__(self, batches, gpus, base_port, cohort_dir, cohort, round_name):
        self.pending = queue.Queue()
        for batch in batches:
            self.pending.put(batch)
        self.gpus, self.base_port = gpus, base_port
        self.cohort_dir, self.cohort = Path(cohort_dir), cohort
        self.run_root = self.cohort_dir / "runner_runs"
        self.round_name = round_name
        self.lock, self.stopping = threading.Lock(), threading.Event()
        self.active = {}

    def append_event(self, event, **fields):
        payload = dict(event=event, round=self.round_name, time_utc=utc_now(), **fields)
        with self.lock:
            with (self.cohort_dir / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")

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
            if not wait_for_gpu_memory_release(gpu, self.stopping):
                self.pending.task_done()
                return
            port = self.base_port + batch["index"] * BATCH_PORT_STRIDE
            command = runner_command(batch, gpu, port, self.cohort)
            log_path = self.cohort_dir / "logs" / (
                "%s.%s.log" % (batch["key"], self.round_name))
            self.append_event("batch_started", batch_key=batch["key"], gpu=gpu,
                              port=port, command=command)
            with log_path.open("w") as log:
                try:
                    process = subprocess.Popen(command, cwd=str(REPO),
                                               env=runner_environment(self.run_root), stdout=log,
                                               stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    with self.lock:
                        self.active[slot] = process
                    return_code = process.wait()
                except OSError as error:
                    log.write("[ABORT] runner could not start: %s\n" % error)
                    return_code = 127
                finally:
                    with self.lock:
                        self.active.pop(slot, None)
            match = RUN_DIR_RE.search(log_path.read_text(errors="replace"))
            self.append_event("batch_finished", batch_key=batch["key"], gpu=gpu,
                              port=port, return_code=return_code,
                              run_dir=match.group(1) if match else None,
                              log=str(log_path))
            self.pending.task_done()

    def run(self):
        threads = [threading.Thread(target=self.worker, args=(slot, gpu), daemon=True)
                   for slot, gpu in enumerate(self.gpus)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


def _positive(value):
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return bool(value)


def _verified_row(row, task, commit):
    if row["countable"] != "True":
        raise ValueError("runner marked route INFRA_INVALID")
    if (int(row["route"]) != task["route"] or
            int(row["repeat"]) != task["repeat"] or
            int(row["rsu_fifo_delay_ms"]) != task["rsu_delay_ms"] or
            int(row["ego_output_fifo_delay_ms"]) != task["ego_delay_ms"]):
        raise ValueError("route/repeat/delay mismatch")
    if Path(row["agent_config"]).name != "pnp_config_%s.yaml" % task["agent"]:
        raise ValueError("agent config mismatch")
    if row["git_commit"] != commit:
        raise ValueError("git provenance mismatch")
    for field in ("ego_json", "rsu_audit_csv", "ego_output_audit_csv"):
        if not Path(row[field]).is_file():
            raise ValueError("missing %s" % field)
    checkpoint = json.loads(Path(row["ego_json"]).read_text())["_checkpoint"]
    record = checkpoint["global_record"]
    status = row["status"]
    if status not in TERMINAL_DRIVING_STATUSES or record["status"] != status:
        raise ValueError("driving status mismatch")
    ds, rc = float(row["DS"]), float(row["RC"])
    if not math.isfinite(ds) or not math.isfinite(rc):
        raise ValueError("non-finite score")
    infractions = record.get("infractions") or {}
    timeout = _positive(infractions.get("route_timeout")) or status == "Failed - Agent timed out"
    return {"task_key": task["key"], "status": status,
            "DS": 0.0 if timeout else ds, "RC": rc,
            "route_timeout": timeout,
            "vehicle_blocked": _positive(infractions.get("vehicle_blocked")),
            "collision": any(_positive(value) for name, value in infractions.items()
                             if name.startswith("collisions_"))}


def joint_arm_label(task):
    """Return the runner/result arm namespace declared by an official manifest task."""
    agent = task.get("agent", "")
    if "_jointc2_" in agent:
        namespace = "jointc2"
    elif "_joint_" in agent:
        namespace = "joint"
    else:
        raise ValueError("task agent has no joint namespace")
    if not task.get("key", "").startswith(namespace + "_"):
        raise ValueError("task key/agent joint namespace mismatch")
    return "%s_rsu%d_ego%d" % (namespace, task["rsu_delay_ms"], task["ego_delay_ms"])


def collect_verified_results(cohort_dir, batches, round_name):
    cohort_dir = Path(cohort_dir)
    tasks = {task["key"]: task for task in
             json.loads((cohort_dir / "manifest.json").read_text())}
    metadata = json.loads((cohort_dir / "metadata.json").read_text())
    events_path = cohort_dir / "events.jsonl"
    events = (json.loads(line) for line in events_path.read_text().splitlines()
              if line.strip()) if events_path.is_file() else ()
    finished = collections.defaultdict(list)
    for event in events:
        if event.get("event") == "batch_finished" and event.get("round") == round_name:
            finished[event.get("batch_key")].append(event)
    accepted, invalid = {}, {}
    for batch in batches:
        keys = batch["task_keys"]
        invalid.update({key: "batch did not finish" for key in keys})
        matches = finished.get(batch["key"], [])
        if len(matches) != 1 or not matches[0].get("run_dir"):
            continue
        event = matches[0]
        log_path = Path(event.get("log") or "")
        if not log_path.is_file():
            continue
        identity = RUN_IDENTITY_RE.search(log_path.read_text(errors="replace"))
        expected_task = tasks[keys[0]]
        expected = {
            "cohort": metadata["cohort"],
            "arm": joint_arm_label(expected_task),
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
        try:
            row_routes = [int(row["route"]) for row in rows]
        except (KeyError, TypeError, ValueError):
            invalid.update({key: "malformed summary route" for key in keys})
            continue
        if (len(row_routes) != len(route_keys) or
                len(row_routes) != len(set(row_routes)) or
                set(row_routes) != set(route_keys) or
                any(None in row for row in rows)):
            invalid.update({key: "summary route set mismatch" for key in keys})
            continue
        seen = set()
        for row in rows:
            try:
                route = int(row["route"])
                if route not in route_keys or route in seen:
                    raise ValueError("duplicate or unexpected route")
                seen.add(route)
                key = route_keys[route]
                accepted[key] = _verified_row(row, tasks[key], metadata["git_commit"])
                invalid.pop(key, None)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                if row.get("route", "").isdigit() and int(row["route"]) in route_keys:
                    key = route_keys[int(row["route"])]
                    accepted.pop(key, None)
                    invalid[key] = str(error)
    return accepted, invalid


def collect_partial_verified_results(cohort_dir, batches, round_name):
    """Revalidate complete route rows even when their parent batch was interrupted."""
    cohort_dir = Path(cohort_dir)
    tasks = {task["key"]: task for task in
             json.loads((cohort_dir / "manifest.json").read_text())}
    metadata = json.loads((cohort_dir / "metadata.json").read_text())
    events_path = cohort_dir / "events.jsonl"
    events = (json.loads(line) for line in events_path.read_text().splitlines()
              if line.strip()) if events_path.is_file() else ()
    finished = collections.defaultdict(list)
    for event in events:
        if event.get("event") == "batch_finished" and event.get("round") == round_name:
            finished[event.get("batch_key")].append(event)
    accepted, invalid = {}, {}
    for batch in batches:
        keys = batch["task_keys"]
        invalid.update({key: "route result missing" for key in keys})
        matches = finished.get(batch["key"], [])
        if len(matches) != 1 or not matches[0].get("run_dir"):
            continue
        event = matches[0]
        log_path = Path(event.get("log") or "")
        if not log_path.is_file():
            continue
        identity = RUN_IDENTITY_RE.search(log_path.read_text(errors="replace"))
        expected_task = tasks[keys[0]]
        expected = {
            "cohort": metadata["cohort"],
            "arm": joint_arm_label(expected_task),
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
        try:
            row_routes = [int(row["route"]) for row in rows]
        except (KeyError, TypeError, ValueError):
            invalid.update({key: "malformed summary route" for key in keys})
            continue
        if (len(row_routes) != len(set(row_routes)) or
                not set(row_routes).issubset(route_keys) or any(None in row for row in rows)):
            invalid.update({key: "summary route set mismatch" for key in keys})
            continue
        for row in rows:
            key = route_keys[int(row["route"])]
            try:
                accepted[key] = _verified_row(row, tasks[key], metadata["git_commit"])
                invalid.pop(key, None)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                accepted.pop(key, None)
                invalid[key] = str(error)
    return accepted, invalid


def load_reused_results(source_cohort, tasks, allow_partial=False,
                        require_manifest_match=False):
    """Revalidate and select accepted evidence from an earlier cohort."""
    source_cohort = Path(source_cohort).resolve()
    batches_path = source_cohort / "batches.json"
    retry_path = source_cohort / "retry_batches.json"
    if not batches_path.is_file():
        raise ValueError("reuse cohort has no batches.json: %s" % source_cohort)
    if require_manifest_match:
        source_tasks = json.loads((source_cohort / "manifest.json").read_text())
        if manifest_hash(source_tasks) != manifest_hash(tasks):
            raise ValueError("reuse cohort manifest does not match exact-age target")
    batches = json.loads(batches_path.read_text())
    retry_batches = json.loads(retry_path.read_text()) if retry_path.is_file() else []
    collector = collect_partial_verified_results if allow_partial else collect_verified_results
    first, first_invalid = collector(source_cohort, batches, "first")
    retry, retry_invalid = collector(source_cohort, retry_batches, "retry")
    selected = {task["key"] for task in tasks}
    inherited, inherited_invalid = {}, {}
    inherited_path = source_cohort / "reused_results.json"
    if inherited_path.is_file():
        persisted = json.loads(inherited_path.read_text())
        for key, value in persisted.items():
            try:
                if key not in selected or value.get("task_key") != key:
                    raise ValueError("inherited task identity mismatch")
                if value.get("status") not in TERMINAL_DRIVING_STATUSES:
                    raise ValueError("inherited driving status invalid")
                ds, rc = float(value["DS"]), float(value["RC"])
                if not math.isfinite(ds) or not math.isfinite(rc):
                    raise ValueError("inherited score non-finite")
                inherited[key] = value
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                inherited_invalid[key] = str(error)
    merged = dict(inherited)
    merged.update({key: value for key, value in first.items() if key in selected})
    merged.update({key: value for key, value in retry.items()
                   if key in selected and key not in first})
    report = {
        "source_cohort": str(source_cohort),
        "source_inherited_accepted": len(inherited),
        "source_inherited_invalid": len(inherited_invalid),
        "source_first_accepted": len(first),
        "source_retry_accepted": len(retry),
        "source_first_invalid": len(first_invalid),
        "source_retry_invalid": len(retry_invalid),
        "accepted": len(merged),
    }
    return merged, report


def summarize_tasks(tasks, first, retry, reused=None):
    reused = reused or {}
    outcomes = []
    for task in tasks:
        key = task["key"]
        result = reused.get(key) or first.get(key) or retry.get(key)
        outcomes.append(dict(
            task_key=key, scenario=task["scenario"], route=task["route"],
            rsu_delay_ms=task["rsu_delay_ms"], ego_delay_ms=task["ego_delay_ms"],
            outcome=("reused" if key in reused else
                     "first_pass" if key in first else
                     "retry" if key in retry else "infra_failure_zeroed"),
            status=result["status"] if result else "INFRA_INVALID",
            DS=result["DS"] if result else 0.0,
            RC=result["RC"] if result else None,
            route_timeout=result.get("route_timeout", False) if result else False,
            vehicle_blocked=result.get("vehicle_blocked", False) if result else False,
            collision=result.get("collision", False) if result else False,
        ))
    by_arm = {}
    for task in tasks:
        arm = joint_arm_label(task)
        by_arm.setdefault(arm, [])
    for task, outcome in zip(tasks, outcomes):
        arm = joint_arm_label(task)
        by_arm[arm].append(outcome)
    arm_summary = {}
    for arm, rows in by_arm.items():
        valid = [row for row in rows if row["outcome"] != "infra_failure_zeroed"]
        zeroed = len(rows) - len(valid)
        arm_summary[arm] = {
            "planned": len(rows), "countable": len(valid),
            "reused": sum(row["outcome"] == "reused" for row in rows),
            "zeroed_after_retry": zeroed,
            "infra_failure_rate": zeroed / len(rows),
            "first_pass_failure_rate": sum(row["outcome"] in ("retry", "infra_failure_zeroed")
                                           for row in rows) / len(rows),
            "mean_DS": sum(row["DS"] for row in rows) / len(rows),
            "mean_RC": sum(row["RC"] for row in valid) / len(valid) if valid else None,
            "Completed": sum(row["status"] == "Completed" for row in valid),
            "Failed": sum(row["status"].startswith("Failed") for row in valid),
            "route_timeout": sum(row["route_timeout"] for row in valid),
            "vehicle_blocked": sum(row["vehicle_blocked"] for row in valid),
            "collision_episodes": sum(row["collision"] for row in valid),
        }
    countable = sum(row["outcome"] != "infra_failure_zeroed" for row in outcomes)
    summary = {
        "planned": len(tasks), "countable": countable,
        "reused": sum(row["outcome"] == "reused" for row in outcomes),
        "zeroed_after_retry": len(tasks) - countable,
        "infra_failure_rate": (len(tasks) - countable) / len(tasks) if tasks else None,
        "mean_DS": sum(row["DS"] for row in outcomes) / len(tasks) if tasks else None,
        "by_arm": arm_summary,
        "zeroed_task_keys": [row["task_key"] for row in outcomes
                             if row["outcome"] == "infra_failure_zeroed"],
    }
    return summary, outcomes


def _finished_keys(cohort_dir, round_name):
    path = Path(cohort_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [event.get("batch_key") for event in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())
            if event.get("event") == "batch_finished" and event.get("round") == round_name]


def summarize_cohort(cohort_dir):
    cohort_dir = Path(cohort_dir)
    tasks = json.loads((cohort_dir / "manifest.json").read_text())
    batches = json.loads((cohort_dir / "batches.json").read_text())
    retry_path = cohort_dir / "retry_batches.json"
    retry_batches = json.loads(retry_path.read_text()) if retry_path.is_file() else []
    reused_path = cohort_dir / "reused_results.json"
    reused = json.loads(reused_path.read_text()) if reused_path.is_file() else {}
    first, first_invalid = collect_verified_results(cohort_dir, batches, "first")
    retry, retry_invalid = collect_verified_results(cohort_dir, retry_batches, "retry")
    summary, outcomes = summarize_tasks(tasks, first, retry, reused)
    retry_expected = {key for batch in plan_retry_batches(batches, first)
                      for key in batch["task_keys"]}
    retry_planned = {key for batch in retry_batches for key in batch["task_keys"]}
    summary.update({
        "cohort_dir": str(cohort_dir),
        "first_pass_failures": len(first_invalid),
        "first_pass_finished_batches": len(_finished_keys(cohort_dir, "first")),
        "retry_finished_batches": len(_finished_keys(cohort_dir, "retry")),
        "retry_invalid": len(retry_invalid),
        "complete": (len(_finished_keys(cohort_dir, "first")) == len(batches) and
                     len(_finished_keys(cohort_dir, "retry")) == len(retry_batches) and
                     retry_expected == retry_planned and
                     summary["countable"] + summary["zeroed_after_retry"] == len(tasks)),
    })
    return summary, outcomes, first_invalid, retry_invalid


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
    parser.add_argument("--base-port", type=int, default=44000)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cohort")
    parser.add_argument("--strict-effective", action="store_true",
                        help="run only the four arms with strict actual action > RSU age")
    parser.add_argument("--exact-age-grid", action="store_true",
                        help="run the official cadence=2 ten-arm exact-age grid")
    parser.add_argument("--reuse-cohort",
                        help="revalidate and reuse accepted evidence from this cohort")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.check_results:
            summary, outcomes, _, _ = summarize_cohort(args.check_results)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if summary["complete"] else 1
        worker_count = parse_gpu_pool(os.environ.get("GPU_POOL"))
        if args.exact_age_grid and args.strict_effective:
            raise ValueError("--exact-age-grid cannot be combined with --strict-effective")
        cadence = 2 if args.exact_age_grid else 4
        tasks = build_manifest(
            args.seed if args.carla_seed is None else args.carla_seed,
            args.seed if args.traffic_seed is None else args.traffic_seed,
            args.repeat, cadence=cadence)
        if args.reuse_cohort and not (args.strict_effective or args.exact_age_grid):
            raise ValueError("--reuse-cohort requires --strict-effective or --exact-age-grid")
        if args.strict_effective:
            tasks = select_tasks_by_arms(tasks, STRICT_EFFECTIVE_ARMS)
        reused, reuse_report = ({}, None)
        if args.reuse_cohort:
            reused, reuse_report = load_reused_results(
                args.reuse_cohort, tasks, allow_partial=args.exact_age_grid,
                require_manifest_match=args.exact_age_grid)
        all_batches = build_batches(tasks)
        batches = plan_retry_batches(all_batches, reused)
        gpus = None if args.dry_run else query_idle_gpus(worker_count)
        cohort_dir = create_joint_cohort(args.output_root, args.cohort)
        cohort = cohort_dir.name
        preview = build_preview(batches, worker_count, args.base_port, cohort)
        build_preview(batches, worker_count, args.base_port + RETRY_PORT_OFFSET, cohort)
        metadata = {
            "cohort": cohort, "created_utc": utc_now(), "result_policy": "two_pass_zero_ds",
            "output_root": str(cohort_dir.parent.resolve()),
            "run_root": str((cohort_dir / "runner_runs").resolve()),
            "manifest_sha256": manifest_hash(tasks), "planned_tasks": len(tasks),
            "planned_batches": len(batches), "matrix_batches": len(all_batches),
            "reused_tasks": len(reused),
            "remaining_tasks": sum(len(batch["task_keys"]) for batch in batches),
            "strict_effective": args.strict_effective,
            "strict_effective_arms": [list(arm) for arm in STRICT_EFFECTIVE_ARMS]
            if args.strict_effective else None,
            "reuse_cohort": args.reuse_cohort, "reuse_report": reuse_report,
            "worker_count": worker_count,
            "cadence": cadence, "exact_age_grid": args.exact_age_grid,
            "independent_exact_age_cohort": args.exact_age_grid and not args.reuse_cohort,
            "cohort_origin": ("exact_age_resume" if args.exact_age_grid and args.reuse_cohort
                              else "independent_exact_age" if args.exact_age_grid
                              else "standard_joint"),
            "base_port": args.base_port,
            "retry_base_port": args.base_port + RETRY_PORT_OFFSET,
            "code_sha256": {name: file_sha256(REPO / "scripts" / name) for name in
                            ("joint_latency_full_manifest.py", "joint_latency_full_sweep.py",
                             "dual_fifo_multiroute_sc.sh", "dual_fifo_result.py")},
        }
        metadata.update(git_provenance())
        write_json(cohort_dir / "manifest.json", tasks)
        write_json(cohort_dir / "batches.json", preview)
        write_json(cohort_dir / "reused_results.json", reused)
        write_json(cohort_dir / "metadata.json", metadata)
        if args.dry_run:
            print(json.dumps({"mode": "dry-run", "cohort_dir": str(cohort_dir),
                              "planned_tasks": len(tasks), "planned_batches": len(batches),
                              "reused_tasks": len(reused),
                              "remaining_tasks": metadata["remaining_tasks"],
                              "manifest_sha256": metadata["manifest_sha256"]}, sort_keys=True))
            return 0
        metadata["selected_gpus"] = gpus
        write_json(cohort_dir / "metadata.json", metadata)
        first_executor = SweepExecutor(batches, gpus, args.base_port, cohort_dir,
                                       cohort, "first")
        previous = {signum: signal.getsignal(signum) for signum in
                    (signal.SIGINT, signal.SIGTERM)}
        try:
            for signum in previous:
                signal.signal(signum, lambda _signum, _frame: first_executor.terminate_active())
            print(json.dumps({"mode": "run", "cohort_dir": str(cohort_dir),
                              "planned_tasks": len(tasks), "planned_batches": len(batches),
                              "reused_tasks": len(reused),
                              "remaining_tasks": metadata["remaining_tasks"],
                              "selected_gpus": gpus}, sort_keys=True), flush=True)
            first_executor.run()
            first, first_invalid = collect_verified_results(cohort_dir, preview, "first")
            write_json(cohort_dir / "first_pass_failures.json", first_invalid)
            retry_batches = plan_retry_batches(batches, first)
            retry_preview = build_preview(retry_batches, worker_count,
                                          args.base_port + RETRY_PORT_OFFSET, cohort)
            write_json(cohort_dir / "retry_batches.json", retry_preview)
            if retry_batches and not first_executor.stopping.is_set():
                retry_executor = SweepExecutor(retry_batches, gpus,
                                               args.base_port + RETRY_PORT_OFFSET,
                                               cohort_dir, cohort, "retry")
                for signum in previous:
                    signal.signal(signum, lambda _signum, _frame: retry_executor.terminate_active())
                retry_executor.run()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        summary, outcomes, _, retry_invalid = summarize_cohort(cohort_dir)
        write_json(cohort_dir / "retry_failures.json", retry_invalid)
        write_json(cohort_dir / "task_outcomes.json", outcomes)
        write_json(cohort_dir / "final_summary.json", summary)
        print(json.dumps({"cohort_dir": str(cohort_dir), "complete": summary["complete"],
                          "countable": summary["countable"],
                          "zeroed_after_retry": summary["zeroed_after_retry"],
                          "infra_failure_rate": summary["infra_failure_rate"]},
                         sort_keys=True), flush=True)
        return 0 if summary["complete"] else 1
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("[ABORT] %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
