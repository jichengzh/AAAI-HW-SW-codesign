#!/usr/bin/env python3
"""Own a fresh Orin publisher/worker/proxy stack for one HIL attempt."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        content = Path("/proc/%s/stat" % pid).read_text()
        return content.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def record_process(path, pid):
    write_json(path, {"pid": pid, "start_ticks": process_identity(pid)})


def kill_record(path, timeout_s=4):
    if not Path(path).is_file():
        return
    owner = json.loads(Path(path).read_text())
    pid = owner["pid"]
    if not owner.get("start_ticks") or process_identity(pid) != owner["start_ticks"]:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and process_identity(pid) == owner["start_ticks"]:
        time.sleep(0.05)
    if process_identity(pid) == owner["start_ticks"]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_services(config):
    state = Path(config["state_dir"])
    for service in reversed(config["services"]):
        owner_path = state / (service["name"] + ".owner.json")
        if service.get("container"):
            inside = service["container_state_dir"] + "/" + service["name"] + ".owner.json"
            command = ["docker", "exec", service["container"], "python3",
                       service["container_script"], "--mode", "kill-child",
                       "--owner-file", inside]
            subprocess.run(command, check=True, timeout=15)
        else:
            kill_record(owner_path)


def completion_policy(config):
    """Opt in finite workers; publisher exit zero means its ACK gate finished."""
    policy = config.get("completion_policy")
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy.get("mode") != "publisher_ack":
        raise ValueError("unsupported completion policy")
    names = [service["name"] for service in config["services"]]
    worker, publisher = policy.get("worker"), policy.get("publisher")
    if (not isinstance(worker, str) or not isinstance(publisher, str)
            or worker == publisher or names.count(worker) != 1 or names.count(publisher) != 1):
        raise ValueError("completion policy requires distinct owned worker and publisher")
    timeout = policy.get("drain_timeout_s")
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("completion drain timeout must be finite and positive")
    return dict(policy)


def completed_services(children, policy):
    finite = (policy["worker"], policy["publisher"]) if policy else ()
    completed = []
    for service, child in children:
        status = child.poll()
        if status is None:
            continue
        ready = service.get("ready_file")
        if (status != 0 or service["name"] not in finite
                or (ready and not Path(ready).is_file())):
            raise RuntimeError("%s exited with status %s" % (service["name"], status))
        completed.append(service["name"])
    return completed


def serve(config):
    policy = completion_policy(config)
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    children = []
    handles = []
    stopped = [False]
    previous = {}
    completed = False
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    record_process(state / "supervisor.owner.json", os.getpid())
    try:
        for service in config["services"]:
            ready = service.get("ready_file")
            if ready and Path(ready).exists():
                raise RuntimeError("fresh attempt already has service readiness marker")
            log = (state / (service["name"] + ".log")).open("ab", buffering=0)
            handles.append(log)
            child = subprocess.Popen(service["argv"], env=dict(os.environ, **service.get("env", {})),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((service, child))
            if not service.get("container"):
                record_process(state / (service["name"] + ".owner.json"), child.pid)
        deadline = time.monotonic() + config.get("ready_timeout_s", 180)
        while not stopped[0]:
            completed_services(children, policy)
            ready = all(not service.get("ready_file") or Path(service["ready_file"]).is_file()
                        for service, _ in children)
            if ready:
                write_json(state / "ready.json", {"ready": True, "services": [s["name"] for s, _ in children]})
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("service startup readiness deadline exceeded")
            time.sleep(0.1)
        drain_deadline = None
        while not stopped[0]:
            finished = completed_services(children, policy)
            if policy and len(finished) == 2:
                completed = True
                break
            if policy and finished:
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + policy["drain_timeout_s"]
                if time.monotonic() >= drain_deadline:
                    raise RuntimeError("finite service completion deadline exceeded")
            time.sleep(0.2)
    except Exception as error:
        write_json(state / "failure.json", {"error": str(error)})
        raise
    finally:
        cleanup_error = None
        try:
            stop_services(config)
        except Exception as error:
            cleanup_error = str(error)
        for _, child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for handle in handles:
            handle.close()
        write_json(state / "stopped.json", {"cleaned": cleanup_error is None, "error": cleanup_error})
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if completed:
        if cleanup_error:
            write_json(state / "failure.json", {"error": cleanup_error})
            raise RuntimeError("service completion cleanup failed: %s" % cleanup_error)
        write_json(state / "completed.json", {"completed": True, "mode": "publisher_ack",
                                             "services": [policy["worker"], policy["publisher"]]})


def start(config, config_path):
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    if (state / "supervisor.owner.json").exists():
        raise RuntimeError("attempt services already started; use a fresh attempt directory")
    with (state / "supervisor.log").open("ab") as output:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--mode", "serve",
                                  "--config", str(config_path)], stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + config.get("ready_timeout_s", 180) + 5
    while time.monotonic() < deadline:
        if (state / "failure.json").exists():
            raise RuntimeError("episode service startup failed; inspect supervisor and service logs")
        if (state / "ready.json").exists():
            if child.poll() is None or (state / "completed.json").exists():
                return
        if child.poll() is not None:
            raise RuntimeError("episode service startup failed; inspect supervisor and service logs")
        time.sleep(0.1)
    kill_record(state / "supervisor.owner.json")
    raise RuntimeError("episode services did not report readiness")


def stop(config):
    state = Path(config["state_dir"])
    kill_record(state / "supervisor.owner.json", timeout_s=15)
    if not (state / "stopped.json").exists():
        stop_services(config)
        write_json(state / "stopped.json", {"cleaned": True, "error": None})
    result = json.loads((state / "stopped.json").read_text())
    if not result["cleaned"]:
        raise RuntimeError("service cleanup incomplete: %s" % result.get("error"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("start", "serve", "stop", "child", "kill-child"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--owner-file")
    parser.add_argument("--command-json")
    args = parser.parse_args(argv)
    if args.mode == "child":
        if os.getpgrp() != os.getpid():
            os.setsid()
        record_process(Path(args.owner_file), os.getpid())
        command = json.loads(args.command_json)
        os.execvpe(command[0], command, os.environ.copy())
    elif args.mode == "kill-child":
        kill_record(args.owner_file)
    else:
        config = json.loads(Path(args.config).read_text())
        {"start": lambda: start(config, args.config), "stop": lambda: stop(config),
         "serve": lambda: serve(config)}[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
