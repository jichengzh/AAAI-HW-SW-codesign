#!/usr/bin/env python3
"""Runner hooks for a new Orin service stack and evidence per episode attempt."""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


def episode_context(environ):
    names = ("RUN_DIR", "TASK_ID", "ARM_ID", "COHORT_ID", "ATTEMPT")
    context = {}
    for name in names:
        value = environ.get("COPT_HIL_" + name)
        if not value:
            raise ValueError("missing runner episode context: " + name)
        context[name.lower()] = value
    if not context["attempt"].isdigit() or int(context["attempt"]) <= 0:
        raise ValueError("attempt must be positive")
    identity = json.dumps(context, sort_keys=True).encode("utf-8")
    context["token"] = hashlib.sha256(identity).hexdigest()[:24] + "-a" + context["attempt"]
    return context


def checked_command(command, **kwargs):
    return subprocess.run(command, check=True, timeout=240, **kwargs)


def remote_command(args, command):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.remote,
            " ".join(shlex.quote(str(item)) for item in command)]


def service_configuration(args, context):
    drain_timeout = getattr(args, "completion_drain_timeout_s", 135)
    if (type(drain_timeout) not in (int, float) or not math.isfinite(drain_timeout)
            or drain_timeout <= 0):
        raise ValueError("completion drain timeout must be finite and positive")
    depth = getattr(args, "pipeline_depth", 0)
    if type(depth) is not int or depth not in (0, 2):
        raise ValueError("pipeline depth must be 0 or 2")
    if depth and not getattr(args, "serial_optimized", False):
        raise ValueError("pipeline requires serial optimized mode")
    if getattr(args, "serial_optimized", False):
        if getattr(args, "wire_compression", "none") != "none":
            raise ValueError("serial results must be uncompressed")
        if getattr(args, "trigger_policy", "fifo") != "fifo":
            raise ValueError("serial results require exact FIFO triggers")
        warmup_count = getattr(args, "runtime_warmup_count", 1)
        if type(warmup_count) is not int or warmup_count < 0:
            raise ValueError("runtime warmup count must be a non-negative integer")
        if not context.get("arm_id"):
            raise ValueError("serial startup requires the episode arm_id")
    if type(args.result_count) is not int or args.result_count <= 0:
        raise ValueError("result count must be a positive integer")
    runtime = args.runtime.rstrip("/")
    oldroot = str(Path(runtime).parent)
    data_root = getattr(args, "data_root", None) or oldroot + "/input"
    input_manifest = getattr(args, "input_manifest", None) or oldroot + "/input/local_input_manifest.json"
    container_runtime = (getattr(args, "container_runtime", None) or "/work").rstrip("/")
    native_repo_root = getattr(args, "native_repo_root", None) or oldroot + "/native/V2Xverse"
    host_state_root = getattr(args, "host_state_root", None) or oldroot + "/runtimestate/p4-recovery"
    container_state_root = getattr(args, "container_state_root", None) or "/state/p4-recovery"
    host_state = host_state_root.rstrip("/") + "/" + context["token"]
    container_state = container_state_root.rstrip("/") + "/" + context["token"]
    container_script = container_runtime + "/scripts/coptv2x_hil_orin_services.py"
    services = []

    def docker_service(name, command, ready=None):
        owner = container_state + "/" + name + ".owner.json"
        launcher = ["python3", container_script, "--mode", "child", "--owner-file", owner,
                    "--command-json", json.dumps(command)]
        shell = "source /opt/ros/foxy/setup.bash && exec " + " ".join(shlex.quote(item) for item in launcher)
        return {"name": name, "argv": ["docker", "exec", "-e", "ROS_DOMAIN_ID=71", "-e",
                                         "PYTHONPATH=" + container_runtime + ":" + container_runtime + "/simulation/leaderboard/team_code/closedloop",
                                         args.container, "bash", "-lc", shell],
                "container": args.container, "container_state_dir": container_state,
                "container_script": container_script,
                "ready_file": host_state + "/" + ready if ready else None}

    publisher = ["python3", container_runtime + "/scripts/ditl_ros2_edge_bridge.py", "result-publisher",
                 "--result-port", str(args.result_port), "--max-results", str(args.result_count),
                 "--ready-file", container_state + "/publisher.ready", "--discovery-timeout-s", "90",
                 "--ack-timeout-s", "120", "--max-payload-mib", "64", "--socket-timeout-s", "300"]
    services.append(docker_service("publisher", publisher, "publisher.ready"))
    worker = [args.orin_python, runtime + "/scripts/coptv2x_hil_orin_worker.py",
              "--data-root", data_root, "--manifest", input_manifest,
              "--artifact-root", args.artifact_root, "--config-dir", runtime + "/simulation/leaderboard/team_code/agent_config/coptv2x_hil",
              "--native-repo-root", native_repo_root, "--worker-port", str(args.worker_port),
              "--result-port", str(args.result_port), "--result-count", str(args.result_count),
              "--max-samples", str(args.result_count), "--max-total-tensor-mib", "262144",
              "--max-payload-mib", "64", "--wire-compression", getattr(args, "wire_compression", "none"),
              "--ready-file", host_state + "/worker.ready"]
    if getattr(args, "serial_optimized", False):
        worker.extend(["--serial-optimized", "--startup-arm-id", context["arm_id"],
                       "--runtime-warmup-count", str(warmup_count)])
    if depth:
        worker.extend(["--pipeline-depth", str(depth)])
    if getattr(args, "preload_static_inputs", False):
        worker.extend(["--preload-static-inputs", "--max-preloaded-tensor-mib",
                       str(getattr(args, "max_preloaded_tensor_mib", 2048))])
    pythonpath = ":".join((runtime + "/pydeps", "/usr/lib/python3.8/dist-packages",
                           runtime + "/simulation/leaderboard/team_code/closedloop", runtime))
    services.append({"name": "worker", "argv": worker, "env": {"PYTHONPATH": pythonpath},
                     "container": None, "ready_file": host_state + "/worker.ready"})
    proxy = ["python3", container_runtime + "/scripts/coptv2x_hil_ros_proxy.py", "--forward-port", str(args.forward_port),
             "--worker-port", str(args.worker_port), "--queue-capacity", "4096",
             "--trigger-policy", getattr(args, "trigger_policy", "fifo")]
    if depth:
        proxy.extend(["--worker-pipeline-depth", str(depth)])
    services.append(docker_service("proxy", proxy))
    return {"state_dir": host_state, "ready_timeout_s": args.ready_timeout_s,
            "completion_policy": {"mode": "publisher_ack", "worker": "worker",
                                  "publisher": "publisher", "drain_timeout_s": drain_timeout},
            "episode": context, "services": services}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("start", "stop"), required=True)
    parser.add_argument("--remote", default="jichengzhi@172.16.62.222")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--runtime", default="/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/runtime")
    parser.add_argument("--container", default="coptv2x-hil-p4-smoke-ros")
    parser.add_argument("--orin-python", default="/home/jichengzhi/miniforge3/envs/uniad/bin/python")
    parser.add_argument("--artifact-root", default="/home/jichengzhi/lane_c_orin_stage6_five_config_20260724")
    parser.add_argument("--native-repo-root", help="explicit frozen native tree for a new runtime deployment")
    parser.add_argument("--container-runtime", default="/work", help="runtime path inside the existing ROS container")
    parser.add_argument("--host-state-root", help="host state root already mounted into the ROS container")
    parser.add_argument("--container-state-root", default="/state/p4-recovery")
    parser.add_argument("--wire-compression", choices=("none", "zlib", "lz4"), default="none")
    parser.add_argument("--serial-optimized", action="store_true")
    parser.add_argument("--pipeline-depth", type=int, choices=(0, 2), default=0)
    parser.add_argument("--runtime-warmup-count", type=int, default=1,
                        help="synthetic startup forwards for serial mode only; discarded before readiness")
    parser.add_argument("--preload-static-inputs", action="store_true",
                        help="bounded prerecorded replay initialization, not live sensing")
    parser.add_argument("--max-preloaded-tensor-mib", type=int, default=2048)
    parser.add_argument("--trigger-policy", choices=("fifo", "latest"), default="fifo")
    parser.add_argument("--data-root", help="Override Orin-local input directory; does not imply live observations")
    parser.add_argument("--input-manifest", help="Override Orin-local input manifest; P4 cache is the default")
    parser.add_argument("--forward-port", type=int, default=56811)
    parser.add_argument("--worker-port", type=int, default=56812)
    parser.add_argument("--result-port", type=int, default=55022)
    parser.add_argument("--result-count", type=int, default=2225)
    parser.add_argument("--ready-timeout-s", type=float, default=180)
    parser.add_argument("--completion-drain-timeout-s", type=float, default=135,
                        help="maximum drain after worker or ACK publisher exits; not episode duration")
    args = parser.parse_args(argv)
    for port in (args.forward_port, args.worker_port, args.result_port):
        if not 1 <= port <= 65535:
            parser.error("service port must be valid")
    context = episode_context(os.environ)
    config = service_configuration(args, context)
    remote_config = args.remote_root.rstrip("/") + "/episodes/" + context["token"] + "/services.json"
    script = args.runtime.rstrip("/") + "/scripts/coptv2x_hil_orin_services.py"
    if args.mode == "start":
        payload = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
        writer = ("import base64,pathlib; p=pathlib.Path(%r); p.parent.mkdir(parents=True,exist_ok=True);"
                  " pathlib.Path(%r).mkdir(parents=True,exist_ok=True); p.write_bytes(base64.b64decode(%r))"
                  % (remote_config, config["state_dir"], payload))
        checked_command(remote_command(args, [args.orin_python, "-c", writer]))
    failure = None
    try:
        checked_command(remote_command(args, [args.orin_python, script, "--mode", args.mode,
                                              "--config", remote_config]))
    except Exception as error:
        failure = error
        if args.mode == "start":
            try:
                checked_command(remote_command(args, [args.orin_python, script, "--mode", "stop", "--config", remote_config]))
            except Exception:
                pass
    finally:
        evidence = Path(context["run_dir"]) / "evidence" / "orin"
        evidence.mkdir(parents=True, exist_ok=True)
        checked_command(["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                         args.remote + ":" + config["state_dir"] + "/", str(evidence) + "/"])
    if failure:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
