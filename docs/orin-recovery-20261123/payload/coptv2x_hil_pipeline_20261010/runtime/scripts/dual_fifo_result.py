#!/usr/bin/env python
"""Parse dual-FIFO route outcomes without counting infrastructure failures."""

import argparse
from bisect import bisect_right
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


SIM_RATE_HZ = 20.0
TERMINAL_DRIVING_STATUSES = frozenset((
    "Completed", "Failed", "Failed - Agent timed out",
    "Failed - Agent got blocked", "Failed - Agent deviated from the route",
))
FIELDS = (
    "route,repeat,rsu_fifo_delay_ms,rsu_delay_frames,"
    "ego_output_fifo_delay_ms,ego_delay_frames,DS,RC,status,countable,"
    "coll_lay,coll_veh,ego_json,rsu_audit_csv,ego_output_audit_csv,"
    "agent_config,sim_rate_hz,git_commit,git_dirty"
).split(",")
INFRA_PATTERN = re.compile(
    r"(?:agent|simulation|run scenario) crashed|Traceback \(most recent call last\)|"
    r"CUDA[^\n]*(?:error|out of memory|fail)|"
    r"(?:connection refused|failed to connect|connection timed out)|"
    r"time-out of .*while waiting for the simulator|"
    r"CARLA[^\n]*(?:error|timeout|failed)|"
    r"audit[^\n]*(?:dump|flush|close)[^\n]*(?:fail|error)|"
    r"audit[^\n]*(?:fail|error)[^\n]*(?:dump|flush|close)", re.IGNORECASE,
)
RSU_HEADERS = ("consume_frame submitted_rsu_source_frame submitted_rsu_ready_frame "
               "used_rsu_source_frame used_rsu_ready_frame configured_delay_ms delay_frames "
               "information_age_frames is_zoh_held is_ego_only_startup "
               "source_sim_time_s consume_sim_time_s step delta "
               "used_rsu_frame_id latency_ms_source zoh_age_frames").split()
RSU_SIM_TIME_HEADERS = ("source_sim_time_s", "consume_sim_time_s")
LEGACY_RSU_HEADERS = [key for key in RSU_HEADERS if key not in RSU_SIM_TIME_HEADERS]
EGO_HEADERS = ("apply_frame generated_source_frame generated_ready_frame applied_source_frame "
               "applied_ready_frame configured_delay_ms delay_frames control_age_frames "
               "is_startup_neutral generated_throttle generated_steer generated_brake "
               "applied_throttle applied_steer applied_brake step source_step delay_ms "
               "age_frames is_startup_hold").split()


def infrastructure_invalid(error):
    return {"status": "INFRA_INVALID", "DS": 0.0, "RC": 0.0,
            "coll_lay": 0.0, "coll_veh": 0.0, "countable": False,
            "error": str(error)}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("route metric must be a JSON number, got %r" % value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite route metric")
    return number


def _infraction_value(value):
    if isinstance(value, list):
        return float(len(value))
    return _number(0.0 if value is None else value)


def _required_scores(record):
    scores = record.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scores must be a mapping containing score_composed and score_route")
    return _number(scores["score_composed"]), _number(scores["score_route"])


def _validate_terminal_status(status):
    if not isinstance(status, str) or status not in TERMINAL_DRIVING_STATUSES:
        raise ValueError("unsupported terminal driving status: %r" % status)


def _validate_checkpoint_statuses(checkpoint, record):
    _validate_terminal_status(record["status"])
    for route_record in checkpoint.get("records", []):
        _validate_terminal_status(route_record["status"])
    for exception in record.get("meta", {}).get("exceptions", []):
        _validate_terminal_status(exception[2])


def _diagnostic_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _diagnostic_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _diagnostic_strings(item)


def parse_route_result(path):
    """Keep raw driving scores, except explicit route timeouts have DS=0."""
    try:
        with open(path) as stream:
            checkpoint = json.load(stream)["_checkpoint"]
        record = checkpoint["global_record"]
        if not isinstance(record, dict) or not record.get("status"):
            raise ValueError("missing or malformed global_record/status")
        status = record["status"]
        _validate_checkpoint_statuses(checkpoint, record)
        # Only diagnostic fields may carry infrastructure errors. Matching the
        # whole JSON also joins unrelated values and keys on one line (for
        # example carlamotors.carlacola followed by route_timeout).
        for meta in (record.get("meta"), checkpoint.get("meta")):
            if isinstance(meta, dict):
                for key in ("error", "errors", "exception", "exceptions", "traceback"):
                    for diagnostic in _diagnostic_strings(meta.get(key)):
                        if INFRA_PATTERN.search(diagnostic):
                            raise ValueError("infrastructure failure in checkpoint %s: %s" %
                                             (key, diagnostic))
        ds, rc = _required_scores(record)
        infractions = record.get("infractions", {})
        timeout = (_infraction_value(infractions.get("route_timeout", 0.0)) > 0 or
                   status == "Failed - Agent timed out")
        return {"status": status, "DS": 0.0 if timeout else ds,
                "RC": rc,
                "coll_lay": _infraction_value(infractions.get("collisions_layout", 0.0)),
                "coll_veh": _infraction_value(infractions.get("collisions_vehicle", 0.0)),
                "countable": True}
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        return infrastructure_invalid("%s: %s" % (path, error))


def _audit_path(path, route, repeat, filename):
    """Resolve from the JSON's experiment tree, never the ambient cwd."""
    route_name = "r%s_repeat%s" % (route, repeat)
    route_dir = next((parent for parent in Path(path).absolute().parents
                      if parent.name == route_name), None)
    if route_dir is None:
        return ""
    root = route_dir.parents[2]
    image_route = root / "image" / route_dir.relative_to(root)
    audits = sorted(image_route.glob("*/meta/" + filename))
    return str(audits[0]) if len(audits) == 1 else ""


def _read_audit(path, required):
    content = Path(path).read_text()
    if not content or not content.endswith("\n"):
        raise ValueError("empty or truncated audit: " + path)
    records = list(csv.reader(io.StringIO(content), strict=True))
    header = records[0]
    if len(header) != len(set(header)) or not set(required).issubset(header):
        raise ValueError("missing/duplicate audit headers: " + path)
    if len(records) < 2 or any(len(row) != len(header) for row in records[1:]):
        raise ValueError("empty, truncated or malformed audit rows: " + path)
    return [dict(zip(header, row)) for row in records[1:]]


def _csv_number(row, key):
    number = float(row[key])
    if not math.isfinite(number):
        raise ValueError("non-finite audit field: " + key)
    return number


def _expect_fields(row, expected):
    for key, value in expected.items():
        actual = row[key]
        if isinstance(value, bool):
            matches = actual == str(value)
        elif value is None:
            matches = actual == ""
        elif isinstance(value, (int, float)):
            matches = _csv_number(row, key) == value
        else:
            matches = actual == value
        if not matches:
            raise ValueError("audit %s mismatch: expected %r got %r" % (key, value, actual))


def _audit_config(config, rsu, ego):
    import yaml
    single_variable = (rsu in (0, 200, 500, 800) and
                       ego in (0, 200, 500, 800) and not (rsu and ego))
    joint = (rsu in (100, 200, 300, 400) and
             ego in (200, 300, 400, 500) and ego > rsu)
    if not (single_variable or joint):
        raise ValueError("unsupported audit delay pair")
    joint_c2 = joint and Path(config).name == (
        "pnp_config_codriving_dualfifo_jointc2_rsu%d_ego%d_l1.yaml" % (rsu, ego))
    arm = ("jointc2_rsu%d_ego%d" % (rsu, ego) if joint_c2 else
           "joint_rsu%d_ego%d" % (rsu, ego) if joint else
           "rsu%d" % rsu if rsu else "ego%d" % ego if ego else "baseline")
    if Path(config).name != "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm:
        raise ValueError("agent config does not match expected arm")
    try:
        with open(config) as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError("malformed agent config: %s" % error)
    if not isinstance(document, dict) or not isinstance(document.get("simulation"), dict):
        raise ValueError("agent config and simulation must be mappings")
    simulation = document["simulation"]
    for key, expected in (("rsu_fifo_delay_ms", rsu), ("ego_output_fifo_delay_ms", ego),
                          ("ego_num", 1), ("tau_ego_ms", 0), ("latency_audit", True)):
        actual = simulation.get(key)
        if actual != expected or (isinstance(actual, bool) != isinstance(expected, bool)):
            raise ValueError("audit/config mismatch: " + key)
    skip = simulation.get("skip_frames")
    if type(skip) is not int or skip <= 0:
        raise ValueError("invalid skip_frames in agent config")
    return skip


def _validate_ego_audit(rows, delay):
    frames = delay // 50
    generated = [tuple(_csv_number(row, "generated_" + key) for key in
                       ("throttle", "steer", "brake")) for row in rows]
    for frame, row in enumerate(rows):
        source = frame - frames if frame >= frames else None
        startup = source is None
        _expect_fields(row, dict(apply_frame=frame, generated_source_frame=frame,
                                generated_ready_frame=frame + frames,
                                applied_source_frame=source,
                                applied_ready_frame=None if startup else frame,
                                configured_delay_ms=delay, delay_frames=frames,
                                control_age_frames=None if startup else frames,
                                is_startup_neutral=startup, step=frame, source_step=source,
                                delay_ms=delay, age_frames=None if startup else frames,
                                is_startup_hold=startup))
        applied = tuple(_csv_number(row, "applied_" + key) for key in ("throttle", "steer", "brake"))
        if applied != ((0., 0., 0.) if startup else generated[source]):
            raise ValueError("applied controls do not match neutral/generated source at frame %d" % frame)


def _validate_rsu_audit(rows, delay, ego_count, skip, validate_sim_time=True):
    frames = delay // 50
    calls = [frame for frame in range(ego_count) if frame <= skip or frame % skip == 0]
    if len(rows) != len(calls):
        raise ValueError("RSU audit inference coverage differs from recorded ego ticks")
    previous = None
    for frame, row in zip(calls, rows):
        index = bisect_right(calls, frame - frames) - 1
        source = calls[index] if index >= 0 else None
        startup = source is None
        ready = None if startup else source + frames
        _expect_fields(row, dict(consume_frame=frame, submitted_rsu_source_frame=frame,
                                submitted_rsu_ready_frame=frame + frames,
                                used_rsu_source_frame=source, used_rsu_ready_frame=ready,
                                configured_delay_ms=None if startup else delay,
                                delay_frames=None if startup else frames,
                                information_age_frames=None if startup else frame - source,
                                is_zoh_held=source == previous, is_ego_only_startup=startup,
                                step=frame, delta=None if startup else frames,
                                used_rsu_frame_id=source,
                                latency_ms_source="" if startup else "injected_from_E7",
                                zoh_age_frames=None if startup else frame - ready))
        sim_times = {"consume_sim_time_s": frame / SIM_RATE_HZ,
                     "source_sim_time_s": None if startup else source / SIM_RATE_HZ} if validate_sim_time else {}
        for key, expected in sim_times.items():
            if expected is None:
                _expect_fields(row, {key: None})
            elif row[key] == "" or not math.isclose(
                    _csv_number(row, key), expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("audit %s mismatch: expected %r got %r" %
                                 (key, expected, row[key]))
        previous = source


def _explicit_audit_paths(path, audit_paths):
    """Return the canonical pair in the canonical JSON's route/meta subtree."""
    if not isinstance(audit_paths, (tuple, list)) or len(audit_paths) != 2:
        raise ValueError("explicit audit paths must be an RSU/ego pair")
    paths = tuple(Path(item) for item in audit_paths)
    if any(not item.is_absolute() for item in paths):
        raise ValueError("explicit audit paths must be absolute")
    try:
        paths = tuple(item.resolve(strict=True) for item in paths)
    except (OSError, RuntimeError) as error:
        raise ValueError("cannot resolve explicit audit paths: %s" % error)
    route = path.parents[1]
    root = route.parents[2]
    image_route = root / "image" / route.relative_to(root)
    for audit, filename in zip(paths, ("latency_align_audit.csv", "ego_output_fifo_audit.csv")):
        if (audit.name != filename or audit.parent.name != "meta"
                or audit.parent.parent.parent != image_route):
            raise ValueError("explicit audit path does not match route/meta provenance")
    if paths[0].parent != paths[1].parent:
        raise ValueError("explicit audit paths must share one meta directory")
    return tuple(str(item) for item in paths)


def validate_audits(path, route, repeat, rsu_delay, ego_delay, agent_config, *,
                    allow_legacy_rsu_schema=False, audit_paths=None):
    """Default to current audits; immutable historical evidence must opt in."""
    if type(allow_legacy_rsu_schema) is not bool:
        raise ValueError("allow_legacy_rsu_schema must be a boolean")
    if any(value is None for value in (route, repeat, rsu_delay, ego_delay, agent_config)):
        raise ValueError("audit acceptance requires route/repeat/config and both expected delays")
    if isinstance(rsu_delay, bool) or isinstance(ego_delay, bool):
        raise ValueError("expected numeric audit delays, not booleans")
    rsu, ego = float(rsu_delay), float(ego_delay)
    skip = _audit_config(agent_config, rsu, ego)
    rsu, ego = int(rsu), int(ego)
    path = Path(path).absolute() if audit_paths is None else Path(path).resolve(strict=True)
    expected_tail = ("v2x_final", "town05_short_collab", "r%s_repeat%s" % (route, repeat),
                     "ego_vehicle_0", "results.json")
    if path.parts[-5:] != expected_tail:
        raise ValueError("route JSON path does not match expected route/repeat/single ego")
    if audit_paths is None:
        rsu_path = _audit_path(path, route, repeat, "latency_align_audit.csv")
        ego_path = _audit_path(path, route, repeat, "ego_output_fifo_audit.csv")
    else:
        rsu_path, ego_path = _explicit_audit_paths(path, audit_paths)
    if not rsu_path or not ego_path or Path(rsu_path).parent != Path(ego_path).parent:
        raise ValueError("missing or ambiguous matching RSU/ego audit pair")
    ego_rows = _read_audit(ego_path, EGO_HEADERS)
    required = LEGACY_RSU_HEADERS if allow_legacy_rsu_schema else RSU_HEADERS
    rsu_rows = _read_audit(rsu_path, required)
    headers = set(rsu_rows[0])
    legacy = headers == set(LEGACY_RSU_HEADERS)
    if headers != set(RSU_HEADERS) and not (allow_legacy_rsu_schema and legacy):
        raise ValueError("unsupported RSU audit headers (expected current 17 or explicit legacy 15)")
    _validate_ego_audit(ego_rows, ego)
    _validate_rsu_audit(rsu_rows, rsu, len(ego_rows), skip, validate_sim_time=not legacy)
    # These contracts establish coverage only through the last recorded ego tick.
    # A coherent common suffix loss cannot be detected without an independent end frame.
    return rsu_path, ego_path


def build_summary_row(path, route, repeat, rsu_delay, ego_delay,
                      agent_config, git_commit, git_dirty):
    parsed = classify_execution(path, route=route, repeat=repeat, rsu_delay=rsu_delay,
                                ego_delay=ego_delay, agent_config=agent_config)
    return [route, repeat, rsu_delay, int(math.ceil(float(rsu_delay) * SIM_RATE_HZ / 1000)),
            ego_delay, int(math.ceil(float(ego_delay) * SIM_RATE_HZ / 1000)),
            parsed["DS"], parsed["RC"], parsed["status"], parsed["countable"],
            parsed["coll_lay"], parsed["coll_veh"], str(path),
            _audit_path(path, route, repeat, "latency_align_audit.csv"),
            _audit_path(path, route, repeat, "ego_output_fifo_audit.csv"),
            agent_config, SIM_RATE_HZ, git_commit, git_dirty]


def classify_execution(path, eval_log=None, process_exit=0, infra_error=None,
                       route=None, repeat=None, rsu_delay=None, ego_delay=None, agent_config=None):
    if infra_error:
        return infrastructure_invalid(infra_error)
    if process_exit:
        return infrastructure_invalid("evaluator process exit=%s log=%s" % (process_exit, eval_log))
    if eval_log:
        try:
            with open(eval_log, errors="replace") as stream:
                for line in stream:
                    if INFRA_PATTERN.search(line):
                        return infrastructure_invalid("%s: %s" % (eval_log, line.strip()))
        except OSError as error:
            return infrastructure_invalid("cannot read evaluator log: %s" % error)
    parsed = parse_route_result(path)
    if parsed["countable"]:
        try:
            validate_audits(path, route, repeat, rsu_delay, ego_delay, agent_config)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, csv.Error) as error:
            return infrastructure_invalid("audit validation: %s" % error)
    return parsed


def append_summary_row(summary_path, row):
    needs_header = not Path(summary_path).exists() or Path(summary_path).stat().st_size == 0
    with open(summary_path, "a", newline="") as stream:
        writer = csv.writer(stream)
        if needs_header:
            writer.writerow(FIELDS)
        writer.writerow(row)


def _process_info(pid):
    fields = (Path("/proc") / str(int(pid)) / "stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[2]), int(fields[3]), fields[19], fields[0], int(fields[1])


def process_start_token(pid, owner_pid):
    """Capture a live direct child, even before it has entered its own session."""
    _, _, token, state, parent = _process_info(pid)
    if int(pid) <= 1 or parent != int(owner_pid) or state == "Z":
        raise ValueError("not a live direct child of the recorded owner")
    return token


def process_token(pid):
    group, session, token, state, _ = _process_info(pid)
    if int(pid) <= 1 or group != int(pid) or session != int(pid) or state == "Z":
        raise ValueError("not a live owned session leader")
    return token


def publish_owned_identity(path):
    pid = os.getpid()
    identity = dict(pid=pid, ppid=os.getppid(), pgrp=os.getpgrp(),
                    session=os.getsid(0), start_token=process_token(pid))
    if identity["pgrp"] != pid or identity["session"] != pid:
        raise ValueError("identity is not an owned session leader")
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(identity))
    temporary.replace(path)
    return identity


def await_owned_identity(path, pid, start_token, owner_pid, timeout_s=5):
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or not 0 <= timeout <= 5:
        raise ValueError("ownership timeout must be between 0 and 5 seconds")
    expected = dict(pid=int(pid), ppid=int(owner_pid), pgrp=int(pid),
                    session=int(pid), start_token=start_token)
    deadline = time.monotonic() + timeout
    while True:
        if not start_token or process_start_token(pid, owner_pid) != start_token:
            raise ValueError("direct-child identity changed during ownership handshake")
        try:
            identity = json.loads(Path(path).read_text())
        except FileNotFoundError:
            identity = None  # Publication is atomic; absence alone is retryable.
        if identity is not None:
            if identity != expected or process_token(pid) != start_token:
                raise ValueError("published identity does not match the recorded child")
            if process_start_token(pid, owner_pid) != start_token:
                raise ValueError("owner changed during ownership handshake")
            return start_token
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("timed out waiting for owned identity")
        time.sleep(min(.02, remaining))


def _direct_child_alive(pid, token, owner_pid):
    try:
        return bool(pid and token) and process_start_token(pid, owner_pid) == token
    except (OSError, ValueError, IndexError):
        return False


def stop_direct_child(pid, start_token, owner_pid):
    """Only signal the exact child; a published supervisor handles group cleanup."""
    deadline = time.monotonic() + .5
    for signum in (signal.SIGTERM, signal.SIGKILL):
        if not _direct_child_alive(pid, start_token, owner_pid):
            return
        try:
            os.kill(int(pid), signum)
        except ProcessLookupError:
            return
        while signum == signal.SIGTERM and time.monotonic() < deadline:
            if not _direct_child_alive(pid, start_token, owner_pid):
                return
            time.sleep(.02)


def process_identity_gone(pid, start_token):
    """Do not allow a shell wait merely because an ownership check failed."""
    try:
        _, _, current, state, _ = _process_info(pid)
        return state == "Z" or bool(start_token and current != start_token)
    except (FileNotFoundError, ProcessLookupError):
        return True


def owned_process_alive(pid, token):
    try:
        return bool(pid and token) and process_token(pid) == token
    except (OSError, ValueError, IndexError):
        return False  # A child exiting during inspection is an ordinary lifecycle race.


def stop_owned_process(pid, token):
    """Only signal the still-live, identity-matching session created by this run."""
    if owned_process_alive(pid, token):
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # The verified group exited between identity check and signal.


def required_ports_free(port):
    # Keep all probes bound together until the preflight is complete.
    probes = []
    try:
        for offset in (0, 1, 2, 5):
            for family, address in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
                probe = socket.socket(family, socket.SOCK_STREAM)
                probes.append(probe)
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind((address, int(port) + offset))
    finally:
        for probe in probes:
            probe.close()


def owned_listener_ready(pid, token, port):
    if not owned_process_alive(pid, token):
        return False
    inodes = set()
    for name in ("tcp", "tcp6"):
        for line in (Path("/proc/net") / name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == int(port):
                inodes.add("socket:[%s]" % fields[9])
    for process in Path("/proc").glob("[0-9]*"):
        try:
            group, session, _, state, _ = _process_info(process.name)
            if group == int(pid) and session == int(pid) and state != "Z":
                if any(os.readlink(str(fd)) in inodes for fd in (process / "fd").iterdir()):
                    return owned_process_alive(pid, token)
        except (OSError, ValueError, IndexError):
            continue  # Unrelated/inaccessible or already-exited processes are never signalled.
    return False


def supervise(status_path, token_path, command):
    """Retain a live group leader after child completion, until owner cleanup.

    This prevents losing the identity anchor while descendants still exist.
    The status file is private to the owning attempt; publication is atomic.
    """
    # Install before publication: owner TERM may arrive before it claims our token.
    # KILL cannot recurse through this handler and removes only our own session.
    def stop_session(signum, frame):
        stop_owned_process(os.getpid(), process_token(os.getpid()))

    signal.signal(signal.SIGTERM, stop_session)
    signal.signal(signal.SIGINT, stop_session)
    publish_owned_identity(token_path)
    result = subprocess.call(command)
    temporary = Path(status_path + ".tmp")
    temporary.write_text(str(result))
    temporary.replace(status_path)
    while True:
        signal.pause()


def process_command(argv):
    command, values = argv[1], argv[2:]
    if command == "--supervise":
        supervise(values[0], values[1], values[2:])
    elif command == "--ports-free":
        required_ports_free(values[0])
    elif command == "--process-token":
        print(process_token(values[0]))
    elif command == "--process-start-token":
        print(process_start_token(*values))
    elif command == "--await-owned":
        print(await_owned_identity(*values))
    elif command == "--stop-child":
        stop_direct_child(*values)
    elif command == "--identity-gone":
        return 0 if process_identity_gone(*values) else 1
    elif command == "--stop-owned":
        stop_owned_process(*values)
    elif command == "--owned-alive":
        return 0 if owned_process_alive(*values) else 1
    elif command == "--owned-ready":
        return 0 if owned_listener_ready(*values) else 1
    return 0


def main(argv):
    if len(argv) > 1 and argv[1] in ("--supervise", "--ports-free", "--process-token",
                                   "--process-start-token", "--await-owned", "--stop-child",
                                   "--identity-gone", "--stop-owned", "--owned-alive", "--owned-ready"):
        try:
            return process_command(argv)
        except (OSError, ValueError, IndexError, TypeError) as error:
            print("[ABORT] process/port check: %s" % error, file=sys.stderr)
            return 3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-result", action="store_true")
    parser.add_argument("--eval-log")
    parser.add_argument("--process-exit", type=int, default=0)
    parser.add_argument("--infra-error")
    for name in ("route", "repeat", "rsu-delay", "ego-delay", "agent-config"):
        parser.add_argument("--" + name)
    parser.add_argument("values", nargs="+")
    args = parser.parse_args(argv[1:])
    expected = 1 if args.check_result else 9
    if len(args.values) != expected:
        parser.error("expected EGO_JSON" if args.check_result else
                     "expected EGO_JSON ROUTE REPEAT RSU_MS EGO_MS SUMMARY AGENT_CONFIG GIT_COMMIT GIT_DIRTY")
    path = args.values[0]
    context = dict(route=args.route, repeat=args.repeat, rsu_delay=args.rsu_delay,
                   ego_delay=args.ego_delay, agent_config=args.agent_config)
    if not args.check_result:
        path, route, repeat, rsu, ego, summary, config, commit, dirty = args.values
        context = dict(route=route, repeat=repeat, rsu_delay=rsu, ego_delay=ego, agent_config=config)
    parsed = classify_execution(path, args.eval_log, args.process_exit, args.infra_error, **context)
    if not args.check_result:
        row = build_summary_row(path, route, repeat, rsu, ego, config, commit, dirty)
        # Infrastructure evidence overrides even an otherwise parseable checkpoint.
        row = row[:6] + [parsed[key] for key in
                         ("DS", "RC", "status", "countable", "coll_lay", "coll_veh")] + row[12:]
        append_summary_row(summary, row)
    print("[result] status=%s countable=%s DS=%.3f RC=%.3f%s" % (
        parsed["status"], parsed["countable"], parsed["DS"], parsed["RC"],
        " error=" + parsed["error"] if "error" in parsed else ""))
    return 1 if args.check_result and not parsed["countable"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
