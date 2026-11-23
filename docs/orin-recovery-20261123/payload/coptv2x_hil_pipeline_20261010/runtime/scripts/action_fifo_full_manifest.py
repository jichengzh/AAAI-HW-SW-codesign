#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the deterministic Town05 action-FIFO full-sweep manifest."""

from __future__ import print_function

import argparse
import json
import math
import sys


# Source authority: scripts/batch_eval_driving_e2e.sh at f63e92b.
SUBSETS = (
    ("1s", "_1", "8_10", (2, 3, 5, 8, 10, 13, 18, 31, 135, 146, 147, 161, 300, 310, 326, 327)),
    ("2s", "_2", "5_10", (2, 3, 7, 12, 19, 22, 24, 25, 27, 136, 138, 161, 300, 320, 321, 322, 328, 329)),
    ("3s", "_3", "5_10", (7, 8, 16, 31, 300, 301, 305, 310, 311, 313, 314, 316, 320, 324, 329, 330, 331)),
    ("4s", "_4", "8_10", (1, 8, 9, 12, 14, 15, 16, 28, 140, 145, 160, 311, 312, 317, 321, 323, 328, 331)),
    ("5s", "_5", "5_50", (1, 6, 7, 8, 18, 20, 28, 31, 142, 145, 146, 157, 306, 310, 315, 318)),
)
DELAYS_MS = (0, 200, 500, 800)
SMOKE_SUBSETS = (
    ("1s", "_1", "8_10", (2,)),
    ("2s", "_2", "5_10", (2,)),
    ("5s", "_5", "5_50", (1,)),
)
SMOKE_DELAYS_MS = (0, 800)
TSV_FIELDS = (
    "subset", "scenario", "route", "profile", "delay_ms", "delay_frames",
    "carla_seed", "traffic_seed", "repeat", "key", "tag", "agent",
    "result_json", "eval_log", "carla_log", "eval_setting",
)


def build_manifest(carla_seed=2000, traffic_seed=2000, repeat=0, smoke=False):
    """Return all official authority cells crossed with the requested delays."""
    subsets = SMOKE_SUBSETS if smoke else SUBSETS
    delays_ms = SMOKE_DELAYS_MS if smoke else DELAYS_MS
    experiment = "smoke" if smoke else "full"
    tasks = []
    for delay_ms in delays_ms:
        for subset, scenario, profile, routes in subsets:
            scenario_id = scenario.lstrip("_")
            for route in routes:
                if carla_seed == traffic_seed:
                    seed_identity = "seed%d" % carla_seed
                else:
                    seed_identity = "cseed%d_tseed%d" % (
                        carla_seed, traffic_seed
                    )
                key_stem = "s%s_sc%s_r%d_d%d_%s_rep%d" % (
                    subset, scenario_id, route, delay_ms, seed_identity, repeat
                )
                key = "%s_%s" % (experiment, key_stem) if smoke else key_stem
                tag = "action_fifo_%s_%s" % (experiment, key_stem)
                eval_setting = "v2x_final/town05_short_collab/%s" % key
                result_root = "results/results_driving_%s" % tag
                log_root = "results/action_fifo_%s/logs" % experiment
                tasks.append(
                    {
                        "experiment": experiment,
                        "subset": subset,
                        "scenario": scenario,
                        "route": route,
                        "profile": profile,
                        "delay_ms": delay_ms,
                        "delay_frames": int(math.ceil(delay_ms * 20.0 / 1000.0)),
                        "seed": carla_seed if carla_seed == traffic_seed else None,
                        "carla_seed": carla_seed,
                        "traffic_seed": traffic_seed,
                        "repeat": repeat,
                        "key": key,
                        "tag": tag,
                        "agent": "codriving_%s_fifo%d_l1" % (profile, delay_ms),
                        "result_json": "%s/%s/ego_vehicle_0/results.json"
                        % (result_root, eval_setting),
                        "eval_log": "%s/eval_%s.log" % (log_root, key),
                        "carla_log": "%s/carla_%s.log" % (log_root, key),
                        "eval_setting": eval_setting,
                    }
                )
    return tasks


def select_shard(tasks, shard_index=None, shard_count=None):
    if shard_index is None and shard_count is None:
        return list(tasks)
    if shard_index is None or shard_count is None:
        raise ValueError("--shard-index and --shard-count must be provided together")
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard must satisfy 0 <= index < count")
    return [task for index, task in enumerate(tasks) if index % shard_count == shard_index]


def _is_driving_failure(status):
    return (
        isinstance(status, str)
        and status.startswith("Failed")
        and "crash" not in status.lower()
    )


def _has_single_route_driving_failure(checkpoint):
    records = checkpoint.get("records")
    global_record = checkpoint.get("global_record")
    if not isinstance(records, list) or len(records) != 1:
        return False
    if not isinstance(global_record, dict):
        return False

    route_record = records[0]
    metadata = global_record.get("meta")
    if not isinstance(route_record, dict):
        return False
    if not isinstance(metadata, dict):
        return False
    exceptions = metadata.get("exceptions")
    if not isinstance(exceptions, list) or len(exceptions) != 1:
        return False

    exception = exceptions[0]
    if not isinstance(exception, (list, tuple)) or len(exception) < 3:
        return False
    route_status = route_record.get("status")
    exception_status = exception[2]
    return (
        route_status == exception_status
        and _is_driving_failure(route_status)
    )


def result_status(path):
    try:
        with open(path) as stream:
            checkpoint = json.load(stream)["_checkpoint"]
    except (IOError, OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(checkpoint, dict):
        return None
    record = checkpoint.get("global_record")
    if not isinstance(record, dict):
        return None
    status = record.get("status")
    if status == "Completed":
        return status
    if status == "Failed" and _has_single_route_driving_failure(checkpoint):
        return status
    return None


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--carla-seed", type=int)
    parser.add_argument("--traffic-seed", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--check-result", metavar="PATH")
    parser.add_argument("--summarize-results", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check_result:
        status = result_status(args.check_result)
        if status is None:
            print("result_infrastructure_missing path=%s" % args.check_result)
            return 1
        print("result_valid status=%s path=%s" % (status, args.check_result))
        return 0

    try:
        common_seed = 2000 if args.seed is None else args.seed
        carla_seed = common_seed if args.carla_seed is None else args.carla_seed
        traffic_seed = common_seed if args.traffic_seed is None else args.traffic_seed
        if carla_seed < 0 or traffic_seed < 0 or args.repeat < 0:
            raise ValueError("seeds and repeat must be non-negative integers")
        tasks = select_shard(
            build_manifest(carla_seed, traffic_seed, args.repeat, args.smoke),
            args.shard_index,
            args.shard_count,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.summarize_results:
        valid = sum(result_status(task["result_json"]) is not None for task in tasks)
        missing = len(tasks) - valid
        print(
            "FULL_SWEEP_RESULT_GATE total=%d valid=%d missing=%d"
            % (len(tasks), valid, missing)
        )
        return 0 if missing == 0 else 4
    if args.format == "json":
        print(json.dumps(tasks, indent=2, sort_keys=True))
    else:
        for task in tasks:
            print("\t".join(str(task[field]) for field in TSV_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
