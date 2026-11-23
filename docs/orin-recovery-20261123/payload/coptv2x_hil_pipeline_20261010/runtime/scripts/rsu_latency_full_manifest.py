#!/usr/bin/env python
"""Build the deterministic 85-cell, RSU-only dual-FIFO manifest."""

from __future__ import print_function

import argparse
import json
import sys

from action_fifo_full_manifest import SUBSETS


DELAYS_MS = (0, 200, 500, 800)
AGENTS = {
    0: "codriving_dualfifo_baseline_l1",
    200: "codriving_dualfifo_rsu200_l1",
    500: "codriving_dualfifo_rsu500_l1",
    800: "codriving_dualfifo_rsu800_l1",
}
TSV_FIELDS = (
    "subset", "scenario", "route", "rsu_delay_ms", "ego_delay_ms",
    "delay_frames", "carla_seed", "traffic_seed", "repeat", "key",
    "tag", "agent", "eval_setting",
)


def build_manifest(carla_seed=2000, traffic_seed=2000, repeat=0):
    """Return the official 85 cells crossed with four RSU delays."""
    tasks = []
    seed_identity = (
        "seed%d" % carla_seed
        if carla_seed == traffic_seed
        else "cseed%d_tseed%d" % (carla_seed, traffic_seed)
    )
    for delay_ms in DELAYS_MS:
        for subset, scenario, _profile, routes in SUBSETS:
            scenario_id = scenario.lstrip("_")
            for route in routes:
                key = "rsu_sc%s_r%d_d%d_%s_rep%d" % (
                    scenario_id, route, delay_ms, seed_identity, repeat
                )
                tasks.append({
                    "subset": subset,
                    "scenario": scenario,
                    "route": route,
                    "rsu_delay_ms": delay_ms,
                    "ego_delay_ms": 0,
                    "delay_frames": delay_ms // 50,
                    "carla_seed": carla_seed,
                    "traffic_seed": traffic_seed,
                    "repeat": repeat,
                    "key": key,
                    "tag": "rsu_latency_full_%s" % key,
                    "agent": AGENTS[delay_ms],
                    "eval_setting": (
                        "v2x_final/town05_short_collab/r%d_repeat%d"
                        % (route, repeat)
                    ),
                })
    return tasks


def select_shard(tasks, shard_index=None, shard_count=None):
    if shard_index is None and shard_count is None:
        return list(tasks)
    if shard_index is None or shard_count is None:
        raise ValueError("--shard-index and --shard-count must be provided together")
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard must satisfy 0 <= index < count")
    return [task for index, task in enumerate(tasks)
            if index % shard_count == shard_index]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--carla-seed", type=int)
    parser.add_argument("--traffic-seed", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    common_seed = 2000 if args.seed is None else args.seed
    carla_seed = common_seed if args.carla_seed is None else args.carla_seed
    traffic_seed = common_seed if args.traffic_seed is None else args.traffic_seed
    try:
        if carla_seed < 0 or traffic_seed < 0 or args.repeat < 0:
            raise ValueError("seeds and repeat must be non-negative integers")
        tasks = select_shard(
            build_manifest(carla_seed, traffic_seed, args.repeat),
            args.shard_index,
            args.shard_count,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(tasks, indent=2, sort_keys=True))
    else:
        for task in tasks:
            print("\t".join(str(task[field]) for field in TSV_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
