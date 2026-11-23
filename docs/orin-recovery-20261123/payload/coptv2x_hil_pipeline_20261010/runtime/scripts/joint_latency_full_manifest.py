#!/usr/bin/env python
"""Build the official 85-cell, ten-arm joint RSU/action latency manifest."""

import argparse
import json
import sys

from action_fifo_full_manifest import SUBSETS


ACTION_DELAYS_MS = (200, 300, 400, 500)
RSU_DELAYS_MS = (100, 200, 300, 400)
ARMS = tuple((rsu, action) for action in ACTION_DELAYS_MS
             for rsu in RSU_DELAYS_MS if action > rsu)
TSV_FIELDS = ("subset", "scenario", "route", "profile", "rsu_delay_ms",
              "ego_delay_ms", "carla_seed", "traffic_seed", "repeat", "key",
              "tag", "agent", "eval_setting")


def build_manifest(carla_seed=2000, traffic_seed=2000, repeat=0, cadence=4):
    if min(carla_seed, traffic_seed, repeat) < 0:
        raise ValueError("seeds and repeat must be non-negative")
    if cadence not in (2, 4):
        raise ValueError("cadence must be 2 or 4")
    seed = ("seed%d" % carla_seed if carla_seed == traffic_seed else
            "cseed%d_tseed%d" % (carla_seed, traffic_seed))
    namespace = "jointc2" if cadence == 2 else "joint"
    tasks = []
    for rsu, action in ARMS:
        for subset, scenario, _source_profile, routes in SUBSETS:
            for route in routes:
                key = "%s_sc%s_r%d_rsu%d_ego%d_%s_rep%d" % (
                    namespace, scenario.lstrip("_"), route, rsu, action, seed, repeat)
                tasks.append({
                    "subset": subset, "scenario": scenario, "route": route,
                    "profile": "5_10", "rsu_delay_ms": rsu,
                    "ego_delay_ms": action, "rsu_delay_frames": rsu // 50,
                    "ego_delay_frames": action // 50,
                    "carla_seed": carla_seed, "traffic_seed": traffic_seed,
                    "repeat": repeat, "key": key,
                    "tag": "%s_latency_full_%s" % (namespace, key),
                    "agent": "codriving_dualfifo_%s_rsu%d_ego%d_l1" % (namespace, rsu, action),
                    "eval_setting": "v2x_final/town05_short_collab/r%d_repeat%d" % (
                        route, repeat),
                })
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--carla-seed", type=int)
    parser.add_argument("--traffic-seed", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--cadence", type=int, choices=(2, 4), default=4)
    args = parser.parse_args(argv)
    try:
        tasks = build_manifest(
            args.seed if args.carla_seed is None else args.carla_seed,
            args.seed if args.traffic_seed is None else args.traffic_seed,
            args.repeat, args.cadence)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(tasks, indent=2, sort_keys=True))
    else:
        for task in tasks:
            print("\t".join(str(task[field]) for field in TSV_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
