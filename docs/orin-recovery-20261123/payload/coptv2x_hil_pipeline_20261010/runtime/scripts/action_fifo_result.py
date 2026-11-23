#!/usr/bin/env python
"""Parse one action-FIFO route result into a trustworthy summary row."""

from __future__ import print_function

import csv
import glob
import json
import math
import os
import sys


def _infraction_value(value):
    if isinstance(value, list):
        return float(len(value))
    return float(value or 0.0)


def parse_route_result(path):
    result = {
        "status": "FAILED",
        "DS": 0.0,
        "RC": 0.0,
        "coll_lay": 0.0,
        "coll_veh": 0.0,
    }
    try:
        with open(path) as stream:
            record = json.load(stream)["_checkpoint"]["global_record"]
    except Exception:
        return result

    scores = record.get("scores", {})
    infractions = record.get("infractions", {})
    return {
        "status": record.get("status", "UNKNOWN"),
        "DS": float(scores.get("score_composed", 0.0) or 0.0),
        "RC": float(scores.get("score_route", 0.0) or 0.0),
        "coll_lay": _infraction_value(infractions.get("collisions_layout", 0.0)),
        "coll_veh": _infraction_value(infractions.get("collisions_vehicle", 0.0)),
    }


def build_summary_row(path, route, repeat, delay):
    parsed = parse_route_result(path)
    delay_frames = int(math.ceil(float(delay) * 20.0 / 1000.0))
    tag = "fifo_d%s_r%s_n%s" % (delay, route, repeat)
    audit_pattern = os.path.join(
        "results",
        "results_driving_" + tag,
        "image",
        "v2x_final",
        "town05_short_collab",
        "r%s_repeat%s" % (route, repeat),
        "*",
        "meta",
        "action_fifo_audit.csv",
    )
    audits = sorted(glob.glob(audit_pattern))
    return [
        route,
        repeat,
        delay,
        delay_frames,
        parsed["DS"],
        parsed["RC"],
        parsed["status"],
        parsed["coll_lay"],
        parsed["coll_veh"],
        path,
        audits[-1] if audits else "",
    ]


def append_summary_row(summary_path, row):
    with open(summary_path, "a", newline="") as stream:
        csv.writer(stream).writerow(row)


def main(argv):
    if len(argv) != 6:
        raise SystemExit(
            "usage: action_fifo_result.py EGO_JSON ROUTE REPEAT DELAY SUMMARY"
        )
    path, route, repeat, delay, summary = argv[1:]
    row = build_summary_row(path, route, repeat, delay)
    append_summary_row(summary, row)
    print(
        "[result] r%s fifo=%sms DS=%.3f RC=%.3f status=%s audit=%s"
        % (route, delay, row[4], row[5], row[6], row[-1] or "MISSING")
    )


if __name__ == "__main__":
    main(sys.argv)
