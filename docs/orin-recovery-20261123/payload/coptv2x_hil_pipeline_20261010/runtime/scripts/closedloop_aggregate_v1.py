# -*- coding: utf-8 -*-
"""closedloop_aggregate_v1.py -- Sim-C sweep result aggregator (design §4.D).

Walks results/results_driving_lat_d*/ (+ lat_norsu) directories, and for each
(route, arm):
  - from ego_vehicle_0/results.json -> _checkpoint/global_record/{scores,
    infractions} + records[0]/meta/duration_* + route_length
  - from the paired image/.../meta/latency_align_audit.csv -> mean zoh_age_frames,
    mean delta, and is_zoh_held fraction
joins them and appends one row per (route, arm) to results/closedloop_sweep_v1.csv.

Pure stdlib (json/csv/glob/argparse) -- no pandas, runs on the eval venv as-is.
No GPU, no CARLA: it only reads files already on disk.

Usage:
    python scripts/closedloop_aggregate_v1.py --results-root results \
        --out results/closedloop_sweep_v1.csv
"""
import argparse
import csv
import glob
import json
import os
import re


# arm tag -> (arm_label, inject_ms or None for no-roadside)
def _parse_arm_from_dirname(dirname):
    """results_driving_lat_d108 -> ('d108', 108); results_driving_lat_norsu ->
    ('norsu', None). Returns None if the dir is not a sweep arm dir."""
    m = re.match(r"results_driving_lat_(d(\d+)|norsu)$", dirname)
    if not m:
        return None
    if m.group(1) == "norsu":
        return ("norsu", None)
    return (m.group(1), int(m.group(2)))


def _route_id_from_dir(route_dir):
    """.../r0_repeat0 -> 0 ; .../r13_repeat0 -> 13. None if unparseable."""
    m = re.search(r"/r(\d+)_repeat\d+$", route_dir.rstrip("/"))
    return int(m.group(1)) if m else None


def _read_ego_results(ego_json):
    """Extract scores+infractions+meta from an ego_vehicle_0/results.json.
    Returns a flat dict, or None if the file is missing/malformed."""
    try:
        d = json.load(open(ego_json))
    except Exception:
        return None
    gr = d.get("_checkpoint", {}).get("global_record", {})
    scores = gr.get("scores", {})
    infr = gr.get("infractions", {})
    status = gr.get("status", "Missing")
    # per-route meta lives under records[0]
    recs = gr_records = d.get("_checkpoint", {}).get("records", [])
    meta = recs[0].get("meta", {}) if recs else {}
    return {
        "status": status,
        "entry_status": d.get("entry_status", ""),
        "driving_score": scores.get("score_composed"),
        "route_completion": scores.get("score_route"),
        "infraction_penalty": scores.get("score_penalty"),
        "collisions_pedestrian": infr.get("collisions_pedestrian"),
        "collisions_vehicle": infr.get("collisions_vehicle"),
        "collisions_layout": infr.get("collisions_layout"),
        "red_light": infr.get("red_light"),
        "outside_route_lanes": infr.get("outside_route_lanes"),
        "route_dev": infr.get("route_dev"),
        "vehicle_blocked": infr.get("vehicle_blocked"),
        "route_timeout": infr.get("route_timeout"),
        "duration_game": meta.get("duration_game"),
        "duration_system": meta.get("duration_system"),
        "route_length": meta.get("route_length"),
    }


def _audit_stats(results_root, arm_dir, route_id):
    """Find the paired latency_align_audit.csv for this (arm, route) and return
    mean zoh_age_frames, mean delta, zoh_held fraction, n_frames. The audit dir
    name carries a run timestamp, so glob over it. Returns dict of None if absent
    (e.g. the no-roadside arm has no aligner -> no audit csv, which is correct)."""
    pattern = os.path.join(
        results_root, arm_dir, "image", "v2x_final", "town05_short_collab",
        "r%d_repeat0" % route_id, "*", "meta", "latency_align_audit.csv")
    matches = sorted(glob.glob(pattern))
    empty = {"zoh_age_mean": None, "delta_mean": None,
             "zoh_held_frac": None, "audit_frames": 0}
    if not matches:
        return empty
    path = matches[-1]  # newest run if multiple
    ages, deltas, held = [], [], []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                z = row.get("zoh_age_frames", "")
                dl = row.get("delta", "")
                h = row.get("is_zoh_held", "")
                if z not in ("", None):
                    try:
                        ages.append(float(z))
                    except ValueError:
                        pass
                if dl not in ("", None):
                    try:
                        deltas.append(float(dl))
                    except ValueError:
                        pass
                if h not in ("", None):
                    held.append(1.0 if str(h).strip().lower() == "true" else 0.0)
    except Exception:
        return empty
    n = max(len(ages), len(deltas), len(held))
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {
        "zoh_age_mean": mean(ages),
        "delta_mean": mean(deltas),
        "zoh_held_frac": mean(held),
        "audit_frames": n,
    }


FIELDNAMES = [
    "route_id", "arm", "inject_ms", "isolation",
    "driving_score", "route_completion", "infraction_penalty",
    "collisions_pedestrian", "collisions_vehicle", "collisions_layout",
    "red_light", "outside_route_lanes", "route_dev",
    "vehicle_blocked", "route_timeout",
    "zoh_age_mean", "delta_mean", "zoh_held_frac", "audit_frames",
    "duration_game", "duration_system", "route_length",
    "status", "entry_status", "ego_json",
]


def aggregate(results_root, out_csv, isolation="unknown"):
    rows = []
    for arm_dir in sorted(os.listdir(results_root)):
        full = os.path.join(results_root, arm_dir)
        if not os.path.isdir(full):
            continue
        parsed = _parse_arm_from_dirname(arm_dir)
        if parsed is None:
            continue
        arm_label, inject_ms = parsed
        route_glob = os.path.join(
            full, "v2x_final", "town05_short_collab", "r*_repeat0")
        for route_dir in sorted(glob.glob(route_glob)):
            rid = _route_id_from_dir(route_dir)
            if rid is None:
                continue
            ego_json = os.path.join(route_dir, "ego_vehicle_0", "results.json")
            res = _read_ego_results(ego_json)
            if res is None:
                print("[warn] missing/bad ego json: %s" % ego_json)
                continue
            audit = _audit_stats(results_root, arm_dir, rid)
            row = {
                "route_id": rid, "arm": arm_label, "inject_ms": inject_ms,
                "isolation": isolation, "ego_json": ego_json,
            }
            row.update(res)
            row.update(audit)
            rows.append(row)

    rows.sort(key=lambda r: (r["route_id"],
                             999999 if r["inject_ms"] is None else r["inject_ms"]))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[agg] wrote %d rows -> %s" % (len(rows), out_csv))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="results/closedloop_sweep_v1.csv")
    ap.add_argument("--isolation", default="unknown",
                    help="single_card_shared (injected-only) | separate_gpu (isolated)")
    args = ap.parse_args()
    aggregate(args.results_root, args.out, args.isolation)


if __name__ == "__main__":
    main()
