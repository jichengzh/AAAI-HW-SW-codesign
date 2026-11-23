#!/usr/bin/env python
"""Read-only verification of an explicit saved seven-arm evidence manifest."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import dual_fifo_result as gate


EXPECTED = {"baseline": (0, 0), "rsu200": (200, 0), "rsu500": (500, 0),
            "rsu800": (800, 0), "ego200": (0, 200), "ego500": (0, 500), "ego800": (0, 800)}
PATH_FIELDS = ("ego_json", "rsu_audit_csv", "ego_output_audit_csv", "agent_config")


def _summary_identity(summary):
    for key in ("route", "repeat"):
        if not isinstance(summary[key], str) or not re.fullmatch(r"[0-9]+", summary[key]):
            raise ValueError("invalid " + key)
    commit = summary["git_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid git_commit")
    if summary["git_dirty"] != "false" or summary["countable"] != "True":
        raise ValueError("evidence requires git_dirty=false and countable=True")
    return summary["route"], summary["repeat"], commit


def _verify_summary(arm, summary):
    rsu, ego = EXPECTED[arm]
    expected = {"rsu_fifo_delay_ms": rsu, "ego_output_fifo_delay_ms": ego,
                "rsu_delay_frames": rsu // 50, "ego_delay_frames": ego // 50,
                "sim_rate_hz": 20.0}
    for key, value in expected.items():
        if isinstance(summary[key], bool) or float(summary[key]) != value:
            raise ValueError("%s: incorrect %s" % (arm, key))
    for key in PATH_FIELDS:
        value = summary[key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("%s must be an absolute evidence path" % key)
    if Path(summary["agent_config"]).name != "pnp_config_codriving_dualfifo_%s_l1.yaml" % arm:
        raise ValueError("agent config basename does not match arm")
    canonical_json = str(Path(summary["ego_json"]).resolve(strict=True))
    paths = (summary["rsu_audit_csv"], summary["ego_output_audit_csv"])
    validated = gate.validate_audits(
        canonical_json, summary["route"], summary["repeat"], rsu, ego,
        summary["agent_config"], allow_legacy_rsu_schema=True, audit_paths=paths)
    parsed = gate.parse_route_result(canonical_json)
    if not parsed["countable"] or parsed["status"] != summary["status"]:
        raise ValueError("%s: route JSON is not countable or status differs" % arm)
    for key in ("DS", "RC", "coll_lay", "coll_veh"):
        if isinstance(summary[key], bool):
            raise ValueError("%s: summary %s must not be boolean" % (arm, key))
        number = float(summary[key])
        if not math.isfinite(number) or number != parsed[key]:
            raise ValueError("%s: route JSON/summary %s mismatch" % (arm, key))
    return _audit_report(arm, summary, validated)


def _audit_report(arm, summary, validated_paths):
    with open(validated_paths[0], newline="") as stream:
        rsu = csv.DictReader(stream)
        legacy = set(rsu.fieldnames) == set(gate.LEGACY_RSU_HEADERS)
        rsu_count = sum(1 for _ in rsu)
    with open(validated_paths[1], newline="") as stream:
        ego_count = sum(1 for _ in csv.DictReader(stream))
    return {"arm": arm, "summary": dict(summary), "assertions": "PASS",
            "rsu_rows": rsu_count, "ego_rows": ego_count,
            "rsu_schema": "legacy_15" if legacy else "current_17", "ego_schema": "v2_20",
            "missing_sim_time_columns": list(gate.RSU_SIM_TIME_HEADERS) if legacy else [],
            "sim_time_validation": "unavailable_in_legacy_schema" if legacy else "PASS"}


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number: %s" % value)
    return number


def verify_evidence(path):
    with open(path) as stream:
        records = json.load(stream, parse_constant=_finite_json_float, parse_float=_finite_json_float)
    if not isinstance(records, list) or len(records) != len(EXPECTED):
        raise ValueError("evidence must contain exactly seven arms")
    if any(not isinstance(row, dict) or not isinstance(row.get("arm"), str) for row in records):
        raise ValueError("each arm entry must be a mapping with an arm name")
    arms = [row["arm"] for row in records]
    if len(set(arms)) != len(EXPECTED) or set(arms) != set(EXPECTED):
        raise ValueError("duplicate, missing or unknown arm")
    identities = []
    for row in records:
        if not isinstance(row.get("summary"), dict):
            raise ValueError("arm summary must be a mapping")
        identities.append(_summary_identity(row["summary"]))
    if len(set(identities)) != 1:
        raise ValueError("arms must share route, repeat and git_commit")
    return [_verify_summary(row["arm"], row["summary"]) for row in records]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_json")
    args = parser.parse_args(argv)
    try:
        reports = verify_evidence(args.evidence_json)
        output = json.dumps(reports, indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError, csv.Error) as error:
        print("evidence verification failed: %s" % " ".join(str(error).split()), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
