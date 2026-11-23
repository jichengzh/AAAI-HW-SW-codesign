#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from ditl_stage_summary import summarize_stage_records
from ditl_trace_schema import read_trace_jsonl


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize DitL v2 stage traces.")
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup-count", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = read_trace_jsonl(args.trace_jsonl)
    summary = summarize_stage_records(records, args.warmup_count)
    target = Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps({"output_json": str(target), "samples": summary["samples"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
