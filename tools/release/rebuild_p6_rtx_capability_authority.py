#!/usr/bin/env python3
"""Rebuild path-free RTX capability authority in the approved TVM runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.p6_capability_context_v1 import (  # noqa: E402
    canonical_cuda_target,
)
from framework.stage6.p6_capability_observation_v1 import (  # noqa: E402
    rebuild_probe_records,
)
from framework.stage6.p6_capability_probe_specs_v1 import (  # noqa: E402
    NEUTRAL_PROBE_IDS,
    PRUNING_PROBE_IDS,
)
from framework.stage6.p6_capability_runtime_authority_v1 import (  # noqa: E402
    AUTHORITY_SCHEMA_VERSION,
)
from framework.stage6.p6_capability_tvm_v1 import (  # noqa: E402
    collect_tvm_runtime_identity,
)
from framework.stage6.p6_tvm_runtime_authority_v1 import (  # noqa: E402
    canonical_tvm_support_tree_sha256,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _path(raw: Path, *, want_dir: bool) -> Path:
    try:
        resolved = raw.resolve(strict=True)
        info = raw.lstat()
    except OSError as error:
        raise ValueError("authority input invalid") from error
    valid_type = stat.S_ISDIR(info.st_mode) if want_dir else stat.S_ISREG(info.st_mode)
    if (
        not raw.is_absolute()
        or resolved != raw
        or raw.is_symlink()
        or not valid_type
        or not want_dir
        and info.st_nlink != 1
    ):
        raise ValueError("authority input invalid")
    return resolved


def _read_context(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_path(path, want_dir=False).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("authority input invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("authority input invalid")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    tvm_site = _path(args.tvm_site, want_dir=True)
    dependency_root = _path(args.dependency_root, want_dir=True)
    nvlibs_file = _path(args.nvlibs_file, want_dir=False)
    support_root = _path(args.support_root, want_dir=True)
    if (
        not isinstance(args.support_root_sha256, str)
        or canonical_tvm_support_tree_sha256(support_root)
        != args.support_root_sha256
    ):
        raise ValueError("authority input invalid")
    sys.path.insert(0, str(dependency_root))
    sys.path.insert(0, str(tvm_site))
    context = _read_context(args.context)
    evidence = context.get("measurement_evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("artifact_blobs"), list):
        raise ValueError("authority input invalid")
    blobs = evidence["artifact_blobs"]
    records = {
        "neutral": rebuild_probe_records(
            [item for item in blobs if isinstance(item, dict) and item.get("family") == "neutral"],
            family="neutral",
            probe_ids=NEUTRAL_PROBE_IDS,
        ),
        "pruning": rebuild_probe_records(
            [item for item in blobs if isinstance(item, dict) and item.get("family") == "pruning"],
            family="pruning",
            probe_ids=PRUNING_PROBE_IDS,
        ),
    }
    runtime = collect_tvm_runtime_identity(
        tvm_site=tvm_site,
        nvlibs_file=nvlibs_file,
        support_root_sha256=args.support_root_sha256,
    )
    if runtime["target"] != canonical_cuda_target("sm89"):
        raise ValueError("authority input invalid")
    return {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "runtime_identity": runtime,
        "probe_records": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--tvm-site", type=Path, required=True)
    parser.add_argument("--nvlibs-file", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--support-root-sha256", required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(parse_args(argv))
    except Exception:
        print("capability_runtime_authority_failed", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
