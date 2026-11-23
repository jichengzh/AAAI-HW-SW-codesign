#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.leaderboard.team_code.closedloop.coptv2x_hil_artifacts import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CONFIG_DIR,
    HilArtifactError,
    preflight_hil_artifacts,
)


def _parser():
    parser = argparse.ArgumentParser(
        description="Verify the frozen five-arm CoptV2X HIL artifact registry."
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        records = preflight_hil_artifacts(
            config_dir=arguments.config_dir,
            artifact_root=arguments.artifact_root,
        )
    except HilArtifactError as error:
        print("CoptV2X HIL preflight failed: %s" % error, file=sys.stderr)
        return 1
    payload = {
        "schema_version": "coptv2x_hil_preflight_v1",
        "status": "ok",
        "artifact_count": len(records),
        "artifacts": list(records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
