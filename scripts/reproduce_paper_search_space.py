#!/usr/bin/env python3
"""Reproduce the public CoptV2X paper search-space counts on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.reproduction.coptv2x_paper_space_v1 import (  # noqa: E402
    SUPPORTED_MODELS,
    build_paper_space_summaries,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only reproduction of the three public CoptV2X paper spaces."
    )
    parser.add_argument(
        "--model",
        choices=(*SUPPORTED_MODELS, "all"),
        default="all",
        help="paper model to reproduce",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rows = build_paper_space_summaries(args.model)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    for row in rows:
        q_modes = ",".join(row["q_modes"])
        print(
            f"{row['model']}: structures={row['structures']} "
            f"candidates={row['candidates']} q_modes={q_modes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
