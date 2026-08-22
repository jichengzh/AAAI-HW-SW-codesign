"""Provision one redacted private P6 full-chain configuration pair."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.p6_full_chain_bootstrap_v1 import (  # noqa: E402
    FullChainBootstrapError,
    materialize_full_chain_binding,
)
from tools.release.provision_p6_history_local_config import (  # noqa: E402
    NvidiaSmiGpuProbe,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--legacy-local-config", required=True, type=_absolute_path)
    parser.add_argument("--runner-template", required=True, type=_absolute_path)
    parser.add_argument("--local-output-root", required=True, type=_absolute_path)
    parser.add_argument("--binding-output", required=True, type=_absolute_path)
    parser.add_argument("--config-output", required=True, type=_absolute_path)
    parser.add_argument("--source-wrapper-profile", type=_absolute_path)
    parser.add_argument("--external-training-binding", type=_absolute_path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize the private pair without exposing private input details."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        materialize_full_chain_binding(
            args.legacy_local_config,
            args.runner_template,
            args.local_output_root,
            args.binding_output,
            args.config_output,
            NvidiaSmiGpuProbe(),
            source_wrapper_profile=args.source_wrapper_profile,
            external_training_binding=args.external_training_binding,
        )
    except FullChainBootstrapError as error:
        sys.stderr.write(f"{error.category}\n")
        return 1
    except Exception:
        sys.stderr.write("bootstrap_invalid\n")
        return 1

    sys.stdout.write("p6_full_chain_config_written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
