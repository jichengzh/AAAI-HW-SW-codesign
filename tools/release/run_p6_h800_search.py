"""Run the shared local P6.1 Pyramid/TVM hardware-profile search."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.coptv2x_h800_search_v2 import (  # noqa: E402
    P6CoptV2XContractError,
    P6CoptV2XExecutionError,
    load_local_config,
    load_public_contract,
    run_p6_coptv2x_search,
    validate_code_revision,
)
from framework.stage6.p6_full_chain_bootstrap_v1 import (  # noqa: E402
    FullChainBootstrapError,
    materialize_full_chain_binding,
)
from framework.stage6.p6_gpu_policy_v1 import (  # noqa: E402
    canonical_gpu_indices,
    parse_gpu_indices_csv,
    parse_runtime_gpu_pool,
)
from framework.stage6.p6_history_binding_v1 import (  # noqa: E402
    validate_history_execution_binding,
)
from tools.release.preflight_p6_materializer_training_bridge import (  # noqa: E402
    preflight_materializer_training_bridge,
)
from tools.release.provision_p6_history_local_config import (  # noqa: E402
    NvidiaSmiGpuProbe,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _revision_label(value: str) -> str:
    try:
        return validate_code_revision(value)
    except P6CoptV2XContractError as error:
        raise argparse.ArgumentTypeError("invalid code revision") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--local-config", type=Path)
    parser.add_argument("--code-revision", required=True, type=_revision_label)
    parser.add_argument("--legacy-local-config", type=Path)
    parser.add_argument("--runner-template", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--binding-output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--source-wrapper-profile", type=Path)
    parser.add_argument("--external-training-binding", type=Path)
    parser.add_argument("--post-source-adapter-profile", type=Path)
    args = parser.parse_args(argv)
    fresh_names = (
        "legacy_local_config",
        "runner_template",
        "local_output_root",
        "binding_output",
        "config_output",
        "source_wrapper_profile",
        "external_training_binding",
        "post_source_adapter_profile",
    )
    fresh_values = tuple(getattr(args, name) for name in fresh_names)
    if args.local_config is not None:
        if any(value is not None for value in fresh_values):
            parser.error("local and fresh modes are mutually exclusive")
    elif any(value is None for value in fresh_values):
        parser.error("fresh mode requires the complete private input group")
    return args


def _run_command(argv: tuple[str, ...], cwd: Path) -> int:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode


def _binding_path_from_local(local: object) -> Path:
    paths: list[Path] = []
    for step_name in ("source_registry_step", "measurement_step"):
        argv = getattr(getattr(local, step_name), "argv")
        if argv.count("--binding") != 1:
            raise P6CoptV2XContractError("runtime GPU binding is unavailable")
        position = argv.index("--binding") + 1
        if position >= len(argv):
            raise P6CoptV2XContractError("runtime GPU binding is unavailable")
        paths.append(Path(argv[position]))
    if paths[0] != paths[1] or not paths[0].is_absolute():
        raise P6CoptV2XContractError("runtime GPU binding is unavailable")
    return paths[0]


def _existing_gpu_binding_indices(local: object, gpu_count: int) -> tuple[int, ...]:
    try:
        binding_path = _binding_path_from_local(local)
        raw = json.loads(binding_path.read_text(encoding="utf-8"))
        interface = validate_history_execution_binding(
            raw,
            profile=getattr(local, "hardware_profile"),
        )
        values = interface["environment"]["values"]
        bound = parse_gpu_indices_csv(values["CUDA_VISIBLE_DEVICES"]["value"])
        if len(bound) != gpu_count:
            raise ValueError
        return bound
    except Exception as error:
        if isinstance(error, P6CoptV2XContractError):
            raise
        raise P6CoptV2XContractError(
            "runtime GPU binding does not match GPU_POOL"
        ) from error


def _materialize_and_preflight(
    args: argparse.Namespace, gpu_count: int
) -> tuple[int, ...]:
    binding = materialize_full_chain_binding(
        args.legacy_local_config,
        args.runner_template,
        args.local_output_root,
        args.binding_output,
        args.config_output,
        NvidiaSmiGpuProbe(),
        source_wrapper_profile=args.source_wrapper_profile,
        external_training_binding=args.external_training_binding,
        post_source_adapter_profile=args.post_source_adapter_profile,
        runtime_gpu_count=gpu_count,
    )
    try:
        selected_indices = canonical_gpu_indices(binding["gpu_policy"]["indices"])
    except (KeyError, TypeError, ValueError) as error:
        raise P6CoptV2XContractError("runtime GPU binding is unavailable") from error
    if len(selected_indices) != gpu_count:
        raise P6CoptV2XContractError("runtime GPU binding does not match GPU_POOL")
    preflight_materializer_training_bridge(
        public_contract_path=args.contract,
        local_config_path=args.config_output,
        private_binding_path=args.binding_output,
        runner_template_path=args.runner_template,
        source_wrapper_profile_path=args.source_wrapper_profile,
        external_training_binding_path=args.external_training_binding,
        post_source_adapter_profile_path=args.post_source_adapter_profile,
    )
    return selected_indices


def main(argv: Sequence[str] | None = None) -> int:
    """Load a legacy H800 or selected v3 profile and return stable exit codes."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    fresh_run = args.local_config is None
    runtime_gpu_count: int | None = None
    if fresh_run or "GPU_POOL" in os.environ:
        try:
            runtime_gpu_count = parse_runtime_gpu_pool(os.environ)
        except ValueError:
            sys.stderr.write("contract_error\n")
            return 2

    try:
        contract = load_public_contract(args.contract)
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2

    if fresh_run:
        try:
            if runtime_gpu_count is None:
                raise P6CoptV2XContractError("runtime GPU pool is required")
            runtime_gpu_indices = _materialize_and_preflight(args, runtime_gpu_count)
        except FullChainBootstrapError as error:
            if error.category == "gpu_admission":
                sys.stderr.write("gpu_admission\n")
                return 1
            sys.stderr.write("contract_error\n")
            return 2
        except (P6CoptV2XContractError, P6CoptV2XExecutionError):
            sys.stderr.write("contract_error\n")
            return 2
        except Exception:
            sys.stderr.write("execution_failed\n")
            return 1
        local_config_path = args.config_output
    else:
        local_config_path = args.local_config
        runtime_gpu_indices = None

    try:
        local = load_local_config(local_config_path, contract)
        if runtime_gpu_count is not None and not fresh_run:
            runtime_gpu_indices = _existing_gpu_binding_indices(
                local, runtime_gpu_count
            )
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2

    try:
        state = run_p6_coptv2x_search(
            contract,
            local,
            args.code_revision,
            _run_command,
            runtime_gpu_indices=runtime_gpu_indices,
        )
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2
    except P6CoptV2XExecutionError as error:
        if error.failure_code == "unsafe_output":
            sys.stderr.write("contract_error\n")
            return 2
        sys.stderr.write("execution_failed\n")
        return 1
    except OSError:
        sys.stderr.write("execution_failed\n")
        return 1
    if state.status != "completed":
        sys.stderr.write("execution_failed\n")
        return 1
    sys.stdout.write("completed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
