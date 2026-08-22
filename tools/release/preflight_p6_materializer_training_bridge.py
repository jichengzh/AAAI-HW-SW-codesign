"""Read-only freshness gate for a P6 materializer/training controller run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal, Mapping

from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XExecutionError,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_history_binding_v1 import validate_history_execution_binding
from framework.stage6.p6_history_measurement_v1 import plan_validated_history_round_paths
from framework.stage6.p6_history_training_contract_v1 import (
    validate_recipe_v2_training_template,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_source_reuse_evidence_v1 import plan_source_reuse_paths
from framework.stage6.p6_source_wrapper_profile_v1 import (
    validate_self_contained_source_wrapper,
)


@dataclass(frozen=True)
class P6MaterializerPreflightReport:
    schema_version: Literal["p6_materializer_training_bridge_preflight_v1"]
    status: Literal["accepted"]
    validated_round_count: int
    wrapper_marker: str
    training_required: bool
    historical_process_launch_count: int
    gpu_probe_count: int


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError
    return payload


def _require_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        pass
    raise P6CoptV2XExecutionError("unsafe_destination", "unsafe destination")


def _private_root(binding: Mapping[str, Any]) -> Path:
    root = binding.get("private_root")
    if not isinstance(root, str):
        raise ValueError
    return Path(root)


def preflight_materializer_training_bridge(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
    runner_template_path: Path,
    source_wrapper_profile_path: Path,
) -> P6MaterializerPreflightReport:
    """Validate only static inputs and deterministic absent destinations."""
    try:
        contract = load_public_contract(public_contract_path)
        local = load_local_config(local_config_path, contract)
        binding = _load_json_mapping(private_binding_path)
        interface = validate_history_execution_binding(binding)
        private_root = _private_root(binding)
        template = binding.get("source_contract_template")
        if not isinstance(template, Mapping):
            raise ValueError
        validated_template = validate_recipe_v2_training_template(
            template, private_root=private_root
        )
        runner_template = validate_pre_provision_runner_template(
            runner_template_path, private_root, require_exact_history_environment=True
        )
        wrapper = validate_self_contained_source_wrapper(
            runner_template, source_wrapper_profile=source_wrapper_profile_path
        )
        reuse_paths = plan_source_reuse_paths(local.local_output_root)
        for path in (
            reuse_paths.metadata_root,
            reuse_paths.run_context,
            reuse_paths.receipt_root,
            local.local_output_root / "materialized",
            local.local_output_root / "source_registry.json",
            local.local_output_root / "pyramid_candidate_plan.json",
            local.local_output_root / "state.json",
        ):
            _require_absent(path)
        if local.candidate_source_mode == "framework_stage2_search_space":
            if local.stage2_search_space_path is None:
                raise ValueError
            _require_absent(local.stage2_search_space_path)
        for round_index in range(4):
            _require_absent(local.local_output_root / f"round-{round_index:02d}")
            planned = plan_validated_history_round_paths(
                interface, private_root, round_index
            )
            for path in planned.values():
                _require_absent(path)
        marker = wrapper.marker_basename
        if marker != "stage5_materialize_round_sources_v1.sh":
            raise ValueError
        return P6MaterializerPreflightReport(
            schema_version="p6_materializer_training_bridge_preflight_v1",
            status="accepted",
            validated_round_count=4,
            wrapper_marker=marker,
            training_required=validated_template["training_required"] is True,
            historical_process_launch_count=0,
            gpu_probe_count=0,
        )
    except P6CoptV2XExecutionError:
        raise
    except Exception:
        raise P6CoptV2XExecutionError(
            "history_execution_invalid", "history execution invalid"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--runner-template", type=Path, required=True)
    parser.add_argument("--source-wrapper-profile", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        report = preflight_materializer_training_bridge(
            public_contract_path=args.contract,
            local_config_path=args.local_config,
            private_binding_path=args.binding,
            runner_template_path=args.runner_template,
            source_wrapper_profile_path=args.source_wrapper_profile,
        )
    except BaseException:
        sys.stderr.write("preflight_failed\n")
        return 1
    sys.stdout.write(json.dumps(asdict(report), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
