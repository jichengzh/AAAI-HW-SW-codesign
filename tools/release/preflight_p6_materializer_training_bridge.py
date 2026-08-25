"""Read-only freshness gate for a P6 materializer/training controller run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.coptv2x_h800_search_v2 import (  # noqa: E402
    P6CoptV2XExecutionError,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_history_binding_v1 import (  # noqa: E402
    validate_history_execution_binding,
)
from framework.stage6.p6_external_training_binding_v1 import (  # noqa: E402
    external_training_binding_from_contract,
    external_training_binding_to_mapping,
    load_external_training_binding,
    validate_external_training_binding,
)
from framework.stage6.p6_history_measurement_v1 import (  # noqa: E402
    plan_validated_history_round_paths,
)
from framework.stage6.p6_history_execution_closure_v1 import (  # noqa: E402
    validate_post_source_wrapper_runner_binding,
)
from framework.stage6.p6_history_training_contract_v1 import (  # noqa: E402
    validate_recipe_v2_training_template,
)
from framework.stage6.p6_runner_template_validator_v1 import (  # noqa: E402
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (  # noqa: E402
    load_post_source_adapter_profile,
)
from framework.stage6.p6_post_source_wrapper_template_v1 import (  # noqa: E402
    validate_post_source_adapter_wrappers,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (  # noqa: E402
    plan_source_reuse_paths,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (  # noqa: E402
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


def _validate_post_source_profile(
    profile_path: Path | None,
    *,
    private_root: Path,
    runner_template_path: Path,
) -> None:
    if profile_path is None:
        if (private_root / "post-source-adapter-profile.yaml").is_file():
            raise ValueError
        return
    profile = load_post_source_adapter_profile(profile_path, private_root=private_root)
    wrapper_paths = validate_post_source_adapter_wrappers(
        profile,
        private_root=private_root,
    )
    validate_post_source_wrapper_runner_binding(
        runner_template_path,
        normalized_private_root=private_root,
        post_source_wrapper_paths=wrapper_paths,
    )


def _reserved_paths(
    local: Any,
    interface: Mapping[str, Any],
    private_root: Path,
    private_binding_path: Path,
    local_config_path: Path,
) -> tuple[Any, list[Path]]:
    reuse_paths = plan_source_reuse_paths(local.local_output_root)
    fixed = [
        private_binding_path,
        local_config_path,
        reuse_paths.metadata_root,
        reuse_paths.run_context,
        reuse_paths.receipt_root,
        local.local_output_root / "materialized",
        local.local_output_root / "source_registry.json",
        local.local_output_root / "pyramid_candidate_plan.json",
        local.local_output_root / "state.json",
    ]
    planned = [
        path
        for round_index in range(4)
        for path in plan_validated_history_round_paths(
            interface, private_root, round_index
        ).values()
    ]
    return reuse_paths, [*fixed, *planned]


def _require_fresh_destinations(
    local: Any,
    interface: Mapping[str, Any],
    private_root: Path,
    reuse_paths: Any,
) -> None:
    fixed = (
        reuse_paths.metadata_root,
        reuse_paths.run_context,
        reuse_paths.receipt_root,
        local.local_output_root / "materialized",
        local.local_output_root / "source_registry.json",
        local.local_output_root / "pyramid_candidate_plan.json",
        local.local_output_root / "state.json",
    )
    for path in fixed:
        _require_absent(path)
    if local.candidate_source_mode == "framework_stage2_search_space":
        if local.stage2_search_space_path is None:
            raise ValueError
        _require_absent(local.stage2_search_space_path)
    for round_index in range(4):
        _require_absent(local.local_output_root / f"round-{round_index:02d}")
        planned = plan_validated_history_round_paths(interface, private_root, round_index)
        for path in planned.values():
            _require_absent(path)


def _load_preflight_inputs(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
    runner_template_path: Path,
    source_wrapper_profile_path: Path,
    post_source_adapter_profile_path: Path | None,
) -> tuple[Any, Mapping[str, Any], Path, Mapping[str, Any], Any]:
    contract = load_public_contract(public_contract_path)
    local = load_local_config(local_config_path, contract)
    if local.candidate_source_mode != "framework_stage2_search_space":
        raise ValueError
    binding = _load_json_mapping(private_binding_path)
    interface = validate_history_execution_binding(binding)
    private_root = _private_root(binding)
    template = binding.get("source_contract_template")
    if not isinstance(template, Mapping):
        raise ValueError
    validated_template = validate_recipe_v2_training_template(
        template, private_root=private_root
    )
    _validate_post_source_profile(
        post_source_adapter_profile_path,
        private_root=private_root,
        runner_template_path=runner_template_path,
    )
    runner_template = validate_pre_provision_runner_template(
        runner_template_path, private_root, require_exact_history_environment=True
    )
    wrapper = validate_self_contained_source_wrapper(
        runner_template, source_wrapper_profile=source_wrapper_profile_path
    )
    return local, interface, private_root, validated_template, wrapper


def _validate_external_preflight(
    *,
    local: Any,
    interface: Mapping[str, Any],
    private_root: Path,
    private_binding_path: Path,
    local_config_path: Path,
    external_training_binding_path: Path,
    validated_template: Mapping[str, Any],
) -> Any:
    reuse_paths, reserved_paths = _reserved_paths(
        local, interface, private_root, private_binding_path, local_config_path
    )
    external = validate_external_training_binding(
        load_external_training_binding(external_training_binding_path),
        code_toolchain_root=private_root,
        local_output_root=local.local_output_root,
        reserved_paths=reserved_paths,
    )
    if external_training_binding_to_mapping(
        external
    ) != external_training_binding_from_contract(validated_template):
        raise ValueError
    _require_fresh_destinations(local, interface, private_root, reuse_paths)
    return external


def preflight_materializer_training_bridge(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
    runner_template_path: Path,
    source_wrapper_profile_path: Path,
    external_training_binding_path: Path,
    post_source_adapter_profile_path: Path | None = None,
) -> P6MaterializerPreflightReport:
    """Validate only static inputs and deterministic absent destinations."""
    try:
        local, interface, private_root, validated_template, wrapper = (
            _load_preflight_inputs(
                public_contract_path=public_contract_path,
                local_config_path=local_config_path,
                private_binding_path=private_binding_path,
                runner_template_path=runner_template_path,
                source_wrapper_profile_path=source_wrapper_profile_path,
                post_source_adapter_profile_path=post_source_adapter_profile_path,
            )
        )
        external = _validate_external_preflight(
            local=local,
            interface=interface,
            private_root=private_root,
            private_binding_path=private_binding_path,
            local_config_path=local_config_path,
            external_training_binding_path=external_training_binding_path,
            validated_template=validated_template,
        )
        marker = wrapper.marker_basename
        if marker != "stage5_materialize_round_sources_v1.sh":
            raise ValueError
        return P6MaterializerPreflightReport(
            schema_version="p6_materializer_training_bridge_preflight_v1",
            status="accepted",
            validated_round_count=4,
            wrapper_marker=marker,
            training_required=external.training_required,
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
    parser.add_argument("--external-training-binding", type=Path, required=True)
    parser.add_argument("--post-source-adapter-profile", type=Path)
    try:
        args = parser.parse_args(argv)
        report = preflight_materializer_training_bridge(
            public_contract_path=args.contract,
            local_config_path=args.local_config,
            private_binding_path=args.binding,
            runner_template_path=args.runner_template,
            source_wrapper_profile_path=args.source_wrapper_profile,
            external_training_binding_path=args.external_training_binding,
            post_source_adapter_profile_path=args.post_source_adapter_profile,
        )
    except Exception:
        sys.stderr.write("preflight_failed\n")
        return 1
    sys.stdout.write(json.dumps(asdict(report), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
