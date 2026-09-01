"""Contracts and local state machine for the P6 CoptV2X hardware search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal

import yaml

from framework.stage1_bridge import load_stage2_search_space
from framework.stage2.canonical_search_v3 import validate_capability_profile
from framework.stage5.production_search_v1 import predict_candidate_rows
from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_measurement_request,
    build_task_candidate_manifest,
    fit_initial_coldstart_bundle,
    fit_online_bundle,
    freeze_initial_coldstart,
    select_task_batch,
    validate_search_task,
    validate_task_feedback_history,
)
from framework.stage6.pyramid_search_space_adapter_v1 import build_pyramid_candidate_plan
from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    default_hardware_execution_profile,
    load_hardware_execution_profile,
    validate_profile_backend,
)
from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    project_source_materialization_request,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    P6SourceReuseEvidenceError,
    create_fresh_run_context,
)


PUBLIC_SCHEMA_VERSION = "p6_h800_coptv2x_search_contract_v2"
LOCAL_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
PUBLIC_SCHEMA_VERSION_V3 = "p6_coptv2x_search_contract_v3"
LOCAL_SCHEMA_VERSION_V3 = "p6_coptv2x_local_v3"
METRIC_NAMES = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
SUCCESS_STATUS = "measured_success_gold"
TRUE_FAILURE_STATUSES = frozenset({"feasibility_failure", "numerical_feasibility_failure"})
FEEDBACK_SCHEMA_VERSION = "p6_h800_coptv2x_feedback_v2"
FAILURE_SCHEMA_VERSION = "p6_h800_coptv2x_failure_v2"
FIXED_TARGET = "h800"
FIXED_MODEL = "pyramid"
FIXED_BACKEND = "tvm_auto"
FIXED_SAMPLE_BUDGET = 16
FIXED_BATCH_SIZE = 4
FIXED_ROUND_COUNT = 4
FIXED_SOURCE_GROUP_COUNT = 343
FIXED_ELIGIBLE_GENOME_COUNT = 686
FIXED_COLDSTART_PROFILE_COUNT = 2
GRAPH_METADATA_FIELDS = frozenset({"group_id", "model", "width", "input_dims"})
STRUCTURAL_GRAPH_FEATURE_FIELDS = frozenset(
    {
        "add_count",
        "arithmetic_intensity_proxy",
        "concat_count",
        "conv_count",
        "conv_flops",
        "conv_macs",
        "conv_output_elements",
        "conv_parameter_elements",
        "conv_transpose_count",
        "depthwise_conv_count",
        "group_conv_count",
        "input_channels",
        "input_elements",
        "input_height",
        "input_width",
        "kernel1_conv_count",
        "kernel3_conv_count",
        "max_conv_channels",
        "node_count",
        "parameter_elements",
        "relu_count",
        "stride2_conv_count",
    }
)
EXPECTED_CLOSURE_HEADS = {
    "latency_ms": "extra_trees_log",
    "energy_j": "extra_trees_log",
    "ap70": "lgbm_huber_residual",
}
EXPECTED_UNCERTAINTY_POLICY = "lgbm_quantile_plus_group_conformal"
EXPECTED_ACQUISITION_POLICY = "predicted_frontier_diversity"
LOCAL_INPUT_NAMES = frozenset(
    {"gold176_rows", "gold176_graph_features", "capability_profiles", "closure"}
)
ALLOWED_TEMPLATE_TOKENS = frozenset(
    {
        "{local_output_root}",
        "{stage1_partition_manifest}",
        "{source_registry_json}",
        "{pyramid_candidate_plan}",
        "{measurement_request}",
        "{feedback_json}",
        "{round_output_root}",
    }
)
PUBLIC_KEYS = frozenset(
    {
        "schema_version",
        "search_id",
        "target",
        "target_model",
        "execution_backend",
        "seed",
        "sample_budget",
        "batch_size",
        "round_count",
        "configuration_label",
        "candidate_space_label",
        "metric_names",
        "assets",
    }
)
PUBLIC_V3_KEYS = PUBLIC_KEYS | {"hardware_profile"}
LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "local_input_paths",
        "candidate_source_mode",
        "stage2_search_space_path",
        "stage1_scan_step",
        "source_registry_step",
        "measurement_step",
        "local_output_root",
    }
)
LOCAL_V3_KEYS = LOCAL_KEYS | {"hardware_profile"}
_OPTIONAL_LEGACY_LOCAL_KEYS = frozenset(
    {"candidate_source_mode", "stage2_search_space_path", "stage1_scan_step"}
)

_ASSET_KEYS = frozenset({"label", "version", "license_status"})
_STEP_KEYS = frozenset({"name", "argv"})
_SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "bash.exe",
        "cmd",
        "cmd.exe",
        "csh",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "sh.exe",
        "tcsh",
        "zsh",
    }
)
_COMMAND_WRAPPER_EXECUTABLES = frozenset({"env", "env.exe"})
_PUBLIC_RESTRICTED_VALUE_PATTERNS = (
    re.compile(r"(?:^|[-_\s])host(?:name)?(?:[-_\s]|\d|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])raw[-_\s]?logs?(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])checkpoints?(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(
        r"(?:^|[-_\s])candidate(?:[-_\s]+id)?[-_\s]+[a-z0-9]+(?:[-_\s]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[-_\s])(?:sha(?:1|224|256|384|512)?|hash)(?:[:=_-]|$)", re.IGNORECASE),
    re.compile(
        r"^(?:python(?:\d+(?:\.\d+)?)?|bash|sh|zsh|cmd(?:\.exe)?|powershell|pwsh)\b",
        re.IGNORECASE,
    ),
)
_RAW_DIGEST_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}|[0-9a-f]{128}", re.IGNORECASE)
_PUBLIC_FAILURE_REASON_PATTERN = re.compile(r"^[a-z0-9_-]+$")
_RECOVERABLE_ROUND_FAILURE_CODES = frozenset(
    {
        "command_failed",
        "feedback_missing",
        "feedback_invalid_json",
        "feedback_request_mismatch",
        "feedback_candidate_mismatch",
        "feedback_metrics_invalid",
        "feedback_terminal_status_invalid",
        "history_execution_invalid",
    }
)


class P6CoptV2XContractError(ValueError):
    """Raised when a P6 public or local CoptV2X contract is invalid."""


class P6CoptV2XExecutionError(RuntimeError):
    """Raised when an argv-only local P6 execution step fails."""

    def __init__(self, failure_code: str, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


def validate_code_revision(value: str) -> str:
    """Validate one explicit public-safe revision label for local state."""
    return _require_public_identifier(value, "code_revision")


@dataclass(frozen=True)
class RegisteredAsset:
    label: str
    version: str
    license_status: str


@dataclass(frozen=True)
class PublicP6CoptV2XContract:
    search_id: str
    target: str
    target_model: str
    execution_backend: str
    seed: int
    sample_budget: int
    batch_size: int
    round_count: int
    configuration_label: str
    candidate_space_label: str
    assets: tuple[RegisteredAsset, ...]
    metric_names: tuple[str, ...]
    hardware_profile: HardwareExecutionProfile = field(
        default_factory=default_hardware_execution_profile
    )


@dataclass(frozen=True)
class LocalExecutionStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class LocalP6CoptV2XConfig:
    asset_paths: Mapping[str, Path]
    local_input_paths: Mapping[str, Path]
    candidate_source_mode: Literal[
        "coptv2x_static_registry", "framework_stage2_search_space"
    ]
    stage2_search_space_path: Path | None
    stage1_scan_step: LocalExecutionStep | None
    source_registry_step: LocalExecutionStep
    measurement_step: LocalExecutionStep
    local_output_root: Path
    hardware_profile: HardwareExecutionProfile = field(
        default_factory=default_hardware_execution_profile
    )


@dataclass(frozen=True)
class P6CoptV2XRunState:
    schema_version: str
    status: Literal["completed", "failed"]
    completed_rounds: int
    measured_candidate_count: int
    failure_code: str | None
    local_state_path: Path


CommandRunner = Callable[[tuple[str, ...], Path], int]


def run_p6_coptv2x_search(
    contract: PublicP6CoptV2XContract,
    local: LocalP6CoptV2XConfig,
    code_revision: str,
    command_runner: CommandRunner,
) -> P6CoptV2XRunState:
    """Run the fixed four-round local Pyramid/TVM search state machine."""
    if contract.hardware_profile.profile_id != local.hardware_profile.profile_id:
        raise P6CoptV2XContractError(
            "local hardware profile must match the public contract"
        )
    code_revision = validate_code_revision(code_revision)
    _prepare_local_output_root(local.local_output_root)
    if local.candidate_source_mode == "framework_stage2_search_space":
        _require_controller_destinations_absent(local)
    _run_framework_stage1_scan(local, command_runner)
    frozen_gold, gold_graphs, capability_profiles, profile = _load_search_inputs(
        local, contract
    )
    task = _build_search_task(contract, profile)
    try:
        source_registry, framework_plan = _build_source_registry(local, command_runner)
        _validate_p6_source_space(source_registry, task, framework_plan=framework_plan)
    except (P6CoptV2XContractError, ValueError):
        raise P6CoptV2XExecutionError(
            "source_registry_invalid", "source registry invalid"
        ) from None
    if local.candidate_source_mode == "framework_stage2_search_space":
        context_registry = _normalize_recipe_v2_registry_paths(source_registry)
        _write_json(local.local_output_root / "source_registry.json", context_registry)
        task_contract = validate_search_task(task)
        if framework_plan is None:
            raise P6CoptV2XExecutionError(
                "source_registry_invalid", "source registry invalid"
            )
        try:
            create_fresh_run_context(
                local_output_root=local.local_output_root,
                task_contract=task_contract,
                code_revision=code_revision,
                candidate_plan=framework_plan,
                source_registry=context_registry,
            )
        except P6SourceReuseEvidenceError as error:
            raise P6CoptV2XExecutionError(
                error.public_category, "history execution invalid"
            ) from None
    online_feedback_rows: list[dict[str, Any]] = []
    measured_row_ids: set[str] = set()
    frozen_holdout_group_ids = {str(row["group_id"]) for row in frozen_gold}
    gold_selection_rows = _attach_graph_features(frozen_gold, gold_graphs)
    for round_index in range(contract.round_count):
        try:
            released_rows = _run_search_round(
                contract=contract,
                local=local,
                task=task,
                capability_profiles=capability_profiles,
                source_registry=source_registry,
                frozen_gold=frozen_gold,
                gold_graphs=gold_graphs,
                gold_selection_rows=gold_selection_rows,
                online_feedback_rows=online_feedback_rows,
                measured_row_ids=measured_row_ids,
                frozen_holdout_group_ids=frozen_holdout_group_ids,
                round_index=round_index,
                command_runner=command_runner,
            )
        except P6CoptV2XExecutionError as error:
            if error.failure_code not in _RECOVERABLE_ROUND_FAILURE_CODES:
                raise
            _write_round_failure(
                local.local_output_root,
                round_index=round_index,
                failure_code=error.failure_code,
                completed_rounds=round_index,
            )
            return _failed_run_state(
                local,
                code_revision=code_revision,
                completed_rounds=round_index,
                measured_candidate_count=len(measured_row_ids),
                failure_code=error.failure_code,
            )
        online_feedback_rows = [*online_feedback_rows, *released_rows]
        measured_row_ids.update(str(row["row_id"]) for row in released_rows)
        if round_index < contract.round_count - 1:
            validate_task_feedback_history(
                online_feedback_rows,
                task=task,
                completed_rounds=round_index + 1,
            )
    return _complete_run_state(
        local,
        code_revision=code_revision,
        completed_rounds=contract.round_count,
        measured_candidate_count=len(measured_row_ids),
    )


def _run_framework_stage1_scan(
    local: LocalP6CoptV2XConfig, command_runner: CommandRunner
) -> None:
    """Generate and validate the Stage1 manifest that feeds framework candidates."""
    if local.candidate_source_mode != "framework_stage2_search_space":
        return
    try:
        if local.stage1_scan_step is None or local.stage2_search_space_path is None:
            raise P6CoptV2XContractError("framework Stage1 scan configuration is missing")
        _require_local_output_absent(local.stage2_search_space_path)
        _run_step(
            local.stage1_scan_step,
            {
                "{stage1_partition_manifest}": local.stage2_search_space_path,
                "{local_output_root}": local.local_output_root,
            },
            cwd=local.local_output_root,
            runner=command_runner,
        )
        _validate_stage1_partition_manifest(
            local.stage2_search_space_path, local.hardware_profile
        )
    except Exception:
        raise P6CoptV2XExecutionError(
            "stage1_scan_invalid", "stage1 scan invalid"
        ) from None


def _validate_stage1_partition_manifest(
    path: Path, hardware_profile: HardwareExecutionProfile
) -> None:
    manifest = _load_mapping(path, "stage1 partition manifest")
    if (
        manifest.get("schema") != "stage1_partition_manifest_v1"
        or manifest.get("stage") != "stage1_partition"
        or manifest.get("model") != "pyramid_lidar"
        or manifest.get("scan_status") != "ok"
    ):
        raise P6CoptV2XContractError("Stage1 partition manifest is invalid")
    hardware = manifest.get("hw_capability")
    hardware_name = hardware.get("name") if isinstance(hardware, Mapping) else None
    if not _matches_profile_hardware_target(hardware_profile, hardware_name):
        raise P6CoptV2XContractError(
            "Stage1 hardware target does not match the hardware profile"
        )
    search_space = load_stage2_search_space(path)
    _validate_stage2_hardware_target(search_space, hardware_profile)


def _matches_profile_hardware_target(
    hardware_profile: HardwareExecutionProfile, hardware_name: object
) -> bool:
    if hardware_name == hardware_profile.target_hardware_id:
        return True
    return (
        hardware_profile.profile_id == "h800"
        and isinstance(hardware_name, str)
        and hardware_name.startswith(f"{FIXED_TARGET}_")
    )


def _validate_stage2_hardware_target(
    search_space: Mapping[str, Any], hardware_profile: HardwareExecutionProfile
) -> None:
    hardware = search_space.get("hardware_target")
    hardware_name = hardware.get("name") if isinstance(hardware, Mapping) else None
    if not _matches_profile_hardware_target(hardware_profile, hardware_name):
        raise P6CoptV2XContractError(
            "Stage2 hardware target does not match the hardware profile"
        )


def _load_search_inputs(
    local: LocalP6CoptV2XConfig,
    contract: PublicP6CoptV2XContract | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    gold_rows = _require_mapping_rows(
        _read_local_input_json(local.local_input_paths["gold176_rows"]),
        "gold176 rows",
    )
    gold_graphs = _normalize_gold_graph_features(
        _require_mapping_rows(
            _read_local_input_json(local.local_input_paths["gold176_graph_features"]),
            "gold176 graph features",
        )
    )
    raw_profiles = _require_mapping_rows(
        _read_local_input_json(local.local_input_paths["capability_profiles"]),
        "capability profiles",
    )
    closure = _read_local_input_json(local.local_input_paths["closure"])
    _validate_closure(closure)
    frozen_gold = freeze_initial_coldstart(gold_rows)
    capability_profiles = [
        validate_capability_profile(profile) for profile in raw_profiles
    ]
    coldstart_profile_ids = {
        str(row.get("capability_profile_id") or "") for row in frozen_gold
    }
    capability_profile_ids = {
        str(profile.get("capability_profile_id") or "")
        for profile in capability_profiles
    }
    if (
        len(coldstart_profile_ids) != FIXED_COLDSTART_PROFILE_COUNT
        or len(capability_profiles) != FIXED_COLDSTART_PROFILE_COUNT
        or capability_profile_ids != coldstart_profile_ids
    ):
        raise P6CoptV2XContractError("coldstart capability context is incomplete")
    profile = _select_profile(capability_profiles, contract)
    return frozen_gold, gold_graphs, capability_profiles, profile


def _normalize_gold_graph_features(
    graph_features: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project legacy graph evidence to the reviewed structural feature schema."""
    normalized: list[dict[str, Any]] = []
    for raw_graph in graph_features:
        graph = dict(raw_graph)
        normalized.append(
            {
                str(name): copy.deepcopy(value)
                if str(name) in GRAPH_METADATA_FIELDS
                else float(value)
                for name, value in graph.items()
                if str(name) in GRAPH_METADATA_FIELDS
                or (str(name) in STRUCTURAL_GRAPH_FEATURE_FIELDS and _finite(value))
            }
        )
    return normalized


def _build_source_registry(
    local: LocalP6CoptV2XConfig, command_runner: CommandRunner
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    source_registry_path = local.local_output_root / "source_registry.json"
    if local.candidate_source_mode == "framework_stage2_search_space":
        _require_local_output_absent(source_registry_path)
    else:
        _validate_local_output_leaf(source_registry_path)
    plan_path: Path | None = None
    plan: Mapping[str, Any] | None = None
    if local.candidate_source_mode == "framework_stage2_search_space":
        assert local.stage2_search_space_path is not None
        search_space = load_stage2_search_space(local.stage2_search_space_path)
        _validate_stage2_hardware_target(search_space, local.hardware_profile)
        plan = build_pyramid_candidate_plan(
            search_space, profile=local.hardware_profile
        )
        plan_path = local.local_output_root / "pyramid_candidate_plan.json"
        _require_local_output_absent(plan_path)
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _run_step(
        local.source_registry_step,
        {
            "{local_output_root}": local.local_output_root,
            "{source_registry_json}": source_registry_path,
            **(
                {"{pyramid_candidate_plan}": plan_path}
                if plan_path is not None
                else {}
            ),
        },
        cwd=local.local_output_root,
        runner=command_runner,
    )
    _validate_local_output_leaf(source_registry_path)
    source_registry = _read_source_registry_json(source_registry_path)
    if plan is not None:
        _validate_framework_registry_plan(source_registry, plan)
    return source_registry, plan


def _validate_framework_registry_plan(
    source_registry: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    """Require local materialization to preserve every dynamic candidate identity."""
    if source_registry.get("schema_version") != "stage5_candidate_source_registry_v2":
        raise P6CoptV2XContractError("framework registry schema must be v2")
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        raise P6CoptV2XContractError("framework registry plan identities are invalid")
    expected = _framework_plan_candidate_mapping(plan)
    actual_entries: list[tuple[tuple[tuple[int, ...], str], tuple[str, ...]]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise P6CoptV2XContractError("framework registry plan identities are invalid")
        width = _framework_width_identity(group)
        available_q_modes = group.get("available_q_modes")
        source_point_ids_by_q_mode = group.get("source_point_ids_by_q_mode")
        if (
            not isinstance(available_q_modes, list)
            or not available_q_modes
            or any(not isinstance(q_mode, str) for q_mode in available_q_modes)
            or not isinstance(source_point_ids_by_q_mode, Mapping)
            or set(source_point_ids_by_q_mode) != set(available_q_modes)
        ):
            raise P6CoptV2XContractError("framework registry plan identities are invalid")
        for q_mode in available_q_modes:
            actual_entries.append(
                (
                    (width, q_mode),
                    _framework_source_point_ids(source_point_ids_by_q_mode[q_mode]),
                )
            )
    actual = dict(actual_entries)
    if len(actual) != len(actual_entries) or actual != expected:
        raise P6CoptV2XContractError("framework registry plan identities do not match")


def _normalize_recipe_v2_registry_paths(
    source_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Expose recipe-v2 declared leaves to the Task 4 context freshness gate."""
    try:
        normalized = json.loads(json.dumps(source_registry, allow_nan=False))
        groups = normalized["groups"]
        if not isinstance(groups, list) or not groups:
            raise ValueError
        for group in groups:
            contract = group["source_contract"]
            paths = contract["shared_source_paths"]
            if not isinstance(contract, dict) or not isinstance(paths, Mapping):
                raise ValueError
            for key in (
                "checkpoint_path", "checkpoint_dir", "config_path", "training_done_marker",
                "onnx_path", "onnx_report_path", "calibration_root", "calibration_npz",
                "calibration_summary", "trt_calibration_dir", "source_done_marker",
            ):
                if key not in paths or key in contract:
                    raise ValueError
                contract[key] = paths[key]
            group["source_contract_sha256"] = _canonical_json_sha256(contract)
        return normalized
    except (KeyError, TypeError, ValueError):
        raise P6CoptV2XExecutionError(
            "source_registry_invalid", "source registry invalid"
        ) from None


def _framework_width_identity(value: Mapping[str, Any]) -> tuple[int, ...]:
    width = value.get("width")
    if (
        not isinstance(width, list)
        or not width
        or any(isinstance(item, bool) or not isinstance(item, int) for item in width)
    ):
        raise P6CoptV2XContractError("framework candidate width is invalid")
    return tuple(width)


def _framework_source_point_ids(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise P6CoptV2XContractError("framework candidate provenance is invalid")
    return tuple(value)


def _framework_plan_candidate_mapping(
    plan: Mapping[str, Any],
) -> dict[tuple[tuple[int, ...], str], tuple[str, ...]]:
    if plan.get("schema_version") != "p6_pyramid_candidate_plan_v2":
        raise P6CoptV2XContractError("framework candidate plan schema is invalid")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise P6CoptV2XContractError("framework candidate plan identities are invalid")
    entries: list[tuple[tuple[tuple[int, ...], str], tuple[str, ...]]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise P6CoptV2XContractError("framework candidate plan identities are invalid")
        q_mode = candidate.get("q_mode")
        if q_mode not in {"fp16", "int8"}:
            raise P6CoptV2XContractError("framework candidate q_mode is invalid")
        entries.append(
            (
                (_framework_width_identity(candidate), q_mode),
                _framework_source_point_ids(candidate.get("source_point_ids")),
            )
        )
    mapping = dict(entries)
    if len(mapping) != len(entries):
        raise P6CoptV2XContractError("framework candidate plan identities are duplicated")
    return mapping


def _validate_p6_source_space(
    source_registry: Mapping[str, Any],
    task: SearchTask,
    *,
    framework_plan: Mapping[str, Any] | None,
) -> None:
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        raise P6CoptV2XContractError("P6 source space groups are invalid")
    if framework_plan is None and len(groups) != FIXED_SOURCE_GROUP_COUNT:
        raise P6CoptV2XContractError("P6 source space must contain exactly 343 groups")
    manifest = build_task_candidate_manifest(
        source_registry, task=task, measured_row_ids=set()
    )
    if framework_plan is None:
        if (
            manifest.get("eligible_row_count") != FIXED_ELIGIBLE_GENOME_COUNT
            or len(manifest.get("rows") or []) != FIXED_ELIGIBLE_GENOME_COUNT
        ):
            raise P6CoptV2XContractError("P6 source space must contain exactly 686 genomes")
        return
    candidates = framework_plan.get("candidates")
    candidate_count = framework_plan.get("candidate_count")
    if (
        not isinstance(candidates, list)
        or isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(candidates)
    ):
        raise P6CoptV2XContractError("framework candidate count is invalid")
    _validate_framework_manifest_identities(
        manifest, framework_plan, source_registry
    )
    eligible_row_count = manifest.get("eligible_row_count")
    if (
        isinstance(eligible_row_count, bool)
        or not isinstance(eligible_row_count, int)
        or eligible_row_count < task.sample_budget
    ):
        raise P6CoptV2XContractError("framework source space is below the sample budget")


def _validate_framework_manifest_identities(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_registry: Mapping[str, Any],
) -> None:
    """Require Stage5 rows to equal the materializable registry-plan subset."""
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise P6CoptV2XContractError("framework manifest identities are invalid")
    plan_identities = set(_framework_plan_candidate_mapping(plan))
    expected_entries = _framework_materializable_registry_identities(source_registry)
    expected = set(expected_entries)
    if len(expected) != len(expected_entries) or not expected <= plan_identities:
        raise P6CoptV2XContractError("framework manifest identities are invalid")
    actual_entries: list[tuple[tuple[int, ...], str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("q_mode") not in {"fp16", "int8"}:
            raise P6CoptV2XContractError("framework manifest identities are invalid")
        actual_entries.append((_framework_width_identity(row), str(row["q_mode"])))
    actual = set(actual_entries)
    if len(actual) != len(actual_entries) or actual != expected:
        raise P6CoptV2XContractError("framework manifest identities do not match")


def _framework_materializable_registry_identities(
    source_registry: Mapping[str, Any],
) -> list[tuple[tuple[int, ...], str]]:
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        raise P6CoptV2XContractError("framework manifest identities are invalid")
    identities: list[tuple[tuple[int, ...], str]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise P6CoptV2XContractError("framework manifest identities are invalid")
        source_status = group.get("source_status")
        if source_status not in {"ready", "materializable", "unavailable"}:
            raise P6CoptV2XContractError(
                "framework registry source status is invalid"
            )
        if source_status == "unavailable":
            continue
        q_modes = group.get("available_q_modes")
        if not isinstance(q_modes, list):
            raise P6CoptV2XContractError("framework manifest identities are invalid")
        width = _framework_width_identity(group)
        identities.extend((width, str(q_mode)) for q_mode in q_modes)
    return identities


def _validate_closure(closure: object) -> None:
    if not isinstance(closure, Mapping):
        raise P6CoptV2XContractError("closure must be an object")
    source_rows = closure.get("training_source_rows")
    valid = (
        closure.get("schema_version") == "stage4_p1_p3_closure_audit_v1"
        and closure.get("stage4_closed") is True
        and closure.get("stage5_search_ready") is True
        and isinstance(source_rows, Mapping)
        and source_rows.get("initial_coldstart") == 176
        and closure.get("canonical_value_heads") == EXPECTED_CLOSURE_HEADS
        and closure.get("uncertainty_policy") == EXPECTED_UNCERTAINTY_POLICY
        and closure.get("selected_acquisition_policy") == EXPECTED_ACQUISITION_POLICY
    )
    if not valid:
        raise P6CoptV2XContractError("closure does not admit the P6 search")


def _run_search_round(
    *,
    contract: PublicP6CoptV2XContract,
    local: LocalP6CoptV2XConfig,
    task: SearchTask,
    capability_profiles: Sequence[Mapping[str, Any]],
    source_registry: Mapping[str, Any],
    frozen_gold: Sequence[Mapping[str, Any]],
    gold_graphs: Sequence[Mapping[str, Any]],
    gold_selection_rows: Sequence[Mapping[str, Any]],
    online_feedback_rows: Sequence[Mapping[str, Any]],
    measured_row_ids: set[str],
    frozen_holdout_group_ids: set[str],
    round_index: int,
    command_runner: CommandRunner,
) -> list[dict[str, Any]]:
    successful_feedback_rows = [
        row
        for row in online_feedback_rows
        if str(row.get("terminal_status")) == SUCCESS_STATUS
    ]
    training_rows = [*frozen_gold, *successful_feedback_rows]
    training_graphs = _unique_graph_features(
        [*gold_graphs, *[dict(row["graph_features"]) for row in successful_feedback_rows]]
    )
    bundle = (
        fit_initial_coldstart_bundle(
            frozen_gold, gold_graphs, capability_profiles, seed=contract.seed
        )
        if round_index == 0
        else fit_online_bundle(
            training_rows, training_graphs, capability_profiles, seed=contract.seed
        )
    )
    manifest = build_task_candidate_manifest(
        source_registry,
        task=task,
        measured_row_ids=measured_row_ids,
        frozen_holdout_group_ids=frozen_holdout_group_ids,
    )
    predicted = predict_candidate_rows(bundle, manifest["rows"], capability_profiles)
    selection = select_task_batch(
        predicted,
        [*gold_selection_rows, *online_feedback_rows],
        training_graphs,
        task=task,
    )
    request = build_measurement_request(
        task=task, selected_rows=selection["selected_rows"], round_index=round_index
    )
    round_root, request_path, feedback_path = _prepare_round_output(
        local.local_output_root,
        round_index,
        reject_existing=local.candidate_source_mode == "framework_stage2_search_space",
    )
    try:
        request = dict(project_source_materialization_request(request).request)
    except P6HistorySourceMaterializationError:
        raise P6CoptV2XExecutionError(
            "history_execution_invalid", "history execution invalid"
        ) from None
    _write_json(request_path, request)
    try:
        feedback_path.unlink(missing_ok=True)
    except OSError:
        raise P6CoptV2XExecutionError(
            "feedback_cleanup_failed", "could not prepare local feedback output"
        ) from None
    _run_step(
        local.measurement_step,
        {
            "{measurement_request}": request_path,
            "{feedback_json}": feedback_path,
            "{round_output_root}": round_root,
        },
        cwd=round_root,
        runner=command_runner,
    )
    _validate_local_output_leaf(feedback_path)
    return _release_feedback_rows(_read_feedback_json(feedback_path), request, task=task)


def _complete_run_state(
    local: LocalP6CoptV2XConfig,
    *,
    code_revision: str,
    completed_rounds: int,
    measured_candidate_count: int,
) -> P6CoptV2XRunState:
    state_path = local.local_output_root / "state.json"
    state = P6CoptV2XRunState(
        schema_version="p6_h800_coptv2x_local_state_v2",
        status="completed",
        completed_rounds=completed_rounds,
        measured_candidate_count=measured_candidate_count,
        failure_code=None,
        local_state_path=state_path,
    )
    _validate_local_output_leaf(state_path)
    _write_state(state, code_revision=code_revision)
    return state


def _failed_run_state(
    local: LocalP6CoptV2XConfig,
    *,
    code_revision: str,
    completed_rounds: int,
    measured_candidate_count: int,
    failure_code: str,
) -> P6CoptV2XRunState:
    state_path = local.local_output_root / "state.json"
    state = P6CoptV2XRunState(
        schema_version="p6_h800_coptv2x_local_state_v2",
        status="failed",
        completed_rounds=completed_rounds,
        measured_candidate_count=measured_candidate_count,
        failure_code=failure_code,
        local_state_path=state_path,
    )
    _validate_local_output_leaf(state_path)
    _write_state(state, code_revision=code_revision)
    return state


def _prepare_local_output_root(local_output_root: Path) -> None:
    if local_output_root.is_symlink() or (
        local_output_root.exists() and not local_output_root.is_dir()
    ):
        raise P6CoptV2XExecutionError("unsafe_output", "local output is unsafe")
    try:
        local_output_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise P6CoptV2XExecutionError(
            "output_setup_failed", "could not prepare local output"
        ) from None
    if local_output_root.is_symlink() or not local_output_root.is_dir():
        raise P6CoptV2XExecutionError("unsafe_output", "local output is unsafe")


def _prepare_round_output(
    local_output_root: Path, round_index: int, *, reject_existing: bool = True
) -> tuple[Path, Path, Path]:
    _prepare_local_output_root(local_output_root)
    round_root = local_output_root / f"round-{round_index:02d}"
    if round_root.is_symlink() or (
        reject_existing and round_root.exists()
    ) or (round_root.exists() and not round_root.is_dir()):
        raise P6CoptV2XExecutionError("unsafe_output", "round output is unsafe")
    try:
        round_root.mkdir(exist_ok=True)
    except OSError:
        raise P6CoptV2XExecutionError(
            "output_setup_failed", "could not prepare round output"
        ) from None
    request_path = round_root / "measurement_request.json"
    feedback_path = round_root / "feedback.json"
    for output_path in (request_path, feedback_path):
        _validate_local_output_leaf(output_path)
    return round_root, request_path, feedback_path


def _write_round_failure(
    local_output_root: Path,
    *,
    round_index: int,
    failure_code: str,
    completed_rounds: int,
) -> None:
    round_root = local_output_root / f"round-{round_index:02d}"
    if round_root.is_symlink() or not round_root.is_dir():
        raise P6CoptV2XExecutionError("unsafe_output", "round output is unsafe")
    failure_path = round_root / "failure.json"
    _validate_local_output_leaf(failure_path)
    _write_json(
        failure_path,
        {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "failure_code": failure_code,
            "completed_rounds": completed_rounds,
        },
    )


def _validate_local_output_leaf(output_path: Path) -> None:
    if output_path.is_symlink() or (output_path.exists() and not output_path.is_file()):
        raise P6CoptV2XExecutionError("unsafe_output", "local output leaf is unsafe")


def _require_local_output_absent(output_path: Path) -> None:
    """Refuse stale controller leaves instead of deleting or reusing them."""
    try:
        output_path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        pass
    raise P6CoptV2XExecutionError("unsafe_output", "local output leaf is unsafe")


def _require_controller_destinations_absent(local: LocalP6CoptV2XConfig) -> None:
    """Check deterministic public controller outputs before process execution."""
    paths = [
        local.local_output_root / "source_registry.json",
        local.local_output_root / "pyramid_candidate_plan.json",
        local.local_output_root / "state.json",
        *(local.local_output_root / f"round-{index:02d}" for index in range(FIXED_ROUND_COUNT)),
    ]
    if local.candidate_source_mode == "framework_stage2_search_space":
        assert local.stage2_search_space_path is not None
        paths.append(local.stage2_search_space_path)
    for path in paths:
        _require_local_output_absent(path)


def _build_search_task(
    contract: PublicP6CoptV2XContract, profile: Mapping[str, Any]
) -> SearchTask:
    return SearchTask(
        task_id=contract.search_id,
        target_model=contract.target_model,
        hardware_id=contract.hardware_profile.target_hardware_id,
        capability_profile=profile,
        sample_budget=contract.sample_budget,
        batch_size=contract.batch_size,
        round_count=contract.round_count,
    )


def _select_profile(
    profiles: Sequence[Mapping[str, Any]],
    contract: PublicP6CoptV2XContract | None = None,
) -> dict[str, Any]:
    try:
        validated = [validate_capability_profile(profile) for profile in profiles]
    except ValueError as exc:
        raise P6CoptV2XContractError("capability profile invalid") from exc
    hardware_profile = (
        contract.hardware_profile
        if contract is not None
        else default_hardware_execution_profile()
    )
    hardware_target = hardware_profile.target_hardware_id
    execution_backend = (
        contract.execution_backend if contract is not None else FIXED_BACKEND
    )
    if any(
        str(profile["hardware_target"]).lower() != hardware_target
        for profile in validated
    ):
        raise P6CoptV2XContractError(
            "capability profiles must match the hardware profile"
        )
    matching = [
        profile
        for profile in validated
        if str(profile["hardware_target"]).lower() == hardware_target
        and str(profile["dispatch_key"]) == execution_backend
    ]
    if len(matching) != 1:
        raise P6CoptV2XContractError(
            "exactly one profile-compatible capability profile is required"
        )
    return matching[0]


def _attach_graph_features(
    rows: Sequence[Mapping[str, Any]], graph_features: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    graph_by_group = {str(graph.get("group_id")): dict(graph) for graph in graph_features}
    if len(graph_by_group) != len(graph_features):
        raise P6CoptV2XContractError("gold176 graph feature identities must be unique")
    if any(str(row.get("group_id")) not in graph_by_group for row in rows):
        raise P6CoptV2XContractError("gold176 graph features are incomplete")
    return [
        {**dict(row), "graph_features": graph_by_group[str(row["group_id"])]}
        for row in rows
    ]


def _unique_graph_features(
    graph_features: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for raw_graph in graph_features:
        graph = dict(raw_graph)
        group_id = str(graph.get("group_id") or "")
        if not group_id:
            raise P6CoptV2XContractError("graph feature group identity is missing")
        if group_id in by_group and by_group[group_id] != graph:
            raise P6CoptV2XContractError("graph feature identity drift")
        by_group[group_id] = graph
    return list(by_group.values())


def _release_feedback_rows(
    feedback: object,
    request: Mapping[str, Any],
    *,
    task: SearchTask,
) -> list[dict[str, Any]]:
    if not isinstance(feedback, Mapping) or feedback.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
        raise P6CoptV2XExecutionError("feedback_invalid_json", "feedback schema invalid")
    if feedback.get("measurement_request_sha256") != request.get(
        "measurement_request_sha256"
    ):
        raise P6CoptV2XExecutionError("feedback_request_mismatch", "feedback request mismatch")
    request_rows = request.get("rows")
    feedback_rows = feedback.get("rows")
    if (
        not isinstance(request_rows, list)
        or not isinstance(feedback_rows, list)
        or len(request_rows) != task.batch_size
        or len(feedback_rows) != len(request_rows)
        or not all(isinstance(row, Mapping) for row in [*request_rows, *feedback_rows])
    ):
        raise P6CoptV2XExecutionError(
            "feedback_candidate_mismatch", "feedback candidate mismatch"
        )
    request_ids = [str(row.get("row_id") or "") for row in request_rows]
    feedback_ids = [str(row.get("row_id") or "") for row in feedback_rows]
    if (
        any(not row_id for row_id in [*request_ids, *feedback_ids])
        or len(set(request_ids)) != len(request_ids)
        or len(set(feedback_ids)) != len(feedback_ids)
        or set(request_ids) != set(feedback_ids)
    ):
        raise P6CoptV2XExecutionError(
            "feedback_candidate_mismatch", "feedback candidate mismatch"
        )
    by_request = dict(zip(request_ids, request_rows))
    by_feedback = dict(zip(feedback_ids, feedback_rows))
    released: list[dict[str, Any]] = []
    for row_id, request_row in by_request.items():
        row = dict(by_feedback[row_id])
        status = str(row.get("terminal_status") or "")
        base = {
            **copy.deepcopy(dict(request_row)),
            "training_source": "online_feedback",
            "terminal_status": status,
        }
        if status == SUCCESS_STATUS:
            metrics = {metric: row.get(metric) for metric in METRIC_NAMES}
            if (
                any(not _finite(metric) for metric in metrics.values())
                or any(float(metrics[metric]) <= 0.0 for metric in ("latency_ms", "energy_j"))
                or any(
                    not 0.0 <= float(metrics[metric]) <= 1.0
                    for metric in ("ap30", "ap50", "ap70")
                )
            ):
                raise P6CoptV2XExecutionError(
                    "feedback_metrics_invalid", "feedback metrics invalid"
                )
            released.append({**base, **{metric: float(metrics[metric]) for metric in METRIC_NAMES}})
        elif status in TRUE_FAILURE_STATUSES:
            reason = str(row.get("failure_reason") or "")
            if not reason:
                raise P6CoptV2XExecutionError(
                    "feedback_terminal_status_invalid", "failure reason missing"
                )
            released.append({**base, "failure_reason": _redacted_reason(reason)})
        else:
            raise P6CoptV2XExecutionError(
                "feedback_terminal_status_invalid", "feedback status invalid"
            )
    return released


def _finite(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _redacted_reason(reason: str) -> str:
    return reason if _PUBLIC_FAILURE_REASON_PATTERN.fullmatch(reason) else "unspecified"


def _read_feedback_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise P6CoptV2XExecutionError("feedback_missing", "feedback missing") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise P6CoptV2XExecutionError("feedback_invalid_json", "feedback invalid") from None


def _read_source_registry_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise P6CoptV2XExecutionError(
            "source_registry_missing", "source registry missing"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise P6CoptV2XExecutionError(
            "source_registry_invalid", "source registry invalid"
        ) from None
    if not isinstance(payload, Mapping):
        raise P6CoptV2XExecutionError(
            "source_registry_invalid", "source registry invalid"
        )
    return payload


def _read_local_input_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise P6CoptV2XExecutionError("local_input_invalid", "local input invalid") from None


def _require_mapping_rows(value: object, description: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise P6CoptV2XContractError(f"{description} must be a list of objects")
    return [dict(row) for row in value]


def _replace_exact_token(arg: str, replacements: Mapping[str, Path]) -> str:
    replacement = replacements.get(arg)
    return str(replacement) if replacement is not None else arg


def _run_step(
    step: LocalExecutionStep,
    replacements: Mapping[str, Path],
    *,
    cwd: Path,
    runner: CommandRunner,
) -> None:
    argv = tuple(_replace_exact_token(arg, replacements) for arg in step.argv)
    runner_failed = False
    try:
        return_code = runner(argv, cwd)
    except Exception:
        runner_failed = True
        return_code = None
    if runner_failed:
        raise P6CoptV2XExecutionError("command_failed", "local command failed")
    if return_code != 0:
        raise P6CoptV2XExecutionError("command_failed", "local command failed")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_state(state: P6CoptV2XRunState, *, code_revision: str) -> None:
    _write_json(
        state.local_state_path,
        {
            "schema_version": state.schema_version,
            "status": state.status,
            "code_revision": code_revision,
            "completed_rounds": state.completed_rounds,
            "measured_candidate_count": state.measured_candidate_count,
            "failure_code": state.failure_code,
        },
    )


def load_public_contract(path: Path) -> PublicP6CoptV2XContract:
    """Load the fixed, non-executable public P6 CoptV2X search contract."""
    payload = _load_mapping(path, "public contract")
    _reject_public_execution_details(payload)
    schema_version = payload.get("schema_version")
    if schema_version == PUBLIC_SCHEMA_VERSION:
        _require_exact_keys(payload, PUBLIC_KEYS, "public contract")
        hardware_profile = default_hardware_execution_profile()
    elif schema_version == PUBLIC_SCHEMA_VERSION_V3:
        _require_exact_keys(payload, PUBLIC_V3_KEYS, "public contract")
        try:
            hardware_profile = load_hardware_execution_profile(
                payload["hardware_profile"]
            )
        except ValueError:
            raise P6CoptV2XContractError("hardware profile is invalid") from None
    else:
        raise P6CoptV2XContractError("public schema_version is invalid")
    target = _require_public_identifier(payload["target"], "target")
    if target != hardware_profile.target_hardware_id:
        if schema_version == PUBLIC_SCHEMA_VERSION:
            raise P6CoptV2XContractError("target must be h800")
        raise P6CoptV2XContractError(
            "target does not match the hardware profile"
        )
    if _require_public_identifier(payload["target_model"], "target_model") != FIXED_MODEL:
        raise P6CoptV2XContractError("target_model must be pyramid")
    execution_backend = _require_public_identifier(
        payload["execution_backend"], "execution_backend"
    )
    try:
        validate_profile_backend(hardware_profile, execution_backend)
    except ValueError:
        if schema_version == PUBLIC_SCHEMA_VERSION:
            raise P6CoptV2XContractError(
                "execution_backend must be tvm_auto"
            ) from None
        raise P6CoptV2XContractError(
            "execution_backend does not match the hardware profile"
        ) from None
    if _require_positive_integer(payload["sample_budget"], "sample_budget") != FIXED_SAMPLE_BUDGET:
        raise P6CoptV2XContractError("sample_budget must be 16")
    if _require_positive_integer(payload["batch_size"], "batch_size") != FIXED_BATCH_SIZE:
        raise P6CoptV2XContractError("batch_size must be 4")
    if _require_positive_integer(payload["round_count"], "round_count") != FIXED_ROUND_COUNT:
        raise P6CoptV2XContractError("round_count must be 4")
    metric_names = _parse_string_list(payload["metric_names"], "metric_names")
    if metric_names != METRIC_NAMES:
        raise P6CoptV2XContractError("metric_names must exactly match runtime metrics")
    return PublicP6CoptV2XContract(
        search_id=_require_public_identifier(payload["search_id"], "search_id"),
        target=target,
        target_model=FIXED_MODEL,
        execution_backend=execution_backend,
        seed=_require_positive_integer(payload["seed"], "seed"),
        sample_budget=FIXED_SAMPLE_BUDGET,
        batch_size=FIXED_BATCH_SIZE,
        round_count=FIXED_ROUND_COUNT,
        configuration_label=_require_public_identifier(
            payload["configuration_label"], "configuration_label"
        ),
        candidate_space_label=_require_public_identifier(
            payload["candidate_space_label"], "candidate_space_label"
        ),
        assets=_parse_assets(payload["assets"]),
        metric_names=metric_names,
        hardware_profile=hardware_profile,
    )


def load_local_config(path: Path, contract: PublicP6CoptV2XContract) -> LocalP6CoptV2XConfig:
    """Load private, argv-only execution details without publishing them."""
    if not isinstance(contract, PublicP6CoptV2XContract):
        raise P6CoptV2XContractError("public contract is invalid")
    payload = _load_mapping(path, "local config")
    schema_version = payload.get("schema_version")
    if schema_version == LOCAL_SCHEMA_VERSION:
        allowed_keys = LOCAL_KEYS
        required_keys = LOCAL_KEYS - _OPTIONAL_LEGACY_LOCAL_KEYS
    elif schema_version == LOCAL_SCHEMA_VERSION_V3:
        allowed_keys = LOCAL_V3_KEYS
        required_keys = LOCAL_V3_KEYS - _OPTIONAL_LEGACY_LOCAL_KEYS
    else:
        raise P6CoptV2XContractError("local schema_version is invalid")
    unknown_keys = set(payload) - allowed_keys
    if unknown_keys:
        raise P6CoptV2XContractError(
            f"local config contains unknown keys: {sorted(unknown_keys)}"
        )
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise P6CoptV2XContractError(
            f"local config is missing required keys: {sorted(missing_keys)}"
        )
    if schema_version == LOCAL_SCHEMA_VERSION:
        hardware_profile = default_hardware_execution_profile()
    else:
        try:
            hardware_profile = load_hardware_execution_profile(
                payload["hardware_profile"]
            )
        except ValueError:
            raise P6CoptV2XContractError("hardware profile is invalid") from None
    if hardware_profile.profile_id != contract.hardware_profile.profile_id:
        raise P6CoptV2XContractError(
            "local hardware profile must match the public contract"
        )
    if _require_nonempty_string(payload["target"], "local target") != contract.target:
        raise P6CoptV2XContractError("local target must match the public contract")
    asset_paths = _parse_path_mapping(payload["asset_paths"], "asset_paths")
    if set(asset_paths) != {asset.label for asset in contract.assets}:
        raise P6CoptV2XContractError("local asset labels must exactly match public asset labels")
    local_input_paths = _parse_path_mapping(payload["local_input_paths"], "local_input_paths")
    if set(local_input_paths) != LOCAL_INPUT_NAMES:
        raise P6CoptV2XContractError("local input labels are invalid")
    candidate_source_mode = _require_nonempty_string(
        payload.get("candidate_source_mode", "coptv2x_static_registry"),
        "candidate_source_mode",
    )
    if candidate_source_mode not in {
        "coptv2x_static_registry",
        "framework_stage2_search_space",
    }:
        raise P6CoptV2XContractError("candidate_source_mode is invalid")
    stage2_search_space_path: Path | None
    stage1_scan_step: LocalExecutionStep | None
    if candidate_source_mode == "framework_stage2_search_space":
        stage2_search_space_path = _parse_absolute_path(
            payload.get("stage2_search_space_path"), "stage2_search_space_path"
        )
        if "stage1_scan_step" not in payload:
            raise P6CoptV2XContractError(
                "stage1_scan_step is required in framework mode"
            )
        stage1_scan_step = _load_step(
            payload["stage1_scan_step"],
            expected_name="build_stage1_partition",
            required_tokens={"{stage1_partition_manifest}", "{local_output_root}"},
        )
        source_required_tokens = {
            "{local_output_root}",
            "{source_registry_json}",
            "{pyramid_candidate_plan}",
        }
    else:
        if payload.get("stage2_search_space_path") is not None:
            raise P6CoptV2XContractError(
                "stage2_search_space_path is only valid in framework mode"
            )
        if payload.get("stage1_scan_step") is not None:
            raise P6CoptV2XContractError(
                "stage1_scan_step is only valid in framework mode"
            )
        stage2_search_space_path = None
        stage1_scan_step = None
        source_required_tokens = {"{local_output_root}", "{source_registry_json}"}
    return LocalP6CoptV2XConfig(
        asset_paths=MappingProxyType(asset_paths),
        local_input_paths=MappingProxyType(local_input_paths),
        candidate_source_mode=candidate_source_mode,
        stage2_search_space_path=stage2_search_space_path,
        stage1_scan_step=stage1_scan_step,
        source_registry_step=_load_step(
            payload["source_registry_step"],
            expected_name="build_source_registry",
            required_tokens=source_required_tokens,
        ),
        measurement_step=_load_step(
            payload["measurement_step"],
            expected_name="measure_batch",
            required_tokens={"{measurement_request}", "{feedback_json}", "{round_output_root}"},
        ),
        local_output_root=_parse_absolute_path(payload["local_output_root"], "local_output_root"),
        hardware_profile=hardware_profile,
    )


def _load_step(
    payload: object, *, expected_name: str, required_tokens: set[str]
) -> LocalExecutionStep:
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise P6CoptV2XContractError("step keys invalid")
    if set(payload) != _STEP_KEYS:
        raise P6CoptV2XContractError("step keys invalid")
    if payload["name"] != expected_name:
        raise P6CoptV2XContractError("step name invalid")
    argv = _validate_argv(payload["argv"])
    present = {token for arg in argv for token in ALLOWED_TEMPLATE_TOKENS if token in arg}
    if present != required_tokens:
        raise P6CoptV2XContractError("template tokens invalid")
    return LocalExecutionStep(name=expected_name, argv=argv)


def _validate_argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise P6CoptV2XContractError("step argv must be a non-empty list")
    argv = tuple(_require_nonempty_string(token, "argv token") for token in value)
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if executable in _SHELL_EXECUTABLES:
        raise P6CoptV2XContractError("step argv must not invoke a shell executable")
    if executable in _COMMAND_WRAPPER_EXECUTABLES:
        raise P6CoptV2XContractError("step argv must not invoke a command wrapper")
    for token in argv:
        if ("{" in token or "}" in token) and token not in ALLOWED_TEMPLATE_TOKENS:
            raise P6CoptV2XContractError("argv template contains an unknown or embedded placeholder")
    return argv


def _load_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P6CoptV2XContractError(f"could not load {description}") from error
    if not isinstance(loaded, Mapping):
        raise P6CoptV2XContractError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise P6CoptV2XContractError(f"{description} keys must be strings")
    return loaded


def _reject_public_execution_details(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _is_forbidden_public_key(key):
                raise P6CoptV2XContractError("public contract contains forbidden execution detail")
            _reject_public_execution_details(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_public_execution_details(nested_value)


def _is_forbidden_public_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("-", "_")
    return (
        "path" in normalized
        or "command" in normalized
        or "cmd" in normalized
        or "host" in normalized
        or normalized in {"argv", "candidate_id"}
        or "hash" in normalized
        or normalized.startswith("sha")
        or normalized in {"raw_log", "raw_logs", "checkpoint", "measurements"}
    )


def _require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], description: str) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = allowed - keys
    if unknown:
        raise P6CoptV2XContractError(f"{description} contains unknown keys: {sorted(unknown)}")
    if missing:
        raise P6CoptV2XContractError(f"{description} is missing required keys: {sorted(missing)}")


def _parse_assets(value: object) -> tuple[RegisteredAsset, ...]:
    if not isinstance(value, list) or not value:
        raise P6CoptV2XContractError("assets must be a non-empty list")
    assets: list[RegisteredAsset] = []
    labels: set[str] = set()
    for raw_asset in value:
        if not isinstance(raw_asset, Mapping) or not all(isinstance(key, str) for key in raw_asset):
            raise P6CoptV2XContractError("each asset must be a mapping")
        _require_exact_keys(raw_asset, _ASSET_KEYS, "asset")
        asset = RegisteredAsset(
            label=_require_public_identifier(raw_asset["label"], "asset label"),
            version=_require_public_identifier(raw_asset["version"], "asset version"),
            license_status=_require_public_identifier(raw_asset["license_status"], "asset license_status"),
        )
        if asset.label in labels:
            raise P6CoptV2XContractError("asset labels must be unique")
        labels.add(asset.label)
        assets.append(asset)
    return tuple(assets)


def _parse_string_list(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise P6CoptV2XContractError(f"{description} must be a non-empty list")
    parsed = tuple(_require_public_identifier(item, description) for item in value)
    if len(set(parsed)) != len(parsed):
        raise P6CoptV2XContractError(f"{description} must contain unique values")
    return parsed


def _parse_path_mapping(value: object, description: str) -> dict[str, Path]:
    if not isinstance(value, Mapping) or not value:
        raise P6CoptV2XContractError(f"{description} must be a non-empty mapping")
    return {
        _require_nonempty_string(key, f"{description} label"): _parse_absolute_path(
            raw_path, f"{description}[{key}]"
        )
        for key, raw_path in value.items()
    }


def _parse_absolute_path(value: object, description: str) -> Path:
    path = Path(_require_nonempty_string(value, description))
    if not path.is_absolute():
        raise P6CoptV2XContractError(f"{description} must be an absolute path")
    return path


def _require_nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P6CoptV2XContractError(f"{description} must be a non-empty string")
    return value


def _require_public_identifier(value: object, description: str) -> str:
    identifier = _require_nonempty_string(value, description)
    if "/" in identifier or "\\" in identifier or any(character.isspace() for character in identifier):
        raise P6CoptV2XContractError(f"{description} contains restricted public information")
    if _RAW_DIGEST_PATTERN.search(identifier) or any(
        pattern.search(identifier) for pattern in _PUBLIC_RESTRICTED_VALUE_PATTERNS
    ):
        raise P6CoptV2XContractError(f"{description} contains restricted public information")
    return identifier


def _require_positive_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise P6CoptV2XContractError(f"{description} must be a positive integer")
    return value
