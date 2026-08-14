"""Contracts and local state machine for the P6 CoptV2X H800/TVM search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal

import yaml

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
    validate_task_feedback_history,
)


PUBLIC_SCHEMA_VERSION = "p6_h800_coptv2x_search_contract_v2"
LOCAL_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
METRIC_NAMES = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
FIXED_TARGET = "h800"
FIXED_MODEL = "pyramid"
FIXED_BACKEND = "tvm_auto"
FIXED_SAMPLE_BUDGET = 16
FIXED_BATCH_SIZE = 4
FIXED_ROUND_COUNT = 4
FIXED_SOURCE_GROUP_COUNT = 343
FIXED_ELIGIBLE_GENOME_COUNT = 686
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
        "{source_registry_json}",
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
LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "local_input_paths",
        "source_registry_step",
        "measurement_step",
        "local_output_root",
    }
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


class P6CoptV2XContractError(ValueError):
    """Raised when a P6 public or local CoptV2X contract is invalid."""


class P6CoptV2XExecutionError(RuntimeError):
    """Raised when an argv-only local P6 execution step fails."""

    def __init__(self, failure_code: str, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


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


@dataclass(frozen=True)
class LocalExecutionStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class LocalP6CoptV2XConfig:
    asset_paths: Mapping[str, Path]
    local_input_paths: Mapping[str, Path]
    source_registry_step: LocalExecutionStep
    measurement_step: LocalExecutionStep
    local_output_root: Path


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
    """Run the fixed four-round local Pyramid/H800/TVM search state machine."""
    local.local_output_root.mkdir(parents=True, exist_ok=True)
    frozen_gold, gold_graphs, profile = _load_search_inputs(local)
    task = _build_search_task(contract, profile)
    source_registry = _build_source_registry(local, command_runner)
    _validate_p6_source_space(source_registry, task)
    online_feedback_rows: list[dict[str, Any]] = []
    measured_row_ids: set[str] = set()
    frozen_holdout_group_ids = {str(row["group_id"]) for row in frozen_gold}
    gold_selection_rows = _attach_graph_features(frozen_gold, gold_graphs)
    for round_index in range(contract.round_count):
        released_rows = _run_search_round(
            contract=contract,
            local=local,
            task=task,
            profile=profile,
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


def _load_search_inputs(
    local: LocalP6CoptV2XConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gold_rows = _require_mapping_rows(
        _read_json_object_or_list(local.local_input_paths["gold176_rows"]),
        "gold176 rows",
    )
    gold_graphs = _require_mapping_rows(
        _read_json_object_or_list(local.local_input_paths["gold176_graph_features"]),
        "gold176 graph features",
    )
    raw_profiles = _require_mapping_rows(
        _read_json_object_or_list(local.local_input_paths["capability_profiles"]),
        "capability profiles",
    )
    closure = _read_json_object_or_list(local.local_input_paths["closure"])
    _validate_closure(closure)
    frozen_gold = freeze_initial_coldstart(gold_rows)
    profile = _select_profile(raw_profiles)
    return frozen_gold, gold_graphs, profile


def _build_source_registry(
    local: LocalP6CoptV2XConfig, command_runner: CommandRunner
) -> Mapping[str, Any]:
    source_registry_path = local.local_output_root / "source_registry.json"
    _run_step(
        local.source_registry_step,
        {
            "{local_output_root}": local.local_output_root,
            "{source_registry_json}": source_registry_path,
        },
        cwd=local.local_output_root,
        runner=command_runner,
    )
    source_registry = _read_json_object_or_list(source_registry_path)
    if not isinstance(source_registry, Mapping):
        raise P6CoptV2XContractError("source registry must be an object")
    return source_registry


def _validate_p6_source_space(
    source_registry: Mapping[str, Any], task: SearchTask
) -> None:
    groups = source_registry.get("groups")
    if not isinstance(groups, list) or len(groups) != FIXED_SOURCE_GROUP_COUNT:
        raise P6CoptV2XContractError("P6 source space must contain exactly 343 groups")
    manifest = build_task_candidate_manifest(
        source_registry, task=task, measured_row_ids=set()
    )
    if (
        manifest.get("eligible_row_count") != FIXED_ELIGIBLE_GENOME_COUNT
        or len(manifest.get("rows") or []) != FIXED_ELIGIBLE_GENOME_COUNT
    ):
        raise P6CoptV2XContractError("P6 source space must contain exactly 686 genomes")


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
    profile: Mapping[str, Any],
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
    training_rows = [*frozen_gold, *online_feedback_rows]
    training_graphs = _unique_graph_features(
        [*gold_graphs, *[dict(row["graph_features"]) for row in online_feedback_rows]]
    )
    bundle = (
        fit_initial_coldstart_bundle(frozen_gold, gold_graphs, [profile], seed=contract.seed)
        if round_index == 0
        else fit_online_bundle(training_rows, training_graphs, [profile], seed=contract.seed)
    )
    manifest = build_task_candidate_manifest(
        source_registry,
        task=task,
        measured_row_ids=measured_row_ids,
        frozen_holdout_group_ids=frozen_holdout_group_ids,
    )
    predicted = predict_candidate_rows(bundle, manifest["rows"], [profile])
    selection = select_task_batch(
        predicted,
        [*gold_selection_rows, *online_feedback_rows],
        training_graphs,
        task=task,
    )
    request = build_measurement_request(
        task=task, selected_rows=selection["selected_rows"], round_index=round_index
    )
    round_root = local.local_output_root / f"round-{round_index:02d}"
    round_root.mkdir(parents=True, exist_ok=True)
    request_path = round_root / "measurement_request.json"
    feedback_path = round_root / "feedback.json"
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
    return _release_feedback_rows(_read_json_object_or_list(feedback_path), request)


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
    _write_state(state, code_revision=code_revision)
    return state


def _build_search_task(
    contract: PublicP6CoptV2XContract, profile: Mapping[str, Any]
) -> SearchTask:
    return SearchTask(
        task_id=contract.search_id,
        target_model=contract.target_model,
        hardware_id=contract.target,
        capability_profile=profile,
        sample_budget=contract.sample_budget,
        batch_size=contract.batch_size,
        round_count=contract.round_count,
    )


def _select_profile(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        validated = [validate_capability_profile(profile) for profile in profiles]
    except ValueError as exc:
        raise P6CoptV2XContractError("capability profile invalid") from exc
    matching = [
        profile
        for profile in validated
        if str(profile["hardware_target"]).lower() == FIXED_TARGET
        and str(profile["dispatch_key"]) == FIXED_BACKEND
    ]
    if len(matching) != 1:
        raise P6CoptV2XContractError("exactly one H800 TVM capability profile is required")
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
    feedback: object, request: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(feedback, Mapping):
        raise P6CoptV2XContractError("feedback must be an object")
    if feedback.get("schema_version") != "p6_h800_coptv2x_feedback_v2":
        raise P6CoptV2XContractError("feedback schema is invalid")
    if feedback.get("measurement_request_sha256") != request.get(
        "measurement_request_sha256"
    ):
        raise P6CoptV2XContractError("feedback request identity mismatch")
    request_rows = request.get("rows")
    feedback_rows = feedback.get("rows")
    if not isinstance(request_rows, list) or not isinstance(feedback_rows, list):
        raise P6CoptV2XContractError("feedback rows are invalid")
    if (
        len(request_rows) != FIXED_BATCH_SIZE
        or len(feedback_rows) != len(request_rows)
        or not all(isinstance(row, Mapping) for row in [*request_rows, *feedback_rows])
    ):
        raise P6CoptV2XContractError("feedback must contain the requested four rows")
    request_ids = [str(row.get("row_id") or "") for row in request_rows]
    feedback_ids = [str(row.get("row_id") or "") for row in feedback_rows]
    if (
        any(not row_id for row_id in [*request_ids, *feedback_ids])
        or len(set(request_ids)) != len(request_ids)
        or len(set(feedback_ids)) != len(feedback_ids)
        or set(request_ids) != set(feedback_ids)
    ):
        raise P6CoptV2XContractError("feedback must contain the requested four rows")
    request_by_id = dict(zip(request_ids, request_rows))
    feedback_by_id = dict(zip(feedback_ids, feedback_rows))
    if (
        len(request_by_id) != 4
        or len(feedback_by_id) != 4
        or set(request_by_id) != set(feedback_by_id)
    ):
        raise P6CoptV2XContractError("feedback must contain the requested four rows")
    released: list[dict[str, Any]] = []
    for row_id, request_row in request_by_id.items():
        feedback_row = feedback_by_id[row_id]
        if feedback_row.get("terminal_status") != "measured_success_gold":
            raise P6CoptV2XContractError("Task 2 feedback must be measured success")
        metrics = {name: feedback_row.get(name) for name in METRIC_NAMES}
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
            or any(float(metrics[name]) <= 0.0 for name in ("latency_ms", "energy_j"))
            or any(
                not 0.0 <= float(metrics[name]) <= 1.0
                for name in ("ap30", "ap50", "ap70")
            )
        ):
            raise P6CoptV2XContractError("feedback metrics are outside physical bounds")
        released.append(
            {
                **dict(request_row),
                **metrics,
                "terminal_status": "measured_success_gold",
                "training_source": "online_feedback",
            }
        )
    return released


def _read_json_object_or_list(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P6CoptV2XContractError("local input invalid") from exc


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
    try:
        return_code = runner(argv, cwd)
    except Exception:
        raise P6CoptV2XExecutionError("command_failed", "local command failed") from None
    if return_code != 0:
        raise P6CoptV2XExecutionError("command_failed", "local command failed")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


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
    _require_exact_keys(payload, PUBLIC_KEYS, "public contract")
    if payload["schema_version"] != PUBLIC_SCHEMA_VERSION:
        raise P6CoptV2XContractError("public schema_version is invalid")
    if _require_public_identifier(payload["target"], "target") != FIXED_TARGET:
        raise P6CoptV2XContractError("target must be h800")
    if _require_public_identifier(payload["target_model"], "target_model") != FIXED_MODEL:
        raise P6CoptV2XContractError("target_model must be pyramid")
    if _require_public_identifier(payload["execution_backend"], "execution_backend") != FIXED_BACKEND:
        raise P6CoptV2XContractError("execution_backend must be tvm_auto")
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
        target=FIXED_TARGET,
        target_model=FIXED_MODEL,
        execution_backend=FIXED_BACKEND,
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
    )


def load_local_config(path: Path, contract: PublicP6CoptV2XContract) -> LocalP6CoptV2XConfig:
    """Load private, argv-only execution details without publishing them."""
    if not isinstance(contract, PublicP6CoptV2XContract):
        raise P6CoptV2XContractError("public contract is invalid")
    payload = _load_mapping(path, "local config")
    _require_exact_keys(payload, LOCAL_KEYS, "local config")
    if payload["schema_version"] != LOCAL_SCHEMA_VERSION:
        raise P6CoptV2XContractError("local schema_version is invalid")
    if _require_nonempty_string(payload["target"], "local target") != contract.target:
        raise P6CoptV2XContractError("local target must match the public contract")
    asset_paths = _parse_path_mapping(payload["asset_paths"], "asset_paths")
    if set(asset_paths) != {asset.label for asset in contract.assets}:
        raise P6CoptV2XContractError("local asset labels must exactly match public asset labels")
    local_input_paths = _parse_path_mapping(payload["local_input_paths"], "local_input_paths")
    if set(local_input_paths) != LOCAL_INPUT_NAMES:
        raise P6CoptV2XContractError("local input labels are invalid")
    return LocalP6CoptV2XConfig(
        asset_paths=MappingProxyType(asset_paths),
        local_input_paths=MappingProxyType(local_input_paths),
        source_registry_step=_load_step(
            payload["source_registry_step"],
            expected_name="build_source_registry",
            required_tokens={"{local_output_root}", "{source_registry_json}"},
        ),
        measurement_step=_load_step(
            payload["measurement_step"],
            expected_name="measure_batch",
            required_tokens={"{measurement_request}", "{feedback_json}", "{round_output_root}"},
        ),
        local_output_root=_parse_absolute_path(payload["local_output_root"], "local_output_root"),
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
