"""Strict, side-effect-free parsing for P6 H800 search execution contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Literal, Protocol

import yaml

from framework.stage5.production_search_v1 import (
    build_candidate_manifest,
    fit_production_bundle,
    predict_candidate_rows,
    select_predicted_frontier_diversity,
)


PUBLIC_SCHEMA_VERSION = "p6_h800_search_contract_v1"
LOCAL_SCHEMA_VERSION = "p6_h800_search_local_v1"
SUMMARY_SCHEMA_VERSION = "p6_h800_search_summary_v1"
_STAGE5_INPUT_NAMES = frozenset(
    {"measurements", "candidate_registry", "graph_features", "capability_profiles", "closure"}
)
_MEASUREMENT_METRICS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
_EXPECTED_ARMS = frozenset(
    {
        ("tvm_auto", "fp16"),
        ("tvm_auto", "int8"),
        ("trt_engine", "fp16"),
        ("trt_engine", "int8"),
    }
)
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
_PUBLIC_KEYS = frozenset(
    {
        "schema_version",
        "search_id",
        "target",
        "target_model",
        "seed",
        "max_rounds",
        "batch_size",
        "configuration_label",
        "assets",
        "metric_names",
        "candidate_space_label",
    }
)
_LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "stage5_input_paths",
        "steps",
        "result_step",
        "result_path_template",
        "local_output_root",
    }
)
_ASSET_KEYS = frozenset({"label", "version", "license_status"})
_STEP_KEYS = frozenset({"name", "argv"})
_ALLOWED_TEMPLATE_TOKENS = frozenset(
    {"{candidate_request}", "{result_json}", "{local_output_root}"}
)
_SUMMARY_METRIC_KEYS = frozenset({"count", "min", "max", "mean"})
_SUMMARY_FAILURE_CODES = frozenset(
    {
        "capability_target_invalid",
        "command_failed",
        "local_input_invalid",
        "local_record_write_failed",
        "result_candidate_mismatch",
        "result_invalid_json",
        "result_metrics_invalid",
        "result_missing",
        "stage5_selection_failed",
    }
)
_PUBLIC_RESTRICTED_VALUE_PATTERNS = (
    re.compile(r"(?:^|[-_\s])host(?:name)?(?:[-_\s]|\d|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])raw[-_\s]?logs?(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])checkpoint(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(
        r"(?:^|[-_\s])candidate(?:[-_\s]+id)?[-_\s]+[a-z0-9]+(?:[-_\s]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[-_\s])(?:sha(?:1|224|256|384|512)?|hash)(?:[:=_-]|$)", re.IGNORECASE),
    re.compile(r"^(?:python(?:\d+(?:\.\d+)?)?|bash|sh|zsh|cmd(?:\.exe)?|powershell|pwsh)\b", re.IGNORECASE),
)
_RAW_DIGEST_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}|[0-9a-f]{128}", re.IGNORECASE)


class H800SearchContractError(ValueError):
    """Raised when a P6 public or local search contract is unsafe or incomplete."""


class H800SearchExecutionError(RuntimeError):
    """Controlled local H800 execution failure with a stable public code."""

    def __init__(self, failure_code: str, detail: str) -> None:
        super().__init__(detail)
        self.failure_code = failure_code


class CommandRunner(Protocol):
    """Narrow injection boundary for one argv-only local execution step."""

    def __call__(self, argv: tuple[str, ...], cwd: Path) -> int: ...


@dataclass(frozen=True)
class RegisteredAsset:
    label: str
    version: str
    license_status: str


@dataclass(frozen=True)
class PublicH800SearchContract:
    search_id: str
    target: str
    target_model: str
    seed: int
    max_rounds: int
    batch_size: int
    configuration_label: str
    assets: tuple[RegisteredAsset, ...]
    metric_names: tuple[str, ...]
    candidate_space_label: str


@dataclass(frozen=True)
class LocalExecutionStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class LocalH800SearchConfig:
    asset_paths: Mapping[str, Path]
    stage5_input_paths: Mapping[str, Path]
    steps: tuple[LocalExecutionStep, ...]
    result_step: str
    result_path_template: Path
    local_output_root: Path


@dataclass(frozen=True)
class H800SearchSummary:
    schema: str
    target: str
    code_revision: str
    seed: int
    configuration_label: str
    assets: tuple[RegisteredAsset, ...]
    status: Literal["completed", "failed"]
    planned_rounds: int
    completed_rounds: int
    successful_candidate_count: int
    aggregate_metrics: Mapping[str, Mapping[str, float]]
    failure_code: str | None


def validate_code_revision(value: str) -> str:
    """Validate one explicit, public-safe code revision label."""
    return _require_public_identifier(value, "code_revision")


def summary_to_public_dict(summary: H800SearchSummary) -> dict[str, object]:
    """Render the fixed public summary surface without local execution details."""
    _validate_public_summary(summary)
    return {
        "schema": SUMMARY_SCHEMA_VERSION,
        "target": summary.target,
        "code_revision": summary.code_revision,
        "seed": summary.seed,
        "configuration_label": summary.configuration_label,
        "assets": [
            {
                "label": asset.label,
                "version": asset.version,
                "license_status": asset.license_status,
            }
            for asset in summary.assets
        ],
        "status": summary.status,
        "planned_rounds": summary.planned_rounds,
        "completed_rounds": summary.completed_rounds,
        "successful_candidate_count": summary.successful_candidate_count,
        "aggregate_metrics": {
            name: {
                "count": values["count"],
                "min": values["min"],
                "max": values["max"],
                "mean": values["mean"],
            }
            for name in _MEASUREMENT_METRICS
            if (values := summary.aggregate_metrics.get(name)) is not None
        },
        "failure_code": summary.failure_code,
    }


def write_public_summary(path: Path, summary: H800SearchSummary) -> None:
    """Atomically publish a JSON summary through a temporary sibling file."""
    payload = summary_to_public_dict(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _validate_public_summary(summary: H800SearchSummary) -> None:
    if not isinstance(summary, H800SearchSummary) or summary.target != "h800":
        raise H800SearchContractError("summary target is invalid")
    validate_code_revision(summary.code_revision)
    _require_public_identifier(summary.configuration_label, "configuration_label")
    _require_positive_integer(summary.seed, "seed")
    if not summary.assets or not all(
        isinstance(asset, RegisteredAsset) for asset in summary.assets
    ):
        raise H800SearchContractError("summary assets are invalid")
    asset_labels: set[str] = set()
    for asset in summary.assets:
        label = _require_public_identifier(asset.label, "asset label")
        _require_public_identifier(asset.version, "asset version")
        _require_public_identifier(asset.license_status, "asset license_status")
        if label in asset_labels:
            raise H800SearchContractError("summary asset labels must be unique")
        asset_labels.add(label)

    if summary.status == "completed":
        if summary.failure_code is not None:
            raise H800SearchContractError("completed summary cannot have a failure code")
    elif summary.status == "failed":
        if summary.failure_code not in _SUMMARY_FAILURE_CODES:
            raise H800SearchContractError("summary failure code is invalid")
    else:
        raise H800SearchContractError("summary status is invalid")

    counts = (
        (summary.planned_rounds, "planned_rounds", True),
        (summary.completed_rounds, "completed_rounds", False),
        (summary.successful_candidate_count, "successful_candidate_count", False),
    )
    for value, description, positive in counts:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (1 if positive else 0)
        ):
            raise H800SearchContractError(f"summary {description} is invalid")
    if summary.completed_rounds > summary.planned_rounds:
        raise H800SearchContractError("summary completed_rounds is invalid")

    if not isinstance(summary.aggregate_metrics, Mapping) or not set(
        summary.aggregate_metrics
    ).issubset(_MEASUREMENT_METRICS):
        raise H800SearchContractError("summary aggregate metric names are invalid")
    for values in summary.aggregate_metrics.values():
        if not isinstance(values, Mapping) or set(values) != _SUMMARY_METRIC_KEYS:
            raise H800SearchContractError("summary aggregate metric fields are invalid")
        if not all(_finite_number(value) for value in values.values()):
            raise H800SearchContractError("summary aggregate metric values are invalid")
        count = float(values["count"])
        minimum = float(values["min"])
        maximum = float(values["max"])
        mean = float(values["mean"])
        if count < 1 or not count.is_integer() or not minimum <= mean <= maximum:
            raise H800SearchContractError("summary aggregate metric values are invalid")


def load_public_contract(path: Path) -> PublicH800SearchContract:
    """Load a public, location-free H800 search declaration without side effects."""
    raw_contract = _load_mapping(path, "public contract")
    _reject_public_execution_details(raw_contract)
    _require_exact_keys(raw_contract, _PUBLIC_KEYS, "public contract")

    if raw_contract["schema_version"] != PUBLIC_SCHEMA_VERSION:
        raise H800SearchContractError("public schema_version is invalid")
    if raw_contract["target"] != "h800":
        raise H800SearchContractError("target must be h800")

    assets = _parse_assets(raw_contract["assets"])
    metric_names = _parse_string_list(raw_contract["metric_names"], "metric_names")
    if frozenset(metric_names) != frozenset(_MEASUREMENT_METRICS):
        raise H800SearchContractError("metric_names must exactly match runtime metrics")
    return PublicH800SearchContract(
        search_id=_require_public_identifier(raw_contract["search_id"], "search_id"),
        target="h800",
        target_model=_require_public_identifier(raw_contract["target_model"], "target_model"),
        seed=_require_positive_integer(raw_contract["seed"], "seed"),
        max_rounds=_require_positive_integer(raw_contract["max_rounds"], "max_rounds"),
        batch_size=_require_positive_integer(raw_contract["batch_size"], "batch_size"),
        configuration_label=_require_public_identifier(
            raw_contract["configuration_label"], "configuration_label"
        ),
        assets=assets,
        metric_names=metric_names,
        candidate_space_label=_require_public_identifier(
            raw_contract["candidate_space_label"], "candidate_space_label"
        ),
    )


def load_local_config(path: Path, contract: PublicH800SearchContract) -> LocalH800SearchConfig:
    """Load local execution details without touching referenced assets or outputs."""
    raw_config = _load_mapping(path, "local config")
    _require_exact_keys(raw_config, _LOCAL_KEYS, "local config")

    if raw_config["schema_version"] != LOCAL_SCHEMA_VERSION:
        raise H800SearchContractError("local schema_version is invalid")
    if raw_config["target"] != contract.target:
        raise H800SearchContractError("local target must match the public contract")

    asset_paths = _parse_path_mapping(raw_config["asset_paths"], "asset_paths")
    registered_labels = {asset.label for asset in contract.assets}
    if set(asset_paths) != registered_labels:
        raise H800SearchContractError("local asset labels must exactly match public asset labels")

    stage5_input_paths = _parse_path_mapping(raw_config["stage5_input_paths"], "stage5_input_paths")
    steps = _parse_steps(raw_config["steps"])
    result_step = _require_nonempty_string(raw_config["result_step"], "result_step")
    if result_step != steps[-1].name:
        raise H800SearchContractError("result_step must reference the last step")
    if "{result_json}" not in steps[-1].argv:
        raise H800SearchContractError("result_step must produce {result_json}")

    result_path_template = _parse_absolute_path(
        raw_config["result_path_template"], "result_path_template"
    )
    local_output_root = _parse_absolute_path(
        raw_config["local_output_root"], "local_output_root"
    )
    _require_path_beneath(result_path_template, local_output_root)

    return LocalH800SearchConfig(
        asset_paths=MappingProxyType(asset_paths),
        stage5_input_paths=MappingProxyType(stage5_input_paths),
        steps=steps,
        result_step=result_step,
        result_path_template=result_path_template,
        local_output_root=local_output_root,
    )


def run_h800_search(
    contract: PublicH800SearchContract,
    local: LocalH800SearchConfig,
    code_revision: str,
    command_runner: CommandRunner,
) -> H800SearchSummary:
    """Run a fail-closed H800 feedback loop through injected argv execution."""
    code_revision = validate_code_revision(code_revision)
    feedback_rows: tuple[Mapping[str, Any], ...] = ()
    feedback_graphs: tuple[Mapping[str, Any], ...] = ()
    measured_metrics: tuple[Mapping[str, float], ...] = ()
    completed_rounds = 0
    successful_candidate_count = 0

    try:
        for round_index in range(1, contract.max_rounds + 1):
            stage5_inputs = _load_stage5_inputs(contract, local)
            base_rows = stage5_inputs["measurements"]
            base_graphs = stage5_inputs["graph_features"]
            measured_rows = (*base_rows, *feedback_rows)
            measured_graphs = (*base_graphs, *feedback_graphs)
            selection = _select_round(
                contract=contract,
                measured_rows=measured_rows,
                measured_graphs=measured_graphs,
                source_registry=stage5_inputs["candidate_registry"],
                capability_profiles=stage5_inputs["capability_profiles"],
                closure=stage5_inputs["closure"],
            )
            groups = selection.get("groups")
            if not isinstance(groups, list) or not groups:
                raise H800SearchExecutionError(
                    "stage5_selection_failed", "Stage5 returned an empty selection"
                )

            round_feedback_rows: tuple[Mapping[str, Any], ...] = ()
            round_feedback_graphs: tuple[Mapping[str, Any], ...] = ()
            for candidate_index, selected_group in enumerate(groups, start=1):
                result_rows, result_graph, metrics = _execute_selected_group(
                    local=local,
                    command_runner=command_runner,
                    selected_group=selected_group,
                    round_index=round_index,
                    candidate_index=candidate_index,
                    candidate_count=len(groups),
                    feedback_count=len(measured_rows) + len(round_feedback_rows),
                )
                round_feedback_rows = (*round_feedback_rows, *result_rows)
                if not any(
                    graph["group_id"] == result_graph["group_id"]
                    for graph in (*feedback_graphs, *round_feedback_graphs)
                ):
                    round_feedback_graphs = (*round_feedback_graphs, result_graph)
                measured_metrics = (*measured_metrics, *metrics)
                successful_candidate_count += 1

            feedback_rows = (*feedback_rows, *round_feedback_rows)
            feedback_graphs = (*feedback_graphs, *round_feedback_graphs)
            completed_rounds += 1
    except H800SearchExecutionError as error:
        _write_failure_record(local.local_output_root, error)
        return _build_summary(
            contract=contract,
            code_revision=code_revision,
            status="failed",
            completed_rounds=completed_rounds,
            successful_candidate_count=successful_candidate_count,
            measured_metrics=measured_metrics,
            failure_code=error.failure_code,
        )

    return _build_summary(
        contract=contract,
        code_revision=code_revision,
        status="completed",
        completed_rounds=completed_rounds,
        successful_candidate_count=successful_candidate_count,
        measured_metrics=measured_metrics,
        failure_code=None,
    )


def _load_stage5_inputs(
    contract: PublicH800SearchContract,
    local: LocalH800SearchConfig,
) -> dict[str, Any]:
    for asset in contract.assets:
        path = local.asset_paths[asset.label]
        if not path.exists() or not (path.is_file() or path.is_dir()):
            raise H800SearchExecutionError("local_input_invalid", "registered asset is unavailable")
    if set(local.stage5_input_paths) != _STAGE5_INPUT_NAMES:
        raise H800SearchExecutionError(
            "local_input_invalid", "Stage5 input labels do not match the required inputs"
        )

    loaded: dict[str, Any] = {}
    for name in sorted(_STAGE5_INPUT_NAMES):
        path = local.stage5_input_paths[name]
        if not path.is_file():
            raise H800SearchExecutionError("local_input_invalid", "Stage5 input is unavailable")
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise H800SearchExecutionError(
                "local_input_invalid", "Stage5 input could not be loaded"
            ) from error

    if not isinstance(loaded["measurements"], list):
        raise H800SearchExecutionError("local_input_invalid", "measurements must be a list")
    if not isinstance(loaded["graph_features"], list):
        raise H800SearchExecutionError("local_input_invalid", "graph features must be a list")
    if not isinstance(loaded["capability_profiles"], list):
        raise H800SearchExecutionError("local_input_invalid", "capability profiles must be a list")
    if any(
        not isinstance(profile, Mapping) or profile.get("hardware_target") != contract.target
        for profile in loaded["capability_profiles"]
    ):
        raise H800SearchExecutionError(
            "capability_target_invalid", "capability profiles must target h800"
        )
    if not isinstance(loaded["candidate_registry"], Mapping):
        raise H800SearchExecutionError("local_input_invalid", "candidate registry must be an object")
    if not isinstance(loaded["closure"], Mapping):
        raise H800SearchExecutionError("local_input_invalid", "closure must be an object")
    return loaded


def _select_round(
    *,
    contract: PublicH800SearchContract,
    measured_rows: Sequence[Mapping[str, Any]],
    measured_graphs: Sequence[Mapping[str, Any]],
    source_registry: Mapping[str, Any],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        bundle = fit_production_bundle(
            measured_rows,
            measured_graphs,
            capability_profiles,
            closure,
            seed=contract.seed,
        )
        manifest = build_candidate_manifest(
            source_registry,
            measured_group_ids={str(row["group_id"]) for row in measured_rows},
            frozen_holdout=closure.get("frozen_holdout") or {"groups": []},
            capability_profiles=capability_profiles,
        )
        candidate_rows = manifest.get("rows")
        if not isinstance(candidate_rows, list) or not candidate_rows:
            raise ValueError("candidate manifest is empty")
        target_group_ids = {
            str(row["group_id"])
            for row in candidate_rows
            if row.get("model") == contract.target_model
        }
        if not target_group_ids:
            raise ValueError("target model has no eligible candidate groups")
        predicted = predict_candidate_rows(bundle, candidate_rows, capability_profiles)
        selection = select_predicted_frontier_diversity(
            predicted,
            measured_rows,
            measured_graphs,
            group_budget_by_model={
                contract.target_model: min(contract.batch_size, len(target_group_ids))
            },
        )
    except Exception as error:
        raise H800SearchExecutionError(
            "stage5_selection_failed", f"Stage5 fitting or selection failed: {error}"
        ) from error
    if not isinstance(selection, Mapping):
        raise H800SearchExecutionError(
            "stage5_selection_failed", "Stage5 selection must be an object"
        )
    return selection


def _execute_selected_group(
    *,
    local: LocalH800SearchConfig,
    command_runner: CommandRunner,
    selected_group: Any,
    round_index: int,
    candidate_index: int,
    candidate_count: int,
    feedback_count: int,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
    tuple[Mapping[str, float], ...],
]:
    if not isinstance(selected_group, Mapping):
        raise H800SearchExecutionError(
            "stage5_selection_failed", "selected candidate group must be an object"
        )
    candidate_id = selected_group.get("group_id")
    rows = selected_group.get("rows")
    if not isinstance(candidate_id, str) or not candidate_id or not isinstance(rows, list) or not rows:
        raise H800SearchExecutionError(
            "stage5_selection_failed", "selected candidate group is incomplete"
        )
    if not all(isinstance(row, Mapping) for row in rows):
        raise H800SearchExecutionError(
            "stage5_selection_failed", "selected candidate rows are invalid"
        )
    arms = {
        (str(row.get("dispatch_key")), str(row.get("q_mode")))
        for row in rows
    }
    row_ids = [str(row.get("row_id") or "") for row in rows]
    if (
        len(rows) != 4
        or arms != _EXPECTED_ARMS
        or any(row.get("group_id") != candidate_id for row in rows)
        or any(not row_id for row_id in row_ids)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise H800SearchExecutionError(
            "stage5_selection_failed", "selected candidate must be one complete four-arm group"
        )

    round_directory = local.local_output_root / f"round-{round_index:04d}"
    candidate_directory = (
        round_directory
        if candidate_count == 1
        else round_directory / f"candidate-{candidate_index:04d}"
    )
    request_path = candidate_directory / "candidate_request.json"
    request = {
        "schema_version": "p6_h800_candidate_request_v1",
        "round_index": round_index,
        "candidate_index": candidate_index,
        "candidate_id": candidate_id,
        "feedback_count": feedback_count,
        "rows": [dict(row) for row in rows],
    }
    _write_json_record(request_path, request)

    result_path = local.result_path_template
    replacements = {
        "{candidate_request}": str(request_path),
        "{result_json}": str(result_path),
        "{local_output_root}": str(local.local_output_root),
    }
    for step in local.steps:
        if step.name == local.result_step:
            try:
                result_path.unlink(missing_ok=True)
            except OSError as error:
                raise H800SearchExecutionError(
                    "local_record_write_failed", "stale result could not be removed"
                ) from error
        argv = tuple(replacements.get(token, token) for token in step.argv)
        try:
            return_code = command_runner(argv, candidate_directory)
        except Exception as error:
            raise H800SearchExecutionError("command_failed", "local command runner failed") from error
        if isinstance(return_code, bool) or not isinstance(return_code, int) or return_code != 0:
            raise H800SearchExecutionError("command_failed", "local command returned nonzero")

    result = _load_result(result_path)
    return _validate_result(candidate_id, rows, result)


def _write_json_record(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise H800SearchExecutionError(
            "local_record_write_failed", "local execution record could not be written"
        ) from error


def _load_result(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise H800SearchExecutionError("result_missing", "result JSON was not produced")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise H800SearchExecutionError("result_invalid_json", "result JSON could not be loaded") from error
    if not isinstance(result, Mapping):
        raise H800SearchExecutionError("result_invalid_json", "result JSON must be an object")
    return result


def _validate_result(
    candidate_id: str,
    selected_rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
    tuple[Mapping[str, float], ...],
]:
    if set(result) != {"candidate_id", "measurements"}:
        raise H800SearchExecutionError("result_metrics_invalid", "result fields are invalid")
    if result.get("candidate_id") != candidate_id:
        raise H800SearchExecutionError(
            "result_candidate_mismatch", "result candidate does not match request"
        )
    measurements = result.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(selected_rows):
        raise H800SearchExecutionError(
            "result_metrics_invalid", "result measurements are incomplete"
        )

    expected_keys = {"candidate_id", *_MEASUREMENT_METRICS}
    feedback_rows: list[Mapping[str, Any]] = []
    metric_rows: list[Mapping[str, float]] = []
    for selected_row, measurement in zip(selected_rows, measurements, strict=True):
        if not isinstance(measurement, Mapping) or set(measurement) != expected_keys:
            raise H800SearchExecutionError(
                "result_metrics_invalid", "measurement fields are invalid"
            )
        if measurement.get("candidate_id") != selected_row.get("row_id"):
            raise H800SearchExecutionError(
                "result_candidate_mismatch", "measurement candidate does not match request"
            )
        metrics = {name: measurement[name] for name in _MEASUREMENT_METRICS}
        if not all(_finite_number(value) for value in metrics.values()):
            raise H800SearchExecutionError(
                "result_metrics_invalid", "measurement values must be finite numbers"
            )
        detached_metrics = MappingProxyType(
            {name: float(value) for name, value in metrics.items()}
        )
        source = {
            name: value
            for name, value in selected_row.items()
            if name not in {"schema_version", "graph_features", "predictions", "prediction_intervals", "prediction_bundle_sha256"}
        }
        feedback_rows.append(
            MappingProxyType(
                {
                    **source,
                    **detached_metrics,
                    "training_source": "online_feedback",
                    "terminal_status": "measured_success_gold",
                }
            )
        )
        metric_rows.append(detached_metrics)

    graph = selected_rows[0].get("graph_features")
    if not isinstance(graph, Mapping):
        raise H800SearchExecutionError(
            "stage5_selection_failed", "selected candidate graph features are missing"
        )
    return tuple(feedback_rows), dict(graph), tuple(metric_rows)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _aggregate_metrics(
    measured_metrics: Sequence[Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    aggregates: dict[str, Mapping[str, float]] = {}
    for name in _MEASUREMENT_METRICS:
        values = [float(row[name]) for row in measured_metrics]
        if values:
            aggregates[name] = MappingProxyType(
                {
                    "count": float(len(values)),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
            )
    return MappingProxyType(aggregates)


def _build_summary(
    *,
    contract: PublicH800SearchContract,
    code_revision: str,
    status: Literal["completed", "failed"],
    completed_rounds: int,
    successful_candidate_count: int,
    measured_metrics: Sequence[Mapping[str, float]],
    failure_code: str | None,
) -> H800SearchSummary:
    return H800SearchSummary(
        schema=SUMMARY_SCHEMA_VERSION,
        target=contract.target,
        code_revision=code_revision,
        seed=contract.seed,
        configuration_label=contract.configuration_label,
        assets=contract.assets,
        status=status,
        planned_rounds=contract.max_rounds,
        completed_rounds=completed_rounds,
        successful_candidate_count=successful_candidate_count,
        aggregate_metrics=_aggregate_metrics(measured_metrics),
        failure_code=failure_code,
    )


def _write_failure_record(output_root: Path, error: H800SearchExecutionError) -> None:
    try:
        _write_json_record(
            output_root / "failure.json",
            {"failure_code": error.failure_code, "detail": str(error)},
        )
    except H800SearchExecutionError:
        return


def _load_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise H800SearchContractError(f"could not load {description}") from error
    if not isinstance(loaded, Mapping):
        raise H800SearchContractError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise H800SearchContractError(f"{description} keys must be strings")
    return loaded


def _reject_public_execution_details(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _is_forbidden_public_key(key):
                raise H800SearchContractError("public contract contains forbidden execution detail")
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
        or normalized == "candidate_id"
        or "hash" in normalized
        or normalized.startswith("sha")
        or normalized in {"raw_log", "raw_logs", "checkpoint"}
    )


def _require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], description: str) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = allowed - keys
    if unknown:
        raise H800SearchContractError(f"{description} contains unknown keys: {sorted(unknown)}")
    if missing:
        raise H800SearchContractError(f"{description} is missing required keys: {sorted(missing)}")


def _parse_assets(value: Any) -> tuple[RegisteredAsset, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError("assets must be a non-empty list")

    assets: list[RegisteredAsset] = []
    labels: set[str] = set()
    for raw_asset in value:
        if not isinstance(raw_asset, Mapping) or not all(isinstance(key, str) for key in raw_asset):
            raise H800SearchContractError("each asset must be a mapping")
        _require_exact_keys(raw_asset, _ASSET_KEYS, "asset")
        asset = RegisteredAsset(
            label=_require_public_identifier(raw_asset["label"], "asset label"),
            version=_require_public_identifier(raw_asset["version"], "asset version"),
            license_status=_require_public_identifier(
                raw_asset["license_status"], "asset license_status"
            ),
        )
        if asset.label in labels:
            raise H800SearchContractError("asset labels must be unique")
        labels.add(asset.label)
        assets.append(asset)
    return tuple(assets)


def _parse_string_list(value: Any, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError(f"{description} must be a non-empty list")
    parsed = tuple(_require_public_identifier(item, description) for item in value)
    if len(set(parsed)) != len(parsed):
        raise H800SearchContractError(f"{description} must contain unique values")
    return parsed


def _parse_path_mapping(value: Any, description: str) -> dict[str, Path]:
    if not isinstance(value, Mapping) or not value:
        raise H800SearchContractError(f"{description} must be a non-empty mapping")
    parsed: dict[str, Path] = {}
    for key, raw_path in value.items():
        label = _require_nonempty_string(key, f"{description} label")
        parsed[label] = _parse_absolute_path(raw_path, f"{description}[{label}]")
    return parsed


def _parse_steps(value: Any) -> tuple[LocalExecutionStep, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError("steps must be a non-empty list")

    steps: list[LocalExecutionStep] = []
    names: set[str] = set()
    for raw_step in value:
        if not isinstance(raw_step, Mapping) or not all(isinstance(key, str) for key in raw_step):
            raise H800SearchContractError("each step must be a mapping")
        _require_exact_keys(raw_step, _STEP_KEYS, "step")
        name = _require_nonempty_string(raw_step["name"], "step name")
        if name in names:
            raise H800SearchContractError("step names must be unique")
        names.add(name)
        steps.append(LocalExecutionStep(name=name, argv=_parse_argv(raw_step["argv"])))
    return tuple(steps)


def _parse_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError("step argv must be a non-empty list")
    argv = tuple(_require_nonempty_string(token, "argv token") for token in value)
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if executable in _SHELL_EXECUTABLES:
        raise H800SearchContractError("step argv must not invoke a shell executable")
    if executable in _COMMAND_WRAPPER_EXECUTABLES:
        raise H800SearchContractError("step argv must not invoke a command wrapper")
    for token in argv:
        if ("{" in token or "}" in token) and token not in _ALLOWED_TEMPLATE_TOKENS:
            raise H800SearchContractError("argv contains an unknown or embedded placeholder")
    return argv


def _parse_absolute_path(value: Any, description: str) -> Path:
    path = Path(_require_nonempty_string(value, description))
    if not path.is_absolute():
        raise H800SearchContractError(f"{description} must be an absolute path")
    return path


def _require_path_beneath(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise H800SearchContractError("result_path_template is invalid") from error
    if resolved_root not in resolved_path.parents:
        raise H800SearchContractError(
            "result_path_template must be beneath local_output_root"
        )


def _require_nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise H800SearchContractError(f"{description} must be a non-empty string")
    return value


def _require_public_identifier(value: Any, description: str) -> str:
    identifier = _require_nonempty_string(value, description)
    if "/" in identifier or "\\" in identifier or any(character.isspace() for character in identifier):
        raise H800SearchContractError(f"{description} contains restricted public information")
    if _RAW_DIGEST_PATTERN.search(identifier) or any(
        pattern.search(identifier) for pattern in _PUBLIC_RESTRICTED_VALUE_PATTERNS
    ):
        raise H800SearchContractError(f"{description} contains restricted public information")
    return identifier


def _require_positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise H800SearchContractError(f"{description} must be a positive integer")
    return value
