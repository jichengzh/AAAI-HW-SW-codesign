"""P6 post-source quantization fanout adapter."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from framework.stage6.p6_round_adapter_runtime_v1 import (
    LeafRunner,
    P6RoundAdapterRuntimeError,
    RoundContext,
    advance_task_state,
    load_round_context,
)


QUANT_SCHEMA = "stage3_tvm_int8_quant_contract_v3"


class P6QuantizationRoundAdapterError(ValueError):
    """Stable path-free quantization adapter failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def run_quantization_round(
    profile: ValidatedPostSourceAdapterProfile,
    task_state: Path,
    round_root: Path,
    runner: LeafRunner,
) -> None:
    """Run one quantization fanout and advance state after validation."""
    try:
        context = load_round_context(
            profile,
            task_state,
            round_root,
            prior_stage="initialized",
        )
        leaf = _quant_leaf(context.profile)
        int8_outputs = _run_int8_leaf_calls(context, leaf, runner)
        for output_path in int8_outputs:
            _require_native_quant_contract(output_path)
        advance_task_state(context, "quantization")
    except (P6QuantizationRoundAdapterError, P6RoundAdapterRuntimeError):
        raise P6QuantizationRoundAdapterError() from None
    except Exception:
        raise P6QuantizationRoundAdapterError() from None


def _run_int8_leaf_calls(
    context: RoundContext,
    leaf: PostSourceLeaf,
    runner: LeafRunner,
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    int8_call_index = 0
    for row in context.request["rows"]:
        if row["q_mode"] != "int8":
            continue
        output_path = _quant_output_path(context.round_root, row["width"])
        _run_one_leaf(context, leaf, row["source_contract"], output_path, runner, int8_call_index)
        outputs.append(output_path)
        int8_call_index += 1
    return tuple(outputs)


def _run_one_leaf(
    context: RoundContext,
    leaf: PostSourceLeaf,
    contract: Mapping[str, Any],
    output_path: Path,
    runner: LeafRunner,
    call_index: int,
) -> None:
    argv = (
        str(context.profile.project_python),
        str(leaf.implementation),
        "--onnx",
        _contract_path(contract, "onnx_path"),
        "--calibration-npz",
        _contract_path(contract, "calibration_npz"),
        "--calibration-summary",
        _contract_path(contract, "calibration_summary"),
        "--output-json",
        str(output_path),
    )
    completed = runner.run(
        argv,
        cwd=leaf.implementation_cwd,
        env=_leaf_env(context, call_index),
        shell=False,
    )
    if isinstance(completed.returncode, bool) or completed.returncode != 0:
        raise P6QuantizationRoundAdapterError()


def _quant_leaf(profile: ValidatedPostSourceAdapterProfile) -> PostSourceLeaf:
    matches = tuple(leaf for leaf in profile.leaves if leaf.name == "quant_contract")
    if len(matches) != 1:
        raise P6QuantizationRoundAdapterError()
    return matches[0]


def _quant_output_path(round_root: Path, width: object) -> Path:
    if (
        not isinstance(width, list)
        or len(width) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in width)
    ):
        raise P6QuantizationRoundAdapterError()
    return round_root / "quant_contracts" / "x".join(str(item) for item in width) / "tensor_quant_params.json"


def _contract_path(contract: Mapping[str, Any], key: str) -> str:
    value = contract.get(key)
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\r\n"):
        raise P6QuantizationRoundAdapterError()
    return value


def _leaf_env(context: RoundContext, call_index: int) -> dict[str, str]:
    inherited = _validated_incoming_env(context)
    return {
        "CUDA_VISIBLE_DEVICES": context.gpu_indices[call_index % len(context.gpu_indices)],
        **inherited,
        "PATH": os.pathsep.join((str(context.profile.project_python.parent), "/usr/bin", "/bin")),
        "PYTHONPATH": os.pathsep.join((str(context.profile.private_root), str(Path.cwd()))),
    }


def _validated_incoming_env(context: RoundContext) -> dict[str, str]:
    keys = (
        "P6_HISTORY_RUN_MODE",
        "P6_HISTORY_PRIVATE_ROOT",
        "P6_HISTORY_TASK_STATE",
        "P6_HISTORY_ROUND_OUTPUT_ROOT",
    )
    inherited = {key: os.environ[key] for key in keys if key in os.environ}
    if set(inherited) != set(keys):
        raise P6QuantizationRoundAdapterError()
    try:
        private_root = Path(inherited["P6_HISTORY_PRIVATE_ROOT"]).resolve(strict=False)
        task_state = Path(inherited["P6_HISTORY_TASK_STATE"]).resolve(strict=False)
        round_root = Path(inherited["P6_HISTORY_ROUND_OUTPUT_ROOT"]).resolve(strict=False)
    except OSError:
        raise P6QuantizationRoundAdapterError()
    if (
        private_root != context.profile.private_root.resolve(strict=False)
        or task_state != context.task_state_path
        or round_root != context.round_root
    ):
        raise P6QuantizationRoundAdapterError()
    return inherited


def _require_native_quant_contract(path: Path) -> None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise OSError
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise P6QuantizationRoundAdapterError() from None
    if not isinstance(payload, Mapping) or payload.get("schema") != QUANT_SCHEMA:
        raise P6QuantizationRoundAdapterError()
