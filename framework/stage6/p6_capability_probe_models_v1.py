"""Fixed model-independent ONNX probes for measured RTX compiler context."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any


OPSET = 13
_NEUTRAL = {
    "P1": ([1, 16, 16, 16], [1, 16, 16, 16], 1, 1),
    "P2": ([1, 16, 16, 16], [1, 16, 8, 8], 1, 2),
    "P3": ([1, 16, 16, 16], [1, 16, 16, 16], 1, 1),
    "P4": ([1, 16, 16, 16], [1, 16, 16, 16], 1, 1),
    "P5": ([1, 16, 16, 16], [1, 16, 16, 16], 4, 1),
    "P6": ([1, 16, 16, 16], [1, 16, 16, 16], 1, 1),
}
_PRUNING = {
    "aligned_channels": ([1, 64, 8, 8], [1, 96, 8, 8], 1, None),
    "misaligned_channels": ([1, 30, 8, 8], [1, 46, 8, 8], 1, None),
    "small_channels": ([1, 4, 8, 8], [1, 6, 8, 8], 1, None),
    "group_packed": ([1, 64, 8, 8], [1, 96, 8, 8], 4, None),
    "group_unpacked": ([1, 24, 8, 8], [1, 40, 8, 8], 4, None),
    "off_diagonal_two_stage_boundary": (
        [1, 32, 8, 8],
        [1, 64, 8, 8],
        1,
        [1, 48, 8, 8],
    ),
}


def _onnx_package() -> Any:
    root = Path(__file__).resolve().parents[2]
    loaded = sys.modules.get("onnx")
    if loaded is not None and hasattr(loaded, "helper"):
        return loaded
    if loaded is not None:
        del sys.modules["onnx"]
    original = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() != root]
        return importlib.import_module("onnx")
    finally:
        sys.path[:] = original


def _initializer(onnx: Any, name: str, shape: list[int], q_mode: str) -> Any:
    import numpy as np

    dtype = np.float16 if q_mode == "fp16" else np.float32
    return onnx.numpy_helper.from_array(np.full(shape, 0.03125, dtype=dtype), name=name)


def _append_qdq(
    onnx: Any, nodes: list[Any], initializers: list[Any], source: str, prefix: str
) -> str:
    import numpy as np

    scale, zero = f"{prefix}_scale", f"{prefix}_zero"
    quantized, dequantized = f"{prefix}_quantized", f"{prefix}_dequantized"
    initializers.extend(
        [
            onnx.numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), name=scale),
            onnx.numpy_helper.from_array(np.asarray(0, dtype=np.int8), name=zero),
        ]
    )
    nodes.extend(
        [
            onnx.helper.make_node(
                "QuantizeLinear", [source, scale, zero], [quantized], name=f"{prefix}_Q"
            ),
            onnx.helper.make_node(
                "DequantizeLinear",
                [quantized, scale, zero],
                [dequantized],
                name=f"{prefix}_DQ",
            ),
        ]
    )
    return dequantized


def _append_conv(
    onnx: Any,
    nodes: list[Any],
    initializers: list[Any],
    *,
    source: str,
    output: str,
    name: str,
    input_channels: int,
    output_channels: int,
    group: int,
    stride: int,
    q_mode: str,
) -> None:
    weight = f"{name}_weight"
    initializers.append(
        _initializer(
            onnx,
            weight,
            [output_channels, input_channels // group, 3, 3],
            q_mode,
        )
    )
    conv_source, conv_weight = source, weight
    if q_mode == "int8":
        conv_source = _append_qdq(onnx, nodes, initializers, source, f"{name}_activation")
        conv_weight = _append_qdq(onnx, nodes, initializers, weight, f"{name}_weight")
    nodes.append(
        onnx.helper.make_node(
            "Conv",
            [conv_source, conv_weight],
            [output],
            name=name,
            group=group,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[stride, stride],
        )
    )


def _neutral_graph(
    onnx: Any, probe_id: str, q_mode: str
) -> tuple[list[Any], list[Any], list[int], list[int]]:
    input_shape, output_shape, group, stride = _NEUTRAL[probe_id]
    nodes: list[Any] = []
    initializers: list[Any] = []
    if probe_id == "P3":
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="input",
            output="hidden",
            name="conv1",
            input_channels=16,
            output_channels=16,
            group=1,
            stride=1,
            q_mode=q_mode,
        )
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="hidden",
            output="output",
            name="conv2",
            input_channels=16,
            output_channels=16,
            group=1,
            stride=1,
            q_mode=q_mode,
        )
    else:
        target = "conv_output" if probe_id in {"P4", "P6"} else "output"
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="input",
            output=target,
            name="conv",
            input_channels=16,
            output_channels=16,
            group=group,
            stride=stride,
            q_mode=q_mode,
        )
        if probe_id == "P4":
            nodes.append(
                onnx.helper.make_node("Add", [target, "input"], ["output"], name="residual_add")
            )
        elif probe_id == "P6":
            source = (
                _append_qdq(onnx, nodes, initializers, target, "output_boundary")
                if q_mode == "int8"
                else target
            )
            nodes.append(
                onnx.helper.make_node("Identity", [source], ["output"], name="output_identity")
            )
    return nodes, initializers, input_shape, output_shape


def _pruning_graph(
    onnx: Any, probe_id: str, q_mode: str
) -> tuple[list[Any], list[Any], list[int], list[int]]:
    input_shape, output_shape, group, boundary = _PRUNING[probe_id]
    nodes: list[Any] = []
    initializers: list[Any] = []
    if boundary is not None:
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="input",
            output="stage1_output",
            name="stage1_conv",
            input_channels=input_shape[1],
            output_channels=boundary[1],
            group=1,
            stride=1,
            q_mode=q_mode,
        )
        nodes.append(
            onnx.helper.make_node(
                "Relu", ["stage1_output"], ["stage_boundary"], name="boundary_relu"
            )
        )
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="stage_boundary",
            output="output",
            name="stage2_conv",
            input_channels=boundary[1],
            output_channels=output_shape[1],
            group=1,
            stride=1,
            q_mode=q_mode,
        )
    else:
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="input",
            output="output",
            name="conv",
            input_channels=input_shape[1],
            output_channels=output_shape[1],
            group=group,
            stride=1,
            q_mode=q_mode,
        )
    return nodes, initializers, input_shape, output_shape


def build_probe_onnx(family: str, probe_id: str, q_mode: str) -> bytes:
    """Build and checker-validate one deterministic canonical probe."""
    if q_mode not in {"fp16", "int8"}:
        raise ValueError("probe precision invalid")
    onnx = _onnx_package()
    if family == "neutral" and probe_id in _NEUTRAL:
        nodes, initializers, input_shape, output_shape = _neutral_graph(onnx, probe_id, q_mode)
    elif family == "pruning" and probe_id in _PRUNING:
        nodes, initializers, input_shape, output_shape = _pruning_graph(onnx, probe_id, q_mode)
    else:
        raise ValueError("probe identity invalid")
    element_type = onnx.TensorProto.FLOAT16 if q_mode == "fp16" else onnx.TensorProto.FLOAT
    graph = onnx.helper.make_graph(
        nodes,
        f"p6_{family}_{probe_id}_{q_mode}",
        [onnx.helper.make_tensor_value_info("input", element_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", element_type, output_shape)],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="p6_capability_probe_models_v1",
        opset_imports=[onnx.helper.make_opsetid("", OPSET)],
    )
    model.ir_version = min(int(model.ir_version), 8)
    onnx.checker.check_model(model)
    return model.SerializeToString(deterministic=True)


__all__ = ["build_probe_onnx"]
