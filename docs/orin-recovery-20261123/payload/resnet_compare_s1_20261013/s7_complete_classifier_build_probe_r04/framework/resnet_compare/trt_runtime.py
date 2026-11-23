"""TensorRT 8.5 target-local build and CUDA-completed inference (Python 3.8+).

TensorRT and CUDA are imported only when used, so CPU protocol tools remain
usable. Engine inspector tensor formats are evidence of actual execution graph
precision, not a guarantee of every fused arithmetic accumulator's precision.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from .runtime import _platform_name


def inspect_precisions(information):
    """Preserve raw actual inspector evidence; never infer from engine filename."""
    rows = []
    for layer in information.get("Layers", []):
        outputs = layer.get("Outputs", [])
        formats = [str(output.get("Format/Datatype", "")) for output in outputs]
        found = sorted({precision for precision in ("FP32", "FP16", "INT8", "INT32")
                        if any(precision in value.upper() for value in formats)})
        rows.append({"name": layer.get("Name", "unknown"),
                     "precision": "+".join(found) if found else "UNKNOWN",
                     "output_formats": formats, "inspector": layer,
                     "precision_evidence": "engine_inspector_output_tensor_format_not_arithmetic_guarantee"})
    return rows


def _require_orin():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 7) or "orin" not in _platform_name().lower():
        raise ValueError("Build/inference must run on actual Jetson Orin sm87")


def classifier_components(network, layers):
    """Recognize the parser's Gemm = matrix multiply plus its own bias SUM."""
    heads = [layer for layer in layers if layer.name == '/fc/Gemm']
    if len(heads) != 1:
        raise ValueError('Fixed policy requires exactly one /fc/Gemm layer')
    head = heads[0]
    kind = str(head.type).rsplit('.', 1)[-1]
    if kind in {'CONVOLUTION', 'FULLY_CONNECTED'}:
        return ['/fc/Gemm']
    if kind != 'MATRIX_MULTIPLY':
        raise ValueError('Unexpected classifier operation')
    tensor_name = head.get_output(0).name
    consumers = [layer for layer in layers if any(layer.get_input(i) is not None
                 and layer.get_input(i).name == tensor_name for i in range(layer.num_inputs))]
    if len(consumers) != 1:
        raise ValueError('Classifier matrix output must have exactly one bias consumer')
    bias_add = consumers[0]
    if (str(bias_add.type).rsplit('.', 1)[-1] != 'ELEMENTWISE'
            or str(bias_add.op).rsplit('.', 1)[-1] != 'SUM' or bias_add.num_inputs != 2):
        raise ValueError('Classifier tail must be its bias SUM')
    if network.num_outputs != 1 or network.get_output(0).name != bias_add.get_output(0).name:
        raise ValueError('Additional operations after classifier are outside fixed policy')
    producers = {layer.get_output(i).name: layer for layer in layers for i in range(layer.num_outputs)}
    other = [bias_add.get_input(i) for i in range(2) if bias_add.get_input(i).name != tensor_name]
    if len(other) != 1:
        raise ValueError('Classifier bias input is ambiguous')
    producer = producers.get(other[0].name)
    if producer is not None and str(producer.type).rsplit('.', 1)[-1] == 'SHUFFLE':
        producer = producers.get(producer.get_input(0).name)
    if producer is None or str(producer.type).rsplit('.', 1)[-1] != 'CONSTANT' or producer.name != 'fc.bias':
        raise ValueError('Classifier SUM does not originate from fc.bias constant')
    bias_add.name = '/fc/Gemm/bias_add'
    return ['/fc/Gemm', '/fc/Gemm/bias_add']


def apply_fc_only_policy(network, config, trt):
    """One fixed policy: final classifier FP16, every other float layer FP32."""
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    components = classifier_components(network, layers)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    constraints = []
    structural = []
    for layer in layers:
        kind = str(getattr(layer, 'type', '')).rsplit('.', 1)[-1]
        if kind in {'SHUFFLE', 'IDENTITY', 'CONSTANT', 'SHAPE'}:
            structural.append({'name': layer.name, 'type': kind, 'constraint': 'preserve_dataflow_dtype'})
            continue
        dtype = trt.float16 if layer.name in components else trt.float32
        floating = [i for i in range(layer.num_outputs)
                    if layer.get_output(i).dtype in (trt.float32, trt.float16)]
        if not floating:
            continue
        layer.precision = dtype
        for i in floating:
            layer.set_output_type(i, dtype)
        constraints.append({'name': layer.name, 'precision': 'FP16' if layer.name in components else 'FP32',
                            'output_indices': floating})
    return {'policy_id': 'fc_only_fp16_backbone_fp32_v1', 'constraints': constraints,
            'classifier_components': components, 'constraint_mode': 'OBEY_PRECISION_CONSTRAINTS', 'structural_unconstrained': structural,
            'interface_revision': 'complete_classifier_bias_and_structural_dtype_flow_v2'}


def add_fp32_output_cast(network, trt):
    """Move the FP32 IO contract to an explicit cast, not a reshape constraint."""
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    evidence = []
    for index, tensor in enumerate(outputs):
        original_name = tensor.name
        network.unmark_output(tensor)
        tensor.name = original_name + '_before_explicit_fp32_cast'
        cast = network.add_identity(tensor)
        if cast is None:
            raise ValueError('Cannot create explicit FP32 output cast')
        cast.name = 'explicit_fp32_output_cast_' + str(index)
        cast.set_output_type(0, trt.float32)
        output = cast.get_output(0)
        output.name = original_name
        output.dtype = trt.float32
        network.mark_output(output)
        evidence.append({'name': cast.name, 'output_name': original_name, 'output_dtype': 'FP32'})
    return evidence


def build_engine(onnx_path, engine_path, precision, *, calibrator=None):
    """Build once on Orin; no 4090 engine transfer, TF32, or INT8 fake ranges.

    A real IInt8Calibrator backed by Task2's frozen calibration manifest is
    mandatory for INT8. Caller must supply and retain its cache/evidence.
    """
    if precision == "INT8" and calibrator is None:
        raise ValueError("INT8 calibration required; do not fabricate tensor ranges")
    if precision not in ("FP32", "FP16", "INT8"):
        raise ValueError("Unsupported TensorRT precision")
    path = Path(engine_path)
    if path.exists():
        raise FileExistsError("Existing engine preserved; explicitly reuse it instead of rebuilding")
    _require_orin()
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "FP16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Target lacks fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "INT8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Target lacks fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    if precision != 'FP16':
        raise ValueError('Isolated fc-only runtime requires FP16 build request')
    precision_policy = apply_fc_only_policy(network, config, trt)
    precision_policy = {**precision_policy, 'explicit_output_casts': add_fp32_output_cast(network, trt)}
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed; no engine produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(bytes(serialized))
    return {"precision_requested": precision, "tf32_enabled": False, "precision_policy": precision_policy,
            "build_seconds": time.perf_counter() - started,
            "engine_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest(),
            "target": _platform_name(), "compute_capability": [8, 7],
            "tensorrt_version": trt.__version__, "workspace_bytes": 1 << 30}


class TensorRTRuntime:
    """TRT explicit static batch-one execution using torch-owned CUDA buffers."""
    backend = "tensorrt"

    def __init__(self, engine_path):
        _require_orin()
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution context creation failed")
        inputs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        outputs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one complete classification input/output")
        self.input_index, self.output_index = inputs[0], outputs[0]
        if tuple(self.engine.get_binding_shape(self.input_index)) != (1, 3, 224, 224) or tuple(
                self.engine.get_binding_shape(self.output_index)) != (1, 1000):
            raise ValueError("Engine binding shapes differ from RN50 RGB224/1000-logit contract")
        supported = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.buffers = [torch.empty(tuple(self.engine.get_binding_shape(i)),
                                   dtype=supported[self.engine.get_binding_dtype(i)], device="cuda:0")
                        for i in range(self.engine.num_bindings)]
        self.bindings = [tensor.data_ptr() for tensor in self.buffers]
        self.host_input = torch.empty_like(self.buffers[self.input_index], device="cpu", pin_memory=True)
        self.host_output = torch.empty_like(self.buffers[self.output_index], device="cpu", pin_memory=True)
        self.stream = torch.cuda.Stream(device=0)
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        inspector = self.engine.create_engine_inspector()
        inspector.execution_context = self.context
        self.inspection = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        self.layer_precision = inspect_precisions(self.inspection)
        self.target_graph_verified = bool(self.layer_precision)
        self.hardware = {"name": torch.cuda.get_device_name(0), "capability": [8, 7],
                         "platform": _platform_name()}

    def infer(self, tensor):
        started = time.perf_counter()
        # CPU packing is included in service latency, not in H2D/L1 CUDA time.
        self.host_input.copy_(tensor)
        with torch.cuda.stream(self.stream):
            self.events[0].record(self.stream)
            self.buffers[self.input_index].copy_(self.host_input, non_blocking=True)
            self.events[1].record(self.stream)
            if not self.context.execute_async_v2(self.bindings, self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v2 failed")
            self.events[2].record(self.stream)
            self.host_output.copy_(self.buffers[self.output_index], non_blocking=True)
            self.events[3].record(self.stream)
        wait_start = time.perf_counter()
        self.events[3].synchronize()
        wait_ms = (time.perf_counter() - wait_start) * 1000
        logits = self.host_output.clone().float()
        stages = {"h2d": self.events[0].elapsed_time(self.events[1]),
                  "forward": self.events[1].elapsed_time(self.events[2]),
                  "d2h": self.events[2].elapsed_time(self.events[3]), "wait": wait_ms}
        return {"logits": logits, "completion": "cuda_event_synchronized", "stages_ms": stages,
                "l1_device_ms": stages["forward"], "source_to_done_ms": (time.perf_counter() - started) * 1000}
