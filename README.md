# Stage1 Model Scanner

This repository contains the Stage1 Python runtime for dense-core graph scanning, hardware-capability filtering, evidence ingestion, and model classification.

## Install

Use Python 3.10+.

```bash
git clone <repo-url>
cd stage1-model-scanner-aaai
python -m venv .venv
source .venv/bin/activate
pip install torch torch-pruning pyyaml pydantic numpy
export PYTHONPATH=.
```

For built-in CoDriving/Pyramid/V2X-ViT/HEAL adapters, also install the corresponding `opencood` codebase and set local paths:

```bash
export HEAL_ROOT=/path/to/HEAL
export HEAL_CKPT_ROOT=/path/to/heal/checkpoints
export V2XVERSE_ROOT=/path/to/V2Xverse
export V2XVERSE_CKPT_ROOT=/path/to/V2Xverse/checkpoints
```

H800 TVM evidence scripts require a local TVM/Relax/MetaSchedule build. The classifier treats TRT only as historical evidence, not as a new measurement backend.

## Layout

```text
framework/stage1/                 Stage1 hardware scan, trace adapters, DepGraph scan, predictors, classifier
framework/capability_schema.py     optional hardware YAML validation
framework/stage1_bridge.py         manifest-to-search-space adapter
tools/configurable/depgraph_*.py   built-in Pyramid and V2X-ViT trace wrappers
scripts/stage1_*.py                report/classifier CLIs
scripts/phase2/stage1_*.py         optional S2/S2.5/S3/S4 evidence utilities
```

Create local input/output directories before running:

```bash
mkdir -p configs/hardware framework/partitions results/stage1_model_predict
```

## Hardware YAML

Put a capability file at `configs/hardware/<name>.yaml`, or pass `--hw`.

Minimal example:

```yaml
name: rtx3090
arch: Ampere sm86
ips:
  gpu:
    precisions: [FP32, TF32, FP16, INT8]
    tensor_core_gen: 3
    sparse_tc: true
alignment:
  int8_channel: 32
  fp16_channel: 8
  int8_pack_factor: 4
  alignment_enforcement: hard
quant_constraints:
  bit_widths_w: [8, 16]
  granularity_w: [per_tensor, per_channel]
  per_channel_activation_supported: false
  symmetric_only: true
memory:
  capacity_gb: 24
```

This is a static capability description. To claim measured behavior on a new device, attach that hardware and run local probes/evidence generation.

## Scan A Model

Register the model in `framework/stage1/adapters.py`. The adapter must provide:

- `build_trace_net(device) -> (torch.nn.Module, dummy_input)`
- `ignored_layers(net)` for output heads or fixed interface layers
- `skipped_modules` or `skipped_subgraphs` for sparse, fusion, attention, routing, or custom parts outside the dense core
- `semantic_bucket(layer_name)` if the default buckets are insufficient

Then run:

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model <registry_name> \
  --hw configs/hardware/<name>.yaml \
  --device cuda \
  --out-dir framework/partitions \
  --profile-latency auto
```

Output:

```text
framework/partitions/<registry_name>_partition.yaml
```

Use `--device cpu --profile-latency off` for a structure-only scan.

## Classify

Manifest-only classification:

```bash
PYTHONPATH=. python scripts/stage1_classify_models.py \
  --manifest framework/partitions/<registry_name>_partition.yaml \
  --evidence-dir results/stage1_model_predict \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

The classifier emits three coarse classes:

- `CO_ACCELERATION_REQUIRED`: needs joint/co-optimized acceleration
- `SEPARABLE_ACCELERATION`: separable acceleration is supported within the stated scope
- `SCAN_FAILED`: trained-checkpoint model scan is unavailable or failed

It also emits detailed verdicts, blockers, next gates/probes, evidence sources, historical evidence sources, unsupported conclusions, and `no_overpromotion`.

## Optional Evidence

Place optional evidence under `results/stage1_model_predict/`:

```text
s2_schedule_anchor_audit_v1.json
s2_probe_results/stage1_s2_probe_completion_v1.json
s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json
s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json
s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json
```

Generate or refresh evidence with:

```bash
PYTHONPATH=. python scripts/phase2/stage1_s2_anchor_runner.py --help
PYTHONPATH=. python scripts/phase2/stage1_s2_probe_completion_report.py --help
PYTHONPATH=. python scripts/phase2/stage1_s2_5_s3_evidence_report.py --help
PYTHONPATH=. python scripts/phase2/stage1_s4_three_arm_validation.py --help
```

## Add A New Model

Use either `TraceAdapter` in `framework/stage1/adapters.py` or the implemented `AutoTraceAdapter` in `framework/stage1/auto_trace.py`. `run_scan` supports both registries.

See the adapter tutorial: [docs/add-new-model-adapter.zh-CN.md](docs/add-new-model-adapter.zh-CN.md).

For a new model family, extend `framework/stage1/model_classifier.py` if the default low-confidence rule is not specific enough.

## Add A New Hardware Target

Add a hardware YAML and run the scan with `--hw`. Static YAML is enough for structural legality checks. Measured latency/AP evidence requires the target hardware locally attached and a matching local probe backend. New measured backend policy is H800 TVM/Relax/MetaSchedule unless you deliberately update the policy and classifier tests.

## Current Limits

- The scanner operates on the traceable dense core, not automatically on every full-model subgraph.
- Sparse VFE, geometry projection, multi-agent fusion, routing, attention, and custom operators must be traced explicitly or recorded as blockers.
- Hardware YAML is a static capability layer; it is not a substitute for measurement on an unseen device.
- Existing built-in model paths require local HEAL/V2Xverse source and checkpoint roots.
- Classifier rules are conservative evidence combiners, not a learned universal classifier for arbitrary unseen architectures.
