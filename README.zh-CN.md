# Stage1 Model Scanner

这是 Stage1 扫描与模型分类的最小开源运行库，包含 dense-core 计算图扫描、硬件 capability 约束、S2/S2.5/S3/S4 证据读取和三分类模型分类器。

## 安装

建议 Python 3.10+。

```bash
git clone <repo-url>
cd stage1-model-scanner-aaai
python -m venv .venv
source .venv/bin/activate
pip install torch torch-pruning pyyaml pydantic numpy
export PYTHONPATH=.
```

如果使用内置 CoDriving / Pyramid / V2X-ViT / HEAL adapter，还需要本地安装对应 `opencood` 代码，并设置路径：

```bash
export HEAL_ROOT=/path/to/HEAL
export HEAL_CKPT_ROOT=/path/to/heal/checkpoints
export V2XVERSE_ROOT=/path/to/V2Xverse
export V2XVERSE_CKPT_ROOT=/path/to/V2Xverse/checkpoints
```

H800 TVM 证据脚本需要本地 TVM/Relax/MetaSchedule 环境。TRT 在分类器中只作为历史证据，不作为新增实测后端。

## 目录

```text
framework/stage1/                 Stage1 硬件扫描、trace 边界检测、DepGraph 扫描、预测器、分类器
framework/capability_schema.py     硬件 YAML 校验
framework/stage1_bridge.py         manifest 到搜索空间的转换
tools/configurable/depgraph_*.py   Pyramid 与 V2X-ViT trace wrapper
scripts/stage1_*.py                Stage1 报告和分类 CLI
scripts/phase2/stage1_*.py         可选 S2/S2.5/S3/S4 证据工具
```

首次运行前创建本地输入输出目录：

```bash
mkdir -p configs/hardware framework/partitions results/stage1_model_predict
```

## 硬件 YAML

将硬件能力文件放在 `configs/hardware/<name>.yaml`，或运行时通过 `--hw` 指定。

最小示例：

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

硬件 YAML 是静态 capability 描述，只能支持结构合法性判断。若要声称新硬件上的实测延迟或精度结论，必须接入该硬件并运行本地 probe。

## 扫描模型

内置模型可以直接扫描：

```bash
PYTHONPATH=. python -m framework.stage1.run_scan --model all --device cpu --profile-latency off
```

单模型运行：

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model <registry_name> \
  --hw configs/hardware/<name>.yaml \
  --device cuda \
  --out-dir framework/partitions \
  --profile-latency auto
```

输出：

```text
framework/partitions/<registry_name>_partition.yaml
```

如果只做结构扫描，可以使用：

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model <registry_name> \
  --hw configs/hardware/<name>.yaml \
  --device cpu \
  --profile-latency off
```

扫描时框架会加载模型，构造或自动识别 dense trace candidate，执行 forward dry-run，构建 `torch-pruning` DepGraph，执行 0.5 剪枝 dry-run，并把 included / ignored / skipped / rejected trace 边界信息写入 manifest。

## 模型分类

仅基于 manifest 的分类：

```bash
PYTHONPATH=. python scripts/stage1_classify_models.py \
  --manifest framework/partitions/<registry_name>_partition.yaml \
  --evidence-dir results/stage1_model_predict \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

分类器输出三类：

- `CO_ACCELERATION_REQUIRED`：需要协同加速
- `SEPARABLE_ACCELERATION`：在声明范围内支持可分离加速
- `SCAN_FAILED`：训练 checkpoint 模型扫描失败或不可用

同时输出细粒度 verdict、blocker、下一步 gate/probe、证据来源、历史证据来源、禁止外推项和 `no_overpromotion`。

## 可选证据

可选证据放在 `results/stage1_model_predict/`：

```text
s2_schedule_anchor_audit_v1.json
s2_probe_results/stage1_s2_probe_completion_v1.json
s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json
s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json
s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json
```

证据脚本入口：

```bash
PYTHONPATH=. python scripts/phase2/stage1_s2_anchor_runner.py --help
PYTHONPATH=. python scripts/phase2/stage1_s2_probe_completion_report.py --help
PYTHONPATH=. python scripts/phase2/stage1_s2_5_s3_evidence_report.py --help
PYTHONPATH=. python scripts/phase2/stage1_s4_three_arm_validation.py --help
```

## 添加新模型

推荐路径是在 `framework/stage1/auto_trace.py` 中新增一个很薄的 `AutoTraceAdapter` 注册。

用户需要提供：

- 模型名
- config 路径
- checkpoint 路径
- 一个返回完整 `torch.nn.Module` 的最小加载函数
- 如果配置无法自动解析，再提供 input-shape hint

之后 Stage1 会自动扫描模块树，识别 dense candidate path，按启发式排除 sparse / fusion / routing / postprocess 区域，生成 wrapper candidate，执行 dry-run validation 和 DepGraph 验证，最后输出 trace-boundary manifest。用户主要负责审核 manifest 是否合理。只有自动候选不合理时，才使用手写 `TraceAdapter` 或手写 wrapper 兜底。

新增 adapter 教程见：[docs/add-new-model-adapter.zh-CN.md](docs/add-new-model-adapter.zh-CN.md)。

如果新模型家族需要更具体的判定规则，再扩展 `framework/stage1/model_classifier.py`。

## 添加新硬件

新增硬件时先写 capability YAML，然后用 `--hw` 运行扫描。静态 YAML 只能用于结构过滤；实测结论必须在目标硬件上运行 probe。当前新增实测后端策略是 H800 TVM/Relax/MetaSchedule；如果要引入其他后端，需要同步修改 backend policy 和分类器测试。

## 当前局限

- 当前 validation 验证的是可 trace dense candidate，不是完整模型 AP/精度。
- sparse VFE、几何投影、多车 fusion、routing、attention、自定义算子通常会被排除并记录为 skipped subgraph；除非提供模型专用插件，否则不会自动进入 dense DepGraph。
- 自动 trace 边界检测已覆盖当前内置协同感知模型族，但新架构仍需要用户审核 manifest，必要时提供 detector/plugin override。
- 硬件 YAML 不是实测数据，不能替代新硬件上的 probe。
- 内置模型依赖用户本地提供 HEAL/V2Xverse 源码和 checkpoint。
- 分类器是保守的证据合并器，不是可直接泛化到任意未知架构的学习型分类模型。
