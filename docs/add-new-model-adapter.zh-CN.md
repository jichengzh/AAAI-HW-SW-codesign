# 新增模型 Adapter 教程

Stage1 的目标不是让用户手工指定 dense core，而是让用户提供模型入口后，由框架自动产生可审计的扫描边界。

正确流程是：

```text
用户提供 config + checkpoint + model name
        ↓
Stage1 加载完整模型
        ↓
自动扫描模块树
        ↓
自动识别 dense candidate path
        ↓
自动排除 sparse / fusion / routing / postprocess
        ↓
自动生成 wrapper candidate
        ↓
forward dry-run
        ↓
torch-pruning DepGraph 验证
        ↓
0.5 pruning dry-run 验证
        ↓
生成 included / ignored / skipped / rejected manifest
        ↓
分类器保守读取 manifest 和后续证据
```

用户仍然需要写一小段加载代码，因为框架无法预先知道任意研究仓库的 config parser、model factory 和 checkpoint 格式。这段代码只负责“如何加载完整模型”，不负责手工选择要扫描哪些层。

## 推荐路径：AutoTraceAdapter

### 1. 写完整模型加载器

在 `framework/stage1/auto_trace.py` 中新增一个 build 函数。这个函数应加载完整模型，然后调用 `TraceBoundaryDetector` 自动生成 trace candidate。

```python
from pathlib import Path

import torch

from framework.stage1.trace_plan import TraceBoundaryDetector, WrapperSynthesizer


def _build_my_model(config_path: str, ckpt_path: str, device: str):
    from my_project.config import load_config
    from my_project.models import build_model

    cfg = load_config(config_path)
    full = build_model(cfg)

    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    full.load_state_dict(state_dict, strict=False)
    full = full.to(device).eval()

    trace_plan = TraceBoundaryDetector().detect(
        full,
        model_name="my_model",
        config_path=config_path,
        ckpt_path=ckpt_path,
        ckpt_status="ok" if Path(ckpt_path).is_file() else "missing_architecture_scan_only",
        input_shape=(1, 64, 256, 512),
    )
    net = WrapperSynthesizer().synthesize(full, trace_plan["selected_candidate"]).to(device).eval()
    net._stage1_trace_plan = trace_plan
    return net
```

这里的 `input_shape` 是 dense BEV 或 dense feature 入口的形状 hint。它不是手工指定扫描层，只是帮助 wrapper dry-run 构造输入。

### 2. 注册 AutoTraceAdapter

仍在 `framework/stage1/auto_trace.py` 的 `AUTO_REGISTRY` 中注册：

```python
AUTO_REGISTRY["my_model"] = AutoTraceAdapter(
    name="my_model",
    model_class="MyModel",
    config_path="/path/to/config.yaml",
    ckpt_path="/path/to/checkpoint.pth",
    build_fn=_build_my_model,
    bev_shape=(1, 64, 256, 512),
    ckpt_status="ok",
    skipped_desc=[
        "pillar_vfe (sparse VFE, auto-skip)",
        "scatter (sparse scatter, auto-skip)",
        "fusion_net (multi-agent fusion, auto-skip)",
    ],
    trace_note="full model loaded; TraceBoundaryDetector selects dense candidate",
)
```

`skipped_desc` 是给报告的初始语义提示。最终 manifest 还会包含 `TraceBoundaryDetector` 生成的 typed `skipped_subgraphs`、`included_modules`、`ignored_layers` 和 `rejected_candidates`。

### 3. 运行扫描

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model my_model \
  --hw configs/hardware/rtx3090.yaml \
  --device cpu \
  --profile-latency off
```

输出：

```text
framework/partitions/my_model_partition.yaml
```

关键字段：

```yaml
trace_plan:
  detector: TraceBoundaryDetector
  selected_candidate:
    validation:
      wrapper_forward_dryrun: ok
      depgraph_build: ok
      prune_dryrun: ok
      boundary_validator: BoundaryValidator.graph_scan_integrated_v1
  included_modules: [...]
  ignored_layers: [...]
  skipped_subgraphs: [...]
  rejected_candidates: [...]
```

用户主要审核这些问题：

- included 是否确实是 dense backbone / neck / head candidate。
- skipped 是否覆盖 sparse VFE、scatter、fusion、routing、postprocess、自定义算子。
- ignored 是否包含输出 head、接口保护层或不应被剪枝的层。
- rejected candidate 的失败原因是否合理。
- validation 是否至少通过 forward、DepGraph 和 pruning dry-run。

## 手工兜底：TraceAdapter

如果自动候选明显错误，可以临时使用 `framework/stage1/adapters.py` 中的 `TraceAdapter` 兜底。手写 adapter 应被视为 override，而不是默认路径。

```python
class MyModelAdapter(TraceAdapter):
    name = "my_model_manual"
    model_class = "MyModel"
    config_path = "/path/to/config.yaml"
    ckpt_path = "/path/to/checkpoint.pth"
    ckpt_status = "ok"
    skipped_modules = [
        "pillar_vfe (sparse VFE)",
        "fusion_net (multi-agent fusion)",
    ]
    trace_note = "manual override: backbone -> neck -> heads"

    def build_trace_net(self, device):
        full = build_my_full_model(self.config_path, self.ckpt_path, device)
        net = MyManualTraceNet(full).to(device).eval()
        x = torch.zeros(1, 64, 256, 512, device=device)
        return net, x

    def ignored_layers(self, net):
        return [net.cls_head, net.reg_head]
```

使用手工兜底时，manifest / classifier 会把它视为需要 review 的低自动化边界；不要把它包装成自动扫描成功。

## 没有 Checkpoint 怎么办

没有 checkpoint 时可以做 architecture-only scan，但必须设置：

```python
ckpt_status="missing_architecture_scan_only"
```

这类扫描只能证明结构路径能否被扫描，不能代表训练模型分类，也不能作为精度或真实性能结论。

## 新硬件怎么办

新增硬件先写 `configs/hardware/<name>.yaml`，再用 `--hw` 指定。硬件 YAML 只提供静态 capability，如 SM 架构、支持精度、INT8 channel alignment、pack factor、DLA/NPU/GPU IP 和 op whitelist。

如果要声称新硬件上的实测延迟、吞吐或 AP，需要把该硬件接入本地环境并运行 probe。静态 YAML 不能替代实测。

## 检查清单

- 完整模型能通过 config + checkpoint 加载。
- `TraceBoundaryDetector` 产生了 `selected_candidate`。
- `wrapper_forward_dryrun == ok`。
- `depgraph_build == ok`。
- `prune_dryrun == ok`。
- manifest 中有 typed `skipped_subgraphs`。
- 没有 checkpoint 时没有写成 trained checkpoint scan。
- 分类器输出中的 `unsupported_conclusions` 和 `blockers` 没有被手工删除。

## 常见错误

- 把“需要提供模型加载器”误解为“需要手工指定扫描层”。
- 用随机初始化 timing 代表训练模型。
- 只因为 `groups=1` 或 no-cliff 就声称模型级可分离。
- 跳过 fusion / attention / routing 后仍声称 full-model separability。
- 用静态硬件 YAML 代表新硬件实测。
- 把 TRT 当作新增测量后端。当前新增实测后端策略是 H800 TVM/Relax/MetaSchedule。
