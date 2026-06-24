# 新增模型 Adapter 教程

Stage1 支持两种 adapter：

- `TraceAdapter`：正式底层接口，写在 `framework/stage1/adapters.py`。适合模型结构特殊、需要手写 wrapper 的情况。
- `AutoTraceAdapter`：已经实现，写在 `framework/stage1/auto_trace.py`。它可以自动处理 dummy input、ignored head 探测和语义桶复用，但仍需要你提供一个 `build_fn`，返回已经包装好的 trace-ready dense net。

`AutoTraceAdapter` 不是“任意 PyTorch 模型输入后自动识别完整 dense core”的系统。当前自动化边界是：你负责把模型加载、checkpoint 加载、dense-core wrapper、不可 trace 子图说明写清楚；Stage1 负责 DepGraph 扫描、结构特征提取、硬件约束过滤和 manifest 输出。

## 推荐路径：AutoTraceAdapter

### 1. 写 trace-ready wrapper

假设原模型完整 forward 依赖 sparse VFE、scatter、fusion 或多车输入，Stage1 不直接 trace 完整 forward。你需要写一个 wrapper，让它只接收 dense BEV tensor。

示例：

```python
class MyModelTraceNet(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.backbone = full.backbone
        self.neck = full.neck
        self.cls_head = full.cls_head
        self.reg_head = full.reg_head

    def forward(self, spatial_features):
        x = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
        x = self.neck(x)
        return self.cls_head(x), self.reg_head(x)
```

规则：

- wrapper 输入必须是固定 shape 的 dense tensor，例如 `(1, 64, 256, 512)`。
- sparse VFE、scatter、geometry projection、multi-agent fusion、attention、routing 如果不进入 wrapper，必须记录为 skipped subgraph。
- 输出 head 通常不剪枝，应该被 `ignored_layers` 捕获或显式指定。

### 2. 写 build_fn

在 `framework/stage1/auto_trace.py` 中新增一个构建函数：

```python
def _build_my_model(config_path: str, ckpt_path: str, device: str):
    _add_path("/path/to/model/repo")

    from my_project.config import load_config
    from my_project.models import build_model

    cfg = load_config(config_path)
    full = build_model(cfg)

    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    full.load_state_dict(state_dict, strict=False)

    full = full.to(device).eval()
    return MyModelTraceNet(full).to(device).eval()
```

如果模型没有可用 checkpoint，不要把随机初始化结果写成训练模型扫描。应设置 `ckpt_status="missing"` 或 `ckpt_status="missing_architecture_scan_only"`，分类器会保守处理。

### 3. 注册 AutoTraceAdapter

在 `AUTO_REGISTRY` 中加入：

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
        "pillar_vfe (sparse VFE, not traced)",
        "scatter (sparse scatter, not traced)",
        "fusion_net (multi-agent fusion, not traced)",
    ],
    trace_note=(
        "trace core = backbone -> neck -> cls/reg heads; "
        "sparse preprocessing and fusion are skipped"
    ),
    ignored_attr_names=["cls_head", "reg_head"],
)
```

如果 head 名称符合 `cls_head/reg_head/dir_head/single_head`，可以省略 `ignored_attr_names`，让 `AutoTraceAdapter` 自动探测。

`bev_shape` 不会在导入时自动读取外部配置，目的是让没有安装外部模型仓库的用户也能打开 CLI help。添加新模型时请根据模型真实 dense BEV 入口显式填写。

### 4. 运行 Stage1 扫描

`framework.stage1.run_scan` 已同时支持 `REGISTRY` 和 `AUTO_REGISTRY`：

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model my_model \
  --hw configs/hardware/rtx3090.yaml \
  --device cuda \
  --out-dir framework/partitions \
  --profile-latency auto
```

输出：

```text
framework/partitions/my_model_partition.yaml
```

如果只验证结构：

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model my_model \
  --hw configs/hardware/rtx3090.yaml \
  --device cpu \
  --profile-latency off
```

### 5. 运行分类器

```bash
PYTHONPATH=. python scripts/stage1_classify_models.py \
  --manifest framework/partitions/my_model_partition.yaml \
  --evidence-dir results/stage1_model_predict \
  --out-json results/stage1_model_predict/model_classifier/my_model_classification.json \
  --out-md results/stage1_model_predict/model_classifier/my_model_classification.md
```

如果没有 S2/S2.5/S3/S4 证据，分类器会保守输出 blocker 和下一步 gate/probe，而不是直接给 full-model separability 结论。

## 备选路径：手写 TraceAdapter

如果模型需要更强控制，可以在 `framework/stage1/adapters.py` 直接继承 `TraceAdapter`：

```python
class MyModelAdapter(TraceAdapter):
    name = "my_model"
    model_class = "MyModel"
    config_path = "/path/to/config.yaml"
    ckpt_path = "/path/to/checkpoint.pth"
    ckpt_status = "ok"
    skipped_modules = [
        "pillar_vfe (sparse VFE)",
        "fusion_net (multi-agent fusion)",
    ]
    trace_note = "trace core = backbone -> neck -> heads"

    def build_trace_net(self, device):
        full = build_my_full_model(self.config_path, self.ckpt_path, device)
        net = MyModelTraceNet(full).to(device).eval()
        x = torch.randn(1, 64, 256, 512, device=device)
        return net, x

    def ignored_layers(self, net):
        return [net.cls_head, net.reg_head]
```

然后注册：

```python
REGISTRY["my_model"] = MyModelAdapter
```

## 检查清单

新增模型前后检查这些点：

- `build_trace_net` 或 `build_fn` 返回的是 `eval()` 模式下的 `nn.Module`。
- dummy input shape 与真实 dense BEV 入口一致。
- `net(dummy_input)` 可以在 CPU 上跑通。
- 输出 head、固定接口层、fusion 输入维度保护层已加入 ignored。
- 所有未 trace 的 sparse/fusion/attention/routing/custom 子图都写入 skipped 描述。
- 没有 checkpoint 时不要写成 trained checkpoint scan。
- 分类器输出中的 `unsupported_conclusions` 和 `blockers` 不应被手动删除。

## 常见错误

- 只写 `groups=1` 或 “no cliff”，但跳过了 fusion/attention 子图：这不能证明模型级可分离。
- 用随机初始化 fusion timing 代表训练模型：不允许。
- 用静态硬件 YAML 代表新硬件实测：不允许。
- 把 TRT 当作新增测量后端：不允许。当前新增实测后端策略是 H800 TVM/Relax/MetaSchedule。
