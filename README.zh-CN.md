# GEAR 协同设计复现包

本仓库提供 CPU-only smoke 工作流、小型已验证 Stage4 制品，以及用于审计投稿的
Stage1–7 选择、分析、发布检查和外部私有硬件运行契约。仓库不打包数据集、模型
checkpoint、ONNX 文件、编译 engine、TVM cache、raw logs、私有测量输出或私有源码树。
下方公开 CPU 工作流不会执行硬件测量、AP 评估、TVM 或 TensorRT。

证据边界请参阅 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) 和
[ARTIFACTS.md](ARTIFACTS.md)。审稿匿名包的入口为 `README.anonymous.md`；它并非
公开项目的 README。
持续维护的发布交接、当前状态及到完整开源的计划见
[docs/AAAI27_RELEASE_AUDIT.md](docs/AAAI27_RELEASE_AUDIT.md)。

## 快速开始：干净克隆与 CPU-only smoke

请使用 Python 3.10--3.13。匿名审稿期间，请从仓库页面取得 HTTPS 克隆 URL 并仅在本机
设置；不把 URL 写入源码可保持匿名边界。浅克隆和 partial clone 不会下载无关历史：

```bash
export GEAR_REPOSITORY_URL='<从仓库页面取得的 HTTPS 克隆 URL>'
git clone --depth 1 --filter=blob:none --single-branch \
  --branch release/aaai27-reproducibility "$GEAR_REPOSITORY_URL" gear-codesign
cd gear-codesign
```

创建 CPU-only 环境并运行确定性 smoke：

```bash
python --version
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python scripts/reproduce/reproduce_all.py --mode smoke --output-root ./repro-smoke-output
```

Smoke 是确定性的、离线的、CPU-only 的，仅使用 `data/demo/` 中的小型 fixture。
它只是契约演练，并非论文证据：每条输入记录都明确标注为
`paper_evidence: false`。输出的 `run_manifest.json` 记录输入和输出 SHA-256
标识、源文件哈希、冻结随机种子与执行边界。对内容相同且已完成的输出目录重复运行
将不产生新结果；内容不同的目录会被拒绝。

如需从检入的净化 scanner evidence 复现三个公开模型的正式搜索空间，运行：

```bash
python scripts/reproduce_paper_search_space.py --model all
```

该命令是离线、CPU-only 的。它只把 Pyramid、CoDriving 和 F-Cooper 的
scanner-contract fixture 送入公开 Stage1 bridge 与通用 Stage2 formal planner，
并打印派生得到的结构数与候选数；不会训练、测量 latency/energy/AP、调用
TVM/TensorRT，也不会访问数据集、checkpoint、GPU、SSH 或私有路径。

如需独立核验“新 clone + 新虚拟环境 + smoke”闭环，可从任意已有检出目录运行：

```bash
bash scripts/reproduce/smoke_clean_clone.sh \
  --repo-url "$GEAR_REPOSITORY_URL" \
  --ref release/aaai27-reproducibility
```

该脚本不会下载数据集、checkpoint、ONNX、编译 engine、TVM/TensorRT 制品或硬件测量；
为便于检查失败现场，它会保留临时工作目录。

同一入口也可审计检入的 verified bundle：

```bash
python scripts/reproduce/reproduce_all.py --mode verified --output-root ./repro-verified-output
```

该命令当前按设计应以非零状态退出。它会校验现有 Stage4 制品字节，然后因包内没有
必需的 Stage6 证据和 Stage7 正式 aggregate 而报告 `unavailable`。该非零结果是
可审计的可用性检查，不能视作成功复现论文。

## 外部私有 RTX4090 全链路运行

仓库还包含经过检查的公开 controller/verifier 接口，可用于外部、hardware-specific 的
RTX4090 运行。该路径不是 CPU clean-clone smoke 的一部分。它需要用户自行提供并负责
授权的私有资产与本地运行时：数据集、checkpoint/模型源码、必要的 ONNX 或 calibration
输入、可用的 CUDA/TVM sm89 工具链、私有 source-map 与 runner-template 文件，以及 Git
忽略的本地输出目录。这些材料不会随仓库发布。

材料所有权分为以下三类；自动生成的文件不是额外输入：

- **用户提供的私有输入：**私有 history root、私有 source-map、
  pre-normalization runner-template、RTX local config 或 locator、授权的数据集、模型源码/checkpoint、必要的
  ONNX 或 calibration 输入、CUDA/TVM sm89 工具链，以及 fresh output root。所有路径必须
  位于仓库外或保持 Git ignored。source-map 与 runner-template 描述受许可约束的历史代码，
  因此仓库不会提供可执行的公开示例。
- **仓库提供的公开参考：**`configs/execution/p6_rtx4090_search.example.yaml` 是四轮搜索的
  公开 contract；`configs/execution/p6_external_training_binding.example.yaml` 是
  仅空值 schema 示例。应将后者复制到 ignored 位置，使用运维方确认的值替换所有必填空值，再由
  validator 计算或核验稳定文件的 SHA-256。仓库中的空值示例按设计不可直接执行。
- **`normalize` 自动生成的私有输出：**
  `<abs-normalized-private-dir>/runner-template.yaml`、
  `<abs-normalized-private-dir>/source-wrapper-profile.yaml`、
  `<abs-normalized-private-dir>/external-training-binding.yaml` 和
  `<abs-normalized-private-dir>/post-source-adapter-profile.yaml`。不要分别手写这些 normalized
  authority；应按下方命令把同一次 normalize 生成的文件交给 controller。

RTX4090 运行使用公开 contract `configs/execution/p6_rtx4090_search.example.yaml` 作为
profile authority。保留的入口名 `tools/release/run_p6_h800_search.py` 是历史名称；当
它接收 v3 RTX4090 contract 时，会加载选定的 `rtx4090` hardware profile 并运行共享
controller 路径，而不是 H800-only 路径。

完整外部运行按以下顺序使用现有私有链路：

```bash
python tools/release/derive_p6_history_recipe.py \
  --source-map <abs-private-source-map.yaml> \
  --runner-template <abs-pre-normalization-private-runner-template.yaml> \
  --recipe-json <abs-output-recipe.json>

python tools/release/normalize_p6_history_root.py \
  --source-map <abs-private-source-map.yaml> \
  --history-root <abs-private-history-root> \
  --private-dir <abs-normalized-private-dir> \
  --runner-template <abs-pre-normalization-private-runner-template.yaml>

GPU_POOL=7 python tools/release/run_p6_h800_search.py \
  --contract configs/execution/p6_rtx4090_search.example.yaml \
  --code-revision "$(git rev-parse --short=12 HEAD)" \
  --legacy-local-config <abs-rtx-local-config-or-locator.yaml> \
  --runner-template <abs-normalized-private-dir>/runner-template.yaml \
  --local-output-root <abs-fresh-output-root> \
  --binding-output <abs-fresh-output-root>/binding.json \
  --config-output <abs-fresh-output-root>/local-config.yaml \
  --source-wrapper-profile <abs-source-wrapper-profile.yaml> \
  --external-training-binding <abs-external-training-binding.yaml> \
  --post-source-adapter-profile <abs-post-source-adapter-profile.yaml>

python tools/release/verify_p6_materializer_training_run.py \
  --contract configs/execution/p6_rtx4090_search.example.yaml \
  --local-config <abs-fresh-output-root>/local-config.yaml \
  --binding <abs-fresh-output-root>/binding.json
```

`--legacy-local-config` 这个 flag 名也是历史名称；在 RTX4090 运行中，它指向已批准的
RTX local config 或 locator，并由 provision 转换成 fresh binding/config pair。`derive`
和 `normalize` 使用 pre-normalization 私有 runner template。fresh-run 模式下，历史名称的
controller 入口会完成 `GPU_POOL` 晚绑定、生成 fresh binding/config、执行静态 preflight，
再启动四轮 controller；它不会修改 normalized authority：
`<abs-normalized-private-dir>/runner-template.yaml`。

`GPU_POOL` 是严格的正整数 GPU 数量，也是 fresh-run 模式下唯一的 GPU 选择入口。
`GPU_POOL=1` 表示申请一张可用卡，`GPU_POOL=3` 表示申请三张，`GPU_POOL=7` 表示申请
七张；调用方不指定物理 GPU 索引，也不提前持久化绑定 UUID。live admission 会从符合
hardware profile 且稳定空闲的设备中自动选择所需数量，只在本次 fresh private binding
中记录实际索引和实时 UUID。controller
会从已准入的有序策略派生各 leaf binding，而不是假设所有 native leaf 都接收同一个
pool。部分 leaf 会接收完整 pool，例如 native performance planning/execution 通过
`gpu_pool` 接入；source materialization、quantization 与 AP shard 则在 adapter contract
要求的位置接收单卡分配。同一轮内，候选/source 工作可以在不同 GPU 间并行；分配到同一
GPU 的工作会串行执行。四轮搜索本身仍按顺序执行，因为后一轮必须消费前一轮已验证的反馈。

旧的 `--local-config` 模式继续兼容。若同时设置 `GPU_POOL`，其数量必须与已有 binding
一致；自动选择设备时应使用 fresh-run 模式。

RTX4090 测量是 hardware-specific 证据。它可以验证 sm89 硬件上的端到端机制，但不能
重标记为 H800 结果，也不能通过数值调整声称复现 H800 论文表格。
当前维护实现已使用 `GPU_POOL=3` 完成一次私有 RTX4090 机制验证：四轮、16 条
selected/measured rows，并由独立 verifier 通过。该结构性事实不公开私有结果包，也不扩大
仓库的论文证据声明。

## 包含内容

```text
data/demo/                     确定性的、仅供 smoke 的 fixture
artifacts/verified/            小型、已脱敏的 Stage4 审计制品及 SHA-256 清单
framework/stage1/              模型扫描与分类契约
framework/stage4/              嵌套分组 cost-model 选择
framework/stage5/              选择与 measurement-request 契约
framework/stage6/              外部证据 paper-table adapter
framework/stage7/              仅选择的 online-ablation 契约与统计
framework/reproduction/        公开 CPU-only 论文搜索空间复现 gate
scripts/reproduce/             CPU-only smoke 与 verified-boundary 入口
tools/release/                 公开发布、私有配置与 verifier CLI
tests/                         单元、集成和发布检查
```

Smoke 工作流会调用公开的 Stage4 选择器、Stage5 selection request、Stage6
representative-selection adapter，以及一轮固定的 Stage7 selection-only；它绝不
运行设备后端，也不会声称得到实测结果。

## 范围与外部边界

发布的 Stage1–7 代码是输入校验、选择、聚合和审计逻辑。硬件 latency、energy、AP、
TVM、TensorRT、checkpoint、ONNX 或 engine 结果，只有在单独提供其 provenance、不可
变输入和已验证哈希后才可作为证据。本仓库既不会下载这些材料，也不会用合成数据替代。

随包 demo 数据不是数据集获取方式。完整数据集、模型源代码、训练 checkpoint 和硬件
系统均属外部资源，并受相应提供者条款约束。此类输入只能经显式命令接口接入，且其
provenance 必须与该匿名/公开包保持分离。

## 复现规则

冻结的 Stage4 选择 seed 为 `20260716`；Stage5 为 `20260717`；正式 Stage7 trajectory
使用 `20260718`、`20260719` 和 `20260720`。Seed 可使固定输入上的选择或分析确定，
但不会产生一次硬件运行。一次 algorithm run 是在固定输入上完整执行指定选择/分析
过程；timing iteration、编译重试、cache probe 和重复设备测量都只是外部执行中的观测，
不是额外 algorithm run。精确次数与证据状态见
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) 的 evidence matrix。

## 开发检查

可在本地执行面向发布的文档和身份检查：

```bash
pytest tests/release/test_identity_scan.py -q
```

使用 `python scripts/reproduce/reproduce_all.py --help` 查看公开 CPU 复现入口的参数。
完整测试套件可能需要额外 Python 依赖；上述 smoke 快速开始使用固定的
`requirements.txt`，然后以 `pip install --no-deps -e .` 安装当前检出目录。

## 许可证与引用

本包采用 [Apache-2.0 许可证](LICENSE)。如使用本软件，请引用
[CITATION.cff](CITATION.cff) 中的集体元数据。
