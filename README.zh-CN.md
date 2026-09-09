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
设置；不把 URL 写入源码可保持匿名边界。浅克隆和 partial clone 不会下载无关历史。
`feat/runtime-gpu-pool` 是当前完整链路候选分支；稳定匿名审稿 CPU-only release 分支仍为
`release/aaai27-reproducibility`：

```bash
export GEAR_REPOSITORY_URL='<从仓库页面取得的 HTTPS 克隆 URL>'
git clone --depth 1 --filter=blob:none --single-branch \
  --branch feat/runtime-gpu-pool "$GEAR_REPOSITORY_URL" gear-codesign
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
  --ref feat/runtime-gpu-pool
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

## 外部私有 H800 / RTX4090 全链路运行

H800 与 RTX4090 共用同一条 Stage1、Stage2、GPU admission、controller 和 verifier
链路，只有 hardware profile、CUDA/TVM 架构及其外部资产不同。该路径不是 CPU smoke；
仓库也不会下载或伪造数据集、checkpoint、模型源码、ONNX/calibration、TVM cache 或
硬件测量。

材料所有权分为三类：

- **用户提供的私有输入：**私有 history root（必须是精确的 Git top-level）、填好的 v5 私有 source-map、
  pre-normalization runner-template、授权的数据集、模型源码/checkpoint、必要的 ONNX 或 calibration 输入、
  与目标 profile 匹配的 CUDA/TVM 工具链（RTX4090 需要 CUDA/TVM sm89 工具链），
  以及两个尚不存在的 normalized 目录与 fresh output root。过去文档称为 RTX local config 或 locator 的文件不再是
  用户输入。
- **仓库提供的公开参考：**H800/RTX4090 的四轮 contract 分别是
  `configs/execution/p6_h800_search.example.yaml` 与
  `configs/execution/p6_rtx4090_search.example.yaml`；单命令 manifest 模板分别是
  `configs/execution/p6_h800_full_chain.example.yaml` 与
  `configs/execution/p6_rtx4090_full_chain.example.yaml`。完整格式还可参考
  `configs/execution/p6_history_source_map.example.yaml`、
  `configs/execution/p6_history_runner_template.example.yaml` 和仅空值 schema 示例
  `configs/execution/p6_external_training_binding.example.yaml`。四份 source JSON 的格式分别
  由 `p6_gold176_rows.example.json`、`p6_gold176_graph_features.example.json`、
  `p6_closure.example.json`，以及二选一的
  `p6_h800_capability_profiles.example.json` / `p6_rtx4090_capability_context.example.json`
  给出，均位于 `configs/execution/`。这些模板均已脱敏；`template_only: true` 的 manifest
  按设计不可执行。
- **`normalize` 自动生成的私有输出：**
  `<abs-normalized-private-dir>/legacy.local.yaml`、
  `<abs-normalized-private-dir>/runner-template.yaml`、
  `<abs-normalized-private-dir>/source-wrapper-profile.yaml`、
  `<abs-normalized-private-dir>/external-training-binding.yaml`、
  `<abs-normalized-private-dir>/post-source-adapter-profile.yaml` 及派生 recipe。不要手写或
  混用这些 authority。`configs/execution/p6_h800_local_locator.example.yaml` 与
  `configs/execution/p6_rtx4090_local_locator.example.yaml` 只用于审阅生成格式，不能代替
  `normalize` 的输出。显式 `hardware_profile: h800` 使用同构 v3 locator；省略 profile 的
  历史 H800 authority 仍兼容 v2。

先把所选 full-chain manifest 复制到仓库外或 Git-ignored 位置，将 `template_only` 改成
`false`，把全部路径替换成绝对路径，并确保其中的 `hardware_profile`、source-map 与
contract 一致。静态检查会解析四份 subordinate JSON，验证其数量、身份、closure、profile
及自洽摘要；它不探测 GPU，也不创建 normalized 或运行输出：

```bash
python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-full-chain-manifest.yaml> \
  --check-inputs
```

检查通过后，H800 和 RTX4090 都从同一个推荐入口启动；`N` 只表示所需 GPU 数量：

```bash
# H800：manifest 基于 p6_h800_full_chain.example.yaml
GPU_POOL=N python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-h800-full-chain-manifest.yaml>

# RTX4090：manifest 基于 p6_rtx4090_full_chain.example.yaml
GPU_POOL=N python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-rtx4090-full-chain-manifest.yaml>
```

命令启动后会依次完成 derive、normalize、fresh provision、四轮 controller 与独立
verify；成功前不需要中途人工编辑、选卡或复制文件。任一步失败都会保留明确的阶段化
错误并停止，不会把部分结果升级为完成结果。

这些 JSON 是**格式参考，不是可执行数据**。真实 Gold 输入必须有 176 行，并为每个
`group_id` 提供一条匹配的 graph feature；两份 H800 capability profile 必须保留历史
authority 的精确摘要。RTX4090 capability context 必须由实测 probe/rebuild 流程连同内嵌
原始字节和摘要自动产生，不能照模板手填。数据集、checkpoint、配置、模型源码、ONNX
及 calibration 保持上游原生格式；v5 source-map 绑定它们的路径以及当前支持的
checkpoint/config SHA-256 身份。

成功后，`<fresh_output_root>/state.json` 是 controller 完成状态，`binding.json` 与
`local-config.yaml` 绑定本次准入环境，`round-00/feedback.json` 至
`round-03/feedback.json` 保存 16 条 released rows 及其 `latency_ms`、`energy_j`、
`ap30`、`ap50`、`ap70`。normalized runner 中
`execution_interface.actual_feedback` 指定的路径保留对应的私有原生证据和 receipts。
只有独立 verifier 输出 `completed` 报告后，才可把这些指标视为本次完成结果；verifier
证明四轮和 16 条互异测量，但不会代替研究者选“最佳”行，也不会公开私有数值。当前
dataset 仍是路径绑定的外部输入而不是 snapshot digest，因此这里证明的是全链路机制，
不是论文精确数值复现。

### 仅用于 debug 的历史四步接口

单命令入口内部仍调用 `derive_p6_history_recipe.py`、`normalize_p6_history_root.py`、历史
名称 `run_p6_h800_search.py` 和 `verify_p6_materializer_training_run.py`。只有定位阶段性失败
时才应分别运行它们；常规复现不要手工串接。第三步的完整参数形状如下，以便诊断旧日志：

```bash
GPU_POOL=7 python tools/release/run_p6_h800_search.py \
  --contract configs/execution/p6_rtx4090_search.example.yaml \
  --code-revision "$(git rev-parse --short=12 HEAD)" \
  --legacy-local-config <abs-normalized-private-dir>/legacy.local.yaml \
  --runner-template <abs-normalized-private-dir>/runner-template.yaml \
  --local-output-root <abs-fresh-output-root> \
  --binding-output <abs-fresh-output-root>/binding.json \
  --config-output <abs-fresh-output-root>/local-config.yaml \
  --source-wrapper-profile <abs-normalized-private-dir>/source-wrapper-profile.yaml \
  --external-training-binding <abs-normalized-private-dir>/external-training-binding.yaml \
  --post-source-adapter-profile <abs-normalized-private-dir>/post-source-adapter-profile.yaml
```

`--legacy-local-config` 和 `run_p6_h800_search.py` 都只是历史名称；profile authority 来自
manifest 选定的 v3 public contract，而 locator 始终来自本次 normalize。

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

H800 与 RTX4090 测量都是 hardware-specific 证据，不能跨 profile 重标记、合并或通过
数值调整互相替代。RTX4090 可以验证 sm89 端到端机制，但不能冒充 H800 论文结果。
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
