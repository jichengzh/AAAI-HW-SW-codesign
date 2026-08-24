# GEAR 协同设计复现包

本仓库提供 CPU-only smoke 工作流、小型已验证 Stage4 制品，以及用于审计投稿的
Stage1–7 选择与分析契约。仓库不打包、也不执行外部硬件测量、AP 评估、模型
checkpoint、ONNX 文件、编译 engine、TVM 或 TensorRT。

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

使用 `python scripts/reproduce/reproduce_all.py --help` 查看支持的复现参数。完整测试套件
可能需要额外 Python 依赖；上述 smoke 快速开始只依赖已声明的 `repro` 和 `dev` extras。

## 许可证与引用

本包采用 [Apache-2.0 许可证](LICENSE)。如使用本软件，请引用
[CITATION.cff](CITATION.cff) 中的集体元数据。
