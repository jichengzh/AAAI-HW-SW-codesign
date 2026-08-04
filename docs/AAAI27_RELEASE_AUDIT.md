# AAAI 2027 双制品发布审计

## 结论摘要

本审计覆盖两种不同的制品，并明确禁止互相替代：

| 制品 | 审计对象 | 结果 |
| --- | --- | --- |
| 公开源码树 | 以 `330b003f487db69ec94db47ac2ba1270137ea273` 为基准、包含本轮 Stage4/Stage5 源身份加固的发布候选工作树 | 源树全局测试、静态检查和编译通过；在 pytest 9 环境中全局覆盖率 **82.13%**。 |
| 匿名评审 ZIP | `aaai27_code_data_anonymous.zip` | SHA-256 `f4b573e1950e9f6a3fae7a3a4f4b333ca1fc28e3d8bbfa997949b618cc881ff7`；本轮全新解包 venv 验收通过，338 tests、全局覆盖率 **80.71%**。 |

审计起点为 `5f56fcaa3e23e7a4eea51398207636a6f3ee57c4`。本文件记录发布前最终源代码树和 ZIP 哈希；最终审计提交由 Git 历史记录，不在文件中自引用其尚未生成的提交哈希。

匿名 ZIP 可用于 CPU-only 的接口、数据完整性和选择流程复现；它**不能**用于宣称已复现外部硬件、TVM/TensorRT、AP、能耗、延迟或最终 Stage6/Stage7 论文结果。

## 可复现环境与命令

本轮全新环境从最终 ZIP 解压后创建，运行时移除了 `PYTHONPATH`，并设置 `CUDA_VISIBLE_DEVICES=''`。环境为 Python 3.13.12、pip 26.2；editable 安装和 `framework` 导入均解析至该新解包目录，而非源码树或旧 venv。关键已解析版本为：

| 包 | 版本 |
| --- | --- |
| numpy | 2.5.1 |
| PyYAML | 6.0.3 |
| pydantic | 2.13.4 |
| pandas | 2.3.3 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.7.0 |
| pytest / pytest-cov | 9.1.1 / 6.3.0 |
| ruff | 0.16.0 |
| torch / torch-pruning | 2.13.0 / 1.6.1 |

执行的验收序列如下：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' python -m pip install --upgrade 'pip>=26.1.2'
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' python -m pip install -e '.[repro,dev]'
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' python scripts/reproduce/reproduce_all.py --mode smoke --output-root <safe-absolute-temp-output>
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' ruff check framework scripts tests tools
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' python -m compileall -q framework scripts tools
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

本轮发布候选源树的精确 pytest 命令退出码为 0：`338 passed`，7,783 条语句中 1,391 条未覆盖，覆盖率 **82.13%**。同一精确命令在新解包 ZIP venv 中退出码也为 0：`338 passed`，7,783 条语句中 1,501 条未覆盖，覆盖率 **80.71%**。两者均由全局 `--cov-fail-under=80` 门禁约束；差异仅记录为两个独立环境的实测覆盖率，不改变 80% 发布门槛结论。

为使声明的 `.[repro,dev]` 真正覆盖 Stage1 公共测试入口，`dev` extra 现显式包含 `torch>=2.0` 与 `torch-pruning>=1.4`，并将已知漏洞版本的 pytest 约束升级为 `pytest>=9.0.3,<10`；`scan` extra 保留相同扫描依赖。pytest 9.1.1 与 pytest-cov 6.3.0 已由完整门禁验证兼容。未使用 `omit`、`no cover`、全局 `noqa` 或合成模块来规避覆盖率门禁。

## 双制品边界

### 公开源码树

公开树保留公开 README、中文 README、引用元数据、完整文档链接和发布元数据检查。CI 增加以下强制项：

- `ruff check framework scripts tests tools`；
- `python -m compileall -q framework scripts tools`；
- 精确的全局 80% 覆盖率命令。
- 在安装依赖前显式升级 `pip>=26.1.2`。
- Stage4 与 Stage5 发布 CLI 在读取输入或写入制品前，先比较当前模块 SHA-256 与明确的期望 SHA-256；manifest 分别记录 `expected`、`current` 与仅作迁移谱系的历史 SHA-256，失配即失败关闭。

### 匿名评审 ZIP

ZIP 中 `README.anonymous.md` 单向映射为根 `README.md`。测试强制其为匿名标题，且 ZIP 中不得同时存在 `README.anonymous.md`、公开根 README 或中文 README。匿名制品严格保留：根匿名 README、`REPRODUCIBILITY.md`、`ARTIFACTS.md`、清单/校验和、可执行命令和必要测试。

允许项采用显式 allowlist，而非目录级放宽。除 `tools/release/**` 外，新增的精确闭包路径为：

```text
.github/workflows/ci.yml
.gitignore
scripts/phase2/b4_integrate.py
scripts/phase2/b5_verify_convergence.py
scripts/phase2/closedloop_objective_query.py
scripts/stage2_update_evidence.py
```

隐藏路径例外仅为两个字面路径 `.github/workflows/ci.yml` 与 `.gitignore`。回归测试确认 `.env` 和 `.github/ISSUE_TEMPLATE/*` 仍被排除。模型、检查点、ONNX、引擎、缓存、结果目录、公开身份 README、中文 README 和引用元数据不进入匿名 ZIP。

本轮最终 ZIP 有 96 个成员；归档 verifier、manifest 与 `.sha256` sidecar 一致。逐成员禁用模式扫描结果为：本地路径、代码托管 URL、邮箱、IPv4、令牌和符号链接均为 0。

## 闭包失败与修复记录

失败记录保留在此处，避免把修复后的通过结果误写成首次即成功。

| 阶段 | 发现 | 影响 | 最小修复 | 复验 |
| --- | --- | --- | --- | --- |
| 初始 ZIP 工具闭包 | ZIP 未包含自己的 archive builder/verifier 与 lint 路径。 | 解压后集成测试不能导入 release 工具。 | 仅加入 `tools/release/**`。 | archive 集成测试通过。 |
| 第一轮制品闭包 | 第二个全新环境中，17 个测试失败且覆盖率 67.61%；缺少 CI、`.gitignore`、匿名制品应有的文档契约以及三个 Stage2/Evidence CLI 路径。 | ZIP 不能作为独立审稿制品执行完整测试。 | 双制品感知测试；只加入 CI、`.gitignore`、B4、B5、evidence-update 三个精确路径。 | 第三轮环境继续执行，暴露下一条真实闭包。 |
| 第二轮制品闭包 | 第三个全新环境中，4 个 Stage2 CLI 测试失败、全局覆盖率 69.52%；`framework.run_b4_ablation` 和 `framework.search_three_arm` 都导入 `scripts.phase2.closedloop_objective_query`。 | CLI 在解析 `--output-root` 前即报 `ModuleNotFoundError`。 | 先写 RED 归档成员断言，再只加入该单一、仅依赖 `math` 的直接 helper。未加入 `scripts/phase2/**` 通配符。 | 第四个环境在旧依赖下通过；第五个环境在升级后的 pip/pytest 约束下 `332 passed`、80.65%。 |

第三个环境的 smoke 输出目录曾因使用相对路径被安全边界拒绝；改为显式绝对临时路径后 smoke 通过。这是调用参数不符合 `resolve_safe_output_root` 契约，不是 ZIP 依赖缺失。

## 论文证据边界

| 论文表/图或阶段 | 可审计证据 | 状态 | 本审计的结论 |
| --- | --- | --- | --- |
| Table 1 / Table 2 / 图编号 | 包内没有可靠的精确编号映射。 | unavailable | 不把任何描述性分析冒充为某个已完成论文表或图。 |
| Stage4 成本模型选择 | 已验证的小型 sanitized artifact：176 measurements、44 groups、5 outer folds、最多 3 inner folds、seed `20260716`。 | verified | 仅可复核该小型 Stage4 选择审计。 |
| Stage5 选择请求 | 公共 smoke 路径的确定性 demo。 | demo | `paper_evidence: false`；不得作为论文结果。 |
| Stage6 论文表适配 | 需要外部 terminal measurement、AP、能耗和独立验证记录。 | unavailable / external | 未运行硬件/AP/能耗流程，不能复现或声称 Stage6 表。 |
| Stage7 在线消融 | 正式设计要求 4 variants × 3 seeds 的 12 trajectories 与 192 selected events；正式 aggregate 未提供。 | unavailable / external | smoke 只生成 selection-only 请求，不形成实验聚合或论文结论。 |

`REPRODUCIBILITY.md` 与 `ARTIFACTS.md` 是上述状态的规范来源。外部数据、模型、权重、设备遥测、TVM/TensorRT 产物和 AP/延迟/能耗记录必须以可审阅的独立包、内容哈希、非路径 provenance、配置、seed 和独立验证状态提供，才可升级证据状态。

## 安全与供应链检查

- `git diff --check` 退出码为 0。
- 当前变更与新增文件的本地路径、邮箱、IPv4 与令牌扫描均为 0。完整 tracked 树的唯一代码托管 URL 命中位于本次新增并有意保留在公开源码树中的 `docs/AAAI27_DUAL_ARTIFACT_RELEASE_PLAN.md`；该文件被匿名 ZIP 排除，且匿名 ZIP 的逐成员代码托管 URL 扫描结果为 0。
- 共享 Conda host 的 `python -m pip_audit` 退出码为 1，报告 18 个包中的 59 项已知漏洞及若干不可审计 Conda 包。这是 host finding，不代表最终 ZIP 环境的依赖结论。
- 第四个干净 ZIP venv 的 scoped audit 曾报告 bootstrap `pip 25.3` 的 5 项漏洞以及 direct dev dependency `pytest 8.4.2` 的 `PYSEC-2026-1845`（修复版本 9.0.3）。这是已修复的历史 finding：CI 和验收命令先升级 `pip>=26.1.2`，开发依赖改为 `pytest>=9.0.3,<10`，并由 pytest 9.1.1 完整门禁验证。
- 对本轮干净 ZIP venv 执行 `pip-audit --path <venv-site-packages>`：`--strict` 退出 1 的唯一原因是 editable `gear-codesign` 未发布到 PyPI、因而不可审计；`--skip-editable` 退出 0，输出 `No known vulnerabilities found`。已解析的 direct 与 transitive declared dependencies 均未报告已知漏洞。

因此，当前 scoped declared-dependency audit **不阻断**匿名 ZIP 的 CPU runtime/repro 验收，也没有已知开发依赖漏洞。CI 尚未把 pip-audit 设为持续门禁；这是低优先级后续改进项，不在本次将审计工具加入开发 extra，避免形成审计工具自依赖。

## 交接状态与完整开源台账

本节是本仓库唯一的**持续交接台账**。它把最初项目状态、当前可验证状态、未完成工作和发布授权边界放在同一处；其他文档分别承担证据定义、制品清单或历史实施细节，不能替代本节。

### 使用与更新规则

- 每完成一项计划，必须在**同一提交**中先更新本节的状态、可验证证据、影响范围和下一项，再提交代码、配置或文档；只改变代码而不更新台账视为该项未完成。
- 每次更新保留一条追加式变更记录，写明日期、计划编号、完成内容、提交（或“待提交”）、实际命令/结果和后续状态。发现先前记录错误时追加更正，不重写历史结论。
- 同一时刻只有一个计划项可以标为“进行中”。受权限、数据许可、硬件访问或论文匿名期限制的工作标为“阻塞”，并写明解除条件；不得把“暂未执行”写成“已验证”。
- 本台账和提交信息**不记录**密码、令牌、私有 IP、绝对本机路径、个人邮箱、未脱敏数据内容或可直接访问的受限资源地址。它只记录可公开的类别、哈希、相对路径、命令和非敏感状态。
- 发布前由维护者按本节的完成定义逐项复核。任何推送、合并、仓库可见性变更、tag 或 GitHub Release 都是外部动作，仍须单独得到明确授权。

### 起始状态

公开代码工作的可追溯起点是 2026-06-24 的 Stage1 model scanner。随后建立本发布分支时，项目缺少统一的依赖元数据、许可证、测试/CI 闭环和匿名制品导出器；完整实验、模型物化与硬件执行仍位于独立的私有工作仓库，不能直接复制到发布树。

2026-07-29 至 2026-07-30 的实现把公开树整理为可测试的 Stage1--Stage7 接口、确定性 demo、已验证的小型 Stage4 审计、证据边界、匿名 ZIP 构建器和发布审计。最初实施范围、源码迁移谱系与历史基线见 [双制品实施计划](AAAI27_DUAL_ARTIFACT_RELEASE_PLAN.md)；该计划是历史设计依据，不是当前进度台账。

从起点起始终有效的边界如下：模型 checkpoint、ONNX、编译 engine、TVM/TensorRT 产物、缓存、可再生成结果、原始硬件日志和未经许可的数据均不入库。若某项实验需要它们，公开仓库只能提供合法获取说明、版本、SHA-256、配置与运行接口，而不能用 demo 或替代数值伪装为原实验。

### 当前状态

**状态日期：2026-08-04；此处记录 P1 最终候选冻结时的已验证工作树状态。** 当前工作分支为 `release/aaai27-reproducibility`。P1 最终候选为 `c65b5fc7eb9d20e2d34928f7c948aea90f822754`，tree 为 `e9adfe9888a21ae0b5f819858e1cea001acbcecb`；冻结时相对配置的远端发布分支为 behind 0、ahead 8，工作树干净且尚未推送。P1 结项文档提交不属于该已验收候选；其发生后应以 `git status --short --branch` 取得实时差异，不能把这里的冻结数字误作实时远端状态。

| 范围 | 当前状态 | 可验证证据 | 仍缺少的内容 |
| --- | --- | --- | --- |
| 公开 CPU 冒烟闭环 | 已完成，并经 P1 重验 | `requirements.txt` 固定 CPU 依赖；`scripts/reproduce/smoke_clean_clone.sh` 创建全新 clone/venv 后运行 smoke；P1 候选上实际通过 27 项 clean-clone 检查。 | 真实公开远端在推送后仍需由 CI 再验证。 |
| 公开文档与新手入口 | 已完成 | `README.md`、`README.zh-CN.md`、`REPRODUCIBILITY.md` 说明浅克隆、固定依赖和 smoke；匿名 README 使用同一依赖入口。 | 全量训练/硬件运行说明尚未迁入。 |
| 匿名审稿 ZIP | P1 本地验收完成 | allowlist、逐字节安全扫描、archive verifier 和解包后的新 venv 已验收；P1 ZIP 为 100 个成员。 | 每次候选 HEAD 改变都必须重建并记录新哈希。 |
| 论文证据 | 受限 | 小型 Stage4 审计为 `verified`；demo 明确为非论文证据。规范定义见 [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) 与 [ARTIFACTS.md](../ARTIFACTS.md)。 | Stage6/Stage7 及真实硬件/AP/能耗结果仍为 `external` 或 `unavailable`。 |
| 完整项目源代码 | P3 进行中 | 当前树含公开的接口、验证、选择和聚合逻辑；P3-1 已补入 Stage6 纯 Python 合约层，P3-3 已补入不读取或回显外部位置、已复审的 Stage4 闭环审计，P3-5 已补入 Stage2 脱敏延迟异常策略。 | 私有完整训练、模型物化、硬件调度/测量和正式实验执行代码尚未逐项脱敏、迁入并验收。 |
| 私有源迁移清单 | P2a + P2b 本地只读盘点完成 | [P2 脱敏汇总](release-manifests/P2_PRIVATE_SOURCE_SUMMARY.json) 与 v1.1.0 inventory 工具已记录 1,551 个候选及复核/排除分类；P2b 重新盘点替代旧候选计数。 | P3 必须逐批完成语义、许可与依赖审阅后才迁移任何候选；当前清单不是自动上传许可。 |
| 公开发布 | 未开始 | P1 本地发布候选已冻结并验收；P2 私有源盘点已完成；尚未执行外部推送或可见性变更。 | 需要完成 P3--P8，并在最后取得推送/发布的明确授权。 |

历史的 `338 passed`、覆盖率和 ZIP SHA-256 记录仍是其对应审计提交的证据，不能自动外推到当前 HEAD。当前 HEAD 的每一次新验收结果必须通过本台账追加记录。

### 到完整开源的执行计划

| 编号 | 状态 | 操作 | 交付物与验收条件 |
| --- | --- | --- | --- |
| P0 | 已完成 | 建立本交接台账及其回归测试，并从 README 入口链接；修复本轮真实匿名 ZIP 闭包发现。 | 同一提交包含台账、链接、测试与 CI 修复；346 passed，真实 ZIP build/verify 通过。 |
| P1 | 已完成（本地） | 冻结当前发布候选，重建匿名 ZIP，并对候选源树和解包 ZIP 分别做干净环境验收。 | 记录候选提交、ZIP SHA-256、成员数、依赖版本、测试数、覆盖率、identity scan、clean-clone 结果；旧审计结果不能复用。 |
| P2 | 已完成（P2a + P2b 本地只读盘点） | 审计私有完整仓库：按“源码/文档/小型脱敏 fixture/外部数据描述/禁止上传生成物”清点全部路径和依赖关系；P2b 加固嵌套数据边界和稳定读取后重新盘点。 | 机器可读的机械分类清单和敏感项扫描结果；每个保留、改写、外置或排除项的语义理由、许可证/许可状态在 P3 逐批批准时补齐。 |
| P3 | 进行中（发布级逐项复审重启） | 逐模块脱敏迁移完整 Python、配置和设计 Markdown；将测试所需的小型固定输入迁入受版本控制的 fixture 目录，不再从 `results/` 读取。 | P3-1、P3-3 和 P3-5 是保留的实现历史；P3-2、P3-4 和 P3-6 未批准直接迁入；P3-7 已建立关闭账本工具。此前 P3-13--P3-17 以机械分流得出的 1,551 项结论和全量 ledger 已撤回，不能作为 P4 或发布依据；其他历史 P3 批次的计数也不计入新的发布级账本。P3-18--P3-30 已按新协议复核 145 项；P3-32 已完成主审但待独立复审，尚未计入。P3-26/P3-27 的冲突本地 records 已通过保留旧记录的 supersession manifest 解决，规范 records 已重新验证并重建账本。新的逐项职责、依赖、许可证和安全输出复审见 [P3 重审协议](release-manifests/P3_REAUDIT_PROTOCOL.md) 与各批次记录。 |
| P4 | 待开始 | 为完整实验的数据、模型和可再生成大文件建立外部获取清单和验证接口。 | 每个外部输入记录许可、获取方式、版本、SHA-256、大小、用途、预期目录和缺失时的失败信息；默认命令不下载数据/权重。 |
| P5 | 待开始 | 固化可复现环境：CPU 基线继续保持；分别提供 4090 与 H800 的环境/驱动/CUDA 约束、硬件探测和最小运行命令。 | CPU、4090、H800 配置文件互不混淆；不含 SSH 信息；每个环境均可执行依赖检查和相应的最小测试。 |
| P6 | 待开始 | 接入全流程训练、评测、硬件执行与结果汇总；生成不可伪造的执行 manifest。 | 每一步消费的输入、代码版本、随机种子、配置和输出 SHA-256 可追溯；缺失外部证据失败关闭；Stage6/Stage7 只在正式数据及独立验证满足后更新证据状态。 |
| P7 | 待开始 | 完成开源前安全、合规、文档和供应链审查。 | 全树/待发布历史凭据扫描为零；第三方许可和数据使用权明确；README、架构/设计说明、数据卡、复现指南、贡献/引用信息与真实入口一致；依赖漏洞处置或记录完成。 |
| P8 | 待开始 | 形成发布候选并进行外部发布。 | 新目录 HTTPS clone、CPU smoke、全量测试、CI、匿名包（如仍在匿名期）和 4090/H800 最小验证均通过；获得明确授权后才推送、PR/合并、打 tag、设置可见性和发布版本。 |

#### P1：发布候选冻结

1. 从当前工作分支创建只读候选提交，确认工作树只含已审查文件。
2. 用空输出目录构建匿名 ZIP；使用独立 verifier 复核 ZIP、manifest、sidecar、成员数和扫描结果。
3. 在新 clone/new venv 中分别运行 CPU smoke、`ruff`、`compileall`、全量 pytest 与覆盖率门槛；另执行 clean-clone 脚本。
4. 将实际命令、版本、提交和每项退出状态追加到本台账；任一失败则记录问题并回到相应计划项修复。

#### P1 完成证据：2026-08-04 本地候选

候选为 `c65b5fc7eb9d20e2d34928f7c948aea90f822754`（tree
`e9adfe9888a21ae0b5f819858e1cea001acbcecb`）。它由一个新的 shallow local
clone 产生；冻结时 `git diff --check` 为 0，工作树干净，分支相对远端为
behind 0、ahead 8。整个 P1 均未推送、创建 PR、打 tag、修改可见性或访问真实
GPU/硬件/外部实验数据。

源树在新的 Python 3.13.12 venv 中验收。运行时为 pip 26.2、NumPy 2.2.5、
PyYAML 6.0.2、pydantic 2.11.4、pandas 2.2.3、SciPy 1.15.3、scikit-learn
1.6.1、LightGBM 4.6.0、pytest 9.0.3、pytest-cov 6.3.0、Ruff 0.16.1、CPU
PyTorch 2.9.0、torch-pruning 1.6.1 与 build 1.5.0；`pip check` 无损坏依赖。
CPU smoke 的 manifest 为 `aaai27_reproduction_run_manifest_v1`、`completed`、
`paper_evidence: false`，含 6 个 stage、34 个输出和 3 个冻结 seed。包 wheel/sdist
构建、Ruff、compileall 和 source 全量覆盖率门禁均通过：**347 passed，80.71%**。
同一新环境执行 clean-clone 脚本通过：**27 passed**，输出 commit 与候选一致。
本地路径 clone 对 `--depth`/`--filter` 只发出 Git 的预期本地 clone 警告；它不影响
clone 内容或验收结果。公开远端 HTTP(S) clone 的传输行为仍由推送后的 CI 复验。

匿名 ZIP 使用 `anonymous-archive-builder/1.0.0` 从同一候选生成，并经独立 verifier
验证。ZIP 有 100 个成员，manifest 亦列 100 个成员，SHA-256 为
`aa83bc98b4ada4f7865a5db0ee0c12c92215599b50d9734ff3f123c6833cf4b7`。解包目录
不含 `.git` 和 `README.anonymous.md`，根 `README.md` 正确映射为匿名入口；第二个全新
venv 的 `framework` 导入路径确认来自该解包目录。该环境的 smoke manifest 与源树摘要
相同，wheel/sdist、Ruff、compileall 和覆盖率门禁均通过：**344 passed, 3 skipped，
80.71%**。3 项跳过均是公开交接/公开 README 专用测试，在匿名 ZIP 中按制品边界显式
跳过；没有跳过 smoke、归档验证、身份扫描或实验代码测试。

安全检查中，匿名 archive builder/verifier 的逐成员 allowlist 扫描通过，故其
local-path/github-url/email/ipv4/token/unknown-binary 类别均为 0。候选完整 tracked
源码树的同一模式扫描结果为 `release_tree_local-path=1` 与
`release_tree_github-url=1`，其余 email/ipv4/token/unknown-binary 均为 0：前者位于
`docs/superpowers/plans/2026-07-30-scoped-reproducibility-disclosure.md`，后者位于
`docs/AAAI27_DUAL_ARTIFACT_RELEASE_PLAN.md`。两份文件不进入匿名 ZIP；但在 P7 对公开
源码树脱敏或作出明确公开豁免前，本仓库不得宣称已可公开开源。

#### P2--P4：从私有工作树安全迁移

1. 先生成清单，再迁移；不对私有树执行批量提交、历史重写或上传。清单必须将 Markdown/Python 与 PNG/PDF、LaTex 临时文件、压缩包、模型、engine、ONNX、校准缓存和结果目录分开处理。
2. 每个候选文本文件先经凭据、IP、绝对路径、邮箱、私有域名和受限数据扫描；命中后改用环境变量示例、抽象路径或脱敏 fixture，不能仅靠 `.gitignore` 掩盖已跟踪内容。
3. 每个可生成大文件只保留生成脚本、最小参数和输入/输出 SHA-256；每个不可再生成但合法可分发的输入必须先确认许可，之后才可建立下载/校验清单。
4. 新增设计 Markdown 必须说明模块目的、关键决策、输入输出契约、复现命令、已知边界和与论文/实验阶段的关系。

#### P2a 与 P2b 完成证据：2026-08-04 私有源只读盘点

P2a 在私有完整工作树上执行只读机械盘点；没有修改、复制、压缩、上传或提交该工作树中的任何文件。盘点时主工作树有 5,127 个 tracked 文件、88 个未跟踪文件待考虑、19 个已跟踪修改；另识别出两个独立嵌套 Git 仓库。它们在 P3 中必须保持仓库边界，不能作为普通子目录盲目合并。

随后独立复审发现 P2a 工具的嵌套数据边界与稳定读取仍有缺口，因此旧报告只能作为历史证据，不能再用于选择 P3 候选。P2b 以 [安全加固记录](release-manifests/P2B_INVENTORY_SAFETY_HARDENING.md) 中的 v1.1.0 工具重跑只读盘点；原始报告只在临时目录短暂保存并已清理，没有修改或上传私有树内容。

工具为 `tools/release/inventory_private_source.py`（`private-source-inventory/1.1.0`）。它只接受 Git 根目录；输出必须在该工作树外，文本候选最大读取 1 MiB，复用匿名制品的路径/内容禁用模式扫描。精确数据目录段在任意路径层级均会排除，而 `tests` 源码目录不会误伤；候选读取逐层禁止跟随符号链接，拒绝硬链接，并在读取前后核验普通文件的设备、inode、mode、link-count、size、mtime 和 ctime。通过扫描的候选才写出相对路径；模型、结果、数据、缓存、生成物、符号链接、脚本、超大文本和命中禁用模式的条目只以分类、大小和不可逆 ID 记录。该工具还拒绝将输出写回私有工作树、拒绝覆盖竞争输出，且失败信息不回显私有路径，因此不会把盘点报告重新纳入待扫描集合。

复查或开始每一批 P3 迁移前，均从公开仓库根目录运行：`python tools/release/inventory_private_source.py --source-root <private-git-root> --source-label <opaque-label> --output <directory-outside-private-worktree>/P2_PRIVATE_SOURCE_INVENTORY.json`。原始 JSON 仅在受控本地位置短期保存，每批迁移核对完成后应清理；提交前只更新其中的脱敏计数、哈希与本节证据。

P2a 的历史原始 JSON 为 992,298 bytes，SHA-256 为 `819207c036a35c7a4d8d61f7628aaaf1ea4ec9cc8b87186d6f6a18891e9057ad`。P2b 的替代原始 JSON 为 992,291 bytes，SHA-256 为 `6fafdace7dc0eae3e9982763cb1c5c098c0ae83d1715af67ff1e08d7a1d12a79`。两份原始逐文件报告均不提交：即使通过禁用模式扫描，完整私有目录结构在 P3 逐批批准前也不应扩大公开暴露面。其可再生成命令由上述工具提供；提交的是不含路径/内容的 [P2 脱敏汇总](release-manifests/P2_PRIVATE_SOURCE_SUMMARY.json)，其中绑定当前原始报告哈希和分类计数。

P2b 盘点得到 1,551 个候选：1,008 `migrate_code`、96 `migrate_config`、439 `migrate_document` 和 8 `fixture_candidate`。另有 3,664 个不透明条目：1,108 `exclude_generated`、1,884 `external_input`、75 `review_other`、31 `review_paper_source`、455 `review_sensitive` 和 111 `review_shell`。候选并非自动迁移许可；P3 每一批都必须复扫、审阅许可与依赖，并将最终“保留/改写/外置/排除”决定写回台账。`exclude_generated` 永不直接复制，外部输入由 P4 处理。

私有树规模约 242 GiB，包含 checkpoint、模型导出、engine、训练/评测输出、缓存和数据目录等十 GB 级边界。盘点不读取这些二进制内容，也不把它们计入候选。文件名预审还发现敏感候选、特殊文件名和符号链接；它们均进入不透明人工复核，不能通过普通 shell 通配符、压缩包或批量复制迁移。

P2a 对公开发布树的回归结果为 `350 passed`；P2 工具定向、台账和匿名 archive 集成测试为 `35 passed`，Ruff、`compileall` 与 diff 空白检查通过。当前工作树构建的匿名 ZIP 已独立 verify：102 个成员，SHA-256 为 `9cb20e3716958c1f0a69b2f03e4431915d48b18e688bf509efce2170a65f0faa`。`pip-audit -r requirements.txt` 未发现已知漏洞；带 `+cpu` 本地版本标记的 PyTorch 不在 PyPI 审计索引中，因此被工具报告为未审计，须保留为 P7 的已知供应链边界。

#### P3-1：Stage6 纯 Python 合约层（已完成，本地）

P3 首批选择三个没有网络、GPU、子进程或隐式文件 I/O 的 Stage6 源模块；对应设计、脱敏和后续依赖决定在 [P3-1 批次记录](release-manifests/P3_BATCH_01_STAGE6_CONTRACTS.md) 中维护。原始来源的固定生成结果位置已改为结构化外部证据描述符，因而公开代码不再从 `results/` 或维护者目录发现输入。审查后，候选选择输出改为字段白名单，两后端源证据哈希漂移和畸形合约输入均失败关闭。测试先以缺少模块失败，再在迁移实现后通过；本地全量回归为 `366 passed`，Ruff、`compileall`、匿名 ZIP build/verify 与禁用模式扫描均通过。当前 ZIP 有 109 个成员，SHA-256 为 `72b8f3c813fd29407044c863c62d66f50ae7c8a4296ae43727bdce1ea75f1b9f`。本批次未推送、未发布，P3 的其余模块仍待审阅。

#### P3-2：Stage5 候选审阅（未批准迁移）

P3-2 重新审阅四个 Stage5 候选，没有复制任何私有文件。源注册表和测量计划会处理外部证据、checkpoint 或结果根，故保留给 P4--P6；`closure_v3` 虽为内存聚合器，却会将调用者输入中的证据/结果位置写回报告。为防止私有路径回流，本批次不迁移实现，必须先按 [P3-2 审阅记录](release-manifests/P3_BATCH_02_STAGE5_AUDIT.md) 将其改为制品 ID、SHA-256 与可用性状态的显式契约，并补失败关闭测试。此拒绝是 P3 的安全结论，不是发布障碍的忽略或自动豁免。

#### P3-3：Stage4 纯内存闭环审计（已完成，本地）

P3-3 迁入没有 I/O、网络、GPU 或子进程依赖的 Stage4 闭环审计器。它只处理传入内存的协议汇总、逻辑组标识和 SHA-256 一致性，不读取或输出证据/结果位置。首轮审查发现 `after_feedback` 行未逐条绑定到合成测量组、采集指标允许负值/零分母，且无效指标会回显原始值；相应失败关闭测试和实现均已补齐。最终复审未发现 blocker；全量回归为 `378 passed`、总覆盖率 81.52%、新模块覆盖率 90%，Ruff、`compileall` 和 diff 空白检查通过。匿名 ZIP 已独立 build/verify（111 成员，SHA-256 `aca6c8426e5f1d413586e18e315dc4655fcc5446d90cd494d0643d1e28cea32c`）；详细范围和排除见 [P3-3 批次记录](release-manifests/P3_BATCH_03_STAGE4_CLOSURE_AUDIT.md)。本批次未推送或发布，P3 其余模块仍待逐批审阅。

#### P3-4：Stage2 S1 探针指标候选审阅（未批准迁移）

P3-4 审阅一个无 I/O 的 Stage2 结构探针指标模块。它会拒绝性能标签，但仍会在报告和异常中回显调用者提供的探针标识与特征名。为防止私有标识或位置经聚合报告回流，未复制实现或测试；必须先按 [P3-4 审阅记录](release-manifests/P3_BATCH_04_STAGE2_S1_PROBE_METRICS_AUDIT.md) 改为公开字段白名单/位置索引、固定错误代码和畸形输入失败关闭，才能在新批次重新审阅。

#### P3-5：Stage2 脱敏延迟异常策略（已完成，本地）

P3-5 将一个只处理内存测量字典的 Stage2 延迟质量策略重写为公开契约。输出只有稳定的 `row_index`、固定质量状态和原因代码；调用方提供的行、运行、配置、模型、候选、调度、路径和 URI 信息绝不回显。分组字段必须均为非空字符串，缺失、空白或非字符串分组只能失败关闭；有限正数边界也安全处理超大数值。全量回归为 `386 passed`、总覆盖率 81.72%、新模块覆盖率 93%，独立复审通过；匿名 ZIP 已独立 build/verify（113 成员，SHA-256 `6d821d88ab4915a586192b0e38592dfea22c51a8541fef194487576f0a3196b4`）。详细范围和排除见 [P3-5 批次记录](release-manifests/P3_BATCH_05_STAGE2_LATENCY_OUTLIER_POLICY.md)。本批次未推送、未发布，P3 仍须逐批完成其余候选的处置。

#### P3-6：下一批候选只读审阅（未批准直接迁入）

P3-6 审阅三个最小候选而未复制任何私有内容。一个纯内存冷启动筛选器会回显调用方 label 且含历史实验叙述，必须改为中性说明和匿名投影；一个 Stage6 请求构造器会深拷贝调用方行和固定实验标签，必须先定义匿名行 schema；已有的 Stage6 协议 smoke 不重复迁入。完整决定见 [P3-6 审阅记录](release-manifests/P3_BATCH_06_NEXT_CANDIDATE_AUDIT.md)。

#### P3-7：关闭验证与远端更新判定（P3 未关闭）

P3-7 新增公开、路径无关的 disposition ledger 生成器和回归测试，用于把本地私有 inventory/decisions 文件转换为只含 HMAC-SHA256 候选 ID、P2 分类、disposition、固定 reason/evidence token 和后续阶段的公开账本。工具拒绝不完整 decisions、重复候选、非法 disposition、符号链接、硬链接、组/他人可读 HMAC key、已存在或不安全输出路径，错误信息不回显私有路径。定向验证为 18 passed，Ruff、`compileall` 和 `git diff --check` 通过。

本轮关闭验证的硬结论是：尚未存在覆盖 P2b 1,551 个候选的本地逐路径 decisions 文件，也未生成可提交的公开 disposition ledger。因此 P3 仍未完成，当前 HEAD 不能作为完整可复现公开版本远端更新。阶段性分支同步也只能在维护者明确授权后执行，且不能被称为开源发布。详见 [P3-7 关闭验证记录](release-manifests/P3_BATCH_07_CLOSURE_VERIFICATION.md)。

#### P3-8：候选处置启动审计（已完成，本地）

P3-8 重新生成 P2b 私有 inventory 并核对 SHA-256，确认 1,551 个 P3 候选和 3,664 个不透明条目均未漂移。随后将候选与当前公开仓库做同路径哈希比对：9 个候选同路径且内容一致，38 个同路径但内容不同，1,504 个当前无公开同路径。首轮无路径功能审阅还将 26 个 Stage4 代码候选分为 19 个需脱敏改写的纯内存功能候选、3 个公开等价候选和 4 个待单独审阅候选。该结果说明 P3 不能依赖已有公开树自动关闭，必须通过本地 decisions 和公开脱敏 ledger 完成逐项处置。

本批次没有迁入源码或生成公开 ledger，只固定了后续 decisions 的 `reason`/`evidence` token 口径、必须使用 `blocked_license_or_permission` 的许可/隐私情形，以及第一批 9 个同内容代码候选的账本演练范围。完整规则见 [P3-8 候选处置启动审计](release-manifests/P3_BATCH_08_CANDIDATE_TRIAGE_AUDIT.md)。下一步需要在公开仓库外维护 HMAC key 和本地 decisions 文件；完成 9 项演练仍不能关闭 P3，最终仍需覆盖全部 1,551 个候选。

#### P3-9：Stage4 纯内存脱敏改写（四个单元完成）

P3-9 已从 P2b 清单派生 9 项仅本机保存的账本演练子集。它们均为当前公开仓库已跟踪、同路径且同 SHA-256 的代码；使用受限 HMAC key 生成 9 条路径无关 ledger 记录，验证通过。该演练只验证账本机制，不能替代完整 P3 ledger，也不会提交 private inventory、decisions、key 或演练输出。

随后按测试优先方法完成四个 Stage4 纯内存模块：`feedback_update_eval_v1`、`selection_completion_v1`、`ranking_pareto_v1` 与 `uncertainty_replay_v1`。它们只接受白名单内存字段；匿名组 ID 限定为 `g<number>`，四臂和 fold/manifest 均严格绑定；不得复制私有实现或接入文件、网络、GPU、子进程、模型、结果或硬件输入。ranking/Pareto 的二次算法另限制为最多 256 组/1,024 候选。四个模块的定向和 closure-audit 回归为 91 passed，完整 Stage4 回归为 119 passed，单元覆盖率依次为 89%、88%、90% 与 90%。本批候选的全仓验收为 502 passed、86% coverage；交接/身份/账本回归为 35 passed；匿名 ZIP build/verify 为 124 个成员、SHA-256 `a6951ed6310541a0266a14d6af93228fe20313f93f41508231b773e54644d264`。Ruff、`compileall`、diff 空白检查通过，依赖审计未发现已知漏洞（PyPI 不提供 CPU 专用 `torch 2.9.0+cpu` 的审计记录）。完整进度见 [P3-9 批次记录](release-manifests/P3_BATCH_09_STAGE4_REWRITE.md)。

#### P3-10：同路径差异候选逐项审计（已完成，本地）

以当前公开工作树复核后，P3-8 中“同路径但内容不同”的 38 项已变为 39 项，说明公开树在 P3-8 后继续演进，旧计数不能外推。39 项均逐项审阅：9 项由当前公开、已测试的安全契约替代，13 项仍需先完成公开改写，2 项仍需来源/许可证依据，15 项仍需逐项确认安全改写、外部契约、排除或阻塞结论。只有前 9 项形成 P3-10 的本地 `duplicate_or_superseded` 子集，采用 `public_contract_supersedes` 与 `public_contract_review` 固定 token；它与 P3-9 的 9 项同内容子集合计仅为 18/1,551 项 decisions。私有 inventory、decisions、HMAC key 和生成 ledger 均未提交。详见 [P3-10 审计](release-manifests/P3_BATCH_10_DIVERGENT_CANDIDATE_AUDIT.md)。

#### P3-11：小型 fixture 逐项处置（已完成，本地）

8 项小型 YAML/JSON fixture 均单独审阅。7 项 synthetic Stage1/Stage2 测试输入已被当前公开测试和 demo 生成契约替代，形成 `duplicate_or_superseded`；1 项仅描述外部 checkpoint/dataset 获取和校验的 manifest 形成 `external_contract_p4`。相应公开 Stage1/Stage2 工作流回归为 36 passed。本机 ledger 检查点含 8 条路径无关记录且泄露扫描干净；累计 26/1,551 项有可复核 decisions。详见 [P3-11 记录](release-manifests/P3_BATCH_11_FIXTURE_DISPOSITION.md)。

#### P3-12：配置候选逐项处置（已完成，本地）

96 项 JSON/YAML 配置逐项验证后，86 项量化/剪枝实验参数转入 P6 execution contract，3 项硬件/环境描述转入 P5，6 项外部数据转换元数据转入 P4，1 项内部代理运行复盘排除。所有本地 ledger 记录均为路径无关 HMAC 条目且通过泄露扫描；累计 122/1,551 项有可复核 decisions。详见 [P3-12 记录](release-manifests/P3_BATCH_12_CONFIGURATION_DISPOSITION.md)。

#### P3：发布级重审重置（进行中）

此前 P3-13--P3-17 在短时间内以静态信号、目录和关键词做批量分流；它们只证明了机械计数与路径无关 ledger 的结构，不足以证明每项职责、调用关系、公开替代物、来源/许可证或安全输出结论。相应文件和全量 ledger 已从当前分支撤回，早期结论不用于 P4、P5、P6、P7 或发布判断。P2b 基线保持不变，真实复审协议与每批最小证据见 [P3 重审协议](release-manifests/P3_REAUDIT_PROTOCOL.md)。

#### P3-18：合成 fixture 发布级复审（已完成，本地）

重审按 P3 协议从 8 个维护者授权的小型合成 fixture 开始。7 项经内容、用途、公开消费者、来源授权和安全输出复核后原样迁入；1 项含“measured”文件名的布尔 anchor 被增加合成/非论文证据元数据后迁入。新增测试先失败、后通过；Stage1 与 release 安全回归为 79 passed。该批只是公开测试输入迁移，不提供真实外部制品，P4 尚未开始。详见 [P3-18 记录](release-manifests/P3_BATCH_18_SYNTHETIC_FIXTURE_REAUDIT.md)。

#### P3-19：外部边界发布级复审（已完成，本地）

按真实职责而非关键词复核 5 项直接涉及外部制品、环境或测量执行的候选：一项分别进入 P4、P5、P6 工作链，一项由已有公开 capability parser/测试替代，一项因用户专属运行时路径而阻塞。没有迁入任何源码或真实制品；P4 尚未开始。5 项均有受限的内容 SHA-256、调用关系、来源/许可证和安全输出记录，私有逻辑回归为 52 passed，公开替代契约为 6 passed。新的发布级 re-audit 目前只有 13/1,551 项可接受 decisions。详见 [P3-19 记录](release-manifests/P3_BATCH_19_EXTERNAL_BOUNDARY_REAUDIT.md)。

#### P3-20：正式执行与私有测试边界复审（已完成，本地）

12 项候选逐项完成职责、导入/调用、公开缺口、来源和安全输出审查：8 项进入 P6，1 项进入 P4，3 项因用户专属挂载路径阻塞。它们没有被执行，也没有迁入公开树；这是避免运行真实模型、数据、GPU、运行时或私有制品的安全边界。独立抽样复审已覆盖 P4、P6 和阻塞三类并通过；新的 re-audit 累计仅 25/1,551 项。详见 [P3-20 记录](release-manifests/P3_BATCH_20_FORMAL_EXECUTION_REAUDIT.md)。

#### P3-21：外部执行脚本复审（已完成，本地）

12 个非测试执行脚本已逐项审阅：7 项进入 P6，3 项进入 P4，2 项因用户专属仓库根路径阻塞。一个 token-like 静态命中已依据上下文排除为凭据误报。未运行任何私有执行器或真实制品，未迁入公开树；独立抽样复审覆盖三种 disposition 并通过，新的 re-audit 累计仅 37/1,551 项。详见 [P3-21 记录](release-manifests/P3_BATCH_21_EXTERNAL_EXECUTION_REAUDIT.md)。

#### P3-22：运行时编排与上游配置复审（已完成，本地）

12 项候选逐项审阅后，9 项进入 P6、1 项进入 P4、2 项因上游来源许可/NOTICE 不明阻塞。一个 token 静态匹配经上下文确认是进程参数而不是凭据。未运行私有运行时或真实制品，未迁入公开树；独立抽样复审覆盖三种 disposition 并通过，新的 re-audit 累计仅 49/1,551 项。详见 [P3-22 记录](release-manifests/P3_BATCH_22_RUNTIME_AND_UPSTREAM_REAUDIT.md)。

#### P3-23：上游配置与外部物化复审（已完成，本地）

12 项候选按当前内容逐项审阅：8 项上游协同感知配置具有 Apache-2.0 代码来源依据，但数据集与 checkpoint 的再分发条件尚须由 P4 明确，因此均进入 P4 外部配置契约；4 项校准、保留集特征、测量行执行和模型导出工具均需外部数据、模型、设备或生成物，均进入 P6。未运行执行器或迁入源码；公开树中没有把这些外部行为伪装成可默认运行的等价物。受限证据已再次核对到当前源码，新的 re-audit 累计仅 61/1,551 项，P4 尚未开始。详见 [P3-23 记录](release-manifests/P3_BATCH_23_UPSTREAM_CONFIG_REAUDIT.md)。

#### P3-24：Stage2 制品与执行边界复审（已完成，本地）

12 项候选逐项审阅后，5 项只应在路径脱敏、依赖补齐和固定输入测试后作为 P4 证据/覆盖契约处理，3 项属于 P6 外部执行或探测契约，4 项因未受控动态解析、私有运行根或派生模型/制品权限而保持 P7 阻塞。没有运行数据、模型、GPU、编译产物或执行器，亦没有迁入公开树；受限记录已核对到当前源码。新的 re-audit 累计仅 73/1,551 项，P4 尚未开始。详见 [P3-24 记录](release-manifests/P3_BATCH_24_STAGE2_ARTIFACT_EXECUTION_REAUDIT.md)。

#### P3-25：TRT、环境清单与遗留入口复审（已完成，本地）

12 项候选逐项审阅后，3 项进入 P4 安全证据/验证契约，2 项进入 P5 环境/清单契约，4 项进入 P6 外部执行契约，2 项因私有根或模型资产许可保持 P7 阻塞，1 项已有公开 Stage4 契约替代。4 项在入队快照后发生内容变化；它们不继承旧队列摘要，而是按当前内容重新读取、绑定受限 SHA-256、复查调用关系和安全输出后才形成决定。未运行任何外部执行器或真实制品，未迁入公开树；新的 re-audit 累计仅 85/1,551 项，P4 尚未开始。详见 [P3-25 记录](release-manifests/P3_BATCH_25_TRT_RUNTIME_AND_LEGACY_REAUDIT.md)。

#### P3-26：公开替代与执行边界复审（已完成，本地）

12 项候选已完成独立逐项复审：5 项已有公开同路径实现或更安全替代并由 Stage1/demo/smoke 测试覆盖，1 项进入 P4 外部测量行/制品注册表契约，6 项进入 P6 ONNX/TVM/TRT、补全队列、源物化、反馈回合或实际执行支持契约。当前内容漂移为 0；未运行执行器、未迁入源码或真实制品。冲突的本地 records 被保留、标为 superseded，唯一规范 records 已重新核验并重建账本；该批接受时累计为 133/1,551，后续接受批次使当前 re-audit 为 181/1,551 项，P4 尚未开始。详见 [P3-26 记录](release-manifests/P3_BATCH_26_PUBLIC_STAGE_AND_EXECUTION_REAUDIT.md)。

#### P3-27：旧搜索基础模块与真实 ablation 边界复审（已完成，本地）

12 项候选已完成独立逐项复审：10 项由公开同路径模块或公开 Stage2/5/6/7 契约替代，1 项进入 P4 上游模型变体 baseline/映射 adapter 契约，1 项真实 PQS ablation runner 进入 P6。当前内容漂移为 0；未运行执行器、未迁入源码或真实制品。冲突的本地 records 被保留、标为 superseded，唯一规范 records 已重新核验并重建账本；该批接受时累计为 145/1,551，后续接受批次使当前 re-audit 为 181/1,551 项，P4 尚未开始。详见 [P3-27 记录](release-manifests/P3_BATCH_27_LEGACY_SEARCH_AND_REAL_ABLATION_REAUDIT.md)。

#### P3-28：公开重写与 Stage6 evidence 边界复审（已完成，本地）

12 项候选逐项审阅并通过独立样本复审：7 项已有公开 release-safe 重写，2 项进入 P4 外部 probe/readiness/evidence 制品契约，2 项进入 P6 外部执行/测量计划契约，1 项进入 P5 Orin 度量与平台协议契约。当前内容漂移为 0；未运行执行器、未迁入新的源码或真实制品。该批接受时累计为 97/1,551，后续接受批次使当前 re-audit 为 181/1,551 项，P4 尚未开始。详见 [P3-28 记录](release-manifests/P3_BATCH_28_PUBLIC_REWRITES_AND_STAGE6_EVIDENCE_REAUDIT.md)。

#### P3-29：Stage6/Stage7 契约边界复审（已完成，本地）

12 项候选由独立逐项阅读完成处置：3 项具有已测试的公开脱敏重写，5 项进入 P4 外部证据契约，2 项进入 P5 环境契约，2 项进入 P6 执行契约。公开重写被实际比较职责与测试，而非根据同路径或关键词认定；其余项不运行、不迁入公开树。本批关闭时的累计为 109/1,551；随后 P3-26/P3-27 的规范 records 已通过独立复审和 supersession 处置，P3-31、P3-33、P3-34 也已按协议接受，当前累计为 181/1,551，P4 尚未开始。详见 [P3-29 记录](release-manifests/P3_BATCH_29_STAGE67_CONTRACT_REAUDIT.md)。

#### P3-30：Stage7 执行与测试边界复审（已完成，本地）

12 项候选由独立逐项阅读完成处置：8 项进入 P4 外部证据/制品契约，1 项进入 P5 环境契约，2 项进入 P6 执行契约，1 项由现有公开预测器工作流和测试链替代。所有项均重新绑定当前内容；未运行任何模型、数据、设备、子进程或外部服务，未迁入公开树。本批关闭时的累计为 121/1,551；随后 P3-26/P3-27 的规范 records 已通过独立复审和 supersession 处置，P3-31、P3-33、P3-34 也已按协议接受，当前累计为 181/1,551，P4 尚未开始。详见 [P3-30 记录](release-manifests/P3_BATCH_30_STAGE7_EXECUTION_TEST_REAUDIT.md)。

#### P3-31：F-Cooper 测试与证据边界复审（已接受，本地）

12 项候选逐项完成当前内容、P2 metadata/字节数、职责、导入/调用、公开替代物、来源/许可证和安全输出复审，并由独立语义审阅确认。4 项进入 P4 外部 ONNX、heldout capture、capability 与 Stage6 graph evidence 契约；3 项进入 P5 硬件能力、native benchmark 与 Orin 环境协议；5 项进入 P6 formal round、TVM measurement 与 GPU exclusivity 执行契约。所有项均重新绑定当前内容；没有执行私有测试、数据、模型、设备、子进程或网络，亦没有迁入公开树。详见 [P3-31 记录](release-manifests/P3_BATCH_31_FCOOPER_TEST_BOUNDARY_REAUDIT.md)。

#### P3-32：F-Cooper TVM 测试边界复审（已接受，本地）

12 项候选完成主审与独立语义复审：5 项进入 P4 外部证据/制品/请求契约，2 项进入 P5 GPU/调度环境协议，5 项进入 P6 TVM/ONNX/AP bridge 与 worker 执行契约。当前内容漂移为 0；未运行私有测试、执行器或真实制品，未迁入公开树。逐项 decision/evidence 与 HMAC ledger 已重新绑定并可重新计算，本批使当前已接受总数为 193/1,551 项；P4 尚未开始。详见 [P3-32 记录](release-manifests/P3_BATCH_32_FCOOPER_TVM_TEST_BOUNDARY_REAUDIT.md)。

#### P3-33：执行环境与制品边界复审（已接受，本地）

12 项候选完成逐项当前内容复核和独立语义审阅：4 项进入 P4 外部 checkpoint、ONNX、校准与部署证据契约；2 项进入 P5 GPU/共享内存环境协议；3 项进入 P6 GPU 资源、edge runner 与后端测量执行契约；3 项可在不依赖真实资产的前提下重写为公开合成逻辑。每项均已与 P2 metadata/字节数和当前 SHA-256 绑定，公开树未发现可直接替代的同职责测试。详见 [P3-33 记录](release-manifests/P3_BATCH_33_EXECUTION_ENVIRONMENT_ARTIFACT_REAUDIT.md)。

#### P3-34：Stage2 覆盖与导出边界复审（已接受，本地）

12 项候选完成逐项当前内容复核和独立语义审阅：1 项进入 P4 外部 checkpoint-to-ONNX evidence 契约；2 项进入 P5 H800/FP16 环境协议；5 项进入 P6 coverage job、artifact task、CLI 与 cold-start 执行契约；4 项可重写为使用合成行、profile 或 module spec 的纯内存公开逻辑。没有自动根据路径、名称或关键词认定公开替代物。详见 [P3-34 记录](release-manifests/P3_BATCH_34_STAGE2_COVERAGE_AND_EXPORT_REAUDIT.md)。

#### P3-35：Stage2 测试边界复审（已接受，本地）

主审和独立语义审阅均逐项读取 12 项候选并确认当前内容无漂移，但其中 8 项对主导 P4/P5/P6 边界的结论不同，因此没有按汇总计数提前落账。第三次只读裁决逐项复核职责、公开替代物与测试/API 覆盖后，确认公开 Stage2 contract/demo 测试只覆盖部分组合职责，不能将该候选批次误判为公开替代。最终 5 项进入 P4 外部证据/制品契约、2 项进入 P5 GPU 环境/调度契约、5 项进入 P6 native bridge、coverage job 与 lane runner 执行契约。公开 Stage2 contract/demo 定向回归为 20 passed；受限 evidence/decision 与 HMAC ledger 已重新绑定并可重新计算，本批使已接受总数为 205/1,551。详见 [P3-35 记录](release-manifests/P3_BATCH_35_STAGE2_TEST_BOUNDARY_ADJUDICATION.md)。

#### P3-36：Stage2 外部执行线索复审（已接受，本地）

12 项候选均完成主审和独立语义读取并确认当前内容无漂移。两项“公开同路径差异实现是否等价”的分歧由第三次职责/API/测试裁决解决：公开实现保留核心 B4 ablation 统计/报告和 three-arm search API，并由公开 demo pipeline 测试覆盖，因此两项为 `duplicate_or_superseded`；其余 10 项仍按职责转交。最终 5 项进入 P4 外部测量/制品/registry/readiness 契约、2 项进入 P5 环境/部署协议、3 项进入 P6 测量/控制/CLI 执行契约、2 项由公开实现替代。公开 demo pipeline 定向回归为 9 passed；受限 evidence/decision 与 HMAC ledger 已重新绑定并可重新计算。P3-36 中有 8 项此前已处置的 P2 身份，故截至本批的真实唯一覆盖为 209/1,551。详见 [P3-36 记录](release-manifests/P3_BATCH_36_STAGE2_EXECUTION_LEADS_ADJUDICATION.md)。

#### P3-37：Stage2/Stage4 运行边界复审（已接受，本地）

12 项完成主审和独立语义复审：5 项进入 P4 外部证据/制品契约、3 项进入 P5 硬件/编译环境契约、4 项进入 P6 训练/测量执行契约。当前内容无漂移；受限 evidence/decision 与 HMAC ledger 已重新绑定并可重新计算。P3-37 含 1 项此前已处置的 P2 身份，故截至本批的真实唯一覆盖为 220/1,551，而非按 records 累加的 253。没有执行私有模型、数据、设备或子进程，P4 尚未开始。

#### P3-38：Stage35 完成与证据边界复审（已接受，本地）

12 项完成主审、独立语义复审和针对两个 finalizer P4/P6 分歧的根级逐项裁决。裁决读取了测试与被测入口：两个 finalizer 只接收 manifest/state rows、验证结构并输出 schema/audit，测试中的 CLI 仅覆盖 `--help`，不实际执行模型、设备或 job，因此均归 P4 而非 P6。最终 6 项进入 P4 ONNX/evidence/finalization 契约、6 项进入 P6 supervisor/source queue/repeat plan 执行契约。受限 evidence/decision 与 HMAC ledger 已重新绑定并可重新计算，本批使真实唯一覆盖到 232/1,551。详见 [P3-38 记录](release-manifests/P3_BATCH_38_STAGE35_FINALIZATION_ADJUDICATION.md)。

#### P3-39：Stage35 Gold 队列与充分性复审（已接受，本地）

12 项完成主审和独立语义复审，当前内容无漂移且结论一致：6 项进入 P4 的 Gold merge、sufficiency、integrity audit 与 repair-manifest 外部 evidence 契约，6 项进入 P6 的 source queue、AP/performance plan 与 runner 执行契约。公开树没有同路径或职责等价实现；没有运行私有队列、模型、数据、设备或子进程。受限 evidence/decision 与 HMAC ledger 已重新绑定并可重新计算，本批使真实唯一覆盖到 244/1,551。详见 [P3-39 记录](release-manifests/P3_BATCH_39_STAGE35_GOLD_QUEUE_REAUDIT.md)。

#### P3-40--P3-42：当前交接更正（已接受，本地）

P3-40 经第三次逐项裁决后为 5 项 P4、7 项 P6；P3-41 为 3 项 P4、3 项 P6、6 项 `duplicate_or_superseded`；P3-42 经双审为 8 项 P4、4 项 P6。三批均完成当前内容绑定和路径无关 HMAC ledger 重算，且未运行私有实验。P3-41 的 6 项公开替代均已逐项以 Stage4 实现、测试或 API 职责证据核实，并完成 119 项定向回归。全量接受 records 虽为 289 条，但 11 条是对既有 P2 身份的复审，故当前真实唯一覆盖为 **278/1,551**；P3 仍未关闭，P4 尚未开始。详见 [P3-41 记录](release-manifests/P3_BATCH_41_STAGE4_EQUIVALENCE_REAUDIT.md) 与 [覆盖身份对账](release-manifests/P3_COVERAGE_IDENTITY_RECONCILIATION.md)。

#### P3-43：Stage5/Stage6 执行与外部证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审与许可证/安全审查；第 4、5 项的 P4/P5/P6 主导边界由第三次只读裁决，均归入 P6。最终为 4 项 P4（ONNX、checkpoint/source registry、终端摘要与 native AP 证据）和 8 项 P6（任务计划、闭环控制、独立验证与反馈轮次执行）。源码继承项目级 Apache-2.0；该许可不覆盖外部制品，且所有公开后续契约均须使用路径无关引用。该批新增 12 个不重复的冻结身份，使当前真实唯一覆盖到 **290/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-43 记录](release-manifests/P3_BATCH_43_STAGE5_STAGE6_EXECUTION_REAUDIT.md)。

#### P3-44：Stage6 证据、环境与执行边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。5 项进入 P4（Stage6/F-Cooper 证据、TVM 基线与 native checkpoint 绑定），1 项进入 P5（H800 范围与 runner readiness），4 项进入 P6（F-Cooper 正式闭环、paper closure、五臂协议和 schedule-only 执行），2 项为已有发布安全公开改写。第 4 项的主导边界与第 6、10 项的“改写/等价替代”分类均经第三次只读裁决：前者的 GPU guard 服务于执行闭环，后两者有明确公开迁移记录且公开 API/测试更严格，故分别归 P6 与 `rewritten_public`。该批新增 12 个不重复身份，使真实唯一覆盖到 **302/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-44 记录](release-manifests/P3_BATCH_44_STAGE6_EVIDENCE_AND_EXECUTION_REAUDIT.md)。

#### P3-45：Stage7 执行、反馈与证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终 5 项进入 P4（actual-v3 反馈/轨迹、cache 与 formal input/evidence 契约）和 7 项进入 P6（GPU 租约、恢复生命周期、物理管线、actual-v3 执行桥、顶层编排和 schedule 控制）。第 1 项 GPU 调度器不只是环境发现，而管理租约、控制器与重试生命周期，故经第三次裁决归 P6；第 7、8 项不启动硬件或控制器，只绑定/推进反馈与物化证据状态，故归 P4。该批新增 12 个不重复身份，使真实唯一覆盖到 **314/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-45 记录](release-manifests/P3_BATCH_45_STAGE7_EXECUTION_AND_EVIDENCE_REAUDIT.md)。

#### P3-46：Stage7 环境、执行与最终证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 4 项 P4（trajectory、formal finalizer、scanner blocker 与 fail-closed evidence facade）、5 项 P5（no-GPU、部署 bundle/cross-host override 与 H800 probe/lock）和 3 项 P6（round controller、worker 与 dispatch）。双审的逐项结论完全一致；独立审阅的汇总行笔误经其按索引复核后更正为 P4=4、P5=5、P6=3。该批新增 12 个不重复身份，使真实唯一覆盖到 **326/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-46 记录](release-manifests/P3_BATCH_46_STAGE7_ENVIRONMENT_AND_FINALIZATION_REAUDIT.md)。

#### P3-47：Stage7 source 与 feedback 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 5 项 P4（private ablation、physical terminal-evidence promotion、freeze/search/relocation） 、3 项 P5（formal prepare 与 source scheduling/lease 环境）和 4 项 P6（GPU runtime、source materialization、source lease controller 与 round shell）。第 2 项不运行硬件或测量，只认证并晋升已产生的 Stage3 terminal evidence，故经第三次裁决归 P4；第 9 项持锁调用 resolver、提交 canonical source result 并推进执行状态，故归 P6。该批新增 12 个不重复身份，使真实唯一覆盖到 **338/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-47 记录](release-manifests/P3_BATCH_47_STAGE7_SOURCE_AND_FEEDBACK_REAUDIT.md)。

#### P3-48：Stage2 执行与外部制品边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 3 项 P4（校准、ONNX 与 coldstart96 测量行）、8 项 P6（AP queues/watchers、TRT/TVM build/eval/measure）和 1 项 `excluded_nonessential`（无公开支持链调用的替代图）。第 1、8 项没有实际运行数据集、checkpoint、CUDA 或导出，而是验证制品接口/证据，故经第三次裁决归 P4；第 12 项只产出可选展示图，公开 Stage4 实现、CLI、测试和 verified artifact 已承担选择证据，故可排除。该批新增 12 个不重复身份，使真实唯一覆盖到 **350/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-48 记录](release-manifests/P3_BATCH_48_STAGE2_EXECUTION_AND_ARTIFACT_REAUDIT.md)。

#### P3-49：Stage7 deployment 与 TVM 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 2 项 P4（论文 cost-model 图/源 CSV provenance 与 checkpoint alias）、1 项 P5（不使用 GPU 的 receipt archive）、8 项 P6（TVM benchmark/tune、部署 sidecar、恢复与 cooldown 控制）和 1 项 `excluded_nonessential`（无公开支持链的可选示意图）。第 1、6、10 项经第三次只读裁决：论文引用的图表需要 P4 补齐来源和制品契约，receipt archive 只保存环境/部署证据而不测量 GPU，故归 P5；cooldown wrapper 会改变实时部署状态，故归 P6。该批新增 12 个不重复身份，使真实唯一覆盖到 **362/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-49 记录](release-manifests/P3_BATCH_49_STAGE7_DEPLOYMENT_AND_TVM_REAUDIT.md)。

#### P3-50：release 与 deployment 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 3 项 P4（release asset integrity 与 checkpoint alias 契约）、5 项 P6（实时 deployment recovery、takeover 与 source sidecar）、2 项 `duplicate_or_superseded`（有公开 clean-clone/synthetic smoke API、测试和支持链职责证据）和 2 项 `blocked_license_or_permission`（外部 CUDA 源码 patcher）。第 1、4、5 项经第三次只读裁决：公开 archive/manifest 的相邻测试不足以替代 release-asset 验证器，故归 P4；后两者虽含环境检查，但主导职责是实时部署事务、状态变更和回滚，故归 P6。外部 CUDA patcher 保持 P7 许可/隐私阻塞，不得以“可选”或“可生成”理由排除。该批新增 12 个不重复身份，使真实唯一覆盖到 **374/1,551**。P3-50 入账前还对此前 373 条 accepted records 重建了 P2 metadata、当前 SHA-256 和 HMAC ledger 绑定，确认其去重覆盖为 362。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-50 记录](release-manifests/P3_BATCH_50_RELEASE_AND_DEPLOYMENT_REAUDIT.md)。

#### P3-51：results 与 runtime 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 10 项 P4（闭环驾驶结果、图表以及 checkpoint→ONNX/engine 制品配置）、1 项 P5（分布式训练/评测运行配置）和 1 项 P6（CARLA/CUDA 感知—规划全流程运行时）。第 11、12 项经第三次只读裁决：它们配置外部 checkpoint、ONNX 与 engine 的导出/物化边界，并不以驱动、设备发现或环境准入为主，故归 P4。闭环结果/图表脚本只消费既有结果而不执行实验，但其输入结果、路由元数据、图源和论文数值 provenance 仍必须由 P4 约束。该批新增 12 个不重复身份，使真实唯一覆盖到 **386/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-51 记录](release-manifests/P3_BATCH_51_RESULTS_AND_RUNTIME_REAUDIT.md)。

#### P3-52：training 与 plugin 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 3 项 P4（tiny 训练/评估所需数据、权重、annotation 与 checkpoint 制品配置）和 9 项 P6（plugin registry、训练/评测 hook、loss、runner、推理收集、training API 及 dense heads）。第 1--4、7 项经第三次只读裁决：前三项虽含 worker、epoch 和日志设置，主导仍是外部数据/权重/checkpoint 绑定，故归 P4；plugin registry 和 API shim 不是环境探测，而是把调用方接入 CUDA/DDP、runner、hook 与 checkpoint 执行链，故归 P6。Motion/occupancy heads 的第三方 TRT 适配来源仍需在后续执行合同中单独完成 NOTICE/provenance 审查。该批新增 12 个不重复身份，使真实唯一覆盖到 **398/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-52 记录](release-manifests/P3_BATCH_52_TRAINING_AND_PLUGIN_REAUDIT.md)。

#### P3-53：model 与 plugin 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。12 项均归 P6：planning/tracking/segmentation heads、assignment、CUDA autograd/custom-op、transformer/BEV attention 以及 ONNX/TRT export 支撑均处于模型训练、推理或导出执行链。双审逐项一致；未找到有公开 API/test support-chain 的等价实现，因此没有以 duplicate/excluded 替代。候选中含 OpenMMLab 修改头或外部 TRT 适配来源的部分，必须在 P6 之前补齐来源、NOTICE、CUDA/plugin 可用性和路径安全失败关闭契约。该批新增 12 个不重复身份，使真实唯一覆盖到 **410/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-53 记录](release-manifests/P3_BATCH_53_MODEL_AND_PLUGIN_REAUDIT.md)。

#### P3-54：quantization 与 evidence 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 3 项 P4（coldstart、engine/probe 与 graph evidence 外部制品）和 9 项 P6（pruning/quantization、GPU guard、GPU benchmark、ONNX preparation 与 recovery training）。GPU exclusivity gate 经第三次只读裁决归 P6：GPU telemetry/锁/quiet-window 是准入守卫，但文件的主导行为是被保护命令的运行、监控、隔离和进程终止。量化候选有 QuantV2X/OpenCOOD 等外部来源或依赖信号；其 NOTICE/provenance、calibration 数据隐私和 CUDA 运行时必须在 P6 前完成。该批新增 12 个不重复身份，使真实唯一覆盖到 **422/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-54 记录](release-manifests/P3_BATCH_54_QUANTIZATION_AND_EVIDENCE_REAUDIT.md)。

#### P3-55：recovery 与 TVM 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 3 项 P4（baseline admission、capability rebind 与 AP feedback repair 外部 evidence）和 9 项 P6（checkpoint recovery、search replay、watchdog、round/GPU scheduler、TVM measurement、ONNX quant contract 与 Relax worker）。双审逐项一致：ONNX quant-contract builder 会以 calibration 样本运行 ONNX Runtime inference，TVM/Relax worker 会加载编译模块并执行，故均归 P6；只验证、重写或修复已有 checkpoint/ONNX/AP evidence 而不执行模型或硬件的项目归 P4。该批新增 12 个不重复身份，使真实唯一覆盖到 **434/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-55 记录](release-manifests/P3_BATCH_55_RECOVERY_AND_TVM_REAUDIT.md)。

#### P3-56：DP4A 与 baseline 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 7 项 P4（scheduler/feedback integrity、H800/4090 baseline、Pareto predictor 与 accuracy evidence）和 5 项 P6（TVM worker 与 DP4A compile/tune/benchmark gates）。第 10、11 项经第三次只读裁决归 P4：它们使用私有 baseline 或训练 predictor 作离线 Pareto 派生，主导风险是制品 provenance、许可和校准依据，而非候选/硬件/模型执行。DP4A gates 则创建、调优、编译、导出并 benchmark CUDA/TVM 代码，归 P6。该批新增 12 个不重复身份，使真实唯一覆盖到 **446/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-56 记录](release-manifests/P3_BATCH_56_DPA4_AND_BASELINE_REAUDIT.md)。

#### P3-57：engine 与 baseline 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 9 项 P4（latency mapping、baseline integration 与 predictor/Pareto analysis 外部 evidence）和 3 项 P6（ONNX export、TensorRT engine build 与 CUDA benchmark 执行）。双审逐项一致：只消费既有测量、baseline 或 predictor 的映射、整合和预测分析归 P4；会物化 runtime engine、导出 ONNX 或执行 CUDA benchmark 的候选归 P6。公开 CPU-only 选择逻辑没有承担完整的输入、输出和执行职责，不能据此认定公开替代。该批新增 12 个不重复身份，使真实唯一覆盖到 **458/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-57 记录](release-manifests/P3_BATCH_57_ENGINE_AND_BASELINE_REAUDIT.md)。

#### P3-58：公开表面与证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 7 项 P4（measurement budget、baseline/predictor search、ratio lookup 与 ablation evidence）、2 项已公开迁入、2 项已安全改写和 1 项非必要排除。第 1、2、11、12 项经第三次只读裁决：同字节公开 package marker 是候选本身的迁入，受测试覆盖的更窄安全导出面是改写，未被受支持流程调用的纯说明 legacy marker 可排除。公开相邻算法不构成完整等价物；其余候选仍需 P4 提供外部 evidence 的来源、许可、哈希、schema 和匿名 fixture。该批新增 12 个不重复身份，使真实唯一覆盖到 **470/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-58 记录](release-manifests/P3_BATCH_58_PUBLIC_SURFACE_AND_EVIDENCE_REAUDIT.md)。

#### P3-59：合同与证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 6 项已安全改写、5 项 P4（cost bundle、evidence registry、native route、probe profile 与 measurement feedback evidence）和 1 项 P7 阻塞。第 7 项经第三次只读裁决：尽管逻辑是纯内存，它会在输出和异常中回显未净化的 probe/feature 标识，公开安全改写和测试尚不存在，故必须保持阻塞，不能以未来可改写而提前记为已改写。其余 P4 候选需由后续外部契约提供来源、许可、哈希、schema 和匿名 fixture。该批新增 12 个不重复身份，使真实唯一覆盖到 **482/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-59 记录](release-manifests/P3_BATCH_59_CONTRACT_AND_EVIDENCE_REAUDIT.md)。

#### P3-60：Stage5--Stage7 边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 7 项已安全改写、2 项 P4（closure feedback 与 independent measurement request evidence）和 3 项 P5（executor admission、recovery root 与 full frozen/sidecar/pre-scan 环境状态）。第 4、10、11、12 项经第三次只读裁决：不启动测量的 request builder 归 P4；进程/根目录/状态准备而不运行 executor 的候选归 P5；公开 selection-only 子集不能替代完整私有 stateful 角色。该批新增 12 个不重复身份，使真实唯一覆盖到 **494/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-60 记录](release-manifests/P3_BATCH_60_STAGE567_BOUNDARY_REAUDIT.md)。

#### P3-61：遗留测试边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 11 项非必要排除和 1 项公开替代。第 2--5、8--12 项经第三次只读裁决：旧 pytest 本身不启动其被测目标中的模型、GPU、TVM 或子进程，且不被当前公开支持链调用，不能仅因被测脚本的潜在执行边界而转 P4/P6。第 8 项由公开 Stage1 coupling predictor API 与 workflow 测试承担同类契约。该批新增 12 个不重复身份，使真实唯一覆盖到 **506/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-61 记录](release-manifests/P3_BATCH_61_LEGACY_TEST_REAUDIT.md)。

#### P3-62：遗留执行测试边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 11 项非必要排除和 1 项公开替代。第 1--9、11、12 项经第三次裁决：旧测试不执行其被测 replay、watchdog、TVM、TRT 或硬件工具，且不被当前公开支持链调用，不能因被测目标的潜在边界而转 P4--P6。第 12 项由公开 Stage1 autoscan/census/predictor workflow 测试替代。该批新增 12 个不重复身份，使真实唯一覆盖到 **518/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-62 记录](release-manifests/P3_BATCH_62_LEGACY_EXECUTION_TEST_REAUDIT.md)。

#### P3-63：Stage1/Stage2 遗留测试复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 7 项公开替代和 5 项非必要排除。第 3、4、6、10 项经第三次裁决：公开 Stage1 predictor/classifier/trace-plan 和 Stage2 canonical-search/genome 测试覆盖净化后的公开合同；但完整私有 QxS/SMBO entry 未被公开 API/test 替代，因其不在当前公开支持链中排除。该批新增 12 个不重复身份，使真实唯一覆盖到 **530/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-63 记录](release-manifests/P3_BATCH_63_STAGE12_LEGACY_TEST_REAUDIT.md)。

#### P3-64：Stage1--Stage4 遗留测试复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 1 项 P7 阻塞、2 项公开替代和 9 项非必要排除。第 4--10、12 项经第三次裁决：旧测试不执行被测 AP、TVM、TRT、calibration 或 repair 工具，且不被当前公开支持链调用，不能仅由底层执行边界转 P4--P6。第 1 项会回显未净化 probe/feature 标识，公开安全改写尚不存在，故保持 P7。该批新增 12 个不重复身份，使真实唯一覆盖到 **542/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-64 记录](release-manifests/P3_BATCH_64_STAGE35_LEGACY_TEST_REAUDIT.md)。

#### P3-65：Stage5/Stage6 遗留测试复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 2 项公开替代和 10 项非必要排除。双审逐项一致：公开 Stage5 genome 与 Stage6 six-arm contract 测试覆盖相同职责；其余旧测试不执行被测 AP plan、repair、validation、audit、TVM helper 或 controller 工具，且不在当前公开支持链中，不能仅由底层执行边界转 P4--P6。该批新增 12 个不重复身份，使真实唯一覆盖到 **554/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-65 记录](release-manifests/P3_BATCH_65_STAGE56_LEGACY_TEST_REAUDIT.md)。

#### P3-66：Stage6/Stage7 遗留测试复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 2 项公开替代和 10 项非必要排除。第 3 项经第三次裁决：公开 Stage6 manifest validator 已以可校验、失败关闭 contract 承担 INT8 自动路由并禁止 legacy DP4A/native/hand rewrite 的职责。其余旧测试不在当前公开支持链，也不执行被测运行工具。该批新增 12 个不重复身份，使真实唯一覆盖到 **566/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-66 记录](release-manifests/P3_BATCH_66_STAGE67_LEGACY_TEST_REAUDIT.md)。

#### P3-67：deployment 与论文证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。第 1 项双审一致；第 2--12 项经第三次只读裁决。最终为 2 项 P4（未公开论文 latency/curve 图源）、1 项 P5（部署 diagnostic bytecode archive）、8 项 P6（Stage7 transactional deployment/runtime repair）和 1 项 P7（外部 process patcher 的本机路径与再分发权限阻塞）。裁决确认：论文证据图没有公开数据/图生成契约，不能仅因不在当前支持链而排除；archive 只依赖部署环境状态而不调度进程；会替换或重绑 deployed/frozen runtime、scheduler、source resolver 或 live wrapper 的候选属于执行链。该批新增 12 个不重复身份，使真实唯一覆盖到 **578/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-67 记录](release-manifests/P3_BATCH_67_DEPLOYMENT_AND_EVIDENCE_REAUDIT.md)。

#### P3-68：deployment、metrics 与证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。第 6--10 项双审一致；第 1--5、11、12 项经第三次只读裁决。最终为 2 项 P4（未公开 measured CSV 的图/插值证据）、5 项 P6（Stage7 transactional deployment 与 live orchestrator 维护）、4 项 P7（外部 patch/config 与第三方派生 metrics 的许可/路径/输出阻塞）和 1 项非必要 legacy synthetic test 排除。裁决确认：会替换 deployed/frozen runtime、更新 deployment state 或认证并控制 live 进程的包装器属于执行链；未公开 measured CSV→论文图的职责属于外部制品契约，即使该脚本不在当前公开调用链。该批新增 12 个不重复身份，使真实唯一覆盖到 **590/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-68 记录](release-manifests/P3_BATCH_68_DEPLOYMENT_METRICS_AND_EVIDENCE_REAUDIT.md)。

#### P3-69：UniV2X 模型支持边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。第 1、4--12 项双审一致；第 2、3 项经第三次只读裁决。最终为 1 项 P4（TRT Phase-2 config 的 checkpoint/ONNX/engine/dataset/anchor 制品来源）、9 项 P6（UniV2X/MMDet registry、assigner、coder、match-cost 和 tensor 执行支持）和 2 项非必要排除。裁决确认：零字节 private package marker 不承担可发布职责；模型 config 的主导边界是外部制品身份而非环境或执行器。该批新增 12 个不重复身份，使真实唯一覆盖到 **602/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-69 记录](release-manifests/P3_BATCH_69_UNIV2X_MODEL_SUPPORT_REAUDIT.md)。

#### P3-70：UniV2X 执行支持边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。主审初稿的队列身份错误已整体作废；重新绑定实际队列后，双审对全部 12 项逐项一致。最终为 12 项 P6：distributed evaluation hook、training loss，以及 UniV2X model plugin/dense-head import 支持。公开 CPU-safe 流程没有同职责 API/test 支持链；这些候选支撑私有模型训练、评估、配置注册与执行链。该批新增 12 个不重复身份，使真实唯一覆盖到 **614/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-70 记录](release-manifests/P3_BATCH_70_UNIV2X_EXECUTION_SUPPORT_REAUDIT.md)。

#### P3-71：UniV2X tracking 与执行支持边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查，双审逐项一致。最终为 12 项 P6：分割 transformer/metric、tracking memory/query/state/tracker、detector registry、ONNX/TRT plugin functions、fusion modules、训练 hook 与 BEV/attention modules。公开 CPU-safe 流程没有同职责 API/test 支持链；这些候选支撑私有模型训练、跟踪、推理、导出或执行状态更新。该批新增 12 个不重复身份，使真实唯一覆盖到 **626/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-71 记录](release-manifests/P3_BATCH_71_UNIV2X_TRACKING_EXECUTION_REAUDIT.md)。

#### P3-72：pruning 与量化边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。前六项双审一致为 P6：transformer、结构化 pruning 与 temporal cache 执行支持。后六项经第三次只读许可证裁决为 P7：量化包、AdaRound、communication quant、BN folding 及 BEVFormer/downstream wrappers 明确具有 QuantV2X/OpenCOOD copy/port 来源；可见上游许可不允许进一步转让，且没有可公开的双许可、NOTICE 或干净替代实现。该批新增 12 个不重复身份，使真实唯一覆盖到 **638/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-72 记录](release-manifests/P3_BATCH_72_PRUNING_AND_QUANTIZATION_REAUDIT.md)。

#### P3-73：evidence 与量化边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 6 项 P4（真实测量、formal feedback、hardware baseline 与结果制品）、2 项 P6（formal feedback 原子发布与 TVM remeasurement 控制）、3 项 P7（QuantV2X/OpenCOOD copy/port 许可阻塞）和 1 项非必要排除。q6、q8、q11 经第三次裁决：atomic finalizer/remeasurement request 会推进发布或执行控制闭环，归 P6；私有早期 sanity 不在公开支持链且不承载正式证据，归排除。该批新增 12 个不重复身份，使真实唯一覆盖到 **650/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-73 记录](release-manifests/P3_BATCH_73_EVIDENCE_AND_QUANTIZATION_REAUDIT.md)。

#### P3-74：attention 执行与证据边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 5 项 P4（AP/latency/H800/SMBO 外部结果制品）、3 项 P6（模型评估与 TVM/CUDA 执行）、2 项 P7（私有根目录与破坏性工作目录语义）和 3 项已安全公开改写。六项经第三次裁决：执行 runner 的实际加载/编译/计时职责归 P6；嵌入路径仅为结果来源示例的 builder 归 P4；当前回显私有根目录或破坏性工作目录的 runner 保持 P7。该批新增 12 个不重复身份，使真实唯一覆盖到 **662/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-74 记录](release-manifests/P3_BATCH_74_ATTENTION_EXECUTION_REAUDIT.md)。

#### P3-75：dataset 与能耗执行边界复审（已接受，本地）

12 项完成 P2 metadata/bytes、当前内容绑定、主审、独立语义复审、项目级许可证继承和安全输出审查。最终为 4 项 P4（dataset、coverage、active-sample 与网格外部制品）、6 项 P6（ONNX/TRT/TVM/CUDA/NVML 执行与能耗测量）和 2 项 P7（私有根目录或外部 OpenCOOD 来源绑定）。七项经第三次裁决：真实 load/compile/measure/NVML runner 均归 P6，纯配置计划归 P4。该批新增 12 个不重复身份，使真实唯一覆盖到 **674/1,551**。没有执行私有模型、数据、设备、子进程或网络任务；P4 尚未开始。详见 [P3-75 记录](release-manifests/P3_BATCH_75_DATASET_AND_ENERGY_REAUDIT.md)。

#### P5--P6：环境与全流程复现

1. CPU smoke 始终只用受控 fixture，且不得隐式发现本机 `results/`、GPU、缓存或外部目录。
2. 4090 与 H800 使用各自明确的环境文件；文档只写主机类别、软件版本和执行接口，绝不记录登录方式或密钥材料。
3. 训练、评测和硬件执行命令必须显式指定输入、输出、随机种子和配置；输出写入被忽略的目录，生成 manifest 后方可进入后续汇总。
4. 任何论文数值从 `demo` 升级为 `verified` 前，必须具备原始输入哈希、运行 manifest、独立验证和数据/硬件许可依据。

#### P7--P8：发布门禁与外部动作

1. 审查暂存差异、发布候选和（如计划公开现有历史）Git 历史。若历史含敏感凭据，先轮换凭据；经明确授权后才可使用历史重写并强推。
2. 完成第三方代码、数据集和模型许可检查；发布的许可证和引用元数据必须与实际可发布内容一致。
3. 在全新目录完成 `git clone`、固定依赖安装、CPU smoke、全量测试和 CI；对 4090/H800 运行声明的最小验证。将结果写入本台账。
4. 只有在维护者明确授权后，才能推送分支、合并、设置公开可见性、打版本 tag 或创建 Release；完成外部动作后将 URL、tag、提交和可复现命令以不含凭据的形式追加到本节。

### 完成定义与发布门槛

“项目上传成功、可开源”不等于 Git 工作树干净，也不等于 smoke 通过。必须同时满足：

1. 支持的每一项实验都有已跟踪的源代码与设计说明，或有合法、可验证的外部输入获取清单；不依赖维护者的私有目录、缓存、模型文件或口头知识。
2. 所有固定小型测试输入在仓库内且脱敏；模型、checkpoint、engine、ONNX、缓存、压缩包、结果图和其他可生成大文件均被忽略并由脚本或 manifest 替代。
3. 新手能按 README 在新目录完成 clone、安装、CPU smoke 和测试；完整实验能按环境文档在 4090/H800 上进行其声明的最小验证。
4. 所有发布证据（输入/输出 SHA-256、种子、配置、代码版本、许可证、验证状态）可被独立复核；缺失证据时流程失败关闭，不把 demo 作为论文结果。
5. 安全/隐私/许可审查通过，工作树和计划公开的历史不含有效凭据或未授权材料；外部发布操作已经取得明确授权并记录结果。

### 变更记录

| 日期 | 计划项 | 状态变化 | 证据与备注 |
| --- | --- | --- | --- |
| 2026-06-24 | 起始状态 | 已建立 | 初始公开内容为 Stage1 model scanner；完整实验仍在独立私有工作树。 |
| 2026-07-29--2026-07-30 | 历史发布基线 | 已完成 | 建立可测试的双制品发布树、匿名 archive、demo/verified 证据边界和初版审计；详细提交与当时结果保留在本文件前述章节。 |
| 2026-07-30 | 历史范围披露 | 已完成 | 明确 scoped CPU smoke 不等价于完整训练、硬件执行或论文结果；对应文档提交见 Git 历史。 |
| 2026-08-03 | P0 前置发布加固 | 已完成 | `9df8299` 新增固定 CPU requirements、干净 clone/new venv smoke、CI 任务、匿名 ZIP 依赖闭包及生成物忽略规则；当时全量回归为 344 passed。 |
| 2026-08-03 | P0 交接台账 | 进行中 | 本节、README 链接和台账回归测试在同一提交中建立。真实 ZIP build 发现 public-smoke CI 中的 `file://` 会触发匿名制品安全扫描；已增加回归测试并改为本地路径 clone，待重新 build/verify 后结项。 |
| 2026-08-03 | P0 交接台账 | 已完成 | 回归测试将 CI 禁止使用 `file://` 固定下来；真实匿名 ZIP build/verify 通过，100 个成员，SHA-256 `9d93031c7a4cb91ada2b9b92ce70839b5df47630654b4c9b21241acb279c53df`；最终全量回归为 346 passed，Ruff、compileall 与 diff 检查通过。下一项为 P1 发布候选冻结。 |
| 2026-08-04 | P1 首个候选 | 未通过，修复中 | 候选 `1a67bd624e2ceee68dcf609a9c5148d48351ae1f` 的独立源树验收通过（346 passed、80.71% coverage、26 项 clean-clone 检查）；匿名 ZIP build/verify 也通过，SHA-256 为 `9d93031c7a4cb91ada2b9b92ce70839b5df47630654b4c9b21241acb279c53df`、100 个成员。但解包后的定向测试发现四项闭包缺失：公开交接文档和 README 断言不适用于匿名映射 README，且 smoke 脚本的 `--help` 过早要求 Git。已新增制品感知回归测试、将公开断言在匿名 ZIP 中跳过，并推迟 Git 根目录解析；必须以修复后的新提交重新冻结和完整复验，不能复用本候选作为 P1 完成证据。 |
| 2026-08-04 | P1 第二个候选 | 未通过，修复中 | 候选 `fe524b736b637a3fae2121788647a12f154f60a8` 的独立源树验收通过（347 passed、80.71% coverage、27 项 clean-clone 检查）。其匿名 ZIP 通过 build/verify（100 个成员，SHA-256 `80e091112d29fecf9c67d5115b42b5623a2d4ad7e924b79fabdebf98db83f85f`），解包后的 CPU smoke、构建、Ruff、compileall 和 coverage 均通过；全量测试为 343 passed、3 skipped、1 failed。唯一失败是 archive 集成测试仍把匿名 README 固定为源树路径 `README.anonymous.md`，没有接受 ZIP 的正确根 `README.md` 映射。测试现已改为两种制品路径均可验证；必须以修复后的新提交重新冻结和完整复验。 |
| 2026-08-04 | P1 第三个候选 | 已完成（本地） | 候选 `c65b5fc7eb9d20e2d34928f7c948aea90f822754` 以两个新 venv 完成源树与解包 ZIP 验收；源树为 347 passed/80.71%，解包 ZIP 为 344 passed、3 skipped/80.71%，ZIP SHA-256 为 `aa83bc98b4ada4f7865a5db0ee0c12c92215599b50d9734ff3f123c6833cf4b7`。完整命令、依赖、manifest、扫描和 P7 风险见上节。P2 是下一项；没有外部发布动作。 |
| 2026-08-04 | P2a 私有源盘点 | 已完成（本地只读） | 新增安全 inventory 工具与脱敏汇总。5,127 tracked 与 88 未跟踪条目被分类为 1,553 个安全候选及 3,662 个不透明复核/排除项；原始报告 SHA-256 为 `819207c036a35c7a4d8d61f7628aaaf1ea4ec9cc8b87186d6f6a18891e9057ad`，不提交。公开树回归为 350 passed，当前匿名 ZIP 已 verify（102 成员，`9cb20e3716958c1f0a69b2f03e4431915d48b18e688bf509efce2170a65f0faa`）。未修改私有工作树，未迁移或上传文件。下一项为 P3 逐批脱敏迁移。 |
| 2026-08-04 | P2b 盘点安全加固 | 已完成（本地只读） | 独立复审发现 P2a 的嵌套数据边界和稳定读取缺口后，以 v1.1.0 工具补充路径段、硬链接与 ctime/link-count 失败关闭测试并获复审批准。只读重新盘点得到 1,551 个候选及 3,664 个不透明复核/排除项；新原始报告 SHA-256 为 `6fafdace7dc0eae3e9982763cb1c5c098c0ae83d1715af67ff1e08d7a1d12a79`，临时报告已清理。P2a 计数仅保留历史证据。 |
| 2026-08-04 | P3-1 Stage6 合约层 | 已完成（本地） | 迁入 3 个确定性纯 Python 模块及其 unit/integration tests；将固定生成物位置改为外部证据描述符，不迁入数据、模型、结果或硬件执行器。审查后的字段白名单、证据哈希一致性和畸形输入失败关闭均有回归测试。公开树回归为 366 passed，匿名 ZIP 已 verify（109 成员，`72b8f3c813fd29407044c863c62d66f50ae7c8a4296ae43727bdce1ea75f1b9f`）。完整范围和排除决定见 P3-1 批次记录。 |
| 2026-08-04 | P3-2 Stage5 审阅 | 已完成（未批准迁移） | 审阅 4 个 Stage5 候选：3 个处理外部制品/测量输入，closure 审计器会回显调用者提供的证据/结果位置。未复制代码或数据；改造要求和下批筛选条件见 P3-2 审阅记录。 |
| 2026-08-04 | P3-3 Stage4 闭环审计 | 已完成（本地） | 首轮迁入后的 `after_feedback` 行绑定、采集数值契约和无效指标回显缺口均以新增失败关闭测试修复，并获独立复审批准。最终源树回归为 378 passed、81.52% coverage；匿名 ZIP 独立 verify 通过（111 成员，`aca6c8426e5f1d413586e18e315dc4655fcc5446d90cd494d0643d1e28cea32c`）。未推送、未发布。 |
| 2026-08-04 | P3-4 Stage2 S1 探针指标审阅 | 已完成（未批准迁移） | 候选无 I/O 且会拒绝性能字段，但报告/错误会回显调用者探针标识与特征名。未复制代码、测试、数据或执行器；改造条件已记录，下一批继续筛选。 |
| 2026-08-04 | P3-5 Stage2 延迟异常策略 | 已完成（本地） | 迁入脱敏的纯内存质量策略，并以定向回归固定分组字段、超大数值和标识不回显的失败关闭行为。全量回归为 386 passed、81.72% coverage，新模块为 93%；独立复审通过。匿名 ZIP 已 verify（113 成员，`6d821d88ab4915a586192b0e38592dfea22c51a8541fef194487576f0a3196b4`）。未推送、未发布。 |
| 2026-08-04 | P3-1 Stage6 加固复审 | 已完成（本地） | 独立复审发现了字段回显、证据绑定、嵌套形状、q-mode 和数值异常链缺口；均以测试优先改为公开字段白名单、规范哈希绑定、固定失败码和无异常链的有限数值验证。最终独立复审批准；Stage6 定向为 72 passed、89% coverage。当前 HEAD 全树为 405 passed、81.95% coverage；匿名 ZIP 已 verify（113 成员，`96a3bc9b6980e2f25bb9ab102cc950ba2ff57f9fb7c2f35142d2367818b319f6`）。 |
| 2026-08-04 | P3-6 下一批候选审阅 | 已完成（未批准直接迁入） | 冷启动筛选器会回显 label，Stage6 独立验证器会深拷贝输入行；两者均要求匿名 schema/投影重写。已有 Stage6 协议 smoke 不重复迁入。未复制代码、测试、数据或生成物；详细条件见 P3-6 审阅记录。 |
| 2026-08-04 | P3-7 关闭验证 | 已完成（P3 未关闭） | 新增 P3 disposition ledger 工具与公开执行面回归。全量验证为 423 passed、81.95% coverage；匿名 ZIP 已 verify（116 成员，`707dbe2ac21dd276d8eb25192527c0486a8ccf86f73afc50174e2a849e8fb06a`）。由于缺少覆盖 P2b 1,551 个候选的本地 decisions 文件和可提交公开 ledger，P3 仍不能标记完成；当前 HEAD 不应作为完整公开发布推送。 |
| 2026-08-04 | P3-8 候选处置启动审计 | 已完成（本地只读） | 重新生成 P2b inventory 并确认 SHA-256 未漂移；同路径哈希比对显示 9 个候选已同内容公开、38 个同路径但不同、1,504 个无公开同路径；Stage4 无路径功能筛选进一步确定 P3-9 的 19 项纯内存脱敏改写范围。已固定 decisions token、许可/隐私阻塞规则和第一批 9 项账本演练范围；未迁入源码、未生成公开 ledger、未推送。 |
| 2026-08-04 | P3-9 Stage4 四个脱敏单元 | 已完成（P3 继续） | 本机 9 项 HMAC ledger 演练保持本地；新增/重写 `feedback_update_eval_v1`、selection completion、ranking/Pareto 与 uncertainty replay 四个纯内存模块，使用匿名组/行投影、字段白名单与失败关闭。ranking/Pareto 限制为最多 256 组/1,024 候选。四个模块及 closure-audit 定向为 91 passed，完整 Stage4 为 119 passed，模块覆盖率依次 89%、88%、90%、90%；全仓为 502 passed、86% coverage，匿名 ZIP 已 verify（124 成员，`a6951ed6310541a0266a14d6af93228fe20313f93f41508231b773e54644d264`），Ruff、compileall、diff 与依赖审计通过（CPU torch 的 PyPI 审计例外已记录）。未推送、未发布。 |
| 2026-08-04 | P3-10 同路径差异候选审计 | 已完成（本地） | 当前公开树重比对发现 39 项同路径不同内容候选，较 P3-8 的 38 项发生计数漂移。逐项审阅后，9 项已由安全公开契约替代，13 项待公开改写，2 项待来源/许可证核查，15 项待逐项安全边界结论；仅前 9 项写入第二个本地 HMAC ledger 子集。加上 P3-9 演练，目前仅 18/1,551 项具有可复核本地决定；私有 inventory、decisions、key 和 ledger 均未提交、未推送。 |
| 2026-08-04 | P3-11 小型 fixture 处置 | 已完成（本地） | 8 项 YAML/JSON fixture 逐项审阅后，7 项由公开 Stage1/Stage2 synthetic test/demo contracts 替代，1 项转入 P4 外部制品契约；相关公开回归为 36 passed。本机路径无关 ledger 含 8 条记录并通过泄露扫描；累计 26/1,551 项有可复核 decisions。私有 inventory、decisions、key 和 ledger 均未提交、未推送。 |
| 2026-08-04 | P3-12 配置候选处置 | 已完成（本地） | 96 项配置逐项审阅后，86 项转入 P6 执行契约、3 项转入 P5 环境契约、6 项转入 P4 外部数据契约、1 项排除；本机路径无关 ledger 通过泄露扫描。累计 122/1,551 项有可复核 decisions；私有 inventory、decisions、key 和 ledger 均未提交、未推送。 |
| 2026-08-04 | P3 发布级重审重置 | 进行中 | 撤回 P3-13--P3-17 的机械批量分流、全量 ledger 和“1,551 项完成”表述：它们不足以证明逐项职责、依赖、许可证和安全输出结论。旧本机机械 artifacts 已隔离，不能作为新账本输入。P2b 基线未变；从 P3 重审协议开始，每项必须保留受限证据并按小批次独立复核。未推送、未发布。 |
| 2026-08-04 | P3-18 合成 fixture 发布级复审 | 已完成（本地） | 8 项维护者授权的合成 fixture 完成逐项内容、职责、来源授权、公开消费者和安全输出复核；7 项原样迁入、1 项增加合成/非论文证据元数据后迁入。新增测试先失败后通过；Stage1 与 release 安全回归为 79 passed。仅有新的 8/1,551 项可接受 decisions，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-19 外部边界发布级复审 | 已完成（本地） | 5 项外部制品/环境/执行相关候选逐项完成内容、调用关系、来源/许可证和安全输出审查：1 项各归 P4/P5/P6，1 项由公开 capability parser/测试替代，1 项因用户专属运行时路径阻塞。私有逻辑回归为 52 passed，公开替代契约为 6 passed；没有迁入源码或真实制品。新的 re-audit 累计仅 13/1,551 项可接受 decisions，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-20 正式执行与私有测试边界复审 | 已完成（本地） | 12 项私有执行/测试候选逐项完成职责、导入/调用、来源/许可证与安全输出审查：8 项进入 P6、1 项进入 P4、3 项因用户专属挂载路径保持 P7 阻塞。独立抽样复审覆盖三种 disposition 并通过；未运行私有执行器、未迁入源码或真实制品。新的 re-audit 累计仅 25/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-21 外部执行脚本复审 | 已完成（本地） | 12 项外部执行脚本逐项完成职责、导入/调用、来源/许可证与安全输出审查：7 项进入 P6、3 项进入 P4、2 项因用户专属仓库根路径保持 P7 阻塞。token-like 静态命中经上下文排除为凭据误报；独立抽样复审覆盖三种 disposition 并通过。未运行私有执行器、未迁入源码或真实制品；新的 re-audit 累计仅 37/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-22 运行时编排与上游配置复审 | 已完成（本地） | 12 项运行时/上游候选逐项完成职责、导入/调用、来源/许可证与安全输出审查：9 项进入 P6、1 项进入 P4、2 项因非维护者来源且许可/NOTICE 不明保持 P7 阻塞。token 静态匹配经上下文排除为进程参数误报；独立抽样复审覆盖三种 disposition 并通过。未运行私有运行时、未迁入源码或真实制品；新的 re-audit 累计仅 49/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-23 上游配置与外部物化复审 | 已完成（本地） | 12 项逐项完成职责、调用、来源/许可证与安全输出审查：8 项进入 P4 外部配置契约、4 项进入 P6 外部物化/执行契约。上游代码的 Apache-2.0 依据不覆盖数据集或 checkpoint 的再分发条件。未运行执行器、未迁入源码或真实制品；新的 re-audit 累计仅 61/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-24 Stage2 制品与执行边界复审 | 已完成（本地） | 12 项逐项完成职责、调用、来源/许可证与安全输出审查：5 项进入 P4、3 项进入 P6、4 项因动态解析、私有运行根或派生模型/制品权限保持 P7 阻塞。未运行执行器、未迁入源码或真实制品；新的 re-audit 累计仅 73/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-25 TRT、环境清单与遗留入口复审 | 已完成（本地） | 12 项逐项审阅后，3 项进入 P4、2 项进入 P5、4 项进入 P6、2 项保持 P7 阻塞、1 项由公开 Stage4 契约替代。4 项在入队后内容漂移，全部按当前内容重新绑定受限证据并复审，未沿用旧摘要。未运行执行器、未迁入源码或真实制品；新的 re-audit 累计仅 85/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-26 公开替代与执行边界复审 | 已完成（本地） | 独立复审接受 12 项逐项结论：5 项已有公开实现或更安全替代，1 项进入 P4，6 项进入 P6。冲突的本地 records 已保留并由 supersession manifest 排除，唯一规范 records 当前内容复验和账本重建通过；未运行或迁入私有执行面。P3-28--P3-30 已先完成，解决冲突后当前可接受总数为 133/1,551；未推送、未发布。 |
| 2026-08-04 | P3-27 旧搜索基础模块与真实 ablation 边界复审 | 已完成（本地） | 独立复审接受 12 项逐项结论：10 项由公开契约替代，1 项进入 P4，1 项进入 P6。冲突的本地 records 已保留并由 supersession manifest 排除，唯一规范 records 当前内容复验和账本重建通过；未运行或迁入私有执行面。当前可接受总数为 145/1,551；未推送、未发布。 |
| 2026-08-04 | P3-28 公开重写与 Stage6 evidence 边界复审 | 已完成（本地） | 12 项逐项审阅并通过独立样本复审：7 项已有公开 release-safe 重写，2 项进入 P4，2 项进入 P6，1 项进入 P5。当前内容漂移为 0；未运行执行器、未迁入新的源码或真实制品；新的 re-audit 累计仅 97/1,551 项，P4 尚未开始；未推送、未发布。 |
| 2026-08-04 | P3-29 Stage6/Stage7 契约边界复审 | 已完成（本地） | 12 项独立逐项审阅后，3 项由已测试公开重写替代、5 项进入 P4、2 项进入 P5、2 项进入 P6。没有基于路径或关键词自动认定替代；未运行或迁入私有执行面。本批关闭时总数为 109/1,551，P3-26/P3-27 后续解决冲突并被接受后当前为 145/1,551；未推送、未发布。 |
| 2026-08-04 | P3-30 Stage7 执行与测试边界复审 | 已完成（本地） | 12 项独立逐项审阅后，8 项进入 P4、1 项进入 P5、2 项进入 P6、1 项由公开预测器工作流和测试链替代。全部重新绑定当前内容，未运行或迁入私有执行面。本批关闭时总数为 121/1,551，P3-26/P3-27 后续解决冲突并被接受后当前为 145/1,551；未推送、未发布。 |
| 2026-08-04 | P3-31 F-Cooper 测试与证据边界复审 | 已接受（本地） | 12 项完成当前内容、P2 bytes、职责、调用、公开反证、来源/许可证与安全输出审查，并通过独立语义复审：4 项转 P4、3 项转 P5、5 项转 P6。受限 HMAC ledger 可重新计算；未运行或迁入私有执行面。本批使已接受总数到 157/1,551；未推送、未发布。 |
| 2026-08-04 | P3-32 F-Cooper TVM 测试边界复审 | 已接受（本地） | 主审与独立语义复审均确认 12 项无内容漂移：5 项转 P4、2 项转 P5、5 项转 P6；逐项 evidence/decision 和 HMAC ledger 已重新绑定并可重新计算。该批使已接受总数到 193/1,551；未推送、未发布。 |
| 2026-08-04 | P3-33 执行环境与制品边界复审 | 已接受（本地） | 12 项完成当前内容、P2 bytes、职责、调用、公开反证、来源/许可证与安全输出审查，并通过独立语义复审：4 项转 P4、2 项转 P5、3 项转 P6、3 项待公开合成重写。受限 HMAC ledger 可重新计算；未运行或迁入私有执行面。本批使已接受总数到 169/1,551；未推送、未发布。 |
| 2026-08-04 | P3-34 Stage2 覆盖与导出边界复审 | 已接受（本地） | 12 项完成当前内容、P2 bytes、职责、调用、公开反证、来源/许可证与安全输出审查，并通过独立语义复审：1 项转 P4、2 项转 P5、5 项转 P6、4 项待公开合成重写。受限 HMAC ledger 可重新计算；当前已接受总数为 181/1,551；未推送、未发布。 |
| 2026-08-04 | P3-35 Stage2 测试边界复审 | 已接受（本地） | 前两次逐项审阅的 8 项主导边界不一致；第三次裁决逐项确认公开 Stage2 contract/demo 测试仅部分覆盖，不能误判为公开替代，最终为 5 项 P4、2 项 P5、5 项 P6。公开 Stage2 定向回归为 20 passed；受限 evidence/decision 和 HMAC ledger 已重新绑定并可重新计算。该批使已接受总数到 205/1,551；未推送、未发布。 |
| 2026-08-04 | P3-36 Stage2 外部执行线索复审 | 已接受（本地） | 两项公开替代分歧经第三次职责/API/测试裁决解决：公开同路径实现和 demo pipeline 覆盖核心 ablation/search 责任，成为 2 项公开替代；其余为 5 项 P4、2 项 P5、3 项 P6。公开 demo pipeline 定向回归为 9 passed；受限 evidence/decision 和 HMAC ledger 已重新绑定并可重新计算。8 条为对 P3-19/P3-20 既有身份的复审，因此真实唯一覆盖仅增至 209/1,551；未推送、未发布。 |
| 2026-08-04 | P3-37 Stage2/Stage4 运行边界复审 | 已接受（本地） | 12 项主审与独立语义复审一致：5 项 P4、3 项 P5、4 项 P6；受限 evidence/decision 与 HMAC ledger 已重新绑定并可重新计算。1 条为对 P3-20 既有身份的复审，因此真实唯一覆盖增至 220/1,551；未推送、未发布。 |
| 2026-08-04 | P3-38 Stage35 完成与证据边界复审 | 已接受（本地） | 12 项经历主审、独立复审和两个 P4/P6 finalizer 分歧的逐项测试/入口裁决，最终为 6 项 P4、6 项 P6。受限 evidence/decision 和 HMAC ledger 已重新绑定并可重新计算；该批使真实唯一覆盖到 232/1,551；未推送、未发布。 |
| 2026-08-04 | P3-39 Stage35 Gold 队列与充分性复审 | 已接受（本地） | 12 项主审与独立语义复审一致：6 项 P4、6 项 P6；公开树无同路径或职责等价实现。受限 evidence/decision 和 HMAC ledger 已重新绑定并可重新计算；该批使真实唯一覆盖到 244/1,551；未推送、未发布。 |
| 2026-08-05 | P3-43 Stage5/Stage6 执行与外部证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查；双审对 10 项一致，第 4、5 项经第三次裁决归 P6。最终 4 项 P4、8 项 P6，新增 12 个不重复身份；HMAC ledger 已重新计算，真实唯一覆盖为 290/1,551；未推送、未发布。 |
| 2026-08-05 | P3-44 Stage6 证据、环境与执行边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 5 项 P4、1 项 P5、4 项 P6、2 项发布安全公开改写；第 4、6、10 项经第三次裁决。新增 12 个不重复身份，HMAC ledger 已重新计算，真实唯一覆盖为 302/1,551；未推送、未发布。 |
| 2026-08-05 | P3-45 Stage7 执行、反馈与证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 5 项 P4、7 项 P6；第 1、7、8 项经第三次裁决。新增 12 个不重复身份，HMAC ledger 已重新计算，真实唯一覆盖为 314/1,551；未推送、未发布。 |
| 2026-08-05 | P3-46 Stage7 环境、执行与最终证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。逐项双审一致；独立审阅汇总笔误已按索引复核。最终 4 项 P4、5 项 P5、3 项 P6，HMAC ledger 已重新计算，真实唯一覆盖为 326/1,551；未推送、未发布。 |
| 2026-08-05 | P3-47 Stage7 source 与 feedback 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 5 项 P4、3 项 P5、4 项 P6；第 2、9 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 338/1,551；未推送、未发布。 |
| 2026-08-05 | P3-48 Stage2 执行与外部制品边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 3 项 P4、8 项 P6、1 项非必要排除；第 1、8、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 350/1,551；未推送、未发布。 |
| 2026-08-05 | P3-49 Stage7 deployment 与 TVM 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 2 项 P4、1 项 P5、8 项 P6、1 项非必要排除；第 1、6、10 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 362/1,551；未推送、未发布。 |
| 2026-08-05 | P3-50 release 与 deployment 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 3 项 P4、5 项 P6、2 项公开等价替代、2 项 P7 许可/隐私阻塞；第 1、4、5 项经第三次裁决。此前 373 条 accepted records 的 P2/current-SHA/HMAC 重建验证通过，新增 12 个身份后真实唯一覆盖为 374/1,551；未推送、未发布。 |
| 2026-08-05 | P3-51 results 与 runtime 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 10 项 P4、1 项 P5、1 项 P6；第 11、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 386/1,551；未推送、未发布。 |
| 2026-08-05 | P3-52 training 与 plugin 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 3 项 P4、9 项 P6；第 1--4、7 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 398/1,551；未推送、未发布。 |
| 2026-08-05 | P3-53 model 与 plugin 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。12 项均归 P6；双审逐项一致。OpenMMLab/TRT 适配来源留待 P6 前 NOTICE/provenance 审查。HMAC ledger 已重新计算，真实唯一覆盖为 410/1,551；未推送、未发布。 |
| 2026-08-05 | P3-54 quantization 与 evidence 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 3 项 P4、9 项 P6；GPU exclusivity gate 经第三次裁决归 P6。HMAC ledger 已重新计算，真实唯一覆盖为 422/1,551；未推送、未发布。 |
| 2026-08-05 | P3-55 recovery 与 TVM 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 3 项 P4、9 项 P6；双审逐项一致。HMAC ledger 已重新计算，真实唯一覆盖为 434/1,551；未推送、未发布。 |
| 2026-08-05 | P3-56 DP4A 与 baseline 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 7 项 P4、5 项 P6；第 10、11 项经第三次裁决归 P4。HMAC ledger 已重新计算，真实唯一覆盖为 446/1,551；未推送、未发布。 |
| 2026-08-05 | P3-57 engine 与 baseline 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 9 项 P4、3 项 P6；双审逐项一致，无需第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 458/1,551；未推送、未发布。 |
| 2026-08-05 | P3-58 公开表面与证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 7 项 P4、2 项已公开迁入、2 项已安全改写、1 项非必要排除；第 1、2、11、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 470/1,551；未推送、未发布。 |
| 2026-08-05 | P3-59 合同与证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 6 项已安全改写、5 项 P4、1 项 P7 阻塞；第 7 项经第三次裁决，未来可改写不构成当前已改写。HMAC ledger 已重新计算，真实唯一覆盖为 482/1,551；未推送、未发布。 |
| 2026-08-05 | P3-60 Stage5--Stage7 边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 7 项已安全改写、2 项 P4、3 项 P5；第 4、10、11、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 494/1,551；未推送、未发布。 |
| 2026-08-05 | P3-61 遗留测试边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 11 项非必要排除、1 项公开替代；第 2--5、8--12 项经第三次裁决，旧测试未因被测脚本的潜在执行边界而误转 P4/P6。HMAC ledger 已重新计算，真实唯一覆盖为 506/1,551；未推送、未发布。 |
| 2026-08-05 | P3-62 遗留执行测试边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 11 项非必要排除、1 项公开替代；第 1--9、11、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 518/1,551；未推送、未发布。 |
| 2026-08-05 | P3-63 Stage1/Stage2 遗留测试复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 7 项公开替代、5 项非必要排除；第 3、4、6、10 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 530/1,551；未推送、未发布。 |
| 2026-08-05 | P3-64 Stage1--Stage4 遗留测试复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 1 项 P7 阻塞、2 项公开替代、9 项非必要排除；第 4--10、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 542/1,551；未推送、未发布。 |
| 2026-08-05 | P3-65 Stage5/Stage6 遗留测试复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 2 项公开替代、10 项非必要排除；双审逐项一致。HMAC ledger 已重新计算，真实唯一覆盖为 554/1,551；未推送、未发布。 |
| 2026-08-05 | P3-66 Stage6/Stage7 遗留测试复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 2 项公开替代、10 项非必要排除；第 3 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 566/1,551；未推送、未发布。 |
| 2026-08-05 | P3-67 deployment 与论文证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 2 项 P4、1 项 P5、8 项 P6、1 项 P7；第 2--12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 578/1,551；未推送、未发布。 |
| 2026-08-05 | P3-68 deployment、metrics 与证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 2 项 P4、5 项 P6、4 项 P7、1 项非必要排除；第 1--5、11、12 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 590/1,551；未推送、未发布。 |
| 2026-08-05 | P3-69 UniV2X 模型支持边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 1 项 P4、9 项 P6、2 项非必要排除；第 2、3 项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 602/1,551；未推送、未发布。 |
| 2026-08-05 | P3-70 UniV2X 执行支持边界复审 | 已接受（本地） | 12 项在重新绑定实际队列后完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查；主审队列身份错误已整体作废，复核后双审一致。最终 12 项 P6，HMAC ledger 已重新计算，真实唯一覆盖为 614/1,551；未推送、未发布。 |
| 2026-08-05 | P3-71 UniV2X tracking 与执行支持边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查；双审逐项一致。最终 12 项 P6，HMAC ledger 已重新计算，真实唯一覆盖为 626/1,551；未推送、未发布。 |
| 2026-08-05 | P3-72 pruning 与量化边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 6 项 P6、6 项 P7；后六项经第三次许可裁决。HMAC ledger 已重新计算，真实唯一覆盖为 638/1,551；未推送、未发布。 |
| 2026-08-05 | P3-73 evidence 与量化边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 6 项 P4、2 项 P6、3 项 P7、1 项非必要排除；q6、q8、q11 经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 650/1,551；未推送、未发布。 |
| 2026-08-05 | P3-74 attention 执行与证据边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 5 项 P4、3 项 P6、2 项 P7、3 项已安全公开改写；六项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 662/1,551；未推送、未发布。 |
| 2026-08-05 | P3-75 dataset 与能耗执行边界复审 | 已接受（本地） | 12 项完成 P2/当前内容、职责、调用、公开反证、项目级许可与安全输出审查。最终 4 项 P4、6 项 P6、2 项 P7；七项经第三次裁决。HMAC ledger 已重新计算，真实唯一覆盖为 674/1,551；未推送、未发布。 |

## 未执行的外部动作与后续授权

本审计未推送、创建 PR、合并、发布制品、上传数据、访问外部硬件或修改第三方资源。任何下列行动须在新的明确授权后执行：

- 推送提交、创建/合并 PR、打 tag 或发布；
- 运行真实 GPU、设备、TVM/TensorRT、AP、延迟或能耗实验；
- 接入外部数据、模型、检查点或硬件日志；
- 为后续新发现的 declared dependency 漏洞调整约束并进行兼容性/安全回归，或将 scoped pip-audit 纳入持续 CI 门禁。
