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

**状态日期：2026-08-03；此处记录的是台账提交前的已验证工作树状态。** 当前工作分支为 `release/aaai27-reproducibility`，最后一个功能提交为 `9df8299`（干净克隆 CPU smoke）；分支相对配置的远端发布分支尚有 5 个本地提交，尚未推送。该数字在推送或新增提交后必须更新。

| 范围 | 当前状态 | 可验证证据 | 仍缺少的内容 |
| --- | --- | --- | --- |
| 公开 CPU 冒烟闭环 | 已完成 | `requirements.txt` 固定 CPU 依赖；`scripts/reproduce/smoke_clean_clone.sh` 创建全新 clone/venv 后运行 smoke；当前分支上曾以隔离本地 Git 源实际通过。 | 真实公开远端在推送后仍需由 CI 再验证。 |
| 公开文档与新手入口 | 已完成 | `README.md`、`README.zh-CN.md`、`REPRODUCIBILITY.md` 说明浅克隆、固定依赖和 smoke；匿名 README 使用同一依赖入口。 | 全量训练/硬件运行说明尚未迁入。 |
| 匿名审稿 ZIP | 已完成，但需随发布候选重建 | allowlist、逐字节安全扫描、archive verifier 和既有审计记录均在仓库。 | 旧 ZIP 哈希仅对应旧审计提交；每次候选 HEAD 改变都必须重建并记录新哈希。 |
| 论文证据 | 受限 | 小型 Stage4 审计为 `verified`；demo 明确为非论文证据。规范定义见 [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) 与 [ARTIFACTS.md](../ARTIFACTS.md)。 | Stage6/Stage7 及真实硬件/AP/能耗结果仍为 `external` 或 `unavailable`。 |
| 完整项目源代码 | 未完成 | 当前树含公开的接口、验证、选择和聚合逻辑。 | 私有完整训练、模型物化、硬件调度/测量和正式实验执行代码尚未逐项脱敏、迁入并验收。 |
| 公开发布 | 未开始 | 本地发布候选已存在，尚未执行外部推送或可见性变更。 | 需要完成 P1--P8，并在最后取得推送/发布的明确授权。 |

历史的 `338 passed`、覆盖率和 ZIP SHA-256 记录仍是其对应审计提交的证据，不能自动外推到当前 HEAD。当前 HEAD 的每一次新验收结果必须通过本台账追加记录。

### 到完整开源的执行计划

| 编号 | 状态 | 操作 | 交付物与验收条件 |
| --- | --- | --- | --- |
| P0 | 已完成 | 建立本交接台账及其回归测试，并从 README 入口链接；修复本轮真实匿名 ZIP 闭包发现。 | 同一提交包含台账、链接、测试与 CI 修复；346 passed，真实 ZIP build/verify 通过。 |
| P1 | 待开始 | 冻结当前发布候选，重建匿名 ZIP，并对候选源树和解包 ZIP 分别做干净环境验收。 | 记录候选提交、ZIP SHA-256、成员数、依赖版本、测试数、覆盖率、identity scan、clean-clone 结果；旧审计结果不能复用。 |
| P2 | 待开始 | 审计私有完整仓库：按“源码/文档/小型脱敏 fixture/外部数据描述/禁止上传生成物”清点全部路径和依赖关系。 | 机器可读迁移清单；每个保留、改写、外置或排除项有理由、许可证/许可状态和敏感项扫描结果。 |
| P3 | 待开始 | 逐模块脱敏迁移完整 Python、配置和设计 Markdown；将测试所需的小型固定输入迁入受版本控制的 fixture 目录，不再从 `results/` 读取。 | 每个迁入模块有单元/集成测试、相对路径入口、设计理由和从私有运行假设中移除的说明；不含密码、主机地址、个人路径或二进制生成物。 |
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

#### P2--P4：从私有工作树安全迁移

1. 先生成清单，再迁移；不对私有树执行批量提交、历史重写或上传。清单必须将 Markdown/Python 与 PNG/PDF、LaTex 临时文件、压缩包、模型、engine、ONNX、校准缓存和结果目录分开处理。
2. 每个候选文本文件先经凭据、IP、绝对路径、邮箱、私有域名和受限数据扫描；命中后改用环境变量示例、抽象路径或脱敏 fixture，不能仅靠 `.gitignore` 掩盖已跟踪内容。
3. 每个可生成大文件只保留生成脚本、最小参数和输入/输出 SHA-256；每个不可再生成但合法可分发的输入必须先确认许可，之后才可建立下载/校验清单。
4. 新增设计 Markdown 必须说明模块目的、关键决策、输入输出契约、复现命令、已知边界和与论文/实验阶段的关系。

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

## 未执行的外部动作与后续授权

本审计未推送、创建 PR、合并、发布制品、上传数据、访问外部硬件或修改第三方资源。任何下列行动须在新的明确授权后执行：

- 推送提交、创建/合并 PR、打 tag 或发布；
- 运行真实 GPU、设备、TVM/TensorRT、AP、延迟或能耗实验；
- 接入外部数据、模型、检查点或硬件日志；
- 为后续新发现的 declared dependency 漏洞调整约束并进行兼容性/安全回归，或将 scoped pip-audit 纳入持续 CI 门禁。
