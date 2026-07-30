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

## 未执行的外部动作与后续授权

本审计未推送、创建 PR、合并、发布制品、上传数据、访问外部硬件或修改第三方资源。任何下列行动须在新的明确授权后执行：

- 推送提交、创建/合并 PR、打 tag 或发布；
- 运行真实 GPU、设备、TVM/TensorRT、AP、延迟或能耗实验；
- 接入外部数据、模型、检查点或硬件日志；
- 为后续新发现的 declared dependency 漏洞调整约束并进行兼容性/安全回归，或将 scoped pip-audit 纳入持续 CI 门禁。
