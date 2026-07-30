# Task 6（Stage7 四变体选择与统计核心）实施报告

## 结论

已发布仅包含 selection/statistics 的 Stage7 公开合同：`full`、`without_surrogate`、`without_measured_feedback`、`backend_blind`，固定三 seed（`20260718`、`20260719`、`20260720`）和四轮×四候选（`T=16`）。实现中保留真实策略名称 `predicted_frontier_diversity`；没有任何位置将其称为 NSGA-II。

## 冻结源与起点核验

实施前 HEAD：`54839707c299815f3ad9e756044eb0eec1f9bb12`（一致）。实际执行：

```text
sha256sum /home/jichengzhi/V2X/framework/stage7/online_component_ablation_v1.py \
  /home/jichengzhi/V2X/framework/stage7/search_policy_v1.py \
  /home/jichengzhi/V2X/framework/stage7/ablation_statistics_v2.py
```

结果逐项一致：

```text
ac2785f86f9cbeeac8d623eeb47bce762966e58f99b4637f134b6971e2af8960  online_component_ablation_v1.py
23207c57f609f8f127b2b0fa4ba3d44d16bee817e973f596380fcf8597b0ba7c  search_policy_v1.py
27d2f7436205675ce79d676db387def813994e96814b42f7482a2dadff4252a0  ablation_statistics_v2.py
```

冻结源中的统计模块依赖 Task6 范围外的 Stage7 actual/execution/formal-evidence 模块。因此没有复制不同源；以已核验冻结源为基线，受限迁移了其中 selection/statistics 合同，剔除了不允许发布的执行路径与其依赖。

## TDD：RED → GREEN

1. 先新增 `tests/stage7/test_ablation_contracts.py`、`test_search_policy.py`、`test_statistics.py`。
   - RED 命令：`pytest tests/stage7 -q`
   - RED 结果：3 个收集错误，均为预期的 `ModuleNotFoundError: No module named 'framework.stage7'`。
2. 最小实现 `framework/stage7/__init__.py`、三个核心模块后重跑。
   - GREEN 命令：`pytest tests/stage7 -q`
   - GREEN 结果：`11 passed`。
3. 先新增 CLI 边界测试 `tests/stage7/test_selection_cli.py`。
   - RED 命令：`pytest tests/stage7/test_selection_cli.py -q`
   - RED 结果：预期失败，`stage7_selection.py` 不存在。
4. 实现 CLI 后重跑。
   - GREEN 结果：初始 Stage7 全集为 `16 passed`。

覆盖补充测试覆盖 JSON/JSONL、输出幂等且拒绝不同内容覆盖、路径约束、A2 显式 frozen bundle/feedback、完整 192 终态事件矩阵、CLI 错误边界。测试均使用真实模块与临时显式文件，不 mock 选择或统计行为。

审查后追加一轮 TDD：先增加 mAP/AP70/delta_hv 标签逃逸、backend-blind 实际选点 score 污染、A2 自一致替换 bundle、later-round 缺失 previous request、以及 CLI 反馈身份错误测试。RED 命令 `pytest tests/stage7/test_ablation_contracts.py tests/stage7/test_search_policy.py tests/stage7/test_selection_cli.py -q` 得到预期 `8 failed`（缺口均对应新合同）；修复后 `pytest tests/stage7 -q` 为 `21 passed`。

## 分段差异

- `framework/stage7/online_component_ablation_v1.py`
  - 公开四个变体及单变量隔离审计；固定 seeds/B=4/rounds=4/T=16。
  - A1 使用局部 `random.Random(seed).sample` 均匀无放回。
  - A2 只记录反馈；训练视图、图特征视图、bundle SHA 均保持冻结，`feedback_refit=False`。
  - backend-blind 仅根据 backend/capability/profile/dispatch 派生 provenance 删除特征；保留静态模型和图特征。
  - `selection_candidate_view` 在选择前拒绝 latency、energy、AP/AP70、mAP/map_score、delta_hv、terminal、failure、cache、Pareto/frontier 标签。
- `framework/stage7/search_policy_v1.py`
  - 生成确定性四项 selection request；排除已选 ID。
  - `full` / A2 / backend-blind 使用真实名称 `predicted_frontier_diversity`；A1 不调用 surrogate。
  - 反馈必须为四条、唯一 row ID、终态、并精确匹配显式 previous request 的 `request_identity`；实际 selection 路径调用该验证。
  - A2 later round 还必须接受调用者显式提供的 round-0 `frozen_payload_sha256`，并再次经过 label firewall；自一致但不同的 replacement bundle 被拒绝。
  - backend-blind 实际选点调用 blind 投影，按 blind numeric matrix 重算排序，完全不消费调用者的 `score`/`acquisition_score`。
  - trajectory/request binding 使用 canonical SHA，拒绝变体、seed、round 或 request 漂移。
- `framework/stage7/ablation_statistics_v2.py`
  - 无缺省补齐：强制 12 条 trajectory、192 条唯一 terminal events 的 4×3×4×4 矩阵。
  - 只生成 DeltaHV-AUC、mean、sample std、paired delta、median/range、direction consistency；显式 `significance_tests_performed=False`。
- `scripts/reproduce/stage7_selection.py`
  - 仅提供 `select` 和 `summarize`；所有输入和输出路径由调用者显式传入。
  - 拒绝 symlink/父路径逃逸、目录覆盖、不同输出覆盖；同字节输出可幂等重跑。
  - `summarize` 仅读取显式 root 下 `events.jsonl`，不完整合同非零退出。
- `.github/workflows/ci.yml`
  - Ruff 范围扩展到 Task7 文件和测试。
  - 添加 Stage7 focused coverage 与逐文件 80% 门禁；未启用 Task9 匿名归档或 Task11 全局覆盖门禁。

未引入 GPU leasing、cache execution、TVM build、AP evaluation、cluster orchestration、私有运行路径。

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `pytest tests/stage7 -q`（RED-1） | 预期 3 个缺模块收集错误 |
| `pytest tests/stage7 -q`（GREEN，审查修复后） | `21 passed in 0.46s` |
| `pytest tests/stage7/test_selection_cli.py -q`（RED-2） | 预期因 CLI 文件不存在而失败 |
| `coverage run -m pytest -q tests/stage7` | `21 passed` |
| `ruff check framework/stage7 scripts/reproduce/stage7_selection.py tests/stage7` | `All checks passed!` |
| `python -m compileall -q framework/stage7 scripts/reproduce/stage7_selection.py` | 退出 0 |
| `python scripts/reproduce/stage7_selection.py --help` | 退出 0 |
| `python scripts/reproduce/stage7_selection.py select --help` | 退出 0，列出四变体与三 seeds |
| `python scripts/reproduce/stage7_selection.py summarize --help` | 退出 0 |
| `pytest -q tests/stage1 tests/stage2 tests/stage4 tests/stage5 tests/stage6 tests/stage7` | `219 passed in 62.31s` |
| `python -c 'import yaml; ...safe_load(.github/workflows/ci.yml)...'` | `ci yaml: valid` |
| `git diff --check` | 退出 0，无输出 |

确定性双跑由 `test_select_requires_explicit_files_and_is_idempotent` 执行同一 CLI `select` 两次：两次返回值均为 0，第二次验证同字节产物幂等。

## 逐文件覆盖率

最终 `coverage json`（Stage7 focused）结果：

| 文件 | 覆盖率 |
|---|---:|
| `framework/stage7/online_component_ablation_v1.py` | 85.96% |
| `framework/stage7/search_policy_v1.py` | 87.86% |
| `framework/stage7/ablation_statistics_v2.py` | 89.23% |
| `scripts/reproduce/stage7_selection.py` | 83.69% |

每个新增生产文件均不低于 80%。

## 自审

- 四变体而非第五个 scanner 变体；full 计入四条 trajectory variants。
- A1 是局部 RNG 的均匀无放回；A2 不在线 refit；backend-blind 不删除静态/图特征。
- backend-blind 实际排序仅消费 blind 投影；不会消费调用者预计算 score。
- A2 later round 锁定 round-0 bundle SHA 并重新执行 label firewall。
- selection 前拒绝 label/cache（含 mAP/map_score/delta_hv）；反馈精确绑定前一显式请求身份。
- 汇总缺 12 trajectories 或 192 terminal events 时失败，不使用缺失值或 surrogate 补齐。
- 统计无 p 值、置信区间或其他显著性检验。
- 文件改动只在分配的 Stage7、CLI、tests、CI 和本报告内；未观察到并行代理冲突。

## 提交

初始提交：`cac69b410a376c492afcb73a777852519011107d`（`feat: publish online ablation selection contracts`）。

## Fix Round 1（独立复审修复）

独立复审确认初版存在四个问题：A2 later round 由调用者 pool 顺序重建排名；request/feedback 只检查 64 位摘要形状；full/A2/backend-blind 没有调用 Stage5 的真实 acquisition；以及 backend-blind 只在隔离 helper 中投影。本轮先完成根因确认：`framework.stage5.single_target_search_v2.select_task_batch` 是仓库中 `predicted_frontier_diversity` 的唯一实现，负责 predicted frontier、objective anchors、uncertainty 和 feature diversity；它需要验证过的 `SearchTask`、预测候选、measured rows 与 graph features。旧 Stage7 的 `score`/numeric-sum 排序绕开了该实现。

### 新增 RED → GREEN

1. 将 `tests/stage7/test_search_policy.py` 改为构造完整的 Stage5-compatible S7 task/candidate/prediction fixture，并增加：A2 反转 caller pool 不变、later 排除已选 ID、伪造 request rows/variant/seed/round 拒绝、backend-blind 不消费 caller score。
   - RED：`pytest tests/stage7/test_search_policy.py -q`
   - 结果：`11 failed`，均为旧 API 没有 task 与真实 Stage5 selection 合同。
2. 将 CLI 测试改为显式传入 `--search-task`、`--measured-rows`、`--measured-graph-features`。
   - RED：`pytest tests/stage7/test_selection_cli.py -q`
   - 结果：`3 failed`，CLI 尚未接受新的显式输入。
3. 修复后：`pytest tests/stage7 -q` → `26 passed`。

### 修复内容

- `select_stage7_round` 现在必须接收冻结的 Stage7 `SearchTask` 与显式 measured rows/graph features。full、A2、backend-blind 均直接调用 `framework.stage5.single_target_search_v2.select_task_batch`；Stage7 没有复制、重写或 fallback 到 score/sum acquisition。
- A2 round 0 冻结按 row identity 排序的 predictions、frozen measured views 及 round-0 Stage5 acquisition；later round 先校验 caller candidate identity set 与 frozen payload SHA，再只从 frozen candidate predictions 中排除已选 ID，并以冻结 measured views 调用同一 `select_task_batch`。caller pool 反转不会影响结果。
- backend-blind 先删除 backend/capability/profile/dispatch 派生 graph/model/features/provenance，再将投影后的候选和 measured inputs 交给同一 Stage5 selector；caller `score` 未被 Stage7 acquisition 消费。
- Stage7 request 的 `request_identity` 与 `measurement_request_sha256` 都明确表示同一 canonical body（schema、variant、seed、round、task SHA、Stage5 request SHA、rows、selected row IDs）。`validate_measurement_request` 每次重算两者；feedback 与 request binding 均先调用它。伪造 rows/variant/seed/round 或任一 identity 均 fail-closed。
- CLI 新增三个严格的显式 JSON 输入。伪造 previous request 的 CLI 回归验证非零退出，不含 traceback 或调用者路径。

### Fix Round 1 验证

| 命令 | 结果 |
|---|---|
| `pytest tests/stage7 -q` | `26 passed in 8.87s` |
| `coverage run -m pytest -q tests/stage7` | `26 passed` |
| `pytest -q tests/stage1 tests/stage2 tests/stage4 tests/stage5 tests/stage6 tests/stage7` | `224 passed in 68.25s` |
| Stage7 per-file coverage | ablation 85.96%、policy 88.89%、statistics 89.23%、CLI 82.91% |
| `ruff check framework/stage7 scripts/reproduce/stage7_selection.py tests/stage7` | `All checks passed!` |
| `python -m compileall -q framework/stage7 scripts/reproduce/stage7_selection.py` | 退出 0 |
| 三个 CLI help 命令与 `git diff --check` | 退出 0 |

Fix Round 1 commit：`fix: preserve stage7 acquisition identities`（最终 SHA 由本次提交后的版本控制历史记录）。

Fix Round 1 独立只读复审：无新增 finding；复审确认三条非 A1 路径均经 `_stage5_acquisition` 调用 Stage5 `select_task_batch`，A2 SHA/候选 identity/exclusion 生效，canonical feedback 验证与 CLI 脱敏测试均覆盖。
