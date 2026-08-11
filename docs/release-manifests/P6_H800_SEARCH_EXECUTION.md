# P6 H800 自动搜索执行边界

状态：**进行中（本地）**。本说明固定 P6 第一阶段的公开边界和维护者本地执行约束；它不表示 P6 已完成或论文结果已关闭。

## 范围与非自动化原则

P6 第一阶段只面向 H800，覆盖维护者在已本地准备资产上的候选选择、训练/评测、反馈与汇总闭环。其他本项目硬件配置不属于 P6 论文目标。Orin 是后续的独立执行线，不能由本阶段结果代替。

公开 runner `tools/release/run_p6_h800_search.py` 仅接受明确给出的 `--contract`、`--local-config`、`--public-summary` 与 `--code-revision`。它不下载资产、不自动探测硬件，也不自动启动硬件；维护者必须在本地明确决定何时执行。缺少本地资产、输入不合规或执行失败时，流程失败关闭。

## 公共契约与本地配置

公开 YAML `configs/execution/p6_h800_search.example.yaml` 是可审阅的 H800 执行声明。它只声明 schema、搜索标签、目标模型、随机种子、轮次与批大小、配置标签、指标名称、候选空间标签，以及脱敏资产的标签、版本和许可状态；它不声明本地位置、主机、资产内容或执行细节。

本地 YAML 位于 Git 忽略的 `configs/local/` 目录，且只能经显式 `--local-config` 传入。它承载本地资产和 Stage5 输入的位置、argv 步骤、结果模板和唯一的本地输出根；输出根不从公共 YAML 或默认位置推断。该目录及本地输出均不进入匿名归档。TVM 与 TensorRT 可作为维护者本地 argv 步骤的一部分，但完整 argv 和本地运行记录不公开。

## 公共摘要契约

公共摘要 schema 为 `p6_h800_search_summary_v1`。允许的顶层字段仅为：

- `schema`、`target`、`code_revision`、`seed`、`configuration_label`
- `assets`（每项仅 `label`、`version`、`license_status`）
- `status`、`planned_rounds`、`completed_rounds`、`successful_candidate_count`
- `aggregate_metrics`（每个指标仅 `count`、`min`、`max`、`mean`）
- `failure_code`

摘要可聚合的指标为 `latency_ms`、`energy_j`、`ap30`、`ap50` 与 `ap70`。它不发布路径、主机信息、完整 argv、资产内容、逐轮结果、逐条测量、候选的唯一标识、模型权重或运行记录。

失败时，公开的稳定失败码仅为 `capability_target_invalid`、`command_failed`、`local_input_invalid`、`local_record_write_failed`、`result_candidate_mismatch`、`result_invalid_json`、`result_metrics_invalid`、`result_missing` 与 `stage5_selection_failed`。成功摘要的 `failure_code` 为 `null`。

## 匿名归档与完成准入

匿名归档可以包含公共 runner、`framework/stage6/h800_search_execution_v1.py` 和示例公共契约；它不得包含 `configs/local/`、本地输出或任何 sentinel 运行记录。归档规则仍保留对其余公开 `framework/**`、`tools/release/**` 与 `configs/**` 的既有覆盖。

P6 只有在实际 H800 运行按预算完成、公共摘要通过结构与泄露检查、全量测试和匿名归档验证都完成后，才具备本地关闭的准入条件。在这些条件满足前，P6 始终保持进行中（本地）。
