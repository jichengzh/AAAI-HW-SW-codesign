# P3：完整源码处置与验证关闭计划

状态：进行中；此前机械分流形成的 P3-13--P3-17 结论与全量 ledger 已撤回，不能作为公开发布或 P4 的依据。真实逐项复审按 [P3_REAUDIT_PROTOCOL.md](P3_REAUDIT_PROTOCOL.md) 执行；本文件定义其关闭条件，不代表已完成公开发布。

P3-18--P3-39 的已接受 records 按批次相加曾为 253 条，但该数字不是去重后的覆盖数：P3-36 与 P3-37 中有 9 条重复 P2 身份，因此截至 P3-39 的真实唯一覆盖为 244 项。P3-25 中 4 项在入队快照后发生内容变化，已按当前内容重新读取、重新绑定受限 SHA-256 证据后才保留；P3-26/P3-27 的冲突本地 records 已由保留旧记录的 supersession manifest 解决，唯一规范 records 已重新验证当前内容并重建账本；P3-28--P3-39 中已接受批次当前内容漂移均为 0，且均已有独立语义复审。P3-35 的前两次审阅有 8 个边界分歧，已由第三次逐项裁决、公开 Stage2 契约回归和重新计算账本解决；P3-36 的两项公开替代分歧也已由第三次职责/API/测试裁决解决；P3-38 的两个 P4/P6 finalizer 分歧已由逐项读取测试与被测入口的裁决解决。所有受限 evidence/decision/HMAC ledger 检查点均不构成关闭账本；P4 尚未开始。

## 当前更正（以本节为准）

P3-40 经第三次逐项裁决、P3-41 经公开职责与定向回归验证、P3-42 经主审与独立复审后，三批均完成当前内容绑定和路径无关 HMAC ledger 重算。P3-41 的 6 项 `duplicate_or_superseded` 已逐项由公开 Stage4 实现与测试/API 职责证据支持，定向回归为 119 passed。P3-18--P3-42 records 合计 289 条，但 P3-40 的 2 条重复 P2 身份与此前 9 条重叠合并后，基线唯一覆盖为 278。P3-43--P3-47 依次新增 12 个身份；P3-48 再以 3 项 P4、8 项 P6、1 项非必要排除新增 12 个身份。P3-49 以 2 项 P4、1 项 P5、8 项 P6、1 项非必要排除新增 12 个身份。P3-50 以 3 项 P4、5 项 P6、2 项有公开 API/测试职责证据的 `duplicate_or_superseded` 和 2 项 `blocked_license_or_permission` 新增 12 个身份。P3-51 以 10 项 P4（闭环结果、图表与 TRT/ONNX 制品）、1 项 P5（分布式运行配置）和 1 项 P6（CARLA/CUDA 全流程运行时）新增 12 个身份；P3-52 以 3 项 P4（训练/评估数据、权重与 checkpoint 配置）和 9 项 P6（插件、训练、评测、损失与模型 head 执行面）新增 12 个身份。P3-50 的 release-asset verifier、completed-round deployment rehydrate 与 dependency-pin transaction 均经第三次只读裁决，分别归 P4、P6、P6；P3-51 的两个 TRT/ONNX 配置也经第三次裁决归 P4，因为主导的是 checkpoint-to-ONNX/engine 制品边界而非环境探测；P3-52 的三份训练配置和 plugin/API 入口也分别经第三次裁决归 P4/P6。两个外部 CUDA patcher 保持 P7 许可/隐私阻塞，不能降为非必要。历史 373 条 accepted records 亦已按其规范 manifest/ledger 组合重建：P2 metadata、当前 SHA-256 和 HMAC 均重新验证，去重为 362；P3-52 后当前真实唯一覆盖为 **398/1,551**。P3 仍在进行中，P4 尚未开始。详见 [覆盖身份对账](P3_COVERAGE_IDENTITY_RECONCILIATION.md)、[P3-51 记录](P3_BATCH_51_RESULTS_AND_RUNTIME_REAUDIT.md) 与 [P3-52 记录](P3_BATCH_52_TRAINING_AND_PLUGIN_REAUDIT.md)。

## 冻结基线

P3 以 P2b 的私有源只读盘点为唯一候选基线：原始盘点 SHA-256 为 `6fafdace7dc0eae3e9982763cb1c5c098c0ae83d1715af67ff1e08d7a1d12a79`，共 1,551 个候选：1,008 个代码、96 个配置、439 个 Markdown 和 8 个小 fixture。另有 3,664 个生成物、外部输入或敏感/特殊项，它们不直接进入 P3。两个嵌套仓库保持独立边界，不能默认合并。

每一次 P3 批次开始前必须用 v1.1.0 inventory 工具在受控临时目录重跑盘点；若基线哈希或计数改变，先记录增量并处置新增/删除项。原始逐路径报告、私有路径和其映射绝不提交。静态扫描、AST、目录和扩展名仅可用于生成复审线索，不能代替逐项职责、依赖、来源和许可证证据。

## 处置账本

关闭前必须由 `build_p3_disposition_ledger.py` 校验一个本地逐路径 decisions 文件，并输出可提交的脱敏账本。维护者提供私有 HMAC key；公开账本只保存 HMAC-SHA256 派生的不可逆 `candidate_id`，不得保存路径、内容、地址、凭据、人员标识或该 key。

每个候选恰好一条、且仅可使用以下决定之一：

- `migrated_public`：可公开迁入并有测试；
- `rewritten_public`：语义保留但已脱敏改写；
- `external_contract_p4`：外部数据、模型或大文件已有获取/校验契约；
- `environment_contract_p5`：环境、驱动或硬件约束已有公开配置；
- `execution_contract_p6`：训练、测量或汇总执行器已有公开执行契约；
- `duplicate_or_superseded`：已有公开等价物；
- `excluded_nonessential`：与支持的复现链无关，且已记录依赖反证；
- `blocked_license_or_permission`：不可公开；若其为复现关键依赖，P3 不得关闭。

每条公开记录还须绑定 P2 分类、批次、固定 reason code、许可结论、公开替代物/契约、测试或审阅证据及后续阶段。每个本机 decision 还必须遵循 P3_REAUDIT_PROTOCOL 的内容 hash、职责、调用关系、来源/许可证和安全输出证据要求。账本不得含 `pending` 或 `reviewing`；计数必须精确回填至 1,551。

## 执行链关闭

每一项支持的公开流程必须建立“设计说明 → 模块 → 命令/API → 合成测试 → 输入契约”的映射。迁入或改写的模块只能使用脱敏小输入；默认 CPU smoke 不读取维护者目录、`results/`、缓存、GPU 或外部硬件。无法提供的外部输入必须以明确的 `unavailable` 失败关闭，而不是以 demo 替代论文结果。

当前 CPU smoke 已满足上述离线边界，但只复现 scoped smoke。正式 Stage6 终端证据、Stage7 轨迹/聚合、训练物化、完整数据和硬件执行仍为 external/unavailable；这些事项不因 P3 完成而被视为已复现，分别由 P4--P6 处理。

公开执行面仍须逐一处置非 smoke 历史入口：每个入口必须被标为有完整公开契约，或被明确外置/延后。P3-7 已修复已发现的个人绝对路径、生成物忽略缺口和过时输出说明；其余入口的全量处置仍由本账本约束。

## P3 关闭验证

只有同时满足以下条件才可将 P3 标记为“已完成（本地）”：

1. 脱敏账本由当前 P2b 基线生成，1,551 条全部处置且分类/总数对账通过；
2. 所有 `migrated_public`、`rewritten_public`、`duplicate_or_superseded` 和 `excluded_nonessential` 都有公开证据；转交 P4--P6 的关键项已有可验证契约；
3. 账本机械测试、全量 pytest（覆盖率不少于 80%）、Ruff、`compileall`、身份/凭据扫描、匿名 ZIP build/verify 均针对同一 HEAD 通过；
4. 独立复审确认不存在路径/身份/凭据回显、隐式本机依赖或未处置候选。

完成后在总台账记录候选基线、提交、命令、测试数、覆盖率和 ZIP 哈希，并保留 P4--P8 的未完成边界。

## 远端更新判定

当前不允许将远端称为完整公开发布版本：P3 的发布级逐项处置账本尚未重新生成，P4--P8 也未完成。只有完成逐项复审及 P4--P7 实际交付后，才可以评估将通过同一 HEAD 验证的分支同步到已确认私有的协作远端；这只是阶段性备份，不是开源发布。任何 `git push`、可见性变更、tag 或 Release 都需要维护者对此次外部动作的明确授权。
