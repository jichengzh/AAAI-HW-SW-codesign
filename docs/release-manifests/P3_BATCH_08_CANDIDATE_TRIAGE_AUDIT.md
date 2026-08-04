# P3-8：候选处置启动审计

状态：已完成本轮只读审计；未迁移源码、未生成公开 ledger、未推送。

## 审计目标

本批次从 P2b 冻结基线重新开始，而不是沿用人工记忆或旧批次选择。目标是确认 1,551 个 P3 候选的处置口径、许可/隐私阻塞规则和第一批最安全的账本演练范围。

本批次只读取重新生成的本地 inventory，且只提交聚合结论。原始 inventory、私有相对路径、文件正文和后续 decisions 文件仍必须保留在公开仓库外。

## 基线复核

使用 `private-source-inventory/1.1.0` 重新盘点私有 Git 根目录，得到的原始 JSON SHA-256 仍为 `6fafdace7dc0eae3e9982763cb1c5c098c0ae83d1715af67ff1e08d7a1d12a79`，与 P2b 冻结值一致。候选总数未漂移：

| 分类 | 数量 |
| --- | ---: |
| `migrate_code` | 1,008 |
| `migrate_config` | 96 |
| `migrate_document` | 439 |
| `fixture_candidate` | 8 |
| 合计 | 1,551 |

同次盘点仍有 3,664 个不透明条目：1,884 个外部输入、1,108 个生成物、455 个敏感复核项、111 个 shell 脚本、75 个其他类型和 31 个论文源文件。它们不进入 P3 直接迁移清单；外部输入转 P4，环境和执行约束转 P5--P6，许可/历史问题转 P7。

## 当前公开覆盖

将 P3 候选与当前公开仓库按同一相对位置做逐项哈希比对，得到：

| 状态 | 数量 | 处置含义 |
| --- | ---: | --- |
| 同路径且内容一致 | 9 | 可作为第一批账本演练对象，但仍需测试和依赖证据。 |
| 同路径但内容不同 | 38 | 不能自动视为已公开；需判断公开版本是否为脱敏重写或语义替代。 |
| 当前无公开同路径 | 1,504 | 需要逐批决定迁入、改写、外置、排除或阻塞。 |

按粗粒度角色看，候选主要集中在 framework、scripts、tools、实验区、论文区和其他私有实验区域。公开仓库现有覆盖非常小，因此 P3 关闭必须依赖完整 decisions/ledger，而不是依赖“已有公开树大致可用”的人工判断。

## Token 口径

本批次为后续本地 decisions 文件建立路径无关、内容无关的 `reason` 和 `evidence` token 注册表。当前 ledger 会机械拒绝不安全 token；维护流程还要求使用下列注册 token。若确有新增 token，必须先更新本文件或后续 P3 批次记录并完成审阅，随后才能用于 decisions：

| token 类型 | 建议 token |
| --- | --- |
| `reason` | `already_public_same_hash`、`public_rewrite_required`、`public_contract_supersedes`、`external_artifact_dependency`、`environment_specific_dependency`、`execution_pipeline_dependency`、`nonessential_to_supported_flow`、`third_party_license_marker`、`private_identifier_risk`、`nested_repo_boundary`、`paper_source_boundary` |
| `evidence` | `p2b_inventory_hash_match`、`path_hash_comparison`、`content_forbidden_scan_clean`、`public_test_coverage`、`public_contract_review`、`license_marker_review`、`dependency_boundary_review`、`no_public_path_yet`、`first_batch_manual_review` |

`reason` 解释为什么作出该处置，`evidence` 解释该处置由哪类审阅支持。二者都不得编码目录名、人员、机器、数据集私有名称、实验路径或文件内容摘要。

## 必须阻塞的情形

以下情形必须使用 `blocked_license_or_permission`，除非先完成脱敏重写、取得明确授权或转为可验证外部契约：

1. 文件包含或需要公开凭据、SSH 信息、口令、token、私有服务器地址、个人邮箱、个人身份标识或维护者本机路径。
2. 文件明显来自第三方、嵌套仓库、论文模板、供应商示例、外部项目改写或复制片段，但许可证、引用和再分发权限不明确。
3. 文件依赖不可公开数据、模型、checkpoint、engine、硬件日志或实验结果，且无法用公开契约替代。
4. 文件会在错误、日志、报告或 manifest 中回显调用者传入的私有标识、路径、URI、运行名或原始配置。
5. 文件属于复现关键路径，但只能在未公开硬件、未授权数据或未审查环境中运行。

若阻塞文件不是支持的公开复现链所必需，可在证明无依赖后改为 `excluded_nonessential`；若它是关键依赖，P3 不得关闭。

## 第一批安全候选

第一批不应从 1,504 个无公开同路径候选开始，而应先处理 9 个“同路径且内容一致”的代码候选。它们分布为 7 个 framework 代码候选和 2 个 scripts 代码候选。建议目标是验证本地 decisions 文件、HMAC key 权限、公开 ledger 生成、ledger 测试和台账更新流程，而不是扩大迁移范围。

这 9 个候选的可接受处置通常是：

- `duplicate_or_superseded`：公开仓库已有同内容文件，且测试/文档已经覆盖其公开职责；
- `migrated_public`：公开仓库已有同内容文件，但需要在 ledger 中作为已迁入源码明确登记。

不能因为同内容就自动通过。每个候选仍需补齐测试证据、依赖边界、许可标记复核和公开执行链映射。

## 下一内容批次的范围

作为第一批账本演练之外的下一内容批次，本轮仅按无路径的功能族统计 Stage4 相关候选代码。26 个候选中，19 个属于纯内存的反馈更新评估、选择闭包完成度或不确定性/排序/Pareto 评估功能族；它们尚无公开等价物，预期处置是 `rewritten_public`，而不是复制私有实现。另有 3 个已存在公开等价物，可在逐项核对测试、依赖边界和许可后考虑 `duplicate_or_superseded`；剩余 4 个不在本批范围，仍待单独审阅。

这个排序只确定 P3-9 的审阅边界，不构成自动迁入或自动账本决定。任何公开改写仍须先以脱敏合成输入写测试，随后进行身份/路径回显、第三方许可、文件/网络/GPU/子进程边界复审，并把可公开模块接入“设计说明 → 模块 → 命令/API → 合成测试 → 输入契约”链。

## 后续步骤

1. 在公开仓库外创建维护者私有 HMAC key，并保证文件权限不允许 group/other 读取。
2. 在公开仓库外创建第一批 9 项本地 decisions 文件，使用本文件固定的 token。
3. 用 `tools/release/build_p3_disposition_ledger.py` 生成临时公开 ledger，并确认不含路径、正文、人员或环境信息。
4. 若第一批通过，以本节定义的 19 个 Stage4 纯内存候选启动 P3-9 脱敏改写审阅；之后再按“公开同路径但内容不同”和“无公开同路径”的候选分层推进；每批结束都更新总台账和对应 P3 批次记录。

完成上述 9 项只能证明 P3 处置流程可用，不能关闭 P3。P3 关闭仍要求 1,551 个候选全部恰好一条处置决定，并通过同一 HEAD 的全量验证。
