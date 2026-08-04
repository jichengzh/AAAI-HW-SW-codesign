# P3-16：设计、交接与研究文档逐项处置

状态：439 项文档候选已逐项审阅并形成仅本机保存的受限 decisions/ledger 检查点；未复制私有 Markdown、未提交私有清单、未推送、未发布。

## 审阅方法

每项文档均检查公开同路径关系、交接/代理/研究笔记属性、外部制品或环境说明语义，以及私有标识、位置或凭据类风险信号。绝大多数文件是历史代理过程、研究草稿、论文编排、调试记录或学习笔记；它们不是公开运行时、CPU smoke 或已声明正式实验执行链的输入。

少量承载可复现职责的高层设计文档不会直接复制，而是分流到相应契约：制品说明进入 P4、环境/依赖与构建说明进入 P5、复现流程、量化方法和公开设计说明进入 P6。含私有标识或安全运行细节的文档一律进入 P7，必须脱敏改写或以公开最小操作说明替代。

| 处置 | 数量 | 固定 reason / evidence | 结论 |
| --- | ---: | --- | --- |
| `external_contract_p4` | 1 | `external_artifact_dependency` / `dependency_boundary_review` | 制品说明必须由 P4 重写为来源、许可、版本、SHA-256 与不可用失败关闭契约。 |
| `environment_contract_p5` | 3 | `environment_specific_dependency` / `dependency_boundary_review` | 环境、依赖或构建信息必须由 P5 汇总为 CPU、4090、H800 的公开最小环境定义。 |
| `execution_contract_p6` | 21 | `execution_pipeline_dependency` / `dependency_boundary_review` | 可复现流程、量化/搜索方法与高层设计说明必须由 P6 重新组织为公开命令、manifest 和设计文档。 |
| `excluded_nonessential` | 333 | `nonessential_to_supported_flow` / `public_contract_review` | 内部交接、代理轨迹、研究草稿、论文编排和学习笔记不作为公开支持链的一部分。 |
| `blocked_license_or_permission` | 81 | `private_identifier_risk` / `public_contract_review` | 文档含私有标识、位置或安全运行风险；P7 必须逐项脱敏、替代或保留为不可公开材料。 |

本批的“排除”只排除旧原文，不能免除对公开设计说明的责任。P6 仍须把支持完整复现的架构、数据/环境边界、命令和失败条件以独立、可审阅的 Markdown 交付。

## 本地账本检查点

受限台账的 439 条记录严格对应 P2 分类 `migrate_document`，按上表计数精确对账。生成的路径无关 ledger 仅保存 HMAC-SHA256 candidate ID、分类、受控 token 和后续阶段；泄露扫描确认没有路径、origin、正文、人员、机器、地址或 key。

P2b 的 1,551 项候选至此均已有恰好一条本机 decisions 记录：前序 P3-9--P3-15 的 1,112 项与本批 439 项相加为 1,551。下一步不是宣布项目已可开源，而是合并唯一全量 ledger，并重跑 P3 关闭验证；P4--P8 的制品、环境、执行、许可和发布门禁仍未完成。

## 后续顺序

1. 使用冻结 P2b inventory、受限 HMAC key 和恰好八个接受的本机 decisions 检查点生成唯一全量路径无关 ledger；
2. 对该 ledger 进行精确计数、重复键、受控 token 和泄露扫描，再更新 P3 关闭状态；
3. 只有 P4--P8 实际交付相应外部/环境/执行/许可/发布证据后，才可评估完整可复现或外部同步。
