# P3-15：实验代码边界逐项处置

状态：610 项剩余实验代码候选已逐项审阅并形成仅本机保存的受限 decisions/ledger 检查点；未复制私有脚本、未提交私有清单、未推送、未发布。

## 审阅方法

每项候选均按 Python 语法、公开同路径差异、公开执行面依赖、外部制品词汇、硬件/工具链语义、私有标识信号和历史测试/调试边界进行检查。全部候选可解析；当前公开同路径但不同内容的候选不被视为自动等价物，而按自身风险和所需公开契约独立分流。

本批采用以下单项优先级：不属于公开支持链的历史测试、调试或论文编排辅助项先排除；带私有标识信号的代码进入 P7 阻塞；其余先按外部数据、checkpoint、模型导出或结果输入转入 P4，再按 CUDA、TensorRT、TVM 或设备测量环境转入 P5，最后才将纯执行编排、验证或汇总代码转入 P6。任何转交都要求后续阶段以脱敏重新实现、manifest 或失败关闭契约替代旧源码，不能直接恢复文件。

| 处置 | 数量 | 固定 reason / evidence | 结论 |
| --- | ---: | --- | --- |
| `external_contract_p4` | 210 | `external_artifact_dependency` / `dependency_boundary_review` | 代码依赖外部数据、checkpoint、模型导出、engine、测量输入或结果制品；P4 必须提供来源、许可、版本、SHA-256 和不可用时的失败关闭。 |
| `environment_contract_p5` | 187 | `environment_specific_dependency` / `dependency_boundary_review` | 代码依赖 GPU、CUDA、TensorRT、TVM、设备或精度工具链；P5 必须分别固定 CPU、4090、H800 的环境边界。 |
| `execution_contract_p6` | 69 | `execution_pipeline_dependency` / `dependency_boundary_review` | 代码承担实验编排、验证、汇总或受控执行职责；P6 必须以版本化命令、seed、输入输出 hash 和 manifest 建立公开接口。 |
| `excluded_nonessential` | 68 | `nonessential_to_supported_flow` / `public_contract_review` | 历史测试、调试、论文编排或本机辅助项未被公开支持运行时导入，不作为公开复现链的一部分。 |
| `blocked_license_or_permission` | 76 | `private_identifier_risk` / `public_contract_review` | 代码含私有标识或位置风险，不能直接公开；P7 必须脱敏重写或以受控契约替代，并确认不会回显调用者输入。 |

本批并没有把任何实验数值、设备测量或论文结论升级为公开可复现证据；P4--P6 和 P7 的具体完成仍是发布前置条件。

## 本地账本检查点

受限台账的 610 条记录严格对应 P2 分类 `migrate_code`，按上表计数精确对账。生成的路径无关 ledger 仅保存 HMAC-SHA256 candidate ID、分类、受控 token 和后续阶段；泄露扫描确认没有路径、origin、正文、人员、机器、地址或 key。

P3 当前有 1,112/1,551 项可复核 decisions：前序 P3-9--P3-14 的 502 项与本批 610 项。剩余 439 项均为设计/交接/研究 Markdown；本批完成后不会将 P3 标记关闭，直到它们也被逐项处置、全量 ledger 建成并重新通过发布门禁。

## 后续顺序

1. P3 对 439 项 Markdown 逐项判断其是否为可公开设计文档、内部交接、论文/研究笔记、第三方资料或敏感运行记录；
2. 合并所有本机 decisions 时只采用本批的更正检查点，早期未引用的生成草稿不进入全量 ledger；
3. 完成 1,551 项后，重新建立单一全量路径无关 ledger，再由 P4--P8 处理外部制品、环境、执行、许可和发布验证。
