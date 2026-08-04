# P3-11：小型固定 fixture 逐项处置

状态：8 项 fixture 候选均已逐项审阅，并已形成仅本机保存的受限 decisions/ledger 检查点；未复制私有 fixture、未提交私有清单、未推送、未发布。

## 审阅结果

P2b 冻结清单中的 8 项 `fixture_candidate` 共 7,561 bytes；每项均为小型 YAML/JSON 测试输入，而非模型、数据集、checkpoint、engine、ONNX、日志或结果。审阅逐项检查了其声明的来源、是否为 synthetic fixture、是否携带可测量结论，以及当前公开测试和合成生成契约是否已覆盖同一公开职责。

| 处置 | 数量 | 固定 reason / evidence | 结论 |
| --- | ---: | --- | --- |
| `duplicate_or_superseded` | 7 | `public_contract_supersedes` / `public_test_coverage` | Stage1 的合成模型/扫描/标准卷积锚点和 Stage2 的合成分类、AP、延迟输入均已由当前公开测试与 demo 生成契约覆盖；私有 fixture 不再是支持链的唯一输入。 |
| `external_contract_p4` | 1 | `external_artifact_dependency` / `dependency_boundary_review` | 该 manifest 只描述正式运行前必须获得并校验的外部 checkpoint 与数据集；它不含制品本身，转由 P4 的来源、许可、下载与 SHA-256 契约处理。 |

对应公开回归为 36 passed，覆盖 Stage1 的扫描、预测与标准卷积工作流，以及 Stage2 的 demo/contract 工作流。此验证证明公开契约不依赖这些私有 fixture；它不把 demo 或 fixture 值升级为论文或硬件证据。

## 本地账本检查点

维护者在仓库外的权限受限目录中创建了 8 项派生 inventory 和 decisions，并用同一受限 HMAC key 生成路径无关 ledger。该 ledger 的计数恰为 7 条 `duplicate_or_superseded` 与 1 条 `external_contract_p4`；记录仅含 HMAC-SHA256 candidate ID、P2 分类、受控 token 和后续阶段。泄露检查确认不含路径、origin、正文、人员、机器、地址或 key。

P3 当前有 26/1,551 项可复核 decisions：P3-9 的 9 项同内容替代、P3-10 的 9 项公开契约替代，以及本批的 8 项 fixture 处置。三个检查点均是局部、仅本机的审计材料，不能替代完整 1,551 项 ledger 或 P3 关闭条件。

## 后续顺序

1. 将本批外部制品 descriptor 与所有后续 `external_contract_p4` 候选纳入 P4 的许可、获取、版本和 SHA-256 清单；
2. 对 P3-10 中待改写、待许可证和高风险的 30 项继续逐项完成安全结论；
3. 按同一标准处理 1,504 项当前无公开同路径候选。任何尚未具有测试、许可和依赖证据的候选仍不得写入最终 ledger。
