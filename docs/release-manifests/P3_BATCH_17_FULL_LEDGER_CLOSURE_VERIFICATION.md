# P3-17：全量处置账本与关闭验证

状态：P2b 的 1,551 项候选均已有恰好一条受限本机 decision，唯一的公开路径无关 ledger 已生成并提交；P3 的“候选审计与账本”工作完成，但“完整可复现发布关闭”仍受 P4--P7 未交付项阻塞。

## 全量账本

全量 ledger 见 [P3_DISPOSITION_LEDGER.json](P3_DISPOSITION_LEDGER.json)。它由冻结的 P2b inventory、受限 HMAC key 和八个接受的本机检查点生成；原始 inventory、逐路径 decisions 和 key 从未提交。生成器因当前共享工作树的组写目录拒绝直接输出，故在权限安全的全新本地 clone 中生成、验证并提交同一字节内容，再以本地快进合并回本分支；没有推送、创建 PR 或修改远端。

ledger SHA-256：`c2ee5a5af17d37966fbcc52e90c0f3f90557f09a2d2724ccefbdc487ee613afb`。

| 分类 | 数量 |
| --- | ---: |
| `migrate_code` | 1,008 |
| `migrate_config` | 96 |
| `migrate_document` | 439 |
| `fixture_candidate` | 8 |
| **总计** | **1,551** |

| 最终处置 | 数量 | 含义 |
| --- | ---: | --- |
| `duplicate_or_superseded` | 48 | 已有公开安全契约或公开测试替代。 |
| `excluded_nonessential` | 668 | 不属于声明的公开支持链；旧原文不迁入。 |
| `external_contract_p4` | 218 | 等待 P4 的来源、许可、版本、SHA-256 和失败关闭。 |
| `environment_contract_p5` | 193 | 等待 P5 的 CPU、4090、H800 环境契约。 |
| `execution_contract_p6` | 193 | 等待 P6 的公开命令、seed、输入输出 hash、manifest 和设计说明。 |
| `blocked_license_or_permission` | 231 | 等待 P7 的许可证、权限或私有标识脱敏结论。 |
| **总计** | **1,551** | 每个候选仅一条决定。 |

机械复核确认：ledger 有 1,551 条 entry；分类和处置计数均精确对账；所有 candidate ID 为 HMAC-SHA256 值；内容不含路径、origin、正文、人员、机器、地址或 HMAC key。

## 同一 HEAD 的关闭验证

| 验证 | 结果 |
| --- | --- |
| 全量 pytest + 覆盖率门禁 | 502 passed，83.09%，满足 80% 门槛。 |
| P3 ledger/身份/交接测试 | 35 passed。 |
| 身份扫描与匿名 archive 集成测试 | 50 passed。 |
| Ruff / `compileall` / diff 空白检查 | 通过。 |
| 匿名 ZIP build/verify | 124 个成员，SHA-256 `a6951ed6310541a0266a14d6af93228fe20313f93f41508231b773e54644d264`。 |

以上验证仅证明当前公开代码、账本和匿名制品在其声明范围内自洽；不访问 GPU、外部数据、模型、checkpoint、engine 或真实硬件，也不把 CPU smoke 升级为论文实验结果。

## 关闭结论与下一步

P3 的逐项审计、受限 decisions 与公开 ledger 已完成，因此不再存在未处置的 P2b 候选。P3 的完整发布关闭条件仍**未满足**：218 项 P4、193 项 P5、193 项 P6 和 231 项 P7 决定尚未落实为公开、可验证的契约或许可结论。故当前分支不能称为“完整可复现开源版本”，也不能据此推送、公开或打 tag。

下一阶段依次是：P4 外部制品与数据契约、P5 CPU/4090/H800 环境契约、P6 执行 manifest 与脱敏设计说明、P7 许可证/隐私解除，再进入 P8 的新 clone/CI/外部发布授权。
