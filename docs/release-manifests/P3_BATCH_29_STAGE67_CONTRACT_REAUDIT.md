# P3-29：Stage6/Stage7 契约边界逐项复审

状态：已完成（本地）。本批逐项读取 12 个候选的当前内容，核对 P2 metadata/字节数、当前 SHA-256、职责、调用关系、公开替代物、来源/许可证和安全输出。没有运行私有脚本、模型、数据、设备、子进程或生成物。

## 逐项结论

| 职责类别 | 数量 | 暂定处置 | 复审结论 |
| --- | ---: | --- | --- |
| Stage6 表格投影与 Stage7 统计/消融视图 | 3 | `rewritten_public` | 当前公开树已有已测试的安全重写，保留公共输入校验和匿名化统计职责，不再暴露私有 evidence 或执行 surface。 |
| Stage6 独立验证、Stage7 selection/cache/evidence 适配 | 5 | `external_contract_p4` | 主导输入为外部证据、制品、部署 bundle 或其摘要；P4 需先建立路径无关的 schema、固定输入和校验边界。 |
| Stage7 部署 bundle 与 no-GPU 隔离 | 2 | `environment_contract_p5` | 需要环境、部署清单或隔离语义；P5 只提供脱敏的环境规范和失败关闭验证。 |
| Stage7 actual runner 与物理 pipeline | 2 | `execution_contract_p6` | 包含实际执行/调度边界，必须在 P6 以受控执行契约处理，不能作为默认公开运行器。 |

公开替代物逐项比较职责和公开测试后才接受，未以同路径或关键词作自动判断。代码许可证依据不覆盖外部制品、模型、设备或运行环境。

## 证据与阶段边界

12 项均完成当前内容 SHA-256 与 P2 metadata/字节数对账、AST 解析、导入/调用审阅、公开替代物检查、来源/许可证结论和安全输出审查。P2b inventory 不带每文件历史 digest，当前 SHA-256 仅绑定本轮读取内容；以后变更必须重新审计。

受限 HMAC evidence/decision records 与路径无关账本检查点均通过，但不构成 P3 关闭账本。本批不迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果；本批关闭时的累计为 109/1,551。随后 P3-26/P3-27 的规范 records 通过独立复审和 supersession 处置，P3-31、P3-33、P3-34 也已按协议接受，当前累计为 181/1,551；P4 尚未开始。
