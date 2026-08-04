# P3-30：Stage7 执行与测试边界逐项复审

状态：已完成（本地）。本批逐项读取 12 个候选的当前内容，核对 P2 metadata/字节数、当前 SHA-256、职责、调用关系、公开替代物、来源/许可证和安全输出。没有运行私有脚本、模型、数据、设备、子进程或生成物。

## 逐项结论

| 职责类别 | 数量 | 暂定处置 | 复审结论 |
| --- | ---: | --- | --- |
| Stage7 输出、物理规划/反馈、回合状态机与 ONNX/校准测试 | 8 | `external_contract_p4` | 这些项消费或产生外部 evidence、制品、校准或结果结构。P4 必须先用脱敏 schema、固定输入和路径无关输出定义可公开边界。 |
| Stage7 source lease | 1 | `environment_contract_p5` | 绑定设备清单、租约和环境准入，不能公开私有环境假设；P5 需改为环境规范与探测。 |
| Stage7 runtime 与 source tuning 测试 | 2 | `execution_contract_p6` | 这些项的职责是受控硬件/子进程执行或其验证，必须转为 P6 失败关闭的执行契约。 |
| 校准预测器测试 | 1 | `duplicate_or_superseded` | 当前公开预测器工作流和测试已覆盖其核心 no-overpromotion、模型 verdict 和报告语义；不迁入私有测试 surface。 |

代码许可证依据不覆盖 ONNX、checkpoint、测量日志、部署 bundle 或生成制品。未发现直接凭据，但多个非阻塞项可能处理制品/运行时 provenance，故仍不得直接迁入公开树。

## 证据与阶段边界

12 项均完成当前内容 SHA-256 与 P2 metadata/字节数对账、AST 解析、导入/调用审阅、公开替代物检查、来源/许可证结论和安全输出审查。P2b inventory 不带每文件历史 digest，当前 SHA-256 仅绑定本轮读取内容；以后变更必须重新审计。

受限 HMAC evidence/decision records 与路径无关账本检查点均通过，但不构成 P3 关闭账本。本批不迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果；本批关闭时的累计为 121/1,551。随后 P3-26/P3-27 的规范 records 通过独立复审和 supersession 处置，P3-31、P3-33、P3-34 也已按协议接受，当前累计为 181/1,551；P4 尚未开始。
