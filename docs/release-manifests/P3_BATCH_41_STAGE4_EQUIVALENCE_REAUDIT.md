# P3-41 Stage4 公开等价职责复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4 启动授权。

## 范围与结论

本批对冻结 P2b 基线中的 12 项候选逐项完成主审、独立复审、当前内容绑定、受限证据/决定记录和路径无关 HMAC ledger 重算。最终处置为 3 项 P4、3 项 P6、6 项 `duplicate_or_superseded`。

六项 `duplicate_or_superseded` 并非仅依据相似文件名处置：每项均已核实公开 Stage4 模块或其测试/API 对应的职责范围，确认其承担同一可复现流程责任；公开定向回归覆盖 closure audit、cost-model selection、feedback update evaluation、ranking/Pareto、selection completion 和 uncertainty replay，共 **119 passed**。

| 处置 | 数量 | 后续责任 |
| --- | ---: | --- |
| P4 | 3 | 外部证据与制品契约 |
| P6 | 3 | 可复现实验执行与测量契约 |
| `duplicate_or_superseded` | 6 | 由公开 Stage4 实现和定向回归承担等价职责 |

## 验证边界

复审只读取冻结基线和公开替代实现；未执行私有模型、数据、设备任务或外部网络操作。受限原始记录不进入公开树；公开账本仅保留路径无关、可复算的 HMAC 承诺。该批完成后，P3 已接受总数为 **289/1,551**。
