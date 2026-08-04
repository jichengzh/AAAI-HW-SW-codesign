# P3-38：Stage35 完成与证据边界逐项裁决

状态：已接受（本地）。本批 12 个候选经主审、独立语义复审和根级逐项裁决。每次读取均核对 P2 metadata/字节数与队列绑定的当前 SHA-256，内容漂移为 0；没有运行私有脚本、模型、数据、设备、子进程或网络。

## 分歧处理

前两次审阅的总计数相同，但两个 finalizer 的 P4/P6 主导边界互换。裁决直接读取这两个测试和被测入口：它们接收 manifest 与 state rows，检查固定组/arms/schema，输出 final rows 与 audit；测试中的 CLI 只验证帮助文本，不实际执行模型、设备、job 或测量。因此两个 finalizer 的主导职责是外部制品/evidence schema，而非执行器，统一归入 P4。

## 逐项结论

| 职责类别 | 数量 | 处置 | 裁决结论 |
| --- | ---: | --- | --- |
| ONNX graph feature、Gold finalizer、AP repair、merge、sufficiency | 6 | `external_contract_p4` | 主导依赖为 ONNX、manifest、checkpoint/engine state、AP evidence 或输出 schema。P4 必须建立公开 schema、固定合成输入、许可/下载说明和哈希校验。 |
| supervisor poll、CoDriving source queue、targeted finalizer、repeat/AP plans | 6 | `execution_contract_p6` | 主导职责是 supervisor、source/job queue、runner、repeat 或 AP 执行编排，不能直接迁入默认公开 CI。 |

根许可证信号为 Apache-2.0；它不覆盖模型、checkpoint、ONNX、engine、数据、测量制品或硬件环境。所有候选均有受限的职责、调用、来源/许可证和安全输出记录；路径无关 HMAC ledger 可重新计算，但不提交，也不构成 P3 全量关闭账本。

本批没有迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。它使已接受重审总数达到 229/1,551；P3-37 仍待独立复审，P4 尚未开始。
