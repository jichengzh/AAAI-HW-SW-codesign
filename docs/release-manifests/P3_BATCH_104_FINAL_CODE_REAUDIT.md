# P3-104 最后代码候选边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

这是冻结基线剩余的全部 8 项 `migrate_code` 候选。每项完成 P2 metadata/bytes、当前 SHA-256、职责/调用、公开等价物、许可证/来源和安全输出的双独立只读审查；1 项经第三方只读裁决。

| 处置 | 数量 |
| --- | ---: |
| P4 外部制品契约 | 1 |
| P6 执行契约 | 7 |

P4 项仅转换既有结果、配对和标定制品。P6 项执行量化评测/敏感度、TRT head forward、剪枝评测、训练、AdaRound/PTQ 验证或 calibration dump。第三方裁决确认 TRT head smoke 实际构造 head 并执行 CUDA dummy forward，因此归 P6，不能仅因依赖环境而降为 P5。`migrate_code` 分类现已完全审阅；后续批次将从 `migrate_config` 开始。没有运行候选、模型、数据、设备、子进程或网络任务；受限 evidence 和 HMAC 不进入公开树。本批新增 8 个身份后真实唯一覆盖为 **1,018/1,551**。
