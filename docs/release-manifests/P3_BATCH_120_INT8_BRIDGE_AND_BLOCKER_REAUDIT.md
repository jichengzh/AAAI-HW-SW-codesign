# P3-120 INT8 Bridge 与 Blocker 复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P7 启动授权。

本批完成 12 项 `migrate_document` 候选的 P2 metadata/bytes、当前内容、实际职责/支持链、公开等价物、许可证/来源及路径、凭据、网络、身份和受控输出风险的双独立只读审查。10 项职责分歧经第三方只读裁决后才接受。

| 处置 | 数量 |
| --- | ---: |
| P4 外部制品契约 | 7 |
| P5 环境契约 | 0 |
| P6 执行契约 | 5 |
| P7 许可或公开阻塞 | 0 |

FP16 true-eval、INT8 adapter/route、AP gate、数值对齐、权重和 output-scale blocker 文档以已测状态、制品、根因和不可声明边界为主导依赖，归 P4；量化补点、AP 收口、TVM worker bridge、Original60 重测和真实 activation/AP gate 计划会直接编排 build/run/bridge/评估流程，归 P6。第三方裁决确认：记录 bridge 或后续修复不自动覆盖以证据为主的 blocker 文档；而会执行跨进程 worker、重测或 full AP gate 的方案属于 P6。未发现真实凭据或连接秘密，因此没有 P7。未运行候选、模型、数据、设备、子进程或网络任务；受限 evidence 和 HMAC 不进入公开树。本批新增 12 个身份后真实唯一覆盖为 **1,208/1,551**，剩余 **343** 项文档候选。
