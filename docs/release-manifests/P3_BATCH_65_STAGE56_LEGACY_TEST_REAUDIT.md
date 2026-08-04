# P3-65 Stage5/Stage6 遗留测试复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批 12 个此前未覆盖候选均完成 P2 metadata/bytes、当前 SHA-256、职责/调用、公开 API/test 对比、许可证/来源和安全输出的只读审查。

| 处置 | 数量 |
| --- | ---: |
| 公开替代 | 2 |
| 非必要排除 | 10 |

两份独立审阅逐项一致。公开 Stage5 genome contract 和 Stage6 six-arm manifest contract 及其测试覆盖两项相同职责的旧测试；其余候选是当前公开链未调用的 AP plan、repair、独立验证、completion audit、TVM helper 或 shell controller 测试。它们不执行被测运行工具，不能仅因工具存在硬件、AP、TVM 或 controller 边界而转入 P4--P6。

本批没有运行候选、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **554/1,551**。
