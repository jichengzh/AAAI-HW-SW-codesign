# P3-121 INT8 证据与执行边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P7 启动授权。

本批完成 12 项 `migrate_document` 候选的 P2 metadata/bytes、当前内容、实际职责/输入输出/支持链、公开等价物、许可证/来源及路径、凭据、网络、身份和受控输出风险的双独立只读审查。7 项真实职责分歧经第三方只读裁决后才接受；其余 5 项在执行阶段上收敛，独立审查中的协议外标签未被采用。

| 处置 | 数量 |
| --- | ---: |
| P4 外部制品契约 | 7 |
| P5 环境契约 | 0 |
| P6 执行契约 | 5 |
| P7 许可或公开阻塞 | 0 |

已测 INT8 route/trace、数值与 sanity blocker、FP16 AP measured-row 来源、calibration/reference-range 制品、reference-range 实测与 calibrated AP smoke/row-gate 以受控测量或制品证据为主导依赖，归 P4。energy/AP 补点、completion queue/simulator、hook capture、reference-range capture/AP 收口和 latency 补点会直接编排测量、采集、构建或评估流程，归 P6。第三方裁决确认：文档即使带有后续执行说明，只要主导职责是已测制品、来源或 blocker 证据，仍归 P4；执行队列与采集/评估闭环则归 P6。

未发现真实凭据或连接秘密，故无 P7；不存在经公开实现与测试/API 证明的同职责替代，故未声明迁入、改写或重复。未运行候选、模型、数据、设备、子进程或网络任务；受限 evidence、candidate identity 和 HMAC 不进入公开树。本批新增 12 个身份后真实唯一覆盖为 **1,220/1,551**，剩余 **331** 项文档候选。
