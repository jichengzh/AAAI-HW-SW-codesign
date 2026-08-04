# P3-63 Stage1/Stage2 遗留测试复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批的 12 个此前未覆盖代码候选均完成 P2 metadata/bytes、当前 SHA-256、职责/调用、公开 API/test 对比、许可证/来源和安全输出的只读审查。

| 处置 | 数量 |
| --- | ---: |
| 公开替代 | 7 |
| 非必要排除 | 5 |

第 3、4、6、10 项经第三次裁决。公开 Stage1 predictor/classifier/trace-plan 和 Stage2 canonical-search/genome 测试覆盖其净化后的公开合同；但公开 canonical search 没有承担完整私有 QxS/SMBO entry 的行为，后者因不在当前公开支持链中排除。TVM/Route-B 遗留测试没有执行被测运行工具，也不被公开流程调用，不能仅由底层执行风险转入 P4--P6。

本批没有运行候选、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **530/1,551**。
