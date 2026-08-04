# P3-66 Stage6/Stage7 遗留测试复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批 12 个此前未覆盖候选均完成 P2 metadata/bytes、当前 SHA-256、职责/调用、公开 API/test 对比、许可证/来源和安全输出的只读审查。

| 处置 | 数量 |
| --- | ---: |
| 公开替代 | 2 |
| 非必要排除 | 10 |

第 3 项经第三次裁决：私有 router source-text guard 的实际职责是保持 INT8 自动路由且禁止 legacy DP4A/native/hand rewrite；公开 Stage6 manifest validator 已以可校验、失败关闭 contract 承担同一职责，故为公开替代。其余遗留 formal-runner、validation、TRT、Stage7 scheduler/CLI、AP queue 和可选图测试不在当前公开支持链，也不执行被测工具。

本批没有运行候选、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **566/1,551**。
