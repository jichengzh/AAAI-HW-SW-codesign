# P3-49 Stage7 deployment 与 TVM 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 350 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 2 | 论文图/源 CSV provenance 与 checkpoint alias 的外部制品契约 |
| P5 | 1 | 不使用 GPU 的环境与部署 receipt archive |
| P6 | 8 | TVM benchmark/tune、部署 sidecar、恢复与 cooldown 等执行契约 |
| `excluded_nonessential` | 1 | 不参与公开支持链的可选示意图 |

第三次只读裁决确认：论文正文引用的 cost-model 图表不能仅凭公开选择实现视为已处置，仍需在 P4 建立源 CSV 与图表 provenance；no-GPU receipt archive 不运行测量或硬件，但保存环境/部署回执，故归 P5；精确 GPU cooldown wrapper 会推进实时部署状态，故归 P6。可选示意图没有公开 support-chain caller，公开实现、CLI、测试和已验证制品已承担必要职责，故归 `excluded_nonessential`；若日后将其纳入论文支持链，必须重新进行来源审计。

候选源码继承项目级 Apache-2.0；该许可不覆盖数据、模型、checkpoint、ONNX、engine、校准缓存、GPU/TVM/TRT 环境、测量结果或图源 CSV。后续 P4--P6 必须使用路径无关引用、显式环境变量、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。该批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **362/1,551**。
