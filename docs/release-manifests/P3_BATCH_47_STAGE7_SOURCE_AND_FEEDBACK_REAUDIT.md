# P3-47 Stage7 source 与 feedback 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 326 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 5 | private ablation、physical terminal-evidence promotion、frozen/search/relocation 外部制品契约 |
| P5 | 3 | formal prepare、source plan/lease 与 scheduler 环境契约 |
| P6 | 4 | GPU runtime、source materialization、source lease controller 与 round shell 执行契约 |

两个混合候选经第三次只读裁决。physical feedback 候选只认证、分类和晋升 Stage3/AP terminal evidence，未构建或运行硬件测量，故归 P4；source lease controller 在 no-GPU gate 后取得锁、调用 resolver、提交 canonical result 并推进 cache/reveal，故归 P6。

候选源码继承项目级 Apache-2.0；该许可不覆盖 checkpoint、ONNX、engine、AP/性能证据、GPU 租约或运行输出。后续 P4/P5/P6 必须以路径无关引用、显式环境变量、受控输出根和失败关闭处理这些边界。本批没有运行候选代码、模型、数据、设备、子进程或网络任务。受限 records 和 HMAC key 不进入公开树；HMAC ledger 已重算。该批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **338/1,551**。
