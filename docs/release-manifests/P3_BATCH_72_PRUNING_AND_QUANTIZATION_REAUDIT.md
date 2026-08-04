# P3-72 pruning 与量化边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批 12 项完成 P2 metadata/bytes、当前 SHA-256、职责/调用、公开等价物反证、许可证/来源和安全输出的只读审查。前六项双审一致；后六项经第三次只读许可证裁决。

| 处置 | 数量 |
| --- | ---: |
| P6 执行契约 | 6 |
| P7 许可/隐私阻塞 | 6 |

P6 项覆盖 transformer、结构化 pruning 与 temporal cache 执行支持。量化包及其 AdaRound、communication quant、BN folding、BEVFormer/downstream wrappers 具有明确 QuantV2X/OpenCOOD copy/port 来源；可见上游许可不允许进一步转让，且无可公开的双许可、NOTICE 或干净替代实现，故保持 P7。

没有运行候选、模型、数据、设备、子进程或网络任务。受限 evidence 和 HMAC 不进入公开树；本批新增 12 个身份后真实唯一覆盖为 **638/1,551**。
