# P3-46 Stage7 环境、执行与最终证据边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4 启动授权。

## 范围与结论

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 314 个唯一已覆盖身份中的代码候选。逐项完成 P2 metadata/bytes、当前内容 SHA-256、职责/输入/输出/调用者、公开职责反证、许可证/来源和安全输出的只读检查；全部当前内容绑定通过。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 4 | trajectory、formal finalizer、scanner blocker 与 fail-closed evidence facade 的外部证据契约 |
| P5 | 5 | no-GPU、部署 bundle/cross-host override 与 H800 probe/lock 环境契约 |
| P6 | 3 | online round controller、worker 和 executor dispatch 执行契约 |

主审和独立复审的 12 条逐项结论完全一致。独立复审最初的汇总行写错了 P4/P5 数量；复审者随后逐项复核并确认真实汇总为 P4=4、P5=5、P6=3。该笔误已写入受限 evidence 与本交接文档，且没有改变任何候选的语义处置。

## 许可与安全边界

候选源码继承可验证的项目级 Apache-2.0，未见独立第三方许可证或来源标记；没有因源码许可证进入 `blocked_license_or_permission`。该许可不覆盖 ONNX、checkpoint、cache、部署清单、GPU/进程信息、运行输出或其他外部资产。P4--P6 后续契约必须明确其可得性、校验与公开许可。

候选可能处理 GPU UUID、进程/控制器身份、部署 manifest、路径、artifact hash、finalizer 输出与日志。任何公开实现必须使用路径无关标识、受控输出根、显式环境变量和失败关闭；不得迁入真实资产、硬件分配或原始运行拓扑。

## 验证边界

本批只读取源码、调用关系、公开树反证和项目级许可证；没有运行候选代码、私有模型、数据、设备、子进程或网络任务。受限逐路径 evidence、decisions 和 HMAC key 不进入公开树；路径无关 HMAC ledger 已由当前 inventory 与 decisions 重算。该批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **326/1,551**。
