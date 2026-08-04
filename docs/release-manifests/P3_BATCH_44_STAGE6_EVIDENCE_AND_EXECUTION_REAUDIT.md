# P3-44 Stage6 证据、环境与执行边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4 启动授权。

## 范围与结论

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 290 个唯一已覆盖身份中的代码候选。逐项完成 P2 metadata/bytes、当前内容 SHA-256、职责/输入/输出/调用者、公开职责反证、许可证/来源和安全输出的只读检查；全部当前内容绑定通过。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 5 | Stage6/F-Cooper 证据、TVM 默认基线与 native checkpoint 绑定契约 |
| P5 | 1 | H800 硬件范围和 runner readiness 环境契约 |
| P6 | 4 | F-Cooper 正式闭环、paper closure、五臂协议和 schedule-only 执行契约 |
| `rewritten_public` | 2 | 已有发布安全的 Stage6 formal-plan 与 paper-table 公开改写 |

主审与独立复审对多数结论一致。第三次只读裁决确认：F-Cooper formal finalizer 的 GPU guard 与证据输入服务于已执行管线的闭环和原子终结，主导归 P6；两项公开 Stage6 实现则具有明确的私有责任迁移记录、同职责 API/测试和额外的输入/路径安全保护，精确处置为 `rewritten_public`，而不是偶然的同名重复。

## 许可与安全边界

候选源码继承可验证的项目级 Apache-2.0，未见独立第三方许可证或来源标记；没有因源码许可证进入 `blocked_license_or_permission`。该许可不覆盖数据、权重、checkpoint、ONNX、engine、TVM 数据库、GPU 环境、测量制品或日志。P4--P6 后续契约必须分别证明这些外部输入的可得性、校验方式和公开许可。

候选处理证据、GPU guards、runtime command、设备/结果根和日志时存在位置与环境泄露风险。任何后续公开实现必须使用路径无关标识、显式环境变量、受控输出根和失败关闭；不得迁入本机默认路径、真实资产或硬件分配。

## 验证边界

本批只读取源码、调用关系、公开实现/测试和设计记录；没有运行候选代码、私有模型、数据、设备、子进程或网络任务。受限逐路径 evidence、decisions 和 HMAC key 不进入公开树；路径无关 HMAC ledger 已由当前 inventory 与 decisions 重算。该批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **302/1,551**。
