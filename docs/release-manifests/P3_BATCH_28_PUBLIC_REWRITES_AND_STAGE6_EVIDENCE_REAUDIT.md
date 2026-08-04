# P3-28：公开重写与 Stage6 evidence 边界逐项复审

状态：已完成（本地）。本批逐项读取 12 个候选的当前内容，核对其 P2 metadata/字节数、当前 SHA-256、职责、导入/调用、公开替代物、来源/许可证和安全输出；独立样本复审已通过。没有执行私有脚本、数据、模型、设备、子进程或生成物。

## 逐项结论

| 职责类别 | 数量 | 暂定处置 | 复审结论 |
| --- | ---: | --- | --- |
| 已由公开树重写的 Stage1、Stage2、Stage4 和 Stage6 契约 | 7 | `rewritten_public` | 公开树已经提供 release-safe 的同路径或同职责实现，并由 Stage1、Stage2、Stage4 或 Stage6 测试覆盖；私有版本中的路径、证据位置或模型表面不再进入发布包。 |
| 外部 probe/readiness/evidence 制品契约 | 2 | `external_contract_p4` | 这些候选主导输入是 ONNX probe/readiness/evidence bundle 或外部测量记录。P4 必须先定义制品许可、版本、SHA-256、用途、目录投影和缺失时失败信息。 |
| 外部执行/测量计划契约 | 2 | `execution_contract_p6` | 这些候选需要外部模型制品、硬件执行或测量计划闭环；公开树只能提供失败关闭的执行契约，不能默认运行。 |
| Orin 度量与平台协议 | 1 | `environment_contract_p5` | 该候选属于平台指标、功耗解析和硬件协议边界；P5 应定义环境输入 schema、采集工具约束和路径安全输出。 |

逐项安全审查未发现直接凭据、邮箱或 IP。候选中的运行时、制品或平台 provenance 信号均通过公开重写、P4、P5 或 P6 阶段边界处置。

## 证据与阶段边界

12 项均完成当前内容 SHA-256 与 P2 metadata/字节数对账，当前内容漂移为 0。P2b inventory 没有每文件历史 digest，当前 SHA-256 仅绑定本轮读取的内容；源码变化时必须重新审计。

本批不迁入新的源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。受限 HMAC evidence/decision records 仅用于本机核验，不构成 P3 关闭账本。新的发布级 re-audit 从 85/1,551 增至 97/1,551；P4 尚未开始。
