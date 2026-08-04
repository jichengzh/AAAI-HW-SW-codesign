# P3-27：旧搜索基础模块与真实 ablation 边界逐项复审

状态：已完成（本地）。本批已逐项读取 12 个候选的当前内容，核对其 P2 metadata/字节数、当前 SHA-256、职责、导入/调用、公开替代物、来源/许可证和安全输出。没有执行私有脚本、数据、模型、设备、子进程或生成物。独立完整语义复审已接受本批结论；冲突的旧本地 records 已保留但被唯一的 supersession manifest 明确排除，不再作为账本或进度输入。

## 逐项结论

| 职责类别 | 数量 | 暂定处置 | 复审结论 |
| --- | ---: | --- | --- |
| 公开树已覆盖的 capability、Stage1 和旧搜索基础职责 | 10 | `duplicate_or_superseded` | 3 项与公开树字节完全一致，1 项由公开同路径安全版本替代，其余旧配置、约束、传播、特征编码、延迟估计和随机搜索职责已由公开 Stage2 canonical search、Stage5 selection、Stage6 formal plan 与 Stage7 policy 契约重写。 |
| 上游模型变体 baseline/映射 adapter | 1 | `external_contract_p4` | 该 adapter 读取外部 baseline 表和延迟映射制品。P4 必须先定义这些输入的许可、版本、SHA-256、用途、目录投影和缺失时失败信息。 |
| 真实 PQS ablation runner | 1 | `execution_contract_p6` | 该 runner 消费真实测量 lookup，生成结果和图，并依赖外部证据。公开树只能在 P6 提供受控执行配方或报告 schema，不能把该真实执行器放进默认 CI。 |

逐项安全审查未发现直接凭据、邮箱或 IP。部分旧模块包含本机目录、结果目录、硬件或制品 provenance 信号；这些信号已通过公开替代或 P4/P6 阶段边界处理。

## 证据与阶段边界

12 项均完成当前内容 SHA-256 与 P2 metadata/字节数对账，当前内容漂移为 0。P2b inventory 没有每文件历史 digest，当前 SHA-256 仅绑定本轮读取的内容；源码变化时必须重新审计。

本批不迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。受限 HMAC evidence/decision records 仅用于本机核验，不构成 P3 关闭账本。经规范 records 的当前内容复验和账本重建后，本批成为已接受的 12 项；当前发布级 re-audit 为 145/1,551，P4 尚未开始。
