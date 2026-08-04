# P3-22：运行时编排与上游配置逐项复审

状态：已完成（本地）。本批逐项复审 12 个未审计的 Python 候选，覆盖设备测量、Stage5/Stage7 编排、TensorRT profiler、checkpoint 恢复工具，以及两个上游端到端配置。队列由外部输入线索建立，最终结论基于逐项职责、调用、来源和安全审查。独立抽样复审覆盖 P4、P6 和许可阻塞；12/12 受限 records 也完成一致性检查，未发现需撤回的问题。

## 逐项结论

| 职责类别 | 数量 | 暂定处置 | 复审结论 |
| --- | ---: | --- | --- |
| Orin/TensorRT 测量、Stage5/Stage7 编排、profiling 与 checkpoint 恢复 | 9 | `execution_contract_p6` | 实际输入涉及 scanner manifest、checkpoint、数据、ONNX/engine、设备运行时或正式 evidence，输出为测量、缓存、重定位、恢复或收口记录；公开树没有同路径或等价安全执行契约。 |
| Stage5 source-aware registry | 1 | `external_contract_p4` | 主导输入是 checkpoint inventory 与 materialization evidence；P4 必须先定义外部制品许可、版本、checksum、目录和路径安全投影。 |
| 上游端到端配置 | 2 | `blocked_license_or_permission` | Git 来源不是当前维护者，文件中未发现可确认的许可证/NOTICE。即使它们未含凭据或私有路径，也没有再分发依据，必须保持阻塞。 |

一处 `token` 静态命中已经上下文复核：它是进程生命周期函数的参数，不读取环境变量、不含硬编码值，也不构成凭据。未发现邮箱、IPv4、私有挂载路径或实际凭据；但所有非阻塞项仍可能输出运行时/制品 provenance，故不得直接迁入公开树。

## 证据与阶段边界

12 项均完成 P2 metadata/字节数与当前内容 SHA-256 对账、AST 解析、导入/调用审阅、公开同路径检查、来源历史和安全输出审查。P2b inventory 不带每文件历史 digest，因此不能将当前 SHA-256 写成历史哈希；源内容变化后须重新审计。

没有运行私有运行时、设备、数据、模型、engine、checkpoint、子进程或真实制品；每项记录了不可安全运行的理由。本批的受限 HMAC records 不构成关闭账本，P3 新标准累计数从 37/1,551 增至 49/1,551，P4 尚未启动。
