# P3-14：第三方工程与执行配置代码逐项处置

状态：92 项工程代码候选已逐项审阅并形成仅本机保存的受限 decisions/ledger 检查点；未复制私有工程、未提交私有清单、未推送、未发布。

## 审阅方法

本批对每项候选分别检查 Python 可解析性、公开同路径替代、公开运行时导入、外部数据/checkpoint/硬件执行语义、网络与文件 I/O 信号，以及代码来源和许可证证据。当前公开支持运行时对私有 vendor package 的导入数为零；该 vendor package 也没有可随源码保留的项目级许可证或 NOTICE 证据。因此不能把其中任何文件当作已获再分发授权的代码迁入。

实验配置代码则逐项核对为可解析的执行配置，并确认不含凭据、维护者位置或公开运行时直接导入。它们定义训练/评测输入、模型权重、数据与硬件执行组合，应由 P6 固化为有版本、seed、输入输出 hash 的执行 manifest，而不是发布旧配置本身。

| 处置 | 数量 | 固定 reason / evidence | 结论 |
| --- | ---: | --- | --- |
| `blocked_license_or_permission` | 74 | `third_party_license_marker` / `license_marker_review` | 该 vendor package 的来源与再分发许可未形成可公开核验的完整证据链；P7 必须确定原始来源、许可证、修改范围和 NOTICE 后才能考虑发布。 |
| `execution_contract_p6` | 17 | `execution_pipeline_dependency` / `dependency_boundary_review` | Python 形式的实验配置属于训练/评测执行边界；P6 必须转写为可验证 manifest 和可用性状态。 |
| `excluded_nonessential` | 1 | `nonessential_to_supported_flow` / `public_contract_review` | 私有 package marker 不被当前公开运行时导入，也不构成公开支持链。 |

`blocked_license_or_permission` 是明确的安全与许可结论，不是迁移批准，也不意味着这些候选可以在开源前被忽略。若 P7 不能取得可再分发许可，正式可复现实验须使用独立公开实现或以 `unavailable` 失败关闭。

## 本地账本检查点

受限台账的 92 条记录严格对应 P2 分类 `migrate_code`，计数为 74 条许可证/权限阻塞、17 条 P6 执行契约和 1 条非必要排除。生成的路径无关 ledger 仅保存 HMAC-SHA256 candidate ID、分类、受控 token 和后续阶段；泄露扫描确认没有路径、origin、正文、人员、机器、地址或 key。

P3 当前有 502/1,551 项可复核 decisions：前序 P3-9--P3-13 的 410 项与本批 92 项。它们都只是本机 audit checkpoint，不能替代最终全量 ledger 或 P3 关闭；尤其 P7 的 74 项许可阻塞尚未解除。

## 后续顺序

1. P7 对许可证阻塞项逐项确认来源、原始许可证、修改清单、NOTICE 和再分发权；
2. P6 为仍在公开支持链内的执行配置建立版本化 manifest，而不恢复旧实验配置；
3. P3 继续审阅实验脚本、工具与设计文档，并在每一项决定后维护累计账本。
