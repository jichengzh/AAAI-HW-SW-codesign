# P3-12：配置候选逐项处置

状态：96 项配置候选已逐项审阅并形成仅本机保存的受限 decisions/ledger 检查点；未迁入私有配置、未公开外部数据元数据、未提交私有清单、未推送。

## 审阅方法

每项配置均按其实际 JSON/YAML 内容而非文件名单独检查：可解析性、声明的消费者、是否包含正式实验参数/硬件环境/外部数据元数据、是否携带凭据或不可公开位置，以及是否属于公开支持链。量化/剪枝项还要求为非空、可解析的参数对象；硬件项要求包含明确的 CUDA、硬件或工具链语义；数据转换项要求为 JSON object/list。候选清单已在 P2b 执行敏感内容扫描，本批额外拒绝任何实验配置中出现 URL、SSH、口令、secret 或 API-key 类字段。

| 处置 | 数量 | 固定 reason / evidence | 结论 |
| --- | ---: | --- | --- |
| `execution_contract_p6` | 86 | `execution_pipeline_dependency` / `dependency_boundary_review` | 量化和剪枝参数定义真实实验的候选空间；不直接发布旧配置，而由 P6 以输入 hash、随机种子、命令和生成 manifest 固化。 |
| `environment_contract_p5` | 3 | `environment_specific_dependency` / `dependency_boundary_review` | 硬件 capability、工具链与 H800/4090 环境说明属于 P5 的 CPU/4090/H800 环境契约；旧环境值不能直接视为正式硬件证据。 |
| `external_contract_p4` | 6 | `external_artifact_dependency` / `dependency_boundary_review` | 数据转换元数据含外部数据集的类别、传感器、日志或标注引用；P4 必须确定数据许可、获取、版本与 checksum，不能把元数据当作公开数据集。 |
| `excluded_nonessential` | 1 | `nonessential_to_supported_flow` / `public_contract_review` | 内部代理运行复盘不参与公开 CPU smoke、Stage1--Stage7 契约或正式实验执行。 |

## 本地账本检查点

受限台账的 96 条记录严格对应 P2 分类 `migrate_config`，计数为 86 条 P6、3 条 P5、6 条 P4 和 1 条非必要排除。生成的路径无关 ledger 仅保存 HMAC-SHA256 candidate ID、分类、受控 token 和后续阶段；泄露扫描确认没有路径、origin、正文、人员、机器、地址或 key。

P3 当前有 122/1,551 项可复核 decisions：前序 P3-9--P3-11 的 26 项与本批 96 项。它们都只是本机 audit checkpoint，不能替代最终全量 ledger 或 P3 关闭。

## 后续顺序

1. P4 将 6 项外部元数据与其他外部数据候选转写为具有许可、获取、版本和 SHA-256 的公开获取契约；
2. P5 分别完成 CPU、4090、H800 环境契约，不公开登录方式、绝对路径或机密；
3. P6 将 86 项实验配置转写为有版本、seed、输入输出 hash 的执行 manifest；
4. P3 继续逐项审阅其余代码与文档候选，尤其是测试、第三方项目、实验脚本和设计文档。
