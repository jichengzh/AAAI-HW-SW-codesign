# P6.1 Pyramid/H800/TVM CoptV2X 本地搜索执行边界

状态：**P6.1 已完成（本地）；P6.2 框架搜索空间接入与离线验证已完成（本地）**。本说明固定 P6.1 的公开契约和维护者本地执行边界；离线接入完成不表示 P6 整体、真实 H800 框架来源闭环、公开论文结果、Stage6 或 Stage7 已关闭。

## 固定范围与预算

P6.1 只验证 Pyramid/H800/TVM 的 CoptV2X 完整搜索闭环。候选空间有 343 个 Pyramid 结构，每个结构按 `q_mode` 展开为 `fp16` 与 `int8`，共 686 个可能的 TVM 候选。Gold176 保持为 176 条不可变 cold-start 证据，供 cost model 通过完整 capability profile 上下文学习跨配置趋势；当前 174 条成功实测记录进入回归，两条已确认真实可行性失败仅保留为可行性证据，不伪造指标、不进入回归，也绝不重新测量。候选预测、选点和真实测量始终只使用 H800/TVM 的目标 profile，绝不把另一后端候选混入本轮预算。

本地循环严格执行 **4 轮 × 4 个候选**，共 16 个测量槽位。每一个选中的候选都必须经过剪枝、训练或微调、模型检查点选择、ONNX 导出、TVM 编译和运行，并得到五项测量证据：`latency_ms`、`energy_j`、`ap30`、`ap50` 与 `ap70`。其中 latency、energy 与 AP70 用于 cost model 选择；AP30 和 AP50 是必须保留的测量证据。

P6.1 不执行 Orin、TensorRT 或其他硬件路线；Orin 和 TensorRT 均为后续独立工作。本阶段不下载资产、不自动探测硬件，也不自动启动硬件。

## P6.1 本地收口

在 Git 忽略的本地执行边界内，P6.1 控制器已完成固定的 4 轮 × 4 个候选闭环。收口核验只确认控制器完成状态、四轮请求/反馈配对完整，以及不存在失败终态；它不公开候选标识、原始测量值、资产位置、命令、日志或生成物。

这项结构性完成事实证明公开 P6.1 契约能够驱动经授权的本地真实闭环，但不等同于 Stage6 或 Stage7 论文证据完成，也不构成可公开复算的论文结果包。

## 公开契约、CLI 与本地输出

受版本控制的 `configs/execution/p6_h800_search.example.yaml` 只描述固定的 Pyramid/H800/TVM 契约、脱敏资产标签及其版本/许可状态和离线验证所需的元数据。它不包含本地路径、命令、主机信息、候选标识或原始测量数据。

公开 CLI `tools/release/run_p6_h800_search.py` 只接受明确给出的 contract、local config 和 public-safe code revision label。成功时仅输出 `completed`；契约或安全输出边界错误仅输出 `contract_error`；在有效契约之后发生的本地执行失败仅输出 `execution_failed`。CLI 不生成公开结果摘要，也不将本地状态、逐轮请求、反馈、候选标识、原始指标、日志、模型检查点、ONNX 或 TVM 产物写入公开面。

本地 YAML 位于 Git 忽略的 `configs/local/`，只能通过显式 local config 输入提供资产位置和 argv-only 适配器。它与配置的本地输出根共同保存 Gold176 输入、候选注册表、逐轮请求/反馈及执行产物；这些内容都不从公共契约推断，也不进入版本控制或匿名归档。共享资产、命令或结果结构不符合契约时必须整批隔离；只有已验证的候选真实可行性失败才消耗测量预算。

## P6.2 Framework Search Space Gate

P6.2 does not replace the P6.1 search loop. It changes only the candidate-source construction step: `stage2_search_space_v1` is converted into `p6_pyramid_structure_plan_v1`, then a Git-ignored local adapter materializes that plan into the CoptV2X source registry consumed by P6.1.

Framework mode is selected only in the Git-ignored local P6 configuration; the public contract stays path-free and contains no search-space or candidate identifiers. The converter rejects non-Pyramid models, non-H800 hardware, missing tuned TVM candidates, incomplete stage1/stage2/stage3 coverage, unsupported quantization policies, and non-buildable active points. It must not silently fall back to the static 343x2 grid（不得静默回退到静态网格）.

The framework source mode is named `framework_stage2_search_space`; this label identifies the conversion contract only and does not expose local configuration or candidate identifiers.

离线转换器和契约验证已完成，但这不是 P6 整体或真实框架来源闭环的完成声明。P6 收口前仍需单独授权真实本地验证；本文件不会启动硬件或训练。

## P6.2 与匿名归档

P6.2 的框架搜索空间接入与离线验证已完成；当前仅剩经单独授权的真实框架来源本地执行/闭环验证，完成后方可关闭 P6。P6.1 已完成不自动启动该验证；P6.1 不新增 P6.2 公开配置或示例。

匿名归档可以包含 v2 控制器、公开 CLI、P6.1 示例契约和纯公开的 framework search-space adapter；它必须排除 `configs/local/`、`outputs/`、`results/` 及任何本地运行材料。P6.1 的本地收口和 P6.2 离线接入完成均不改变这一归档边界；真实框架来源闭环仍待单独授权。
