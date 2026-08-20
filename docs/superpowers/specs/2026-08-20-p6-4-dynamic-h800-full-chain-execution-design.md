# P6.4 动态 H800 全链路执行设计

## 状态与目标

本规格将 P6.2 的动态候选空间与 P6.3 的历史实际测量链连接为一次可自主完成的 H800/TVM 实验。它验证的闭环为：真实 Pyramid 结构扫描、动态 Stage2 空间、Gold176 cold start、四轮 cost-model 选点、每轮四个真实候选的训练/TVM/AP/时延/能耗测量，以及实际反馈后的在线重训。

本规格只定义执行闭环。真实运行产生的路径、配置、候选、checkpoint、日志、原始指标和结果均为 Git 忽略的本地资产；公开仓库不记录它们。

## 已确认事实

- 目标固定为 Pyramid、H800、TVM、Gold176、4 轮、每轮 4 个候选、总预算 16；指标为 AP30、AP50、AP70、时延与能耗。
- Stage1 现有 `pyramid_lidar` 扫描可从真实 checkpoint 构造结构分区；不可追踪的稀疏/协同部分必须保留为扫描边界，不能据此声称完整模型性能。
- Stage2 候选由本次 Stage1 manifest 的 active、buildable、H800/TVM 合法点构造。一个候选有且只有一个全局 `q_mode`（`fp16` 或 `int8`）；不会引入 mixed-precision 候选。
- 已有私有本地 YAML 的资产和四项输入均可用，并可唯一确定一套同时拥有所需输入和历史执行组件的来源仓库。
- 旧本地 YAML 缺少动态 candidate-plan 输入，默认仍是 P6.1 的 static source path；它只能作为私有资产定位和历史执行信息的来源，不能直接启动 P6.4。
- 当前历史来源中没有可直接消费的 P6.3 runner interface，也没有已落盘的 Stage1 partition manifest。因此 P6.4 必须在运行前生成两者。

## 范围与非目标

P6.4 只运行 H800/TVM，并使用私有配置选择的三张 H800 设备。不会下载资产、调用 Orin/TensorRT/CPU/RTX 4090、重测 Gold176、回退 static registry、设置 343/686 固定规模或加入 FLOPs、代理时延、Pareto、宽度范围等预剪枝。

P6.4 不重写已有 Stage5 的训练、导出、TVM 或 AP 逻辑。它不把单次成功运行表述为论文数值已复现；论文表格级对比仍是后续证据整理工作。

## 总体架构

```text
private legacy locator
  -> private execution bootstrap
      -> actual Stage1 Pyramid scan on H800
      -> normalized Stage1 partition manifest
      -> Stage2 dynamic candidate plan
      -> source registry materialization
  -> Gold176 cold-start cost-model fit
  -> four rounds of select(4) -> actual Stage5 execution -> feedback -> refit
  -> completed / failed
```

所有 bootstrap、plan、registry、request、feedback、训练和测量产物写入显式的 Git 忽略输出根。公开 P6 contract 保持不含私有路径、命令、候选标识或实测数值。

## 组件设计

### 1. Stage1 空间构建

新增 framework-mode 的 `stage1_scan_step`。它只在本次私有输出根中写入 Pyramid 分区 manifest，并且必须：

1. 通过既有 `pyramid_lidar` adapter、真实 checkpoint 和 H800 capability 执行 Stage1 scan；
2. 产出带 `schema: stage1_partition_manifest_v1` 的 manifest。现有 scan 输出缺该 schema 字段，P6.4 需要补齐，以便后续 provision 与 Stage2 以同一个产物衔接；
3. 验证 `scan_status`、模型、H800 capability 和 Stage2 loader 可用性；
4. 不以 demo manifest、历史 static registry 或手工候选替代 scan 输出。

`stage2_search_space_path` 指向该预定的私有 manifest 路径。控制器先执行 `stage1_scan_step`，再调用现有 Stage2 loader 生成 `stage2_search_space_v1` 与动态 `p6_pyramid_candidate_plan_v2`。

### 2. 私有执行 bootstrap

bootstrap 以旧本地 YAML 为私有 locator，而非公开配置来源。它验证资产/输入存在性，并从唯一共同来源选择历史组件；不得对大目录做“第一个文件获胜”的猜测。

bootstrap 写入或更新下列 Git 忽略文件：

- 本次 Stage1 partition manifest；
- `p6_history_runner_interface_v1`；
- `p6_history_binding_v1`；
- dynamic `p6_h800_coptv2x_local_v2` config。

runner interface 固定既有 Stage5 的实际执行次序：源码物化、量化、性能、AP、反馈收口。它使用从历史组件定义中解析并验证过的 argv 和环境激活方式，不从文件名猜测参数，也不使用 shell 拼接。历史组件无法提供完整、可验证的接口或反馈布局时，bootstrap 以稳定类别失败，绝不改用模拟适配器。

本设计采用结构性存在性、唯一性和输出格式校验，以及已有 P6 request/row identity 校验；不新增每个外部资产的 SHA256 清单。

### 3. 动态空间与 registry

Stage2 依本次 manifest 的实际输出分别组合全局 FP16 和全局 INT8 候选。候选规模由输出决定，允许 FP16-only、INT8-only 或两者并存，但不产生候选内混合精度。

registry materializer 必须为计划中的每个 `(width, q_mode)` 创建可由历史 Stage5 物化的 source contract。plan、registry 与 Stage5 manifest 的候选身份集合必须精确相等。少于 16 个可测候选、遗漏、重复或无权威 source contract 时，在任何真实测量前失败；不得删除候选后继续。

### 4. 四轮搜索状态机

```text
preflight
  -> stage1_ready
  -> candidate_pool_ready
  -> round_0_coldstart_fit -> select_4 -> actual_measure -> feedback
  -> round_1_online_fit    -> select_4 -> actual_measure -> feedback
  -> round_2_online_fit    -> select_4 -> actual_measure -> feedback
  -> round_3_online_fit    -> select_4 -> actual_measure -> feedback
  -> completed
```

Gold176 只用于初始 cost-model 训练，绝不重新测量。每轮仅在上一轮四条实际反馈的请求、行身份、receipt/barrier 与五项指标均验证通过后进入下一轮。任何失败写入私有失败状态并停止；不跳轮、不补模拟值、不重启为 static mode。

### 5. GPU 调度

仅使用私有配置选择的三张 H800 设备。执行器在 Stage1 前、每轮前和每轮实际测量后检查三卡的 H800 型号、绑定的 index/UUID 映射以及占用阈值。未满足准入时自动等待并重检；准入后发生漂移或不兼容占用时本次运行失败，避免把受污染测量并入 feedback。

Stage1 scan 可以暂时使用私有配置选择的三张 H800 中的一张；它结束后仍须重新通过三卡联合准入，才能进入真实四轮测量。

## 错误语义

以下情况必须停止且不启动测量：私有资产/运行接口不可用，Stage1 失败，manifest schema/Stage2 loader 失败，动态候选不足 16，registry/manifest 身份漂移，或 GPU 不准入。

以下情况必须停止且不进入下一轮：历史 Stage5 子链失败，四行 batch 不完整，request/row identity 或 receipt/barrier 不一致，真实反馈缺少五项指标，时延/能耗非正，或 AP 不在合法区间。

公开 CLI 仅输出稳定失败类别；详细诊断保留在忽略输出根。

## 验收与 P6 收口

离线验收必须采用 TDD，至少覆盖：

1. Stage1 scan manifest 的 schema、真实输入边界、失败状态和 Stage2 可加载性；
2. bootstrap 的唯一来源选择、拒绝歧义、私有输出和无公开泄露；
3. 动态 plan/registry/Stage5 manifest 的 exact identity、非固定候选数量、FP16/INT8 独立性与不少于 16 的门槛；
4. controller 的 Gold176 cold start、四次 online feedback update、四轮四行批次和失败即停；
5. 私有配置选择的三张 H800 的准入、等待、漂移与 Git 忽略输出边界；
6. P6.1 static mode 的既有 343/686 回归不变。

真实运行的 P6 执行闭环收口条件为：本次 Stage1 scan 成功、动态候选池通过 preflight、四轮全部完成、16 个唯一候选均有已验证的实际反馈，最终状态为 `completed`。之后可以将 P6 标记为执行闭环完成，但不得仅凭该次运行宣称论文表格已复现。

## 实施顺序

1. 扩展 Stage1 manifest 与 P6 local framework config，使 Stage1 output 能被动态 controller 消费。
2. 实现私有 bootstrap/runner-interface materialization 与动态 local config 生成。
3. 以 TDD 接入 Stage1 -> Stage2 -> registry -> historical measurement 的 controller 状态机，并保持 static mode 回归。
4. 完成离线质量门禁和 disclosure 检查。
5. 在私有配置选择的三张 H800 准入后执行真实四轮；只在满足收口条件时更新 P6 审计状态。
