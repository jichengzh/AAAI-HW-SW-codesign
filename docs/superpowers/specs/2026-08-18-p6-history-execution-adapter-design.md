# P6.3 历史 CoptV2X 执行适配层设计

## 目标

把 P6.2 的动态 Stage2 Pyramid/H800/TVM 候选空间接入已有 CoptV2X Stage5 的真实四轮测量链。接入后，P6 仍负责 Gold176 cold start、预测、`predicted_frontier_diversity` 选点和四轮反馈更新；历史 Stage5 链仍负责候选物化、训练、checkpoint/ONNX、TVM、时延、能耗、AP 与实际反馈。

本设计不以手工 YAML 代替协议接入。它新增一个私有配置生成器和两个窄适配器，使已验证的历史组件成为 P6 framework mode 的本地执行后端。

## 已确认的边界

1. P6 选择阶段产出的测量请求已经是 `stage5_measurement_request_v2`；不需要再实现 P6 request 到 Stage5 request 的转换器。
2. 历史 Stage5 source materializer 只消费带完整 `source_contract` 的已认证请求行，不会根据 `(width, q_mode, source_point_ids)` 自行创建训练、导出或校准契约。
3. 因此缺失的是动态计划到 `stage5_candidate_source_registry_v2` 的契约扩展，以及历史实际反馈到 `p6_h800_coptv2x_feedback_v2` 的原子回写。
4. Stage5 是真实测量的执行权威。Stage7 仅可复用 GPU UUID、H800 型号、占用和漂移校验语义；不得把 Stage7 的设备排除策略带入 P6。

## 不变量

- 固定实验口径为 Pyramid、H800、TVM、Gold176、4 轮、每轮 4 个候选、共 16 个真实测量，以及 `predicted_frontier_diversity`。
- 一个候选只有一个全局 `q_mode`，只能为 `fp16` 或 `int8`；不会产生逐 stage mixed-precision 候选。
- framework candidate pool 必须是 Stage2 中所有 active、buildable、H800/TVM 合法点按全局 q mode 的精确组合。不得增加静态计数、FLOPs、代理时延、Pareto 或宽度范围剪枝。
- 所有计划候选必须能够生成完整、经验证的历史 source contract；任一个候选缺少权威物化绑定时，整个 framework preflight 失败。不得通过移除该候选来让实验继续。
- P6.1 static mode 保持 343 structures / 686 candidates 的历史门禁和执行行为不变；framework mode 绝不回退到 static registry。
- 所有真实路径、资产、命令、GPU UUID、候选 ID、请求、反馈、日志、checkpoint 和结果只存在于 Git 忽略的本地边界。公开代码、配置、文档和归档不得包含它们。
- 不下载资产，不调用 Orin、TensorRT、CPU、RTX 4090 或其他后端，不 push 或 merge。

## 架构

```text
public P6 contract + Stage2 search space
  -> private historical-binding discovery
  -> generated ignored P6 local config
  -> P6 dynamic candidate plan
  -> plan-to-registry-v2 adapter
  -> existing Stage5 manifest / cost-model / selection
  -> stage5_measurement_request_v2 (four rows)
  -> Stage5 history measurement adapter
  -> historical actual feedback
  -> P6 feedback adapter
  -> P6 online update; repeat four rounds
```

P6 controller remains generic: it still invokes only its local `source_registry_step` and `measurement_step` as argv-only processes. The generated local configuration points those steps to tracked Python adapters. The adapters may call historical components only through explicit argv and a constrained environment; no `shell=True`, shell interpolation or ambient fallback is permitted.

## 私有历史绑定发现

### 输入与输出

新增一个本地-only discovery/provision command。它接受一个允许的历史实验根、一个 local output root，并从 Git 忽略的私有 runner interface 派生恰好三张有序且唯一的 GPU index policy，静态检查历史组件、资源锚点和环境语义，并原子写出两份 Git 忽略文件：

1. `p6_history_binding_v1`：历史组件、输入资产、模型物化模板、环境激活、输出布局与 GPU UUID 锁的私有绑定。
2. `p6_h800_coptv2x_local_v2`：当前 controller 已接受的 local config；完整填写 `asset_paths`、四个 `local_input_paths`、`candidate_source_mode: framework_stage2_search_space`、`stage2_search_space_path`、两个 adapter step 与 `local_output_root`。

绑定发现只能选择唯一、版本兼容且结构完整的历史组件组合。发现多个候选、组件版本不匹配、缺少输入、路径越出许可根、资产验证失败、私有策略选择的任一 GPU 非 H800 或 UUID 漂移时，稳定失败并且不写可执行 config。

### GPU 准入

generator 和 measurement adapter 都重新读取私有策略选择的三张 GPU 的 index-to-UUID 映射，确认其型号满足 H800、未被不兼容进程占用且 UUID 与 binding 一致。调度只能在这三张卡内等待或分配；不得改用别的 GPU，也不得继承 Stage7 的其他设备排除策略。

## 动态计划到 registry-v2

新增 `plan-to-registry-v2` adapter，输入为：

- `p6_pyramid_candidate_plan_v2`；
- 经过 discovery 验证的 `p6_history_binding_v1`；
- 历史 Pyramid builder/pruner/export/calibration 的权威模板。

对每个计划候选，adapter 确定性创建 Stage5 所需的 source group 和 source contract，包括结构 group、训练/预训练绑定、checkpoint/ONNX/calibration 目标、source evidence 和 canonical contract hash。它输出 `stage5_candidate_source_registry_v2`，每个 group 仅声明该宽度实际存在的 `available_q_modes` 与对应三个 `source_point_ids_by_q_mode`。

输出必须满足以下精确性：

- registry 的 `(width, q_mode) -> source_point_ids` 映射与 P6 plan 完全相等；
- 每个 source contract 都可被历史 Stage5 materializer 验证；
- group 的 source evidence、contract hash 与路径/资产绑定可重算；
- 不包含 metrics、objectives、cache、terminal status 或 failure 内容；
- 不能建立完整 contract 的候选导致 adapter 失败，而非被过滤。

P6 controller 现有 plan/registry/manifest identity gate 继续作为第二道验证；适配器不能绕过它。

## Stage5 测量与 P6 反馈适配

measurement adapter 读取 P6 已生成的 `stage5_measurement_request_v2`，先重新验证：四行、row/request hash、H800、Pyramid、全局 q mode、固定指标、atomic feedback 与预算。随后按历史 Stage5 顺序调用候选物化、量化、性能计划/执行、AP 计划/执行和 feedback finalize。

历史链完成后，feedback adapter 只能从已验证的实际反馈构造 `p6_h800_coptv2x_feedback_v2`：

- `measurement_request_sha256` 必须精确匹配输入；
- row ID 集合必须与四个 request row 完全相等且唯一；
- 成功行必须含五个有限真实指标；
- 可行性失败只能使用 P6 允许的失败状态与脱敏原因；
- 任意 Stage5 receipt、source evidence、row hash 或实际反馈 barrier 不匹配时，adapter fail closed，不把子进程退出码当作成功。

所有中间请求、actual feedback、receipt、模型和日志保留在 private output root。P6 public surface 只会看到控制器的本地 completed/failed 结构状态，且真实结果不会被提交。

## 错误模型

以下情况必须在真实测量前停止：历史锚点不唯一、资源/环境缺失、Stage2 输入不合法、任何动态候选不能物化、registry identity 不一致、少于 16 个 eligible 候选、GPU 范围或 UUID/H800 型号不满足。

以下情况必须在该轮 feedback 前停止并让 P6 写入本地失败状态：Stage5 request 不合法、历史链执行失败、反馈非原子、五指标不完整或不合法、receipt/barrier/hash 漂移、结果不能转换为 P6 feedback。

失败信息面向公开 CLI 时仅保留稳定类别；私有上下文写在忽略 output root，不能进入 Git 或公开文档。

## 离线测试与验收

1. discovery tests 使用临时历史根和合成设备标识，覆盖唯一发现、歧义、缺少组件、允许根逃逸、资产验证、私有三卡策略的 UUID/H800/占用/漂移与原子写入；测试不得访问真实 GPU。
2. generator tests 验证产出能被 `load_local_config()` 接受，处于 Git 忽略目录，并且 public projection 不含路径、命令、GPU UUID 或秘密字段。
3. registry adapter tests 用合成 Stage2 candidate plan 与历史 binding，验证纯 FP16、纯 INT8 和混合候选池的 exact identity；缺一 source contract 就整体拒绝，不发生候选裁剪。
4. measurement adapter tests 用 fake Stage5 runner 和合成 receipt/feedback，验证四行 request、成功 feedback 转换、每个 hash/row/metric/barrier 不一致均 fail closed。
5. controller/CLI integration tests 验证 generator 产物走 framework mode，registry/manifest 与 plan 精确相等，动态非 343/686 池可完成四轮 fake feedback，P6.1 static 回归仍为 343/686。
6. 完整离线 quality、release、anonymous archive、Ruff 和 disclosure guard 全绿后，才可在 H800 上运行 discovery、动态 preflight 和授权的真实四轮实验。

## 非目标

- 不重写历史 Stage5 训练、TVM 或 AP 测量逻辑。
- 不把 Stage7 scheduler 变成 P6 的主执行器。
- 不引入自动候选剪枝、混合精度候选、替代 cost model、替代 acquisition 或替代后端。
- 不把生成的本地 binding/YAML、运行产物或任何实验结果变为受跟踪文件。
