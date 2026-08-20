# P6 历史程序式物化器到声明式 Recipe 桥接器设计

## 背景与根因

P6 的目标链路是 Stage1 Pyramid scan → Stage2 动态候选计划 → registry-v2 → 四个前沿候选实测 → feedback 更新 cost model。当前 normalizer/registry 已能消费 `p6_history_dynamic_materialization_recipe_v1`，但 v1 把训练、checkpoint、ONNX 和 calibration 路径按 `q_mode` 分开。这不符合已确认的 CoptV2X/Pyramid 历史接口：同一 width/group 只训练并导出一份共享 source bundle；FP16/INT8 是搜索维度，只在 quantization、performance 和 AP 阶段分流。

历史 Stage5 链也没有保存一份现成声明式 recipe。公开可验证的程序组件只有以下四个 marker：

- `controller` → `stage5_task_round_controller_v3.sh`
- `source_materializer` → `stage5_materialize_round_sources_v1.sh`
- `performance_plan` → `stage5_build_performance_plan_v2.py`
- `finalizer` → `stage5_finalize_feedback_v2.py`

runner template 的执行阶段是 `source_materialization`、`quantization`、`performance`、`ap`、`finalization`。训练、checkpoint、ONNX 与 calibration 是 `source_materializer` 所产出的共享 bundle 字段，不是可独立证明的 component role；因此不得虚构 `training`、`checkpoint_export`、`onnx_export` 或 `int8_calibration` marker。

本设计增加一个窄桥接器：在 provision 之前，静态验证 private runner template 与四个公开 marker，并从版本化公开 profile 派生 `p6_history_dynamic_materialization_recipe_v2`。registry-v2 按 group 渲染一份共享 source bundle；measurement request 在进入历史物化器前由版本化 projection adapter 转成历史程序所需的 flat `source_contract`，重新计算已有 row/request canonical hashes，再按 group 去重，以 direct argv 调用 source materializer。桥接器不扫描或执行历史代码来推断接口。

## 目标

1. 保持 `p6_history_dynamic_materialization_recipe_v1` 的读取兼容，同时新增语义正确的 recipe v2。
2. recipe v2 为每个 width/group 只渲染一份 `shared_source_path_templates`；同组 FP16/INT8 必须共享 source path，跨组必须无碰撞。
3. 公开 profile 只证明四个程序 marker、五个 execution stage、Pyramid 三阶段 width surface、recipe-v2 路径字段和 source materializer 的 direct-argv 调用形状。
4. 静态 base checkpoint、数据/config 根和训练超参数继续来自 Git 忽略的 private source-contract template；公开 profile 不承载其值。
5. 新增版本化 request projection + source invocation adapter，按 group 去重调用 source materializer，并保持后续 private quantization/performance/AP/finalizer wrapper 不变。
6. 失败复用现有 canonical row/request hash，不新增资产级 hash 账本、目录 Merkle 树或 checkpoint/ONNX hash 清单。
7. 在任何真实训练或测量前，用 injected runner 完成真实 Stage2→registry-v2→request→projection→source-invocation-shape 的 zero-GPU 黑盒验证。

## 非目标

- 不重新下载资产，不安装/重建 TVM，不执行历史 controller。
- 不从脚本正文、文件名或目录结构推断未声明能力。
- 不改变 quantization、performance、AP、finalizer 的 private wrapper 或真实反馈格式。
- 不固定 Stage2 候选数，不恢复 343/686 后备候选集合。
- 不把 private 路径、真实 GPU index/UUID、资产/候选 ID、checkpoint/ONNX/calibration 实例、指标、结果或日志提交到 Git。
- 本桥接计划不启动训练、导出、量化、TVM 调优、AP、时延或能耗测量；真实四轮仍是后续单独授权阶段。

## 采用方案

采用“版本化 profile + 静态 recipe-v2 + 版本化 request projection”方案。

不采用手写 v1 recipe，因为它会继续错误地按 q_mode 重复训练/source bundle；不采用 dry-run tracing，因为 dry-run 仍可能读取资产、初始化环境或产生输出；不采用通用代码推断，因为历史 shell/Python 接口不构成稳定 schema。

## 架构与无环数据流

```text
ignored private source-map v2
  + ignored pre-provision runner template
  + fresh private Git root
  + public profile v1
  -> shared runner-template validator
  -> recipe bridge
  -> p6_history_dynamic_materialization_recipe_v2
  -> normalizer
  -> provisioned private binding
  -> Stage1 scan
  -> Stage2 dynamic candidate plan
  -> registry-v2 (one shared source bundle per group)
  -> Stage5 measurement request v2
  -> request projection (flat source_contract + recomputed existing hashes)
  -> source invocation adapter (one direct-argv call per distinct group)
  -> existing quantization/performance/AP/finalizer wrappers
```

recipe 派生不读取未来 binding；registry 不读取 measurement request；projection 不反向修改 registry 或 Stage2 plan。post-provision 只比较 binding 中的 recipe canonical 值与 pre-provision 派生值，不形成 recipe→binding→recipe 循环。

## 权威接口面

### Runner template

共享 validator 读取并冻结：

- `p6_history_runner_template_v1` schema；
- controller argv0 与 execution-chain argv0；
- 四个公开 marker basename；
- exactly-once execution stages：`source_materialization`、`quantization`、`performance`、`ap`、`finalization`；
- environment、activation argv、output layout 和 GPU policy 所需的既有字段；
- 所有 component path 必须在同一 fresh private Git root 内、非软链接、可执行。

profile 不给 quantization 或 AP executable 指定公开 basename；它们只需作为 runner template 中唯一的相应 stage 通过既有 path/argv 验证。

### Profile registry

公开 profile `p6_stage5_pyramid_h800_tvm_profile_v1` 只包含：

- target：`model=pyramid`、`hardware=h800`、`backend=tvm_auto`；
- exactly 四个 marker role/basename；
- exactly 五个 execution stage 及其顺序；
- width surface：`stage1_width/stage2_width/stage3_width`；
- q-mode surface：`fp16/int8`，并明确 q-mode 不进入 shared source identity/path；
- recipe-v2 的 11 个 shared path template key；
- source materializer direct argv 形状：`<executable> --request <path> --model pyramid --group-id <group_id> --gpu <validated_private_gpu>`。

source-map 不得自报 capability、argv、marker 或输出路径；这些值分别来自公开 profile、validated runner template 和 ignored source-contract template。

## Source-map 与 recipe schema

### Source-map v1 兼容

`p6_history_normalization_source_v1` 与显式 `p6_history_dynamic_materialization_recipe_v1` 行为保持不变。v1 不自动升级、不进入程序式派生，也不用于本轮真实 CoptV2X/Pyramid source materialization。

### Source-map v2

`p6_history_normalization_source_v2` 支持互斥两路：

- `recipe_mode=explicit_dynamic_recipe`：只允许 `dynamic_materialization_recipe`；recipe 可以是受支持的 v1 或 v2。
- `recipe_mode=procedural_profile`：只允许 `procedural_recipe_profile` 和最小 `procedural_recipe_source`；不得同时携带显式 recipe。

程序式 source 仅定位 ignored runner template、fresh private Git root 和四个结构化 role ref；它不能声明 semantic argv、capability、shared path 或训练参数。字段缺失、两路同时存在、两路都不存在或 mode/字段不一致均以 `history_recipe_derivation_invalid` 失败。

### Recipe v2

```yaml
schema_version: p6_history_dynamic_materialization_recipe_v2
stage_width_fields: [stage1_width, stage2_width, stage3_width]
group_id_template: "pyramid|{stage1_width}x{stage2_width}x{stage3_width}"
artifact_id_template: "pyramid-{stage1_width}-{stage2_width}-{stage3_width}"
shared_source_path_templates:
  checkpoint_path: relative-template
  checkpoint_dir: relative-template
  config_path: relative-template
  training_done_marker: relative-template
  onnx_path: relative-template
  onnx_report_path: relative-template
  calibration_root: relative-template
  calibration_npz: relative-template
  calibration_summary: relative-template
  trt_calibration_dir: relative-template
  source_done_marker: relative-template
```

规则：

- 11 个 key 必须完整且无额外字段；模板只允许三阶段 width、`group_id`、`artifact_id`。
- 模板必须是相对于本次 ignored local output root 的路径，不得绝对、包含 `..` 或与 registry/request 输出碰撞。
- 同一 group 的所有 q-mode row 使用同一组渲染结果；q_mode 既不出现在 template placeholder 中，也不进入 artifact ID。
- 对不同 width/group 的采样渲染结果必须逐字段不碰撞。
- recipe 不承载 base checkpoint、dataset/config root、训练超参数或 GPU；这些静态值保留在 ignored source-contract template。

registry-v2 对每个 group 生成一次 `shared_source_paths`，并继续保留 `available_q_modes` 与 `source_point_ids_by_q_mode` 来表达 FP16/INT8 搜索维度。若同组同时存在 FP16/INT8，它们引用同一个 source contract/shared bundle。

## Request projection 与 source invocation

新增版本化 adapter 接受已验证的 `stage5_measurement_request_v2`、recipe-v2 registry source contracts、private round root 和已验证 GPU policy，返回一个新的 request 对象与按 group 排序的 invocation plan；不原地修改输入。

projection 对每行执行：

1. 验证 row/group identity 与 registry group 一致；
2. 从该 group 的 `shared_source_paths` 取得 11 个动态字段；
3. 将 ignored source-contract template 中的静态 base checkpoint/config/training 参数与 11 个动态字段合并成历史 materializer 需要的 flat `source_contract`；
4. 删除仅供声明式 registry 使用的 recipe/shared wrapper 字段；
5. 重新计算该行已有的 `source_contract_sha256` 和 `row_sha256`；
6. 所有行完成后重新计算已有 `measurement_request_sha256`。

同一 group 的 FP16/INT8 row 必须得到完全相同的 11 个 source path 字段。不同 group 的任一路径相同即 `history_execution_invalid`。projection 不添加新的 hash 字段或 hash ledger。

invocation adapter 按 request 中首次出现顺序对 `group_id` 去重。每个 group 只执行一次：

```text
<validated source_materializer executable>
  --request <projected private request path>
  --model pyramid
  --group-id <group_id>
  --gpu <one index from validated private GPU policy>
```

argv 必须以 sequence 直接交给 injected `Runner`，`shell=False`。GPU 由调用方从已验证 private policy 调度，adapter 必须验证其属于 policy 且对应 UUID/model/occupancy gate 已通过；公开代码和测试不得固定真实编号。adapter 不执行 controller。source materialization 成功后，现有 quantization/performance/AP/finalizer wrappers 继续消费 projected request 和已有 private task state，不改变其契约。

## 失败类别与原子性

- `history_recipe_derivation_invalid`：source-map v2 互斥错误、unknown profile、runner template/marker/stage 漂移、recipe-v2 字段/模板/路径碰撞、private root/软链接/ignore 边界错误。
- `history_execution_invalid`：request/registry identity 漂移、flat projection 字段缺失、同组 source path 不一致、跨组碰撞、hash 重算不一致、GPU 不属于 validated policy、source invocation argv 无法构造。
- 历史进程非零或抛异常继续使用既有执行失败类别；不把私有 stderr 暴露到公开输出。

派生、normalize、provision、projection 或 invocation-plan 构造失败时不得写 partial recipe、normalized root、binding pair、projected request 或 task state。projected request 只在全部行和 hashes 验证通过后原子写入 ignored private round root。任何 derivation/execution-invalid 失败都不得调用历史脚本。

## 隐私与安全边界

公开文件可以包含 schema、角色、四个 marker basename、五个 stage 名、11 个字段名、相对模板规则、direct argv flag 名、稳定失败类别和合成测试值。

公开文件不得包含真实 private path、主机名、GPU index/UUID、资产/候选 ID、base checkpoint 值、数据/config 路径、训练超参数、产物实例、AP/latency/energy、结果或日志。private source-map、source-contract template、派生 recipe、binding、local YAML、registry、request 和 projected request 只能存在于 Git 忽略输出根。

这是即将开源的研究仓库，不增加不必要的防御层。只保留 schema 边界、root/symlink、canonical identity/hash、direct argv 和 no-partial-write 门禁；不新增大规模 SHA256 账本或不可能场景的推断分支。

## TDD 验证矩阵

| 场景 | 期望 |
| --- | --- |
| v1 explicit recipe | 继续按原行为通过 |
| known profile + valid runner template | 派生 recipe v2；历史脚本调用数为 0 |
| marker/stage/profile drift | `history_recipe_derivation_invalid`；无输出 |
| recipe v2 key 缺失/额外/q_mode placeholder | `history_recipe_derivation_invalid` |
| Stage2 mixed q-mode same group | registry 中一份 shared bundle；FP16/INT8 row 的 11 个 flat path 相同 |
| two different groups | 11 个路径跨组无碰撞 |
| projection hashes | source-contract、row、request 的既有 canonical hash 全部更新且可复验 |
| invocation shape | 每个 distinct group 一次 direct argv，含完整 `--request/--model/--group-id/--gpu` |
| zero-GPU black box | 真实 Stage2→registry-v2→request→projection→invocation plan；只调用 injected runner，不执行历史脚本 |
| projection/argv drift | `history_execution_invalid`；无 projected request、无 runner call |
| H800 preflight | 只验证到 invocation shape；训练和测量调用数均为 0 |

## 部署与回滚

部署顺序：共享 runner-template validator → profile/recipe-v2 renderer → private derive CLI → source-map/normalizer/registry-v2 → request projection/source invocation adapter → zero-GPU lifecycle → private H800 zero-measurement preflight。

任一步失败都可关闭 source-map v2 procedural mode，并继续保留 v1 explicit 行为。无需迁移历史结果，因为本计划不改写历史资产、不执行训练/测量，也不提交 private 产物。

## Acceptance criteria

1. v1 recipe 兼容，v2 recipe 能完整表达 11 个共享 source path 字段。
2. 公开证据仅使用四个 marker 和五个 execution stage，不存在虚构 training/export/calibration component。
3. 同 group FP16/INT8 共享 source bundle；不同 group 路径不碰撞；候选规模完全由 Stage2 输出决定。
4. private source-contract template 继续提供 base checkpoint/config/training 参数；公开 profile 无私有值。
5. request projection 生成 flat source contracts，并正确重算现有 source-contract/row/request hashes。
6. source materializer 按 group 只调用一次，argv 完整且 GPU 来自 validated private policy；controller 未执行，后续 wrapper 契约未改变。
7. 所有派生失败归类为 `history_recipe_derivation_invalid`，投影/调用契约失败归类为 `history_execution_invalid`，且失败无 partial 写入和历史进程调用。
8. zero-GPU injected-runner 黑盒覆盖真实 Stage2→registry-v2→request→projection→source-invocation shape。
9. H800 retry 只完成 zero-measurement preflight，不开始训练、导出、量化、AP、时延或能耗测量。
