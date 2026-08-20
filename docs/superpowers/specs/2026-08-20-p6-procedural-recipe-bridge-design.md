# P6 程序式物化器到声明式 recipe 桥接器设计

## 背景与根因

P6.4 已经要求真实闭环从 Stage1 Pyramid scan 出发，经 Stage2 动态候选计划、registry-v2、Stage5 历史实际测量适配器和 P6 feedback 更新完成四轮 H800/TVM 执行。当前 normalizer 已能消费显式的 `p6_history_dynamic_materialization_recipe_v1`，并把 private source-map 中的一份 source contract/template 规范化为可由 registry adapter 使用的动态 recipe。

剩余断点是：已确认的 Stage5 历史物化链并不天然保存一份声明式 recipe。它以程序式组件表达训练、checkpoint、ONNX、calibration 和量化输出布局；组件能执行真实物化，但没有直接提供 P6 registry-v2 需要的相对输出模板和候选 identity 渲染规则。若让公开 P6 直接猜测这些程序、执行历史脚本或手写一份 recipe，都会破坏 P6 的 fail-closed、隐私和动态候选身份约束。

根因可以分成三层：

1. Stage5 历史接口是程序式的，权威信息散落在已确认组件的角色、argv 语义、marker basename 和输出布局中；
2. P6 registry-v2 需要纯声明式、可静态验证的 recipe，以便在 zero-measurement 预检中为 Stage2 动态计划里的每个 `(width, q_mode)` 生成 source contract；
3. 公开仓库不能记录私有路径、真实 GPU、资产 ID、候选 ID、checkpoint、ONNX、calibration 实例、指标或结果，因此转换必须发生在 Git 忽略的私有输入边界内，公开代码只持有版本化 profile 和验证逻辑。

本设计新增一个窄桥接器：它把已验证 private binding/template、已选择程序式组件和一个版本化 adapter profile 转换成 `p6_history_dynamic_materialization_recipe_v1`。桥接器只做接口验证与声明式 recipe 渲染，不做通用代码推断，不运行历史脚本，不训练、不导出、不测量。

## 目标

1. 支持从已确认 Stage5 程序式接口派生 `p6_history_dynamic_materialization_recipe_v1`，供现有 normalizer 与 registry-v2 materializer 消费。
2. 引入版本化 adapter profile registry。每个 profile 明确支持的组件角色、marker basename、语义 argv/placeholder surface、三阶段 width surface、FP16/INT8 能力，以及 training/checkpoint/ONNX/calibration 输出能力。
3. 将 private source-map 升级到 schema v2，同时保持 v1 显式 recipe 兼容。v2 必须在 `dynamic_materialization_recipe` 与 `procedural_recipe_source/profile` 之间二选一。
4. 对未知 profile、接口漂移、角色缺失、precision 能力缺失、placeholder 缺失、输出模板碰撞、软链接和越界路径全部 fail closed。
5. 只复用现有契约已有的身份校验：Pyramid canonical group identity、Stage2 plan/registry identity gate、source contract canonical validation 和既有 contract hash。桥接器不新增大规模 hash/SHA 体系。
6. 在真实 H800 四轮之前完成 recipe→normalizer→provision→Stage1→dynamic Stage2→registry-v2 的 zero-measurement 预检。

## 非目标

- 不做通用 Python/shell 代码推断，不从文件名、目录名或脚本内容猜测语义。
- 不执行历史程序、dry-run 历史脚本、训练、量化、导出、TVM 编译、AP 评估、时延或能耗测量。
- 不固定候选数量，不恢复 static 343/686 作为 framework mode 后备。候选数完全由 Stage2 dynamic plan 决定。
- 不把 private source-map、binding、local YAML、registry、Stage1 manifest、Stage2 plan、candidate ID、checkpoint、ONNX、calibration、日志或指标提交到公开仓库。
- 不新增资产内容哈希清单、目录级 Merkle 树或每个 checkpoint/ONNX 的新 SHA 账本。
- 不把本阶段验收等同于真实四轮实验完成。真实四轮仍是后续单独阶段，需在离线门禁后获得独立运行授权。

## 三方案比较

| 方案 | 做法 | 优点 | 风险 | 决策 |
| --- | --- | --- | --- | --- |
| A. 手工声明式 recipe | 由操作者直接写 `p6_history_dynamic_materialization_recipe_v1` | 实现量最小；沿用现有 v1 normalizer | 人工容易遗漏程序式组件真实能力；无法证明 recipe 与已确认 Stage5 接口一致；可能把私有路径或候选痕迹写入公开配置；漂移时不会自动 fail closed | 不采用 |
| B. runtime dry-run tracing | 运行历史物化脚本的 dry-run 或 mock 模式，捕获输出路径 | 看似贴近真实程序 | dry-run 仍会执行历史代码，可能读取资产、初始化 GPU/环境、生成临时输出或泄露路径；不同历史脚本 dry-run 语义不稳定；无法保证 zero-measurement | 不采用 |
| C. 版本化 profile + 静态接口验证 | profile 固定可接受的组件角色、marker basename、semantic argv 和 capabilities；桥接器只验证 selected components 并渲染 recipe | 不执行历史代码；可审计、可测试、可 fail closed；公开 profile 不含私有路径；能与 v1 显式 recipe 并存 | 每个已确认 Stage5 接口需要一个明确 profile；接口漂移必须更新 profile 与测试 | 采用 |

采用 C 的关键理由是：P6 需要的是一个可静态证明的桥接契约，而不是“看起来能跑”的历史命令重放。profile registry 把人工确认结果固化成版本化接口面，桥接器只接受该接口面的精确匹配，避免通用推断和隐式后备。

## 架构与数据流

桥接器拆分为六个部分：

1. profile registry：公开、版本化、无私有值；记录一个 profile 对应的 Stage5 程序式接口面。
2. interface verifier：在私有 source-map v2 中读取已选择程序式组件，验证角色、marker basename、semantic argv、placeholder surface、precision 能力和输出能力。
3. recipe renderer：从已验证的 profile 与组件接口渲染 `p6_history_dynamic_materialization_recipe_v1`，只生成 canonical identity 和 per-q-mode 相对输出模板。
4. normalizer integration：source-map v2 入口在显式 recipe 与程序式来源二者之间选择一路，向现有 normalizer 传入同一份 recipe 结构。
5. CLI：提供 private-only 的 recipe 派生命令，稳定输出成功类别或 `history_recipe_derivation_invalid`。
6. preflight/lifecycle：在任何 measurement 前运行 recipe derivation、normalizer、provision、Stage1、Stage2 dynamic plan 和 registry-v2 identity gate。

数据流如下：

```text
ignored private source-map v2
  + validated private p6_history_binding_v1/template
  + selected procedural Stage5 components
  + public adapter profile name
  -> interface verifier
  -> recipe renderer
  -> p6_history_dynamic_materialization_recipe_v1
  -> existing history normalizer
  -> ignored private root and local config
  -> provision
  -> Stage1 scan
  -> Stage2 dynamic candidate plan
  -> registry-v2 materializer
  -> zero-measurement preflight accepted
```

桥接器不读取 Stage2 candidate rows，也不决定候选集合。recipe 只描述“任意合法三阶段 Pyramid width 与 q_mode 应如何形成 source group 和相对输出布局”。最终 registry 中出现多少 groups 与 q modes，由 Stage2 dynamic plan 的实际 `candidates` 决定。

## Profile registry

profile 是公开代码中的版本化常量或数据对象，名称必须稳定，例如 `p6_stage5_pyramid_h800_tvm_profile_v1`。profile 内容只允许描述接口形状，不允许包含绝对路径、GPU 编号/UUID、私有资产名、候选 ID、checkpoint 名称、ONNX 名称、calibration 实例、指标或结果。

每个 profile 至少包含：

- `profile_name` 与 `profile_schema_version`；
- `target`: `model=pyramid`、`hardware=h800`、`backend=tvm_auto`；
- exact `component_roles`：程序式组件必须覆盖并且只覆盖已批准角色，例如 source materialization、training、checkpoint export、ONNX export、INT8 calibration、performance/AP plan 或 feedback boundary 中 profile 声明的角色；
- 每个角色允许的 `marker_basename` 集合，比较 basename 而非完整路径；
- 每个角色的 required semantic argv surface：参数名、是否必须出现、是否必须引用 width placeholder、q_mode placeholder、private root placeholder 或 output placeholder；
- `stage_width_surface`: exactly three stage axes，按 `stage1_width`、`stage2_width`、`stage3_width` 绑定；不接受单 stage、四 stage、命名漂移或 mixed width surface；
- `precision_surface`: `fp16` 与 `int8` 是否支持，以及每种 q mode 需要哪些组件能力；
- `output_capabilities`: training、checkpoint、onnx、calibration 四类输出能力，以及它们能否为每个 q mode 生成相对模板；
- `identity_policy`: 使用现有 Pyramid `canonical_group_id("pyramid", width)` 作为 group identity，artifact identity 仅由三阶段 width 派生；
- `public_projection_policy`: profile 公开投影中不得出现 private path、GPU、asset/candidate ID、checkpoint、metrics 或 result 字段。

profile 是 allowlist，不是 heuristic。profile 名称未知、profile schema 版本未知、组件多于或少于 profile exact roles、marker basename 不匹配、argv surface 多义、placeholder 缺失或 precision 能力不完整时，桥接器必须失败。

## Source-map schema 与接口

### v1 兼容

现有 `p6_history_normalization_source_v1` 继续表示“调用方已显式提供 `dynamic_materialization_recipe`”。v1 行为不变：normalizer 仍按既有字段验证 source contract、recipe、asset paths 和 input sources。

v1 不新增程序式字段。v1 输入中缺少 recipe 时仍按既有失败类别处理；不得自动进入程序式推断。

### v2 新 schema

新增 `p6_history_normalization_source_v2`。v2 的公共语义是：公共代码可验证字段形状，但文件本身仍是 Git 忽略的私有输入。v2 top-level 字段为：

```text
schema_version: p6_history_normalization_source_v2
history_root: absolute private Git root, ignored
asset_paths: existing private asset mapping
input_sources: existing private JSON input mapping
source_contract: private source group/template
recipe_mode: explicit_dynamic_recipe | procedural_profile
dynamic_materialization_recipe: present only when recipe_mode is explicit_dynamic_recipe
procedural_recipe_source: present only when recipe_mode is procedural_profile
procedural_recipe_profile: present only when recipe_mode is procedural_profile
```

互斥规则：

- `recipe_mode=explicit_dynamic_recipe` 时，必须存在 `dynamic_materialization_recipe`，且不得存在 `procedural_recipe_source` 或 `procedural_recipe_profile`；
- `recipe_mode=procedural_profile` 时，必须存在 `procedural_recipe_source` 与 `procedural_recipe_profile`，且不得存在 `dynamic_materialization_recipe`；
- 同时出现两种来源、两者都缺失、`recipe_mode` 与字段不一致，均以 `history_recipe_derivation_invalid` 失败；
- v2 中派生出的 recipe 只传入 private normalizer/runtime，不写入公开 tracked recipe 文件。

`procedural_recipe_profile` 只包含 profile 名称和 profile schema version。`procedural_recipe_source` 只包含已验证 binding/template 中对程序式组件的私有引用、组件角色、marker basename、semantic argv projection、capability projection 和相对输出 surface。该 projection 可以含私有路径，因为 source-map v2 是 ignored private input；但桥接器不得把这些私有路径复制进公开 profile、公开 recipe 或公开日志。

### 程序式组件 projection

每个 selected procedural component 必须以结构化字段描述：

- `role`: 必须精确命中 profile role；
- `marker_basename`: 只比较 basename；
- `argv_semantics`: 由 profile 定义的 required semantic argv surface 的实例化投影；
- `placeholders`: 组件承认的 placeholder 名称集合，必须覆盖 profile 所需 surface；
- `capabilities`: 组件声明支持的 q mode 与 training/checkpoint/onnx/calibration 能力；
- `output_surface`: 输出 root 必须是 private output root 下的相对模板 source，不能是绝对产物路径；
- `binding_ref`: 指向已验证 private binding/template 的本地引用，不进入公开 recipe。

projection 不是代码扫描结果。它是 private bootstrap 在已批准历史组件上产生或复制的结构化接口说明；桥接器只验证它与 profile 是否精确一致。

## Recipe renderer

renderer 的输出 schema 固定为 `p6_history_dynamic_materialization_recipe_v1`，字段与现有 registry materializer 对齐：

```text
schema_version: p6_history_dynamic_materialization_recipe_v1
stage_width_fields: [stage1_width, stage2_width, stage3_width]
group_id_template: pyramid|{stage1_width}x{stage2_width}x{stage3_width}
artifact_id_template: pyramid-{stage1_width}-{stage2_width}-{stage3_width}
output_path_templates_by_q_mode:
  fp16:
    training_path_template: relative template
    checkpoint_path_template: relative template
    onnx_path_template: relative template
    calibration_path_template: relative template
  int8:
    training_path_template: relative template
    checkpoint_path_template: relative template
    onnx_path_template: relative template
    calibration_path_template: relative template
```

identity 规则：

- group identity 必须等于现有 Pyramid canonical rule：`pyramid|{stage1_width}x{stage2_width}x{stage3_width}`；
- artifact identity 只能由三阶段 width 派生，不能包含 profile 名、q mode、资产 ID、候选 ID、时间戳或 checkpoint；
- q mode 只影响 `output_path_templates_by_q_mode` 下的相对路径模板；
- `fp16` 与 `int8` 是全局 q mode；桥接器不得产生 mixed precision；
- renderer 不枚举候选。Stage2 dynamic plan 中没有的 `(width, q_mode)` 不会出现在 registry。

输出模板规则：

- 模板字段只允许使用 `stage1_width`、`stage2_width`、`stage3_width`、`group_id`、`artifact_id`、`q_mode`；
- 渲染结果必须是 private local output root 下的相对路径，不得是绝对路径，不得包含 `..`，不得指向 registry 输出文件本身；
- training、checkpoint、onnx、calibration 四类路径在同一 q mode 内、跨 q mode、跨 width 采样验证时不得碰撞；
- recipe 中不得包含 private source path、真实 GPU、资产/候选 ID、checkpoint 实例、ONNX 实例、calibration 实例、metrics、objective、cache、terminal status 或 result context。

## 验证与失败语义

桥接器按如下顺序 fail closed：

1. schema gate：只接受 v1 显式 recipe 或 v2 互斥 recipe source；v2 字段组合错误失败；
2. privacy gate：source-map 路径、history root、private root、输出 root 必须是 absolute、无软链接、在允许 root 内，并满足 Git 忽略策略；
3. binding/template gate：private binding/template 已通过现有 `p6_history_binding_v1` 验证，target 为 Pyramid/H800/TVM，status 为 validated；
4. profile gate：profile 名称、schema 版本和 target 精确匹配；
5. role/marker gate：selected components 的 roles 与 profile exact roles 完全一致，marker basename 精确匹配；
6. argv/placeholder gate：required semantic argv 与 placeholders 完全覆盖 profile surface；未知 required placeholder、缺失 width/q_mode/output placeholder、ambient fallback 均失败；
7. capability gate：三阶段 width、FP16、INT8、training、checkpoint、ONNX、calibration 能力全部满足；profile 声明 required 的 q mode 缺失时失败；
8. render gate：recipe identity 与输出模板可渲染、相对、无碰撞、无 result context；
9. normalizer gate：派生 recipe 与 source contract/template 一致，并继续通过既有 source contract validation；
10. lifecycle gate：registry-v2 identity 必须与 Stage2 dynamic plan 完全相等，且 zero measurement。

新的稳定失败类别为 `history_recipe_derivation_invalid`。它覆盖 v2 程序式 recipe 派生阶段的所有失败，包括 unknown profile、profile drift、component role drift、marker drift、argv drift、placeholder drift、precision/capability 缺失、模板碰撞、互斥字段错误、private path 越界、软链接和公开泄露风险。

失败语义：

- 失败时不写 partial recipe、不写 source map 更新、不写 normalized root、不写 YAML、不写 registry；
- 如果派生阶段在 normalizer 前失败，CLI 只输出 `history_recipe_derivation_invalid`；
- 如果派生成功但后续 normalizer/provision/Stage1/Stage2/registry gate 失败，沿用对应阶段稳定失败类别，但不得进入 measurement；
- 任何失败不得启动历史 Stage5 程序、GPU admission、训练、导出、TVM 编译、AP、时延、能耗或 feedback；
- 详细私有诊断只能写入 Git 忽略输出根，公开 stderr/stdout 仅暴露稳定类别。

## 隐私与公开边界

公开 tracked 文件允许包含：

- profile 名称、profile schema version、组件 role 名、marker basename、semantic argv 名、placeholder 名、capability 名；
- recipe schema 名、identity 模板规则、相对模板规则、失败类别、测试 fixture 中的合成值；
- zero-measurement 状态和预检通过/失败类别。

公开 tracked 文件不得包含：

- 私有绝对路径、用户目录、远端路径、真实 GPU index/UUID、主机名；
- 训练数据、模型资产、source asset ID、candidate ID、checkpoint 文件名、ONNX 文件名、calibration 数据名；
- AP、latency、energy、objective、cost-model result、历史结果、terminal status 或日志片段；
- private source-map、binding、local YAML、Stage1 manifest、Stage2 plan、registry-v2、request、feedback 或 normalized root。

private source-map v2、派生 recipe、normalized root、local YAML 和 registry-v2 只允许写在 Git 忽略输出根。bridge CLI 读写路径前必须验证 symlink、root containment 和 ignore policy。公开 recipe 不承载 private path；private recipe 若需要指向物化输出，也只能使用本次 private output root 下相对模板，经 registry materializer 渲染为 ignored local absolute path。

## CLI 与生命周期

新增或扩展 private-only CLI 时，接口应保持 argv-only、无 shell 拼接、无环境隐式 fallback。参数集合固定为：

```text
derive_p6_history_recipe
  --source-map: absolute ignored private source-map v2 path
  --binding: absolute ignored p6_history_binding_v1 path
  --profile: registered adapter profile name
  --recipe-json: absolute ignored output recipe path
```

实现可以把 recipe derivation 内联到 normalizer CLI，也可以作为 normalizer 前置命令；无论采用哪种集成，生命周期必须保持：

```text
derive recipe
  -> normalize history root
  -> provision local config
  -> Stage1 scan
  -> Stage2 dynamic plan
  -> registry-v2 materialization
  -> stop before measurement
```

H800 retry 只允许发生在上述离线 gates 全部通过之后。retry 的对象是私有 H800 预检或后续真实四轮运行，不是 recipe 派生失败；recipe/profile/normalizer 失败必须先修 profile、source-map 或 binding，再重新从零执行预检。

## TDD 验证矩阵

| 覆盖面 | 必测场景 | 期望 |
| --- | --- | --- |
| known profile accepted | 合成 binding/template、selected components 与 `p6_stage5_pyramid_h800_tvm_profile_v1` 完全匹配 | 派生 `p6_history_dynamic_materialization_recipe_v1`，不运行历史程序 |
| unknown profile | profile 名称不存在或 schema version 不支持 | `history_recipe_derivation_invalid`，无写入 |
| profile drift | marker basename、required semantic argv 或 placeholder surface 与 profile 不一致 | `history_recipe_derivation_invalid` |
| missing capability | training/checkpoint/ONNX/calibration 任一 required capability 缺失 | `history_recipe_derivation_invalid` |
| precision capability | profile 要求 FP16/INT8，但组件只声明其中一种或声明 mixed precision | `history_recipe_derivation_invalid` |
| role exactness | 缺少 role、重复 role、额外 role、role 名漂移 | `history_recipe_derivation_invalid` |
| three-stage width | 少于/多于三阶段 width、字段名不是 `stage1_width/stage2_width/stage3_width` | `history_recipe_derivation_invalid` |
| symlink/out-of-root | source-map、binding、component path 或 output root 经软链接/越界 | `history_recipe_derivation_invalid` 或现有 private path gate 失败；无 partial 输出 |
| output collisions | training/checkpoint/ONNX/calibration 模板在 q mode 或 width 采样下碰撞，或碰撞 registry 输出 | `history_recipe_derivation_invalid` |
| mutual exclusivity | v2 同时提供 explicit recipe 和 procedural source，或二者都缺失 | `history_recipe_derivation_invalid` |
| v1 compatibility | 旧 v1 explicit `dynamic_materialization_recipe` 输入 | 继续通过既有 normalizer；不要求 profile |
| v2 explicit compatibility | v2 `recipe_mode=explicit_dynamic_recipe` | 行为等价于 v1 recipe 路径，但受 v2 互斥规则约束 |
| source contract consistency | 派生 recipe 与 source contract/template 中的 recipe 不一致 | fail closed，不写 normalized root |
| no new hash system | 只检查既有 source contract hash/identity gate，不要求资产内容 hash 清单 | 测试证明没有新增资产 SHA 要求 |
| zero measurement | recipe→normalizer→provision→Stage1→dynamic Stage2→registry-v2 | 成功停在 registry-v2 identity gate 后；Stage5 measurement adapter 未调用 |
| H800 retry boundary | 离线 gates 未过时请求 H800 retry | 拒绝 retry；只有离线 gates 全绿后才允许进入私有 H800 预检 |

所有 RED 测试先断言稳定失败类别和无输出，再实现最小逻辑。GREEN 后增加 disclosure guard，扫描新公开文件和测试 fixture，确认不存在私有路径、真实 GPU、资产/候选 ID、checkpoint、ONNX、calibration 实例、metrics 或 result context。

## 部署与回滚

部署顺序：

1. 添加 profile registry 与 interface verifier 的纯单元测试，先覆盖 unknown/drift/missing capability/role/precision；
2. 实现 recipe renderer，并以合成程序式组件验证 canonical group identity、artifact identity 和相对输出模板；
3. 扩展 source-map v2 parser，保持 v1 显式 recipe 回归；
4. 集成 normalizer CLI 或新增 derivation CLI，确保失败不写 partial 文件；
5. 串接 offline preflight：recipe→normalizer→provision→Stage1→dynamic Stage2→registry-v2，并验证 zero measurement；
6. 在 disclosure、lint、unit/integration 和 git diff check 全绿后，才允许申请私有 H800 preflight retry。

回滚策略：

- 若 profile bridge 失败，关闭 v2 `recipe_mode=procedural_profile` 入口，继续保留 v1 explicit recipe 行为；
- 若 source-map v2 parser 引入回归，恢复 normalizer 对 v1 的默认路径，并拒绝 v2 输入；
- 若 renderer 产生 identity 或模板漂移，删除派生输出并重新运行 derivation；不得手改 registry 或 normalized root；
- 若离线 preflight 失败，保留 ignored diagnostics，公开树不提交私有产物，真实四轮不启动。

回滚不需要迁移历史结果，因为本设计不产生公开结果、不修改生产 Stage5 代码、不运行 measurement。

## Acceptance criteria

本规格对应实现完成的条件为：

1. `p6_history_normalization_source_v2` 支持 mutually exclusive explicit recipe 与 procedural profile 两路；v1 explicit recipe 兼容测试仍通过。
2. 至少一个已批准 Stage5 profile 可接受已验证 private binding/template 和 selected procedural components，并派生 `p6_history_dynamic_materialization_recipe_v1`。
3. unknown profile、profile drift、role drift、marker drift、semantic argv/placeholder drift、missing capability、precision drift、symlink/out-of-root、template collision、互斥字段错误均以 `history_recipe_derivation_invalid` fail closed。
4. 派生 recipe 使用现有 Pyramid canonical group identity，artifact identity 只来自三阶段 width，per-q-mode 输出模板只为相对路径；不包含 private path、真实 GPU、资产/候选 ID、checkpoint、ONNX、calibration、metrics 或 result context。
5. 不新增资产级内容 hash 体系；仅复用既有 source contract hash、Pyramid identity validation、Stage2 plan/registry identity gate 和现有契约校验。
6. 任一失败不写 partial recipe、source-map 更新、normalized root、YAML 或 registry，并且不进入 measurement。
7. 远端/私有预检链路 recipe→normalizer→provision→Stage1→dynamic Stage2→registry-v2 成功，registry identity 与 Stage2 dynamic plan 完全一致，且 Stage5 measurement 调用次数为零。
8. 真实 H800 四轮不属于本阶段完成条件；它只能在本规格的 offline gates 全绿后作为后续单独阶段申请执行。
