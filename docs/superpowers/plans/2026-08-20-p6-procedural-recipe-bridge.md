# P6 程序式 Recipe Bridge Implementation Plan

> **执行方式：** 按用户已选择的子代理驱动模式执行；每项任务独立 RED→GREEN→REFACTOR，再做规格复核和代码质量复核，通过后才进入下一项。

**Goal:** 将已确认的 CoptV2X/Pyramid 历史 source materializer 静态桥接为共享 source-bundle recipe v2，并在 zero-GPU 条件下证明 Stage2→registry-v2→request→projection→source invocation shape 完整正确。

**Spec:** `docs/superpowers/specs/2026-08-20-p6-procedural-recipe-bridge-design.md`

## 全局合同

- 同一 width/group 的训练、checkpoint、ONNX、calibration 是一份共享 source bundle；FP16/INT8 仅在 quantization、performance、AP 后续分流。
- 公开 component 证据仅允许四个 marker：controller、source_materializer、performance_plan、finalizer；不得新增 training/export/calibration marker。
- execution stages 必须且仅为 `source_materialization/quantization/performance/ap/finalization`，顺序固定。
- recipe v1 继续可读；本轮 procedural profile 只派生 `p6_history_dynamic_materialization_recipe_v2`。
- 静态 base checkpoint/config/training 参数来自 ignored source-contract template；公开 profile 只保存字段形状和相对模板。
- 候选数量由 Stage2 动态输出决定；不得断言 343/686 或另加自动剪枝。
- 失败分类：派生/recipe 为 `history_recipe_derivation_invalid`；request projection/source invocation contract 为 `history_execution_invalid`。
- 只重算已有 source-contract/row/request canonical hashes；不新增 asset hash ledger、Merkle tree 或逐产物 SHA。
- Task 1–5 禁止 SSH、GPU、训练、导出、量化、TVM 调优、AP、时延和能耗测量。Task 6 也只做 zero-measurement preflight。
- tracked 文件不得包含 private path、真实 GPU、资产/候选 ID、base checkpoint 值、训练参数、产物实例、指标、结果或日志。

## 文件职责

- `p6_runner_template_validator_v1.py`：共享 pre-provision runner-template validator。
- `p6_history_recipe_profiles_v1.py`：公开版本化 profile，只含四 marker、五 stage、recipe-v2 字段/模板形状和 source invocation flags。
- `p6_history_recipe_bridge_v1.py`：source-map procedural profile 验证与 recipe-v2 纯渲染。
- `p6_history_normalization_v1.py`：v1/v2 source-map 互斥解析与 canonical recipe 传递。
- `p6_history_registry_v1.py`：v1 compatibility + v2 每 group 共享 source bundle 渲染。
- `p6_history_source_materialization_v1.py`：immutable request projection、existing hash 重算、按 group invocation plan 和 direct-argv runner boundary。
- `coptv2x_h800_search_v2.py`：在 round request 原子写入前调用纯 projection，使 projected request 成为 feedback 校验的 canonical request。
- `p6_history_measurement_v1.py`：调用 source-materialization adapter，然后保持 quantization/performance/AP/finalizer wrappers 不变；不执行 controller。

---

### Task 1: 抽取共享 pre-provision runner-template validator

**Files**

- Create: `framework/stage6/p6_runner_template_validator_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Test: `tests/stage6/test_p6_runner_template_validator.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces**

```python
@dataclass(frozen=True)
class ValidatedRunnerTemplate:
    history_root: Path
    component_paths: Mapping[str, Path]
    execution_interface: Mapping[str, Any]
    stage_argv: Mapping[str, tuple[str, ...]]

def validate_pre_provision_runner_template(
    runner_template_path: Path,
    history_root: Path,
) -> ValidatedRunnerTemplate: ...
```

validator 必须证明 exactly 四个 role/basename 和 exactly 五个 stage；quantization/AP 只验证唯一 stage argv0 位于 private Git root，不声明公开 marker。返回值深拷贝或 immutable wrapper，调用方不得修改 template 原对象。

- [ ] RED：为合法模板、缺/多/重复 stage、四 marker 漂移、component 越 root、symlink、非 executable、非 ignored template 写测试。
- [ ] RED command：`PYTHONPATH=. pytest tests/stage6/test_p6_runner_template_validator.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py -q`
- [ ] GREEN：把 bootstrap 现有 template/path/argv 校验移动到共享模块；bootstrap 只做错误类别映射和 binding/provision。
- [ ] REFACTOR：确认 bootstrap 公共输出 shape 不变，validator 不 import recipe/normalizer/registry/binding creation。
- [ ] Verify：上述 pytest + `python -m ruff check framework/stage6/p6_runner_template_validator_v1.py framework/stage6/p6_full_chain_bootstrap_v1.py tests/stage6/test_p6_runner_template_validator.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py && git diff --check`
- [ ] Commit：`refactor: share P6 runner template validation`

---

### Task 2: 版本化 profile 与共享 source recipe v2 renderer

**Files**

- Create: `framework/stage6/p6_history_recipe_profiles_v1.py`
- Create: `framework/stage6/p6_history_recipe_bridge_v1.py`
- Test: `tests/stage6/test_p6_history_recipe_bridge.py`

**Interfaces**

```python
RECIPE_V2 = "p6_history_dynamic_materialization_recipe_v2"
PROFILE_V1 = "p6_stage5_pyramid_h800_tvm_profile_v1"
SHARED_SOURCE_PATH_KEYS = (
    "checkpoint_path", "checkpoint_dir", "config_path",
    "training_done_marker", "onnx_path", "onnx_report_path",
    "calibration_root", "calibration_npz", "calibration_summary",
    "trt_calibration_dir", "source_done_marker",
)

def derive_dynamic_recipe_from_procedural_source(
    *, source_map: Mapping[str, Any], runner_template_path: Path,
    history_root: Path,
) -> dict[str, Any]: ...
```

profile exact 内容：四个已存在 marker、五个 stage、三 width fields、FP16/INT8 search surface、11 shared keys，以及 invocation flags `--request/--model/--group-id/--gpu`。profile 不含 training/checkpoint_export/onnx_export/int8_calibration component，不含 private 静态值。

recipe-v2 输出只含 schema、width fields、canonical group/artifact templates 和 `shared_source_path_templates`。所有模板不含 `{q_mode}`；同一 group 只渲染一份路径，跨 group 采样不得碰撞。

- [ ] RED：known profile 派生成功，且历史 runner 调用数为 0。
- [ ] RED：unknown profile、四 marker 漂移、五 stage 漂移、11 key 缺失/额外、absolute/`..`、`{q_mode}`、同 group/跨 group collision 均为 `history_recipe_derivation_invalid`。
- [ ] RED：断言公开 profile 序列化内容不出现虚构 component role 或 private/static values。
- [ ] RED command：`PYTHONPATH=. pytest tests/stage6/test_p6_history_recipe_bridge.py -q`
- [ ] GREEN：实现 immutable profile registry、validated template 对比和 pure renderer；无脚本扫描/执行，无 source-map capability 自证。
- [ ] Verify：pytest + `python -m ruff check framework/stage6/p6_history_recipe_profiles_v1.py framework/stage6/p6_history_recipe_bridge_v1.py tests/stage6/test_p6_history_recipe_bridge.py && git diff --check`
- [ ] Commit：`feat: derive shared P6 source recipes`

---

### Task 3: Private derive CLI、source-map v2 normalizer 与 post-provision equality

**Files**

- Create: `tools/release/derive_p6_history_recipe.py`
- Modify: `framework/stage6/p6_history_normalization_v1.py`
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `tools/release/normalize_p6_history_root.py`
- Modify: `tools/release/provision_p6_full_chain_local_config.py`
- Test: `tests/release/test_derive_p6_history_recipe.py`
- Test: `tests/stage6/test_p6_history_normalization.py`
- Test: `tests/release/test_normalize_p6_history_root.py`
- Test: `tests/stage6/test_p6_history_binding.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces**

```python
def normalize_history_inputs(
    source_map: Mapping[str, Any], history_root: Path, private_dir: Path,
    *, runner_template_path: Path | None = None,
) -> Mapping[str, Path]: ...

def validate_binding_recipe_consistency(
    binding: Mapping[str, Any], expected_recipe: Mapping[str, Any] | None,
) -> None: ...
```

v1 explicit recipe 行为不变。v2 explicit 支持 recipe v1/v2；v2 procedural 必须有 runner template 并派生 recipe v2。两路互斥，source-map 不自报 argv/capability。derived recipe 只写 ignored output，全部验证通过后 atomic replace。provision 在写 binding/config pair 前做 canonical equality；不读取未来 binding 来派生 recipe。

- [ ] RED：CLI success/stdout、失败 stderr redaction、no-partial output、absolute ignored path gates。
- [ ] RED：v1 regression、v2 explicit v1/v2、v2 procedural、互斥错误、缺 runner template、binding recipe drift。
- [ ] RED command：`PYTHONPATH=. pytest tests/release/test_derive_p6_history_recipe.py tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py -q`
- [ ] GREEN：CLI direct argv-only；normalizer 只在 procedural v2 调 bridge；binding equality 发生在 pair 写入前。
- [ ] Verify：上述 pytest + 相关文件 ruff + `git diff --check`。
- [ ] Commit：`feat: normalize P6 procedural source recipes`

---

### Task 4: Registry-v2 共享 bundle 与 immutable request projection

**Files**

- Modify: `framework/stage6/p6_history_registry_v1.py`
- Create: `framework/stage6/p6_history_source_materialization_v1.py`
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Test: `tests/stage6/test_p6_history_registry.py`
- Test: `tests/stage6/test_p6_history_source_materialization.py`
- Test: `tests/stage6/test_coptv2x_h800_search.py`

**Interfaces**

```python
@dataclass(frozen=True)
class ProjectedSourceRequest:
    request: Mapping[str, Any]
    ordered_group_ids: tuple[str, ...]

def project_source_materialization_request(
    request: Mapping[str, Any],
) -> ProjectedSourceRequest: ...
```

registry 的 recipe-v1 路径保持兼容；recipe-v2 对每个 group 渲染 exactly 11 个 `shared_source_paths`，把 ignored template 中的静态 source 参数原样深拷贝到 group contract，并禁止 q_mode 进入 shared paths。

projection 输入是 `build_measurement_request()` 的完整 request。每行把 nested shared bundle 与静态字段组合为 flat `source_contract`；同组 q-mode rows 的 11 path 字段必须一致；跨组路径不得碰撞。它返回新对象，原 request/row/contract 不变，并重算已有 `source_contract_sha256`、`row_sha256`、`measurement_request_sha256`。不新增 hash 字段。

`coptv2x_h800_search_v2._run_one_round` 在 `_write_json(request_path, request)` 前调用 projection；projected request 是 measurement 与 feedback validation 的唯一 canonical request，从而不存在 original/projected request hash 双真值。

- [ ] RED：recipe-v2 FP16-only、INT8-only、同组 mixed q-mode 都生成一份 shared bundle；不同 group 无碰撞。
- [ ] RED：同组两行 11 path 完全相同，q_mode/strategy/genome 仍不同；静态 private template 字段仍存在于 flat contract。
- [ ] RED：原输入未突变，三层既有 hashes 全部按 canonical JSON 更新并可复验。
- [ ] RED：path mismatch/collision、registry identity drift、missing/extra key 为 `history_execution_invalid`，无 request 写入。
- [ ] RED command：`PYTHONPATH=. pytest tests/stage6/test_p6_history_registry.py tests/stage6/test_p6_history_source_materialization.py tests/stage6/test_coptv2x_h800_search.py -q`
- [ ] GREEN：实现 recipe-version dispatch、shared render、pure projection，并将 projection 放在 round request 原子写入前。
- [ ] Verify：上述 pytest + 相关文件 ruff + `git diff --check`。
- [ ] Commit：`feat: project shared P6 source requests`

---

### Task 5: Source invocation adapter 与 zero-GPU 全链条黑盒

**Files**

- Modify: `framework/stage6/p6_history_source_materialization_v1.py`
- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Test: `tests/stage6/test_p6_history_source_materialization.py`
- Test: `tests/stage6/test_p6_history_measurement.py`
- Test: `tests/stage6/test_coptv2x_h800_search.py`
- Test: `tests/release/test_run_p6_h800_search.py`
- Test: `tests/release/test_p6_history_execution_adapters.py`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md`

**Interfaces**

```python
def build_source_invocations(
    projected_request_path: Path,
    ordered_group_ids: Sequence[str],
    *, source_materializer: Path,
    validated_gpu_policy: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]: ...

def run_source_invocations(
    invocations: Sequence[Sequence[str]], *, runner: Runner,
    cwd: Path, env: Mapping[str, str],
) -> None: ...
```

每个 distinct group exactly one argv：

```text
<source_materializer> --request <projected-request> --model pyramid --group-id <group> --gpu <validated-index>
```

GPU 分配只从 binding 已验证 policy 的三张卡选择，测试使用合成 index。argv sequence + `shell=False`，不执行 controller。`p6_history_measurement_v1` 不再按旧的两 positional args 调 source stage；它先调用 adapter，再按原顺序调用 quantization、performance、AP、finalization，wrapper argv/placeholder 不改变。

**必须新增的 zero-GPU 黑盒**

使用公开 Stage2 合成搜索空间真实生成 dynamic plan，再经过 registry-v2、Stage5 selection/request、projection 和 source invocation。injected runner 只记录 argv 并返回成功；不得 subprocess 执行历史脚本。测试必须同时证明：

1. 同组 FP16/INT8 共享 11 path 且只产生一次 source invocation；
2. 不同 group 路径不碰撞且各调用一次；
3. argv 包含完整 flags，GPU 均来自合成 validated policy；
4. projected source-contract/row/request hashes 可复验；
5. controller、真实 source materializer、quant/perf/AP/finalizer 的进程调用次数均为 0；
6. Stage2 plan/registry identity 完全相等，候选数只与动态输出相等。

- [ ] RED：dedupe/order/GPU round-robin or deterministic assignment、argv exactness、non-policy GPU、runner nonzero/exception、no-partial state。
- [ ] RED：上述 Stage2→registry-v2→request→projection→invocation-shape 黑盒。
- [ ] RED command：`PYTHONPATH=. pytest tests/stage6/test_p6_history_source_materialization.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py tests/release/test_p6_history_execution_adapters.py -q`
- [ ] GREEN：接入 measurement adapter，保留后续四个 private wrapper 行为与 feedback contract。
- [ ] Docs：`AAAI27_RELEASE_AUDIT.md` 只记录“offline recipe/projection/invocation-shape gate 完成；H800 preflight/真实四轮未完成”，不记录 private 实例或结果。
- [ ] Full gates：
  - `PYTHONPATH=. pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80`
  - `PYTHONPATH=. pytest tests/release tests/integration/test_anonymous_archive.py -q`
  - `python -m ruff check framework tools scripts tests`
  - `git diff --check`
- [ ] Review：规格复核后再做 Python 质量/隐私复核；CRITICAL/HIGH 全部修复。
- [ ] Commit：`test: gate P6 source materialization offline`

---

### Task 6: H800 private zero-measurement preflight

**Files**

- No tracked changes and no commit。
- 只使用既有 approved SSH control master、ignored source-map/runner template/source-contract、fresh private output root 和已安装 TVM activation argv。

**边界**

本任务不执行 source materializer，不开始训练、checkpoint/ONNX/calibration 生成、quantization、TVM tuning、AP、时延或能耗测量。它只把真实 private 输入验证到 projected request 与 source invocation argv shape；runner 必须在进程边界拦截。

- [ ] `git status --short` 必须为空，Task 1–5 full gates 再次全绿。
- [ ] 通过既有 SSH master 检查连接；不得新建需要口令的连接，不记录主机/GPU/private path 到 tracked 文件。
- [ ] 在 fresh ignored root 依次执行 derive→normalize→provision→Stage1→Stage2 dynamic plan→registry-v2→request→projection。
- [ ] 使用 injected/intercept runner 构造 source invocations，断言每 group 一次、argv flags 完整、GPU 来自 validated policy；拦截点不得执行 argv0。
- [ ] 断言同组 mixed q-mode 共享 source bundle、跨组无碰撞、canonical hashes 可复验、measurement/training process call count 为 0。
- [ ] 失败只在 ignored private diagnostics 记录稳定类别；不得复制路径、ID、指标、结果或日志进 Git。
- [ ] 最后 `git status --short` 仍为空；不 `git add`、不 commit。

## 最终自审清单

- [ ] 无虚构 training/checkpoint_export/onnx_export/int8_calibration marker。
- [ ] recipe v2 shared bundle 不含 q_mode；FP16/INT8 只在后续阶段分流。
- [ ] projection 位于 canonical request 写入之前，没有 original/projected hash 双真值。
- [ ] source materializer 每 group 一次，完整 direct argv，GPU 来自 private validated policy。
- [ ] controller 未执行，quant/perf/AP/finalizer contract 未改变。
- [ ] Task 5 zero-GPU 黑盒覆盖真实 Stage2→registry-v2→request→projection→invocation shape。
- [ ] Task 6 无训练、无测量、无 tracked changes。
