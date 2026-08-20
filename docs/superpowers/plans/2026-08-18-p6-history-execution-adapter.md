# P6.3 历史 CoptV2X 执行适配层实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 P6.2 的动态 Stage2 Pyramid/H800/TVM 候选池通过 Git 忽略的本地绑定，调用现有 CoptV2X Stage5 真实训练、导出、TVM、时延、能耗、AP 与反馈链，完成四轮每轮四候选的闭环；不改变 P6 的 Gold176 cold start、cost model、`predicted_frontier_diversity` 或候选集合。

**Architecture:** 新增一个私有历史绑定发现器，把唯一可验证的历史实验根转换为忽略的 `p6_history_binding_v1` 与现有 controller 可读取的 `p6_h800_coptv2x_local_v2` 配置。plan-to-registry 适配器将每个动态 P6 `(width, q_mode, source_point_ids)` 物化为完整的 `stage5_candidate_source_registry_v2` source contract；measurement 适配器将 P6 已生成的 `stage5_measurement_request_v2` 交给受约束的历史 Stage5 runner，再把经 receipt/barrier 验证的实际反馈转换为 P6 feedback。P6 controller 仍只执行 argv-only 的两个 local step。

**Tech Stack:** Python 3.10+, pytest, PyYAML, `framework.stage1_bridge.load_stage2_search_space`, `framework.stage5.production_search_v1`, `framework.stage5.single_target_search_v2`, `framework.stage6.coptv2x_h800_search_v2`, `tools/release/run_p6_h800_search.py`。

**Design specification:** `docs/superpowers/specs/2026-08-18-p6-history-execution-adapter-design.md`

## Global Constraints

- 固定实验协议为 Pyramid、H800、TVM、Gold176、four rounds、four selected candidates per round、16 measurements 和 `predicted_frontier_diversity`。
- framework mode 的候选严格等于 Stage2 中全部 active、buildable、H800/TVM 合法点按全局 `fp16` / `int8` 生成的组合。不得使用静态数量、FLOPs、proxy latency、Pareto、宽度范围或任何其他自动剪枝。
- 候选只有一个全局 `q_mode`，不能产生 stage-wise mixed precision。FP16-only、INT8-only 或两者并存均由 Stage2 输出决定。
- 任一动态候选无法获得完整且可验证的历史 source contract 时，整个 framework preflight 必须失败；不得删除候选、回退 P6.1 static mode 或选择其他后端。
- P6.1 static mode 的 343 structures / 686 candidates 门禁、执行语义与测试必须不变。
- 真实执行只允许 Git 忽略的私有 runner interface 选择的三张 GPU；每次必须验证 H800 型号、index-to-UUID 锁、占用语义与漂移。不可自动使用其他 GPU，也不可继承 Stage7 的设备排除策略。
- 不下载资产；不得执行 Orin、TensorRT、CPU、RTX 4090 或其他后端。
- 真实路径、资产、命令、GPU UUID、候选 ID、原始请求/反馈、日志、checkpoint、ONNX 和结果只能存在于 Git 忽略的本地边界。不得写入公开文档、测试 fixture、tracked config 或匿名归档。
- 所有外部进程使用 `subprocess.run(..., shell=False)` 与显式 argv。无 `shell=True`、无字符串命令解释、无环境或 GPU 的静默回退。
- 本计划不授权 push、merge、下载、或启动真实 H800 工作；真实运行是最后一项的单独用户授权动作。

## File Structure

- Add `framework/stage6/p6_history_binding_v1.py`: 纯数据模型、历史根发现、路径边界、Git 跟踪边界、GPU admission 和原子本地输出。
- Add `framework/stage6/p6_history_registry_v1.py`: 动态 candidate plan 到 Stage5 registry-v2 的完整 source-contract 转换与验证。
- Add `framework/stage6/p6_history_measurement_v1.py`: request、historical receipt/actual-feedback 与 P6 feedback 的 fail-closed 转换。
- Add `tools/release/provision_p6_history_local_config.py`: 本地-only binding/config generator CLI。
- Add `tools/release/build_p6_history_registry.py`: P6 `source_registry_step` argv CLI。
- Add `tools/release/measure_p6_history_batch.py`: P6 `measurement_step` argv CLI。
- Add `tests/stage6/test_p6_history_binding.py`, `tests/stage6/test_p6_history_registry.py`, and `tests/stage6/test_p6_history_measurement.py`.
- Add `tests/release/test_provision_p6_history_local_config.py` and `tests/release/test_p6_history_execution_adapters.py`.
- Modify `tests/stage6/test_coptv2x_h800_search.py` and `tests/release/test_run_p6_h800_search.py` for generated framework local-config integration, while retaining all P6.1 fixtures.
- Modify `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`, `docs/AAAI27_RELEASE_AUDIT.md`, and `tests/release/test_project_handoff.py` only after offline acceptance; they must describe P6.3 as implemented/offline-validated, not as a completed real run.

## Task 1: Define and Test the Private Historical Binding

**Files:**

- Create `framework/stage6/p6_history_binding_v1.py`
- Create `tests/stage6/test_p6_history_binding.py`

**Consumes:** a caller-supplied historical experiment root and an ignored output directory. Tests use only synthetic temporary roots and injected GPU probes.

**Produces:** canonical `p6_history_binding_v1` plus an atomic, non-public projection that can feed the local-config generator.

- [ ] **Step 1: Write RED tests for unambiguous discovery and no-leak projection**

  Create a synthetic historical root containing one each of the documented Stage5 controller marker, source materializer marker, performance/AP/finalizer markers, one valid Pyramid source-registry template, the four required local inputs, and an H800 probe result for three explicitly synthetic GPU indices.

  Add tests for a public `discover_history_binding()` API with an injected `GpuProbe` protocol. Assert that the returned mapping has:

  ```python
  assert binding["schema_version"] == "p6_history_binding_v1"
  assert binding["target"] == {"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"}
  assert binding["gpu_policy"]["indices"] == list(synthetic_gpu_indices)
  assert set(binding["gpu_policy"]["uuid_by_index"]) == {
      str(index) for index in synthetic_gpu_indices
  }
  assert binding["source_contract_template"]["schema_version"] == "stage5_source_contract_v1"
  assert {"gold176_rows", "gold176_graph_features", "capability_profiles", "closure"} <= set(binding["local_input_paths"])
  assert "private_root" not in public_binding_projection(binding)
  assert not any(str(history_root) in value for value in _walk_strings(public_binding_projection(binding)))
  ```

  The public projection may contain only schema/target/version/status labels. It must contain no absolute path, command, UUID, candidate ID, secret-like value, or registry content.

- [ ] **Step 2: Write RED rejection and atomic-write tests**

  Add independent tests asserting stable `P6HistoryBindingError` categories for:

  1. zero or multiple matching component markers;
  2. required inputs absent, incompatible target/backend/version, or a template that is not a complete Pyramid source contract;
  3. any discovered path resolving outside `history_root`;
  4. GPU policy other than exactly three distinct non-negative indices in canonical ascending order, non-H800 model, duplicate/missing UUID, incompatible occupancy, or a second probe with UUID drift;
  5. an attempted output write where the destination is inside the repository but not `git check-ignore`d, or where an existing target is a symlink.

  Pre-populate both output destinations, trigger an error on the second file, and assert neither prior target was overwritten. Then assert a success writes both JSON files through temp siblings and `os.replace`, without a half-written pair.

- [ ] **Step 3: Run the binding RED tests**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_p6_history_binding.py
  ```

  Expected: the module/imports do not exist, so the new tests fail before implementation.

- [ ] **Step 4: Implement discovery, path safety, GPU admission, and atomic persistence**

  In `framework/stage6/p6_history_binding_v1.py`:

  1. Define `P6HistoryBindingError`, a `GpuProbe` protocol returning immutable `{index, uuid, model_name, occupancy}` records, and non-mutating mapping builders.
  2. Define explicit marker names/relative discovery rules for only the documented historical Stage5 component chain. Resolve every candidate with `Path.resolve(strict=True)` and reject paths outside the resolved history root; require exactly one matching component per role.
  3. Load the historical source registry only to validate/select one complete Pyramid/H800/TVM source-contract template. Never copy metrics, candidate IDs, paths, or raw source contract into public output.
  4. Locate exactly one compatible Gold176, graph, capability, and closure input under the allowed root. Record their real paths only in binding/config output.
  5. Derive exactly three distinct non-negative indices in canonical ascending order from the private runner interface; require unique UUIDs, a normalized H800 model name, and compatible occupancy. Preserve the discovered UUID map for the measurement-time drift check.
  6. Implement `write_private_binding_pair(binding, local_config, binding_path, config_path, repo_root)` using sibling temporary files, fsync where available, and `os.replace` only after both serializations validate. Require every destination to be either outside the resolved repository root or `git check-ignore`d inside it, and require non-symlink files beneath their intended private parent.
  7. Keep `public_binding_projection()` data-only and explicitly deny keys containing path, command, argv, uuid, identifier, request, feedback, output, checkpoint, or metric semantics.

- [ ] **Step 5: Run binding GREEN and lint**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_p6_history_binding.py
  python -m ruff check framework/stage6/p6_history_binding_v1.py tests/stage6/test_p6_history_binding.py
  git diff --check
  ```

  Expected: all synthetic discovery/admission tests pass without accessing a real GPU or historical directory.

- [ ] **Step 6: Commit Task 1**

  ```bash
  git add framework/stage6/p6_history_binding_v1.py tests/stage6/test_p6_history_binding.py
  git commit -m "feat: bind P6 history execution inputs"
  ```

## Task 2: Generate the Ignored P6 Framework Local Configuration

**Files:**

- Create `tools/release/provision_p6_history_local_config.py`
- Create `tests/release/test_provision_p6_history_local_config.py`

**Consumes:** one private `p6_history_binding_v1`, the tracked P6 public contract, and a requested ignored local output root.

**Produces:** validated `p6_h800_coptv2x_local_v2` at an explicitly supplied private path, normally repo-relative `configs/local/p6_h800_search.local.yaml` which is already ignored, plus a paired binding JSON either under the ignored local output root or another explicitly private path.

- [ ] **Step 1: Write RED CLI tests**

  Use `subprocess.run()` black-box tests with a temporary synthetic history root and a temporary Git repository whose ignored config path matches the P6 local-config rule and whose binding path is either outside the repo root or under an ignored output directory. Invoke:

  ```bash
  python tools/release/provision_p6_history_local_config.py \
    --history-root <synthetic-history-root> \
    --local-output-root <ignored-output-root> \
    --binding-output <ignored-binding.json> \
    --config-output <ignored-p6-local.yaml>
  ```

  Assert success produces a config accepted by the existing `load_local_config()` and exactly these controller-facing semantics:

  ```python
  assert config["schema_version"] == "p6_h800_coptv2x_local_v2"
  assert config["candidate_source_mode"] == "framework_stage2_search_space"
  assert set(config["local_input_paths"]) == {
      "gold176_rows", "gold176_graph_features", "capability_profiles", "closure",
  }
  assert "{pyramid_candidate_plan}" in config["source_registry_step"]["argv"]
  assert "{measurement_request}" in config["measurement_step"]["argv"]
  assert "{feedback_json}" in config["measurement_step"]["argv"]
  ```

  Assert the generated argv points only to tracked adapter entry points plus placeholder tokens. The local config is private and may contain the binding path plus tracked adapter module path, but it must not bake the historical controller command or raw history-runner argv into a public-facing file.

- [ ] **Step 2: Write RED negative/projection tests**

  Assert the CLI returns only stable category text and writes nothing for an ambiguous root, invalid public contract, non-private config/binding location, repository-internal output root that is not ignored, or a binding whose GPU policy drifted between discovery and config rendering. A repository-external output root is valid only when it is an absolute private path selected by the caller. Scan every tracked file after a successful test and assert no test path, UUID, raw history command, candidate ID, request, feedback, or metric was introduced.

- [ ] **Step 3: Run generator RED tests**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/release/test_provision_p6_history_local_config.py
  ```

  Expected: the command is absent and tests fail before implementation.

- [ ] **Step 4: Implement the provisioning CLI**

  In `tools/release/provision_p6_history_local_config.py`:

  1. Parse paths with `argparse`; require absolute `--history-root`, `--local-output-root`, `--binding-output`, and `--config-output`. Do not accept arbitrary command/argv or GPU override arguments.
  2. Call Task 1 discovery and render a new local config with the existing exact schema/key set. Populate only private `asset_paths`, the four `local_input_paths` (`gold176_rows`, `gold176_graph_features`, `capability_profiles`, `closure`), the discovered Stage2 path, and the two tracked adapter argv templates.
  3. Construct the two adapter argv lists as direct Python/module invocations with binding/config paths supplied at runtime; preserve controller placeholder tokens as full argv elements. No shell, `bash -c`, `$VAR`, relative binary, or unapproved template token is allowed.
  4. Reuse Task 1 paired atomic write so a discovered binding and controller config cannot disagree. Print only a stable success category and output-file basenames; do not print private values.
  5. Make `main(argv: Sequence[str] | None = None) -> int` testable and map expected domain errors to stable nonzero exits.

- [ ] **Step 5: Run generator GREEN and compatibility tests**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/release/test_provision_p6_history_local_config.py tests/release/test_run_p6_h800_search.py -k 'local_config or provision'
  python -m ruff check tools/release/provision_p6_history_local_config.py tests/release/test_provision_p6_history_local_config.py
  git diff --check
  ```

  Expected: provisioned framework config is accepted by the existing controller loader; static config tests are unchanged.

- [ ] **Step 6: Commit Task 2**

  ```bash
  git add tools/release/provision_p6_history_local_config.py tests/release/test_provision_p6_history_local_config.py
  git commit -m "feat: provision private P6 framework config"
  ```

## Task 3: Materialize Every Dynamic P6 Candidate into Registry-v2

**Files:**

- Create `framework/stage6/p6_history_registry_v1.py`
- Create `tools/release/build_p6_history_registry.py`
- Create `tests/stage6/test_p6_history_registry.py`
- Create `tests/release/test_p6_history_execution_adapters.py`

**Consumes:** `p6_pyramid_candidate_plan_v2`, Task 1 binding, and the validated historical Pyramid source-contract template.

**Produces:** local `stage5_candidate_source_registry_v2`, with one full source contract for every plan candidate and exact `(width, q_mode) -> source_point_ids` provenance.

- [ ] **Step 1: Write RED exact-identity and no-pruning tests**

  Build synthetic dynamic plans that are deliberately not 343/686: FP16-only, INT8-only, and a union of both. Use a complete template binding and assert:

  ```python
  registry = materialize_history_registry(plan, binding, local_registry_root)

  assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
  assert _registry_identity_map(registry) == _plan_identity_map(plan)
  assert sum(len(group["available_q_modes"]) for group in registry["groups"]) == plan["candidate_count"]
  assert all("source_contract" in group for group in registry["groups"])
  ```

  Add tests where exactly one plan identity lacks a build/export/calibration/template mapping, source-contract canonical validation fails, the plan repeats an identity, or template substitution resolves outside the binding output root. In each case assert `P6HistoryRegistryError("source_registry_invalid", ...)`, an empty/nonexistent registry output, and no reduced candidate set.

- [ ] **Step 2: Write RED CLI contract tests**

  Through `subprocess.run()`, call `tools/release/build_p6_history_registry.py` with the exact P6 source-step placeholder shape:

  ```bash
  python tools/release/build_p6_history_registry.py \
    --binding <ignored-binding.json> \
    --pyramid-candidate-plan <plan.json> \
    --source-registry-json <registry.json> \
    --local-output-root <ignored-root>
  ```

  Assert outputs contain no metrics/objectives/status/cache/result fields and that a valid registry can be passed to `build_task_candidate_manifest()` without changing plan identities. Assert malformed argument paths and private binding failures return category-only stderr.

- [ ] **Step 3: Run registry RED tests**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_p6_history_registry.py tests/release/test_p6_history_execution_adapters.py -k registry
  ```

  Expected: imports/CLI are absent and tests fail.

- [ ] **Step 4: Implement deterministic plan-to-registry materialization**

  In `framework/stage6/p6_history_registry_v1.py`:

  1. Parse and validate the canonical v2 plan with `build_pyramid_candidate_plan`-compatible fields. Reject anything other than Pyramid/H800/TVM and global `fp16`/`int8` q modes.
  2. For each plan candidate, derive a source group from the private, complete historical template. Bind its three width values, q mode, exactly three source point IDs, training/checkpoint/ONNX/calibration output locations, and existing source evidence semantics only under the private root.
  3. Recompute the source-contract canonical hash with the existing Stage5 convention after every derived field is fixed. Do not preserve a hash from a differently shaped template.
  4. Group only identities with the same structural width; write sorted `available_q_modes` and `source_point_ids_by_q_mode`. Reject collisions rather than overwriting.
  5. Compare the completed registry identity map to the plan before persistence. Require every group source contract to pass the same validation used by the Stage5 materializer; never emit metrics or terminal values.
  6. Atomically write the registry within the ignored local output root and return the immutable mapping.

  In `tools/release/build_p6_history_registry.py`, parse only four direct file/directory arguments, call the materializer, and emit no private details.

- [ ] **Step 5: Run registry GREEN, Stage5 compatibility, and lint**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_p6_history_registry.py tests/release/test_p6_history_execution_adapters.py -k registry tests/stage5/test_production_search.py tests/stage5/test_single_target_search.py
  python -m ruff check framework/stage6/p6_history_registry_v1.py tools/release/build_p6_history_registry.py tests/stage6/test_p6_history_registry.py tests/release/test_p6_history_execution_adapters.py
  git diff --check
  ```

  Expected: every valid dynamic plan identity reaches the Stage5 manifest unchanged; one missing binding rejects the complete run before measurement.

- [ ] **Step 6: Commit Task 3**

  ```bash
  git add framework/stage6/p6_history_registry_v1.py tools/release/build_p6_history_registry.py tests/stage6/test_p6_history_registry.py tests/release/test_p6_history_execution_adapters.py
  git commit -m "feat: materialize dynamic P6 source registry"
  ```

## Task 4: Adapt Verified Historical Measurements into P6 Feedback

**Files:**

- Create `framework/stage6/p6_history_measurement_v1.py`
- Create `tools/release/measure_p6_history_batch.py`
- Create `tests/stage6/test_p6_history_measurement.py`
- Modify `tests/release/test_p6_history_execution_adapters.py`

**Consumes:** the P6 controller's existing `stage5_measurement_request_v2`, Task 1 private binding, and an actual Stage5 runner receipt/feedback barrier.

**Produces:** one atomic `p6_h800_coptv2x_feedback_v2` for exactly the same four request rows.

- [ ] **Step 1: Write RED request and successful-feedback tests**

  Implement an injected fake historical runner that records argv/environment and writes a synthetic valid Stage5 actual-feedback barrier. Assert `run_history_measurement_batch()`:

  ```python
  feedback = run_history_measurement_batch(request, binding, round_output_root, runner)

  assert feedback["schema_version"] == "p6_h800_coptv2x_feedback_v2"
  assert feedback["measurement_request_sha256"] == request["measurement_request_sha256"]
  assert {row["row_id"] for row in feedback["rows"]} == {row["row_id"] for row in request["rows"]}
  assert len(feedback["rows"]) == 4
  assert all(_has_five_finite_metrics(row) for row in feedback["rows"])
  assert runner.calls[0].argv[0] != "sh"
  assert runner.calls[0].shell is False
  ```

  Assert the adapter executes the documented source-materialization → quantization → performance → AP → finalization sequence through explicit runner arguments and creates no tracked artifact.

- [ ] **Step 2: Write RED fail-closed and GPU-drift tests**

  Add parameterized failure cases for request schema/sha mismatch, not exactly four unique rows, wrong target/backend/q mode, altered row SHA/source evidence, Stage5 runner nonzero exit, missing/duplicate/extra actual-feedback row, incomplete/non-finite five metrics, invalid failure status, receipt/barrier mismatch, result request SHA mismatch, and private-policy GPU UUID/model/occupancy drift before or after the runner.

  Assert none write P6 feedback and all return a stable P6 execution category without exposing private command/path/receipt content.

- [ ] **Step 3: Run measurement RED tests**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_p6_history_measurement.py tests/release/test_p6_history_execution_adapters.py -k measurement
  ```

  Expected: imports/CLI are absent, so the new tests fail before implementation.

- [ ] **Step 4: Implement the fail-closed adapter and CLI**

  In `framework/stage6/p6_history_measurement_v1.py`:

  1. Reuse/validate the current Stage5 request schema; require four unique rows and exact H800/Pyramid/TVM/global-q/budget fields before issuing any process.
  2. Re-probe the binding's private three-device policy, requiring H800 model, UUID equality, and compatible occupancy both before and after historical execution.
  3. Construct only the documented Stage5 chain argv from binding-controlled component paths and per-round private files. Use an injected `Runner` protocol in unit tests and `subprocess.run(shell=False, check=False, ...)` in production. A component failure is not success.
  4. Validate Stage5 actual receipt, source evidence, canonical request SHA, exact row set, and atomic finalization barrier. Translate only valid actual results into the existing P6 feedback schema; preserve no raw receipt or path in P6/public output.
  5. Write P6 feedback atomically only after full validation. Map all expected errors to stable categories and leave detailed material only beneath the ignored round output root.

  In `tools/release/measure_p6_history_batch.py`, require `--binding`, `--measurement-request`, `--feedback-json`, and `--round-output-root`; reject all nonignored output paths and unknown arguments.

- [ ] **Step 5: Run measurement GREEN and controller regression**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_p6_history_measurement.py tests/release/test_p6_history_execution_adapters.py -k measurement tests/stage6/test_coptv2x_h800_search.py
  python -m ruff check framework/stage6/p6_history_measurement_v1.py tools/release/measure_p6_history_batch.py tests/stage6/test_p6_history_measurement.py tests/release/test_p6_history_execution_adapters.py
  git diff --check
  ```

  Expected: fake history execution produces only valid atomic P6 feedback; every identity/GPU/feedback fault stops the round.

- [ ] **Step 6: Commit Task 4**

  ```bash
  git add framework/stage6/p6_history_measurement_v1.py tools/release/measure_p6_history_batch.py tests/stage6/test_p6_history_measurement.py tests/release/test_p6_history_execution_adapters.py
  git commit -m "feat: execute P6 through history measurement chain"
  ```

## Task 5: Prove Generated-Config Four-Round Integration and Close Documentation

**Files:**

- Modify `tests/stage6/test_coptv2x_h800_search.py`
- Modify `tests/release/test_run_p6_h800_search.py`
- Modify `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`
- Modify `docs/AAAI27_RELEASE_AUDIT.md`
- Modify `tests/release/test_project_handoff.py`

**Consumes:** synthetic provisioned binding/config, dynamic Stage2 search space, registry adapter, fake Stage5 runner, and existing public release documentation.

**Produces:** an offline proof that controller → dynamic plan → registry-v2 → Stage5 request → verified feedback traverses exactly four rounds while P6.1 remains unchanged; public status remains honest that no real H800 run occurred.

- [ ] **Step 1: Write RED end-to-end tests through the public CLI**

  Add an in-process controller integration test and a `subprocess.run()` public CLI test. The dynamic Stage2 fixture must have a deliberately non-343/686 candidate count of at least 16. Provision a synthetic local config, substitute fake adapter executables, then assert:

  ```python
  state = run_p6_coptv2x_search(contract, provisioned_config, "test-revision", runner)

  assert state.status == "completed"
  assert state.completed_rounds == 4
  assert state.measured_candidate_count == 16
  assert observed_rounds == [0, 1, 2, 3]
  assert all(len(rows) == 4 for rows in observed_measurement_requests)
  assert observed_registry_identities == observed_plan_identities
  ```

  Add a regression test that P6.1 static mode still rejects a non-343/686 v1 registry. Add a framework-mode test that a candidate unable to materialize fails before any measurement adapter invocation. Assert all output files are in ignored/test-private roots and no test writes an actual path, result, candidate ID, or command into tracked documentation.

- [ ] **Step 2: Run the integration RED tests**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -k 'history or provisioned or framework'
  ```

  Expected: failures identify the absent history adapters/generator or an incomplete end-to-end boundary.

- [ ] **Step 3: Complete integration wiring with the smallest compatible changes**

  Modify existing controller tests/fixtures only as needed to invoke the exact adapter placeholder interfaces from Task 2. Do not add a controller special case for historical execution: its existing source-registry and measurement subcommands remain the sole integration boundary.

  Ensure P6 controller error normalization remains `source_registry_invalid` before measurement and a local round failure after measurement feedback faults. Preserve the current Stage2 dynamic plan, Stage5 cost-model cold-start/retrain loop, four×four task budget, static-mode checks, and no-public-result behavior.

- [ ] **Step 4: Run full offline quality gates**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
  PYTHONPATH=. python -m pytest -q tests/release tests/integration/test_anonymous_archive.py
  python -m ruff check framework/stage6 tools/release tests/stage6 tests/release
  git diff --check
  git status --short
  ```

  Expected: all quality/release/archive tests pass, coverage remains at least 80%, lint/diff are clean, and no generated config, binding, output, log, asset, request, feedback, or measurement result is tracked.

- [ ] **Step 5: Update P6 status documentation and its disclosure tests**

  Update `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md` and `docs/AAAI27_RELEASE_AUDIT.md` to say only: P6.3 historical execution adapters are implemented and offline-validated; the actual H800 framework-source four-round run remains pending local execution. Keep P6.1/P6.2 distinctions and P7/P8 status intact.

  Update `tests/release/test_project_handoff.py` to reject public disclosure of actual paths, commands, GPU UUIDs, candidate IDs, raw metrics/results, checkpoints, logs, requests, feedback, or claims that a real H800 run completed. Do not include synthetic fixture locations in docs.

- [ ] **Step 6: Run final documentation/disclosure checks and commit**

  Run:

  ```bash
  PYTHONPATH=. python -m pytest -q tests/release/test_project_handoff.py tests/release/test_run_p6_h800_search.py tests/stage6/test_coptv2x_h800_search.py
  python -m ruff check framework/stage6 tools/release tests/stage6 tests/release
  git diff --check
  git status --short
  ```

  Then commit:

  ```bash
  git add framework/stage6/p6_history_binding_v1.py framework/stage6/p6_history_registry_v1.py framework/stage6/p6_history_measurement_v1.py tools/release/provision_p6_history_local_config.py tools/release/build_p6_history_registry.py tools/release/measure_p6_history_batch.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_history_registry.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_provision_p6_history_local_config.py tests/release/test_p6_history_execution_adapters.py tests/release/test_run_p6_h800_search.py docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md tests/release/test_project_handoff.py
  git commit -m "feat: adapt P6 history execution chain"
  ```

## Final Independent Review and Real-Run Gate

- [ ] **Step 1: Perform an independent code review before any real execution**

  Review only the completed diff for: silent candidate removal, static-mode fallback, accidental mixed precision, non-argv subprocess calls, unbound input/assets, path traversal/symlink writes, private three-device policy bypass or UUID drift, success on incomplete feedback, and public disclosure leaks. Address every critical/high finding with a dedicated RED → GREEN test and commit before moving on.

- [ ] **Step 2: Obtain a separate, explicit real-H800 execution authorization**

  Before connecting to H800 or running the generator, request confirmation that this specific revision may: use the existing SSH control connection, inspect the selected private historical root, generate the ignored local config at `configs/local/p6_h800_search.local.yaml`, generate the binding and outputs under the ignored/private local output root, reserve only the private runner interface's selected three GPUs, and execute the full four-round H800/TVM search. Do not infer that permission from approval of this plan.

- [ ] **Step 3: Run discovery and dynamic preflight only after authorization**

  Execute the provision CLI, run the P6 CLI through framework-mode source-registry validation, and verify the Stage2 plan/registry/manifest identity set plus `>=16` eligible candidates. If any binding, contract, GPU, asset, or identity gate fails, stop before training/measurement; record only the local failure material and report the stable category.

- [ ] **Step 4: Run and supervise the four-round real loop only after preflight succeeds**

  Require the actual controller state to progress `cold_start -> select/measure/feedback` for rounds `0, 1, 2, 3`; each round must select and receive exactly four distinct rows, use the actual Stage5 chain, atomically feedback, then retrain/update the cost model before the next selection. Stop on any drift/error; do not manually patch a state to advance.

- [ ] **Step 5: Keep real outputs private and update status only from verified evidence**

  After a successful loop, verify all products remain ignored and no tracked diff contains private details. A separate evidence review will decide whether the run qualifies for a P6 completion note; no numerical result or completion claim enters public docs unless that review has passed and the user explicitly authorizes the documentation change.
