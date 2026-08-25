# P6 Post-Source Round Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build four narrow post-source adapters that execute the historical CoptV2X quantization, TVM performance, AP, and finalization leaves and publish the existing P6 completion set, then run a wholly fresh real four-round experiment.

**Architecture:** Keep `p6_history_runner_interface_v1`, the five-stage runner order, and the exact five-key environment. A private source-map leaf binding is normalized into one ignored profile and four generated wrappers. Tracked adapter modules orchestrate historical native files; only finalization projects the existing P6 task-state/feedback/receipt/barrier files.

**Tech Stack:** Python 3.11, stdlib subprocess/json/pathlib/hashlib/tempfile, PyYAML at normalization boundaries, pytest, pytest-cov, Ruff, existing Stage3/Stage5 historical CLIs.

**Spec:** `docs/superpowers/specs/2026-08-25-p6-post-source-round-adapters-design.md`

## Global Constraints

- Preserve Pyramid/H800/TVM, Gold176 cold start, four rounds, four rows per round, and the five metrics.
- Preserve scanner-derived candidate identities and global row-level `q_mode`; do not add search filtering or static-space fallback.
- Keep `p6_history_runner_interface_v1` and exactly the existing five private environment keys.
- Historical Stage3/Stage5 leaves remain authoritative; do not implement proxy metrics or synthetic terminal rows.
- Add no quant/performance/AP P6 receipt schemas; use native historical outputs.
- Validate only component identity, request/state identity, native stage completeness, and final completion-set publication.
- Every production change follows observed RED, minimal GREEN, then refactor.
- Changed production modules require at least 80% branch coverage.
- New or split production files stay below 800 lines; changed functions stay below 50 lines.
- Generated/private profiles, paths, GPU identities, logs, artifacts, and metrics remain Git-ignored and are never copied into tracked fixtures.
- No real SSH, GPU, Stage1, controller, training, or measurement before all implementation and review gates pass.

---

### Task 1: Private Leaf Binding and Normalized Profile

**Files:**
- Create: `framework/stage6/p6_post_source_leaf_binding_v1.py`
- Create: `framework/stage6/p6_post_source_adapter_profile_v1.py`
- Modify: `framework/stage6/p6_history_recipe_normalization_v1.py`
- Modify: `framework/stage6/p6_history_normalization_v1.py`
- Modify: `tools/release/derive_p6_history_recipe.py`
- Test: `tests/stage6/test_p6_post_source_adapter_profile.py`
- Test: `tests/stage6/test_p6_history_normalization.py`
- Test: `tests/release/test_derive_p6_history_recipe.py`

**Interfaces:**
- Consumes: `P6ValidatedExecutionClosure`, copied closure roots, the existing canonical project Python, and source-map schema v3.
- Produces: `ValidatedPostSourceAdapterProfile`, `load_post_source_adapter_profile(path, *, private_root)`, `build_post_source_adapter_profile(...)`, and normalized `post-source-adapter-profile.yaml`.

- [ ] **Step 1: Write source-map and profile RED tests**

Add tests that construct a v3 source map with this exact private leaf binding:

```python
leaf_binding = {
    "schema_version": "p6_post_source_leaf_binding_v1",
    "leaves": {
        name: {
            "closure_id": "historical-chain",
            "entrypoint_relative_path": relative_path,
        }
        for name, relative_path in {
            "quant_contract": "scripts/quant-contract.leaf.py",
            "performance_plan": "scripts/performance-plan.leaf.py",
            "performance_execute": "scripts/performance-execute.leaf.py",
            "ap_plan": "scripts/ap-plan.leaf.py",
            "ap_execute": "scripts/ap-execute.leaf.py",
            "feedback_finalize": "scripts/feedback-finalize.leaf.py",
            "feedback_promote": "scripts/feedback-promote.leaf.py",
        }.items()
    },
}
```

Assert the current code rejects schema v3 or omits the normalized profile. Add parameterized failures for one missing leaf, one extra leaf, unknown closure id, escaping relative path, non-executable leaf, and duplicate resolved leaf.

- [ ] **Step 2: Run RED tests and record exact failures**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_post_source_adapter_profile.py \
  tests/stage6/test_p6_history_normalization.py -k 'post_source or v3'
```

Expected: collection failure for the missing profile module and/or explicit assertions showing v3/profile behavior is absent.

- [ ] **Step 3: Implement the minimal immutable profile contract**

Implement the source-map boundary in `p6_post_source_leaf_binding_v1.py` and
the normalized private profile in `p6_post_source_adapter_profile_v1.py`.
Use frozen dataclasses:

```python
@dataclass(frozen=True)
class PostSourceLeaf:
    name: str
    implementation: Path
    implementation_cwd: Path
    sha256: str

@dataclass(frozen=True)
class PostSourceAdapter:
    stage: str
    implementation: Path
    implementation_cwd: Path

@dataclass(frozen=True)
class ValidatedPostSourceAdapterProfile:
    schema_version: Literal["p6_post_source_adapter_profile_v1"]
    private_root: Path
    project_python: Path
    adapters: tuple[PostSourceAdapter, ...]
    leaves: tuple[PostSourceLeaf, ...]
```

Use exact key sets, seven exact leaf names, four exact adapter stages, canonical relative paths beneath one declared closure root, copied-file SHA-256, and immutable tuples. Add `p6_history_normalization_source_v3` only for source maps containing `post_source_leaf_binding`; preserve v1/v2 behavior.

- [ ] **Step 4: Generate and reload the ignored profile**

After copying the existing closure roots, derive adapter implementations from the four existing closure roles and leaf implementations from the v3 binding. Write `post-source-adapter-profile.yaml` atomically, add it to `.git/info/exclude`, reload it through `load_post_source_adapter_profile`, and assert byte/path/digest equivalence before publishing the normalized root.

- [ ] **Step 5: Verify Task 1 GREEN and coverage**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_post_source_adapter_profile.py \
  tests/stage6/test_p6_history_normalization.py \
  tests/release/test_derive_p6_history_recipe.py
PYTHONPATH=. pytest -q \
  --cov=framework.stage6.p6_post_source_leaf_binding_v1 \
  --cov=framework.stage6.p6_post_source_adapter_profile_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_post_source_adapter_profile.py
python -m ruff check framework/stage6/p6_post_source_leaf_binding_v1.py \
  framework/stage6/p6_post_source_adapter_profile_v1.py \
  framework/stage6/p6_history_recipe_normalization_v1.py \
  framework/stage6/p6_history_normalization_v1.py \
  tools/release/derive_p6_history_recipe.py \
  tests/stage6/test_p6_post_source_adapter_profile.py
```

- [ ] **Step 6: Commit Task 1**

```bash
git add framework/stage6/p6_post_source_leaf_binding_v1.py \
  framework/stage6/p6_post_source_adapter_profile_v1.py \
  framework/stage6/p6_history_recipe_normalization_v1.py \
  framework/stage6/p6_history_normalization_v1.py \
  tools/release/derive_p6_history_recipe.py \
  tests/stage6/test_p6_post_source_adapter_profile.py \
  tests/stage6/test_p6_history_normalization.py \
  tests/release/test_derive_p6_history_recipe.py
git commit -m "feat: bind P6 post-source leaf roles"
```

---

### Task 2: Generated Adapter Wrappers and Preflight Routing

**Files:**
- Create: `framework/stage6/p6_post_source_wrapper_template_v1.py`
- Modify: `framework/stage6/p6_history_execution_closure_v1.py`
- Modify: `framework/stage6/p6_history_normalization_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `tools/release/provision_p6_full_chain_local_config.py`
- Modify: `tools/release/preflight_p6_materializer_training_bridge.py`
- Test: `tests/stage6/test_p6_post_source_wrapper.py`
- Test: `tests/stage6/test_p6_history_execution_closure.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`
- Test: `tests/release/test_preflight_p6_materializer_training_bridge.py`

**Interfaces:**
- Consumes: Task 1 `ValidatedPostSourceAdapterProfile`.
- Produces: `render_post_source_adapter_wrappers(profile, *, private_root) -> Mapping[str, Path]` and a normalized v1 runner whose four downstream stages point to exact generated wrappers.

- [ ] **Step 1: Write wrapper execution RED tests**

For each stage, assert expected wrapper bytes and an execution-level fake adapter sees:

```python
{
    "argv": original_stage_argv,
    "incoming_environment_keys": sorted(EXPECTED_HISTORY_ENV_KEYS),
    "project_python": str(profile.project_python),
    "profile": str(profile_path),
    "cwd": str(adapter.implementation_cwd),
}
```

Add failures for an extra incoming env key, interpreter drift, adapter implementation drift, leaf digest drift, and child nonzero propagation. Assert performance/finalization wrapper basenames are exactly the existing component markers.

- [ ] **Step 2: Run wrapper RED**

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_post_source_wrapper.py
```

Expected: missing module/function and runner still points to copied raw roles.

- [ ] **Step 3: Implement one shared deterministic wrapper template**

Render four scripts from one template. Each generated script uses `/usr/bin/python3`, requires the exact five incoming keys, and calls:

```python
child_argv = (
    str(project_python),
    str(adapter_implementation),
    "--profile",
    str(profile_path),
    *sys.argv[1:],
)
```

Its child env is the five original values plus deterministic `PATH` and `PYTHONPATH`; it uses `subprocess.run(..., shell=False)` and returns the child code.

- [ ] **Step 4: Route the existing v1 runner and bootstrap**

Extend `render_normalized_runner_template` with an optional exact four-stage wrapper mapping. Preserve the existing source wrapper. Point quant/AP at private-runner wrapper names and performance/finalization at the required historical marker basenames. Add `--post-source-adapter-profile` to provision and preflight. Require this argument only for v3 recipe-v2 deployments.

- [ ] **Step 5: Verify Task 2 GREEN**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_post_source_wrapper.py \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/release/test_provision_p6_full_chain_local_config.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_post_source_wrapper_template_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_post_source_wrapper.py
```

- [ ] **Step 6: Commit Task 2**

```bash
git add framework/stage6/p6_post_source_wrapper_template_v1.py \
  framework/stage6/p6_history_execution_closure_v1.py \
  framework/stage6/p6_history_normalization_v1.py \
  framework/stage6/p6_full_chain_bootstrap_v1.py \
  tools/release/provision_p6_full_chain_local_config.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tests/stage6/test_p6_post_source_wrapper.py \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/release/test_provision_p6_full_chain_local_config.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py
git commit -m "feat: render P6 post-source adapter wrappers"
```

---

### Task 3: Shared Round Runtime and Quantization Fanout

**Files:**
- Create: `framework/stage6/p6_round_adapter_runtime_v1.py`
- Create: `framework/stage6/p6_quantization_round_adapter_v1.py`
- Create: `tools/release/run_p6_quantization_round_adapter.py`
- Test: `tests/stage6/test_p6_quantization_round_adapter.py`
- Test: `tests/release/test_run_p6_quantization_round_adapter.py`

**Interfaces:**
- Produces: `RoundContext`, `LeafRunner`, `load_round_context(...)`, `advance_task_state(...)`, and `run_quantization_round(profile, task_state, round_root, runner) -> None`.
- Task 4 and Task 5 consume the shared runtime without expanding its validation surface.

- [ ] **Step 1: Write exact quantization RED tests**

Cover four requests: all FP16, all INT8, mixed FP16/INT8, and same-group mixed q-mode. Assert zero FP16 leaf calls, one call per INT8 row, request-order GPU assignment through child `CUDA_VISIBLE_DEVICES`, native output paths by width, direct argv, deterministic env, and state advancement only after all native contracts validate.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_quantization_round_adapter.py \
  tests/release/test_run_p6_quantization_round_adapter.py
```

- [ ] **Step 3: Implement the shared minimal runtime**

`load_round_context` reads only the canonical request/profile/task-state needed by all stages. It verifies the existing request SHA, row SHA map, four row identities, prior stage, and ordered GPU CSV. `advance_task_state` atomically replaces only the stage string while preserving immutable row identity/status objects.

- [ ] **Step 4: Implement quant fanout and CLI**

Invoke each INT8 leaf with:

```python
(
    str(profile.project_python), str(leaf.implementation),
    "--onnx", contract["onnx_path"],
    "--calibration-npz", contract["calibration_npz"],
    "--calibration-summary", contract["calibration_summary"],
    "--output-json", str(output_path),
)
```

Use one GPU in leaf env per INT8 row. Do not call the leaf for FP16. Require
the historical output field `schema == "stage3_tvm_int8_quant_contract_v3"`
and request-derived output presence; do not invent a replacement contract.

- [ ] **Step 5: Verify Task 3 GREEN and coverage**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_quantization_round_adapter.py \
  tests/release/test_run_p6_quantization_round_adapter.py
PYTHONPATH=. pytest -q \
  --cov=framework.stage6.p6_round_adapter_runtime_v1 \
  --cov=framework.stage6.p6_quantization_round_adapter_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_quantization_round_adapter.py
```

- [ ] **Step 6: Commit Task 3**

```bash
git add framework/stage6/p6_round_adapter_runtime_v1.py \
  framework/stage6/p6_quantization_round_adapter_v1.py \
  tools/release/run_p6_quantization_round_adapter.py \
  tests/stage6/test_p6_quantization_round_adapter.py \
  tests/release/test_run_p6_quantization_round_adapter.py
git commit -m "feat: add P6 quantization round adapter"
```

---

### Task 4: Performance Plan and Execution Adapter

**Files:**
- Create: `framework/stage6/p6_performance_round_adapter_v1.py`
- Create: `tools/release/run_p6_performance_round_adapter.py`
- Test: `tests/stage6/test_p6_performance_round_adapter.py`
- Test: `tests/release/test_run_p6_performance_round_adapter.py`

**Interfaces:**
- Consumes: Task 3 `RoundContext`, `LeafRunner`, `advance_task_state`.
- Produces: `run_performance_round(profile, task_state, round_root, runner) -> None` and native `performance/{performance_manifest.json,performance_jobs.jsonl,performance_state.jsonl}`.

- [ ] **Step 1: Write planner/executor RED tests**

Assert the planner runs once before the executor, both receive exact named argv, ordered GPU CSV is unchanged, `max_workers == len(gpus)`, four manifest/job identities match the request, and task state remains `quantization` on planner/executor nonzero or incomplete native state.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_performance_round_adapter.py \
  tests/release/test_run_p6_performance_round_adapter.py
```

- [ ] **Step 3: Implement minimal plan then execute orchestration**

Call the profile-bound planner and executor with the exact historical CLI flags. Keep the planner's default TVM FP16 trial policy. Accept native success or confirmed performance failure as terminal; reject missing/ready rows.

- [ ] **Step 4: Verify Task 4 GREEN and coverage**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_performance_round_adapter.py \
  tests/release/test_run_p6_performance_round_adapter.py
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_performance_round_adapter_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_performance_round_adapter.py
```

- [ ] **Step 5: Commit Task 4**

```bash
git add framework/stage6/p6_performance_round_adapter_v1.py \
  tools/release/run_p6_performance_round_adapter.py \
  tests/stage6/test_p6_performance_round_adapter.py \
  tests/release/test_run_p6_performance_round_adapter.py
git commit -m "feat: add P6 performance round adapter"
```

---

### Task 5: AP Plan, Sanity, and Full Adapter

**Files:**
- Create: `framework/stage6/p6_ap_round_adapter_v1.py`
- Create: `tools/release/run_p6_ap_round_adapter.py`
- Test: `tests/stage6/test_p6_ap_round_adapter.py`
- Test: `tests/release/test_run_p6_ap_round_adapter.py`

**Interfaces:**
- Consumes: Task 3 shared runtime and Task 4 native performance outputs.
- Produces: `run_ap_round(profile, task_state, round_root, runner) -> None` and native `ap/{ap_plan.json,ap_plan.jsonl,ap_state.jsonl}` plus private shard files.

- [ ] **Step 1: Write AP RED tests**

Cover four ready rows on three GPUs, numerical-feasibility sanity terminal, performance-blocked row, sanity leaf nonzero, full leaf nonzero, and stale/missing plan rows. Assert deterministic round-robin shards, no two concurrent shard processes target the same GPU, sanity completes before full starts, native numerical skip is preserved, and only AP-ready rows require full terminal state.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_ap_round_adapter.py \
  tests/release/test_run_p6_ap_round_adapter.py
```

- [ ] **Step 3: Implement planner, GPU shards, sanity, and full**

Call the planner once. Partition plan rows by `row_index % len(gpus)`, producing at most one shard process per GPU. Run distinct GPU shards concurrently for sanity, wait for all, then repeat for full. Pass the profile project Python as `--univ2x-python`. Concatenate shard state by shard index using a sibling temp and `os.replace`.

- [ ] **Step 4: Verify Task 5 GREEN and coverage**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_ap_round_adapter.py \
  tests/release/test_run_p6_ap_round_adapter.py
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_ap_round_adapter_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_ap_round_adapter.py
```

- [ ] **Step 5: Commit Task 5**

```bash
git add framework/stage6/p6_ap_round_adapter_v1.py \
  tools/release/run_p6_ap_round_adapter.py \
  tests/stage6/test_p6_ap_round_adapter.py \
  tests/release/test_run_p6_ap_round_adapter.py
git commit -m "feat: add P6 AP round adapter"
```

---

### Task 6: Historical Finalization, Promotion, and P6 Projection

**Files:**
- Create: `framework/stage6/p6_finalization_round_adapter_v1.py`
- Create: `tools/release/run_p6_finalization_round_adapter.py`
- Modify: `tools/release/verify_p6_materializer_training_run.py`
- Test: `tests/stage6/test_p6_finalization_round_adapter.py`
- Test: `tests/release/test_run_p6_finalization_round_adapter.py`
- Test: `tests/release/test_verify_p6_materializer_training_run.py`

**Interfaces:**
- Consumes: canonical request, native performance/AP outputs, historical finalizer/promoter leaves, and the six existing finalization argv paths.
- Produces: exact existing private `task-state.json`, `actual-feedback.json`, `receipt.json`, and `barrier.json`; no new public schema.

- [ ] **Step 1: Write finalization RED tests**

Assert finalize runs once before promote; promoted success rows project only five metrics; allowed failure rows project only stable failure reason; row/request/source-evidence mismatch fails; nonfinite metrics fail; and any leaf/projection failure leaves actual feedback, receipt, and barrier absent and task state non-final.

- [ ] **Step 2: Write atomic commit-point RED tests**

Inject a writer failure after each planned publication step. Assert barrier is absent until all earlier leaves and final task state have been written, and a complete run passes the existing `translate_history_feedback` unchanged.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_finalization_round_adapter.py \
  tests/release/test_run_p6_finalization_round_adapter.py \
  tests/release/test_verify_p6_materializer_training_run.py -k 'finalization or promotion'
```

- [ ] **Step 4: Implement finalize, promote, projection, and publication**

Call both historical leaves with their exact named argv. Validate four promoted identities against the canonical request. Build exact existing completion mappings. Publish actual feedback, receipt, atomically replace task state with stage `finalization`, then publish barrier last.

- [ ] **Step 5: Extend verifier minimally**

Require native finalization and promotion outputs to exist beneath each private round root while preserving the current public-safe verifier report fields. Do not expose paths, metrics, row IDs, or adapter details.

- [ ] **Step 6: Verify Task 6 GREEN and coverage**

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_finalization_round_adapter.py \
  tests/release/test_run_p6_finalization_round_adapter.py \
  tests/release/test_verify_p6_materializer_training_run.py
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_finalization_round_adapter_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_finalization_round_adapter.py
```

- [ ] **Step 7: Commit Task 6**

```bash
git add framework/stage6/p6_finalization_round_adapter_v1.py \
  tools/release/run_p6_finalization_round_adapter.py \
  tools/release/verify_p6_materializer_training_run.py \
  tests/stage6/test_p6_finalization_round_adapter.py \
  tests/release/test_run_p6_finalization_round_adapter.py \
  tests/release/test_verify_p6_materializer_training_run.py
git commit -m "feat: finalize P6 historical measurement rounds"
```

---

### Task 7: Full Adapter Integration and Release Gates

**Files:**
- Modify: `tests/release/test_p6_history_execution_adapters.py`
- Modify: `tests/release/test_p6_source_reuse_lifecycle.py`
- Modify: `tests/stage6/test_p6_source_reuse_measurement.py`
- Modify only surfaced normalization/provision/release fixtures that require source-map v3 or the new profile.
- Document: append implementation evidence to the plan-specific ignored SDD reports, not a new top-level tracked document.

**Interfaces:**
- Consumes: Tasks 1–6 complete normalized profile, wrappers, adapter CLIs, and completion projection.
- Produces: one synthetic full round and four-round lifecycle proof through the real public measurement/controller boundaries.

- [ ] **Step 1: Replace synthetic raw-leaf success with adapter-chain RED**

Make the release execution fixture expose seven exact fake historical leaves and four tracked adapters. Assert the old normalized chain fails at the first raw-leaf positional mismatch before implementation commits are applied.

- [ ] **Step 2: Verify one full mixed-q round**

Run:

```bash
PYTHONPATH=. pytest -q tests/release/test_p6_history_execution_adapters.py
```

Assert four feedback rows, exact stage order, FP16 zero quant leaves, INT8 quant fanout, performance plan+execute once, AP sanity/full shards, finalize+promote once, and barrier-last completion.

- [ ] **Step 3: Verify source reuse and four rounds**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/release/test_p6_source_reuse_lifecycle.py
```

Assert reuse changes only source invocation count; all 16 selected row identities traverse all four downstream adapters; Gold176 overlap remains zero.

- [ ] **Step 4: Run changed-module coverage and static gates**

```bash
PYTHONPATH=. pytest -q \
  --cov=framework.stage6.p6_post_source_leaf_binding_v1 \
  --cov=framework.stage6.p6_post_source_adapter_profile_v1 \
  --cov=framework.stage6.p6_post_source_wrapper_template_v1 \
  --cov=framework.stage6.p6_round_adapter_runtime_v1 \
  --cov=framework.stage6.p6_quantization_round_adapter_v1 \
  --cov=framework.stage6.p6_performance_round_adapter_v1 \
  --cov=framework.stage6.p6_ap_round_adapter_v1 \
  --cov=framework.stage6.p6_finalization_round_adapter_v1 \
  --cov-branch --cov-report=term-missing --cov-fail-under=80 \
  tests/stage6/test_p6_post_source_adapter_profile.py \
  tests/stage6/test_p6_post_source_wrapper.py \
  tests/stage6/test_p6_quantization_round_adapter.py \
  tests/stage6/test_p6_performance_round_adapter.py \
  tests/stage6/test_p6_ap_round_adapter.py \
  tests/stage6/test_p6_finalization_round_adapter.py \
  tests/release/test_p6_history_execution_adapters.py
python -m ruff check framework tools/release tests
python -m compileall -q framework tools/release tests
git diff --check
```

- [ ] **Step 5: Run complete repository gates**

```bash
PYTHONPATH=. pytest -q tests/stage6 tests/release
PYTHONPATH=. pytest -q
```

Inspect every failure from its first boundary; migrate only stale fixtures whose contract genuinely changed.

- [ ] **Step 6: Run sequential reviews and fix through RED/GREEN**

Dispatch SPEC review against the design and plan. Only after SPEC passes, dispatch QUALITY, MLE, and security reviews. Reproduce every load-bearing finding with a failing test, implement the minimum correction, rerun the scoped review, then rerun the complete gates.

- [ ] **Step 7: Commit integration fixes**

```bash
git add framework tools/release tests
git commit -m "test: verify P6 post-source adapter chain"
```

---

### Task 8: Fresh Private Deployment and Real Four-Round Experiment

**Files:**
- Create only Git-ignored/private source-map, runner, profile, locator, preflight report, runtime roots, logs, and experiment outputs.
- Modify no tracked production file during deployment or execution.

**Interfaces:**
- Consumes: exact clean reviewed HEAD from Task 7 and existing official derive/normalize/provision/preflight/run/verifier CLIs.
- Produces: a fresh ignored locator and, on success, a verifier-accepted four-round/16-row real run.

- [ ] **Step 1: Build a wholly fresh private source root**

Create a new absent prefix. Copy the exact reviewed HEAD archive, dependency overlay, and historical leaf closure without reusing any prior normalized/output root. Author a v3 ignored source map whose leaf binding points to renamed dependency-only copies of the seven historical leaf scripts. Preserve exact leaf bytes and closure digests.

- [ ] **Step 2: Run official derive and normalize**

Use:

```bash
python tools/release/derive_p6_history_recipe.py \
  --source-map "$P6_ADAPTER_SOURCE_MAP" \
  --runner-template "$P6_ADAPTER_SOURCE_RUNNER" \
  --recipe-json "$P6_ADAPTER_DERIVED_RECIPE"
python tools/release/normalize_p6_history_root.py \
  --source-map "$P6_ADAPTER_SOURCE_MAP" \
  --history-root "$P6_ADAPTER_HISTORY_ROOT" \
  --private-dir "$P6_ADAPTER_NORMALIZED_ROOT" \
  --runner-template "$P6_ADAPTER_SOURCE_RUNNER"
```

The shell variables are loaded from one mode-0600 ignored operator locator. Before each command, resolve every value to an absolute existing parent or required-absent destination.

- [ ] **Step 3: Provision and run zero-process preflight**

```bash
python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "$P6_ADAPTER_LEGACY_CONFIG" \
  --runner-template "$P6_ADAPTER_NORMALIZED_RUNNER" \
  --local-output-root "$P6_ADAPTER_OUTPUT_ROOT" \
  --binding-output "$P6_ADAPTER_BINDING" \
  --config-output "$P6_ADAPTER_LOCAL_CONFIG" \
  --source-wrapper-profile "$P6_ADAPTER_SOURCE_PROFILE" \
  --post-source-adapter-profile "$P6_ADAPTER_POST_SOURCE_PROFILE" \
  --external-training-binding "$P6_ADAPTER_EXTERNAL_TRAINING"
python tools/release/preflight_p6_materializer_training_bridge.py \
  --contract "$P6_ADAPTER_PUBLIC_CONTRACT_JSON" \
  --local-config "$P6_ADAPTER_LOCAL_CONFIG" \
  --binding "$P6_ADAPTER_BINDING" \
  --runner-template "$P6_ADAPTER_NORMALIZED_RUNNER" \
  --source-wrapper-profile "$P6_ADAPTER_SOURCE_PROFILE" \
  --post-source-adapter-profile "$P6_ADAPTER_POST_SOURCE_PROFILE" \
  --external-training-binding "$P6_ADAPTER_EXTERNAL_TRAINING"
```

Require accepted four-round training mode, zero historical processes, zero GPU probe side effects, exact wrapper/profile bytes, exact reviewed code archive digest, and absent controller/output leaves.

- [ ] **Step 4: Run real prelaunch gates**

Re-establish and verify the approved SSH ControlMaster. Under the activation-effective environment, verify registry/controller/measurement/adapter module origins, canonical GPU double snapshots in binding order, runtime interpreters, disk/inodes, zero existing controller/training/measurement processes, and fresh absent output leaves.

- [ ] **Step 5: Launch exactly once and monitor without interruption**

```bash
python tools/release/run_p6_h800_search.py \
  --contract "$P6_ADAPTER_PUBLIC_CONTRACT_JSON" \
  --local-config "$P6_ADAPTER_LOCAL_CONFIG" \
  --code-revision "$(git rev-parse HEAD)"
```

Keep the launch under a persistent waiting parent. Record launch count one. Monitor Stage1, formal Stage2, registry identity, each first-use source group, all four post-source adapters, round feedback, and online refit. Do not interrupt healthy long training/AP work.

- [ ] **Step 6: Verify terminal completion**

```bash
python tools/release/verify_p6_materializer_training_run.py \
  --contract "$P6_ADAPTER_PUBLIC_CONTRACT_JSON" \
  --local-config "$P6_ADAPTER_LOCAL_CONFIG" \
  --binding "$P6_ADAPTER_BINDING"
```

Accept only four completed rounds, 16 selected unique rows, 16 `measured_success_gold` real measurements, zero Gold176 remeasurement, and exact completion files. On the first stable failure, freeze the root and perform read-only diagnosis; never relaunch or repair it in place.
