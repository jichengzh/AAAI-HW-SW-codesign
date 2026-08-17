# P6.2 Dynamic Framework Candidate Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the P6 CoptV2X four-round H800/TVM loop from the complete dynamic Stage2 Pyramid search space, without fixed 343/686 cardinality or a new mixed-precision/pruning method.

**Architecture:** The adapter emits one canonical candidate for every valid three-stage, global-`q_mode` Stage2 combination. A framework registry preserves exactly those candidate identities and their q-mode provenance; Stage5 expands only declared q modes. The P6 controller retains static P6.1 343×2 validation, but framework mode compares plan, registry, and manifest identities exactly and requires only the fixed 16-measurement budget.

**Tech Stack:** Python 3.10+, pytest, PyYAML, `framework.stage1_bridge.load_stage2_search_space`, `framework.stage5.single_target_search_v2`, `framework.stage6.coptv2x_h800_search_v2`.

## Global Constraints

- Keep the existing CoptV2X protocol: H800, TVM, Pyramid, Gold176, four rounds, four selected candidates per round, 16 measurements total, and `predicted_frontier_diversity` acquisition.
- A candidate has exactly one global `q_mode`: `fp16` or `int8`. Never create a per-stage mixed-precision candidate.
- Construct the framework candidate pool from every active, buildable, H800/TVM-compatible Stage2 point; do not add heuristic, proxy, Pareto, FLOPs, or width-range pre-pruning.
- Candidate cardinality in framework mode is dynamic. It must equal the exact Stage2-derived identity set and contain at least 16 eligible unmeasured candidates before round 0.
- Preserve P6.1 static mode and its 343 structures / 686 candidates validation unchanged.
- Framework mode must not fall back to the static registry under any failure condition.
- Do not download assets or execute Orin, TensorRT, CPU, RTX 4090, or unrelated backends.
- Use existing Git-ignored assets, local adapter configuration, and output root only. Never commit paths, host information, commands, candidate IDs, raw measurements, checkpoints, logs, or generated outputs.
- Do not push or merge the branch.

---

## File Structure

- Modify `framework/stage6/pyramid_search_space_adapter_v1.py`: emit a dynamic candidate plan with pure global q modes.
- Modify `tests/stage6/test_pyramid_search_space_adapter.py`: cover independent q-mode products, variable cardinality, and invalid Stage2 records.
- Modify `framework/stage5/production_search_v1.py`: validate source-registry v2 q-mode declarations for the reusable manifest builder.
- Modify `framework/stage5/single_target_search_v2.py`: expand framework groups only into their declared q modes.
- Modify `tests/stage5/test_production_search.py` and `tests/stage5/test_single_target_search.py`: preserve v1 static expansion and prove v2 dynamic expansion.
- Modify `framework/stage6/coptv2x_h800_search_v2.py`: apply dynamic plan/registry/manifest identity gates in framework mode and preserve static gates.
- Modify `tests/stage6/test_coptv2x_h800_search.py` and `tests/release/test_run_p6_h800_search.py`: cover dynamic four-round control flow and CLI compatibility.
- Modify `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`, `docs/AAAI27_RELEASE_AUDIT.md`, and `tests/release/test_project_handoff.py`: correct the P6.2 state until the dynamic offline gate and then the real local run complete.

### Task 1: Emit a Dynamic Stage2 Candidate Plan

**Files:**

- Modify: `framework/stage6/pyramid_search_space_adapter_v1.py`
- Modify: `tests/stage6/test_pyramid_search_space_adapter.py`

**Consumes:** `stage2_search_space_v1` with exact Stage1/2/3 software points and an H800 tuned TVM hardware candidate.

**Produces:**

```python
{
    "schema_version": "p6_pyramid_candidate_plan_v2",
    "source_schema": "stage2_search_space_v1",
    "target_model": "pyramid",
    "hardware_target": "h800",
    "execution_backend": "tvm_auto",
    "candidate_source_mode": "framework_stage2_search_space",
    "structure_count": 5,
    "candidate_count": 7,
    "candidates": [
        {
            "width": [16, 32, 64],
            "q_mode": "fp16",
            "source_point_ids": ["stage1-fp16-16", "stage2-fp16-32", "stage3-fp16-64"],
        }
    ],
}
```

- [ ] **Step 1: Write RED adapter tests for independent global q-mode products**

Add a Stage2 fixture in `tests/stage6/test_pyramid_search_space_adapter.py` with two active/buildable FP16 widths in every stage and one active/buildable INT8 width in every stage. Assert:

```python
plan = build_pyramid_candidate_plan(_space())

assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
assert plan["structure_count"] == 8
assert plan["candidate_count"] == 9
assert {(tuple(row["width"]), row["q_mode"]) for row in plan["candidates"]} == {
    *((width, "fp16") for width in product((16, 32), (32, 64), (64, 96))),
    ((16, 32, 64), "int8"),
}
assert all(row["q_mode"] in {"fp16", "int8"} for row in plan["candidates"])
```

Add a second fixture that has no active/buildable INT8 points. Assert that it emits every FP16 candidate, emits zero INT8 candidates, and does not fail merely because the FP16 widths have no INT8 counterpart.

- [ ] **Step 2: Write RED adapter rejection and determinism tests**

Add tests that call `build_pyramid_candidate_plan()` and expect `PyramidSearchSpaceAdapterError` for: an unsupported schema/model/hardware/backend; an unknown quant policy; an active non-buildable point; duplicate `(stage, width, q_mode)` point identities; missing any dense stage; and an input with no globally complete q-mode product. Reverse all input lists and assert the plan remains byte-for-byte canonical after JSON serialization with sorted keys.

- [ ] **Step 3: Run the adapter RED tests**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/stage6/test_pyramid_search_space_adapter.py
```

Expected: failures identify the old `p6_pyramid_structure_plan_v1` output, q-mode intersection rule, and/or missing `build_pyramid_candidate_plan` function.

- [ ] **Step 4: Implement the candidate-plan conversion**

In `framework/stage6/pyramid_search_space_adapter_v1.py`:

1. Rename the public converter to `build_pyramid_candidate_plan(search_space: Mapping[str, Any]) -> dict[str, Any]` and keep a thin compatibility alias only if a non-P6 caller imports `build_pyramid_structure_plan`.
2. Keep the current schema/model/H800/tuned-TVM and three-stage validation.
3. For each `q_mode` separately, collect only `status == "active" and buildable is True` points in Stage1/2/3. A non-active non-buildable diagnostic point is ignored; an active non-buildable point fails; every other unsupported status/buildable pair fails.
4. Build the Cartesian product independently for FP16 and INT8. Each resulting record carries exactly the corresponding three source point IDs and one q mode.
5. Sort candidates by `(tuple(width), q_mode, tuple(source_point_ids))`; reject duplicate `(width, q_mode)` identities. Set `structure_count` from distinct widths and `candidate_count` from all candidate records. Do not compute a q-mode intersection and do not insert a missing q mode.

- [ ] **Step 5: Run adapter GREEN tests and static checks**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/stage6/test_pyramid_search_space_adapter.py
python -m ruff check framework/stage6/pyramid_search_space_adapter_v1.py tests/stage6/test_pyramid_search_space_adapter.py
git diff --check
```

Expected: all adapter tests pass; no Ruff or whitespace error.

- [ ] **Step 6: Commit Task 1**

```bash
git add framework/stage6/pyramid_search_space_adapter_v1.py tests/stage6/test_pyramid_search_space_adapter.py
git commit -m "feat: derive dynamic P6 framework candidates"
```

### Task 2: Make Stage5 Source Registries q-Mode Exact

**Files:**

- Modify: `framework/stage5/production_search_v1.py`
- Modify: `framework/stage5/single_target_search_v2.py`
- Modify: `tests/stage5/test_production_search.py`
- Modify: `tests/stage5/test_single_target_search.py`

**Consumes:** legacy `stage5_candidate_source_registry_v1` and framework `stage5_candidate_source_registry_v2` source registries.

**Produces:** static v1 groups still expand to both q modes; v2 groups expand only their sorted, unique `available_q_modes` values.

- [ ] **Step 1: Write RED manifest tests for source-registry v2**

In both Stage5 test modules, construct a valid source group with:

```python
{
    "schema_version": "stage5_candidate_source_registry_v2",
    "groups": [
        {
            **valid_source_group,
            "available_q_modes": ["fp16"],
            "source_point_ids_by_q_mode": {"fp16": ["s1", "s2", "s3"]},
        }
    ],
}
```

Assert the manifest has one candidate row with `q_mode == "fp16"`; the same group under v1 without those fields still emits two rows. Add rejection tests for an empty/duplicate/unknown q-mode list, missing q-mode provenance, a provenance list that is not exactly three nonempty strings, and v1 carrying a v2-only field.

- [ ] **Step 2: Run Stage5 RED tests**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/stage5/test_production_search.py tests/stage5/test_single_target_search.py
```

Expected: v2 fixtures fail with an unexpected registry schema or extra q-mode expansion.

- [ ] **Step 3: Implement shared source-registry q-mode validation**

In `framework/stage5/production_search_v1.py`, define and export:

```python
def source_group_q_modes(
    source_registry_schema: str, group: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return v1's two default q modes or validate a v2 declared subset."""
```

Rules:

1. Accept only registry schemas `stage5_candidate_source_registry_v1` and `_v2`.
2. For v1, reject `available_q_modes` and `source_point_ids_by_q_mode`, then return `("fp16", "int8")`.
3. For v2, require a sorted, nonempty, duplicate-free subset of `("fp16", "int8")`; require a mapping with exactly the same q-mode keys; every value must be a three-item sequence of nonempty strings.
4. Modify both `build_candidate_manifest()` and `build_task_candidate_manifest()` to invoke the helper and iterate only its returned q modes.
5. Preserve every existing source-contract SHA, graph feature, holdout, measured-row, and candidate-row validation.

- [ ] **Step 4: Run Stage5 GREEN tests and static checks**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/stage5/test_production_search.py tests/stage5/test_single_target_search.py
python -m ruff check framework/stage5/production_search_v1.py framework/stage5/single_target_search_v2.py tests/stage5/test_production_search.py tests/stage5/test_single_target_search.py
git diff --check
```

Expected: v1 regression fixtures retain two q modes and v2 fixtures retain exactly their declared dynamic modes.

- [ ] **Step 5: Commit Task 2**

```bash
git add framework/stage5/production_search_v1.py framework/stage5/single_target_search_v2.py tests/stage5/test_production_search.py tests/stage5/test_single_target_search.py
git commit -m "feat: preserve framework q-mode candidate identities"
```

### Task 3: Apply Dynamic Framework Gates in the P6 Controller

**Files:**

- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/release/test_run_p6_h800_search.py`

**Consumes:** candidate plan v2, framework registry v2, static registry v1, fixed P6 public task budget.

**Produces:** framework source mode validates exact dynamic identities and at least 16 eligible rows; static mode still validates exactly 343 structures and 686 candidates.

- [ ] **Step 1: Write RED controller tests for a non-343/686 four-round framework pool**

Add a synthetic Stage2 fixture with an FP16 product of 18 candidate widths and an INT8 product of 6 candidate widths. Its framework registry materializer must emit v2 groups with exact `available_q_modes` and `source_point_ids_by_q_mode` from the written plan. Assert:

```python
state = run_p6_coptv2x_search(contract, framework_local, "test-revision", runner)

assert state.status == "completed"
assert state.completed_rounds == 4
assert state.measured_candidate_count == 16
assert observed_plan["candidate_count"] == 24
assert observed_manifest_candidate_identities == observed_plan_candidate_identities
```

Add tests that fail before a measurement argv is observed when: registry v2 omits, adds, duplicates, or changes a plan candidate; manifest has fewer than 16 eligible rows; a static v1 registry has a count other than 343/686; or framework mode receives a v1 registry.

- [ ] **Step 2: Run controller RED tests**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/stage6/test_coptv2x_h800_search.py -k 'framework or source_space'
```

Expected: failures show the fixed 343/686 check and the current plan/registry assumption cannot accept the 24-candidate framework fixture.

- [ ] **Step 3: Implement dynamic framework validation**

In `framework/stage6/coptv2x_h800_search_v2.py`:

1. Replace the framework plan builder call with `build_pyramid_candidate_plan()` and write the resulting local JSON as `pyramid_candidate_plan.json`.
2. Keep the existing local-only source-mode and template-token boundary. Rename the framework-only token to `{pyramid_candidate_plan}` and reject it in static mode.
3. Replace `_validate_framework_registry_plan()` with a mapping comparison from `(tuple(width), q_mode)` to `tuple(source_point_ids)`. Require registry schema v2 and exact equality; use source groups' `available_q_modes` and provenance mapping to construct actual identities.
4. In `_validate_p6_source_space()`, branch by source mode. Static mode retains `FIXED_SOURCE_GROUP_COUNT` and `FIXED_ELIGIBLE_GENOME_COUNT`. Framework mode requires registry v2, validates `plan["candidate_count"] == len(plan["candidates"])`, validates the manifest identity set exactly equals plan identities, and requires `eligible_row_count >= task.sample_budget`.
5. Keep `SearchTask` at B=4, T=16, four rounds, Gold176 loading, capability-profile checks, model fitting, feedback validation, and `predicted_frontier_diversity` unchanged.
6. Preserve error normalization: any plan/registry/manifest violation becomes `P6CoptV2XExecutionError("source_registry_invalid", ...)` before the measurement step starts.

- [ ] **Step 4: Run controller GREEN, CLI, and static-regression tests**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py
python -m ruff check framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py
git diff --check
```

Expected: dynamic 24-candidate framework loop completes with synthetic feedback; invalid dynamic sources fail before measurement; legacy static 343/686 tests remain green.

- [ ] **Step 5: Commit Task 3**

```bash
git add framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py
git commit -m "feat: validate dynamic P6 framework candidate pools"
```

### Task 4: Correct Public P6.2 Status and Archive Boundaries

**Files:**

- Modify: `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify: `tests/release/test_project_handoff.py`
- Modify: `tests/integration/test_anonymous_archive.py` only if archive coverage fails.

**Consumes:** implemented dynamic candidate contract and completed offline tests.

**Produces:** public documentation that distinguishes P6.1 historical static closure, P6.2 dynamic offline integration, and the still-local real H800 framework-derived run.

- [ ] **Step 1: Write RED documentation-state tests**

Update `tests/release/test_project_handoff.py` to require these statements:

```python
assert "P6.1 静态 343×2" in manifest
assert "P6.2 动态框架候选空间" in manifest
assert "不预设为 343 或 686" in manifest
assert "单个候选不是 mixed-precision" in manifest
assert "真实框架来源闭环仍待本地执行" in manifest
```

Keep negative assertions preventing public absolute paths, candidate IDs, raw metrics, P6 overall closure, real framework H800 closure, Stage6/Stage7 evidence completion, and P7/P8 start claims.

- [ ] **Step 2: Run documentation RED tests**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/release/test_project_handoff.py
```

Expected: failure because public status still describes the superseded fixed-cardinality P6.2 converter as complete.

- [ ] **Step 3: Update public documentation**

Update both P6 documents to state:

1. P6.1 static 343×2 closure remains a completed local historical path.
2. P6.2 dynamically derives the candidate pool from Stage2 and does not predeclare 343/686.
3. P6.2 offline implementation/verification is in progress until Tasks 1–3 pass; after they pass, record only offline dynamic completion.
4. A real H800 framework-derived four-round run remains local, Git-ignored, and distinct from offline completion and paper evidence.

Do not write any candidate ID, local path, command, raw result, checkpoint, host, or raw log. Update the anonymous archive test only if it proves a changed public Python module is excluded; do not alter allowlist rules speculatively.

- [ ] **Step 4: Run documentation GREEN tests**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/release/test_project_handoff.py tests/integration/test_anonymous_archive.py
python -m ruff check tests/release/test_project_handoff.py
git diff --check
```

Expected: docs and archive boundaries are consistent; no tracking change exposes private execution material.

- [ ] **Step 5: Commit Task 4**

```bash
git add docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md tests/release/test_project_handoff.py tests/integration/test_anonymous_archive.py
git commit -m "docs: describe dynamic P6.2 candidate space"
```

### Task 5: Offline Release Gate and Authorized Dynamic H800 Execution

**Files:**

- No tracked-file changes are required for the real run.
- Local-only inputs: existing Git-ignored assets, dynamic framework local configuration, source-registry materializer, measurement adapter, and output root.

**Consumes:** Tasks 1–4, user authorization for H800 execution, and the already available local H800/TVM environment.

**Produces:** a Git-ignored real local four-round run state, or a Git-ignored failure state with no public result output.

- [ ] **Step 1: Run the complete offline quality and release gates**

Run:

```bash
PYTHONPATH=. python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
PYTHONPATH=. python -m pytest -q tests/release tests/integration/test_anonymous_archive.py
python -m ruff check framework/stage5/production_search_v1.py framework/stage5/single_target_search_v2.py framework/stage6/pyramid_search_space_adapter_v1.py framework/stage6/coptv2x_h800_search_v2.py tests/stage5 tests/stage6 tests/release/test_project_handoff.py
git diff --check
```

Expected: zero failures, coverage at least 80%, and no whitespace errors.

- [ ] **Step 2: Run local dynamic preflight without the measurement adapter**

Use the existing Git-ignored framework-mode configuration to build the Stage2 candidate plan and source registry. Verify locally that the registry and Stage5 manifest identities are equal, all candidates have a pure global q mode, the manifest contains at least 16 eligible rows, and neither public repository files nor the anonymous archive changed. If any check fails, stop before the first training/measurement command and repair the offline contract.

- [ ] **Step 3: Launch the authorized real CoptV2X loop**

Start the existing P6 release CLI with the Git-ignored dynamic framework local configuration and the current short code revision. The measurement adapter must consume each generated request and perform its existing H800/TVM training, checkpoint selection, ONNX export, compilation, latency, energy, AP30, AP50, and AP70 measurements. Do not download assets and do not publish a result summary.

- [ ] **Step 4: Monitor and validate the local terminal state**

Wait for terminal `completed` or `failed`. On completed, check only local structural facts: four completed rounds, 16 measured candidates, request/feedback pairing, and no failed terminal status. On failure, retain Git-ignored failure evidence, report the normalized failure code, and do not retry a different candidate set without diagnosis.

- [ ] **Step 5: Commit only a redacted status update after successful local validation**

After a successful real run, update the P6 audit/manifest with structural completion only: dynamic Stage2 candidate pool used, four feedback rounds completed, and no public raw results. Add matching negative disclosure tests, run release tests, then commit with:

```bash
git add docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md tests/release/test_project_handoff.py
git commit -m "docs: record dynamic P6.2 local validation"
```

Do not commit or publish local execution files.

## Self-Review Checklist

- Spec coverage: Tasks 1–3 implement dynamic Stage2 candidates, pure q-mode semantics, registry/manifest exactness, static P6.1 compatibility, and the fixed four-round protocol; Task 4 controls public claims; Task 5 defines preflight and authorized execution.
- Placeholder scan: this plan contains concrete files, public interfaces, test commands, expected outcomes, and commit scopes; no incomplete implementation markers remain.
- Type consistency: `p6_pyramid_candidate_plan_v2`, `stage5_candidate_source_registry_v2`, `available_q_modes`, `source_point_ids_by_q_mode`, and `{pyramid_candidate_plan}` are defined before later tasks consume them.
- Scope check: all changed code belongs to the candidate-source boundary or its direct Stage5/controller consumer; no new search heuristic, model family, backend, or external asset action is introduced.
