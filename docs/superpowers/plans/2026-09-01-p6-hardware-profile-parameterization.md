# P6 Hardware-Profile Parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task. Steps use checkbox syntax for tracking. Do not run real SSH, GPU, controller, or measurement work until Tasks 1–8 and their reviews pass.

**Goal:** Parameterize the existing P6 H800 path with an immutable H800/RTX4090 hardware profile registry so the unchanged controller and verifier can execute a real four-card RTX4090 4x4/16 experiment.

**Architecture:** A new registry is the only authority for target, hardware/environment YAML, normalized GPU models, backend scope, TVM arch/cache namespace, occupancy, and RTX cardinality. Public/local v2 remains H800-default compatible; v3 persists a selected profile through the existing Stage1, binding, adapter, release, controller, and verifier chain.

**Tech Stack:** Python 3.11, dataclasses, pathlib, PyYAML, JSON, pytest, pytest-cov, Ruff, existing Stage1/Stage2 scanner, historical training/quantization/TVM/AP leaves.

**Spec:** `docs/superpowers/specs/2026-09-01-p6-hardware-profile-parameterization-design.md`

## Global Constraints

- Preserve current public v2 H800 contract/local-config behavior and existing release CLI arguments.
- Treat `provision_p6_history_local_config.py` only as a legacy H800-only,
  non-materializing diagnostic.  RTX provisioning uses only
  `provision_p6_full_chain_local_config.py`.
- Add no RTX controller, verifier, four-round chain, proxy metric, manual candidate plan, static fallback, TensorRT-as-TVM substitution, or broad historical/security work.
- Keep fixed 4 rounds, 4 rows per round, 16 unique measured candidates, five real metrics, and Gold176 overlap zero.
- `rtx4090` requires one ordered four-card policy; H800 preserves current nonempty ordered-policy compatibility.
- GPU UUIDs, private paths, raw logs/commands, data/checkpoint values, and local outputs remain private/ignored.
- TVM reuse requires an exact profile/candidate/source/toolchain manifest match; otherwise compile through the existing real TVM path.
- Every task follows observed RED, minimal GREEN, focused review, then suite/coverage verification. Do not change unrelated code.

---

### Task 1: Immutable Hardware Registry

**Files:**
- Create: `framework/stage6/hardware_execution_profile_v1.py`
- Create: `tests/stage6/test_hardware_execution_profile.py`

**Interfaces:**
- Produces `HardwareExecutionProfile`, `load_hardware_execution_profile(profile_id)`, `default_hardware_execution_profile()`, `validate_profile_gpu_policy(profile, indices)`, and `validate_profile_gpu_records(profile, records, indices)`.
- Registry ids are exactly `h800` and `rtx4090`; the latter resolves `sm89`, `rtx4090-sm89`, `tvm_auto`, the existing RTX YAMLs, exact RTX model normalizations, four cards, and occupancy `0.05`.

- [ ] Write RED tests asserting frozen dataclass fields, exact registry ids, H800 default, YAML/arch/backend values, unknown-id rejection, RTX four-card rejection for 1/3/5 indices, H800 acceptance of the existing two-card fixture, and model/backend mismatch rejection.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_hardware_execution_profile.py
```

Expected: import failure for `hardware_execution_profile_v1`.

- [ ] Implement the registry with immutable `frozenset`/tuple fields; resolve only repository-relative declared YAML paths and validate them at load time. Do not read GPUs or environment variables in this module.
- [ ] Run GREEN and mutation coverage:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_hardware_execution_profile.py
PYTHONPATH=. pytest -q --cov=framework.stage6.hardware_execution_profile_v1 --cov-branch --cov-fail-under=80 tests/stage6/test_hardware_execution_profile.py
python -m ruff check framework/stage6/hardware_execution_profile_v1.py tests/stage6/test_hardware_execution_profile.py
```

- [ ] Independent review gate: confirm no public/private path, UUID, command, cache artifact, or mutable registry escapes the module.

### Task 2: Backward-Compatible Controller Contracts

**Files:**
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/stage6/test_p6_fresh_run_controller.py`

**Interfaces:**
- `PublicP6CoptV2XContract` and `LocalP6CoptV2XConfig` gain `hardware_profile: HardwareExecutionProfile`.
- `load_public_contract(path)` accepts v2 as implicit H800 and v3 with exact `hardware_profile`; `load_local_config(path, contract)` accepts local v2 as implicit H800 and v3 only when ids agree.

- [ ] Write RED fixtures for v3 H800 and RTX contracts/local configs. Assert v2 objects select H800 unchanged; v3 rejects unknown profile, target/profile disagreement, backend outside `backend_scope`, local/public profile disagreement, and a profile-specific Stage2 hardware target mismatch before any runner call.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_fresh_run_controller.py -k 'hardware_profile or v3'
```

Expected: failures because v3 keys and profile consistency are unsupported.

- [ ] Implement exact v2/v3 key sets and profile-derived checks. Keep `FIXED_SAMPLE_BUDGET`, `FIXED_BATCH_SIZE`, `FIXED_ROUND_COUNT`, metrics, controller loop, and `run_p6_coptv2x_search` signature unchanged.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_fresh_run_controller.py
PYTHONPATH=. pytest -q --cov=framework.stage6.coptv2x_h800_search_v2 --cov-branch --cov-fail-under=80 tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_fresh_run_controller.py
```

- [ ] Independent review gate: compare every existing v2 fixture result and CLI-visible category with its pre-change expectation.

### Task 3: Scanner and Formal Candidate-Plan Profile Consistency

**Files:**
- Modify: `framework/stage6/p6_stage1_bridge_v1.py`
- Modify: `framework/stage6/p6_formal_plan_contract_v1.py`
- Create: `configs/stage1/p6_rtx4090_formal_scan.yaml`
- Create: `tests/stage6/test_p6_hardware_profile_stage1_plan.py`
- Modify: `tests/stage6/test_p6_stage1_bridge.py`
- Modify: `tests/stage6/test_formal_plan_v1.py`

**Interfaces:**
- `build_p6_stage1_partition_manifest(..., profile: HardwareExecutionProfile, ...)` validates the profile hardware YAML and emits its target.
- `validate_p6_candidate_plan(raw_plan, *, profile)` requires plan target/backend to equal that profile.

- [ ] Write RED tests using scanner fakes: H800 preserves current manifest; RTX accepts only the committed RTX YAML and emits `rtx4090`; foreign YAML, `sm90` under RTX, plan target drift, backend drift, and a manually substituted candidate-plan target fail.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_hardware_profile_stage1_plan.py tests/stage6/test_p6_stage1_bridge.py tests/stage6/test_formal_plan_v1.py -k 'profile or rtx'
```

Expected: missing profile-aware interfaces and RTX formal scan fixture.

- [ ] Implement profile-driven validation, add only the RTX scanner config corresponding to existing `configs/hardware/rtx4090.yaml`, and thread the selected profile from controller-local Stage1 invocation. Preserve scanner ownership of candidate generation and all 343/686 assertions.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_hardware_profile_stage1_plan.py tests/stage6/test_p6_stage1_bridge.py tests/stage6/test_formal_plan_v1.py
python -m ruff check framework/stage6/p6_stage1_bridge_v1.py framework/stage6/p6_formal_plan_contract_v1.py tests/stage6/test_p6_hardware_profile_stage1_plan.py
```

- [ ] Independent review gate: verify there is no hand-authored RTX candidate list or conditional scanner bypass.

### Task 4: Binding, Admission, and Runtime Policy

**Files:**
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Modify: `framework/stage6/p6_history_source_materialization_v1.py`
- Modify: `tests/stage6/test_p6_history_binding.py`
- Modify: `tests/stage6/test_p6_history_measurement.py`
- Modify: `tests/stage6/test_p6_history_source_materialization.py`

**Interfaces:**
- Binding `gpu_policy` persists `hardware_profile`; `discover_history_binding(..., profile)` and runtime validation return/use the selected `HardwareExecutionProfile`.
- GPU admission uses `validate_profile_gpu_policy` and `validate_profile_gpu_records`; the existing two-snapshot sequence and ordered UUID binding remain unchanged.

- [ ] Write RED tests for a valid four-card RTX fake probe, policy count/model/order/second-snapshot drift failures, H800 two-card regression acceptance, and source/runtime rejection when binding profile differs from contract/profile.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_p6_history_source_materialization.py -k 'rtx or hardware_profile'
```

Expected: binding lacks a persisted profile and RTX admission fails as H800.

- [ ] Implement profile-id persistence and centralized registry validation; delete duplicated H800 allowlist/occupancy decisions only after all consumers use the registry. Preserve exact error categories and no-launch-before-admission ordering.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_p6_history_source_materialization.py
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_history_binding_v1 --cov=framework.stage6.p6_history_measurement_v1 --cov-branch --cov-fail-under=80 tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_history_measurement.py
```

- [ ] Independent review gate: inspect public projections and stable errors for profile identity only, with no GPU indices, UUIDs, or model text leak.

### Task 5: Normalized Profiles and TVM Cache Evidence

**Files:**
- Modify: `framework/stage6/p6_post_source_adapter_profile_v1.py`
- Modify: `framework/stage6/p6_history_registry_v1.py`
- Modify: `framework/stage6/p6_history_feedback_validation_v1.py`
- Modify: `framework/stage6/p6_performance_round_adapter_v1.py`
- Modify: `tests/stage6/test_p6_post_source_adapter_profile.py`
- Modify: `tests/stage6/test_p6_history_registry.py`
- Modify: `tests/stage6/test_p6_performance_round_adapter.py`

**Interfaces:**
- Normalized profile v4 adds `hardware_profile` and validates its target/backend against the registry; v1–v3 load only as legacy H800 where their existing compatibility surface requires it.
- `validate_tvm_measurement_manifest(manifest, *, profile, candidate_id, source_digest)` requires exact profile id, `tvm_arch`, `tvm_cache_namespace`, toolchain id, candidate id, and source digest; it rejects a TensorRT engine marker.

- [ ] Write RED tests for profile v4 round-trip, H800 v3 legacy behavior, RTX target mismatch, wrong `sm90`/cache namespace/source digest/candidate/toolchain manifest, missing manifest as a compile-required cache miss, and TensorRT-marked result rejection.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_post_source_adapter_profile.py tests/stage6/test_p6_history_registry.py tests/stage6/test_p6_performance_round_adapter.py -k 'hardware_profile or tvm_manifest or rtx'
```

Expected: v4/profile-aware TVM evidence is absent.

- [ ] Implement profile-bound serialization and the narrow manifest validator at the existing performance adapter boundary. Pass the profile’s TVM arch/cache namespace only through existing direct-argv leaf binding; unmatched/missing evidence calls the normal compile path and may not reuse another profile’s artifact.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_post_source_adapter_profile.py tests/stage6/test_p6_history_registry.py tests/stage6/test_p6_performance_round_adapter.py
python -m ruff check framework/stage6/p6_post_source_adapter_profile_v1.py framework/stage6/p6_performance_round_adapter_v1.py
```

- [ ] Independent review gate: demonstrate the only cache reuse predicate is full-manifest equality and that no TensorRT path reaches metric parsing.

### Task 6: Profile-Aware Bootstrap, Provision, and Preflight

**Files:**
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `tools/release/provision_p6_full_chain_local_config.py`
- Modify: `tools/release/provision_p6_history_local_config.py`
- Modify: `tools/release/preflight_p6_materializer_training_bridge.py`
- Modify: `tests/release/test_provision_p6_full_chain_local_config.py`
- Modify: `tests/release/test_provision_p6_history_local_config.py`
- Modify: `tests/release/test_preflight_p6_materializer_training_bridge.py`

**Interfaces:**
- Full-chain materialization derives one profile from the public contract and requires local config, binding, source wrapper, and post-source profile agreement before writing either private output.
- Provision/preflight remain CLI-compatible and receive no GPU override flag; the selected profile comes from the public contract.

- [ ] Write RED tests for profile-agreeing RTX fixtures, mismatch rejection before output/process/probe, RTX four-card fake probe admission, preflight accepted report retaining 4 planned rounds and zero process/GPU launches, and legacy H800 CLI fixtures unchanged.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_provision_p6_history_local_config.py tests/release/test_preflight_p6_materializer_training_bridge.py -k 'hardware_profile or rtx'
```

Expected: private pairs lack profile agreement and preflight cannot validate RTX.

- [ ] Implement profile plumbing without adding CLI values; make `PUBLIC_CONTRACT_PATH` resolve the selected example through the validated contract rather than hard-coding an H800 path. Preserve atomic pair semantics, ignored destinations, and preflight `historical_process_launch_count == 0`, `gpu_probe_count == 0`.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_provision_p6_history_local_config.py tests/release/test_preflight_p6_materializer_training_bridge.py
python -m ruff check framework/stage6/p6_full_chain_bootstrap_v1.py tools/release/provision_p6_history_local_config.py tools/release/provision_p6_full_chain_local_config.py tools/release/preflight_p6_materializer_training_bridge.py
```

- [ ] Independent review gate: confirm no profile mismatch reaches a write, a GPU probe, or process launch.

### Task 7: Shared Controller Runner and Verifier

**Files:**
- Modify: `tools/release/run_p6_h800_search.py`
- Modify: `tools/release/verify_p6_materializer_training_run.py`
- Modify: `tests/release/test_run_p6_h800_search.py`
- Modify: `tests/release/test_verify_p6_materializer_training_run.py`

**Interfaces:**
- Existing runner executable loads either v2 H800 or v3 profile-selected contracts; no new RTX runner exists.
- `P6MaterializerCompletionReport` gains public `hardware_profile: str`; `verify_materializer_training_run` verifies all contract/binding/profile consistency before the existing 4-round/16/Gold checks.

- [ ] Write RED tests for a completed fake RTX four-round evidence tree, profile mismatch rejection, legacy H800 CLI/report compatibility, and verifier failure for a completed state with 4/16 but wrong hardware-profile evidence.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/release/test_run_p6_h800_search.py tests/release/test_verify_p6_materializer_training_run.py -k 'hardware_profile or rtx'
```

Expected: report lacks profile provenance and runner/verifier do not cross-check it.

- [ ] Implement profile verification around existing loaders only. Do not alter controller selection, round iteration, row count, metrics, feedback translation, or completion cardinality.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/release/test_run_p6_h800_search.py tests/release/test_verify_p6_materializer_training_run.py
PYTHONPATH=. pytest -q --cov=tools.release.verify_p6_materializer_training_run --cov-branch --cov-fail-under=80 tests/release/test_verify_p6_materializer_training_run.py
```

- [ ] Independent review gate: verify public report includes only the profile id and preserves redaction of private evidence.

### Task 8: RTX Public Examples and Cross-Hardware Reporting Contract

**Files:**
- Create: `configs/execution/p6_rtx4090_search.example.yaml`
- Create: `tests/release/test_p6_rtx4090_execution_contract.py`
- Modify: `tests/release/test_public_execution_surface.py`

**Interfaces:**
- RTX public example is v3, names `hardware_profile: rtx4090`, target `rtx4090`, backend `tvm_auto`, and retains 16/4/4 and the five metrics.
- Completion/report provenance accepts `comparison_scope: "hardware_specific"`; no shared latency/energy frontier or H800 target label is valid for RTX evidence.

- [ ] Write RED tests loading the RTX example, checking it resolves the registry and formal scan config, rejects H800 profile/target swaps, and rejects reports that combine H800 and RTX latency/energy or identify RTX data as H800.
- [ ] Run RED:

```bash
PYTHONPATH=. pytest -q tests/release/test_p6_rtx4090_execution_contract.py tests/release/test_public_execution_surface.py -k 'rtx4090 or hardware_specific'
```

Expected: RTX example and hardware-specific report rule are absent.

- [ ] Add the minimal v3 public example and enforce reporting labels at the public completion/report boundary. AP remains comparable only when external provenance fields match; latency/energy and Pareto values remain profile-specific.
- [ ] Run GREEN:

```bash
PYTHONPATH=. pytest -q tests/release/test_p6_rtx4090_execution_contract.py tests/release/test_public_execution_surface.py
python -m ruff check tests/release/test_p6_rtx4090_execution_contract.py
```

- [ ] Independent review gate: verify no private locator, GPU policy, paths, raw metrics, or remote endpoint appears in the committed example.

### Task 9: Full Regression and Readiness Review

**Files:**
- Modify only if earlier tests require exact fixture updates: the files named in Tasks 1–8.

- [ ] Run targeted full P6 regression:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_hardware_execution_profile.py tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_fresh_run_controller.py tests/stage6/test_p6_hardware_profile_stage1_plan.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_p6_post_source_adapter_profile.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_preflight_p6_materializer_training_bridge.py tests/release/test_run_p6_h800_search.py tests/release/test_verify_p6_materializer_training_run.py tests/release/test_p6_rtx4090_execution_contract.py tests/release/test_public_execution_surface.py
```

- [ ] Run changed-module lint and coverage at or above 80% branch coverage, then inspect `git diff --check` and the diff for copied controllers/verifiers, proxy metrics, GPU IDs, private paths, or TensorRT substitution.
- [ ] Review readiness evidence before any real run: scanner-generated RTX Stage1/Stage2 plan only; fresh ignored output roots; profile-consistent private binding; exactly two live snapshots at provision; preflight process/GPU counts 0/0; and verifier expected 4/16/zero-Gold checks.
- [ ] Do not start controller/measurement in this task. The user's approved real attempt begins only in Task 10, after this review passes.

### Task 10: Fresh Four-GPU RTX4090 Real Run

**Files:**
- Create only ignored private runtime inputs, locator, evidence, and output roots under the existing deployment workspace.
- Do not add another tracked controller, verifier, round adapter, or measurement implementation.

**Interfaces:**
- Select four currently admitted local RTX4090 cards through a private ordered policy; do not hard-code their indices in tracked files.
- Use the v3 RTX4090 public contract, the existing provision/preflight/controller entrypoints, and fresh output destinations.

- [ ] Reconcile the reviewed clean HEAD, private input modes, fresh-root absence, RTX profile agreement, scanner-derived plan, and zero relevant process/output state.
- [ ] Take the two live snapshots required by the existing provision boundary. Require stable ordered identity, four exact allowed RTX4090 models, occupancy at or below the selected profile threshold, and exactly one binding/config pair publication.
- [ ] Run official preflight and require four planned rounds, training required, historical process launch count zero, and GPU probe count zero.
- [ ] Launch the existing controller exactly once. Run four rounds with four candidates assigned one-per-GPU, retaining real training, quantization, TVM-SM89 performance, AP, finalization, and feedback evidence.
- [ ] Monitor without relaunching the controller. Ordinary software/path/build errors remain in-scope debugging work: diagnose, add a failing test, apply the smallest profile-bound fix, rerun focused verification, and resume through a fresh attempt when required. Stop only for destructive/external interference, unresolvable missing authority/data, or hardware unavailability that prevents truthful measurement.
- [ ] Require terminal state: four completed rounds, 16 unique measured candidates, 16 `measured_success_gold` rows, finite five-metric values, Gold176 overlap/remeasurement zero, no active experiment process, and no proxy/synthetic/manual rows.

### Task 11: Existing Verifier and RTX4090 Result Report

**Files:**
- Use the existing verifier and ignored real-run evidence.
- Create a user-facing result summary in the repository's existing experiment-report location; do not expose private UUIDs, paths, commands, or raw private values.

**Interfaces:**
- The existing verifier consumes the completed RTX4090 run; no RTX-specific verifier is permitted.
- The report labels the run `rtx4090`, `sm89`, and `hardware_specific` and includes reproducibility provenance.

- [ ] Run the unchanged-mechanics verifier against the fresh completion and require the selected profile, 4/16 cardinality, Gold176 zero-overlap, five real metrics, and TVM manifest consistency.
- [ ] Verify every reused TVM artifact has an exact candidate/q-mode/config/checkpoint/code/profile/SM89/toolchain manifest match. Record mismatches as compiled cache misses; never count TensorRT `.engine` files as TVM evidence.
- [ ] Produce the final RTX4090 metric table, per-round progress, selected/Pareto candidates, total run status, compiler/environment fingerprints, seeds/data/checkpoint protocol identities, and comparison limitations.
- [ ] Compare AP only under matching evaluation provenance. Keep latency, energy, rankings, and Pareto conclusions hardware-specific; do not relabel or numerically adjust results to mimic H800/COptV2X paper values.
- [ ] Run final regression, Ruff, coverage (at least 80% for changed modules), `git diff --check`, independent whole-branch code review, and completion verification before reporting success.
