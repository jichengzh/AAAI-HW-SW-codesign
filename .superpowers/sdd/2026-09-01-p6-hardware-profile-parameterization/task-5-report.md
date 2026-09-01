# Task 5 report: Normalized Profiles and Compile-Bound TVM Evidence

## Delivered

- `p6_post_source_adapter_profile_v4` binds the post-source target to a
  registry-owned hardware profile. Existing v1-v3 profiles remain legacy H800.
- Current v5 normalization accepts an optional exact `hardware_profile` id.
  Omission still emits profile v3/H800; explicit `rtx4090` is loaded from the
  canonical registry, carried as that registry object through canonical
  normalization, and emits profile v4. Unknown ids, forged objects, and a
  hardware field on an older source-map schema fail closed.
- The performance adapter now passes `--hardware-profile`, `--tvm-arch`, and
  `--tvm-cache-namespace` through the existing direct-argv planner leaf for v4.
  Legacy v1-v3 planner argv remains unchanged.
- The historical planner parser accepts those three arguments with H800
  defaults. The measurement plan validates rows against the exact selected
  hardware target and rejects mixed H800/RTX or noncanonical architecture and
  namespace labels. Every newly planned TVM job carries an exact expected
  manifest bound to `manifest_job_id` and the validated row source digest.
- The historical executor still runs the real subprocess. Only after a zero
  return code, success/correctness checks, and finite latency/energy checks does
  it reject a conflicting runner-supplied manifest or atomically inject the
  exact job manifest into the real result JSON. The result is hashed after that
  write. Jobs without an expected manifest retain their prior behavior.
- The current adapter validates both the planned expected manifest and the
  successful result manifest against the canonical profile and request row.
  Missing or mismatched evidence on a successful v4 result fails before metric
  parsing. TensorRT runner keys and recursive engine markers remain excluded.

## Current cache behavior

There is no implemented artifact-reuse controller in this path. Current TVM
runners compile fresh, and the historical planner/executor attaches evidence to
that fresh compile result. `validate_tvm_measurement_manifest(...)` is an exact
evidence-acceptance predicate: missing or mismatched evidence returns `False`,
so an old unmanifested `.so` or TensorRT artifact is not accepted as a measured
success. This task does not claim or fabricate a cache hit, bypass a subprocess,
or introduce a reuse path.

## Strict RED -> GREEN evidence

Initial Task 5 RED:

```text
9 failed, 4 passed, 102 deselected
```

Initial focused GREEN:

```text
13 passed, 102 deselected
```

Fix round parent-repository RED contained four expected real-boundary failures:

1. the planner argparse rejected profile/arch/namespace;
2. the measurement plan had no parameter-derived RTX target;
3. a successful real subprocess result was not given its job manifest; and
4. a conflicting result manifest was accepted.

```text
4 failed, 16 deselected
```

The four tests then passed. A separate real CLI test demonstrated that a
self-consistent but noncanonical `rtx4090`/`sm90` tuple was accepted:

```text
1 failed, 4 deselected -> 1 passed, 4 deselected
```

Fix round current-worktree RED demonstrated both missing integration links:

```text
2 failed, 68 deselected
```

The failures were v5 explicit RTX being rejected during recipe canonicalization
and the adapter omitting `--hardware-profile`. Both passed after the minimal
normalization/staging and adapter changes.

## Verification

Parent historical focused plus adjacent Stage5/fixed-batch suites:

```text
25 passed in 0.87s
```

The 21-test focused parent branch-coverage run recorded 76% across the existing full
historical planner and executor files. Ruff (excluding the two pre-existing
`EXE001`/`SIM117` findings), compilation, and scoped `git diff --check` passed.
An attempted broader Stage7 physical-execution run stopped in its deployment
fixture because unrelated dirty parent files no longer match its reviewed
source-SHA table; it fails before reaching Task 5 planner/executor behavior.

Current worktree expanded compatibility and branch-coverage suite:

```text
297 passed in 52.49s
normalization 82%, staging 83%, feedback validation 88%, registry 83%,
performance adapter 78%, post-source profile 77%, aggregate branch coverage 81%
```

Pytest continues to emit the pre-existing temporary-directory cleanup warning
after successful current-worktree runs. It does not change the exit status.

## Commits

- Parent historical-authority repository:
  `0fb3585d96f39ce4a6324bd72b2498a37c50e9b0`
  (`feat: bind historical TVM results to hardware profiles`).
- Initial current-worktree Task 5 commit:
  `213e73c058bc1de03e56761acd0c5cbd5cf1c06a`.
- The exact current fix-round commit hash is included in the final handoff. It
  cannot be embedded in this report inside that same commit because changing
  the report changes the Git commit hash.

## Fix-round files changed

Parent historical-authority repository (path redacted):

- `framework/stage5/measurement_plan_v2.py`
- `scripts/stage5_build_performance_plan_v2.py`
- `scripts/stage3_execute_performance_plan_v3.py`
- `framework/tests/test_stage5_measurement_plan_v2.py`
- `framework/tests/test_stage3_execute_performance_plan_v3.py`

Current worktree:

- `framework/stage6/p6_history_normalization_v1.py`
- `framework/stage6/p6_history_normalization_staging_v1.py`
- `framework/stage6/p6_performance_round_adapter_v1.py`
- `tests/stage6/test_p6_post_source_adapter_profile.py`
- `tests/stage6/test_p6_performance_round_adapter.py`
- `.superpowers/sdd/2026-09-01-p6-hardware-profile-parameterization/task-5-report.md`
