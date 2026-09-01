# Task 5 report: Normalized Profiles and TVM Cache Evidence

## Delivered

- Added `p6_post_source_adapter_profile_v4`, which extends the existing v3
  dependency-overlay profile with the canonical `hardware_profile` identifier.
  Its serialized target is derived from the registry-owned profile, and the
  loader requires exact target/backend/profile agreement. Existing v1-v3
  profiles remain H800-only and continue to serialize without the new key.
- Added canonical hardware profile persistence to the materialized Stage5
  source registry.
- Added `validate_tvm_measurement_manifest(...)` as the single cache-reuse
  predicate. Reuse requires exact equality of schema, hardware profile, TVM
  architecture, cache namespace, TVM toolchain, candidate identity, and source
  digest. Missing or mismatched evidence returns `False`, so it is a
  compile-required miss rather than reusable success.
- Bound v4 performance planner invocations to the selected profile's
  `--tvm-arch` and `--tvm-cache-namespace` through the existing direct-argv
  leaf call. Legacy v1-v3 argv remains unchanged.
- Kept native jobs and executor mechanics unchanged. The adapter carries the
  already-validated row candidate/source identity beside native jobs only for
  result-manifest validation.
- Removed TensorRT dispatch from the performance adapter's accepted runner-key
  mapping. TensorRT/engine markers are recursively rejected before latency or
  energy parsing, including nested engine paths.
- Added canonical-registry validation before a v4 planner launch, so a forged
  `HardwareExecutionProfile` cannot pass architecture/cache values to a leaf.

## Strict RED -> GREEN evidence

The exact focused RED command from the brief was run after adding the tests:

```bash
PYTHONPATH=. pytest -q \
  tests/stage6/test_p6_post_source_adapter_profile.py \
  tests/stage6/test_p6_history_registry.py \
  tests/stage6/test_p6_performance_round_adapter.py \
  -k 'hardware_profile or tvm_manifest or rtx'
```

Observed RED: `9 failed, 4 passed, 102 deselected`. Failures were the expected
missing v4 dataclass/schema field, absent registry profile persistence, absent
manifest predicate, and absent RTX adapter binding. There were no collection or
fixture errors.

After the minimal implementation, the same command passed:

```text
13 passed, 102 deselected
```

The security review added two more focused tests. Before the hardening change,
the nested TensorRT marker and forged registry object tests both failed. After
recursive marker rejection and pre-launch registry validation:

```text
2 passed, 31 deselected
```

## Cache-reuse and metric-order review gate

- `validate_tvm_measurement_manifest` constructs one literal expected mapping
  and returns `True` only when both the key set and complete mapping are equal.
  Wrong `sm90`, H800 cache namespace, source digest, candidate, toolchain, or
  any extra field returns `False`; a missing manifest also returns `False`.
- A v4 successful native state calls the manifest predicate before
  `_validate_success_result_payload`. A false result raises the stable adapter
  error before either metric extractor.
- TensorRT/engine markers are rejected recursively at both manifest and result
  boundaries. Tests replace the latency parser with a counter and prove zero
  calls for missing manifests and TensorRT-marked results.
- `_native_runner_key` accepts only the two existing TVM runner keys. No
  TensorRT result candidate can be selected through a validated request.
- No compiler/controller abstraction, synthetic metric, alternate scheduler,
  shell execution, GPU action, remote call, or private runtime was introduced.

## Verification

Full assigned suites on the final tree:

```text
117 passed in 18.46s
```

Fresh expanded compatibility and branch-coverage run (assigned suites plus
finalization, history measurement, post-source wrapper, and full-chain
bootstrap):

```text
256 passed in 40.72s
feedback validation 88%, registry 83%, performance adapter 78%,
post-source profile 77%, aggregate branch coverage 80%
```

Ruff passed on all seven changed Python files, including the exact two-file
command from the brief. `git diff --check` passed.

Pytest continues to emit a pre-existing temporary-directory cleanup warning
after otherwise successful runs. It was present on the clean baseline and does
not change test exit status or Task 5 evidence.

## Files changed

- `framework/stage6/p6_post_source_adapter_profile_v1.py`
- `framework/stage6/p6_history_registry_v1.py`
- `framework/stage6/p6_history_feedback_validation_v1.py`
- `framework/stage6/p6_performance_round_adapter_v1.py`
- `tests/stage6/test_p6_post_source_adapter_profile.py`
- `tests/stage6/test_p6_history_registry.py`
- `tests/stage6/test_p6_performance_round_adapter.py`
