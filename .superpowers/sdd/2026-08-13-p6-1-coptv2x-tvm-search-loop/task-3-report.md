# Task 3 implementation report

## Status

Completed. The P6 controller now treats `feasibility_failure` and
`numerical_feasibility_failure` as request-bound, budget-consuming candidate
feedback. Malformed feedback batches are quarantined atomically: no rows from
the rejected batch enter online history, the round does not advance, and the
measured-candidate count is unchanged.

## Implementation

- Added controller-local terminal-status and feedback-schema constants.
- Replaced the Task 2 success-only feedback finalizer with an atomic release
  path that checks schema, request digest, exact row identities/cardinality,
  five finite physical metrics for successful rows, and terminal status.
- Preserved the stricter Task 2 metric rules: latency and energy remain
  positive, and AP30/AP50/AP70 remain finite values in `[0, 1]`.
- Accepted only the two true candidate failure statuses. Their non-empty
  `failure_reason` is retained only when it matches `[a-z0-9_-]+`; otherwise
  the released row stores `unspecified`.
- Added recoverable round failure handling. It writes only
  `schema_version`, `failure_code`, and `completed_rounds` to the local
  `round-XX/failure.json`, then writes the redacted failed local state.
- Added stable mappings for missing/invalid source registries and unreadable
  or invalid-JSON controller inputs. Existing closure and structural contract
  errors retain their Task 2 contract-error behavior.
- Preserved Task 2's round-zero Gold176 single-target cold start, later online
  fitting, 343/686 gates, Gold holdout, and local output leaf validation. No
  legacy four-arm fitter, GPU/asset/network execution, CLI work, or ancestor
  symlink/fd-level TOCTOU mechanism was introduced.

`framework/stage5/single_target_search_v2.py` needed no change: its existing
feedback-history validator already accepts the two true failure statuses, and
its online fitter already filters training rows to `measured_success_gold`.
That existing behavior is compatible with this task's requirement that failure
rows consume selection budget without becoming metric-training evidence.

## TDD evidence

### RED: true failures and malformed feedback isolation

```text
PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback tests/stage6/test_coptv2x_h800_search.py::test_run_p6_quarantines_invalid_feedback_batches -q
9 failed in 8.17s
```

The former success-only finalizer rejected both true failure statuses. The
seven malformed feedback variants escaped as contract errors rather than
returning a failed run state with a stable code.

### GREEN: true failures and malformed feedback isolation

```text
PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback tests/stage6/test_coptv2x_h800_search.py::test_run_p6_quarantines_invalid_feedback_batches -q
9 passed in 11.91s
```

### RED/GREEN: stable source and local-input codes

```text
PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_rejects_incomplete_source_space tests/stage6/test_coptv2x_h800_search.py::test_run_p6_reports_stable_source_registry_failures tests/stage6/test_coptv2x_h800_search.py::test_run_p6_reports_invalid_local_input_with_a_stable_code -q
4 failed in 1.59s
```

```text
PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_rejects_incomplete_source_space tests/stage6/test_coptv2x_h800_search.py::test_run_p6_reports_stable_source_registry_failures tests/stage6/test_coptv2x_h800_search.py::test_run_p6_reports_invalid_local_input_with_a_stable_code -q
4 passed in 1.36s
```

## Final verification

```text
PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py tests/stage5/test_single_target_search.py -k 'not cli' -q
95 passed, 13 deselected in 26.36s
```

```text
PYTHONPATH=. pytest tests/stage5/test_production_search.py tests/stage6 -q
183 passed in 37.56s
```

```text
python -m ruff check framework/stage5/single_target_search_v2.py framework/stage6/coptv2x_h800_search_v2.py tests/stage5/test_single_target_search.py tests/stage6/test_coptv2x_h800_search.py
All checks passed!

python -m compileall -q framework/stage5 framework/stage6 tests/stage5 tests/stage6
exit 0

git diff --check
exit 0
```

An explicit full run of `tests/stage5/test_single_target_search.py` has 30
passing and 12 failing CLI cases. Every failing case stops at the existing
`source identity verification failed` gate before its intended assertion. This
is the known Task 4 CLI module-identity boundary recorded in Task 2; Task 3
does not modify `scripts/reproduce/stage5_selection.py` or its identity pin.

## Review

An independent scoped review found no correctness, security, or regression
issues. It confirmed atomic release before history mutation, failure-row
exclusion from metric fitting, stable source/local error mappings, and the
redacted `failure.json` shape.
