# Task 5 report — AP Plan, Sanity, and Full Adapter

## Summary

Implemented Task 5 AP adapter and release CLI:

- `framework/stage6/p6_ap_round_adapter_v1.py`
- `tools/release/run_p6_ap_round_adapter.py`
- `tests/stage6/test_p6_ap_round_adapter.py`
- `tests/release/test_run_p6_ap_round_adapter.py`

The adapter calls the native AP planner once, validates native `stage5_ap_plan_v2` plan shape and request row identity, shards rows by `row_index % len(ordered GPUs)`, runs one AP executor shard per GPU for sanity, waits for all sanity shards, runs full shards, merges shard states by shard index through sibling temp + `os.replace`, and advances task state from `performance` to `ap` only after native AP completeness.

## Native semantics preserved

- Native AP plan field names/statuses used directly:
  - `jobs`
  - `manifest_job_id`
  - `ap_terminal: ready`
  - `blocked_performance_not_success`
  - other non-ready `blocked_*` / feasibility terminals treated as not requiring AP full
- Native numerical feasibility semantics preserved:
  - sanity terminal: `status: failed`, `failure_reason: numerical_feasibility_failure`
  - full terminal: `status: skipped_numerical_feasibility`, `failure_reason: numerical_feasibility_failure`
- Ready rows require sanity success + full success, unless numerical-feasibility skip is present.
- Performance-blocked rows do not require AP full.
- Empty merged AP state is accepted when all rows are native non-ready/blocked.

## TDD evidence

RED:

```text
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> failed during collection with ModuleNotFoundError: framework.stage6.p6_ap_round_adapter_v1
```

Additional RED:

```text
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_accepts_all_performance_blocked_native_rows
=> failed with P6APRoundAdapterError before empty-state validation was adjusted
```

GREEN:

```text
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> 10 passed in 5.24s
```

Coverage workaround:

```text
PYTHONPATH=. coverage run --branch -m pytest -q tests/stage6/test_p6_ap_round_adapter.py && coverage report --include='framework/stage6/p6_ap_round_adapter_v1.py' --fail-under=80
=> 8 passed in 2.30s
=> framework/stage6/p6_ap_round_adapter_v1.py: 83%
```

Syntax/shape:

```text
python -m py_compile framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> exit 0

Task 5 files function-size check
=> no functions > 50 lines
```

## Coverage command caveat

The requested pytest-cov command still fails before collection:

```text
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_ap_round_adapter_v1 --cov-branch --cov-report=term-missing --cov-fail-under=80 tests/stage6/test_p6_ap_round_adapter.py
=> ImportError: cannot load module more than once per process
```

The failure path is through `framework/stage6/__init__.py -> coptv2x_h800_search_v2 -> framework.stage5.production_search_v1 -> numpy`, and the same failure reproduces for the existing performance adapter pytest-cov target:

```text
PYTHONPATH=. pytest -q --cov=framework.stage6.p6_performance_round_adapter_v1 --cov-branch --cov-report=term-missing --cov-fail-under=80 tests/stage6/test_p6_performance_round_adapter.py
=> same NumPy import loader error
```

I did not change `framework/stage6/__init__.py` because Task 5 ownership was limited to Task 5 files.

## Commit

Prepared commit message:

```text
feat: add P6 AP round adapter
```

## NEEDS_EVIDENCE addendum before review

Exact committed SHA:

```text
28c628807bc79965a7eed62530b6b761d09b3d92
```

### Mandatory equivalent pre-import pytest-cov form

Command:

```bash
PYTHONPATH=. python - <<'PY'
import framework.stage6.coptv2x_h800_search_v2
import pytest
raise SystemExit(pytest.main([
    '-q',
    '--cov=framework.stage6.p6_ap_round_adapter_v1',
    '--cov-branch',
    '--cov-report=term-missing',
    '--cov-fail-under=80',
    'tests/stage6/test_p6_ap_round_adapter.py',
]))
PY
```

Full output:

```text
........                                                                 [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.12-final-0 _______________

Name                                         Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------------------------------
framework/stage6/p6_ap_round_adapter_v1.py     174     22     54     16    83%   59-60, 158, 175, 177, 186, 189->184, 222->220, 260, 270-271, 277, 283, 285-286, 288, 295, 297, 299-300, 302, 315, 318, 325
----------------------------------------------------------------------------------------
TOTAL                                          174     22     54     16    83%
Required test coverage of 80% reached. Total coverage: 83.33%
8 passed in 0.44s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-97eb9c49-8274-4a41-992e-42cae48532c9/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-97eb9c49-8274-4a41-992e-42cae48532c9/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-97eb9c49-8274-4a41-992e-42cae48532c9/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-97eb9c49-8274-4a41-992e-42cae48532c9/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-97eb9c49-8274-4a41-992e-42cae48532c9
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-97eb9c49-8274-4a41-992e-42cae48532c9'
  warnings.warn(
```

### Ruff on all Task5 owned files

Command:

```bash
python -m ruff check framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
All checks passed!
```

### compileall and git diff check

Command:

```bash
python -m compileall -q framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
```

Command:

```bash
git diff --check
```

Full output:

```text
```

### Fresh focused rerun

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
..........                                                               [100%]
10 passed in 5.28s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-d911f46f-b509-4afa-89bf-397d77053cae/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-d911f46f-b509-4afa-89bf-397d77053cae/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-d911f46f-b509-4afa-89bf-397d77053cae/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-d911f46f-b509-4afa-89bf-397d77053cae/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-d911f46f-b509-4afa-89bf-397d77053cae
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-d911f46f-b509-4afa-89bf-397d77053cae'
  warnings.warn(
```

## Fix round 1/5 — native planner rc=1 and AP evidence binding

### Root cause

Native `stage5_ap_plan_v2.py` writes `ap_plan.json` / `ap_plan.jsonl` and returns `1` whenever any row has non-ready `ap_terminal`. The adapter treated every planner nonzero as a leaf failure, so valid all-blocked/mixed native plans could not advance. The adapter also accepted AP terminal state based only on `job_id`, `stage`, and `status`, but native `stage3_execute_ap_plan_v3.py` binds terminal rows to `plan_fingerprint` and report evidence, and Stage5 finalization requires real report files/SHA, with full numerical-feasibility skip copying the failed sanity report evidence.

### RED evidence

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
..FF......FFFFFF.......                                                  [100%]
=================================== FAILURES ===================================
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_accepts_all_performance_blocked_native_rows
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_accepts_planner_rc1_for_mixed_native_nonready_plan
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_inconsistent_or_failed_planner_results[<lambda>6]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_stale_or_unbound_native_state_evidence[<lambda>0]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_stale_or_unbound_native_state_evidence[<lambda>1]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_stale_or_unbound_native_state_evidence[<lambda>2]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_stale_or_unbound_native_state_evidence[<lambda>3]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_stale_or_unbound_native_state_evidence[<lambda>4]
8 failed, 15 passed in 5.88s
```

### GREEN focused rerun

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
.......................                                                  [100%]
23 passed in 5.59s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-b2538865-321e-416d-8ac4-d967c4bd4b9c/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-b2538865-321e-416d-8ac4-d967c4bd4b9c/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-b2538865-321e-416d-8ac4-d967c4bd4b9c/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-b2538865-321e-416d-8ac4-d967c4bd4b9c/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-b2538865-321e-416d-8ac4-d967c4bd4b9c
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-b2538865-321e-416d-8ac4-d967c4bd4b9c'
  warnings.warn(
```

### Pre-import branch coverage

Command:

```bash
PYTHONPATH=. python - <<'PY'
import framework.stage6.coptv2x_h800_search_v2
import pytest
raise SystemExit(pytest.main([
    '-q',
    '--cov=framework.stage6.p6_ap_round_adapter_v1',
    '--cov-branch',
    '--cov-report=term-missing',
    '--cov-fail-under=80',
    'tests/stage6/test_p6_ap_round_adapter.py',
]))
PY
```

Full output:

```text
.....................                                                    [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.12-final-0 _______________

Name                                         Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------------------------------
framework/stage6/p6_ap_round_adapter_v1.py     260     26     92     19    87%   72-73, 183, 227, 242, 245->240, 304, 359, 369-370, 376, 387, 394, 396, 398-399, 401, 407, 410, 417, 442-443, 455, 458, 465, 484-485
----------------------------------------------------------------------------------------
TOTAL                                          260     26     92     19    87%
Required test coverage of 80% reached. Total coverage: 87.22%
21 passed in 0.93s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-8cbe72dd-3926-4b3d-8fc9-9b46dd5bc249/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-8cbe72dd-3926-4b3d-8fc9-9b46dd5bc249/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-8cbe72dd-3926-4b3d-8fc9-9b46dd5bc249/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-8cbe72dd-3926-4b3d-8fc9-9b46dd5bc249/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-8cbe72dd-3926-4b3d-8fc9-9b46dd5bc249
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-8cbe72dd-3926-4b3d-8fc9-9b46dd5bc249'
  warnings.warn(
```

### Static/shape gates

Ruff:

```text
python -m ruff check framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> All checks passed!
```

compileall:

```text
python -m compileall -q framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> exit 0
```

git diff check:

```text
git diff --check
=> exit 0
```

Function size:

```text
framework/stage6/p6_ap_round_adapter_v1.py long_functions= []
tools/release/run_p6_ap_round_adapter.py long_functions= []
tests/stage6/test_p6_ap_round_adapter.py long_functions= []
tests/release/test_run_p6_ap_round_adapter.py long_functions= []
```

## Fix round 2/5 — bound full command fingerprint and stale AP output preflight

### Root cause

Native `stage3_execute_ap_plan_v3.py` binds `full_command_state_bindings` before executing full AP, then computes `plan_fingerprint(job, "full")` from that bound full job. The adapter was validating full success terminals against the unbound original plan. Separately, AP planner rc=1 could be paired with stale preexisting `ap_plan*` / `ap_state*` outputs because the adapter did not require adapter-owned AP output targets to be absent before planner invocation.

### RED evidence

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
................FFFFF.......                                             [100%]
=================================== FAILURES ===================================
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_accepts_bound_full_command_state_fingerprint
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_preexisting_ap_outputs_before_planner[<lambda>0]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_preexisting_ap_outputs_before_planner[<lambda>1]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_preexisting_ap_outputs_before_planner[<lambda>2]
FAILED tests/stage6/test_p6_ap_round_adapter.py::test_ap_round_rejects_preexisting_ap_outputs_before_planner[<lambda>3]
5 failed, 23 passed in 6.39s
```

### GREEN focused rerun

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
```

Full output:

```text
............................                                             [100%]
28 passed in 5.87s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-c6b3770d-ee67-40c7-b0dd-d1d2f91759c9/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-c6b3770d-ee67-40c7-b0dd-d1d2f91759c9/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-c6b3770d-ee67-40c7-b0dd-d1d2f91759c9/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-c6b3770d-ee67-40c7-b0dd-d1d2f91759c9/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-c6b3770d-ee67-40c7-b0dd-d1d2f91759c9
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-c6b3770d-ee67-40c7-b0dd-d1d2f91759c9'
  warnings.warn(
```

### Pre-import branch coverage

Command:

```bash
PYTHONPATH=. python - <<'PY'
import framework.stage6.coptv2x_h800_search_v2
import pytest
raise SystemExit(pytest.main([
    '-q',
    '--cov=framework.stage6.p6_ap_round_adapter_v1',
    '--cov-branch',
    '--cov-report=term-missing',
    '--cov-fail-under=80',
    'tests/stage6/test_p6_ap_round_adapter.py',
]))
PY
```

Full output:

```text
..........................                                               [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.12-final-0 _______________

Name                                         Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------------------------------
framework/stage6/p6_ap_round_adapter_v1.py     308     36    116     27    85%   73-74, 184, 228, 235, 247, 250->245, 309, 324, 337, 342, 345->347, 356, 420, 430-431, 437, 448, 455, 457, 459-460, 462, 468, 471, 478, 503-504, 510-513, 515, 528, 531, 538, 557-558
----------------------------------------------------------------------------------------
TOTAL                                          308     36    116     27    85%
Required test coverage of 80% reached. Total coverage: 84.67%
26 passed in 1.10s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-c74b49ad-2d90-47bb-a37f-cf6561bb4d6f/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-c74b49ad-2d90-47bb-a37f-cf6561bb4d6f/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-c74b49ad-2d90-47bb-a37f-cf6561bb4d6f/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-c74b49ad-2d90-47bb-a37f-cf6561bb4d6f/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-c74b49ad-2d90-47bb-a37f-cf6561bb4d6f
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-c74b49ad-2d90-47bb-a37f-cf6561bb4d6f'
  warnings.warn(
```

### Static/shape gates

Ruff:

```text
python -m ruff check framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> All checks passed!
```

compileall:

```text
python -m compileall -q framework/stage6/p6_ap_round_adapter_v1.py tools/release/run_p6_ap_round_adapter.py tests/stage6/test_p6_ap_round_adapter.py tests/release/test_run_p6_ap_round_adapter.py
=> exit 0
```

git diff check:

```text
git diff --check
=> exit 0
```

Function size:

```text
framework/stage6/p6_ap_round_adapter_v1.py long_functions= []
tools/release/run_p6_ap_round_adapter.py long_functions= []
tests/stage6/test_p6_ap_round_adapter.py long_functions= []
tests/release/test_run_p6_ap_round_adapter.py long_functions= []
```
