# Task 4 Report: Performance Plan and Execution Adapter

## Status

Implemented Task 4 in commit:

```text
787e80d feat: add P6 performance round adapter
```

No additional implementation commit was needed after the required evidence rerun because the mandatory equivalent branch coverage reached the 80% threshold.

## Files

Created:

- `framework/stage6/p6_performance_round_adapter_v1.py`
- `tools/release/run_p6_performance_round_adapter.py`
- `tests/stage6/test_p6_performance_round_adapter.py`
- `tests/release/test_run_p6_performance_round_adapter.py`

Report-only evidence file:

- `.superpowers/sdd/2026-08-25-p6-post-source-round-adapters/task-4-report.md`

## Behavior implemented

- Added `run_performance_round(profile, task_state, round_root, runner) -> None`.
- Loads the shared Task 3 `RoundContext` with prior stage `quantization`.
- Resolves exactly two native historical leaves:
  - `performance_plan`
  - `performance_execute`
- Runs planner exactly once before executor.
- Uses direct argv execution through the injected `LeafRunner`; `shell=False`.
- Builds deterministic child environments using the Task 3 env/path binding pattern:
  - ordered `CUDA_VISIBLE_DEVICES` CSV remains unchanged for both leaves
  - deterministic `PATH`
  - deterministic `PYTHONPATH`
  - validated `P6_HISTORY_*` bindings
- Planner argv follows the historical parser:
  - `--request-json`
  - `--remote-artifact-root`
  - `--output-dir`
  - `--quant-contract-root`
  - `--gpus`
- The optional `--tvm-fp16-max-trials` flag is intentionally omitted so the native default policy is preserved.
- Executor argv follows the historical parser:
  - `--jobs-jsonl`
  - `--state-jsonl`
  - `--gpus`
  - `--max-workers`
- `max-workers` is set to `len(gpus)`.
- Native outputs are under `performance/`:
  - `performance_manifest.json`
  - `performance_jobs.jsonl`
  - `performance_state.jsonl`
- State advances from `quantization` to `performance` only after:
  - planner exits zero
  - native manifest identities match the four requested rows
  - native job identities match the four requested rows
  - executor exits zero
  - every native executor state row is terminal: `success` or `confirmed_failure`
- Rejects nonterminal/native incomplete rows such as `ready`, `missing`, and `failed`.
- Rejects planner/executor nonzero exits without advancing task state.
- Rejects missing native leaf bindings without advancing task state.
- No proxy metrics are produced or accepted.
- No SSH, GPU, real Stage1/controller/training/measurement execution was performed.

## TDD evidence

### RED

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Observed failure after tests were in the target worktree and before production implementation:

```text
==================================== ERRORS ====================================
______ ERROR collecting tests/stage6/test_p6_performance_round_adapter.py ______
ImportError while importing test module '/home/jichengzhi/V2X/github/stage1-model-scanner-aaai/.worktrees/p6-materializer-training-bridge/tests/stage6/test_p6_performance_round_adapter.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../../miniconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/stage6/test_p6_performance_round_adapter.py:12: in <module>
    from framework.stage6.p6_performance_round_adapter_v1 import (
E   ModuleNotFoundError: No module named 'framework.stage6.p6_performance_round_adapter_v1'
=========================== short test summary info ============================
ERROR tests/stage6/test_p6_performance_round_adapter.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.45s
```

### GREEN

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Fresh output:

```text
...........                                                              [100%]
11 passed in 4.72s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-9d64feb6-085b-4767-a39f-c485934884ac/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-9d64feb6-085b-4767-a39f-c485934884ac/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-9d64feb6-085b-4767-a39f-c485934884ac/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-9d64feb6-085b-4767-a39f-c485934884ac/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-9d64feb6-085b-4767-a39f-c485934884ac
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-9d64feb6-085b-4767-a39f-c485934884ac'
  warnings.warn(
```

## Mandatory equivalent coverage evidence

The direct requested `pytest --cov=framework.stage6.p6_performance_round_adapter_v1 ...` form previously failed during collection because pytest-cov pre-imported `framework.stage6`, whose package initializer imports NumPy-heavy search code. The mandatory equivalent pre-import form requested for this evidence run was used instead.

Command:

```bash
python - <<'PY'
import framework.stage6.coptv2x_h800_search_v2  # noqa: F401
import pytest
raise SystemExit(pytest.main([
    '-q',
    '--cov=framework.stage6.p6_performance_round_adapter_v1',
    '--cov-branch',
    '--cov-report=term-missing',
    '--cov-fail-under=80',
    'tests/stage6/test_p6_performance_round_adapter.py',
]))
PY
```

Output:

```text
.........                                                                [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.12-final-0 _______________

Name                                                  Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------------------
framework/stage6/p6_performance_round_adapter_v1.py     116     19     34     11    80%   58-59, 128, 132, 166, 171-172, 178, 185, 187-188, 190, 197, 203-204, 206, 212, 218, 221
-------------------------------------------------------------------------------------------------
TOTAL                                                   116     19     34     11    80%
Required test coverage of 80% reached. Total coverage: 80.00%
9 passed in 0.10s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-bb1db75b-149d-482a-8f29-c1cf6b7ca360/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-bb1db75b-149d-482a-8f29-c1cf6b7ca360/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-bb1db75b-149d-482a-8f29-c1cf6b7ca360/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-bb1db75b-149d-482a-8f29-c1cf6b7ca360/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-bb1db75b-149d-482a-8f29-c1cf6b7ca360
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-bb1db75b-149d-482a-8f29-c1cf6b7ca360'
  warnings.warn(
```

Per-module branch result:

- `framework/stage6/p6_performance_round_adapter_v1.py`: 116 statements, 19 missed, 34 branches, 11 partial branches, 80% coverage.

## Ruff evidence

Command:

```bash
python -m ruff check framework/stage6/p6_performance_round_adapter_v1.py tools/release/run_p6_performance_round_adapter.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Output:

```text
All checks passed!
```

## compileall evidence

Command:

```bash
python -m compileall -q framework/stage6/p6_performance_round_adapter_v1.py tools/release/run_p6_performance_round_adapter.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Output:

```text
<no output; exit code 0>
```

## git diff check evidence

Command:

```bash
git diff --check
```

Output:

```text
<no output; exit code 0>
```

## Self-review

- Ownership stayed within Task 4 files plus this required report.
- The adapter uses the shared Task 3 runtime API: `RoundContext`, `LeafRunner`, `advance_task_state`, and `load_round_context`.
- The adapter reuses the deterministic env/path binding pattern from Task 3 without broadening the execution boundary.
- Planner and executor use the authoritative historical CLI parser flags inspected from:
  - `/home/jichengzhi/V2X/scripts/stage5_build_performance_plan_v2.py`
  - `/home/jichengzhi/V2X/scripts/stage3_execute_performance_plan_v3.py`
- The optional TVM FP16 trial flag is omitted.
- State remains `quantization` on planner failure, executor failure, incomplete native state, identity drift, and missing leaf bindings.
- State advances to `performance` only after native plan and state completeness validate.
- Tests cover planner-before-executor order, exact argv, unchanged ordered GPU CSV, `max-workers == len(gpus)`, manifest/job identity checks, terminal state acceptance, nonterminal rejection, nonzero child rejection, missing leaf rejection, and CLI wrapper behavior.
- Production files are below 800 lines and production functions are below 50 lines.
- `git diff --check` is clean.

## Concerns

- Coverage is exactly at the required threshold: 80.00%.
- The pytest cleanup warnings are pre-existing temp-directory cleanup warnings from the broader test environment; the relevant Task 4 assertions passed.
- The direct pytest-cov form can fail when coverage pre-imports `framework.stage6`; the requested pre-import form succeeds and records the Task 4 module coverage.
- No real native GPU execution was run by design; tests use fakes at the historical leaf process boundary.

## Fix round 1/5: native manifest fidelity

### Review finding

The original Task 4 tests fabricated a non-native `performance_manifest.json` using a top-level `rows` field. The historical Stage5 v2 planner writes `performance_manifest.json` with top-level `jobs`, while `performance_jobs.jsonl` contains native Stage5 performance job rows and Stage3 executor state rows use the `stage3_execute_performance_plan_v3_state` schema. Production also accepted both `manifest["jobs"]` and `manifest["rows"]`, which was too permissive.

### Native shape inspection

Authoritative files inspected:

- `/home/jichengzhi/V2X/framework/stage5/measurement_plan_v2.py`
- `/home/jichengzhi/V2X/framework/stage5/measurement_plan_v1.py`
- `/home/jichengzhi/V2X/scripts/stage35_gold32_performance_plan_v1.py`
- `/home/jichengzhi/V2X/scripts/stage3_execute_performance_plan_v3.py`

Native field names confirmed:

- Manifest: `schema_version`, `source_request_schema`, `source_request_sha256`, `task_id`, `task_sha256`, `source_pool`, `genome_count`, `row_count`, `group_count`, `group_ids`, `jobs`.
- Manifest jobs: `schema_version`, `job_id`, `manifest_job_id`, `split`, `source_pool`, `required_metrics`, `source_status`, `source_evidence_path`, `source_evidence_sha256`, `source_contract`, `terminal_status`, plus copied request row identity fields.
- Performance jobs jsonl: `schema_version`, `job_id`, `manifest_job_id`, `group_id`, `model`, `width_key`, `q_mode`, `runner_key`, `dispatch_key`, `split`, `onnx_path`, `calibration_root`, `source_contract`, `command`, `assigned_gpu`, `gpu_pool`, `remote_artifact_root`, `expected_result_json`, `max_attempts`, `terminal_status`.
- Executor state rows: `schema_version`, `job_id`, `attempt`, `status`, `returncode`, `start_time_unix`, `end_time_unix`, `elapsed_s`, `stdout_path`, `stderr_path`, `result_json`, `result_sha256`, `failure_reasons`.
- Terminal statuses accepted by the adapter remain the native terminal rows: `success` and `confirmed_failure`.

### RED evidence

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Output:

```text
..F..........                                                            [100%]
=================================== FAILURES ===================================
________ test_performance_round_rejects_non_native_manifest_rows_alias _________

tmp_path = PosixPath('/tmp/pytest-of-jichengzhi/pytest-4763/test_performance_round_rejects0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fe7f61167b0>

    def test_performance_round_rejects_non_native_manifest_rows_alias(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Break caught: adapter accepts fabricated manifest rows instead of native jobs."""
        _set_runtime_env(monkeypatch, tmp_path)
        profile = _profile(tmp_path)
        round_root = tmp_path / "round"
        request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
        task_state = _write_quantized_task_state(round_root, request)
        original_state = _read_json(task_state)

>       with pytest.raises(P6PerformanceRoundAdapterError):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'framework.stage6.p6_performance_round_adapter_v1.P6PerformanceRoundAdapterError'>

tests/stage6/test_p6_performance_round_adapter.py:208: Failed
=========================== short test summary info ============================
FAILED tests/stage6/test_p6_performance_round_adapter.py::test_performance_round_rejects_non_native_manifest_rows_alias
1 failed, 12 passed in 4.96s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-6fc85cf9-4b7b-44d1-b470-b5864043ab5e/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-6fc85cf9-4b7b-44d1-b470-b5864043ab5e/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-6fc85cf9-4b7b-44d1-b470-b5864043ab5e/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-6fc85cf9-4b7b-44d1-b470-b5864043ab5e/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-6fc85cf9-4b7b-44d1-b470-b5864043ab5e
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-6fc85cf9-4b7b-44d1-b470-b5864043ab5e'
  warnings.warn(
```

### Fix

- Unit fake planner now emits native `manifest["jobs"]`, native `performance_jobs.jsonl`, and native `performance_state.jsonl` rows by default.
- Release fake leaves now emit the same native field names.
- Added an explicit unit test proving native manifest/jobs/state fixtures succeed.
- Added an explicit unit test proving a fabricated manifest with top-level `rows` is rejected.
- Production now validates only `manifest["jobs"]` and only `manifest_job_id` for native manifest/job identity. The `rows` fallback and `row_id` identity alias were removed.

### GREEN evidence

Command:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Output:

```text
.............                                                            [100%]
13 passed in 4.80s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-18e03e77-62e4-4efc-adae-d008793779c7/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-18e03e77-62e4-4efc-adae-d008793779c7/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-18e03e77-62e4-4efc-adae-d008793779c7/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-18e03e77-62e4-4efc-adae-d008793779c7/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-18e03e77-62e4-4efc-adae-d008793779c7
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-18e03e77-62e4-4efc-adae-d008793779c7'
  warnings.warn(
```

### Focused coverage evidence

Command:

```bash
python - <<'PY'
import framework.stage6.coptv2x_h800_search_v2  # noqa: F401
import pytest
raise SystemExit(pytest.main([
    '-q',
    '--cov=framework.stage6.p6_performance_round_adapter_v1',
    '--cov-branch',
    '--cov-report=term-missing',
    '--cov-fail-under=80',
    'tests/stage6/test_p6_performance_round_adapter.py',
]))
PY
```

Output:

```text
...........                                                              [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.12-final-0 _______________

Name                                                  Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------------------
framework/stage6/p6_performance_round_adapter_v1.py     115     18     34     10    81%   58-59, 127, 131, 165, 170-171, 177, 184, 186-187, 189, 196, 202-203, 205, 217, 220
-------------------------------------------------------------------------------------------------
TOTAL                                                   115     18     34     10    81%
Required test coverage of 80% reached. Total coverage: 81.21%
11 passed in 0.12s
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-498a4f14-ca98-4e3c-b508-30920de3427e/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-498a4f14-ca98-4e3c-b508-30920de3427e/test_temporary_cleanup_unlocks0/.execution-closure.locked.tmp'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-498a4f14-ca98-4e3c-b508-30920de3427e/test_temporary_cleanup_unlocks0
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-498a4f14-ca98-4e3c-b508-30920de3427e/test_temporary_cleanup_unlocks0'
  warnings.warn(
/home/jichengzhi/miniconda3/lib/python3.13/site-packages/_pytest/pathlib.py:96: PytestWarning: (rm_rf) error removing /tmp/pytest-of-jichengzhi/garbage-498a4f14-ca98-4e3c-b508-30920de3427e
<class 'OSError'>: [Errno 39] Directory not empty: '/tmp/pytest-of-jichengzhi/garbage-498a4f14-ca98-4e3c-b508-30920de3427e'
  warnings.warn(
```

Per-module branch result:

- `framework/stage6/p6_performance_round_adapter_v1.py`: 115 statements, 18 missed, 34 branches, 10 partial branches, 81% coverage; total coverage 81.21%.

### Static evidence

Ruff command:

```bash
python -m ruff check framework/stage6/p6_performance_round_adapter_v1.py tools/release/run_p6_performance_round_adapter.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

Ruff output:

```text
All checks passed!
```

compileall command:

```bash
python -m compileall -q framework/stage6/p6_performance_round_adapter_v1.py tools/release/run_p6_performance_round_adapter.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
```

compileall output:

```text
<no output; exit code 0>
```

Diff check command:

```bash
git diff --check
```

Diff check output:

```text
<no output; exit code 0>
```

### Post-helper-split rerun

After splitting the release fake leaf body helper to keep functions within the size target, the focused checks were rerun.

Functional/static/size command group:

```bash
PYTHONPATH=. pytest -q tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
python -m ruff check framework/stage6/p6_performance_round_adapter_v1.py tools/release/run_p6_performance_round_adapter.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
python -m compileall -q framework/stage6/p6_performance_round_adapter_v1.py tools/release/run_p6_performance_round_adapter.py tests/stage6/test_p6_performance_round_adapter.py tests/release/test_run_p6_performance_round_adapter.py
git diff --check
```

Output:

```text
.............                                                            [100%]
13 passed in 4.99s
All checks passed!
framework/stage6/p6_performance_round_adapter_v1.py: 231 lines
tools/release/run_p6_performance_round_adapter.py: 100 lines
tests/stage6/test_p6_performance_round_adapter.py: 495 lines
tests/release/test_run_p6_performance_round_adapter.py: 350 lines
```

Final pre-import coverage rerun:

```text
...........                                                              [100%]
Name                                                  Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------------------
framework/stage6/p6_performance_round_adapter_v1.py     115     18     34     10    81%   58-59, 127, 131, 165, 170-171, 177, 184, 186-187, 189, 196, 202-203, 205, 217, 220
-------------------------------------------------------------------------------------------------
TOTAL                                                   115     18     34     10    81%
Required test coverage of 80% reached. Total coverage: 81.21%
11 passed in 0.12s
```
