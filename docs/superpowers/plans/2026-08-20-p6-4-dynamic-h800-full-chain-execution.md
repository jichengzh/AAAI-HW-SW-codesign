# P6.4 Dynamic H800 Full-Chain Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make P6 framework mode generate a real Pyramid Stage1 manifest, derive the dynamic Stage2 candidate pool, and execute the existing actual Stage5 chain through four H800/TVM rounds without static fallback.

**Architecture:** Framework-mode P6 gains a pre-search `stage1_scan_step` that writes one private `stage1_partition_manifest_v1`; the controller consumes that artifact before its dynamic candidate plan. A private bootstrap validates the known local asset locator, renders a P6.3 binding/interface and local config, then the existing registry and measurement adapters retain responsibility for Stage5 source contracts and actual feedback.

**Tech Stack:** Python 3.10+, PyYAML, pytest, Ruff, existing Stage1 graph scanner, Stage2 bridge, Stage5/P6 contracts, `nvidia-smi`, H800/TVM private environment.

**Spec:** `docs/superpowers/specs/2026-08-20-p6-4-dynamic-h800-full-chain-execution-design.md`

## Global Constraints

- Use Pyramid, H800, TVM, Gold176, four rounds, four candidates per round, and 16 total measurement slots.
- Build candidates only from the current real Stage1 manifest through Stage2; do not use demos, static 343/686 counts, pruning heuristics, proxy objectives, or static fallback.
- A candidate has one global `fp16` or `int8` q-mode; never add candidate-level mixed precision.
- Reuse Gold176 for cold-start fitting only; do not measure it again.
- Use GPU 5, 6, and 7 only. Keep all private paths, argv, IDs, measurements, checkpoints, logs, and results Git-ignored.
- Preserve existing request/row identity checks and P6.1 static 343/686 regression. Do not add a per-asset SHA256 inventory.
- Each task is RED -> GREEN -> focused review -> commit. Do not push, merge, download, or start H800 jobs during Tasks 1-5.

## File Structure

| Path | Responsibility |
| --- | --- |
| `framework/stage1/graph_scan.py` | Emit a schema-tagged real Stage1 partition manifest. |
| `framework/stage6/coptv2x_h800_search_v2.py` | Parse and execute a framework-only Stage1 pre-step. |
| `framework/stage6/p6_full_chain_bootstrap_v1.py` | Validate private locator/template structures and render binding/config. |
| `framework/stage6/p6_history_binding_v1.py` | Validate bootstrap-rendered history interface. |
| `tools/release/provision_p6_full_chain_local_config.py` | Redacted CLI for private bootstrap. |
| `tests/stage1/test_trace_graph_adapter_workflow.py` | Stage1 manifest tests. |
| `tests/stage6/test_coptv2x_h800_search.py` | Controller pre-step and four-round tests. |
| `tests/stage6/test_p6_full_chain_bootstrap.py` | Bootstrap structure and no-write tests. |
| `tests/release/test_provision_p6_full_chain_local_config.py` | CLI black-box tests. |
| `tests/release/test_run_p6_h800_search.py` | CLI lifecycle ordering tests. |

---

### Task 1: Tag the real Stage1 partition manifest

**Files:**

- Modify: `framework/stage1/graph_scan.py:636-667`
- Modify: `tests/stage1/test_trace_graph_adapter_workflow.py:90-116`
- Modify: `tests/stage1/test_stage1_bridge.py:10-118`

**Interfaces:**

- Consumes: `graph_scan.scan(adapter, hw, device, profile_latency_mode)`.
- Produces: a mapping with `schema == "stage1_partition_manifest_v1"`, `stage == "stage1_partition"`, and existing fields accepted by `load_stage2_search_space(path)`.

- [ ] **Step 1: Write failing schema and bridge tests**

```python
def test_toy_scan_marks_real_partition_manifest_schema() -> None:
    manifest = scan(_ToyAdapter(), _hardware(), device="cpu", profile_latency_mode="off")
    assert manifest["schema"] == "stage1_partition_manifest_v1"
    assert manifest["stage"] == "stage1_partition"


def test_stage2_loader_accepts_schema_tagged_real_scan_manifest(tmp_path: Path) -> None:
    manifest = scan(_ToyAdapter(), _hardware(), device="cpu", profile_latency_mode="off")
    path = _write_yaml(tmp_path / "partition.yaml", manifest)
    assert load_stage2_search_space(path)["schema"] == "stage2_search_space_v1"
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/stage1/test_trace_graph_adapter_workflow.py tests/stage1/test_stage1_bridge.py -q`

Expected: the new schema assertion fails; existing bridge cases remain green.

- [ ] **Step 3: Add the schema at the manifest source**

```python
manifest = {
    "schema": "stage1_partition_manifest_v1",
    "stage": "stage1_partition",
    "model": adapter.name,
    "model_class": adapter.model_class,
    "config": adapter.config_path,
    "ckpt": adapter.ckpt_path,
    # Preserve the existing trace, search-space, routing, and checks mappings.
}
```

Do not add candidate counts, metrics, paths, or demo markers.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/stage1/test_trace_graph_adapter_workflow.py tests/stage1/test_stage1_bridge.py -q`

Expected: PASS.

- [ ] **Step 5: Review and commit**

Run: `python -m ruff check framework/stage1/graph_scan.py tests/stage1/test_trace_graph_adapter_workflow.py tests/stage1/test_stage1_bridge.py && git diff --check`

```bash
git add framework/stage1/graph_scan.py tests/stage1/test_trace_graph_adapter_workflow.py tests/stage1/test_stage1_bridge.py
git commit -m "feat: tag real Stage1 partition manifests"
```

### Task 2: Run Stage1 before framework candidate construction

**Files:**

- Modify: `framework/stage6/coptv2x_h800_search_v2.py:80-125, 228-239, 260-321, 392-417, 1010-1135`
- Modify: `tests/stage6/test_coptv2x_h800_search.py:300-900, 1716-1800`

**Interfaces:**

- Consumes: framework local key `stage1_scan_step` with `name: "build_stage1_partition"`, argv tokens `{stage1_partition_manifest}` and `{local_output_root}`, and absolute `stage2_search_space_path`.
- Produces: `LocalP6CoptV2XConfig.stage1_scan_step: LocalExecutionStep | None`; static mode uses `None` and framework mode requires the validated step.
- Produces: `_run_framework_stage1_scan(local, command_runner) -> None` before `_build_source_registry()`.

- [ ] **Step 1: Write failing config and ordering tests**

```python
def test_framework_mode_requires_a_stage1_scan_step(tmp_path: Path) -> None:
    payload = _framework_local_payload(tmp_path)
    payload.pop("stage1_scan_step")
    with pytest.raises(P6CoptV2XContractError, match="stage1_scan_step"):
        load_local_config(_write_yaml(tmp_path / "local.yaml", payload), _public_contract())


def test_framework_run_executes_scan_before_registry_or_measurement(tmp_path: Path) -> None:
    local = _framework_local_config_with_stage1_step(tmp_path)
    events: list[str] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        events.append(argv[0])
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(local.stage2_search_space_path)
        elif argv[0] == "fake-registry":
            _write_framework_registry(cwd / "source_registry.json")
        else:
            _write_feedback_from_request(argv)
        return 0

    state = run_p6_coptv2x_search(_public_contract(), local, "test-revision", runner)
    assert state.status == "completed"
    assert events[0] == "fake-stage1"
    assert events.count("fake-measure") == 4
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py -k 'stage1_scan_step or scan_before_registry' -q`

Expected: FAIL because config parsing and pre-step execution do not exist.

- [ ] **Step 3: Implement framework-only parsing, execution, and validation**

```python
def _run_framework_stage1_scan(local: LocalP6CoptV2XConfig, command_runner: CommandRunner) -> None:
    if local.candidate_source_mode != "framework_stage2_search_space":
        return
    assert local.stage1_scan_step is not None
    assert local.stage2_search_space_path is not None
    _validate_local_output_leaf(local.stage2_search_space_path)
    _run_step(local.stage1_scan_step, {
        "{stage1_partition_manifest}": local.stage2_search_space_path,
        "{local_output_root}": local.local_output_root,
    }, cwd=local.local_output_root, runner=command_runner)
    _validate_stage1_partition_manifest(local.stage2_search_space_path)
```

Call it after `_prepare_local_output_root()` and before `_load_search_inputs()` or `_build_source_registry()`. The validator requires the Task 1 schema, `stage1_partition`, `pyramid_lidar`, `scan_status == "ok"`, an H800 capability mapping, and successful `load_stage2_search_space()`. Convert any failure to `P6CoptV2XExecutionError("stage1_scan_invalid", "stage1 scan invalid")`.

- [ ] **Step 4: Verify GREEN and static compatibility**

Run: `PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py -q`

Expected: PASS, including static 343/686 and dynamic framework cases.

- [ ] **Step 5: Review and commit**

Run: `python -m ruff check framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_coptv2x_h800_search.py && git diff --check`

```bash
git add framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_coptv2x_h800_search.py
git commit -m "feat: run Stage1 before dynamic P6 search"
```

### Task 3: Render one private full-chain binding from explicit inputs

**Files:**

- Create: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `framework/stage6/p6_history_binding_v1.py:161-204, 458-620`
- Create: `tests/stage6/test_p6_full_chain_bootstrap.py`

**Interfaces:**

- Consumes: an ignored legacy local YAML, an ignored runner template, a private output root, and injected `GpuProbe`.
- Produces: `FullChainBootstrapError(category: str, detail: str)` and `materialize_full_chain_binding(legacy_local_config: Path, runner_template: Path, local_output_root: Path, binding_output: Path, config_output: Path, gpu_probe: GpuProbe) -> dict[str, Any]`.
- Produces: one validated `p6_history_binding_v1` plus a framework `p6_h800_coptv2x_local_v2` config with `stage1_scan_step`, the existing registry adapter argv, and the existing measurement adapter argv.

The runner template is private input, not a tracked file. Its exact top-level keys are `schema_version`, `stage1_scan`, and `execution_interface`; `schema_version` is `p6_history_runner_template_v1`; `stage1_scan` has exactly `name` and `argv`; and `execution_interface` has the exact existing `p6_history_runner_interface_v1` fields `schema_version`, `controller`, `execution_chain`, `environment`, `output_layout`, and `actual_feedback`. It contains the Stage1 scan argv and five explicit Stage5 interface entries. Executable paths are relative to one verified source root before rendering, so command choice is explicit but never public or inferred from a filename.

- [ ] **Step 1: Write failing synthetic-root tests**

```python
def test_materialize_full_chain_binding_renders_dynamic_config(tmp_path: Path) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    binding = materialize_full_chain_binding(
        legacy_config, template, output_root,
        output_root / "binding.json", output_root / "local.yaml", _gpu_probe(),
    )
    local = load_local_config(output_root / "local.yaml", _public_contract())
    assert binding["schema_version"] == "p6_history_binding_v1"
    assert local.candidate_source_mode == "framework_stage2_search_space"
    assert local.stage1_scan_step is not None


@pytest.mark.parametrize("mutation", ["two_common_roots", "missing_stage", "escape", "bad_template"])
def test_bootstrap_rejects_untrusted_or_ambiguous_private_inputs(tmp_path: Path, mutation: str) -> None:
    legacy_config, template, output_root = _write_invalid_private_inputs(tmp_path, mutation)
    with pytest.raises(FullChainBootstrapError):
        materialize_full_chain_binding(
            legacy_config, template, output_root,
            output_root / "binding.json", output_root / "local.yaml", _gpu_probe(),
        )
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_full_chain_bootstrap.py -q`

Expected: collection failure because the module and function do not exist.

- [ ] **Step 3: Implement structural rendering without ambient discovery**

```python
def materialize_full_chain_binding(
    legacy_local_config: Path,
    runner_template: Path,
    local_output_root: Path,
    binding_output: Path,
    config_output: Path,
    gpu_probe: GpuProbe,
) -> dict[str, Any]:
    locator = _load_legacy_local_locator(legacy_local_config)
    root = _unique_common_history_root(locator)
    interface = _render_runner_interface(root, _load_runner_template(runner_template))
    binding = _build_binding(root, locator, interface, gpu_probe)
    config = _render_full_chain_local_config(binding, interface, local_output_root, binding_output)
    _validate_rendered_pair(binding, config, config_output)
    write_private_binding_pair(binding, config, binding_output, config_output)
    return binding
```

`_unique_common_history_root()` derives candidates from the four resolved local inputs and four required component markers, requires exactly one common Git root, and rejects anything outside the legacy locator root. `_render_runner_interface()` resolves only template-relative executables under that root, preserves the five stage order/placeholders already validated by `p6_history_binding_v1.py`, and validates before writing. It must never use `shell=True`, glob-first selection, or a fallback command.

- [ ] **Step 4: Verify GREEN with existing binding/measurement tests**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_full_chain_bootstrap.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_history_measurement.py -q`

Expected: PASS.

- [ ] **Step 5: Review and commit**

Run: `python -m ruff check framework/stage6/p6_full_chain_bootstrap_v1.py framework/stage6/p6_history_binding_v1.py tests/stage6/test_p6_full_chain_bootstrap.py && git diff --check`

```bash
git add framework/stage6/p6_full_chain_bootstrap_v1.py framework/stage6/p6_history_binding_v1.py tests/stage6/test_p6_full_chain_bootstrap.py
git commit -m "feat: materialize private P6 full-chain binding"
```

### Task 4: Expose the bootstrap through a redacted local-only CLI

**Files:**

- Create: `tools/release/provision_p6_full_chain_local_config.py`
- Create: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces:**

- Consumes: absolute `--legacy-local-config`, `--runner-template`, `--local-output-root`, `--binding-output`, and `--config-output`.
- Produces: exit 0 with `p6_full_chain_config_written`; exit 1 with one stable bootstrap category; exit 2 with `argument_error`.
- Produces: no pair on rejected template, nonignored repository output, invalid GPU probe, or nonunique root.

- [ ] **Step 1: Write subprocess black-box CLI tests**

```python
def test_cli_writes_only_ignored_private_pair(tmp_path: Path) -> None:
    result = _run_cli(*_valid_args(tmp_path))
    assert result.returncode == 0
    assert result.stdout == "p6_full_chain_config_written\n"
    assert _load_local_config_without_echoing_private_values(tmp_path).stage1_scan_step is not None


def test_cli_redacts_private_template_failure_and_writes_nothing(tmp_path: Path) -> None:
    result = _run_cli(*_invalid_template_args(tmp_path))
    assert result.returncode == 1
    assert result.stderr == "execution_interface_unavailable\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/release/test_provision_p6_full_chain_local_config.py -q`

Expected: collection failure because the CLI module does not exist.

- [ ] **Step 3: Implement direct argv parsing and stable error mapping**

```python
def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        materialize_full_chain_binding(
            args.legacy_local_config,
            args.runner_template,
            args.local_output_root,
            args.binding_output,
            args.config_output,
            NvidiaSmiGpuProbe(),
        )
    except FullChainBootstrapError as error:
        sys.stderr.write(f"{error.category}\n")
        return 1
    except (OSError, TypeError, ValueError):
        sys.stderr.write("bootstrap_invalid\n")
        return 1
    sys.stdout.write("p6_full_chain_config_written\n")
    return 0
```

Reuse the fixed `nvidia-smi --id=5,6,7` parser and `git check-ignore` destination policy of `provision_p6_history_local_config.py`. Do not echo supplied paths, argv, template text, UUIDs, or exception detail.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_provision_p6_history_local_config.py -q`

Expected: PASS.

- [ ] **Step 5: Review and commit**

Run: `python -m ruff check tools/release/provision_p6_full_chain_local_config.py tests/release/test_provision_p6_full_chain_local_config.py && git diff --check`

```bash
git add tools/release/provision_p6_full_chain_local_config.py tests/release/test_provision_p6_full_chain_local_config.py
git commit -m "feat: provision private P6 full-chain config"
```

### Task 5: Prove the complete dynamic lifecycle offline

**Files:**

- Modify: `tests/stage6/test_coptv2x_h800_search.py:300-900`
- Modify: `tests/release/test_run_p6_h800_search.py:400-650`
- Modify: `tests/release/test_p6_history_execution_adapters.py:1-220`

**Interfaces:**

- Consumes: the Task 2 framework local config and Task 3 synthetic binding pair.
- Produces: offline proof of `stage1_scan_step -> dynamic plan -> registry-v2 -> Gold176 fit -> four 4-row feedback batches -> completed`.

- [ ] **Step 1: Write failing end-to-end controller and CLI tests**

```python
def test_full_framework_lifecycle_uses_actual_scan_artifact_and_four_feedback_rounds(tmp_path: Path) -> None:
    local, calls = _full_chain_local_config_and_fake_runner(tmp_path)
    state = run_p6_coptv2x_search(_public_contract(), local, "test-revision", calls.runner)
    assert state.status == "completed"
    assert calls.names == ["fake-stage1", "fake-registry", *(["fake-measure"] * 4)]
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16


def test_framework_lifecycle_never_uses_static_registry_after_stage1(tmp_path: Path) -> None:
    local, calls = _full_chain_local_config_and_static_registry_runner(tmp_path)
    with pytest.raises(P6CoptV2XExecutionError, match="source_registry_invalid"):
        run_p6_coptv2x_search(_public_contract(), local, "test-revision", calls.runner)
    assert calls.measurement_count == 0
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -k 'full_framework_lifecycle or never_uses_static_registry' -q`

Expected: FAIL until Tasks 2-4 are wired into the fixtures.

- [ ] **Step 3: Connect fixtures only through public APIs**

```python
def runner(argv: tuple[str, ...], cwd: Path) -> int:
    if argv[0] == "fake-stage1":
        _write_real_stage1_manifest(_stage1_manifest_path(argv))
    elif argv[0] == "fake-registry":
        _write_plan_identical_registry(cwd / "source_registry.json")
    elif argv[0] == "fake-measure":
        _write_valid_feedback(_measurement_request_path(argv), _feedback_path(argv))
    else:
        raise AssertionError(f"unexpected argv: {argv!r}")
    return 0
```

The helper writes only the three public contract shapes; it does not mock the controller’s cost-model, identity gates, or round transition functions. Assert cold-start fit occurs once, online fit occurs after each prior completed round, and every request has four unique rows.

- [ ] **Step 4: Verify GREEN, static regression, and disclosure boundary**

Run: `PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_history_registry.py tests/stage6/test_p6_history_measurement.py tests/release/test_run_p6_h800_search.py -q`

Expected: PASS; static P6.1 retains 343/686 and no synthetic private value appears in public output.

- [ ] **Step 5: Review and commit**

Run: `python -m ruff check tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_history_execution_adapters.py tests/release/test_run_p6_h800_search.py && git diff --check`

```bash
git add tests/stage6/test_coptv2x_h800_search.py tests/stage6/test_p6_history_execution_adapters.py tests/release/test_run_p6_h800_search.py
git commit -m "test: cover P6 dynamic full-chain lifecycle"
```

### Task 6: Gate, deploy, and execute the real private run

**Files:**

- Modify only after successful execution: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify only after successful execution: `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`
- Create outside Git only: private template, binding, local config, Stage1 manifest, registry, state, requests, feedback, logs, models, and measurements.

**Interfaces:**

- Consumes: the Task 4 CLI, existing private asset locator, explicit private runner template, and public P6 contract.
- Produces: a private `completed` state with four rounds/16 measured candidates, or a private failure state with no fabricated feedback.

- [ ] **Step 1: Run the complete offline release gates**

```bash
PYTHONPATH=. python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
PYTHONPATH=. pytest tests/release tests/integration/test_anonymous_archive.py -q
python -m ruff check framework tools scripts tests
git diff --check
```

Expected: tests pass, coverage is at least 80%, and tracked files contain no private artifacts.

- [ ] **Step 2: Verify the approved H800 control connection and private prerequisites**

Use the existing control socket. Run redacted probes only for H800 model, GPU 5/6/7 occupancy, output-root free space, Python/TVM availability, and presence of the private locator/template.

Expected: connection is live; GPU 5/6/7 are H800 and below the configured occupancy threshold; no raw path, data, or checkpoint content reaches public output.

- [ ] **Step 3: Materialize and preflight the private run bundle**

Run `provision_p6_full_chain_local_config.py` with the private locator/template and ignored destinations. Validate the generated pair with `load_local_config()` and run the Stage1 scan step alone.

Expected: one source root, one binding/config pair, successful schema-tagged Stage1 manifest, Stage2-loadable search space, at least 16 dynamic candidates, and no Stage5 measurement.

- [ ] **Step 4: Start exactly one autonomous four-round process after admission**

Run `run_p6_h800_search.py` once with the public contract, generated local config, and code revision. It may wait for GPU admission, then must scan, construct the dynamic space, and execute four rounds without manually replacing candidates, injecting feedback, relaunching a partial round, or measuring Gold176.

Expected: private `completed`, `completed_rounds == 4`, `measured_candidate_count == 16`, and four validated actual feedback batches. Any other terminal state is a failure requiring diagnosis before a new run.

- [ ] **Step 5: Independently verify private completion before documentation**

Check only structural facts: Stage1 schema/status, plan/registry/manifest identity, four rounds, four rows per request, 16 unique rows, five finite metrics per successful row, and final state. Scan tracked/public files for private locations, candidate IDs, raw metrics, and results.

Expected: all closure facts hold and public tree remains clean.

- [ ] **Step 6: Update P6 status only on success, then commit**

Record only the de-identified fact that one dynamic H800/TVM Stage1-to-four-round execution completed. Do not publish numeric results or claim paper-table reproduction. If any prior step fails, do not edit P6 status documents.

```bash
git add docs/AAAI27_RELEASE_AUDIT.md docs/release-manifests/P6_H800_SEARCH_EXECUTION.md
git commit -m "docs: close completed P6 dynamic execution"
```

## Plan Self-Review

- Spec coverage: Tasks 1-2 implement real Stage1-to-Stage2 handoff; Tasks 3-4 implement the explicit private interface/config boundary; Task 5 proves Gold176 plus four rounds offline; Task 6 applies GPU admission, real execution, disclosure control, and the P6 closure condition.
- Static fallback is rejected in Tasks 2 and 5. No task measures Gold176 or adds mixed precision, static candidate counts, per-asset hashes, or another backend.
- `LocalP6CoptV2XConfig.stage1_scan_step`, `_run_framework_stage1_scan`, `materialize_full_chain_binding`, and the bootstrap CLI are all defined before later tasks consume them.
