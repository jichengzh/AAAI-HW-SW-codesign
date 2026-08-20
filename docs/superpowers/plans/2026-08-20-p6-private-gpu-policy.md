# P6 Private GPU Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the P6 H800 execution chain derive its exact three-device policy only from the Git-ignored private runner interface, while preserving strict GPU admission and public redaction.

**Architecture:** The binding validator parses the private `CUDA_VISIBLE_DEVICES` literal into one canonical three-index policy and uses it for the binding-time double probe. The provision and measurement adapters consume that stored private policy to construct direct `nvidia-smi` calls and to revalidate UUID, H800 model, and occupancy before and after execution. Public projections and public contracts retain no device index values.

**Tech Stack:** Python 3.10+, pytest, subprocess direct argv, `nvidia-smi`, Git-ignored local P6 configuration.

**Spec:** Confirmed user design on 2026-08-20; [P6.4 full-chain design](../specs/2026-08-20-p6-4-dynamic-h800-full-chain-execution-design.md).

## Global Constraints

- Exact GPU device indices are present only in a Git-ignored private runner interface and its generated private binding.
- The private `CUDA_VISIBLE_DEVICES` policy must be exactly three distinct non-negative indices in canonical ascending order.
- Every admitted device must be an H800, have a unique non-empty UUID, and have occupancy at most `0.05`.
- Binding creation probes the configured devices twice and rejects UUID drift; measurement rechecks the identical policy before and after execution.
- `nvidia-smi` is always invoked by direct argv; no shell, ambient policy, path discovery, fallback, or default device list is allowed.
- Public contract, public binding projection, tracked design/plan prose, and production source must not record a real device index.
- Preserve P6.1 static 343/686 behavior, P6.4 four rounds × four candidates, Gold176 cold start, TVM backend, and sixteen actual measurements.
- Do not access H800, private assets, or generate a local runner configuration while implementing this change.

---

### Task 1: Private policy parsing and binding admission

**Files:**
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `tests/stage6/test_p6_history_binding.py`
- Modify: `tests/stage6/test_p6_full_chain_bootstrap.py`

**Interfaces:**
- Consumes: `execution_interface.environment.values.CUDA_VISIBLE_DEVICES` with `kind: literal`.
- Produces: private `binding["gpu_policy"]` with canonical `indices`, UUIDs, H800 label, and occupancy ceiling.
- Consumed by: Task 2 probe adapters and Task 3 measurement revalidation.

- [ ] **Step 1: Write failing policy-admission tests**

```python
def test_binding_derives_probe_indices_from_private_cuda_policy() -> None:
    interface = _runner_interface(cuda_visible_devices="17,19,23")
    binding = build_history_binding(..., execution_interface=interface, gpu_probe=probe)
    assert probe.calls == [(17, 19, 23), (17, 19, 23)]
    assert binding["gpu_policy"]["indices"] == [17, 19, 23]


@pytest.mark.parametrize("policy", ("17,17,23", "23,19,17", "17,19", "17,19,x"))
def test_binding_rejects_noncanonical_private_cuda_policy(policy: str) -> None:
    with pytest.raises(P6HistoryBindingError, match="execution_interface"):
        build_history_binding(..., execution_interface=_runner_interface(policy), gpu_probe=probe)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `PYTHONPATH=. python -m pytest tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py -q`

Expected: FAIL because the fixed device policy is still required.

- [ ] **Step 3: Implement canonical private-policy derivation**

```python
def _private_gpu_indices(interface: Mapping[str, Any]) -> tuple[int, int, int]:
    value = interface["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"]
    parts = tuple(value.split(","))
    indices = tuple(int(part) for part in parts)
    if len(indices) != 3 or tuple(sorted(indices)) != indices or len(set(indices)) != 3:
        raise _execution_interface_error()
    if any(index < 0 for index in indices):
        raise _execution_interface_error()
    return indices
```

Use the derived tuple for both binding probes and for the private binding policy. Keep `CUDA_VISIBLE_DEVICES` a literal private interface value, but validate its format rather than one hard-coded value.

- [ ] **Step 4: Run focused tests and refactor only duplication**

Run: `PYTHONPATH=. python -m pytest tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py -q`

Expected: PASS.

- [ ] **Step 5: Run lint and commit**

Run: `python -m ruff check framework/stage6/p6_history_binding_v1.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py && git diff --check`

Commit: `feat: derive P6 GPU policy from private interface`

### Task 2: Policy-driven direct-argv GPU probes

**Files:**
- Modify: `tools/release/provision_p6_history_local_config.py`
- Modify: `tools/release/measure_p6_history_batch.py`
- Modify: `tests/release/test_provision_p6_full_chain_local_config.py`
- Modify: `tests/release/test_p6_history_execution_adapters.py`

**Interfaces:**
- Consumes: `GpuProbe.snapshot(indices: tuple[int, ...])` from Task 1 and its canonical three-index argument.
- Produces: a tuple of matching `GpuRecord` values from direct `nvidia-smi --id=<comma-separated-indices>` argv.
- Consumed by: provision CLI and measurement batch CLI.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_gpu_probe_builds_direct_argv_from_supplied_private_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = NvidiaSmiGpuProbe()
    records = probe.snapshot((17, 19, 23))
    assert observed_argv[0:2] == ("nvidia-smi", "--id=17,19,23")
    assert [record.index for record in records] == [17, 19, 23]


def test_gpu_probe_rejects_noncanonical_policy_before_subprocess() -> None:
    with pytest.raises(ValueError):
        NvidiaSmiGpuProbe().snapshot((19, 17, 23))
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `PYTHONPATH=. python -m pytest tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_p6_history_execution_adapters.py -q`

Expected: FAIL because both adapters still enforce a fixed policy.

- [ ] **Step 3: Implement direct-argv policy rendering**

```python
def _gpu_query_argv(indices: tuple[int, ...]) -> tuple[str, ...]:
    _validate_indices(indices)
    return (
        "nvidia-smi",
        "--id=" + ",".join(str(index) for index in indices),
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
```

Each existing `NvidiaSmiGpuProbe.snapshot()` validates the supplied tuple before launching the direct subprocess and returns records in the supplied canonical order. Do not add a CLI argument or a default policy.

- [ ] **Step 4: Run focused tests and refactor only local duplication**

Run: `PYTHONPATH=. python -m pytest tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_p6_history_execution_adapters.py -q`

Expected: PASS.

- [ ] **Step 5: Run lint and commit**

Run: `python -m ruff check tools/release/provision_p6_history_local_config.py tools/release/measure_p6_history_batch.py tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_p6_history_execution_adapters.py && git diff --check`

Commit: `feat: parameterize P6 GPU probe argv`

### Task 3: Measurement revalidation, integration, and public-redaction regression

**Files:**
- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Modify: `tests/stage6/test_p6_history_measurement.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/release/test_run_p6_h800_search.py`
- Modify: `docs/superpowers/specs/2026-08-20-p6-4-dynamic-h800-full-chain-execution-design.md`
- Modify: `docs/superpowers/plans/2026-08-20-p6-4-dynamic-h800-full-chain-execution.md`

**Interfaces:**
- Consumes: Task 1 private binding `gpu_policy.indices` and Task 2 probe behavior.
- Produces: fail-closed pre- and post-execution admission using the same stored policy.
- Preserves: public binding projection has no indices or UUIDs; P6.1 and P6.4 offline contracts remain unchanged except the generic private-device wording.

- [ ] **Step 1: Write failing runtime and redaction tests**

```python
def test_measurement_revalidates_the_binding_private_policy_before_and_after_execution() -> None:
    run_history_measurement_batch(..., gpu_probe=probe_for((17, 19, 23)))
    assert probe.calls == [(17, 19, 23), (17, 19, 23)]


def test_public_binding_projection_does_not_expose_private_gpu_policy() -> None:
    projection = public_binding_projection(private_binding)
    assert "gpu_policy" not in projection
    assert "17" not in json.dumps(projection)
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `PYTHONPATH=. python -m pytest tests/stage6/test_p6_history_measurement.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -q`

Expected: FAIL because execution still reads fixed indices.

- [ ] **Step 3: Implement stored-policy revalidation and generic documentation**

```python
indices = tuple(policy["indices"])
snapshot = gpu_probe.snapshot(indices)
```

Validate the tuple shape, UUID key set, H800 model, and occupancy before and after running the private chain. Replace fixed device-number prose in the P6.4 design and plan with “the private configuration’s selected three H800 devices”; do not add real indices to tracked documentation.

- [ ] **Step 4: Run focused integration tests**

Run: `PYTHONPATH=. python -m pytest tests/stage6/test_p6_history_measurement.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -q`

Expected: PASS.

- [ ] **Step 5: Run static checks and commit**

Run: `python -m ruff check framework/stage6/p6_history_measurement_v1.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py && git diff --check`

Commit: `fix: revalidate private P6 GPU policy`

### Task 4: Offline acceptance gate

**Files:**
- Modify: no production files unless a failing Task 1–3 regression requires its owning task’s correction.
- Test: `tests/stage6/test_p6_history_binding.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/stage6/test_p6_history_measurement.py`
- Test: `tests/stage6/test_coptv2x_h800_search.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`
- Test: `tests/release/test_p6_history_execution_adapters.py`
- Test: `tests/release/test_run_p6_h800_search.py`

**Interfaces:**
- Consumes: completed Tasks 1–3.
- Produces: evidence that dynamic private policy admission preserves P6.1/P6.4 offline behavior without H800 access.

- [ ] **Step 1: Run focused Stage6 and release suite**

Run: `PYTHONPATH=. python -m pytest tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/stage6/test_p6_history_measurement.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_provision_p6_full_chain_local_config.py tests/release/test_p6_history_execution_adapters.py tests/release/test_run_p6_h800_search.py -q`

Expected: PASS.

- [ ] **Step 2: Run quality checks**

Run: `python -m ruff check framework/stage6/p6_history_binding_v1.py framework/stage6/p6_history_measurement_v1.py tools/release/provision_p6_history_local_config.py tools/release/measure_p6_history_batch.py && git diff --check`

Expected: PASS.

- [ ] **Step 3: Commit acceptance evidence only if code changed in a Task 1–3 fix loop**

No new behavior is introduced by this task. Record commands and outputs in the Git-ignored SDD report.
