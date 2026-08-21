# P6 V2 Materializer Training Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make P6 recipe-v2 materialization fail closed unless every selected Pyramid/H800/TVM candidate has a self-contained private source wrapper, complete static training contract, preserved source hashes, marker-gated downstream execution, and a fresh four-round real run with no Gold176 remeasurement.

**Architecture:** Keep the public adapter boundary strict: direct argv, `shell=False`, private round cwd, and exactly the five-key private environment. Add private-only validators for the source wrapper profile and static Pyramid training contract, then enforce those validators in binding/provisioning, registry materialization, request projection, source invocation, downstream marker gates, preflight, and fresh-run state handling. Split new logic out of already-large binding/registry/measurement modules so implementation remains focused and reviewable.

**Tech Stack:** Python 3.10+, pytest, Ruff, JSON/YAML canonical serialization, existing Stage1/Stage2/Stage5/Stage6 P6 modules, injected runner tests, optional private H800 execution only in final gated task.

**Spec:** `docs/superpowers/specs/2026-08-21-p6-v2-materializer-training-bridge-design.md`

## Global Constraints

- Keep `p6_h800_coptv2x_search_contract_v2`, `p6_h800_coptv2x_local_v2`, `stage5_measurement_request_v2`, `p6_h800_coptv2x_feedback_v2`, and `p6_history_dynamic_materialization_recipe_v2`.
- Target stays exactly Pyramid + H800 + TVM, four rounds, batch size four, sample budget 16, metrics `latency_ms`, `energy_j`, `ap30`, `ap50`, `ap70`.
- Stage1 scans real Pyramid structure for H800, then Stage2 produces the dynamic Pyramid/H800/TVM candidate plan; no fallback to static 343/686 P6.1 registry in framework mode.
- Gold176 is frozen cold-start evidence only; it is never emitted as a measurement request row.
- Round 0 cost-model fit count must be 176; online refit counts after accepted feedback must be 180, 184, and 188.
- Each round selects exactly four unmeasured rows; the full successful run selects 16 unique rows.
- Public adapter must not add `PYTHONPATH`, inherit ambient environment, construct shell command strings, use `shell=True`, run from the history root, leak raw stderr/stdout, or publish private paths, GPU UUIDs, dataset roots, checkpoint paths, host details, static training values, or raw logs. Direct `shell=False` execution of the validated private `/bin/sh` marker wrapper is the approved boundary.
- Source materialization must be one direct argv call per canonical Pyramid group: `<source_materializer> --request <projected-request> --model pyramid --group-id <group_id> --gpu <validated-index>`.
- Recipe-v2 source contracts must enforce `training_required: true`, `training_source_kind`, `base_checkpoint_path`, `dataset_root`, `pyramid_config_path`, `training_parameters`, `stage_widths`, and exactly the 11 existing `shared_source_paths` keys.
- Required `training_parameters` semantic keys are `training_mode`, `epochs`, `seed`, `optimizer`, `learning_rate`, `batch_size`, `dataset_split`, `checkpoint_selection`, and `freeze_policy`.
- Static training input paths must be absolute, symlink-safe, existing where required, and under the validated private history root; rendered output paths must be absolute, symlink-safe, unique, and under the local private output root.
- Fail closed before process launch for incomplete private wrapper, binding, recipe-v2 contract, output layout, source contract, static training contract, hash drift, stale state, unsafe symlink, or path collision.
- Existing v1/static compatibility must be preserved; stricter training enforcement applies only to recipe-v2 P6 real-run contracts.
- Files touched by implementation should stay under 800 lines. Existing files already over the limit must receive only routing glue plus a focused split module.
- Every task is RED→GREEN→REFACTOR, includes a review boundary, and uses a conventional commit message. Do not commit from the planning step.
- Final public docs are conditional: update release/audit docs only after the private preflight and real four-round run actually complete successfully.

---

## File Structure

- Create `framework/stage6/p6_history_training_contract_v1.py`: private-only recipe-v2 static training contract constants, static input path validation, rendered output validation, canonical-preserving contract copier, and public redaction helpers.
- Create `framework/stage6/p6_source_wrapper_profile_v1.py`: deterministic private `repo_cwd_exec_v1` wrapper profile renderer/validator for the self-contained private source materializer wrapper referenced by the validated runner template.
- Create `tools/release/render_p6_source_wrapper.py`: private-only renderer CLI that consumes an ignored `p6_private_source_wrapper_profile_v1` profile, writes a deterministic `/bin/sh` marker wrapper beneath the private Git root, and verifies generated bytes without exposing private paths in public artifacts.
- Create `tools/release/preflight_p6_materializer_training_bridge.py`: zero-process preflight CLI that validates public/local contracts, private binding, wrapper profile, static training fields, Stage1/Stage2 leaves, registry destination, round leaves, marker paths, and hashes without launching historical code or touching GPUs.
- Create `tools/release/verify_p6_materializer_training_run.py`: public-safe post-run verifier that resolves exact receipt/barrier/result paths from the validated private binding, delegates feedback identity checks to existing validators, proves 4×4 unique measurements and marker ordering, and prints only a redacted completion summary.
- Modify `framework/stage6/p6_runner_template_validator_v1.py`: expose the validated source stage argv and exact five-key environment assertions without weakening existing template validation.
- Modify `framework/stage6/p6_history_binding_v1.py`: delegate private template and wrapper checks to split modules; keep public projection private-value-free.
- Modify `framework/stage6/p6_full_chain_bootstrap_v1.py`: require a regenerated binding/config pair with a valid wrapper profile and complete static training contract before atomic write.
- Modify `tools/release/provision_p6_full_chain_local_config.py`: surface stable categories for invalid wrapper/training contracts and add no-private-value stderr guarantees.
- Modify `framework/stage6/p6_history_registry_v1.py`: use the new training contract validator for recipe-v2 group contracts and remove untrusted output aliases before hashing.
- Modify `framework/stage6/p6_history_source_materialization_v1.py`: preserve training fields through projection, reject static drift across q-modes, recompute hashes after projection, validate marker paths, and add marker-gate helpers.
- Modify `framework/stage6/p6_history_measurement_v1.py`: enforce exact five-key environment, call source wrapper under round cwd, verify `training_done_marker` and `source_done_marker` before quantization/performance/AP/finalization, and redact failures.
- Modify `framework/stage6/coptv2x_h800_search_v2.py`: add fresh-run/relaunch guards for Stage1 manifest, registry destination, round roots, request/feedback/task-state/receipt/barrier leaves, and no in-place partial resume.
- Modify tests under `tests/stage6/` and `tests/release/` named in each task.
- Conditionally modify final docs such as `docs/AAAI27_RELEASE_AUDIT.md` and a release manifest only after Task 7 succeeds.

---

### Task 1: Static Pyramid Training Contract Validator

**Files:**

- Create: `framework/stage6/p6_history_training_contract_v1.py`
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `framework/stage6/p6_history_registry_v1.py`
- Test: `tests/stage6/test_p6_history_training_contract.py`
- Test: `tests/stage6/test_p6_history_binding.py`
- Test: `tests/stage6/test_p6_history_registry.py`

**Interfaces:**

- Consumes: private binding `source_contract_template`, recipe-v2 group contracts from `materialize_history_registry()`, and existing `SHARED_SOURCE_PATH_KEYS`.
- Produces:

```python
REQUIRED_TRAINING_CONTRACT_KEYS: tuple[str, ...] = (
    "training_required",
    "training_source_kind",
    "base_checkpoint_path",
    "dataset_root",
    "pyramid_config_path",
    "training_parameters",
    "stage_widths",
)

REQUIRED_TRAINING_PARAMETER_KEYS: tuple[str, ...] = (
    "training_mode",
    "epochs",
    "seed",
    "optimizer",
    "learning_rate",
    "batch_size",
    "dataset_split",
    "checkpoint_selection",
    "freeze_policy",
)

STATIC_TRAINING_INPUT_PATH_KEYS: tuple[str, ...] = (
    "base_checkpoint_path",
    "dataset_root",
    "pyramid_config_path",
)

class P6HistoryTrainingContractError(ValueError):
    category: str

def validate_recipe_v2_training_template(
    template: Mapping[str, Any],
    *,
    private_root: Path,
) -> dict[str, Any]: ...

def validate_recipe_v2_group_training_contract(
    contract: Mapping[str, Any],
    *,
    private_root: Path,
    local_output_root: Path,
    group_id: str,
) -> dict[str, Any]: ...

def public_safe_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]: ...
```

- `validate_recipe_v2_training_template()` returns a deep-copied canonical template with private static fields intact.
- `validate_recipe_v2_group_training_contract()` validates both static input fields and rendered shared output fields.
- `public_safe_contract_projection()` is used only in tests and privacy gates to assert that path/static values are not copied into public projections.

- [ ] **Step 1: Write the failing static-field tests**

Add this test body to `tests/stage6/test_p6_history_training_contract.py`:

```python
def test_training_template_requires_real_training_and_all_static_fields(tmp_path: Path) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    template = _training_template(private_root)

    validated = validate_recipe_v2_training_template(template, private_root=private_root)

    assert validated["training_required"] is True
    assert validated["training_source_kind"] == "selected_candidate_finetune"
    assert set(validated["training_parameters"]) == set(REQUIRED_TRAINING_PARAMETER_KEYS)
    assert validated["training_parameters"]["epochs"] == 2
    assert validated["training_parameters"]["batch_size"] == 1
```

Add parametrized rejection coverage:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_training_required",
        "false_training_required",
        "missing_training_source_kind",
        "missing_base_checkpoint_path",
        "missing_dataset_root",
        "missing_pyramid_config_path",
        "missing_training_parameters",
        "empty_training_parameters",
        "missing_training_mode",
        "zero_epochs",
        "negative_learning_rate",
        "zero_batch_size",
        "outside_private_root",
        "symlinked_training_input",
    ],
)
def test_training_template_rejects_incomplete_or_unsafe_static_contract(
    tmp_path: Path, mutation: str
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    template = _mutated_training_template(private_root, mutation)

    with pytest.raises(P6HistoryTrainingContractError) as captured:
        validate_recipe_v2_training_template(template, private_root=private_root)

    assert captured.value.category == "history_execution_invalid"
```

Helper requirements in this test file:

```python
def _private_root_with_training_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "private-history"
    (root / "checkpoints").mkdir(parents=True)
    (root / "datasets" / "coptv2x").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "checkpoints" / "base.ckpt").write_text("base\n", encoding="utf-8")
    (root / "configs" / "pyramid.py").write_text("config\n", encoding="utf-8")
    return root

def _training_template(private_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "stage5_source_contract_v1",
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "artifact_id": "pyramid-16-32-64",
        "source_status": "ready",
        "source_evidence_sha256": _sha({"evidence": "training"}),
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "base_checkpoint_path": str(private_root / "checkpoints" / "base.ckpt"),
        "dataset_root": str(private_root / "datasets" / "coptv2x"),
        "pyramid_config_path": str(private_root / "configs" / "pyramid.py"),
        "training_parameters": {
            "training_mode": "finetune_selected_width",
            "epochs": 2,
            "seed": 20260821,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "batch_size": 1,
            "dataset_split": "trainval_coptv2x",
            "checkpoint_selection": "best_ap70",
            "freeze_policy": "pyramid_backbone_partial",
        },
        "dynamic_materialization_recipe": _recipe_v2(),
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_training_contract.py::test_training_template_requires_real_training_and_all_static_fields -q`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `framework.stage6.p6_history_training_contract_v1`.

- [ ] **Step 3: Write minimal implementation**

Create `p6_history_training_contract_v1.py` with constants above and these implementation rules:

```python
def validate_recipe_v2_training_template(
    template: Mapping[str, Any],
    *,
    private_root: Path,
) -> dict[str, Any]:
    contract = _detached_mapping(template)
    _require_training_required_true(contract)
    _require_training_source_kind(contract)
    _require_training_parameters(contract)
    _validate_static_training_paths(contract, private_root)
    return contract
```

`_validate_static_training_paths()` must resolve `base_checkpoint_path` and `pyramid_config_path` as existing files, `dataset_root` as an existing directory, reject symlink components, reject non-absolute values, and require resolved paths beneath `private_root`.

- [ ] **Step 4: Run tests to verify contract validation passes**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_training_contract.py -q`

Expected: PASS for the new file; existing binding/registry tests may still fail until the integration steps in this task are complete.

- [ ] **Step 5: Integrate binding template validation**

Modify `framework/stage6/p6_history_binding_v1.py` so `build_history_binding()` validates recipe-v2 templates immediately after `_discover_source_contract(root)`:

```python
if _is_recipe_v2_source_template(source_contract):
    source_contract = validate_recipe_v2_training_template(
        source_contract,
        private_root=root,
    )
```

Also update `public_binding_projection()` tests to prove the projection has only existing public keys and no `base_checkpoint_path`, `dataset_root`, `pyramid_config_path`, `training_parameters`, `gpu_policy`, `execution_interface`, `source_contract_template`, or private component paths.

Keep binding discovery's existing `source_status == "ready"` trust boundary. Add a RED/GREEN test
that a `ready` source template carrying recipe-v2 is discovered and validated, while an unrelated
registry row whose status is only `materializable` is not promoted to the authoritative binding
template. Registry-v2 may emit materializable group rows later; that does not broaden bootstrap
template discovery.

- [ ] **Step 6: Integrate registry group validation**

Modify `framework/stage6/p6_history_registry_v1.py` in `_materialize_groups()` after recipe-v2 `contract.update(...)` and before `_canonical_sha(contract)`:

```python
@dataclass(frozen=True)
class ValidatedSourceTemplate:
    private_root: Path
    template: Mapping[str, Any]
    recipe: Mapping[str, Any]

def _validate_template(binding: Mapping[str, Any]) -> ValidatedSourceTemplate:
    private_root = _resolve_existing_root(
        binding.get("private_root", ""), "history binding root"
    )
    ...
    return ValidatedSourceTemplate(
        private_root=private_root,
        template=validated_template,
        recipe=recipe,
    )
```

Then consume that exact return type in `materialize_history_registry()` and `_materialize_groups()`:

```python
validated = _validate_template(binding)
groups = _materialize_groups(
    plan_mapping,
    validated.template,
    validated.recipe,
    validated.private_root,
    local_output_root,
    output_path,
)

if recipe_version == RECIPE_V2:
    contract = validate_recipe_v2_group_training_contract(
        contract,
        private_root=private_root,
        local_output_root=local_output_root,
        group_id=group_id,
    )
```

Do not use an undefined helper or infer the private root from the template. `_validate_template(binding)` already has the authoritative `binding["private_root"]`; preserve it explicitly in `ValidatedSourceTemplate`. This is a focused split because `p6_history_registry_v1.py` is already near the file-size ceiling.

- [ ] **Step 7: Add registry RED/GREEN coverage**

Add these assertions to `tests/stage6/test_p6_history_registry.py`:

```python
def test_recipe_v2_registry_preserves_training_fields_and_rehashes_contract(
    tmp_path: Path,
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    binding = _binding_with_recipe_v2_training_template(private_root)
    registry = materialize_history_registry(
        _plan(("fp16", "int8")),
        binding,
        tmp_path / "ignored-output",
    )

    first = registry["groups"][0]["source_contract"]
    assert first["training_required"] is True
    assert first["training_source_kind"] == "selected_candidate_finetune"
    assert set(first["training_parameters"]) == set(REQUIRED_TRAINING_PARAMETER_KEYS)
    assert first["base_checkpoint_path"].startswith(str(private_root))
    assert first["dataset_root"].startswith(str(private_root))
    assert first["pyramid_config_path"].startswith(str(private_root))
    assert registry["groups"][0]["source_contract_sha256"] == _sha(first)
```

Add rejection tests for missing/false `training_required`, missing `pyramid_config_path`, static path outside root, shared output outside local root, and cross-group output collision.

- [ ] **Step 8: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_training_contract.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_history_registry.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Quality and privacy gates**

Run:

```bash
python -m ruff check \
  framework/stage6/p6_history_training_contract_v1.py \
  framework/stage6/p6_history_binding_v1.py \
  framework/stage6/p6_history_registry_v1.py \
  tests/stage6/test_p6_history_training_contract.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_history_registry.py
git diff --check
python - <<'PY'
from pathlib import Path
blocked = ("base_checkpoint_path\": \"/", "dataset_root\": \"/", "pyramid_config_path\": \"/", "GPU-")
for path in Path("docs").rglob("*.md"):
    if "docs/superpowers/plans/" in path.as_posix():
        continue
    text = path.read_text(encoding="utf-8")
    if any(token in text for token in blocked):
        raise SystemExit(f"private-looking value in {path}")
PY
```

Expected: all commands exit 0.

- [ ] **Step 10: Review boundary**

Review only:

- new training contract validator,
- binding integration,
- registry integration,
- tests in this task.

Reject if v1/static source contracts are forced to include recipe-v2 training fields, if public projection copies private fields, or if `p6_history_binding_v1.py`/`p6_history_registry_v1.py` grows further without a split.

- [ ] **Step 11: Commit**

```bash
git add \
  framework/stage6/p6_history_training_contract_v1.py \
  framework/stage6/p6_history_binding_v1.py \
  framework/stage6/p6_history_registry_v1.py \
  tests/stage6/test_p6_history_training_contract.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_history_registry.py
git commit -m "feat: enforce P6 training source contracts"
```

---

### Task 2: Self-Contained Private Source Wrapper Profile and Provisioning

**Files:**

- Create: `framework/stage6/p6_source_wrapper_profile_v1.py`
- Create: `tools/release/render_p6_source_wrapper.py`
- Modify: `framework/stage6/p6_runner_template_validator_v1.py`
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `tools/release/provision_p6_full_chain_local_config.py`
- Test: `tests/stage6/test_p6_source_wrapper_profile.py`
- Test: `tests/release/test_render_p6_source_wrapper.py`
- Test: `tests/stage6/test_p6_runner_template_validator.py`
- Test: `tests/stage6/test_p6_history_binding.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces:**

- Consumes: `ValidatedRunnerTemplate`, the validated `source_materialization` stage argv, the ignored private runner template, and the canonical environment-key tuple exported by `p6_history_binding_v1` from its existing `WRAPPER_ENVIRONMENT_SPEC`.
- Produces:

```python
# Import this single source of truth from p6_history_binding_v1; do not redefine it.
from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS

@dataclass(frozen=True)
class ValidatedSourceWrapper:
    executable: Path
    argv_shape: tuple[str, ...]
    marker_basename: str

@dataclass(frozen=True)
class SourceWrapperRenderPlan:
    destination: Path
    wrapper_kind: Literal["repo_cwd_exec_v1"]
    implementation: Path
    implementation_cwd: Path
    expected_bytes: bytes
    expected_sha256: str

class P6SourceWrapperProfileError(ValueError):
    category: str

def render_self_contained_source_wrapper(
    profile: Mapping[str, Any],
    *,
    history_root: Path,
) -> ValidatedSourceWrapper: ...

def validate_self_contained_source_wrapper(
    validated_template: ValidatedRunnerTemplate,
    *,
    source_wrapper_profile: Path,
) -> ValidatedSourceWrapper: ...
```

- `render_self_contained_source_wrapper()` is the deterministic deployment mechanism for the actual private wrapper. It reads an ignored private profile with exactly these keys:

```yaml
schema_version: p6_private_source_wrapper_profile_v1
wrapper_kind: repo_cwd_exec_v1
destination_relative_path: documented-stage5-chain/stage5_materialize_round_sources_v1.sh
implementation_relative_path: private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh
implementation_cwd_relative_path: private-relocated-history-repo
```

All three relative paths live only in the ignored private profile and generated wrapper. They are not written to public docs, public projections, stderr, or tracked artifacts. The renderer rejects absolute paths, `..`, symlinks, shell tokens, destinations outside the private Git root, and any destination basename other than `stage5_materialize_round_sources_v1.sh`.
- Fresh private deployment must copy the minimal verified historical implementation repository/module tree, not only the shell entrypoint, beneath a non-marker private subroot such as `private-relocated-history-repo/`. The relocated entrypoint uses a non-marker path such as `private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh`; its validator/import siblings remain available from `private-relocated-history-repo/`. Omit the raw source marker from the relocated tree, then render the unique marker wrapper at `documented-stage5-chain/stage5_materialize_round_sources_v1.sh`.
- The generated wrapper must be deterministic `/bin/sh`, require `P6_HISTORY_PRIVATE_ROOT`, resolve only the validated relative destination/implementation/cwd under the history root, `cd` to the private implementation repo cwd, and `exec` the relocated original historical materializer with `"$@"`.
- The generated wrapper must not use `/usr/bin/env python3`, Python module imports, `module_name`, `callable_name`, `PYTHONPATH`, `shell=True`, or an unknown runtime.
- Destination write rule: destination must be absent or byte-identical to the expected generated wrapper. Never overwrite a differing existing historical marker.
- `validate_self_contained_source_wrapper()` must recompute expected wrapper bytes from the ignored profile and compare digest/content with the source stage executable. It must not infer self-contained behavior from basename/executable alone.
- It must not require public code to inspect private implementation internals, set `PYTHONPATH`, widen env, call shell, or change cwd to the history root.
- Add `EXPECTED_HISTORY_ENV_KEYS = tuple(WRAPPER_ENVIRONMENT_SPEC)` in `p6_history_binding_v1.py`; wrapper validation and measurement both import this value. Add an assertion that the validated template environment keys equal this tuple's set so provisioning and runtime cannot drift to separate five-key definitions.

- [ ] **Step 0: Write RED wrapper renderer tests**

Add to `tests/release/test_render_p6_source_wrapper.py`:

```python
def test_renderer_writes_self_contained_wrapper_from_private_profile(tmp_path: Path) -> None:
    history_root = _private_git_root(tmp_path)
    profile = _write_private_wrapper_profile(
        tmp_path / "ignored-inputs" / "source-wrapper-profile.yaml",
        implementation_relative_path="private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh",
        implementation_cwd_relative_path="private-relocated-history-repo",
    )
    _write_relocated_materializer(
        history_root / "private-relocated-history-repo" / "bin" / "stage5_materialize_round_sources_v1.original.sh"
    )

    wrapper = render_self_contained_source_wrapper(
        yaml.safe_load(profile.read_text(encoding="utf-8")),
        history_root=history_root,
    )

    assert wrapper.executable == history_root / "documented-stage5-chain" / "stage5_materialize_round_sources_v1.sh"
    assert wrapper.executable.is_file()
    assert wrapper.executable.stat().st_mode & 0o111
    body = wrapper.executable.read_text(encoding="utf-8")
    assert body.startswith("#!/bin/sh\n")
    assert "P6_HISTORY_PRIVATE_ROOT" in body
    assert 'exec "$implementation" "$@"' in body
    assert "/usr/bin/env python3" not in body
    assert "PYTHONPATH" not in wrapper.executable.read_text(encoding="utf-8")
    assert str(history_root) not in wrapper.executable.read_text(encoding="utf-8")
```

Add rejection coverage:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "absolute_destination",
        "destination_parent_escape",
        "wrong_marker_basename",
        "absolute_implementation",
        "implementation_parent_escape",
        "implementation_cwd_parent_escape",
        "raw_marker_implementation_path",
        "differing_existing_marker",
        "symlink_destination",
    ],
)
def test_renderer_rejects_unsafe_private_wrapper_profile(tmp_path: Path, mutation: str) -> None:
    history_root = _private_git_root(tmp_path)
    profile = _mutated_private_wrapper_profile(mutation)

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        render_self_contained_source_wrapper(profile, history_root=history_root)

    assert captured.value.category == "history_execution_invalid"
```

- [ ] **Step 0b: Run renderer RED tests**

Run:

```bash
PYTHONPATH=. pytest tests/release/test_render_p6_source_wrapper.py -q
```

Expected: FAIL with missing renderer function or CLI.

- [ ] **Step 1: Write RED provisioning/profile tests**

Add to `tests/stage6/test_p6_source_wrapper_profile.py`:

```python
def test_source_wrapper_profile_accepts_marker_stage_under_private_root(tmp_path: Path) -> None:
    template, history_root, profile = _write_valid_template_with_generated_wrapper(tmp_path)
    validated = validate_pre_provision_runner_template(template, history_root)

    wrapper = validate_self_contained_source_wrapper(
        validated,
        source_wrapper_profile=profile,
    )

    assert wrapper.executable == history_root / "documented-stage5-chain" / "stage5_materialize_round_sources_v1.sh"
    assert wrapper.argv_shape == (
        "--request",
        "<absolute-private-request-json>",
        "--model",
        "pyramid",
        "--group-id",
        "<canonical-pyramid-group-id>",
        "--gpu",
        "<validated-binding-gpu-index>",
    )
```

Add rejection cases:

```python
@pytest.mark.parametrize(
    "mutation",
    ["source_not_first_stage", "wrong_basename", "outside_root", "symlink", "not_executable"],
)
def test_source_wrapper_profile_rejects_non_self_contained_binding_shape(
    tmp_path: Path, mutation: str
) -> None:
    template, history_root, profile = _write_mutated_template(tmp_path, mutation)
    validated = validate_pre_provision_runner_template(template, history_root)

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        validate_self_contained_source_wrapper(
            validated,
            source_wrapper_profile=profile,
        )

    assert captured.value.category == "history_execution_invalid"
```

- [ ] **Step 2: Write RED black-box repo-cwd determinism test**

Add a subprocess test that runs only a synthetic generated wrapper and relocated implementation, not the production historical chain:

```python
def test_source_wrapper_execs_relocated_implementation_from_private_cwd_with_exact_env(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request = round_root / "measurement-request.json"
    request.write_text("{}", encoding="utf-8")
    task_state = round_root / "state" / "task-state.json"
    task_state.parent.mkdir()
    task_state.write_text("{}", encoding="utf-8")
    implementation = _write_relocated_materializer(
        history_root / "private-relocated-history-repo" / "bin" / "stage5_materialize_round_sources_v1.original.sh",
        required_sibling_module=(
            history_root / "private-relocated-history-repo" / "history_contract_validator.py"
        ),
    )
    profile = _wrapper_profile(
        implementation_relative_path="private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh",
        implementation_cwd_relative_path="private-relocated-history-repo",
    )
    wrapper = render_self_contained_source_wrapper(
        profile,
        history_root=history_root,
    )
    env = {
        "CUDA_VISIBLE_DEVICES": "101,103,107",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(history_root),
        "P6_HISTORY_TASK_STATE": str(task_state),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
    }

    completed = subprocess.run(
        [
            str(wrapper.executable),
            "--request",
            str(request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|16x32x64",
            "--gpu",
            "101",
        ],
        cwd=round_root,
        env=env,
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert set(env) == set(EXPECTED_HISTORY_ENV_KEYS)
    relocated_repo = implementation.parents[1]
    assert (relocated_repo / "observed-cwd.txt").read_text(encoding="utf-8") == str(
        relocated_repo
    )
    assert (relocated_repo / "sibling-import-ok.txt").read_text(encoding="utf-8") == "ok"
```

Add the negative black-box:

```python
def test_validator_rejects_differing_existing_historical_marker(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    marker = history_root / "documented-stage5-chain" / "stage5_materialize_round_sources_v1.sh"
    marker.parent.mkdir(parents=True)
    marker.write_text("#!/bin/sh\nexec /private/raw/source \"$@\"\n", encoding="utf-8")
    marker.chmod(0o700)
    profile = _wrapper_profile(
        implementation_relative_path="private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh",
        implementation_cwd_relative_path="private-relocated-history-repo",
    )

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        render_self_contained_source_wrapper(profile, history_root=history_root)

    assert captured.value.category == "history_execution_invalid"
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/stage6/test_p6_runner_template_validator.py::test_validator_returns_private_copy_for_exact_template \
  -q
```

Expected: FAIL with missing `p6_source_wrapper_profile_v1` module or missing exact argv profile.

- [ ] **Step 4: Implement minimal wrapper profile validator**

Create `p6_source_wrapper_profile_v1.py` and add the renderer plus this validation flow:

```python
def render_self_contained_source_wrapper(
    profile: Mapping[str, Any],
    *,
    history_root: Path,
) -> ValidatedSourceWrapper:
    plan = _validate_render_profile(profile, history_root=history_root)
    _write_wrapper_script(plan, history_root=history_root)
    return _validate_rendered_wrapper(plan.destination, history_root=history_root)
```

The wrapper body must be deterministic and public-safe:

```sh
#!/bin/sh
set -eu
: "${P6_HISTORY_PRIVATE_ROOT:?}"
root="$(cd "${P6_HISTORY_PRIVATE_ROOT}" && pwd -P)"
implementation="${root}/private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh"
implementation_cwd="${root}/private-relocated-history-repo"
[ -f "$implementation" ] && [ -x "$implementation" ]
cd "$implementation_cwd"
implementation_cwd="$(pwd -P)"
exec "$implementation" "$@"
```

Only the ignored generated wrapper contains the private relative implementation/cwd values. Public tests assert tracked diffs and public docs do not contain real private absolute values. The validator recomputes these exact bytes from the ignored profile and compares them to the marker wrapper content before accepting the source stage.

```python
def validate_self_contained_source_wrapper(
    validated_template: ValidatedRunnerTemplate,
    *,
    source_wrapper_profile: Path,
) -> ValidatedSourceWrapper:
    source_argv = tuple(validated_template.stage_argv["source_materialization"])
    executable = Path(source_argv[0])
    if executable.name != "stage5_materialize_round_sources_v1.sh":
        _invalid()
    _require_regular_executable_under_root(executable, validated_template.history_root)
    expected = expected_wrapper_bytes_from_profile(
        source_wrapper_profile,
        history_root=validated_template.history_root,
    )
    if executable.read_bytes() != expected:
        _invalid()
    if tuple(validated_template.stage_argv) != (
        "source_materialization",
        "quantization",
        "performance",
        "ap",
        "finalization",
    ):
        _invalid()
    return ValidatedSourceWrapper(
        executable=executable,
        argv_shape=(
            "--request",
            "<absolute-private-request-json>",
            "--model",
            "pyramid",
            "--group-id",
            "<canonical-pyramid-group-id>",
            "--gpu",
            "<validated-binding-gpu-index>",
        ),
        marker_basename="stage5_materialize_round_sources_v1.sh",
    )
```

- [ ] **Step 5: Integrate provisioning**

Change the exact `materialize_full_chain_binding()` signature to add an optional keyword-only profile path while preserving legacy callers:

```python
def materialize_full_chain_binding(
    legacy_local_config: Path,
    runner_template: Path,
    local_output_root: Path,
    binding_output: Path,
    config_output: Path,
    gpu_probe: GpuProbe,
    *,
    source_wrapper_profile: Path | None = None,
) -> dict[str, Any]: ...
```

`source_wrapper_profile` is optional at the Python interface only for legacy/static compatibility. It is mandatory when the discovered source template uses recipe-v2; recipe-v2 without this profile raises stable `history_execution_invalid`.

In `materialize_full_chain_binding()` call:

```python
validated_template = validate_pre_provision_runner_template(
    runner_template_path,
    history_root,
)
if _is_recipe_v2_source_template(source_contract):
    if source_wrapper_profile is None:
        raise FullChainBootstrapError("history_execution_invalid")
    validate_self_contained_source_wrapper(
        validated_template,
        source_wrapper_profile=source_wrapper_profile,
    )
```

Then validate the source contract template with Task 1’s `validate_recipe_v2_training_template()` before writing the binding/config pair. Existing generated bindings without the complete training contract must fail provisioning; regenerated private input must pass.

Add `--source-wrapper-profile` to `tools/release/provision_p6_full_chain_local_config.py`. It is optional at argparse level to keep legacy/static tests compatible, but recipe-v2 missing profile must fail with stable `history_execution_invalid`. If supplied, run `render_self_contained_source_wrapper()` before template validation, then validate that the runner template source stage resolves to the rendered destination. The renderer output stays under the ignored private history root and is never copied into public artifacts.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/release/test_render_p6_source_wrapper.py \
  tests/stage6/test_p6_runner_template_validator.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_full_chain_bootstrap.py \
  tests/release/test_provision_p6_full_chain_local_config.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Quality and privacy gates**

Run:

```bash
python -m ruff check \
  framework/stage6/p6_source_wrapper_profile_v1.py \
  tools/release/render_p6_source_wrapper.py \
  framework/stage6/p6_runner_template_validator_v1.py \
  framework/stage6/p6_history_binding_v1.py \
  framework/stage6/p6_full_chain_bootstrap_v1.py \
  tools/release/provision_p6_full_chain_local_config.py \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/release/test_render_p6_source_wrapper.py \
  tests/stage6/test_p6_runner_template_validator.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_full_chain_bootstrap.py \
  tests/release/test_provision_p6_full_chain_local_config.py
git diff --check
```

Expected: all commands exit 0 and test stderr contains only stable categories such as `contract_error`, `execution_failed`, `history_execution_invalid`, or `history_recipe_derivation_invalid`.

- [ ] **Step 8: Review boundary**

Review only wrapper profile/provisioning behavior. Reject if public code gains historical import knowledge, `PYTHONPATH`, `shell=True`, public shell-string construction, ambient environment dependency, or private wrapper internals. Direct argv execution of the validated private marker wrapper remains allowed.

- [ ] **Step 9: Commit**

```bash
git add \
  framework/stage6/p6_source_wrapper_profile_v1.py \
  tools/release/render_p6_source_wrapper.py \
  framework/stage6/p6_runner_template_validator_v1.py \
  framework/stage6/p6_history_binding_v1.py \
  framework/stage6/p6_full_chain_bootstrap_v1.py \
  tools/release/provision_p6_full_chain_local_config.py \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/release/test_render_p6_source_wrapper.py \
  tests/stage6/test_p6_runner_template_validator.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_full_chain_bootstrap.py \
  tests/release/test_provision_p6_full_chain_local_config.py
git commit -m "feat: validate P6 private source wrapper profile"
```

---

### Task 3: Projection Preservation, Static Drift Rejection, and Hash Gates

**Files:**

- Modify: `framework/stage6/p6_history_source_materialization_v1.py`
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Test: `tests/stage6/test_p6_history_source_materialization.py`
- Test: `tests/stage6/test_coptv2x_h800_search.py`

**Interfaces:**

- Consumes: `ProjectedSourceRequest`, recipe-v2 group contracts emitted by Task 1 registry, and `build_measurement_request()`.
- Produces:

```python
def project_source_materialization_request(
    request: Mapping[str, Any],
) -> ProjectedSourceRequest: ...

def validate_projected_training_marker_pairs(
    request: Mapping[str, Any],
) -> tuple[tuple[Path, Path], ...]: ...
```

- Projection must flatten recipe-v2 `shared_source_paths` into row `source_contract`, preserve all static training fields, strip legacy dynamic aliases, reject mixed legacy and recipe-v2 rows, reject same-group static training drift, reject same-group shared-path drift, reject cross-group path collision, and recompute `source_contract_sha256`, `row_sha256`, and `measurement_request_sha256`.
- `validate_projected_training_marker_pairs()` returns ordered unique `(training_done_marker, source_done_marker)` pairs by canonical group order. It permits the same pair to be shared by q-modes of the same group, but rejects missing/relative/malformed paths, training/source self-aliasing, different pairs for one group, or any marker collision across groups.

- [ ] **Step 1: Write RED projection-preservation tests**

Add to `tests/stage6/test_p6_history_source_materialization.py`:

```python
def test_projection_preserves_complete_training_contract_and_rehashes() -> None:
    request = _request_with_complete_training_contract()

    projected = project_source_materialization_request(request)

    assert request["measurement_request_sha256"] != projected.request["measurement_request_sha256"]
    for row in projected.request["rows"]:
        contract = row["source_contract"]
        assert contract["training_required"] is True
        assert contract["training_source_kind"] == "selected_candidate_finetune"
        assert contract["base_checkpoint_path"] == "/private/synthetic/base/model.ckpt"
        assert contract["dataset_root"] == "/private/synthetic/dataset"
        assert contract["pyramid_config_path"] == "/private/synthetic/configs/pyramid.py"
        assert set(contract["training_parameters"]) == {
            "training_mode",
            "epochs",
            "seed",
            "optimizer",
            "learning_rate",
            "batch_size",
            "dataset_split",
            "checkpoint_selection",
            "freeze_policy",
        }
        assert "shared_source_paths" not in contract
        assert "training_path" not in contract
        assert row["source_contract_sha256"] == _sha(contract)
        assert projected.request["row_sha256"][row["row_id"]] == _sha(row)
```

Add drift rejections:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "same_group_training_parameters_drift",
        "same_group_base_checkpoint_drift",
        "same_group_dataset_drift",
        "same_group_pyramid_config_drift",
        "same_group_training_required_false",
        "missing_training_done_marker",
        "missing_source_done_marker",
        "shared_path_collision_across_groups",
        "mixed_legacy_and_recipe_v2_rows",
        "stale_source_contract_hash",
    ],
)
def test_projection_rejects_untrusted_training_or_shared_path_drift(mutation: str) -> None:
    request = _mutated_request_with_training_contract(mutation)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
```

- [ ] **Step 2: Write RED atomic request test**

Add to `tests/stage6/test_coptv2x_h800_search.py`:

```python
def test_round_request_write_uses_projected_training_hash_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, local = _framework_mode_setup(tmp_path)
    captured_request: dict[str, Any] = {}

    def command_runner(argv: tuple[str, ...], cwd: Path) -> int:
        if argv[1] == "local_build_registry.py":
            _write_recipe_v2_training_registry(Path(argv[3]))
            return 0
        request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        captured_request.update(request)
        _write_success_feedback(Path(argv[2]), request)
        return 0

    state = run_p6_coptv2x_search(contract, local, "rev-projected-training", command_runner)

    assert state.completed_rounds == 4
    first_row = captured_request["rows"][0]
    assert first_row["source_contract"]["training_required"] is True
    assert captured_request["row_sha256"][first_row["row_id"]] == _sha(first_row)
    body = {key: value for key, value in captured_request.items() if key != "measurement_request_sha256"}
    assert captured_request["measurement_request_sha256"] == _sha(body)
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_source_materialization.py::test_projection_preserves_complete_training_contract_and_rehashes \
  tests/stage6/test_coptv2x_h800_search.py::test_round_request_write_uses_projected_training_hash_only \
  -q
```

Expected: FAIL because projection currently preserves only a partial training fixture and does not validate the full static training field set.

- [ ] **Step 4: Implement projection validation**

In `project_source_materialization_request()` after `flat_contract.update(shared_paths)`:

```python
flat_contract = validate_projected_training_contract(
    flat_contract,
    group_id=group_id,
)
```

Add `validate_projected_training_contract()` in `p6_history_training_contract_v1.py`:

```python
def validate_projected_training_contract(
    contract: Mapping[str, Any],
    *,
    group_id: str,
) -> dict[str, Any]:
    detached = _detached_mapping(contract)
    _require_training_required_true(detached)
    _require_training_source_kind(detached)
    _require_training_parameters(detached)
    _require_stage_widths_match_group(detached, group_id)
    _require_absolute_marker_paths(detached)
    return detached
```

Keep this projected validator lexical only; registry/provisioning already checked private-root existence and symlink safety.

- [ ] **Step 5: Implement marker extraction**

In `p6_history_source_materialization_v1.py`:

```python
def validate_projected_training_marker_pairs(
    request: Mapping[str, Any],
) -> tuple[tuple[Path, Path], ...]:
    projected = _validated_request_copy(request)
    pairs_by_group: dict[str, tuple[Path, Path]] = {}
    owners_by_marker: dict[Path, str] = {}
    for row in projected["rows"]:
        contract = row["source_contract"]
        group_id = str(row["group_id"])
        pair = (
            Path(contract["training_done_marker"]),
            Path(contract["source_done_marker"]),
        )
        _validate_marker_pair(group_id, pair, pairs_by_group, owners_by_marker)
    return tuple(pairs_by_group[group_id] for group_id in sorted(pairs_by_group))
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/stage6/test_coptv2x_h800_search.py::test_round_request_write_uses_projected_training_hash_only \
  tests/stage6/test_coptv2x_h800_search.py::test_run_p6_projection_failure_is_atomic_before_measurement_request_write \
  -q
```

Expected: PASS.

- [ ] **Step 7: Quality and privacy gates**

Run:

```bash
python -m ruff check \
  framework/stage6/p6_history_source_materialization_v1.py \
  framework/stage6/p6_history_training_contract_v1.py \
  framework/stage6/coptv2x_h800_search_v2.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/stage6/test_coptv2x_h800_search.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Review boundary**

Review only projection/hashing changes and round request atomicity. Reject if there are two canonical request hashes, if original unprojected request can reach measurement, or if projection mutates caller input.

- [ ] **Step 9: Commit**

```bash
git add \
  framework/stage6/p6_history_training_contract_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  framework/stage6/coptv2x_h800_search_v2.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/stage6/test_coptv2x_h800_search.py
git commit -m "fix: preserve P6 training fields through projection"
```

---

### Task 4: Runtime Environment Boundary and Marker-Gated Downstream Stages

**Files:**

- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Modify: `framework/stage6/p6_history_source_materialization_v1.py`
- Test: `tests/stage6/test_p6_history_measurement.py`
- Test: `tests/stage6/test_p6_history_source_materialization.py`
- Test: `tests/release/test_p6_history_execution_adapters.py`

**Interfaces:**

- Consumes: projected request written by `run_history_measurement_batch()`, the binding-owned `EXPECTED_HISTORY_ENV_KEYS`, `run_source_invocations()`, and marker pairs from `validate_projected_training_marker_pairs()`.
- Produces:

```python
def assert_source_markers_absent_before_invocation(
    request: Mapping[str, Any],
    *,
    local_output_root: Path,
) -> None: ...

def assert_training_and_source_markers_complete(
    request: Mapping[str, Any],
    *,
    local_output_root: Path,
    request_path: Path,
) -> None: ...
```

- `run_history_measurement_batch()` must call activation, source materializer, marker gate, then quantization/performance/AP/finalization.
- Marker gate must run after all source invocations return 0 and before any downstream wrapper executes.
- Ordering evidence is an explicit fresh-run filesystem contract: both exact marker leaves must be absent immediately before source invocation; after source returns, both must be regular non-symlink files; `measurement-request` mtime must be no later than the training marker, and the training marker mtime must be no later than the source marker. Equal timestamps are allowed for coarse filesystems. Preflight and runtime both reject preexisting marker leaves, so an old marker cannot satisfy this evidence.
- Missing `training_done_marker` or `source_done_marker` maps to `history_execution_invalid`; source wrapper nonzero/import failure maps to redacted `history_execution_invalid` at the source adapter boundary or `history_execution_failed` if raised by the generic private stage executor. Public state must not include private paths, argv, traceback, env, or raw stderr.

- [ ] **Step 1: Write RED exact-env test**

Add to `tests/stage6/test_p6_history_measurement.py`:

```python
def test_measurement_source_wrapper_gets_round_cwd_exact_five_key_env(
    tmp_path: Path,
) -> None:
    binding = _binding_with_recipe_v2_training_contract(tmp_path / "private")
    request = project_source_materialization_request(_request_with_training_contract()).request
    runner = _MarkerWritingRunner()
    gpu_probe = _GpuProbeOk()

    run_history_measurement_batch(request, binding, tmp_path / "private" / "private-runs" / "0", runner, gpu_probe)

    source_calls = [call for call in runner.calls if call.argv[0].endswith("stage5_materialize_round_sources_v1.sh")]
    assert source_calls
    for call in source_calls:
        assert call.cwd.name == "0"
        assert set(call.env) == set(EXPECTED_HISTORY_ENV_KEYS)
        assert "PYTHONPATH" not in call.env
        assert call.shell is False
        assert call.argv[1::2] == ("--request", "--model", "--group-id", "--gpu")
```

- [ ] **Step 2: Write RED marker-gate tests**

Add:

```python
@pytest.mark.parametrize("missing_marker", ["training_done_marker", "source_done_marker"])
def test_missing_source_marker_blocks_downstream_stages(
    tmp_path: Path, missing_marker: str
) -> None:
    binding = _binding_with_recipe_v2_training_contract(tmp_path / "private")
    request = project_source_materialization_request(_request_with_training_contract()).request
    runner = _SelectiveMarkerRunner(missing_marker=missing_marker)

    with pytest.raises(P6HistoryMeasurementError) as captured:
        run_history_measurement_batch(request, binding, tmp_path / "private" / "private-runs" / "0", runner, _GpuProbeOk())

    assert captured.value.category == "history_execution_invalid"
    assert [call.stage for call in runner.calls if call.stage in ("quantization", "performance", "ap", "finalization")] == []
```

Add a reversed-order case that creates both fresh files but sets the source marker mtime before the
training marker with `os.utime()`; assert the same stable failure and zero downstream calls. Add a
preexisting-marker case and prove source invocation itself is not started.

Add source failure redaction:

```python
def test_source_import_failure_is_redacted_and_stops_downstream(tmp_path: Path) -> None:
    binding = _binding_with_recipe_v2_training_contract(tmp_path / "private")
    request = project_source_materialization_request(_request_with_training_contract()).request
    runner = _FailingSourceRunner(stderr="Traceback private/root/module.py")

    with pytest.raises(P6HistoryMeasurementError) as captured:
        run_history_measurement_batch(request, binding, tmp_path / "private" / "private-runs" / "0", runner, _GpuProbeOk())

    assert captured.value.category in {"history_execution_invalid", "history_execution_failed"}
    assert "private" not in str(captured.value).lower()
    assert "traceback" not in str(captured.value).lower()
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_measurement.py::test_measurement_source_wrapper_gets_round_cwd_exact_five_key_env \
  tests/stage6/test_p6_history_measurement.py::test_missing_source_marker_blocks_downstream_stages \
  -q
```

Expected: FAIL because marker gate is not yet enforced before downstream stages and env exactness is not centralized.

- [ ] **Step 4: Implement exact environment assertion**

In `p6_history_measurement_v1._render_environment()` after rendering, import the canonical tuple
from `p6_history_binding_v1` (the same owner used by provisioning) and do not redeclare it in the
measurement or wrapper modules:

```python
if set(rendered) != set(EXPECTED_HISTORY_ENV_KEYS):
    raise P6HistoryMeasurementError("history_execution_invalid")
```

Import `EXPECTED_HISTORY_ENV_KEYS` only from `p6_history_binding_v1`.

- [ ] **Step 5: Implement marker gate**

In `p6_history_source_materialization_v1.py`:

```python
def assert_source_markers_absent_before_invocation(
    request: Mapping[str, Any],
    *,
    local_output_root: Path,
) -> None:
    for training_marker, source_marker in validate_projected_training_marker_pairs(request):
        for marker in (training_marker, source_marker):
            _reject_marker_if_symlink_or_outside(
                marker,
                local_output_root=local_output_root,
            )
            if marker.exists() or marker.is_symlink():
                _invalid()

def assert_training_and_source_markers_complete(
    request: Mapping[str, Any],
    *,
    local_output_root: Path,
    request_path: Path,
) -> None:
    for training_marker, source_marker in validate_projected_training_marker_pairs(request):
        for marker in (training_marker, source_marker):
            _reject_marker_if_symlink_or_outside(
                marker,
                local_output_root=local_output_root,
            )
            if not marker.is_file():
                _invalid()
        if not (
            request_path.stat().st_mtime_ns
            <= training_marker.stat().st_mtime_ns
            <= source_marker.stat().st_mtime_ns
        ):
            _invalid()
```

The allowed marker location rule is: marker must be beneath the validated ignored
`local_output_root` used by registry rendering, not guessed beneath the separate historical
`private_root` or the public round directory. It must never point into a tracked public path;
provisioning/registry and the runtime gate independently enforce containment and symlink safety.
At runtime derive `validated_local_output_root` only as the safe resolved parent of the supplied
public round root after `_validate_controller_round_root()` succeeds; require its lexical
components to be non-symlinks, then require every flattened shared output path and marker in the
already-hashed projected request to remain beneath it. Do not infer the root from a marker path.

In `run_history_measurement_batch()`:

```python
assert_source_markers_absent_before_invocation(...)
run_source_invocations(...)
assert_training_and_source_markers_complete(
    canonical_request,
    local_output_root=validated_local_output_root,
    request_path=paths["measurement_request"],
)
for stage in interface["execution_chain"][1:]:
    _execute(...)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/release/test_p6_history_execution_adapters.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Quality and privacy gates**

Run:

```bash
python -m ruff check \
  framework/stage6/p6_history_measurement_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  framework/stage6/p6_source_wrapper_profile_v1.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/release/test_p6_history_execution_adapters.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Review boundary**

Review only runtime boundary and marker gate. Reject if downstream wrappers can run without both markers, if the source stage receives positional legacy argv, if `PYTHONPATH` appears, or if failure messages include private diagnostics.

- [ ] **Step 9: Commit**

```bash
git add \
  framework/stage6/p6_history_measurement_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/release/test_p6_history_execution_adapters.py
git commit -m "fix: gate P6 downstream stages on training markers"
```

---

### Task 5: Zero-Process Preflight and Relaunch Safety

**Files:**

- Create: `tools/release/preflight_p6_materializer_training_bridge.py`
- Create: `tools/release/verify_p6_materializer_training_run.py`
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Test: `tests/release/test_preflight_p6_materializer_training_bridge.py`
- Test: `tests/release/test_verify_p6_materializer_training_run.py`
- Test: `tests/stage6/test_coptv2x_h800_search.py`
- Test: `tests/release/test_run_p6_h800_search.py`

**Interfaces:**

- Consumes: public contract path, local config path, private binding path, runner template path, ignored private source wrapper profile path, optional expected round count.
- Produces:

```python
@dataclass(frozen=True)
class P6MaterializerPreflightReport:
    schema_version: str
    status: Literal["accepted"]
    validated_round_count: int
    wrapper_marker: str
    training_required: bool
    historical_process_launch_count: int
    gpu_probe_count: int

def preflight_materializer_training_bridge(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
    runner_template_path: Path,
    source_wrapper_profile_path: Path,
) -> P6MaterializerPreflightReport: ...

@dataclass(frozen=True)
class P6MaterializerCompletionReport:
    schema_version: str
    status: Literal["completed"]
    completed_rounds: int
    selected_rows: int
    gold176_remeasured_rows: int

def verify_materializer_training_run(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
) -> P6MaterializerCompletionReport: ...
```

- The CLI prints only public-safe JSON with `historical_process_launch_count: 0` and `gpu_probe_count: 0`. `historical_process_launch_count` counts prohibited activation/history/training/materialization/measurement runners only; it does not count unavoidable current-process validation, JSON/YAML parsing, or read-only `git check-ignore` validation used by existing safety gates.
- It validates schemas, private binding target, exact five-key environment, source wrapper marker, static training fields, static path safety, shared output renderability, registry destination, Stage1 manifest leaf, round leaves, request/feedback/task-state/receipt/barrier absence, and no stale state.
- It must not import historical Python modules, execute wrappers, call GPU probes, call TVM/AP/training code, or read raw private logs.
- The post-run verifier must resolve task-state/result/receipt/barrier through the validated binding's exact templates, not `glob`/`rglob` or guessed basenames. It reuses `translate_history_feedback()` and the controller's canonical request/hash rules, checks the five metric fields, validates marker order, proves 16 unique selected rows, and compares their identities against the frozen Gold176 identities.

- [ ] **Step 1: Write RED zero-process preflight tests**

Add to `tests/release/test_preflight_p6_materializer_training_bridge.py`:

```python
def test_preflight_accepts_regenerated_training_binding_without_process_or_gpu(
    tmp_path: Path,
) -> None:
    public_contract, local_config, binding, runner_template, wrapper_profile = _valid_preflight_inputs(tmp_path)

    report = preflight_materializer_training_bridge(
        public_contract_path=public_contract,
        local_config_path=local_config,
        private_binding_path=binding,
        runner_template_path=runner_template,
        source_wrapper_profile_path=wrapper_profile,
    )

    assert report.status == "accepted"
    assert report.validated_round_count == 4
    assert report.wrapper_marker == "stage5_materialize_round_sources_v1.sh"
    assert report.training_required is True
    assert report.historical_process_launch_count == 0
    assert report.gpu_probe_count == 0
```

Add rejection coverage:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "old_binding_missing_training_required",
        "old_binding_false_training_required",
        "missing_static_training_field",
        "wrapper_not_self_contained",
        "source_registry_preexists_as_directory",
        "stage1_manifest_preexists_stale",
        "round_request_preexists",
        "round_feedback_preexists",
        "task_state_preexists",
        "barrier_preexists",
        "output_root_symlink",
    ],
)
def test_preflight_rejects_stale_or_incomplete_private_state_without_launch(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _mutated_preflight_inputs(tmp_path, mutation)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        preflight_materializer_training_bridge(**inputs)

    assert captured.value.failure_code in {"history_execution_invalid", "unsafe_destination", "source_registry_invalid"}
    assert _process_launch_log(tmp_path) == []
    assert _gpu_probe_log(tmp_path) == []
```

- [ ] **Step 2: Write RED relaunch safety tests**

Add to `tests/stage6/test_coptv2x_h800_search.py`:

```python
def test_run_p6_rejects_preexisting_public_search_round_before_measurement(
    tmp_path: Path,
) -> None:
    contract, local = _framework_mode_setup(tmp_path)
    round_root = local.local_output_root / "round-00"
    round_root.mkdir(parents=True)
    (round_root / "measurement_request.json").write_text("stale\n", encoding="utf-8")
    measurement_started = False

    def command_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_started
        if argv[1] == "local_build_registry.py":
            _write_recipe_v2_training_registry(Path(argv[3]))
            return 0
        measurement_started = True
        return 1

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "rev-no-resume", command_runner)

    assert captured.value.failure_code in {"history_execution_invalid", "unsafe_output"}
    assert measurement_started is False
```

The zero-process preflight mutation matrix separately places a stale file at the exact private
receipt/task-state/barrier path resolved from the validated binding template. Assert that a decoy
file under the public `round-00` cannot hide or satisfy that private stale-state check.

Also assert Stage1 manifest is unlinked/reserved before scan and a no-op Stage1 adapter cannot reuse stale manifest:

```python
def test_stage1_scan_must_rewrite_manifest_not_reuse_stale_file(tmp_path: Path) -> None:
    contract, local = _framework_mode_setup(tmp_path)
    assert local.stage2_search_space_path is not None
    local.stage2_search_space_path.write_text("schema: stale\n", encoding="utf-8")

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "rev-stale-stage1", _noop_stage1_runner)

    assert captured.value.failure_code == "stage1_scan_invalid"
```

- [ ] **Step 2b: Write RED exact-layout completion verifier tests**

Add to `tests/release/test_verify_p6_materializer_training_run.py`:

```python
def test_completion_verifier_uses_binding_templates_and_rejects_gold176_remeasurement(
    tmp_path: Path,
) -> None:
    inputs = _completed_four_round_private_run(tmp_path, nonstandard_private_leaf_names=True)

    report = verify_materializer_training_run(**inputs)

    assert report.status == "completed"
    assert report.completed_rounds == 4
    assert report.selected_rows == 16
    assert report.gold176_remeasured_rows == 0

    _replace_selected_row_with_gold176_identity(inputs)
    with pytest.raises(P6CoptV2XExecutionError):
        verify_materializer_training_run(**inputs)
```

Add mutations for a missing exact receipt, missing exact barrier, receipt/request hash drift,
row/source-contract hash drift, duplicate selected row, missing training/source marker, reversed
marker order, absent/extra/non-finite metric, and a decoy `receipt.json` that must not satisfy a
different binding template.

Add focused path-planning tests: planned private leaves may all be absent and still resolve to safe
lexical destinations; a symlinked existing parent or escape is rejected; the runtime resolver still
rejects an absent supplied public round root; and after that public round exists it returns the same
private paths as the planner.

- [ ] **Step 3: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  tests/stage6/test_coptv2x_h800_search.py::test_run_p6_rejects_preexisting_public_search_round_before_measurement \
  tests/stage6/test_coptv2x_h800_search.py::test_stage1_scan_must_rewrite_manifest_not_reuse_stale_file \
  -q
```

Expected: FAIL with missing preflight/verifier modules and incomplete stale round gating.

- [ ] **Step 4: Expose exact round-path resolution and implement preflight**

First expose two immutable seams from `p6_history_measurement_v1.py`:

```python
def plan_validated_history_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    round_index: int,
) -> Mapping[str, Path]: ...

def resolve_validated_history_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    supplied_public_round_root: Path,
    round_index: int,
) -> Mapping[str, Path]: ...
```

`plan_validated_history_round_paths()` validates templates, lexical components that already exist,
containment, uniqueness, and symlink safety while allowing the planned private round root/leaves to
be absent. `resolve_validated_history_round_paths()` retains the current runtime requirement that
the supplied public round root already exists as a safe directory, then delegates private template
resolution to the planner. Do not weaken the runtime check to make preflight work.

Create the CLI with:

```python
def main(argv: Sequence[str] | None = None) -> int:
    # argparse requires --contract, --local-config, --binding,
    # --runner-template, and --source-wrapper-profile.
    try:
        report = preflight_materializer_training_bridge(
            public_contract_path=args.contract,
            local_config_path=args.local_config,
            private_binding_path=args.binding,
            runner_template_path=args.runner_template,
            source_wrapper_profile_path=args.source_wrapper_profile,
        )
    except (P6CoptV2XContractError, P6CoptV2XExecutionError):
        sys.stderr.write("preflight_failed\n")
        return 1
    sys.stdout.write(json.dumps(dataclasses.asdict(report), sort_keys=True) + "\n")
    return 0
```

Use existing validators from Tasks 1 and 2. Do not call `_run_step()`, `GpuProbe.snapshot()`,
`run_history_measurement_batch()`, any historical/training/materialization/measurement executable,
or historical Python imports. Existing read-only `git rev-parse`/`git check-ignore` subprocesses
inside the repository safety validators are allowed and do not increment
`historical_process_launch_count`; no other subprocess is allowed in preflight.

For every configured round, call `plan_validated_history_round_paths()` with the validated
binding/interface and require the exact private round root, measurement-request, task-state,
actual-feedback, receipt, and finalization-barrier leaves to be absent. Separately require each
public search round directory to be absent. Never apply public `round-00` basenames to the private
history tree.

- [ ] **Step 4b: Implement exact-layout completion verification**

Implement the verifier CLI using `load_public_contract()`, `load_local_config()`,
`validate_history_execution_binding()`, `resolve_validated_history_round_paths()` for the completed
public/private round pair, and
`translate_history_feedback()`. Never reconstruct receipt/barrier names, scan directories, emit
private values, or accept a decoy artifact. The CLI prints only the completion dataclass as sorted
JSON and maps invalid private evidence to a stable public failure category.

- [ ] **Step 5: Implement fresh-run/relaunch gates**

In `coptv2x_h800_search_v2.py`:

```python
def _reject_preexisting_public_search_round(round_root: Path) -> None:
    if round_root.exists() or round_root.is_symlink():
        raise P6CoptV2XExecutionError(
            "history_execution_invalid",
            "public search round output already exists",
        )
```

Call this from `_prepare_round_output()` before creating the public search round. Preserve failed
diagnostics by not deleting scoped round roots automatically. Do not put private history leaf names
in this controller gate: preflight resolves those exact paths from the binding, and
`_initialize_private_round()` retains the final just-in-time rejection for the exact private
measurement-request/task-state/result/receipt/barrier leaves.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  tests/stage6/test_coptv2x_h800_search.py \
  tests/release/test_run_p6_h800_search.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Quality and privacy gates**

Run:

```bash
python -m ruff check \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py \
  framework/stage6/coptv2x_h800_search_v2.py \
  framework/stage6/p6_history_measurement_v1.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  tests/stage6/test_coptv2x_h800_search.py \
  tests/release/test_run_p6_h800_search.py
git diff --check
PYTHONPATH=. python tools/release/preflight_p6_materializer_training_bridge.py --help >/tmp/p6-preflight-help.txt
```

Expected: all commands exit 0; help text contains no private paths.

- [ ] **Step 8: Review boundary**

Review only zero-process preflight, exact-layout completion verification, and stale-run prevention. Reject if preflight launches any non-git process, probes GPUs, imports private historical modules, deletes partial diagnostics, or silently resumes a partial round; the existing read-only Git safety subprocesses are the only exception. Also reject if completion verification guesses filenames, scans directories, bypasses existing feedback validators, or emits private identities/paths.

- [ ] **Step 9: Commit**

```bash
git add \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py \
  framework/stage6/coptv2x_h800_search_v2.py \
  framework/stage6/p6_history_measurement_v1.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  tests/stage6/test_coptv2x_h800_search.py \
  tests/release/test_run_p6_h800_search.py
git commit -m "feat: preflight P6 materializer training bridge"
```

---

### Task 6: Zero-GPU Full Dynamic Lifecycle Gate

**Files:**

- Modify: `tests/release/test_run_p6_h800_search.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/stage6/test_p6_history_measurement.py`
- Modify: `tests/stage6/test_p6_history_source_materialization.py`

**Interfaces:**

- Consumes: Stage1 fake scan, real Stage2 loader/build adapter, recipe-v2 registry builder, projection, source invocation builder, injected measurement runner.
- Produces: no production API; this task is an end-to-end test gate proving the lifecycle without GPU/training/measurement process execution.

- [ ] **Step 1: Write RED zero-GPU black-box lifecycle test**

Add to `tests/release/test_run_p6_h800_search.py`:

```python
def test_cli_zero_gpu_dynamic_lifecycle_trains_markers_without_gold_remeasurement(
    tmp_path: Path,
) -> None:
    paths = _prepare_framework_recipe_v2_training_run(tmp_path)
    process_log = paths["process_log"]

    result = run_cli(
        [
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local_config"]),
            "--code-revision",
            "rev-zero-gpu-training-bridge",
        ]
    )

    assert result.returncode == 0
    state = json.loads((paths["output_root"] / "state.json").read_text(encoding="utf-8"))
    assert state["completed_rounds"] == 4
    assert state["measured_candidate_count"] == 16
    records = [json.loads(line) for line in process_log.read_text(encoding="utf-8").splitlines()]
    assert [record["stage"] for record in records if record["stage"] == "stage1"] == ["stage1"]
    assert [record["fit_count"] for record in records if record["stage"] == "initial_fit"] == [176]
    assert [record["fit_count"] for record in records if record["stage"] == "online_fit"] == [180, 184, 188]
    measured_row_ids = {
        str(row.get("row_id") or row.get("manifest_job_id"))
        for round_index in range(4)
        for row in json.loads((paths["output_root"] / f"round-{round_index:02d}" / "measurement_request.json").read_text(encoding="utf-8"))["rows"]
    }
    gold_row_ids = {
        str(row.get("manifest_job_id") or row.get("row_id"))
        for row in json.loads(paths["gold176_rows"].read_text(encoding="utf-8"))
    }
    assert measured_row_ids.isdisjoint(gold_row_ids)
    assert _all_source_marker_pairs_exist(paths["private_root"])
    assert _no_gpu_or_training_process_was_launched(records)
```

The fake measurement adapter must:

- read the projected request,
- create `training_done_marker` and `source_done_marker` per selected group,
- record source invocation shape only,
- synthesize feedback only after marker creation,
- never call GPU, TVM, AP, latency, energy, or historical code.

- [ ] **Step 2: Add exact dynamic counts and no static fallback assertions**

In the same test, assert:

```python
registry = json.loads((paths["output_root"] / "source_registry.json").read_text(encoding="utf-8"))
plan = json.loads((paths["output_root"] / "pyramid_candidate_plan.json").read_text(encoding="utf-8"))
expected_candidate_count = _expected_candidate_count_from_stage2_fixture(paths["stage2_fixture"])
assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
assert len(registry["groups"]) == plan["structure_count"]
assert sum(len(group["available_q_modes"]) for group in registry["groups"]) == plan["candidate_count"]
assert plan["candidate_count"] == expected_candidate_count
assert expected_candidate_count >= 16
assert plan["candidate_count"] not in {343, 686}
assert paths["candidate_source_mode"] == "framework_stage2_search_space"
```

The synthetic Stage2 fixture may deterministically yield 126 candidates, but 126 is not a product invariant. Compute the expected count from the fixture/adapter and assert the plan and registry match that count. Separately assert framework mode has no 343/686/static fallback. The real H800 run accepts whatever fresh Stage2 outputs if it is at least the 16-row sample budget and passes plan/registry identity validation.

- [ ] **Step 3: Add source invocation and marker-order assertions**

Assert four selected rows per round and one source invocation per distinct group:

```python
for round_index in range(4):
    request = json.loads((paths["output_root"] / f"round-{round_index:02d}" / "measurement_request.json").read_text(encoding="utf-8"))
    assert request["round_index"] == round_index
    assert len(request["rows"]) == 4
    assert all(row["source_contract"]["training_required"] is True for row in request["rows"])
    group_ids = []
    for row in request["rows"]:
        group_ids.append(row["group_id"])
        contract = row["source_contract"]
        training_marker = Path(contract["training_done_marker"])
        source_marker = Path(contract["source_done_marker"])
        assert training_marker.is_file()
        assert source_marker.is_file()
        assert training_marker.stat().st_mtime_ns <= source_marker.stat().st_mtime_ns
    expected_source_calls = sorted(set(group_ids))
    actual_source_calls = _source_calls_for_round(process_log, round_index)
    assert [call["group_id"] for call in actual_source_calls] == expected_source_calls
```

- [ ] **Step 4: Run RED test**

Run:

```bash
PYTHONPATH=. pytest tests/release/test_run_p6_h800_search.py::test_cli_zero_gpu_dynamic_lifecycle_trains_markers_without_gold_remeasurement -q
```

Expected: FAIL until fake dynamic lifecycle fixture and marker-writing measurement adapter are wired to the stricter training contract.

- [ ] **Step 5: Implement only test fixtures and adapter hooks required for zero-GPU**

Keep production behavior unchanged unless a production seam from Tasks 1–5 is missing. The fake measurement adapter must never mark success before it has written both markers for each group:

```python
for row in request["rows"]:
    contract = row["source_contract"]
    Path(contract["training_done_marker"]).parent.mkdir(parents=True, exist_ok=True)
    Path(contract["training_done_marker"]).write_text("trained\n", encoding="utf-8")
    Path(contract["source_done_marker"]).parent.mkdir(parents=True, exist_ok=True)
    Path(contract["source_done_marker"]).write_text("source-ready\n", encoding="utf-8")
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_run_p6_h800_search.py \
  tests/stage6/test_coptv2x_h800_search.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Full coverage and static gates**

Run:

```bash
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=80
PYTHONPATH=. pytest tests/release tests/integration/test_anonymous_archive.py -q
python -m ruff check framework tools scripts tests
git diff --check
```

Expected: all commands exit 0, coverage is at least 80%, and anonymous archive tests find no private values.

- [ ] **Step 8: Review boundary**

Review the full zero-GPU lifecycle only. Reject if Gold176 appears in any measurement request, if static P6.1 registry fallback can satisfy framework mode, if synthetic metrics are accepted without source markers, or if process logs include private paths.

- [ ] **Step 9: Commit**

```bash
git add \
  tests/release/test_run_p6_h800_search.py \
  tests/stage6/test_coptv2x_h800_search.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py
git commit -m "test: gate P6 dynamic training lifecycle offline"
```

---

### Task 7: Private H800 Preflight, Fresh Four-Round Run, and Conditional Final Docs

**Files:**

- Conditionally modify after real success: `docs/AAAI27_RELEASE_AUDIT.md`
- Conditionally create after real success: `docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md`
- Do not modify tracked files if private preflight or real run fails.

**Interfaces:**

- Consumes: completed Tasks 1–6, already-cleared ignored private runner template/source-contract/source-map, regenerated private binding/config, H800 private environment, and existing release CLIs.
- Produces: private ignored run artifacts and public-safe final docs only if all acceptance gates succeed.

- [ ] **Step 1: Confirm clean tracked state and full local gates**

Run:

```bash
git status --short
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=80
python -m ruff check framework tools scripts tests
git diff --check
```

Expected: `git status --short` is empty before private execution; all verification commands pass.

- [ ] **Step 2: Regenerate private binding/config; do not patch old binding**

Use the existing ignored private source-map, runner template, source wrapper profile, source-contract template, and local output root. Before provisioning, build a fresh private component staging root: copy the verified minimal historical implementation repository/module tree beneath the profile's non-marker implementation subroot; preserve the required validator/import siblings; omit the raw `stage5_materialize_round_sources_v1.sh` marker from that relocated tree; copy the separately validated downstream components; then let the renderer create the one unique source marker referenced by the runner template. Do not copy only the source shell entrypoint, infer its repository root, or overwrite a differing marker.

Private paths are supplied by the operator through already-set environment variables. The plan intentionally does not assign illustrative absolute paths:

```bash
: "${P6_PUBLIC_CONTRACT_JSON:?set to the existing public contract path}"
: "${P6_LEGACY_LOCAL_CONFIG:?set to the ignored private locator/config path}"
: "${P6_RUNNER_TEMPLATE_YAML:?set to the ignored staged runner template path}"
: "${P6_SOURCE_WRAPPER_PROFILE_YAML:?set to the ignored wrapper profile path}"
: "${P6_LOCAL_OUTPUT_ROOT:?set to a fresh ignored preflight output root}"
: "${P6_CODE_REVISION_LABEL:?set to a public-safe non-digest revision label}"
```

Run:

```bash
: "${P6_LEGACY_LOCAL_CONFIG:?}"
: "${P6_RUNNER_TEMPLATE_YAML:?}"
: "${P6_SOURCE_WRAPPER_PROFILE_YAML:?}"
: "${P6_LOCAL_OUTPUT_ROOT:?}"
mkdir -p "${P6_LOCAL_OUTPUT_ROOT}"
PYTHONPATH=. python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "${P6_LEGACY_LOCAL_CONFIG}" \
  --runner-template "${P6_RUNNER_TEMPLATE_YAML}" \
  --local-output-root "${P6_LOCAL_OUTPUT_ROOT}" \
  --binding-output "${P6_LOCAL_OUTPUT_ROOT}/binding.json" \
  --config-output "${P6_LOCAL_OUTPUT_ROOT}/local-config.json" \
  --source-wrapper-profile "${P6_SOURCE_WRAPPER_PROFILE_YAML}"
```

Expected:

- old V2 binding lacking `training_required: true` fails preflight/provisioning;
- regenerated binding accepts only if the wrapper is self-contained and the source contract template has all static Pyramid training fields;
- stdout/stderr contains no private paths in tracked logs.

- [ ] **Step 3: Run zero-process private preflight**

Run:

```bash
: "${P6_PUBLIC_CONTRACT_JSON:?}"
: "${P6_LOCAL_OUTPUT_ROOT:?}"
: "${P6_RUNNER_TEMPLATE_YAML:?}"
: "${P6_SOURCE_WRAPPER_PROFILE_YAML:?}"
PYTHONPATH=. python tools/release/preflight_p6_materializer_training_bridge.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_LOCAL_OUTPUT_ROOT}/local-config.json" \
  --binding "${P6_LOCAL_OUTPUT_ROOT}/binding.json" \
  --runner-template "${P6_RUNNER_TEMPLATE_YAML}" \
  --source-wrapper-profile "${P6_SOURCE_WRAPPER_PROFILE_YAML}"
```

Expected public-safe JSON:

```json
{
  "gpu_probe_count": 0,
  "historical_process_launch_count": 0,
  "schema_version": "p6_materializer_training_bridge_preflight_v1",
  "status": "accepted",
  "training_required": true,
  "validated_round_count": 4,
  "wrapper_marker": "stage5_materialize_round_sources_v1.sh"
}
```

If this fails, stop this task, preserve private diagnostics under the ignored output root, do not update docs, and do not relaunch in-place.

- [ ] **Step 4: Start a fresh private output root for the real run**

Use a new ignored private local output root, or explicitly clean only a scoped ignored root after reviewing the exact directory. Do not remove a repository root, home directory, broad workspace, or unscoped path. Do not reuse a partial round directory.

The operator creates and exports a new ignored `P6_FRESH_LOCAL_OUTPUT_ROOT` outside tracked/public artifacts. Validate it instead of assigning a fake example path, then provision it:

```bash
: "${P6_FRESH_LOCAL_OUTPUT_ROOT:?set to a new ignored real-run output root}"
mkdir -p "${P6_FRESH_LOCAL_OUTPUT_ROOT}"
PYTHONPATH=. python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "${P6_LEGACY_LOCAL_CONFIG:?}" \
  --runner-template "${P6_RUNNER_TEMPLATE_YAML:?}" \
  --local-output-root "${P6_FRESH_LOCAL_OUTPUT_ROOT}" \
  --binding-output "${P6_FRESH_LOCAL_OUTPUT_ROOT}/binding.json" \
  --config-output "${P6_FRESH_LOCAL_OUTPUT_ROOT}/local-config.json" \
  --source-wrapper-profile "${P6_SOURCE_WRAPPER_PROFILE_YAML:?}"
```

Fresh-run criteria before launch are checked in two namespaces:

- Stage1 manifest path is absent.
- `source_registry.json` destination is absent and is not a symlink.
- Public search `round-00` through `round-03` directories are absent; these contain only the search controller's request/feedback/failure leaves.
- For private history rounds 0 through 3, the exact round root, measurement-request, task-state, actual-feedback, receipt, and finalization-barrier paths resolved from the validated binding templates are absent. No filename guessing or directory scan is allowed.
- Any marker destinations statically renderable from the fresh registry/plan are absent. Because selected requests do not exist before Stage1/selection, each selected projected marker pair is also checked for absence immediately before its source invocation.
- `state.json` is absent.

- [ ] **Step 5: Execute the fresh H800 run**

Run:

```bash
: "${P6_PUBLIC_CONTRACT_JSON:?}"
: "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}"
: "${P6_CODE_REVISION_LABEL:?}"
PYTHONPATH=. python tools/release/run_p6_h800_search.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_FRESH_LOCAL_OUTPUT_ROOT}/local-config.json" \
  --code-revision "${P6_CODE_REVISION_LABEL}"
```

Expected:

- Stage1 scan runs and writes a fresh manifest.
- Stage2 dynamic candidate plan is built from Stage1/Stage2 surfaces.
- Observed candidate pool is whatever fresh Stage2 emits for Pyramid/H800/TVM, provided it has at least 16 eligible rows and matches the registry identity. Do not require 126, 343, or 686 in the real run.
- Round 0 fits Gold176 cold-start rows only; Gold176 rows are not sent to measurement.
- Four rows are selected in each of four rounds.
- Source materializer no longer fails with the deterministic private import error.
- For each selected group, private evidence shows training/fine-tuning marker before source marker and before downstream TVM/AP/latency/energy stages.
- Online refits occur after accepted feedback for rounds 0, 1, and 2 with counts 180, 184, and 188.
- No failed or fabricated metrics are treated as successful measurements.

- [ ] **Step 6: Validate private completion without leaking details**

Run the reviewed verifier against the exact binding and output-layout contract. It may inspect ignored private artifacts but prints only a public-safe summary:

```bash
PYTHONPATH=. python tools/release/verify_p6_materializer_training_run.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON:?}" \
  --local-config "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}/local-config.json" \
  --binding "${P6_FRESH_LOCAL_OUTPUT_ROOT}/binding.json"
```

Expected: the verifier resolves each task-state/result/receipt/barrier from the validated binding templates, never by directory scan; validates completed state, four rounds of four rows, request/row/source-contract/source-evidence hashes, translated feedback identity, exactly five finite metrics, positive latency/energy, bounded AP values, training-before-source marker order, 16 unique rows, and zero overlap with frozen Gold176 measurement identities. It prints only sorted public-safe JSON equivalent to `status=completed`, `completed_rounds=4`, `selected_rows=16`, and `gold176_remeasured_rows=0`.

- [ ] **Step 7: If real run fails, stop without docs**

If any selected candidate cannot train/fine-tune, import failure persists, markers are missing, feedback fails validation, Gold176 is remeasured, or downstream metrics are absent/invalid:

- preserve ignored private diagnostics,
- do not replace failed feedback with synthetic success,
- do not skip candidates,
- do not switch to static registry,
- do not update final docs,
- start any retry from a new ignored private output root after fixing the cause.

- [ ] **Step 8: Conditionally update public docs after success**

Only after Step 6 succeeds, update docs with stable public-safe facts:

```markdown
## P6 V2 materializer training bridge

- Private preflight accepted the regenerated recipe-v2 binding with `training_required: true`.
- The self-contained source materializer wrapper ran from private round cwd with the strict five-key environment.
- Stage1/Stage2 dynamic Pyramid/H800/TVM candidate generation fed four P6 rounds.
- Gold176 was used only as cold-start cost-model evidence and was not remeasured.
- Four rounds completed with 16 unique selected measurement rows.
- Each selected group produced training and source markers before downstream TVM/AP/latency/energy stages.
```

Do not include private paths, dataset names beyond public labels, checkpoint paths, GPU UUIDs, hostnames, raw logs, candidate IDs, metric values, stdout/stderr, or static hyperparameter values.

- [ ] **Step 9: Run final docs/privacy gates**

Run:

```bash
python -m ruff check framework tools scripts tests
git diff --check
PYTHONPATH=. pytest tests/integration/test_anonymous_archive.py tests/release -q
python - <<'PY'
from pathlib import Path
blocked = ("/home/", "/mnt/", "GPU-", "CUDA_VISIBLE_DEVICES=", "base_checkpoint_path", "dataset_root", "pyramid_config_path", "Traceback")
for path in [Path("docs/AAAI27_RELEASE_AUDIT.md"), Path("docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md")]:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        for token in blocked:
            assert token not in text, f"{token} leaked in {path}"
PY
```

Expected: all commands exit 0.

- [ ] **Step 10: Review boundary**

Review only public docs and the public-safe completion evidence. Reject if docs are updated before real success, if docs include private/static values, or if run closure is claimed with fewer than four completed rounds and 16 selected rows.

- [ ] **Step 11: Commit**

Only if docs were updated after real success:

```bash
git add docs/AAAI27_RELEASE_AUDIT.md docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md
git commit -m "docs: record P6 materializer training bridge completion"
```

If docs were not updated because the private run failed or was not executed, do not commit this task.

---

## Spec Requirement Mapping

| Spec requirement | Task coverage |
| --- | --- |
| Fix deterministic private import failure without widening public env | Task 2 wrapper profile and black-box import determinism; Task 4 exact five-key env |
| Enforce `training_required: true` and static Pyramid training fields | Task 1 binding/registry validation; Task 3 projection preservation |
| Preserve static fields through projection and hashes | Task 3 projection tests and hash recomputation |
| Reject incomplete recipe-v2 source contracts | Task 1 and Task 5 preflight |
| Render one shared source bundle per Pyramid group | Task 1 registry integration; Task 3 projection drift tests; Task 6 lifecycle assertions |
| Remove untrusted output aliases before canonical group contract hash | Task 1 registry integration; Task 3 projection tests |
| Public/private boundary and no private leakage | Tasks 1, 2, 4, 5, 7 privacy gates |
| Source materializer direct argv per group | Tasks 2, 3, 4, 6 |
| Gate downstream stages on training/source markers | Task 4 and Task 6 |
| Zero-process preflight before activation/GPU/historical execution | Task 5 |
| Zero-GPU black-box dynamic lifecycle | Task 6 |
| Fresh Stage1/Stage2 through four rounds | Task 6 fake lifecycle; Task 7 real H800 run |
| No Gold176 remeasurement | Task 6 and Task 7 |
| No in-place partial resume | Task 5 and Task 7 |
| Preserve v1/static compatibility | Tasks 1, 3, 5 focused regression commands |
| Final docs conditional on real completion | Task 7 only |

---

## Self-Review

- [ ] Spec coverage: every goal, invariant, fail-closed category, public/private rule, TDD matrix row, provisioning rule, preflight rule, relaunch rule, and closure criterion maps to at least one task above.
- [ ] Placeholder scan: the plan contains no unresolved placeholder instructions or unbounded “write broad tests later” steps. Task 7 uses concrete operator-supplied private environment variables (`P6_PUBLIC_CONTRACT_JSON`, `P6_LEGACY_LOCAL_CONFIG`, `P6_RUNNER_TEMPLATE_YAML`, `P6_SOURCE_WRAPPER_PROFILE_YAML`, `P6_LOCAL_OUTPUT_ROOT`, `P6_FRESH_LOCAL_OUTPUT_ROOT`, `P6_CODE_REVISION_LABEL`) with shell `${VAR:?}` gates and Python `os.environ`; these are intentional runtime inputs, not unresolved path placeholders.
- [ ] Red-flag scan: run the literal-token scan requested by the reviewer against this plan and keep it clean before handoff.
- [ ] Type consistency: function names and constants are stable across tasks: `validate_recipe_v2_training_template()`, `validate_recipe_v2_group_training_contract()`, `validate_projected_training_contract()`, `validate_projected_training_marker_pairs()`, `validate_self_contained_source_wrapper(..., source_wrapper_profile=...)`, `materialize_full_chain_binding(..., *, source_wrapper_profile: Path | None = None) -> dict[str, Any]`, `plan_validated_history_round_paths()`, `resolve_validated_history_round_paths()`, `preflight_materializer_training_bridge(..., source_wrapper_profile_path=...)`, `verify_materializer_training_run()`, `historical_process_launch_count`, and binding-owned `EXPECTED_HISTORY_ENV_KEYS`.
- [ ] File-size check: new validators are split into focused modules; existing over-limit files receive routing glue only.
- [ ] Security/privacy check: public projection, docs, stderr, tests, and release manifests must not contain private paths, GPU UUIDs, raw logs, checkpoints, datasets, hostnames, candidate IDs, or static training values.
- [ ] Completion check: Task 7 docs are explicitly conditional on private preflight plus successful fresh four-round H800 completion.
