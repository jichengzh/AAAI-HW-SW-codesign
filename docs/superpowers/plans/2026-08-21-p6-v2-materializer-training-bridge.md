# P6 V2 Materializer Training Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make P6 recipe-v2 train each canonical Pyramid source group on first use in one fresh run, reuse only adapter-receipted current-run source evidence for later q-modes, execute q-specific downstream work for all 16 selected rows, and complete without Gold176 remeasurement.

**Architecture:** Keep the public adapter boundary strict: direct argv, `shell=False`, private round cwd, and exactly the five-key private environment. A focused private evidence module creates one immutable fresh-run context, classifies each selected group as first-use or validated current-run reuse, hashes the exact q-independent 11-output bundle, and exclusively publishes adapter-owned per-group receipts. The controller creates the context exactly once after dynamic plan/registry validation; measurement only loads it, invokes unseen groups, revalidates all selected groups, and then runs every q-specific downstream stage.

**Tech Stack:** Python 3.10+, pytest, Ruff, JSON/YAML canonical serialization, existing Stage1/Stage2/Stage5/Stage6 P6 modules, injected runner tests, optional private H800 execution only in final gated task.

**Spec:** `docs/superpowers/specs/2026-08-21-p6-v2-materializer-training-bridge-design.md` (approved authority at `258e37d`)

## Global Constraints

- Keep `p6_h800_coptv2x_search_contract_v2`, `p6_h800_coptv2x_local_v2`, `stage5_measurement_request_v2`, `p6_h800_coptv2x_feedback_v2`, and `p6_history_dynamic_materialization_recipe_v2`.
- Target stays exactly Pyramid + H800 + TVM, four rounds, batch size four, sample budget 16, metrics `latency_ms`, `energy_j`, `ap30`, `ap50`, `ap70`.
- Stage1 scans real Pyramid structure for H800, then Stage2 produces the dynamic Pyramid/H800/TVM candidate plan; no fallback to static 343/686 P6.1 registry in framework mode.
- Gold176 is frozen cold-start evidence only; it is never emitted as a measurement request row.
- Round 0 cost-model fit count must be 176; online refit counts after accepted feedback must be 180, 184, and 188.
- Each round selects exactly four unmeasured rows; the full successful run selects 16 unique rows.
- The 16 identities are row-level `(group_id, q_mode)` genomes. Canonical group ids may repeat under another q-mode in the same or a later round; global group uniqueness is explicitly rejected.
- Public adapter must not add `PYTHONPATH`, inherit ambient environment, construct shell command strings, use `shell=True`, run from the history root, leak raw stderr/stdout, or publish private paths, GPU UUIDs, dataset roots, checkpoint paths, host details, static training values, or raw logs. Direct `shell=False` execution of the validated private `/bin/sh` marker wrapper is the approved boundary.
- Source materialization must be one direct argv call on a canonical group's first use in the fresh run: `<source_materializer> --request <projected-request> --model pyramid --group-id <group_id> --gpu <validated-index>`. A current-run ready group receives zero later source calls; per-row and per-q-mode retraining are explicitly rejected.
- Recipe-v2 source contracts must enforce `training_required: true`, `training_source_kind`, `base_checkpoint_path`, `dataset_root`, `pyramid_config_path`, `training_parameters`, `stage_widths`, and exactly the 11 existing `shared_source_paths` keys.
- Required `training_parameters` semantic keys are `training_mode`, `epochs`, `seed`, `optimizer`, `learning_rate`, `batch_size`, `dataset_split`, `checkpoint_selection`, and `freeze_policy`.
- Static training input paths must be absolute, symlink-safe, existing where required, and under the validated private history root; rendered output paths must be absolute, symlink-safe, unique, and under the local private output root.
- Fail closed before process launch for incomplete private wrapper, binding, recipe-v2 contract, output layout, source contract, static training contract, hash drift, stale state, unsafe symlink, or path collision.
- Bare markers never authorize reuse. Only an adapter-owned `p6_group_source_reuse_receipt_v1` bound to the immutable `p6_materializer_fresh_run_context_v1`, its producer request/row, and recomputed artifact/marker digests may authorize skipping a source call.
- Run context and receipt locations are exact under `.p6-materializer-training-bridge-v1`; no `glob`, `rglob`, basename search, marker-parent inference, or filename guessing may locate them. Bounded traversal is allowed only inside an already-declared directory artifact to compute its digest.
- Contexts and receipts are create-only, atomically published with no-replace semantics, mode `0600` beneath mode-`0700` parents, and never updated, repaired, overwritten, or accepted from a wrapper.
- All root, metadata, receipt, and declared artifact paths must pass canonical lexical spelling, component-wise `lstat`, resolved containment, exact type, symlink rejection, path-escape rejection, and single-link regular-file checks. The evidence path object stores the exact `root.resolve(strict=True)` reached after those checks, and both context create/load hash `str()` of that same resolved object; raw or marker-derived roots are forbidden. Directory digests reject symlinks, hard-linked regular files, sockets, devices, and FIFOs.
- The controller creates one fresh 32-byte nonce and run context only after validating the task, public-safe revision, local-root fingerprint, complete Stage2 plan, and registry identities, and before any measurement/GPU/activation boundary. No nonce/context/receipt/local-root environment key is added.
- Measurement requires an existing valid context and never creates one. `UNSEEN` means receipt plus all 11 leaves are absent; `READY_CURRENT_RUN` means the receipt and all bound bytes validate; partial, stale, or mismatched states stop before downstream execution.
- All four selected rows in every round continue through quantization, performance/TVM, AP, and finalization, including same-group mixed q-mode rows. Reuse skips only source materialization/training.
- Completion requires four exact requests, 16 unique row ids, 16 row-to-receipt mappings, all terminal success evidence and five finite metrics, and zero Gold176 overlap. Multiple rows may share one receipt, and a producer may be from the same or an earlier round.
- Stable private reuse categories are `p6_source_reuse_partial`, `p6_source_reuse_stale`, and `p6_source_reuse_mismatch`; all map to public `history_execution_invalid` without values.
- Existing v1/static compatibility must be preserved; stricter training enforcement applies only to recipe-v2 P6 real-run contracts.
- Files touched by implementation should stay under 800 lines. Existing files already over the limit must receive only routing glue plus a focused split module.
- Every task is RED→GREEN→REFACTOR, includes a review boundary, and uses a conventional commit message. Do not commit from the planning step.
- A failed or partial run is preserved for ignored private diagnosis and never resumed in place. Any retry uses a new scoped ignored local output root.
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
- Create `framework/stage6/p6_source_reuse_evidence_v1.py`: immutable run-context/receipt schemas, canonical hashing and strict JSON loading, deterministic paths, filesystem safety, source-bundle digests, group classification, first-use filtering, producer validation, and exclusive adapter receipt publication.
- Modify `framework/stage6/p6_history_measurement_v1.py`: in Task 4 add private routing glue that returns the validated resolved public-round parent as `local_output_root`; in Task 5 expose the same planned/runtime exact round-path contract without signature or return-key drift; enforce the exact five-key environment, require the existing fresh context, invoke only unseen groups, publish receipts from the public adapter, revalidate all selected groups, then run every downstream stage.
- Modify `framework/stage6/p6_history_source_materialization_v1.py`: preserve Task 3 training/hash projection and canonical marker/invocation extraction, route reusable runtime path safety into the focused evidence module, and remove the contradictory blanket per-invocation marker-absence rule.
- Modify `framework/stage6/coptv2x_h800_search_v2.py`: validate the public-safe revision, perform exact fresh-run/relaunch guards, and create the one context after plan/registry validation and before the round loop.
- Create focused Task 4 tests: `tests/stage6/test_p6_source_reuse_evidence_paths.py`, `tests/stage6/test_p6_source_reuse_evidence_receipts.py`, and `tests/stage6/test_p6_source_reuse_measurement.py`.
- Create Task 5 tests: `tests/stage6/test_p6_history_round_paths.py`, `tests/stage6/test_p6_fresh_run_controller.py`, `tests/release/test_preflight_p6_materializer_training_bridge.py`, and `tests/release/test_verify_p6_materializer_training_run.py`.
- Create Task 6 offline lifecycle fixture/gate: `tests/release/p6_source_reuse_lifecycle_fixture.py` and `tests/release/test_p6_source_reuse_lifecycle.py`.
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

### Task 4: Immutable Current-Run Source-Reuse Evidence and Measurement Gate

**Files:**

- Create: `framework/stage6/p6_source_reuse_evidence_v1.py`
- Modify: `framework/stage6/p6_history_measurement_v1.py` (routing glue only, including the private resolver's validated resolved `local_output_root` result)
- Modify: `framework/stage6/p6_history_source_materialization_v1.py` (projection/invocation routing only)
- Create: `tests/stage6/test_p6_source_reuse_evidence_paths.py`
- Create: `tests/stage6/test_p6_source_reuse_evidence_receipts.py`
- Create: `tests/stage6/test_p6_source_reuse_measurement.py`
- Modify: `tests/stage6/test_p6_history_measurement.py` (migrate/retain the exact-env regression only)
- Modify: `tests/stage6/test_p6_history_source_materialization.py` (move paused blanket marker-gate coverage to the focused files)
- Modify: `tests/release/test_p6_history_execution_adapters.py` (fixture routing only)

**Paused-diff migration rule:** The five unstaged Task 4 files are existing user work, not disposable scaffolding. Preserve their exact five-key environment assertion, local-root containment checks, source failure redaction, regular-file marker checks, first-use marker-order test logic, and compatible private-resolver `local_output_root` routing. Move focused new cases out of the already-large mixed test modules. Replace `assert_source_markers_absent_before_invocation()` as a blanket per-measurement rule with `UNSEEN`/`READY_CURRENT_RUN` classification: absence and mtime ordering apply only to first-use groups; later reuse is authorized by current-run receipt and recomputed digests. Do not discard, stash, reset, or overwrite the paused work wholesale.

**Interfaces:**

- Consumes: Task 3's detached canonical projected request, binding-owned `EXPECTED_HISTORY_ENV_KEYS`, `SHARED_SOURCE_PATH_KEYS`, the exact binding round templates, and existing `build_source_invocations()`/`run_source_invocations()`.
- Produces these exact immutable types and APIs from `p6_source_reuse_evidence_v1.py`:

```python
P6SourceReuseState = Literal[
    "UNSEEN",
    "READY_CURRENT_RUN",
    "INVALID_PARTIAL",
    "INVALID_STALE",
    "INVALID_MISMATCH",
]
ArtifactKind = Literal["regular_file", "directory_tree"]

class P6SourceReuseEvidenceError(ValueError):
    private_category: Literal[
        "p6_source_reuse_partial",
        "p6_source_reuse_stale",
        "p6_source_reuse_mismatch",
    ] | None
    public_category: Literal["history_execution_invalid", "unsafe_destination"]
    def __init__(
        self,
        *,
        public_category: Literal[
            "history_execution_invalid", "unsafe_destination"
        ],
        private_category: Literal[
            "p6_source_reuse_partial",
            "p6_source_reuse_stale",
            "p6_source_reuse_mismatch",
        ] | None = None,
    ) -> None: ...

RUN_METADATA_RELATIVE_ROOT = Path(".p6-materializer-training-bridge-v1")
RUN_CONTEXT_RELATIVE_PATH = RUN_METADATA_RELATIVE_ROOT / "run-context.json"
GROUP_RECEIPT_RELATIVE_ROOT = RUN_METADATA_RELATIVE_ROOT / "group-receipts"
SOURCE_OUTPUT_RELATIVE_ROOT = Path("materialized")

SOURCE_ARTIFACT_KEYS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_dir",
    "config_path",
    "onnx_path",
    "onnx_report_path",
    "calibration_root",
    "calibration_npz",
    "calibration_summary",
    "trt_calibration_dir",
)
SOURCE_MARKER_KEYS: tuple[str, str] = (
    "training_done_marker",
    "source_done_marker",
)

@dataclass(frozen=True)
class P6SourceReusePaths:
    local_output_root: Path
    metadata_root: Path
    run_context: Path
    receipt_root: Path

@dataclass(frozen=True)
class P6FreshRunContext:
    schema_version: Literal["p6_materializer_fresh_run_context_v1"]
    run_nonce: str
    task_id: str
    task_sha256: str
    code_revision: str
    local_output_root_sha256: str
    candidate_plan_schema_version: Literal["p6_pyramid_candidate_plan_v2"]
    candidate_plan_sha256: str
    source_registry_schema_version: Literal[
        "stage5_candidate_source_registry_v2"
    ]
    source_registry_sha256: str
    created_before_round_index: Literal[0]
    run_context_sha256: str

@dataclass(frozen=True)
class P6ArtifactDigest:
    kind: ArtifactKind
    sha256: str

@dataclass(frozen=True)
class P6GroupSourceReceipt:
    schema_version: Literal["p6_group_source_reuse_receipt_v1"]
    status: Literal["READY_CURRENT_RUN"]
    run_context_sha256: str
    run_nonce: str
    task_id: str
    task_sha256: str
    group_id: str
    group_key_sha256: str
    source_contract_sha256: str
    source_evidence_sha256: str
    producer_round_index: int
    producer_measurement_request_sha256: str
    producer_row_id: str
    producer_row_sha256: str
    producer_q_mode: Literal["fp16", "int8"]
    artifact_digests: tuple[tuple[str, P6ArtifactDigest], ...]
    marker_digests: tuple[tuple[str, str], ...]
    receipt_sha256: str

@dataclass(frozen=True)
class P6GroupReuseDecision:
    group_id: str
    state: P6SourceReuseState
    receipt_path: Path

def canonical_json_sha256(value: object) -> str: ...
def plan_source_reuse_paths(local_output_root: Path) -> P6SourceReusePaths: ...
def resolve_existing_source_reuse_paths(
    local_output_root: Path,
) -> P6SourceReusePaths: ...
def receipt_path_for_group(paths: P6SourceReusePaths, group_id: str) -> Path: ...
def create_fresh_run_context(
    *,
    local_output_root: Path,
    task_contract: Mapping[str, Any],
    code_revision: str,
    candidate_plan: Mapping[str, Any],
    source_registry: Mapping[str, Any],
) -> P6FreshRunContext: ...
def load_fresh_run_context(
    *,
    local_output_root: Path,
    expected_task_id: str,
    expected_task_sha256: str,
) -> P6FreshRunContext: ...
def requires_current_run_source_evidence(
    request: Mapping[str, Any],
) -> bool: ...
def classify_selected_group_sources(
    request: Mapping[str, Any],
    *,
    run_context: P6FreshRunContext,
    local_output_root: Path,
    interface: Mapping[str, Any],
    private_root: Path,
) -> tuple[P6GroupReuseDecision, ...]: ...
def first_use_group_ids(
    decisions: Sequence[P6GroupReuseDecision],
) -> tuple[str, ...]: ...
def validate_and_publish_group_receipt(
    request: Mapping[str, Any],
    *,
    group_id: str,
    run_context: P6FreshRunContext,
    local_output_root: Path,
    interface: Mapping[str, Any],
    private_root: Path,
) -> P6GroupSourceReceipt: ...
def require_selected_groups_ready_current_run(
    request: Mapping[str, Any],
    *,
    run_context: P6FreshRunContext,
    local_output_root: Path,
    interface: Mapping[str, Any],
    private_root: Path,
) -> tuple[P6GroupSourceReceipt, ...]: ...
```

- Also owns this Task 4-private routing contract in `p6_history_measurement_v1.py`; Task 5 exports the same semantics without changing the positional arguments or mapping keys:

```python
def _resolve_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    supplied_public_round_root: Path,
    round_index: int,
) -> dict[str, Path]: ...
```

The returned mapping includes the existing `history_root` and binding-resolved private leaves plus `local_output_root`. That root is exactly the validated resolved parent of the already validated supplied public round directory. It is never inferred from a marker, shared-output leaf, private round path, or receipt path.

`artifact_digests` serializes as an exact-key object with the nine `SOURCE_ARTIFACT_KEYS`; expected kinds are `regular_file` for checkpoint/config/ONNX/report/NPZ/summary and `directory_tree` for checkpoint/calibration/TRT directories. A declared file may also appear inside a declared directory-tree digest; this intentional duplicate coverage proves both leaf and tree identities. `marker_digests` serializes as an exact-key object with the two `SOURCE_MARKER_KEYS`. Tuple order in memory is the constant order above; JSON hashes use sorted keys.

- [ ] **Step 1: Write RED canonical hashing, layout, and context tests**

Create `tests/stage6/test_p6_source_reuse_evidence_paths.py` with this canonical-hash anchor and exact path assertions:

```python
def test_canonical_json_hash_and_group_receipt_path_are_exact(tmp_path: Path) -> None:
    root = _safe_local_root(tmp_path)
    resolved_root = root.resolve(strict=True)
    paths = plan_source_reuse_paths(root)
    group_id = "pyramid|16x32x64"
    expected_key = hashlib.sha256(
        b"p6-group-receipt-v1\0" + group_id.encode("utf-8")
    ).hexdigest()

    assert canonical_json_sha256({"b": 2, "a": 1}) == hashlib.sha256(
        b'{"a":1,"b":2}'
    ).hexdigest()
    assert paths.local_output_root == resolved_root
    assert paths.run_context == resolved_root / RUN_CONTEXT_RELATIVE_PATH
    assert receipt_path_for_group(paths, group_id) == (
        resolved_root / GROUP_RECEIPT_RELATIVE_ROOT / f"{expected_key}.json"
    )
```

Add `test_create_context_binds_exact_task_revision_root_plan_and_registry()` using `_validated_task_contract()`, `_candidate_plan()`, and `_source_registry()` fixtures. It must assert a 64-lowercase-hex nonce, `created_before_round_index == 0`, exact plan/registry hashes, root fingerprint
`sha256(b"p6-local-output-root-v1\0" + str(root.resolve(strict=True)).encode()).hexdigest()`, canonical context hash, mode `0600` context, and mode `0700` metadata/receipt directories. Assert both create and load obtain the same `P6SourceReusePaths.local_output_root == root.resolve(strict=True)` and use that exact object for plan/registry leaves and root hashing; no raw caller-path string may enter the hash. Patch `secrets.token_bytes` to a known 32-byte value only in this unit test and assert it is called once.

Add this exact path/context mutation table:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "relative_root",
        "noncanonical_root",
        "missing_root",
        "root_symlink_component",
        "root_not_directory",
        "metadata_symlink",
        "context_preexists",
        "receipt_root_preexists",
        "materialized_root_preexists",
        "one_declared_output_preexists",
        "one_group_receipt_preexists",
        "plan_file_mismatch",
        "registry_file_mismatch",
        "duplicate_json_key",
        "unknown_context_key",
        "noncanonical_context_hash",
        "second_context_create",
    ],
)
def test_context_and_path_boundaries_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _mutated_context_inputs(tmp_path, mutation)
    with pytest.raises(P6SourceReuseEvidenceError) as captured:
        _exercise_context_boundary(mutation, inputs)
    assert captured.value.public_category in {
        "history_execution_invalid",
        "unsafe_destination",
    }
```

`_mutated_context_inputs()` must create the exact plan leaf `pyramid_candidate_plan.json`, registry leaf `source_registry.json`, deterministic receipt leaf, and one of the 11 contract paths without searching. `resolve_existing_source_reuse_paths()` must fail until a valid context and real receipt directory exist; `plan_source_reuse_paths()` must allow both to be absent while validating every existing ancestor.

- [ ] **Step 2: Run the path/context RED slice**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_source_reuse_evidence_paths.py \
  -q
```

Expected: collection FAIL with `ModuleNotFoundError: framework.stage6.p6_source_reuse_evidence_v1`.

- [ ] **Step 3: Implement strict JSON, deterministic paths, and exclusive context publication**

Create `p6_source_reuse_evidence_v1.py`. Use the approved hash bytes exactly:

```python
def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _validated_resolved_local_root(root: Path) -> Path:
    if not root.is_absolute() or os.path.normpath(str(root)) != str(root):
        raise P6SourceReuseEvidenceError(public_category="unsafe_destination")
    _require_existing_nonsymlink_directory_components(root)
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise P6SourceReuseEvidenceError(public_category="unsafe_destination")
    return resolved_root

def plan_source_reuse_paths(local_output_root: Path) -> P6SourceReusePaths:
    resolved_root = _validated_resolved_local_root(local_output_root)
    return P6SourceReusePaths(
        local_output_root=resolved_root,
        metadata_root=resolved_root / RUN_METADATA_RELATIVE_ROOT,
        run_context=resolved_root / RUN_CONTEXT_RELATIVE_PATH,
        receipt_root=resolved_root / GROUP_RECEIPT_RELATIVE_ROOT,
    )

def _local_root_sha256(resolved_root: Path) -> str:
    return hashlib.sha256(
        b"p6-local-output-root-v1\0"
        + str(resolved_root).encode("utf-8")
    ).hexdigest()

def _group_key_sha256(group_id: str) -> str:
    return hashlib.sha256(
        b"p6-group-receipt-v1\0" + group_id.encode("utf-8")
    ).hexdigest()
```

Lexical absolute/canonical spelling, component `lstat`, symlink, and directory-type checks occur before the exact `resolved_root = root.resolve(strict=True)` assignment above. `P6SourceReusePaths.local_output_root` always stores that resolved object. `create_fresh_run_context()` begins with `paths = plan_source_reuse_paths(local_output_root)` and hashes `paths.local_output_root`; `load_fresh_run_context()` begins with `paths = resolve_existing_source_reuse_paths(local_output_root)` and recomputes the hash from the same `paths.local_output_root`. Both resolve plan/registry/context/receipt leaves only from `paths`; neither hashes or joins from the raw function argument, re-resolves a marker-derived path, or permits raw-root drift.

Use these exact initial bindings in the two functions:

```python
# create_fresh_run_context(...)
paths = plan_source_reuse_paths(local_output_root)
resolved_root = paths.local_output_root
root_sha256 = _local_root_sha256(resolved_root)
plan_path = resolved_root / "pyramid_candidate_plan.json"
registry_path = resolved_root / "source_registry.json"

# load_fresh_run_context(...)
paths = resolve_existing_source_reuse_paths(local_output_root)
resolved_root = paths.local_output_root
root_sha256 = _local_root_sha256(resolved_root)
plan_path = resolved_root / "pyramid_candidate_plan.json"
registry_path = resolved_root / "source_registry.json"
```

All subsequent context-path and receipt-path access uses `paths`; the two local `resolved_root` bindings are the sole source of plan/registry joins and local-root fingerprints.

Strict JSON loading uses `object_pairs_hook` to reject duplicate keys and exact schema-key sets; reject booleans where integers are required, non-finite numbers, noncanonical path strings, invalid schema literals, uppercase/non-64-hex hashes, and unknown keys. `_publish_exclusive_json()` writes canonical bytes to a same-directory `O_CREAT|O_EXCL|O_NOFOLLOW` mode-`0600` temp file, flushes and `fsync`s it, publishes with `renameat2(RENAME_NOREPLACE)` or same-filesystem hard-link no-replace semantics, unlinks only its temp, and `fsync`s the parent. Ordinary `os.replace()` is forbidden. If the final appears at any point, fail without modifying it.

`create_fresh_run_context()` must, before calling `secrets.token_bytes(32)`, validate the exact persisted plan/registry bytes against the supplied validated mappings using the leaves beneath `paths.local_output_root`, hash that same resolved object, and require the context, metadata/receipt namespace, deterministic receipt leaf for every registry group, `materialized` root, and every one of the 11 declared group leaves absent. It creates the metadata and empty receipt directories and publishes once. A crash after either directory create is intentionally not repairable in place.

`run_context_sha256` hashes the exact context object excluding only `run_context_sha256`; `receipt_sha256` hashes the exact receipt object excluding only `receipt_sha256`. `load_fresh_run_context()` must re-read the exact `pyramid_candidate_plan.json` and `source_registry.json` leaves beneath its resolved `paths.local_output_root`, validate their schemas and canonical hashes, recompute the root fingerprint from that same resolved object, and reject task/root/plan/registry/context drift without consulting environment state.

- [ ] **Step 4: Write RED artifact, receipt, classification, and tamper tests**

Create `tests/stage6/test_p6_source_reuse_evidence_receipts.py`. The happy path must build all exact leaves, not only markers:

```python
def test_first_use_publishes_adapter_receipt_and_reclassifies_ready(
    tmp_path: Path,
) -> None:
    fixture = _fresh_run_fixture(tmp_path, q_modes=("fp16",))
    before = classify_selected_group_sources(**fixture.classification_kwargs)
    assert [(item.group_id, item.state) for item in before] == [
        (fixture.group_id, "UNSEEN")
    ]
    _write_complete_source_bundle(fixture.request, fixture.group_id)

    receipt = validate_and_publish_group_receipt(
        fixture.request,
        group_id=fixture.group_id,
        **fixture.receipt_kwargs,
    )

    assert receipt.producer_row_id == min(
        row["row_id"]
        for row in fixture.request["rows"]
        if row["group_id"] == fixture.group_id
    )
    assert receipt.producer_measurement_request_sha256 == (
        fixture.request["measurement_request_sha256"]
    )
    assert first_use_group_ids(
        classify_selected_group_sources(**fixture.classification_kwargs)
    ) == ()
    assert require_selected_groups_ready_current_run(
        **fixture.ready_kwargs
    ) == (receipt,)
```

The helper writes regular files with `open(..., "xb")`, real directories, the training marker, then source marker, and sets supplemental first-use order to
`producer_request_mtime <= training_marker_mtime <= source_marker_mtime`. The private producer request is the exact binding-resolved request leaf, parses to the same canonical projected object, and recomputes the same request hash; public/private serializers may use different whitespace. For a same-round mixed-q request, it writes one shared bundle and the canonical producer is the lexicographically smallest row id.

Add this classification/tamper matrix and assert exact state plus zero public detail:

```python
@pytest.mark.parametrize(
    ("mutation", "expected_state"),
    [
        ("bare_training_and_source_markers", "INVALID_PARTIAL"),
        ("one_marker_only", "INVALID_PARTIAL"),
        ("artifact_without_receipt", "INVALID_PARTIAL"),
        ("receipt_only", "INVALID_PARTIAL"),
        ("receipt_missing_one_artifact", "INVALID_PARTIAL"),
        ("cross_run_receipt_copy", "INVALID_STALE"),
        ("different_run_nonce", "INVALID_STALE"),
        ("different_task", "INVALID_STALE"),
        ("different_revision", "INVALID_STALE"),
        ("different_root_fingerprint", "INVALID_STALE"),
        ("different_plan_hash", "INVALID_STALE"),
        ("different_registry_hash", "INVALID_STALE"),
        ("unknown_receipt_key", "INVALID_MISMATCH"),
        ("malformed_receipt_hash", "INVALID_MISMATCH"),
        ("wrong_group_key", "INVALID_MISMATCH"),
        ("artifact_byte_tamper", "INVALID_MISMATCH"),
        ("marker_byte_tamper", "INVALID_MISMATCH"),
        ("producer_request_tamper", "INVALID_MISMATCH"),
        ("producer_row_hash_tamper", "INVALID_MISMATCH"),
        ("consumer_contract_drift", "INVALID_MISMATCH"),
        ("consumer_evidence_drift", "INVALID_MISMATCH"),
        ("artifact_symlink", "INVALID_MISMATCH"),
        ("artifact_hard_link", "INVALID_MISMATCH"),
        ("directory_contains_symlink", "INVALID_MISMATCH"),
        ("directory_contains_hard_link", "INVALID_MISMATCH"),
        ("directory_contains_fifo", "INVALID_MISMATCH"),
        ("wrong_leaf_type", "INVALID_MISMATCH"),
        ("lexical_path_escape", "INVALID_MISMATCH"),
        ("resolved_parent_escape", "INVALID_MISMATCH"),
    ],
)
def test_group_classification_is_exact_and_fail_closed(
    tmp_path: Path, mutation: str, expected_state: P6SourceReuseState
) -> None:
    fixture = _mutated_ready_fixture(tmp_path, mutation)
    decision = classify_selected_group_sources(**fixture.classification_kwargs)[0]
    assert decision.state == expected_state
    assert expected_state not in str(P6HistoryMeasurementError("history_execution_invalid"))
```

Add dedicated tests that directory-tree digests are stable across creation order and hash the exact sorted POSIX-relative canonical entries `{"kind":"directory","path":"relative/path"}` and `{"kind":"regular_file","path":"relative/path","sha256":"...","size":N}`; the declared root itself is omitted and an empty directory hashes `[]`. File size is an integer, not bool; regular-file digests stream raw bytes. Receipt publication refuses a wrapper-created receipt even when its JSON is otherwise byte-perfect. The wrapper-created case must assert source has returned, downstream call log is empty, the leaf is preserved, and a new root is required.

- [ ] **Step 5: Run the receipt RED slice**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_source_reuse_evidence_receipts.py \
  -q
```

Expected: FAIL because classification, digest, producer-binding, and exclusive receipt publication functions are not implemented.

- [ ] **Step 6: Implement exact digest, classification, producer, and publication semantics**

Implement classification without scanning. For each sorted distinct request `group_id`, derive its exact receipt and 11 paths from that group's canonical projected contract:

```python
if receipt_absent and all(output_absent for output in outputs):
    state = "UNSEEN"
elif receipt_absent or any(output_absent for output in outputs):
    state = "INVALID_PARTIAL"
else:
    state = _validate_complete_receipt_or_return_invalid_state(...)
```

Every leaf check uses `lstat`; a symlink counts as existing and invalid, never absent. A complete receipt whose run-level fields disagree with the loaded context is `INVALID_STALE`. A context-bound receipt with schema/hash/group/producer/contract/evidence/type/digest/order drift is `INVALID_MISMATCH`. `first_use_group_ids()` accepts only decisions whose states are all executable, returns sorted `UNSEEN` ids, and raises `P6SourceReuseEvidenceError` with public `history_execution_invalid` for any invalid state.

Producer validation resolves the exact request using `interface["output_layout"]["round_root_template"]` and `producer_round_index`; it never searches. Require producer round `0..3` and `<= consumer round`, exact request hash, exact row hash, group id, q-mode, source-contract hash, and source-evidence hash. The producer row is the lexicographically smallest same-group row in the first-use request. Consumer row id and q-mode may differ, but group source-contract/source-evidence hashes must match.

For first use only, require receipt and all 11 outputs absent immediately before the source call. After a zero-return wrapper, require receipt still absent, all outputs exact and safe, and
`producer_request_mtime <= training_marker_mtime <= source_marker_mtime`. Compute nine artifact and two marker digests and exclusively publish. On later validation, ignore current consumer request mtime and recompute content digests; marker mtimes alone never authorize reuse.

- [ ] **Step 7: Write RED measurement integration tests for first use, same-round mixed q, and later reuse**

Create `tests/stage6/test_p6_source_reuse_measurement.py` with a public adapter fixture whose fake source wrapper writes the complete 11-leaf bundle but never a receipt. The real `run_history_measurement_batch()` must publish receipts.

```python
def test_same_round_fp16_int8_share_one_source_call_but_both_run_downstream(
    tmp_path: Path,
) -> None:
    fixture = _measurement_fixture(
        tmp_path,
        rounds=(("shared", "fp16"), ("shared", "int8"),
                ("other-a", "fp16"), ("other-b", "int8")),
    )
    feedback = run_history_measurement_batch(**fixture.call_kwargs)

    assert fixture.source_group_ids == tuple(sorted({
        fixture.shared_group_id,
        fixture.other_a_group_id,
        fixture.other_b_group_id,
    }))
    assert fixture.downstream_row_ids_by_stage == {
        stage: tuple(sorted(row["row_id"] for row in fixture.request["rows"]))
        for stage in ("quantization", "performance", "ap", "finalization")
    }
    assert {row["row_id"] for row in feedback["rows"]} == {
        row["row_id"] for row in fixture.request["rows"]
    }
```

Add this focused routing regression in the same file; import `framework.stage6.p6_history_measurement_v1 as measurement` so the loader seam is the one used by the public adapter:

```python
def test_measurement_context_root_is_exact_resolved_public_round_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _measurement_fixture(
        tmp_path,
        rounds=(("shared", "fp16"), ("shared", "int8"),
                ("other-a", "fp16"), ("other-b", "int8")),
    )
    expected_root = fixture.public_round_root.parent.resolve(strict=True)
    decoy_marker_parent = tmp_path / "decoy-marker-parent"
    decoy_marker_parent.mkdir()
    (decoy_marker_parent / "training.done").write_text("done\n", encoding="utf-8")
    observed_roots: list[Path] = []
    real_loader = measurement.load_fresh_run_context

    def recording_loader(**kwargs: Any) -> P6FreshRunContext:
        observed_roots.append(kwargs["local_output_root"])
        return real_loader(**kwargs)

    monkeypatch.setattr(measurement, "load_fresh_run_context", recording_loader)
    measurement.run_history_measurement_batch(**fixture.call_kwargs)

    assert decoy_marker_parent.resolve(strict=True) != expected_root
    assert observed_roots == [expected_root]
```

The fixture must make the supplied public round root and its parent existing, absolute, nonsymlink directories. The assertion proves context resolution uses only that exact validated resolved parent; add a decoy marker-shaped path elsewhere and assert it is never passed to the loader so no marker path can become a root oracle.

Add `test_round0_fp16_first_use_then_round2_int8_reuses_source_bundle()` and the reverse INT8-producer/FP16-consumer case. Both create one context once, execute two exact round requests, assert first-use source count one for the repeated group, later source count zero for that group, producer round remains earlier, later q-specific downstream stages contain the later row, and receipt bytes are unchanged. Add a ready-only request with four groups and assert zero source calls plus all four downstream rows.

Add failures for context absent, context invalid, partial/stale/mismatched group, wrapper-created receipt, source nonzero, missing artifact, missing marker, reversed first-use marker order, artifact tamper between classification and downstream revalidation, and private token redaction. Context and initial partial/stale/mismatch failures must assert `gpu_probe.calls == 0` and `runner.calls == []`; failures after valid GPU/activation/source admission must still assert zero downstream calls.

Retain the paused exact-env assertion in focused form:

```python
assert set(source_call.env) == set(EXPECTED_HISTORY_ENV_KEYS)
assert set(source_call.env) == {
    "CUDA_VISIBLE_DEVICES",
    "P6_HISTORY_RUN_MODE",
    "P6_HISTORY_PRIVATE_ROOT",
    "P6_HISTORY_TASK_STATE",
    "P6_HISTORY_ROUND_OUTPUT_ROOT",
}
assert not {
    "PYTHONPATH", "P6_RUN_CONTEXT", "P6_RUN_NONCE",
    "P6_SOURCE_RECEIPT", "P6_LOCAL_OUTPUT_ROOT",
} & set(source_call.env)
assert source_call.cwd == fixture.private_round_root
assert source_call.shell is False
```

- [ ] **Step 8: Run the measurement RED slice**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_source_reuse_measurement.py \
  -q
```

Expected: FAIL on the reviewed Task 3 base because measurement still applies blanket marker absence/builds every group invocation and does not load context or publish receipts. The exact-root routing regression is also RED on the clean Task 3 base; if the compatible paused work already makes only that assertion pass, retain it and continue the remaining RED failures.

- [ ] **Step 9: Route measurement through current-run classification**

Refactor `run_history_measurement_batch()` in this exact order:

First, extend the existing private resolver in Task 4. Its first operations and returned routing keys are exact:

```python
validated_supplied_root = _validate_controller_round_root(
    supplied_public_round_root
)
local_output_root = _validate_controller_round_root(
    validated_supplied_root.parent
)

paths = {
    "history_root": private_root,
    "local_output_root": local_output_root,
    "round_root": round_root,
    "measurement_request": request_path,
    "task_state": task_state,
    "actual_feedback": result,
    "actual_receipt": receipt,
    "finalization_barrier": barrier,
}
```

Here `round_root`, `request_path`, `task_state`, `result`, `receipt`, and `barrier` remain the existing binding-template-resolved private paths. `_validate_controller_round_root()` performs componentwise `lstat`, rejects symlinks and nondirectories, and returns `path.resolve(strict=True)`. It validates the supplied public round before deriving only `validated_supplied_root.parent`, then validates and resolves that parent separately. Do not derive `local_output_root` from a marker, source artifact, private round path, or receipt. Task 5 must promote these exact semantics, arguments, and keys rather than repair an unavailable root later.

Then route the adapter in this order:

```python
projected = project_source_materialization_request(verified_request)
canonical_request = projected.request
private_root, gpu_policy = _validate_binding_runtime(binding)
paths = _resolve_round_paths(
    interface,
    private_root,
    Path(round_output_root),
    canonical_request["round_index"],
)
reuse_required = requires_current_run_source_evidence(canonical_request)
if reuse_required:
    run_context = load_fresh_run_context(
        local_output_root=paths["local_output_root"],
        expected_task_id=canonical_request["task_id"],
        expected_task_sha256=canonical_request["task_sha256"],
    )
    decisions = classify_selected_group_sources(
        canonical_request,
        run_context=run_context,
        local_output_root=paths["local_output_root"],
        interface=interface,
        private_root=private_root,
    )
    unseen_group_ids = first_use_group_ids(decisions)
else:
    run_context = None
    unseen_group_ids = projected.ordered_group_ids
source_invocations = (
    build_source_invocations(
        paths["measurement_request"],
        unseen_group_ids,
        source_materializer=Path(interface["execution_chain"][0]["argv"][0]),
        validated_gpu_policy=gpu_policy,
    )
    if unseen_group_ids
    else ()
)
```

Then run GPU admission, initialize the private request/task-state, render the exact environment, and execute activation. In the recipe-v2 branch, immediately reclassify each unseen group as `UNSEEN`, run only its source invocation, require the wrapper did not create the receipt, and call `validate_and_publish_group_receipt()` once per sorted unseen group. Call `require_selected_groups_ready_current_run()` immediately before `for stage in interface["execution_chain"][1:]`; only then execute quantization, performance, AP, and finalization for the full unchanged four-row request. In the true legacy/v1 branch, preserve the existing all-group source flow and do not create/load context or receipts. Mixed or malformed recipe-v2 identity has already failed Task 3 projection and must never fall back to legacy.

Measurement must not import `secrets`, call `create_fresh_run_context()`, write plan/registry/context bytes, or add an environment key. Keep exact-env enforcement owned by `EXPECTED_HISTORY_ENV_KEYS`. Move compatible paused filesystem helpers into the focused evidence module; `p6_history_source_materialization_v1.py` remains the owner of projection and direct source invocation shape, not receipt state.

- [ ] **Step 10: Run focused GREEN tests and legacy regressions**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_source_reuse_evidence_paths.py \
  tests/stage6/test_p6_source_reuse_evidence_receipts.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/stage6/test_p6_history_source_materialization_training_contract.py \
  tests/release/test_p6_history_execution_adapters.py \
  -q
```

Expected: PASS. Existing v1/static measurement tests remain green without a run context; the context/receipt lane activates only for identifiable recipe-v2 requests.

- [ ] **Step 11: Coverage, size, lint, and privacy gates**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. coverage run \
  --source=framework.stage6.p6_source_reuse_evidence_v1 \
  -m pytest \
  tests/stage6/test_p6_source_reuse_evidence_paths.py \
  tests/stage6/test_p6_source_reuse_evidence_receipts.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  -q
coverage report -m --fail-under=80
python -m ruff check \
  framework/stage6/p6_source_reuse_evidence_v1.py \
  framework/stage6/p6_history_measurement_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  tests/stage6/test_p6_source_reuse_evidence_paths.py \
  tests/stage6/test_p6_source_reuse_evidence_receipts.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/release/test_p6_history_execution_adapters.py
wc -l \
  framework/stage6/p6_source_reuse_evidence_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  tests/stage6/test_p6_source_reuse_evidence_paths.py \
  tests/stage6/test_p6_source_reuse_evidence_receipts.py \
  tests/stage6/test_p6_source_reuse_measurement.py
git diff --check
```

Expected: focused coverage is at least 80%; every listed new/focused file and both touched below-limit production modules are under 800 lines; existing over-limit measurement/release test modules contain only fixture/routing changes; all other commands exit 0. Injected `PRIVATE-TOKEN-P6-REUSE` appears in neither exception text nor public feedback.

- [ ] **Step 12: Fresh implementer/reviewer boundary**

Review only Task 4-owned evidence/runtime files and the migrated paused diff. Reject Critical/Important if any of these hold: global group uniqueness is introduced; a later q-mode retrains; bare markers authorize reuse; measurement creates context; wrapper can publish a receipt; receipt/context is searched or overwritten; directory/file digest safety is incomplete; the private resolver omits `local_output_root`, leaves it unresolved, or derives it from a marker/private leaf; create/load hash a raw root instead of their `P6SourceReusePaths.local_output_root`; a reused group's current request mtime is compared to old markers; source calls include ready groups; downstream omits a q-mode row; or private values reach a public error. Require a fresh reviewer to rerun the Task 4 focused commands before acceptance.

- [ ] **Step 13: Commit**

```bash
git add \
  framework/stage6/p6_source_reuse_evidence_v1.py \
  framework/stage6/p6_history_measurement_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  tests/stage6/test_p6_source_reuse_evidence_paths.py \
  tests/stage6/test_p6_source_reuse_evidence_receipts.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/release/test_p6_history_execution_adapters.py
git commit -m "feat: add current-run P6 source reuse evidence"
```

---

### Task 5: Zero-Process Preflight and Relaunch Safety

**Files:**

- Create: `tools/release/preflight_p6_materializer_training_bridge.py`
- Create: `tools/release/verify_p6_materializer_training_run.py`
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Create: `tests/stage6/test_p6_history_round_paths.py`
- Create: `tests/stage6/test_p6_fresh_run_controller.py`
- Test: `tests/release/test_preflight_p6_materializer_training_bridge.py`
- Test: `tests/release/test_verify_p6_materializer_training_run.py`

**Interfaces:**

- Consumes: Task 4's exact run-context/receipt APIs, validated public/local contracts, the private binding and wrapper profile, existing `validate_search_task()`, Task 3's plan/registry identities, exact round templates, `translate_history_feedback()`, and frozen Gold176 inputs.
- Produces:

```python
@dataclass(frozen=True)
class P6MaterializerPreflightReport:
    schema_version: Literal["p6_materializer_training_bridge_preflight_v1"]
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
    schema_version: Literal["p6_materializer_training_bridge_completion_v1"]
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

def plan_validated_history_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    round_index: int,
) -> dict[str, Path]: ...

def resolve_validated_history_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    supplied_public_round_root: Path,
    round_index: int,
) -> dict[str, Path]: ...
```

`resolve_validated_history_round_paths()` is the exported Task 5 form of Task 4's private `_resolve_round_paths()`: the same four positional arguments, validation order, private path values, `history_root`, and `local_output_root` return key. Its `local_output_root` remains exactly the separately validated `validated_supplied_root.parent.resolve(strict=True)`. Exporting it must not add, remove, rename, or reinterpret a mapping key, and must never derive the root from markers, artifacts, receipts, or private paths.

- The preflight CLI prints only the dataclass fields above. `historical_process_launch_count` counts prohibited activation/history/training/materialization/measurement runners; current-process validation, JSON/YAML parsing, and existing read-only `git check-ignore` checks are not launches.
- Zero-process preflight is run before Stage1. It validates schemas, target, exact five-key environment, self-contained wrapper, static training inputs, and exact planned destinations. It requires absent: `.p6-materializer-training-bridge-v1`, context, receipt namespace/leafs, the fixed `materialized` root (therefore any of the 11 output leaves), Stage1 manifest, plan, registry, state, all four public round roots, and every binding-resolved private round root/request/task-state/result/completion-receipt/barrier destination. Binding/config inputs intentionally present in the local root are allowed.
- After Stage1/Stage2 registry creation, `create_fresh_run_context()` performs the second exact freshness gate over every deterministic group receipt plus every one of the 11 declared leaves. The controller does not duplicate or weaken that gate.
- It must not import historical Python modules, execute wrappers, call GPU probes, call TVM/AP/training code, or read raw private logs.
- Completion resolves the one context, four public/private requests, task-state/result/actual completion receipt/barrier, and group receipts through exact constants/templates only. It reuses `translate_history_feedback()`, requires successful terminal status and five finite metrics for every row, validates current artifact/marker digests and exact producer mapping, proves 16 unique row ids and zero Gold176 overlap, and allows multiple rows to share one group receipt. The public report never emits distinct receipt count or mapping.

- [ ] **Step 1: Write RED planned-absent versus runtime-existing path tests**

Create `tests/stage6/test_p6_history_round_paths.py`:

```python
def test_planner_allows_absent_private_round_but_runtime_requires_public_round(
    tmp_path: Path,
) -> None:
    private_root, interface = _safe_binding_interface(tmp_path)
    planned = plan_validated_history_round_paths(interface, private_root, 2)
    assert planned["round_root"] == private_root / "private-runs/2"
    assert planned["measurement_request"] == (
        private_root / "private-runs/2/measurement-request.json"
    )
    assert all(path.is_absolute() for path in planned.values())

    public_round = tmp_path / "local-output/round-02"
    with pytest.raises(P6HistoryMeasurementError):
        resolve_validated_history_round_paths(
            interface, private_root, public_round, 2
        )

    public_round.mkdir(parents=True)
    resolved = resolve_validated_history_round_paths(
        interface, private_root, public_round, 2
    )
    assert set(resolved) == {
        "history_root",
        "local_output_root",
        "round_root",
        "measurement_request",
        "task_state",
        "actual_feedback",
        "actual_receipt",
        "finalization_barrier",
    }
    assert resolved["local_output_root"] == public_round.parent.resolve(strict=True)
    assert {
        key: value for key, value in resolved.items()
        if key not in {"local_output_root", "history_root"}
    } == planned
```

Add exact rejections for round index `-1`, `4`, boolean index, template escape, duplicate private destinations, symlinked existing parent, non-directory parent, relative private root, and a runtime public-round symlink. Assert neither API creates directories or files.

Add `test_exported_runtime_resolver_preserves_task4_local_root_contract()`: assert the exported result has exactly `history_root`, `local_output_root`, and the planner's private keys; assert its planner-key subset equals `plan_validated_history_round_paths(...)`; and assert `local_output_root` is the exact resolved public-round parent. Include a decoy marker path and assert the result does not depend on it. This locks Task 4's runtime contract without requiring two competing resolver implementations.

- [ ] **Step 2: Run the path RED slice**

Run:

```bash
PYTHONPATH=. pytest tests/stage6/test_p6_history_round_paths.py -q
```

Expected: FAIL because the reviewed Task 4 state has the correct private runtime resolver but does not yet expose `plan_validated_history_round_paths()` or `resolve_validated_history_round_paths()`.

- [ ] **Step 3: Split exact round planning from runtime resolution**

Extract the planned portion of the existing `_resolve_round_paths()` into `plan_validated_history_round_paths()` and promote/rename its runtime portion to `resolve_validated_history_round_paths()`. The planner validates templates, canonical lexical spelling, every existing component with `lstat`, resolved containment beneath `private_root`, destination uniqueness, and round `0..3`, while allowing the round root and leaves absent. The runtime function first validates the supplied public round as an existing absolute nonsymlink directory, derives only its parent, separately validates that parent, stores the resulting `parent.resolve(strict=True)` as `local_output_root`, calls the planner, and returns the planned mapping plus unchanged `history_root` and `local_output_root` keys. Do not infer the local root from a marker or any binding-derived private path, and do not weaken `_initialize_private_round()` create-only checks.

Update Task 4 measurement routing to call the exported runtime function; this is a rename/split of the same behavior, not a second resolver. Preserve the four positional argument types and order and every runtime return key/value from Task 4 so the adapter call site and evidence APIs experience no signature or root-identity drift.

- [ ] **Step 4: Write RED zero-process preflight tests**

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
        "metadata_namespace_preexists",
        "run_context_preexists",
        "receipt_namespace_preexists",
        "one_group_receipt_preexists",
        "materialized_root_preexists",
        "checkpoint_path_preexists",
        "checkpoint_dir_preexists",
        "config_path_preexists",
        "training_done_marker_preexists",
        "onnx_path_preexists",
        "onnx_report_path_preexists",
        "calibration_root_preexists",
        "calibration_npz_preexists",
        "calibration_summary_preexists",
        "trt_calibration_dir_preexists",
        "source_done_marker_preexists",
        "source_registry_preexists",
        "candidate_plan_preexists",
        "stage1_manifest_preexists",
        "state_preexists",
        "public_round_root_preexists",
        "private_round_root_preexists",
        "round_request_preexists",
        "task_state_preexists",
        "actual_feedback_preexists",
        "actual_completion_receipt_preexists",
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

    assert captured.value.failure_code in {
        "history_execution_invalid",
        "unsafe_destination",
        "source_registry_invalid",
    }
    assert _process_launch_log(tmp_path) == []
    assert _gpu_probe_log(tmp_path) == []
```

Each 11-output mutation is generated from the exact `SHARED_SOURCE_PATH_KEYS`, not a guessed filename. Assert injected strings `PRIVATE-PREFLIGHT-PATH` and `GPU-private-preflight` appear in neither CLI stdout nor stderr. Add an AST/mock gate proving the function never calls `_run_step`, `GpuProbe.snapshot`, `run_history_measurement_batch`, `subprocess.run` other than existing Git ignore validation, or any historical wrapper.

- [ ] **Step 5: Run the preflight RED slice**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  -q
```

Expected: FAIL with missing preflight module and missing exact metadata/materialized freshness gates.

- [ ] **Step 6: Implement zero-process preflight and CLI**

Create the report and `main()` with exact arguments `--contract`, `--local-config`, `--binding`, `--runner-template`, and `--source-wrapper-profile`. On accepted input print one sorted JSON object; on any private validation failure print only `preflight_failed\n` to stderr and return 1.

Use `plan_source_reuse_paths()` and `plan_validated_history_round_paths()` only. Require exact destination absence with `lstat` checks; reject the complete metadata and `materialized` namespaces before checking descendants. Validate the binding's recipe-v2 template through Tasks 1/2 without running it. Never delete an existing destination and never generate a nonce/context during preflight.

- [ ] **Step 7: Write RED controller context-order and relaunch tests**

Create `tests/stage6/test_p6_fresh_run_controller.py`:

```python
def test_controller_creates_one_context_after_registry_before_measurement(
    tmp_path: Path,
) -> None:
    fixture = _fresh_controller_fixture(tmp_path)
    events: list[str] = []
    fixture.on_stage1 = lambda: events.append("stage1")
    fixture.on_registry = lambda: events.append("registry")
    fixture.on_context_publish = lambda: events.append("context")
    fixture.on_measurement = lambda: events.append("measurement")

    state = run_p6_coptv2x_search(**fixture.call_kwargs)

    assert state.completed_rounds == 4
    assert events.count("context") == 1
    assert events.index("stage1") < events.index("registry")
    assert events.index("registry") < events.index("context")
    assert events.index("context") < events.index("measurement")
    context = load_fresh_run_context(
        local_output_root=fixture.local_output_root,
        expected_task_id=fixture.task_contract["task_id"],
        expected_task_sha256=fixture.task_contract["task_sha256"],
    )
    assert context.code_revision == "rev-fresh-context"
    assert context.candidate_plan_sha256 == canonical_json_sha256(fixture.plan)
    assert context.source_registry_sha256 == canonical_json_sha256(fixture.registry)
```

Add a parametrized controller freshness table for preexisting Stage1 manifest, plan, registry, state, public round root, metadata/context/receipt namespace, `materialized` root, deterministic group receipt, and each declared 11-output leaf. For every mutation assert context nonce generation count zero, measurement count zero, and stable failure. Add malformed revision and plan/registry drift cases. Add this relaunch anchor:

```python
def test_failed_root_cannot_relaunch_in_place(tmp_path: Path) -> None:
    fixture = _fresh_controller_fixture(tmp_path, fail_round=1)
    first = run_p6_coptv2x_search(**fixture.call_kwargs)
    assert first.status == "failed"

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(**fixture.call_kwargs)

    assert captured.value.failure_code in {
        "history_execution_invalid",
        "unsafe_output",
        "unsafe_destination",
    }
    assert fixture.second_run_processes == []
```

- [ ] **Step 8: Run the controller RED slice**

Run:

```bash
PYTHONPATH=. pytest tests/stage6/test_p6_fresh_run_controller.py -q
```

Expected: FAIL because the controller does not create a context, accepts existing exact output leaves, and permits an existing public round directory.

- [ ] **Step 9: Integrate exactly-once context creation and fail-closed relaunch guards**

At `run_p6_coptv2x_search()` entry validate `code_revision = validate_code_revision(code_revision)`. Keep zero-process preflight as the operator gate before Stage1. Inside the controller, require the exact Stage1 manifest absent before Stage1; exact plan/registry/state/public round destinations absent before writes; and never unlink stale feedback/request leaves.

After `_build_source_registry()` and `_validate_p6_source_space()` succeed in `framework_stage2_search_space` mode, and before the round loop, call exactly:

```python
if local.candidate_source_mode == "framework_stage2_search_space":
    task_contract = validate_search_task(task)
    if framework_plan is None:
        raise P6CoptV2XExecutionError(
            "source_registry_invalid", "source registry invalid"
        )
    try:
        create_fresh_run_context(
            local_output_root=local.local_output_root,
            task_contract=task_contract,
            code_revision=code_revision,
            candidate_plan=framework_plan,
            source_registry=source_registry,
        )
    except P6SourceReuseEvidenceError as error:
        raise P6CoptV2XExecutionError(
            error.public_category, "history execution invalid"
        ) from None
```

The branch above is the dynamic framework recipe-v2 lane. Static/v1 mode preserves its existing context-free compatibility path and receives no synthesized plan. Context creation owns nonce generation and the second exact registry/output freshness gate. There is no context creation in `_run_search_round()`, measurement, preflight, or relaunch. `_prepare_round_output()` rejects an existing round root rather than reusing it. Failure preserves diagnostics and makes the same root non-relaunchable.

- [ ] **Step 10: Write RED exact-layout completion verifier tests**

Add to `tests/release/test_verify_p6_materializer_training_run.py`:

```python
def test_completion_accepts_16_unique_rows_with_shared_current_run_receipts(
    tmp_path: Path,
) -> None:
    inputs = _completed_four_round_private_run(
        tmp_path,
        repeated_group_across_rounds=True,
        same_round_mixed_q=True,
        nonstandard_private_leaf_names=True,
    )

    report = verify_materializer_training_run(**inputs)

    assert report.status == "completed"
    assert report.completed_rounds == 4
    assert report.selected_rows == 16
    assert report.gold176_remeasured_rows == 0
    assert inputs.private_distinct_receipt_count < 16
    assert "receipt" not in dataclasses.asdict(report)
```

The fixture creates a producer in round 0 and a consumer in round 2, plus a same-round mixed-q group. Add this exact mutation table:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "context_missing",
        "context_hash_drift",
        "public_private_request_mismatch",
        "producer_request_missing",
        "producer_request_hash_drift",
        "producer_is_later_round",
        "producer_row_missing",
        "producer_row_hash_drift",
        "group_receipt_missing",
        "group_receipt_cross_run",
        "group_receipt_artifact_tamper",
        "group_receipt_marker_tamper",
        "task_state_missing",
        "actual_result_missing",
        "actual_completion_receipt_missing",
        "finalization_barrier_missing",
        "duplicate_selected_row_id",
        "duplicate_measurement_identity",
        "gold176_overlap",
        "terminal_failure_status",
        "metric_missing",
        "metric_extra",
        "metric_boolean",
        "metric_nan",
        "metric_infinite",
        "latency_nonpositive",
        "energy_nonpositive",
        "ap_out_of_bounds",
        "decoy_context_or_receipt",
    ],
)
def test_completion_rejects_invalid_exact_evidence(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _mutated_completed_run(tmp_path, mutation)
    with pytest.raises(P6CoptV2XExecutionError) as captured:
        verify_materializer_training_run(**inputs.call_kwargs)
    assert captured.value.failure_code == "history_execution_invalid"
    assert "PRIVATE-COMPLETION-TOKEN" not in str(captured.value)
```

The decoy case places plausible `run-context.json`/receipt files at noncanonical names and proves they are ignored. Add a positive case in which 16 rows map to 16 receipts and one with fewer receipts; receipt count is dynamic and never an acceptance constant.

- [ ] **Step 11: Run the verifier RED slice**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_verify_p6_materializer_training_run.py \
  -q
```

Expected: FAIL with missing verifier module and no row-to-current-run-receipt mapping validation.

- [ ] **Step 12: Implement the exact completion verifier and CLI**

Load the public contract/local config/binding and reconstruct the validated `SearchTask` through the same public profile inputs used by the controller. Resolve, never search:

- context through `RUN_CONTEXT_RELATIVE_PATH`;
- public request through `round-{round_index:02d}/measurement_request.json`;
- private request/task-state/result/actual completion receipt/barrier through `resolve_validated_history_round_paths()`;
- group receipt through `receipt_path_for_group()`.

For each round, require public and private projected request JSON to parse to equal canonical objects and both request hashes to recompute; do not require formatting whitespace to match. Call `require_selected_groups_ready_current_run()` and map each row's `group_id` to its validated receipt; repeated mappings are accepted. Revalidate producer request/row and current artifact/marker digests for every row, then call `translate_history_feedback()` and require four successful rows with `latency_ms > 0`, `energy_j > 0`, and finite `ap30`, `ap50`, `ap70` within `[0, 1]`. Accumulate exactly 16 unique row ids and 16 unique measurement identities using `str(row.get("manifest_job_id") or row["row_id"])`; compute frozen Gold176 identities with the same rule and require the sets disjoint without emitting either set.

The CLI accepts only `--contract`, `--local-config`, and `--binding`. On failure print `verification_failed\n`; on success print sorted dataclass JSON. Do not print receipt count, group/producer/q-mode, paths, hashes, nonce, mtimes, artifact details, or metric values.

- [ ] **Step 13: Run focused Task 5 GREEN tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_round_paths.py \
  tests/stage6/test_p6_fresh_run_controller.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  -q
```

Expected: PASS.

- [ ] **Step 14: Run Stage6/release regressions, coverage, static no-search, and privacy gates**

Run:

```bash
PYTHONPATH=. pytest tests/stage6 tests/release -q
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=80
python -m ruff check \
  framework/stage6/coptv2x_h800_search_v2.py \
  framework/stage6/p6_history_measurement_v1.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py \
  tests/stage6/test_p6_history_round_paths.py \
  tests/stage6/test_p6_fresh_run_controller.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py
rg -n '\.(glob|rglob)\(|os\.walk\(|filename guessing|marker-parent' \
  framework/stage6/p6_source_reuse_evidence_v1.py \
  framework/stage6/p6_history_measurement_v1.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py
git diff --check
```

Expected: suites pass, project coverage is at least 80%, Ruff/diff exit 0, and `rg` returns no matches. The two CLIs' stdout/stderr contain no injected private path, GPU id, group id, nonce, hash, or metric value.

- [ ] **Step 15: Fresh implementer/reviewer boundary**

Review only Task 5-owned controller/path/CLI files and tests. Reject if exporting the resolver changes Task 4's four positional arguments, runtime return keys, validation order, or resolved-parent `local_output_root`; if marker/private paths can determine that root; if context creation can occur twice or before task/plan/registry validation; preflight starts Stage1/process/GPU work; planned paths require existence; runtime accepts an absent public round; preexisting context/receipt/11-output/round leaves are ignored; a failed root can resume; completion requires 16 receipts instead of 16 rows; a producer may be later; a decoy file is searched; Gold176 enters measurement; or public output includes private mapping/details. Require a fresh reviewer to rerun Task 5 commands.

- [ ] **Step 16: Commit**

```bash
git add \
  framework/stage6/coptv2x_h800_search_v2.py \
  framework/stage6/p6_history_measurement_v1.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py \
  tests/stage6/test_p6_history_round_paths.py \
  tests/stage6/test_p6_fresh_run_controller.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py
git commit -m "feat: bind P6 reuse to one fresh controller run"
```

---

### Task 6: Zero-GPU Full Dynamic Lifecycle Gate

**Files:**

- Create: `tests/release/p6_source_reuse_lifecycle_fixture.py`
- Create: `tests/release/test_p6_source_reuse_lifecycle.py`
- No production file changes are expected; if the test exposes a missing production seam, return to the owning Task 4 or Task 5 review rather than hiding the gap in a fake.

**Interfaces:**

- Consumes: real `run_p6_coptv2x_search()`, real Stage2 plan/registry validation, real request projection, real Task 4 public measurement adapter and receipt publication, and Task 5 context/relaunch integration. The fixture injects Stage1/registry/selection/process boundaries only.
- Produces no production API. `p6_source_reuse_lifecycle_fixture.py` exports:

```python
@dataclass(frozen=True)
class OfflineReuseLifecycle:
    call_kwargs: Mapping[str, Any]
    local_output_root: Path
    private_root: Path
    gold176_row_ids: frozenset[str]
    selected_rows_by_round: tuple[tuple[tuple[str, str], ...], ...]
    process_records: list[Mapping[str, Any]]
    source_calls: list[Mapping[str, Any]]
    downstream_rows: dict[str, list[tuple[str, ...]]]

def build_offline_reuse_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> OfflineReuseLifecycle: ...
```

The deterministic selected schedule contains 16 unique row ids and intentionally includes both: one canonical group selected as FP16 in round 0 and INT8 in round 2; and another canonical group selected as FP16 plus INT8 in round 1. The test computes expected group/source counts from this schedule and never treats the fixture's candidate-pool or distinct-group count as a product constant.

- [ ] **Step 1: Write the RED repeated-group offline lifecycle fixture and gate**

Create the two files and add:

```python
def test_zero_gpu_dynamic_lifecycle_reuses_group_source_and_runs_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_offline_reuse_lifecycle(tmp_path, monkeypatch)
    state = run_p6_coptv2x_search(**fixture.call_kwargs)

    assert state.status == "completed"
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    selected = [item for round_rows in fixture.selected_rows_by_round for item in round_rows]
    assert len(selected) == len(set(selected)) == 16
    selected_group_ids = {group_id for group_id, _q_mode in selected}
    assert len(fixture.source_calls) == len(selected_group_ids)
    assert len(fixture.source_calls) < len(selected)
    assert {call["group_id"] for call in fixture.source_calls} == selected_group_ids
    assert all(
        len({call["group_id"] for call in fixture.source_calls if call["round"] == round_index})
        == len({
            group_id for group_id, _q_mode in fixture.selected_rows_by_round[round_index]
            if group_id not in {
                prior_group
                for prior_round in fixture.selected_rows_by_round[:round_index]
                for prior_group, _prior_q in prior_round
            }
        })
        for round_index in range(4)
    )
```

Add explicit repeated-group assertions without a literal distinct-group count:

```python
repeated_later_group = fixture.selected_rows_by_round[0][0][0]
same_round_group = fixture.selected_rows_by_round[1][0][0]
assert (repeated_later_group, "int8") in fixture.selected_rows_by_round[2]
assert {
    q_mode for group_id, q_mode in fixture.selected_rows_by_round[1]
    if group_id == same_round_group
} == {"fp16", "int8"}
assert len([call for call in fixture.source_calls if call["group_id"] == repeated_later_group]) == 1
assert len([call for call in fixture.source_calls if call["group_id"] == same_round_group]) == 1
for stage in ("quantization", "performance", "ap", "finalization"):
    assert len(fixture.downstream_rows[stage]) == 4
    assert {
        row_id for batch in fixture.downstream_rows[stage] for row_id in batch
    } == _selected_row_ids(fixture.selected_rows_by_round)
```

The fixture's source-stage fake must:

- receive the exact direct argv/five-key environment from the real public adapter;
- read the exact projected private request and write all nine artifacts plus two markers for only the requested first-use group;
- write directory contents in deliberately non-sorted creation order so the canonical digest is exercised;
- never know, receive, locate, or write the receipt/context path;
- never spawn a subprocess or touch a GPU.

The real public adapter, not the fake wrapper, calls `validate_and_publish_group_receipt()`. The fake downstream stages read the request and write exact task-state/result/actual completion receipt/barrier evidence for all four row ids and record `(row_id, q_mode)` per stage. They never execute TVM, AP, latency, energy, training, or historical code.

- [ ] **Step 2: Add dynamic Stage2/Gold176/context/receipt assertions**

In the same test, load the exact persisted files:

```python
registry = _read_json(fixture.local_output_root / "source_registry.json")
plan = _read_json(fixture.local_output_root / "pyramid_candidate_plan.json")
expected_candidate_count = _expected_candidate_count_from_fixture(fixture)
assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
assert len(registry["groups"]) == plan["structure_count"]
assert sum(len(group["available_q_modes"]) for group in registry["groups"]) == plan["candidate_count"]
assert plan["candidate_count"] == expected_candidate_count
assert expected_candidate_count >= 16
assert plan["candidate_count"] not in {343, 686}
assert fixture.call_kwargs["local"].candidate_source_mode == "framework_stage2_search_space"

context = _read_json(fixture.local_output_root / RUN_CONTEXT_RELATIVE_PATH)
assert context["created_before_round_index"] == 0
assert context["candidate_plan_sha256"] == canonical_json_sha256(plan)
assert context["source_registry_sha256"] == canonical_json_sha256(registry)
assert sum(
    record.get("event") == "context_publish"
    for record in fixture.process_records
) == 1
```

For every exact public request assert four rows, canonical hashes, and Gold176 disjointness:

```python
measured_row_ids: set[str] = set()
for round_index in range(4):
    request = _read_json(
        fixture.local_output_root
        / f"round-{round_index:02d}/measurement_request.json"
    )
    assert request["round_index"] == round_index
    assert len(request["rows"]) == 4
    assert all(row["source_contract"]["training_required"] is True for row in request["rows"])
    assert request["measurement_request_sha256"] == canonical_json_sha256({
        key: value for key, value in request.items()
        if key != "measurement_request_sha256"
    })
    measured_row_ids.update(row["row_id"] for row in request["rows"])
assert len(measured_row_ids) == 16
assert measured_row_ids.isdisjoint(fixture.gold176_row_ids)
assert not any(
    row_id in fixture.gold176_row_ids
    for stage_batches in fixture.downstream_rows.values()
    for batch in stage_batches
    for row_id in batch
)
```

Assert fit input counts are `[176]` for initial and `[180, 184, 188]` for online refits, derived from recorded accepted feedback. Assert no fixed source-call count and no receipt-count field in public state. Load each selected group's exact receipt through `receipt_path_for_group()`, validate it, and assert a repeated group's receipt bytes do not change after its later q-mode.

- [ ] **Step 3: Add explicit no-process/no-wrapper-receipt privacy gate**

Add:

```python
for record in fixture.process_records:
    assert record.get("kind") == "injected_boundary"
    assert record.get("launched") is False
assert not any(
    record.get("stage") in {"gpu", "training", "tvm", "ap", "latency", "energy", "history"}
    and record.get("launched")
    for record in fixture.process_records
)
assert all("receipt" not in call["argv"] and "context" not in call["env"] for call in fixture.source_calls)
public_bytes = (fixture.local_output_root / "state.json").read_bytes()
assert b"PRIVATE-OFFLINE-REUSE-TOKEN" not in public_bytes
```

- [ ] **Step 4: Run the RED lifecycle test**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_p6_source_reuse_lifecycle.py::test_zero_gpu_dynamic_lifecycle_reuses_group_source_and_runs_all_rows \
  -q
```

Expected: FAIL until the deterministic repeated-group fixture routes through the real current-run context/receipt adapter and all downstream row evidence.

- [ ] **Step 5: Implement only the offline fixture and injected boundaries**

The fixture builds a Stage2 space with more than 16 `(group_id, q_mode)` rows and monkeypatches only deterministic selection/model scoring so the schedule is reproducible. It invokes the real controller and measurement adapter in-process. The source fake writes exact kinds:

```python
FILE_KEYS = {
    "checkpoint_path", "config_path", "onnx_path", "onnx_report_path",
    "calibration_npz", "calibration_summary",
}
DIRECTORY_KEYS = {"checkpoint_dir", "calibration_root", "trt_calibration_dir"}
for key in DIRECTORY_KEYS:
    Path(contract[key]).mkdir(parents=True)
for key in FILE_KEYS:
    path = Path(contract[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{group_id}:{key}\n".encode("ascii"))
Path(contract["training_done_marker"]).write_bytes(b"trained\n")
Path(contract["source_done_marker"]).write_bytes(b"source-ready\n")
```

The fixture must not call a fake receipt writer. If adapter receipt publication is absent or wrong, the test stays RED and returns to Task 4.

- [ ] **Step 6: Run lifecycle and cross-task GREEN gates**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_p6_source_reuse_lifecycle.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/stage6/test_p6_fresh_run_controller.py \
  tests/release/test_verify_p6_materializer_training_run.py \
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
wc -l \
  tests/release/p6_source_reuse_lifecycle_fixture.py \
  tests/release/test_p6_source_reuse_lifecycle.py
git diff --check
```

Expected: all commands exit 0, total framework coverage is at least 80%, both new files are under 800 lines, and anonymous archive tests find no private values.

- [ ] **Step 8: Fresh implementer/reviewer boundary**

Review the full zero-GPU lifecycle only. Reject if the fixture publishes receipts, hard-codes a candidate/source/receipt count as a product invariant, omits the later-round or same-round mixed-q case, lets Gold176 enter a request, lets static P6.1 fallback satisfy framework mode, skips downstream work for a reused q-mode, launches GPU/training/TVM/AP/latency/energy/history code, or leaks private paths. Require a fresh reviewer to rerun the lifecycle and full gates.

- [ ] **Step 9: Commit**

```bash
git add \
  tests/release/p6_source_reuse_lifecycle_fixture.py \
  tests/release/test_p6_source_reuse_lifecycle.py
git commit -m "test: gate P6 source reuse lifecycle offline"
```

---

### Task 7: Private H800 Preflight, Fresh Four-Round Run, and Conditional Final Docs

**Files:**

- Conditionally modify after real success: `docs/AAAI27_RELEASE_AUDIT.md`
- Conditionally create after real success: `docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md`
- Do not modify tracked files if private preflight or real run fails.

**Interfaces:**

- Consumes: completed and freshly reviewed Tasks 1–6; ignored private source map, source history root, runner template, wrapper profile, and deployment roots; tracked public contract; actual H800/TVM environment; existing derive/normalize/provision/preflight/run/verify CLIs.
- Produces: one fresh normalized private deployment, one exclusively created context, first-use source training receipts, four exact rounds/16 actual measurements, and public-safe docs only after the completion verifier succeeds.

- [ ] **Step 1: Confirm clean tracked state and rerun all implementation gates**

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

- [ ] **Step 2: Validate operator-supplied private inputs without recording values**

Every real private path remains only in an environment variable or ignored/external artifact. Do not paste values into the shell history transcript, tracked docs, test output, or this plan:

```bash
: "${P6_PUBLIC_CONTRACT_JSON:?set to the existing public contract path}"
: "${P6_SOURCE_MAP_JSON:?set to the ignored procedural source map}"
: "${P6_SOURCE_HISTORY_ROOT:?set to the validated source history root}"
: "${P6_RUNNER_TEMPLATE_YAML:?set to the ignored fresh-deployment runner template}"
: "${P6_SOURCE_WRAPPER_PROFILE_YAML:?set to the ignored wrapper profile path}"
: "${P6_PRIVATE_DERIVATION_DIR:?set to a new ignored derivation directory}"
: "${P6_FRESH_PRIVATE_DIR:?set to a new ignored normalized deployment root}"
: "${P6_FRESH_LOCAL_OUTPUT_ROOT:?set to a new ignored real-run output root}"
: "${P6_CODE_REVISION_LABEL:?set to a public-safe non-digest revision label}"
test -f "${P6_PUBLIC_CONTRACT_JSON}"
test -f "${P6_SOURCE_MAP_JSON}"
test -d "${P6_SOURCE_HISTORY_ROOT}"
test -f "${P6_RUNNER_TEMPLATE_YAML}"
test -f "${P6_SOURCE_WRAPPER_PROFILE_YAML}"
```

Expected: all checks exit 0 without printing a path.

- [ ] **Step 3: Preflight exact freshness before deployment**

The derivation, normalized deployment, and local output roots must all be absent. Do not scan, empty, or repair an existing root:

```bash
test ! -e "${P6_PRIVATE_DERIVATION_DIR:?}"
test ! -e "${P6_FRESH_PRIVATE_DIR:?}"
test ! -e "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}"
install -d -m 0700 "${P6_PRIVATE_DERIVATION_DIR}"
```

If any `test ! -e` fails, stop and choose a new scoped ignored root. Do not delete the existing root.

- [ ] **Step 4: Derive the canonical recipe and normalize/deploy the complete private root**

Derive from the ignored source map and validated runner template, then use the normalizer as the reviewed deployment mechanism. It copies the complete approved component/module tree and required sibling imports, not only the source entrypoint. The relocated tree omits the raw marker wrapper; Task 2's renderer creates the unique validated marker during provisioning.

```bash
PYTHONPATH=. python tools/release/derive_p6_history_recipe.py \
  --source-map "${P6_SOURCE_MAP_JSON:?}" \
  --runner-template "${P6_RUNNER_TEMPLATE_YAML:?}" \
  --recipe-json "${P6_PRIVATE_DERIVATION_DIR:?}/recipe.json"
PYTHONPATH=. python tools/release/normalize_p6_history_root.py \
  --source-map "${P6_SOURCE_MAP_JSON:?}" \
  --history-root "${P6_SOURCE_HISTORY_ROOT:?}" \
  --private-dir "${P6_FRESH_PRIVATE_DIR:?}" \
  --runner-template "${P6_RUNNER_TEMPLATE_YAML:?}"
test -f "${P6_PRIVATE_DERIVATION_DIR}/recipe.json"
test -f "${P6_FRESH_PRIVATE_DIR}/derivation/recipe.json"
test -f "${P6_FRESH_PRIVATE_DIR}/legacy.local.yaml"
```

Expected: stdout is only `p6_history_recipe_derived` and `p6_history_root_normalized`; stderr is empty; tracked Git state remains unchanged. Reject any ambiguous component/source selection, missing import sibling, raw duplicate marker, symlink, or nonignored destination.

- [ ] **Step 5: Provision one fresh binding/config pair; do not patch an old pair**

Create the local root once and provision through the reviewed CLI:

```bash
install -d -m 0700 "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}"
PYTHONPATH=. python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "${P6_FRESH_PRIVATE_DIR:?}/legacy.local.yaml" \
  --runner-template "${P6_RUNNER_TEMPLATE_YAML:?}" \
  --local-output-root "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}" \
  --binding-output "${P6_FRESH_LOCAL_OUTPUT_ROOT}/binding.json" \
  --config-output "${P6_FRESH_LOCAL_OUTPUT_ROOT}/local-config.json" \
  --source-wrapper-profile "${P6_SOURCE_WRAPPER_PROFILE_YAML:?}"
```

Expected:

- old V2 binding lacking `training_required: true` fails preflight/provisioning;
- regenerated binding accepts only if the wrapper is self-contained and the source contract template has all static Pyramid training fields;
- the exact five-key environment remains unchanged;
- stdout is only `p6_full_chain_config_written`; no private path is echoed or stored in tracked output.

- [ ] **Step 6: Run the zero-process private preflight before Stage1/controller launch**

Run:

```bash
: "${P6_PUBLIC_CONTRACT_JSON:?}"
: "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}"
: "${P6_RUNNER_TEMPLATE_YAML:?}"
: "${P6_SOURCE_WRAPPER_PROFILE_YAML:?}"
PYTHONPATH=. python tools/release/preflight_p6_materializer_training_bridge.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_FRESH_LOCAL_OUTPUT_ROOT}/local-config.json" \
  --binding "${P6_FRESH_LOCAL_OUTPUT_ROOT}/binding.json" \
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

This preflight must report zero historical process launches and zero GPU probes. It validates exact absence in two namespaces:

- Stage1 manifest path is absent.
- `pyramid_candidate_plan.json`, `source_registry.json`, `state.json`, and public `round-00` through `round-03` are absent.
- `.p6-materializer-training-bridge-v1`, context, receipt namespace, and `materialized` are absent.
- For private history rounds 0 through 3, exact round root, request, task-state, actual result, actual completion receipt, and barrier paths are absent.
- Binding/config files intentionally present are valid and are not mistaken for stale run output.

If this fails, preserve private diagnostics, do not update docs, and do not relaunch in place.

- [ ] **Step 7: Execute exactly one fresh Stage1→Stage2→Gold176→4×4 H800/TVM controller run**

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
- The controller creates exactly one current-run context before round 0 measurement/GPU/activation.
- Every first-used group performs one real materialization/training call and publishes one adapter-owned receipt after all 11 outputs and first-use marker order validate.
- If the search selects a later q-mode of a group, it performs zero later source calls for that group while still running its quantization/TVM/AP/latency/energy path. If all 16 rows happen to have distinct groups, 16 receipts are valid; sharing is permitted, not forced.
- Same-round mixed q-mode, if selected, performs one source call and both downstream rows.
- Online refits occur after accepted feedback for rounds 0, 1, and 2 with counts 180, 184, and 188.
- The one controller invocation is the only run. No manual wrapper, Gold176 measurement, synthetic feedback, per-row retraining, partial retry, or second controller is allowed.
- No failed, non-finite, or fabricated metrics are treated as success.

- [ ] **Step 8: Run the exact completion verifier once**

Run the reviewed verifier against the exact binding and output-layout contract. It may inspect ignored private artifacts but prints only a public-safe summary:

```bash
PYTHONPATH=. python tools/release/verify_p6_materializer_training_run.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON:?}" \
  --local-config "${P6_FRESH_LOCAL_OUTPUT_ROOT:?}/local-config.json" \
  --binding "${P6_FRESH_LOCAL_OUTPUT_ROOT}/binding.json"
```

Expected: the verifier resolves one context, four exact public/private requests, every task-state/result/actual completion receipt/barrier, and each deterministic group receipt without scans. It accepts fewer than 16 distinct receipts when rows share groups, but requires exactly 16 unique row ids and successful row measurements. It validates current-run context/receipt identities, producer request/row from the same or an earlier round, all current artifact/marker hashes, exactly five finite metrics, positive latency/energy, AP in `[0, 1]`, and zero Gold176 overlap. Public output contains only:

```json
{
  "completed_rounds": 4,
  "gold176_remeasured_rows": 0,
  "schema_version": "p6_materializer_training_bridge_completion_v1",
  "selected_rows": 16,
  "status": "completed"
}
```

- [ ] **Step 9: On any failure, stop without docs or in-place resume**

If source/runtime/context/receipt/artifact/feedback/GPU validation fails, any selected candidate cannot train, Gold176 enters measurement, downstream omits a row, or metrics are absent/invalid:

- preserve ignored private diagnostics,
- do not replace failed feedback with synthetic success,
- do not repair/delete individual context, receipts, markers, artifacts, requests, state, feedback, or barriers,
- do not skip or substitute candidates,
- do not switch to static registry,
- do not update final docs,
- after fixing the cause, restart derive/normalize/provision/preflight/run from new ignored derivation, deployment, and output roots.

- [ ] **Step 10: Conditionally update public docs only after verifier success**

Only after Step 8 succeeds, update docs with stable public-safe facts:

```markdown
## P6 V2 materializer training bridge

- Private preflight accepted the regenerated recipe-v2 binding with `training_required: true`.
- The self-contained source materializer wrapper ran from private round cwd with the strict five-key environment.
- Stage1/Stage2 dynamic Pyramid/H800/TVM candidate generation fed four P6 rounds.
- Gold176 was used only as cold-start cost-model evidence and was not remeasured.
- Four rounds completed with 16 unique selected measurement rows.
- Every selected row mapped to validated adapter-owned current-run source evidence before its downstream q-specific stages.
```

Do not disclose receipt count, which rows shared a receipt, producer row/round/q-mode, group/candidate ids, nonce, hash, path, mtime, artifact name/detail, metric value, dataset/checkpoint, GPU UUID, hostname, raw log, stdout/stderr, or static hyperparameter value.

- [ ] **Step 11: Run final docs/privacy and full repository gates**

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
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=80
```

Expected: all commands exit 0.

- [ ] **Step 12: Fresh real-run reviewer boundary**

Review exact verifier output and the conditional public-doc diff. Reject if closure depends on global group uniqueness, per-row retraining, bare markers, 16 receipts rather than 16 rows, a later-round producer, missing q-specific downstream work, non-finite metrics, Gold176 measurement, a relaunch, or private disclosure. The reviewer inspects ignored evidence locally but copies only stable public status into the review report.

- [ ] **Step 13: Commit only conditional docs**

Only if docs were updated after real success:

```bash
git add docs/AAAI27_RELEASE_AUDIT.md docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md
git commit -m "docs: record P6 materializer training bridge completion"
```

If docs were not updated because the private run failed or was not executed, do not commit this task.

---

## Task Ordering, Sequential Dependencies, and Ownership Conflicts

| Order | Owner | Depends on | Fresh review boundary | Shared-file rule |
| --- | --- | --- | --- | --- |
| Completed Task 1 | Training contract | None | Accepted in ledger | Historical content preserved |
| Completed Task 2 | Wrapper profile/provisioning | Task 1 | Accepted in ledger | Historical content preserved |
| Completed Task 3 | Projection/hash gates | Tasks 1–2 | Accepted in ledger | Historical content preserved |
| Task 4 | Evidence schemas, resolved-root paths/hashes, digests, classification, receipt publication, measurement routing | Task 3 | Fresh implementer + fresh focused reviewer | Owns new evidence module and interim private resolver `local_output_root` routing; only routing in measurement/source modules; migrates paused diff in place |
| Task 5 | Controller context creation, exact path planner/exported resolver, zero-process preflight, completion verifier, relaunch rejection | Task 4 | Fresh implementer + fresh focused reviewer | Formalizes Task 4's resolver semantics without signature/key/root drift, then edits controller; no parallel ownership |
| Task 6 | Offline repeated-group lifecycle | Tasks 4–5 | Fresh test implementer + fresh lifecycle reviewer | Test-only new files; production gaps return to owner |
| Task 7 | Private deploy/run/verification/docs | Tasks 1–6 all green | Fresh real-run evidence/docs reviewer | No tracked code; docs only after verifier success |

Do not implement Tasks 4 and 5 in parallel: both touch `p6_history_measurement_v1.py`, and Task 5 consumes Task 4's stable evidence APIs and validated resolved-parent routing. Task 4 must first add `local_output_root` to its private resolver; Task 5 may then extract the planner and export the runtime resolver while preserving its arguments, keys, and exact root derivation. Do not let Task 6 patch production under a test-only commit. The five paused Task 4 modifications remain in the worktree and are migrated by the Task 4 implementer; no task may reset/stash/discard them. Existing files over 800 lines receive routing or fixture migration only; all new focused files start below 800 lines and must remain there.

## Acceptance Matrix

| Scenario | Classification | Source invocation | Downstream q-specific stages | Acceptance |
| --- | --- | ---: | ---: | --- |
| New group; receipt and all 11 leaves absent | `UNSEEN` | Once for group | Every selected row | Validate bundle, adapter publishes receipt, proceed |
| FP16 producer; later INT8 same group/current run | `READY_CURRENT_RUN` | Zero later | INT8 row runs | Accept after producer/current digest validation |
| INT8 producer; later FP16 same group/current run | `READY_CURRENT_RUN` | Zero later | FP16 row runs | Accept after producer/current digest validation |
| FP16 + INT8 same group/same round | One `UNSEEN` group | Once | Both rows run | Lexicographically smallest row is producer |
| Four ready groups in a later request | All `READY_CURRENT_RUN` | Zero | All four rows run | Accept |
| Bare markers/artifacts without receipt | `INVALID_PARTIAL` | Zero | Zero | `p6_source_reuse_partial`; redacted stop |
| Receipt with any missing artifact/marker | `INVALID_PARTIAL` | Zero | Zero | `p6_source_reuse_partial`; redacted stop |
| Complete receipt copied from another run/root/task/revision/plan/registry | `INVALID_STALE` | Zero | Zero | `p6_source_reuse_stale`; redacted stop |
| Current receipt with artifact/marker/producer/contract hash drift | `INVALID_MISMATCH` | Zero | Zero | `p6_source_reuse_mismatch`; redacted stop |
| Wrapper creates a receipt | `INVALID_MISMATCH` after source | Source already returned | Zero | Preserve leaf; new root required |
| Symlink, hard link, path escape, FIFO/device/socket, wrong type | `INVALID_MISMATCH` | Zero unless created by wrapper | Zero | Redacted stop |
| Four rounds, 16 unique q-level rows, fewer than 16 valid receipts | All valid | Distinct first-use groups only | 16 rows | Completion accepts |
| Four rounds, 16 unique q-level rows, 16 valid receipts | All valid | 16 first-use groups | 16 rows | Completion accepts |
| Any Gold176 row enters measurement | Irrelevant | Stop | Stop | Completion rejects |
| Failed/partial root relaunched | Invalid fresh-run destination | Zero new work | Zero | New root required |

## Failure Category and Public-Privacy Matrix

| Internal condition | Private stable category | Public category/output |
| --- | --- | --- |
| Request/row/q-mode/request hash drift | Existing request validator | `history_request_invalid` |
| Partial source evidence | `p6_source_reuse_partial` | `history_execution_invalid` only |
| Cross-run/root/task/revision/plan/registry evidence | `p6_source_reuse_stale` | `history_execution_invalid` only |
| Receipt/artifact/marker/producer/type/link/path mismatch | `p6_source_reuse_mismatch` | `history_execution_invalid` only |
| Invalid binding/context/source contract/projection | Existing stable validator | `history_execution_invalid` |
| H800 identity/occupancy admission failure | Existing GPU validator | `history_gpu_admission_failed` |
| Validated private process returns nonzero | Existing executor | `history_execution_failed` |
| Preexisting/unsafe planned destination | Exact path gate | `unsafe_destination` or existing `unsafe_output` |
| Outer measurement command fails | Controller step | `command_failed` |

No public exception/report/state may include internal state name, path, argv, environment, group/row/q-mode, producer, receipt count/mapping, nonce, hash, mtime, artifact detail, metric value, GPU UUID, hostname, traceback, or raw stdout/stderr.

## Explicitly Rejected Implementations

- **A — global group uniqueness:** rejected because it changes the row-level `(group_id, q_mode)` search space/acquisition behavior and can prevent 16 valid selected rows.
- **B — per-row or per-q-mode retraining:** rejected because recipe-v2 owns one q-independent trained bundle per canonical group and duplicate training would race on shared paths.
- **Bare-marker reuse:** rejected because marker presence/mtime binds none of task, revision, root, plan, registry, nonce, producer request/row, contract, or artifact content.
- **In-place resume/repair:** rejected because create-only context/receipt/source evidence makes a partial root diagnostic evidence, not a resumable cache.

## Spec Requirement Mapping

| Spec requirement | Task coverage |
| --- | --- |
| Fix deterministic private import failure without widening public env | Task 2 wrapper profile and black-box import determinism; Task 4 exact five-key env |
| Enforce `training_required: true` and static Pyramid training fields | Task 1 binding/registry validation; Task 3 projection preservation |
| Preserve static fields through projection and hashes | Task 3 projection tests and hash recomputation |
| Reject incomplete recipe-v2 source contracts | Task 1 and Task 5 preflight |
| Render one shared source bundle per Pyramid group | Task 1 registry integration; Task 3 projection drift tests; Task 4 first-use receipt; Task 6 lifecycle |
| Remove untrusted output aliases before canonical group contract hash | Task 1 registry integration; Task 3 projection tests |
| Immutable run context bound to task/revision/root/plan/registry/nonce | Task 4 schemas/persistence; Task 5 controller integration |
| Deterministic exact paths and no search | Tasks 4–5 path APIs/static scans |
| Lexical/resolved/symlink/type/hard-link safety | Task 4 path/digest matrix; Task 5 planned/runtime roots |
| Adapter-only exclusive receipt publication | Task 4 publication/wrapper-receipt tests; Task 6 real adapter fake |
| Current-run first-use/reuse classification | Task 4 unit/integration; Task 6 full lifecycle |
| Public/private boundary and no private leakage | Tasks 1, 2, 4, 5, 6, 7 privacy gates |
| Source materializer direct argv once per first-use group | Tasks 2, 3, 4, 6 |
| Gate all downstream stages on validated current-run evidence | Tasks 4 and 6 |
| Same-round mixed q-mode and later-round reuse | Task 4 integration and Task 6 end-to-end |
| Zero-process preflight before Stage1/activation/GPU/history execution | Task 5 CLI; Task 7 invocation order |
| Zero-GPU black-box dynamic lifecycle | Task 6 |
| Fresh Stage1/Stage2 through four rounds | Task 6 fake lifecycle; Task 7 real H800 run |
| No Gold176 remeasurement | Task 6 and Task 7 |
| Completion maps 16 rows to a dynamic number of receipts | Task 5 verifier; Task 6 lifecycle; Task 7 real verifier |
| No in-place partial resume | Tasks 4, 5, and 7 |
| Preserve v1/static compatibility | Tasks 1, 3, 5 focused regression commands |
| Final docs conditional on real completion | Task 7 only |

---

## Final Full Gates Before Task 7 or Merge Handoff

Run from a clean tracked worktree after Task 6:

```bash
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=80
PYTHONPATH=. pytest tests/release tests/integration/test_anonymous_archive.py -q
python -m ruff check framework tools scripts tests
python -m compileall -q framework tools scripts tests
git diff --check
rg -n '\.(glob|rglob)\(|os\.walk\(' \
  framework/stage6/p6_source_reuse_evidence_v1.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py
python - <<'PY'
from pathlib import Path

limits = [
    Path("framework/stage6/p6_source_reuse_evidence_v1.py"),
    Path("tests/stage6/test_p6_source_reuse_evidence_paths.py"),
    Path("tests/stage6/test_p6_source_reuse_evidence_receipts.py"),
    Path("tests/stage6/test_p6_source_reuse_measurement.py"),
    Path("tests/stage6/test_p6_history_round_paths.py"),
    Path("tests/stage6/test_p6_fresh_run_controller.py"),
    Path("tools/release/preflight_p6_materializer_training_bridge.py"),
    Path("tools/release/verify_p6_materializer_training_run.py"),
    Path("tests/release/test_preflight_p6_materializer_training_bridge.py"),
    Path("tests/release/test_verify_p6_materializer_training_run.py"),
    Path("tests/release/p6_source_reuse_lifecycle_fixture.py"),
    Path("tests/release/test_p6_source_reuse_lifecycle.py"),
]
for path in limits:
    count = len(path.read_text(encoding="utf-8").splitlines())
    assert count < 800, f"{path} has {count} lines"
PY
```

Expected: all pytest/lint/compile/diff/privacy gates pass, coverage is at least 80%, the no-search `rg` has no matches, and every focused file is below 800 lines.

## Plan Self-Review

- [ ] Spec coverage: every goal, invariant, fail-closed category, public/private rule, TDD matrix row, provisioning rule, preflight rule, relaunch rule, and closure criterion maps to at least one task above.
- [ ] Placeholder scan: the plan contains no unresolved implementation marker, deferred-fill instruction, unbounded test instruction, or unresolved private example path. Task 7's named environment variables are required operator inputs guarded with `${VAR:?}`, not missing implementation details.
- [ ] Red-flag scan: run the literal-token scan requested by the reviewer against this plan and keep it clean before handoff.
- [ ] Type consistency: Task 4 owns `P6FreshRunContext`, `P6GroupSourceReceipt`, `P6GroupReuseDecision`, `plan_source_reuse_paths()`, `resolve_existing_source_reuse_paths()`, `create_fresh_run_context()`, `load_fresh_run_context()`, `classify_selected_group_sources()`, `first_use_group_ids()`, `validate_and_publish_group_receipt()`, and `require_selected_groups_ready_current_run()`; Tasks 5–7 consume those exact names/types. `P6SourceReusePaths.local_output_root` is always the validated `root.resolve(strict=True)`, and create/load join leaves and hash `str()` from that exact object, never the raw argument. Binding-owned `EXPECTED_HISTORY_ENV_KEYS` remains the sole five-key owner.
- [ ] Sequential consistency: Task 4 adds the exact resolved public-round parent as `local_output_root` to its existing private runtime resolver before measurement loads context. Task 5 exposes the planned/runtime pair and updates routing without positional signature, mapping-key, validation-order, or root-derivation drift. Task 6 changes tests only. Task 7 changes docs only after success.
- [ ] File-size check: new evidence/path/controller/lifecycle test files are split up front; existing over-limit controller/measurement/test files receive routing glue only.
- [ ] Security/privacy check: public projection, docs, stderr, tests, and release manifests must not contain private paths, GPU UUIDs, raw logs, checkpoints, datasets, hostnames, candidate IDs, or static training values.
- [ ] Reuse semantics check: no task enforces global group uniqueness, retrains per q-mode/row, treats bare markers as reuse authority, or compares later consumer request mtime to first-use markers.
- [ ] Completion check: Task 7 docs are explicitly conditional on zero-process preflight, one fresh real run, and the exact completion verifier accepting four rounds/16 rows/zero Gold176 overlap; receipt count may be below 16.
