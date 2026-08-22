# P6 V2 Materializer Training Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make P6 recipe-v2 train each canonical Pyramid source group from an exact ignored external-training binding on first use in one fresh run, reuse only adapter-receipted current-run source evidence for later q-modes, execute q-specific downstream work for all 16 selected rows, and complete without Gold176 remeasurement.

**Architecture:** Keep three independent roots: a normalized private root containing only an explicitly selected self-contained execution/module/wrapper/toolchain closure, read-only operator-owned dataset/checkpoint/Pyramid-config paths validated in place through an ignored `p6_external_training_binding_v1`, and one fresh private output root containing all mutable state plus the exact q-independent 11-output bundles. Keep the public adapter boundary strict: direct argv, `shell=False`, private round cwd, and exactly the five-key private environment. The completed current-run context/receipt design remains authoritative; the external binding is normalized into every recipe-v2 source contract, revalidated before the GPU boundary, and never copied into tracked files or either private destination root.

**Tech Stack:** Python 3.10+, pytest, Ruff, JSON/YAML canonical serialization, existing Stage1/Stage2/Stage5/Stage6 P6 modules, injected runner tests, optional private H800 execution only in final gated task.

**Spec:** `docs/superpowers/specs/2026-08-21-p6-v2-materializer-training-bridge-design.md` (approved external-binding authority at `5fa3715`)

## Revision status and historical boundary

Tasks 1–6 below are completed historical implementation records and remain intact for auditability. Their accepted commits are not rewritten. The external-training correction at `5fa3715` supersedes only contradictory clauses that place `dataset_root`, `base_checkpoint_path`, or `pyramid_config_path` beneath the private history Git root, copy training assets into the normalized root, or let a wrapper/program guess unbound defaults. Tasks 7–10 are the executable continuation: Task 7 defines the exact binding; Task 8 integrates independent external inputs with a self-contained all-role execution closure; Task 9 regenerates ignored private artifacts and reaches a mocked GPU boundary; Task 10 alone may perform the next real controller launch.

Attempts 1–3 and their ignored diagnostics remain preserved and unmodified. Their controller launch count is exactly `0`; they are neither execution evidence nor paper evidence. The next remote run, if Tasks 7–9 pass, uses a fresh derivation root, a fresh normalized code/toolchain root, fresh ignored binding/config artifacts, and a fresh local output root.

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
- Recipe-v2 source contracts must contain one exact nested `external_training_binding` object with `training_required: true`, `training_source_kind: selected_candidate_finetune`, three external paths, two normalized stable-file digests, and the exact `training_parameters`; they also retain `stage_widths` and exactly the 11 existing `shared_source_paths` keys.
- Required `training_parameters` semantic keys are `training_mode`, `epochs`, `seed`, `optimizer`, `learning_rate`, `batch_size`, `dataset_split`, `checkpoint_selection`, and `freeze_policy`.
- `dataset_root`, `base_checkpoint_path`, and `pyramid_config_path` are independent read-only external inputs. Each must be an existing, readable, absolute, lexically canonical POSIX path with every component checked by `lstat`, no symlink component, and resolved spelling equal to lexical spelling. Dataset is a readable/searchable real directory; checkpoint and config are readable regular files. They need not share a Git root with one another or with the normalized code/toolchain root.
- External paths are pairwise distinct and must not equal, contain, be contained by, or inode-alias the normalized code/toolchain root, fresh output root, any private round root, or any declared request/task-state/receipt/marker/artifact/result/feedback/barrier path. External inputs are never copied, hard-linked, vendored, rewritten, synthesized, or used as destinations.
- Checkpoint/config SHA-256 values are optional assertions in the ignored authoring binding and required computed lowercase 64-hex evidence after validation. Dataset trees are never implicitly traversed or hashed; `dataset_split` is the exact dataset selection binding.
- All nine generated artifacts and both markers remain absolute, symlink-safe, unique, canonical, and strictly beneath the fresh local output root. The output root contains exactly the existing 11 `shared_source_paths` leaves per group; no external binding may supply an output/result/receipt/marker path.
- A program-declared default is adopted only when code reads an explicit declaration that names the exact path, proves that exact existing canonical path, and copies that value into the ignored binding before validation. Basename search, sibling/cwd inference, environment fallback, glob, registry scan, and conventional-location guesses are rejected. Initial implementation may decline adoption and require an explicit ignored value; it may never guess.
- Fail closed before process launch for incomplete private wrapper, binding, recipe-v2 contract, output layout, source contract, static training contract, hash drift, stale state, unsafe symlink, or path collision.
- Bare markers never authorize reuse. Only an adapter-owned `p6_group_source_reuse_receipt_v1` bound to the immutable `p6_materializer_fresh_run_context_v1`, its producer request/row, and recomputed artifact/marker digests may authorize skipping a source call.
- Run context and receipt locations are exact under `.p6-materializer-training-bridge-v1`; no `glob`, `rglob`, basename search, marker-parent inference, or filename guessing may locate them. Bounded traversal is allowed only inside an already-declared directory artifact to compute its digest.
- Contexts and receipts are create-only, atomically published with no-replace semantics, mode `0600` beneath mode-`0700` parents, and never updated, repaired, overwritten, or accepted from a wrapper.
- All root, metadata, receipt, and declared artifact paths must pass canonical lexical spelling, component-wise `lstat`, resolved containment, exact type, symlink rejection, path-escape rejection, and single-link regular-file checks. The evidence path object stores the exact `root.resolve(strict=True)` reached after those checks, and both context create/load hash `str()` of that same resolved object; raw or marker-derived roots are forbidden. Directory digests reject symlinks, hard-linked regular files, sockets, devices, and FIFOs.
- Scope is paper reproducibility in a trusted single-user local/H800 environment. The preceding canonical-path checks reject ordinary malformed, missing, corrupt, stale, and observed symlinked artifacts; they are not a claim to close a concurrent directory-root symlink-swap/TOCTOU race, which is a documented nonblocking environmental limitation.
- The controller creates one fresh 32-byte nonce and run context only after validating the task, public-safe revision, local-root fingerprint, complete Stage2 plan, and registry identities, and before any measurement/GPU/activation boundary. No nonce/context/receipt/local-root environment key is added.
- Measurement requires an existing valid context and never creates one. `UNSEEN` means receipt plus all 11 leaves are absent; `READY_CURRENT_RUN` means the receipt and all bound bytes validate; partial, stale, or mismatched states stop before downstream execution.
- All four selected rows in every round continue through quantization, performance/TVM, AP, and finalization, including same-group mixed q-mode rows. Reuse skips only source materialization/training.
- Completion requires four exact requests, 16 unique row ids, 16 row-to-receipt mappings, all terminal success evidence and five finite metrics, and zero Gold176 overlap. Multiple rows may share one receipt, and a producer may be from the same or an earlier round.
- Stable private reuse categories are `p6_source_reuse_partial`, `p6_source_reuse_stale`, and `p6_source_reuse_mismatch`; all map to public `history_execution_invalid` without values.
- Existing v1/static compatibility must be preserved; stricter training enforcement applies only to recipe-v2 P6 real-run contracts.
- Files touched by implementation should stay under 800 lines. Existing files already over the limit must receive only routing glue plus a focused split module.
- Every task is RED→GREEN→REFACTOR, includes a review boundary, and uses a conventional commit message. Do not commit from the planning step.
- A failed or partial run is preserved for ignored private diagnosis and never resumed in place. Any retry uses a new scoped ignored local output root.
- Final public docs are conditional: update release/audit docs only after the private preflight and real four-round run actually complete successfully, with a public-safe protocol/config/results comparison rather than private run detail.
- The tracked `configs/execution/p6_external_training_binding.example.yaml` contains the exact schema and semantic constants with only intentional `null` private-value slots. It is documentation, carries no real path/digest/dataset/checkpoint/config/hyperparameter value, and must fail executable-binding validation.

---

## File Structure

- Create `framework/stage6/p6_external_training_binding_v1.py`: exact YAML loader, schema/type validation, component-wise external-path validation, streaming checkpoint/config digests, overlap/inode-alias rejection, detached normalized binding, and public redaction.
- Create `configs/execution/p6_external_training_binding.example.yaml`: tracked null-only documentation shape for `p6_external_training_binding_v1`; never executable.
- Create `framework/stage6/p6_history_execution_closure_v1.py`: exact all-role execution/module/wrapper/toolchain closure manifest, canonical copy plan, post-copy verification, and training-asset exclusion.
- Modify `framework/stage6/p6_history_normalization_v1.py` and `framework/stage6/p6_history_recipe_normalization_v1.py`: accept the recipe-v2 external object and explicit closure manifest, remove recipe-v2 same-Git-root training-asset assumptions, copy only closure roles/toolchain code, and write the normalized external object into ignored artifacts without copying its inputs.
- Modify `tools/release/derive_p6_history_recipe.py` and `tools/release/normalize_p6_history_root.py`: require the corrected recipe-v2 source-map shape and explicit external-binding/closure inputs while preserving stable redacted stdout/stderr.
- Modify `framework/stage6/p6_runner_template_validator_v1.py` and `framework/stage6/p6_source_wrapper_profile_v1.py`: validate the normalized runner against every declared runner role (`stage1_scan`, `controller`, five execution stages, and `activation`) and the complete source implementation/module closure.
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
- Create Task 7 focused tests: `tests/stage6/test_p6_external_training_binding.py` and the tracked example privacy assertions in `tests/integration/test_anonymous_archive.py`.
- Create Task 8 focused tests: `tests/stage6/test_p6_history_execution_closure.py`, plus external-binding integration cases in the existing normalization, binding, registry, source-materialization, measurement, wrapper, bootstrap, provisioning, and preflight test modules.
- Create Task 9 private-boundary gate: `tests/release/test_p6_external_training_deployment.py`; ignored regenerated artifacts remain outside Git and are never listed with real values in this plan.
- Conditionally modify final docs such as `docs/AAAI27_RELEASE_AUDIT.md` and a release manifest only after Task 10 succeeds.

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

**Trusted-local limitation:** Task 4 keeps canonical-path, containment,
`lstat`, and no-symlink checks for paths as observed. It does not add Task 4.1
or require an adversarial concurrent directory-root symlink-swap/TOCTOU race
to be prevented or tested for acceptance. The fresh local output root is
operator-controlled and exclusively used for the run; malformed, missing,
corrupt, stale, and straightforward symlinked artifacts still fail closed and
all public failures remain redacted.

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

Review only Task 4-owned evidence/runtime files and the migrated paused diff. Reject Critical/Important if any of these hold: global group uniqueness is introduced; a later q-mode retrains; bare markers authorize reuse; measurement creates context; wrapper can publish a receipt; receipt/context is searched or overwritten; malformed, missing, corrupt, stale, or observed symlinked evidence is accepted; the private resolver omits `local_output_root`, leaves it unresolved, or derives it from a marker/private leaf; create/load hash a raw root instead of their `P6SourceReusePaths.local_output_root`; a reused group's current request mtime is compared to old markers; source calls include ready groups; downstream omits a q-mode row; or private values reach a public error. Do not reject acceptance solely because a concurrent adversary could swap the directory root after validation; that documented TOCTOU limitation is outside this trusted-local scope. Require a fresh reviewer to rerun the Task 4 focused commands before acceptance.

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

**Remaining priority 1:** Demonstrate that the controller creates exactly one
fresh context only after the dynamic Stage1/Stage2 plan and registry validate,
then protect the resulting zero-process preflight, exact-layout verification,
and no-relaunch lifecycle. Straightforward canonical/no-symlink validation is
in scope; the documented concurrent root-swap race is not an acceptance gate.

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

Review only Task 5-owned controller/path/CLI files and tests. Reject if exporting the resolver changes Task 4's four positional arguments, runtime return keys, validation order, or resolved-parent `local_output_root`; if marker/private paths can determine that root; if context creation can occur twice or before task/plan/registry validation; preflight starts Stage1/process/GPU work; planned paths require existence; runtime accepts an absent public round; preexisting context/receipt/11-output/round leaves are ignored; a failed root can resume; completion requires 16 receipts instead of 16 rows; a producer may be later; a decoy file is searched; Gold176 enters measurement; or public output includes private mapping/details. An adversarial post-validation directory-root symlink swap remains outside this trusted-local review scope. Require a fresh reviewer to rerun Task 5 commands.

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

**Remaining priority 2:** Run the complete fresh dynamic lifecycle without GPU
work, proving the real controller/context/adapter interfaces across
Stage1→Stage2→Gold176→four rounds while deriving source-call expectations from
the fixture's dynamic candidate schedule rather than a fixed candidate count.

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

### Task 7 (A): Exact External-Training Binding, Digests, and Null-Only Public Example

**Files:**

- Create: `framework/stage6/p6_external_training_binding_v1.py`
- Create: `configs/execution/p6_external_training_binding.example.yaml`
- Modify: `framework/stage6/p6_history_training_contract_v1.py`
- Test: `tests/stage6/test_p6_external_training_binding.py`
- Test: `tests/stage6/test_p6_history_training_contract.py`
- Test: `tests/integration/test_anonymous_archive.py`

**Interfaces:**

- Consumes: one ignored YAML object with the exact `p6_external_training_binding_v1` shape; canonical normalized code/toolchain and fresh output roots; exact planned destination paths from the binding/output-layout APIs.
- Produces:

```python
ExternalTrainingParameter = str | int | float

@dataclass(frozen=True)
class P6ExternalTrainingBinding:
    schema_version: Literal["p6_external_training_binding_v1"]
    training_required: Literal[True]
    training_source_kind: Literal["selected_candidate_finetune"]
    dataset_root: Path
    base_checkpoint_path: Path
    base_checkpoint_sha256: str
    pyramid_config_path: Path
    pyramid_config_sha256: str
    training_parameters: tuple[tuple[str, ExternalTrainingParameter], ...]

class P6ExternalTrainingBindingError(ValueError):
    category: Literal["history_execution_invalid"]

def load_external_training_binding(path: Path) -> dict[str, Any]: ...

def validate_external_training_binding(
    raw: Mapping[str, Any],
    *,
    code_toolchain_root: Path,
    local_output_root: Path,
    reserved_paths: Sequence[Path],
) -> P6ExternalTrainingBinding: ...

def external_training_binding_to_mapping(
    binding: P6ExternalTrainingBinding,
) -> dict[str, Any]: ...

def bind_external_training_contract(
    contract: Mapping[str, Any],
    binding: P6ExternalTrainingBinding,
) -> dict[str, Any]: ...

def external_training_binding_from_contract(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]: ...

def public_safe_external_training_projection(
    binding: Mapping[str, Any],
) -> dict[str, Any]: ...
```

`load_external_training_binding()` accepts only an absolute, existing, readable, non-symlinked YAML file, rejects duplicate YAML keys and files over 1 MiB, and requires `git check-ignore` when the file is inside the tracked repository, except for the one exact tracked example path. It returns a detached mapping and never prints or embeds its path in an exception. `validate_external_training_binding()` requires the exact nine top-level keys and exact nine `training_parameters` keys from the spec. It computes checkpoint/config SHA-256 by streaming raw bytes in 1 MiB chunks; a null assertion is replaced by the computed digest, while a supplied lowercase 64-hex assertion must match. The dataclass stores resolved canonical external `Path` objects; `external_training_binding_to_mapping()` returns canonical strings and non-null computed digests for registry/request hashing. The two boundary roots may be existing canonical directories or lexically planned absent directories whose existing ancestors pass component-wise checks.

`reserved_paths` is the complete exact set of already-known request/task-state/receipt/marker/artifact/result/feedback/barrier paths. Validation rejects equality between external paths, overlap in either direction with either root or any reserved path, and `st_dev/st_ino` identity between either stable external file and any existing regular-file destination. Dataset contents are not traversed or hashed. The known post-validation concurrent root-swap race remains outside the trusted single-user acceptance scope.

The parameter tuple is ordered exactly as `REQUIRED_TRAINING_PARAMETER_KEYS`, so the frozen dataclass does not retain a mutable caller mapping. `public_safe_external_training_projection()` returns only the fixed schema version, `training_required`, and `training_source_kind`; it never returns a path, digest, or parameter.

- [ ] **Step 1: Write the RED schema, type, digest, and unrelated-root tests**

Create `tests/stage6/test_p6_external_training_binding.py` with fixtures that create four sibling roots: `code-root`, `output-root`, `operator-dataset`, and `operator-stable-files`. The external roots are not Git repositories and are not beneath `code-root`.

```python
def test_external_binding_accepts_unrelated_roots_and_computes_null_digests(
    tmp_path: Path,
) -> None:
    fixture = _external_binding_fixture(tmp_path, checkpoint_digest=None, config_digest=None)

    validated = validate_external_training_binding(
        fixture.raw,
        code_toolchain_root=fixture.code_root,
        local_output_root=fixture.output_root,
        reserved_paths=fixture.reserved_paths,
    )
    normalized = external_training_binding_to_mapping(validated)

    assert validated.dataset_root == fixture.dataset_root.resolve(strict=True)
    assert validated.base_checkpoint_path == fixture.checkpoint.resolve(strict=True)
    assert validated.pyramid_config_path == fixture.config.resolve(strict=True)
    assert normalized["base_checkpoint_sha256"] == _raw_sha256(fixture.checkpoint)
    assert normalized["pyramid_config_sha256"] == _raw_sha256(fixture.config)
    assert not validated.dataset_root.is_relative_to(fixture.code_root)
    assert not validated.base_checkpoint_path.is_relative_to(fixture.code_root)
```

Add exact-key/type tests for both semantic constants; null executable paths; unknown/missing keys; all nine required parameters; `epochs`/`batch_size` positive non-bool integers; `seed` nonnegative non-bool integer; `learning_rate` finite positive non-bool JSON number; and the remaining five values as nonempty strings equal to `.strip()`.

```python
@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_top_level_key", "missing_top_level_key", "training_required_false",
        "wrong_source_kind", "null_dataset", "null_checkpoint", "null_config",
        "unknown_parameter", "missing_parameter", "bool_epochs", "zero_epochs",
        "bool_seed", "negative_seed", "bool_learning_rate", "nan_learning_rate",
        "zero_learning_rate", "bool_batch_size", "zero_batch_size",
        "blank_string_parameter", "padded_string_parameter",
        "uppercase_checkpoint_digest", "short_config_digest",
        "checkpoint_digest_mismatch", "config_digest_mismatch",
    ),
)
def test_external_binding_rejects_noncanonical_schema_or_value(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _mutated_external_binding_fixture(tmp_path, mutation)
    with pytest.raises(P6ExternalTrainingBindingError) as captured:
        validate_external_training_binding(
            fixture.raw,
            code_toolchain_root=fixture.code_root,
            local_output_root=fixture.output_root,
            reserved_paths=fixture.reserved_paths,
        )
    assert captured.value.category == "history_execution_invalid"
    assert "PRIVATE-EXTERNAL-TOKEN" not in str(captured.value)
```

- [ ] **Step 2: Write the RED filesystem and alias matrix**

Parametrize missing, unreadable, wrong-type, relative, lexically noncanonical, and symlinked paths. Cover a symlink at the leaf, its immediate parent, and a higher existing ancestor for each external field. Skip unreadability cases only when the test process has effective root privileges and `os.access()` cannot observe the denied mode.

```python
@pytest.mark.parametrize(
    "mutation",
    (
        "relative_dataset", "dot_component_checkpoint", "dotdot_component_config",
        "missing_dataset", "missing_checkpoint", "missing_config",
        "dataset_is_file", "checkpoint_is_directory", "config_is_directory",
        "unreadable_dataset", "unsearchable_dataset", "unreadable_checkpoint",
        "unreadable_config", "dataset_leaf_symlink", "checkpoint_parent_symlink",
        "config_ancestor_symlink", "dataset_equals_code_root", "dataset_contains_code_root",
        "checkpoint_inside_output_root", "config_contains_output_root",
        "checkpoint_equals_reserved_file", "config_inode_aliases_existing_result",
        "checkpoint_equals_config", "unignored_dataset_in_git_repo",
        "unignored_checkpoint_in_git_repo", "unignored_config_in_git_repo",
    ),
)
def test_external_binding_rejects_unsafe_path_or_alias(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _mutated_external_binding_fixture(tmp_path, mutation)
    with pytest.raises(P6ExternalTrainingBindingError):
        validate_external_training_binding(
            fixture.raw,
            code_toolchain_root=fixture.code_root,
            local_output_root=fixture.output_root,
            reserved_paths=fixture.reserved_paths,
        )
```

Add one positive fixture in which all three external assets are beneath a temporary Git repository and matched by its ignore rules. It must pass. The three unignored variants above must fail. Inputs outside every Git repository remain valid without an ignore rule.

- [ ] **Step 3: Write the RED loader/example/privacy tests**

The tracked example bytes are exact and contain no executable value:

```yaml
schema_version: p6_external_training_binding_v1
training_required: true
training_source_kind: selected_candidate_finetune
dataset_root: null
base_checkpoint_path: null
base_checkpoint_sha256: null
pyramid_config_path: null
pyramid_config_sha256: null
training_parameters:
  training_mode: null
  epochs: null
  seed: null
  optimizer: null
  learning_rate: null
  batch_size: null
  dataset_split: null
  checkpoint_selection: null
  freeze_policy: null
```

```python
def test_public_example_is_null_only_and_not_executable(tmp_path: Path) -> None:
    example = load_external_training_binding(
        REPOSITORY_ROOT / "configs/execution/p6_external_training_binding.example.yaml"
    )
    assert example["dataset_root"] is None
    assert example["base_checkpoint_path"] is None
    assert example["pyramid_config_path"] is None
    assert all(value is None for value in example["training_parameters"].values())
    with pytest.raises(P6ExternalTrainingBindingError):
        validate_external_training_binding(
            example,
            code_toolchain_root=_mkdir(tmp_path / "code"),
            local_output_root=_mkdir(tmp_path / "output"),
            reserved_paths=(),
        )
```

Add loader rejection for a duplicate key, tracked non-example private YAML, relative loader path, symlink component, directory input, invalid UTF-8, malformed YAML, oversize input, and private-token redaction. Extend `test_anonymous_archive.py` to permit only this exact example filename and assert it has no absolute path, 64-hex digest, non-null parameter, dataset identity, checkpoint/config value, or private token.

- [ ] **Step 4: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_external_training_binding.py \
  tests/stage6/test_p6_history_training_contract.py \
  tests/integration/test_anonymous_archive.py \
  -q
```

Expected: FAIL because `p6_external_training_binding_v1.py` and its example do not exist and the historical training validator still requires containment beneath `private_root`.

- [ ] **Step 5: Implement the minimal loader and validator**

Create the module and example exactly as specified. Use `Path.lstat()` component by component from the filesystem anchor to the leaf; reject any observed symlink before `resolve(strict=True)`. Require lexical spelling `str(path) == raw`, POSIX absolute paths, `os.access(dataset, os.R_OK | os.X_OK)`, and `os.access(file, os.R_OK)`. Require `stat.S_ISDIR` for dataset and `stat.S_ISREG` for checkpoint/config. For each external asset, resolve a containing Git worktree only through `git -C <existing-path-or-parent> rev-parse --show-toplevel`; when one exists, require `git check-ignore -q -- <exact-relative-path>`. A non-repository path is valid. Keep all exception text equal to `history_execution_invalid`.

Change `validate_recipe_v2_training_template()` and `validate_recipe_v2_group_training_contract()` to consume `contract["external_training_binding"]` without requiring containment beneath `private_root`. Preserve their existing signatures for completed callers; `private_root` now acts only as the normalized code/toolchain exclusion root. Make `bind_external_training_contract()` remove the superseded flat training-input fields and set exactly one detached nested object from `external_training_binding_to_mapping()`; never merge unknown keys and never mutate either input. `external_training_binding_from_contract()` requires that one exact nested object and is the only Task 8 runtime extractor.

- [ ] **Step 6: Run GREEN, refactor, coverage, lint, and privacy gates**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. coverage run \
  --source=framework.stage6.p6_external_training_binding_v1 \
  -m pytest \
  tests/stage6/test_p6_external_training_binding.py \
  tests/stage6/test_p6_history_training_contract.py \
  -q
coverage report -m --fail-under=80
PYTHONPATH=. pytest tests/integration/test_anonymous_archive.py -q
python -m ruff check \
  framework/stage6/p6_external_training_binding_v1.py \
  framework/stage6/p6_history_training_contract_v1.py \
  tests/stage6/test_p6_external_training_binding.py \
  tests/stage6/test_p6_history_training_contract.py \
  tests/integration/test_anonymous_archive.py
git diff --check
```

Expected: tests pass; focused coverage is at least 80%; the example remains null-only; no private token reaches tracked output.

Refactor only duplicated path/digest/key checks into private helpers; keep public types/signatures and stable error text unchanged, then rerun the same commands.

- [ ] **Step 7: Review boundary and commit**

Review only the schema/loader/path/digest/redaction module, the training-validator delegation, the example, and focused tests. Reject if unknown keys pass, bool satisfies a numeric field, dataset is recursively hashed, a digest is trusted without streaming bytes, any external input is required beneath `private_root`, a symlink/noncanonical/unreadable/alias path passes, a returned object aliases caller input, or the public example becomes executable.

```bash
git add \
  framework/stage6/p6_external_training_binding_v1.py \
  framework/stage6/p6_history_training_contract_v1.py \
  configs/execution/p6_external_training_binding.example.yaml \
  tests/stage6/test_p6_external_training_binding.py \
  tests/stage6/test_p6_history_training_contract.py \
  tests/integration/test_anonymous_archive.py
git commit -m "feat: validate external P6 training assets"
```

---

### Task 8 (B): Self-Contained All-Role Closure and Runtime Binding Integration

**Files:**

- Create: `framework/stage6/p6_history_execution_closure_v1.py`
- Modify: `framework/stage6/p6_history_recipe_normalization_v1.py`
- Modify: `framework/stage6/p6_history_normalization_v1.py`
- Modify: `tools/release/derive_p6_history_recipe.py`
- Modify: `tools/release/normalize_p6_history_root.py`
- Modify: `framework/stage6/p6_runner_template_validator_v1.py`
- Modify: `framework/stage6/p6_source_wrapper_profile_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `tools/release/provision_p6_full_chain_local_config.py`
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `framework/stage6/p6_history_registry_v1.py`
- Modify: `framework/stage6/p6_history_source_materialization_v1.py`
- Modify: `framework/stage6/p6_history_measurement_v1.py`
- Modify: `tools/release/preflight_p6_materializer_training_bridge.py`
- Test: `tests/stage6/test_p6_history_execution_closure.py`
- Test: `tests/stage6/test_p6_history_normalization.py`
- Test: `tests/release/test_derive_p6_history_recipe.py`
- Test: `tests/release/test_normalize_p6_history_root.py`
- Test: `tests/stage6/test_p6_runner_template_validator.py`
- Test: `tests/stage6/test_p6_source_wrapper_profile.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`
- Test: `tests/stage6/test_p6_history_binding.py`
- Test: `tests/stage6/test_p6_history_registry.py`
- Test: `tests/stage6/test_p6_history_source_materialization_training_contract.py`
- Test: `tests/stage6/test_p6_source_reuse_measurement.py`
- Test: `tests/release/test_preflight_p6_materializer_training_bridge.py`

**Interfaces:**

- Consumes: Task 7's exact ignored external binding; recipe-v2 source map; current runner template; explicitly selected source closure; completed Tasks 3–6 projection/reuse/preflight APIs.
- Produces:

```python
EXECUTION_CLOSURE_ROLES: tuple[str, ...] = (
    "stage1_scan",
    "controller",
    "source_materializer",
    "quantization",
    "performance",
    "ap",
    "finalization",
    "activation",
)

@dataclass(frozen=True)
class P6ExecutionClosureRoot:
    closure_id: str
    source_root: Path
    destination_relative_root: Path
    sha256: str

@dataclass(frozen=True)
class P6ExecutionClosureRole:
    role: str
    closure_id: str
    entrypoint_relative_path: Path

@dataclass(frozen=True)
class P6ValidatedExecutionClosure:
    schema_version: Literal["p6_execution_code_closure_v1"]
    roots: tuple[P6ExecutionClosureRoot, ...]
    roles: tuple[P6ExecutionClosureRole, ...]

class P6ExecutionClosureError(ValueError):
    category: Literal["history_normalization_invalid"]

def validate_execution_closure_manifest(
    raw: Mapping[str, Any],
    *,
    source_history_root: Path,
    external_training: P6ExternalTrainingBinding,
) -> P6ValidatedExecutionClosure: ...

def copy_execution_closure(
    closure: P6ValidatedExecutionClosure,
    *,
    staged_private_root: Path,
) -> Mapping[str, Path]: ...

def render_normalized_runner_template(
    source_template: ValidatedRunnerTemplate,
    *,
    normalized_private_root: Path,
    copied_role_paths: Mapping[str, Path],
) -> dict[str, Any]: ...

def validate_normalized_runner_closure(
    runner_template_path: Path,
    *,
    normalized_private_root: Path,
    expected_closure: P6ValidatedExecutionClosure,
) -> ValidatedRunnerTemplate: ...
```

The ignored recipe-v2 source map gains exactly two required keys: `external_training_binding` containing the exact Task 7 object and `execution_code_closure` containing exactly `schema_version`, `roots`, and `roles`. `roots` is a nonempty list of exact objects `{closure_id, source_root, destination_relative_root, sha256}`. `roles` has exactly the eight keys in `EXECUTION_CLOSURE_ROLES`; each value is `{closure_id, entrypoint_relative_path}`. Closure ids are canonical nonempty strings; source roots are canonical absolute existing readable nonsymlink directories beneath the explicitly selected source history Git root; destination roots are unique canonical relative paths strictly beneath `execution-closure/`; entrypoints are canonical relative single-link regular executable files beneath their referenced roots. Root digests use the already approved canonical directory-tree digest rules and must match the explicit 64-lowercase-hex assertion. Shared closure ids allow several roles to reuse one copied module tree without duplicating it. This manifest is also the normalized toolchain declaration: each runnable role is bound to a digest-verified entrypoint and its complete copied module root; undeclared repository-local executables or import roots are forbidden.

The closure includes every runner role, including quantization, AP, and activation even though the historical `COMPONENT_MARKERS` map names only four roles. It includes the source materializer's implementation and import siblings; Task 2's deterministic wrapper remains the sole source marker and points to the copied implementation/cwd. No system interpreter is copied merely because it appears in argv; any interpreter is an explicit validated toolchain declaration or an approved system prerequisite, while every repository-local script/module needed by argv/import resolution is in a declared closure root.

Exact signature updates:

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
    external_training_binding: Path | None = None,
) -> dict[str, Any]: ...

def preflight_materializer_training_bridge(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
    runner_template_path: Path,
    source_wrapper_profile_path: Path,
    external_training_binding_path: Path,
) -> P6MaterializerPreflightReport: ...
```

Recipe-v2 requires both keyword arguments; v1/static keeps accepting `None` and follows its historical lane. `normalize_history_inputs()` keeps its existing Python signature and returns three additional paths for recipe-v2: `paths["runner_template"]`, `paths["source_wrapper_profile"]`, and `paths["external_training_binding"]`. Their exact normalized filenames are `runner-template.yaml`, `source-wrapper-profile.yaml`, and `external-training-binding.yaml`. The normalizer derives the wrapper profile from the copied `source_materializer` role: fixed marker destination, copied implementation entrypoint, and copied closure-root cwd. The normalized external YAML stores exact canonical values plus computed stable-file digests, but none of the dataset/checkpoint/config bytes. Provisioning loads that YAML, compares it byte-semantically with the source-contract template, and makes the normalized binding the sole runtime authority.

- [ ] **Step 1: Write RED execution-closure tests**

```python
def test_closure_copies_all_runner_roles_and_source_import_siblings(
    tmp_path: Path,
) -> None:
    fixture = _execution_closure_fixture(tmp_path)
    validated = validate_execution_closure_manifest(
        fixture.manifest,
        source_history_root=fixture.source_history_root,
        external_training=fixture.external_training,
    )
    copied = copy_execution_closure(validated, staged_private_root=fixture.staged_root)

    assert set(copied) == set(EXECUTION_CLOSURE_ROLES)
    assert (fixture.staged_root / "execution-closure/history/history_contract_validator.py").is_file()
    assert all(path.is_relative_to(fixture.staged_root) for path in copied.values())
    assert not any(
        path.is_relative_to(fixture.staged_root)
        for path in (
            fixture.external_training.dataset_root,
            fixture.external_training.base_checkpoint_path,
            fixture.external_training.pyramid_config_path,
        )
    )
```

Add rejection for missing/unknown role, missing/unknown closure root, absolute/escaping destination, role entrypoint escape, source outside the selected history root, symlink at any source component, special file, hard-linked file, digest mismatch, duplicate destination, copied tree drift, external path overlap, and a closure whose source materializer lacks an imported sibling. A subprocess black-box runs the copied source implementation through the rendered marker wrapper from the private round cwd with exactly `EXPECTED_HISTORY_ENV_KEYS` and proves the sibling import succeeds without `PYTHONPATH`.

- [ ] **Step 2: Write RED normalizer/migration tests**

Update recipe-v2 fixtures to supply `external_training_binding` and `execution_code_closure`. Add:

```python
def test_recipe_v2_normalizer_keeps_training_external_and_writes_all_role_runner(
    tmp_path: Path,
) -> None:
    fixture = _external_recipe_v2_normalization_fixture(tmp_path)
    paths = normalize_history_inputs(
        fixture.source_map,
        fixture.source_history_root,
        fixture.normalized_root,
        runner_template_path=fixture.source_runner_template,
    )

    normalized_external = yaml.safe_load(paths["external_training_binding"].read_text())
    assert normalized_external["dataset_root"] == str(fixture.dataset_root)
    assert normalized_external["base_checkpoint_sha256"] == _raw_sha256(fixture.checkpoint)
    assert normalized_external["pyramid_config_sha256"] == _raw_sha256(fixture.config)
    assert set(_runner_role_paths(paths["runner_template"])) == set(EXECUTION_CLOSURE_ROLES)
    assert not _tree_contains_inode(paths["legacy"].parents[0], fixture.checkpoint)
    assert not _tree_contains_inode(paths["legacy"].parents[0], fixture.config)
```

Migration gates:

- v1/static source maps and provisioning remain byte-compatible and do not require the new keys;
- the derive and normalize CLIs load recipe-v2 `.yaml`/`.yml` source maps with duplicate-key rejection; existing JSON source maps remain valid because JSON is a YAML subset and retain their historical compatibility tests;
- recipe-v2 source maps lacking either new key fail `history_normalization_invalid` before destination creation;
- the historical recipe-v2 shape that required training assets beneath one Git root is rejected with a stable migration error rather than silently copied;
- external roots in unrelated directories pass;
- a program default is never searched. A fixture with an exact declared default plus an explicit copied value in `external_training_binding` passes; fixtures offering only a basename, environment variable, sibling file, glob match, registry entry, or conventional directory fail before copy;
- real assets inside this repository must be ignored; repository-external assets need no Git metadata.

This increment introduces no automatic default-adoption API. The operator may copy a program's exact declared absolute value into the ignored object, after which the ordinary existence/canonical/readability/digest gates are the only authority. If the declaration cannot be referenced exactly, the binding stays incomplete and normalization stops.

- [ ] **Step 3: Write RED provisioning, registry, projection, and runtime tests**

Add a provisioning test whose normalized private root has the all-role runner/profile and whose external binding points to unrelated roots. Assert the normalized private tree contains code/module/wrapper closure but neither training asset bytes nor their inodes. Assert binding template, registry group, projected request row, and canonical hashes contain exact canonical external values and computed checkpoint/config digests. Assert public projections/stdout/stderr contain none.

```python
def test_measurement_revalidates_external_binding_before_gpu(
    tmp_path: Path,
) -> None:
    fixture = _external_measurement_fixture(tmp_path)
    fixture.checkpoint.write_bytes(b"drift-after-provision")

    with pytest.raises(P6HistoryMeasurementError) as captured:
        run_history_measurement_batch(**fixture.call_kwargs)

    assert str(captured.value) == "history_execution_invalid"
    assert fixture.gpu_probe.calls == 0
    assert fixture.runner.calls == []
```

Cover config drift, missing/unreadable/symlinked external path after provision, output overlap, and inode alias. The revalidation order is binding → projected request → external path/digest/overlap checks → exact round paths → fresh context → classification → GPU probe. The wrapper receives only the projected request and exact five-key environment; it has no argv/env fallback. A fake wrapper that ignores request fields and uses a program default is rejected by its task-state/source evidence mismatch.

- [ ] **Step 4: Run RED slices**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/stage6/test_p6_history_normalization.py \
  tests/release/test_derive_p6_history_recipe.py \
  tests/release/test_normalize_p6_history_root.py \
  tests/stage6/test_p6_runner_template_validator.py \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/stage6/test_p6_full_chain_bootstrap.py \
  tests/release/test_provision_p6_full_chain_local_config.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_history_registry.py \
  tests/stage6/test_p6_history_source_materialization_training_contract.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  -q
```

Expected: FAIL on absent closure APIs, recipe-v2 same-Git-root assumptions, source-root runner validation, missing external CLI parameters, and missing pre-GPU digest revalidation.

- [ ] **Step 5: Implement minimal closure normalization and normalized runner**

Implement the exact closure interfaces. Copy each validated closure root once with `follow_symlinks=False` into a staging root, preserve executable bits, reject unsupported entries/hard links, recompute the canonical tree digest after copy, and atomically publish the finished normalized root only after all eight roles resolve. Render a normalized runner template whose repository-local argv paths point only beneath the normalized root. Render a normalized wrapper profile whose implementation/cwd point to the copied source closure and whose marker destination remains `documented-stage5-chain/stage5_materialize_round_sources_v1.sh` beneath the normalized root.

Use one duplicate-key-rejecting safe-YAML source-map loader in both release CLIs; reject non-mapping roots and unknown keys before derivation/normalization. JSON inputs continue through the same loader. Remove recipe-v2 validation that requires external training inputs or legacy `training-data`/`model-init` source assets to share the history Git root. Keep the same-root rule for declared code/module closure roots. `_legacy_locator()` continues to populate its legacy three labels with normalized metadata paths for compatibility; none is authority for the external dataset/checkpoint/config.

- [ ] **Step 6: Implement minimal provisioning/registry/runtime integration**

Add `--external-training-binding` to provision and preflight CLIs. For recipe-v2 it is required, loaded through Task 7, checked against normalized contract values, and copied only as canonical YAML/JSON values into ignored binding/config/registry artifacts. Bind the detached normalized object before `source_contract_sha256` and `row_sha256` are computed. The nested object has the exact nine binding keys; `base_checkpoint_sha256` and `pyramid_config_sha256` are non-null after validation.

At measurement, call `external_training_binding_from_contract()` on the detached projected recipe-v2 contract, then call `validate_external_training_binding()` with `private_root`, `paths["local_output_root"]`, the exact 11 shared leaves, and every binding-resolved round destination. Compare recomputed stable-file digests with the projected non-null values before `load_fresh_run_context()`, GPU probe, activation, or runner call. Keep the existing public schemas, direct argv, private round cwd, `shell=False`, and exact five environment keys unchanged.

- [ ] **Step 7: Run GREEN, refactor, compatibility, coverage, and quality gates**

Run:

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/stage6/test_p6_history_normalization.py \
  tests/release/test_derive_p6_history_recipe.py \
  tests/release/test_normalize_p6_history_root.py \
  tests/stage6/test_p6_runner_template_validator.py \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/stage6/test_p6_full_chain_bootstrap.py \
  tests/release/test_provision_p6_full_chain_local_config.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_history_registry.py \
  tests/stage6/test_p6_history_source_materialization.py \
  tests/stage6/test_p6_history_source_materialization_training_contract.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/stage6/test_p6_history_measurement.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_p6_history_execution_adapters.py \
  -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. coverage run \
  --source=framework.stage6.p6_history_execution_closure_v1,framework.stage6.p6_external_training_binding_v1 \
  -m pytest \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/stage6/test_p6_external_training_binding.py \
  tests/stage6/test_p6_history_normalization.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  -q
coverage report -m --fail-under=80
python -m ruff check framework/stage6 tools/release tests/stage6 tests/release
python -m compileall -q framework/stage6 tools/release tests/stage6 tests/release
git diff --check
```

Expected: focused and compatibility tests pass; combined focused coverage is at least 80%; no normalized-root test finds external bytes/inodes; every public failure is redacted.

Refactor only repeated closure-copy, source-map loading, and external-contract extraction logic into the two focused modules. Do not broaden schemas or change completed context/receipt APIs; rerun the same commands after refactoring.

- [ ] **Step 8: Review boundary and commit**

Review only Task 8-owned closure/normalizer/runner/provision/registry/runtime integration. Reject if any runner role is absent, an import sibling is discovered at runtime instead of explicitly copied, a source-tree runner path survives normalization, an external training asset is copied/linked, a same-Git-root rule survives for external inputs, a digest is computed after GPU admission, a wrapper uses an unbound default, v1/static compatibility breaks, or the exact five-key/direct-argv boundary changes.

```bash
git add \
  framework/stage6/p6_history_execution_closure_v1.py \
  framework/stage6/p6_history_recipe_normalization_v1.py \
  framework/stage6/p6_history_normalization_v1.py \
  framework/stage6/p6_runner_template_validator_v1.py \
  framework/stage6/p6_source_wrapper_profile_v1.py \
  framework/stage6/p6_full_chain_bootstrap_v1.py \
  framework/stage6/p6_history_binding_v1.py \
  framework/stage6/p6_history_registry_v1.py \
  framework/stage6/p6_history_source_materialization_v1.py \
  framework/stage6/p6_history_measurement_v1.py \
  tools/release/derive_p6_history_recipe.py \
  tools/release/normalize_p6_history_root.py \
  tools/release/provision_p6_full_chain_local_config.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/stage6/test_p6_history_normalization.py \
  tests/release/test_derive_p6_history_recipe.py \
  tests/release/test_normalize_p6_history_root.py \
  tests/stage6/test_p6_runner_template_validator.py \
  tests/stage6/test_p6_source_wrapper_profile.py \
  tests/stage6/test_p6_full_chain_bootstrap.py \
  tests/release/test_provision_p6_full_chain_local_config.py \
  tests/stage6/test_p6_history_binding.py \
  tests/stage6/test_p6_history_registry.py \
  tests/stage6/test_p6_history_source_materialization_training_contract.py \
  tests/stage6/test_p6_source_reuse_measurement.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py
git commit -m "feat: bind P6 runtime to external training assets"
```

---

### Task 9 (C): Regenerate Ignored Deployment and Validate to the Mocked GPU Boundary

**Files:**

- Create: `tests/release/test_p6_external_training_deployment.py`
- Modify: `tests/release/test_preflight_p6_materializer_training_bridge.py`
- Ignored only: a new source map, external-training YAML, closure manifest, normalized runner template, wrapper profile, derivation root, normalized code/toolchain root, and local output root.
- Do not modify any attempt 1–3 root or diagnostic.

**Interfaces:**

- Consumes: reviewed Tasks 7–8, the tracked null-only example as a shape reference only, exact operator-owned external paths/values, and injected fake GPU/process seams.
- Produces: one ignored regenerated artifact set that passes derive → normalize → provision → zero-process preflight, plus a pure test proving execution stops at the injected GPU admission boundary with zero historical process launches.

- [ ] **Step 1: Write the RED mocked-boundary deployment test**

```python
def test_regenerated_external_deployment_reaches_mocked_gpu_boundary_without_process(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    binding, local_config = fixture.derive_normalize_and_provision()
    report = preflight_materializer_training_bridge(
        public_contract_path=fixture.public_contract,
        local_config_path=local_config,
        private_binding_path=binding,
        runner_template_path=fixture.normalized_runner,
        source_wrapper_profile_path=fixture.normalized_wrapper_profile,
        external_training_binding_path=fixture.normalized_external_binding,
    )

    assert report.historical_process_launch_count == 0
    assert report.gpu_probe_count == 0
    with pytest.raises(P6HistoryMeasurementError) as captured:
        run_history_measurement_batch(
            fixture.projected_request,
            fixture.binding_mapping,
            fixture.public_round_root,
            fixture.fail_if_called_runner,
            fixture.rejecting_gpu_probe,
        )
    assert captured.value.category == "history_gpu_admission_failed"
    assert fixture.rejecting_gpu_probe.calls == 1
    assert fixture.fail_if_called_runner.calls == []
```

The fixture uses unrelated synthetic external roots, an all-role closure with a real sibling import, a dynamic Stage2 plan whose candidate count is derived from fixture contents, Gold176 only as frozen fit evidence, and a recipe-v2 request with four rows. It does not launch CUDA, training, TVM, AP, latency, energy, activation, or historical programs.

Add failures proving null-only example, digest drift, missing closure role, copied training asset, preexisting output leaf, and private token all stop before the mocked GPU call.

- [ ] **Step 2: Run the RED deployment-boundary test**

Run:

```bash
PYTHONPATH=. pytest \
  tests/release/test_p6_external_training_deployment.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  -q
```

Expected on the first Task 9 test commit draft: FAIL if Task 8 omitted a pure injected GPU seam or normalized artifact path. Return any production gap to the Task 8 owner and rerun Task 8's review/commit; Task 9 remains test-and-ignored-artifact only. Do not add a validation bypass, GPU dry-run mode, synthetic success, or production process launch.

- [ ] **Step 3: Regenerate the real ignored artifacts without exposing values**

Author a new ignored recipe-v2 source map for this validation attempt. It must use the corrected exact source-map key set, embed one executable non-null `external_training_binding`, embed one `p6_execution_code_closure_v1` with all eight roles and verified root digests, and reference the exact source history root. It must contain no attempt 1–3 output, binding, normalized-root, marker, receipt, or result path. This is the only manual private-value step; values come from operator records or an exact program declaration, never a search or guess.

All variables below are required absolute paths. The shell performs only silent existence/freshness assertions before calling reviewed CLIs:

```bash
: "${P6_EXT_SOURCE_MAP_YAML:?}"
: "${P6_PUBLIC_CONTRACT_JSON:?}"
: "${P6_SOURCE_HISTORY_ROOT:?}"
: "${P6_SOURCE_RUNNER_TEMPLATE_YAML:?}"
: "${P6_PRIVATE_DERIVATION_ROOT:?}"
: "${P6_NORMALIZED_CODE_ROOT:?}"
: "${P6_VALIDATION_OUTPUT_ROOT:?}"
test -f "${P6_EXT_SOURCE_MAP_YAML}"
test -f "${P6_PUBLIC_CONTRACT_JSON}"
test -d "${P6_SOURCE_HISTORY_ROOT}"
test -f "${P6_SOURCE_RUNNER_TEMPLATE_YAML}"
test ! -e "${P6_PRIVATE_DERIVATION_ROOT}"
test ! -e "${P6_NORMALIZED_CODE_ROOT}"
test ! -e "${P6_VALIDATION_OUTPUT_ROOT}"
install -d -m 0700 "${P6_PRIVATE_DERIVATION_ROOT}"
install -d -m 0700 "${P6_VALIDATION_OUTPUT_ROOT}"
PYTHONPATH=. python tools/release/derive_p6_history_recipe.py \
  --source-map "${P6_EXT_SOURCE_MAP_YAML}" \
  --runner-template "${P6_SOURCE_RUNNER_TEMPLATE_YAML}" \
  --recipe-json "${P6_PRIVATE_DERIVATION_ROOT}/recipe.json"
PYTHONPATH=. python tools/release/normalize_p6_history_root.py \
  --source-map "${P6_EXT_SOURCE_MAP_YAML}" \
  --history-root "${P6_SOURCE_HISTORY_ROOT}" \
  --private-dir "${P6_NORMALIZED_CODE_ROOT}" \
  --runner-template "${P6_SOURCE_RUNNER_TEMPLATE_YAML}"
PYTHONPATH=. python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "${P6_NORMALIZED_CODE_ROOT}/legacy.local.yaml" \
  --runner-template "${P6_NORMALIZED_CODE_ROOT}/runner-template.yaml" \
  --local-output-root "${P6_VALIDATION_OUTPUT_ROOT}" \
  --binding-output "${P6_VALIDATION_OUTPUT_ROOT}/binding.json" \
  --config-output "${P6_VALIDATION_OUTPUT_ROOT}/local-config.json" \
  --source-wrapper-profile "${P6_NORMALIZED_CODE_ROOT}/source-wrapper-profile.yaml" \
  --external-training-binding "${P6_NORMALIZED_CODE_ROOT}/external-training-binding.yaml"
```

Expected stdout is exactly the three stable lines `p6_history_recipe_derived`, `p6_history_root_normalized`, and `p6_full_chain_config_written`; stderr is empty. The ignored binding contains exact real values and computed digests. No dataset/checkpoint/config byte or inode appears beneath either new root. No program default was inferred.

- [ ] **Step 4: Run zero-process preflight and offline full gates**

```bash
PYTHONPATH=. python tools/release/preflight_p6_materializer_training_bridge.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_VALIDATION_OUTPUT_ROOT}/local-config.json" \
  --binding "${P6_VALIDATION_OUTPUT_ROOT}/binding.json" \
  --runner-template "${P6_NORMALIZED_CODE_ROOT}/runner-template.yaml" \
  --source-wrapper-profile "${P6_NORMALIZED_CODE_ROOT}/source-wrapper-profile.yaml" \
  --external-training-binding "${P6_NORMALIZED_CODE_ROOT}/external-training-binding.yaml"
PYTHONPATH=. pytest tests/release/test_p6_external_training_deployment.py -q
PYTHONPATH=. pytest tests/release/test_p6_source_reuse_lifecycle.py -q
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-fail-under=80
python -m ruff check framework tools scripts tests
git diff --check
```

Expected preflight report has `historical_process_launch_count: 0`, `gpu_probe_count: 0`, four validated rounds, and accepted status. The mocked-boundary test has one rejected fake GPU probe and zero runner calls. Repository coverage is at least 80%.

- [ ] **Step 5: Review boundary and commit**

Review the regenerated artifact shapes locally without copying values into review output. Reject if an attempt 1–3 path changed, any real value entered Git, the normalized closure is missing a runner role/import sibling, an external byte/inode was copied, validation launched a process/GPU, a default was guessed, or preflight observed a stale destination.

```bash
git add \
  tests/release/test_p6_external_training_deployment.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py
git commit -m "test: gate external P6 training deployment"
```

---

### Task 10 (D): One Fresh Remote Attempt, Exact Verifier, and Conditional Public Docs

**Files:**

- Conditionally modify after verifier success: `docs/AAAI27_RELEASE_AUDIT.md`
- Conditionally create after verifier success: `docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md`
- Do not modify tracked files when preflight, controller, or verifier fails.

**Interfaces:**

- Consumes: completed Tasks 1–9; preserved attempt 1–3 ignored roots with controller count `0`; exact external inputs already validated in Task 9; actual H800/TVM environment.
- Produces: at most one controller process from one fresh attempt root; one verifier report; conditional public status/count documentation only.

- [ ] **Step 0: Re-run fail-closed RED cases and the 80% GREEN repository gate**

```bash
PYTHONPATH=. pytest \
  tests/stage6/test_p6_external_training_binding.py \
  tests/stage6/test_p6_history_execution_closure.py \
  tests/release/test_p6_external_training_deployment.py \
  tests/release/test_preflight_p6_materializer_training_bridge.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  -q
PYTHONPATH=. pytest -q \
  --cov=framework \
  --cov=scripts/reproduce \
  --cov-report=term-missing \
  --cov-fail-under=80
python -m ruff check framework tools scripts tests
git diff --check
git status --short
```

Expected: fail-closed tests prove invalid binding/closure/stale-root/incomplete-verifier states are rejected; the full GREEN gate passes with at least 80% coverage; tracked state is clean before remote work.

- [ ] **Step 1: Prove preserved history and fresh roots before any launch**

```bash
: "${P6_ATTEMPT1_DIAGNOSTIC_ROOT:?}"
: "${P6_ATTEMPT2_DIAGNOSTIC_ROOT:?}"
: "${P6_ATTEMPT3_DIAGNOSTIC_ROOT:?}"
: "${P6_PUBLIC_CONTRACT_JSON:?}"
: "${P6_REMOTE_SOURCE_MAP_YAML:?}"
: "${P6_REMOTE_SOURCE_HISTORY_ROOT:?}"
: "${P6_REMOTE_SOURCE_RUNNER_TEMPLATE_YAML:?}"
: "${P6_REMOTE_DERIVATION_ROOT:?}"
: "${P6_REMOTE_NORMALIZED_CODE_ROOT:?}"
: "${P6_REMOTE_OUTPUT_ROOT:?}"
: "${P6_REMOTE_CODE_REVISION_LABEL:?}"
test -d "${P6_ATTEMPT1_DIAGNOSTIC_ROOT}"
test -d "${P6_ATTEMPT2_DIAGNOSTIC_ROOT}"
test -d "${P6_ATTEMPT3_DIAGNOSTIC_ROOT}"
test -f "${P6_PUBLIC_CONTRACT_JSON}"
test ! -e "${P6_REMOTE_DERIVATION_ROOT}"
test ! -e "${P6_REMOTE_NORMALIZED_CODE_ROOT}"
test ! -e "${P6_REMOTE_OUTPUT_ROOT}"
```

The operator record must state only that attempts 1–3 are preserved and each controller launch count is `0`. Do not read any prior root as a source of run output, delete from it, patch it, or reuse it. The new three roots must differ from every attempt 1–3 root. The same validated dataset/checkpoint/config may be rebound because those are read-only inputs, not attempt output.

- [ ] **Step 2: Derive, normalize, and provision only fresh roots**

```bash
install -d -m 0700 "${P6_REMOTE_DERIVATION_ROOT}"
install -d -m 0700 "${P6_REMOTE_OUTPUT_ROOT}"
PYTHONPATH=. python tools/release/derive_p6_history_recipe.py \
  --source-map "${P6_REMOTE_SOURCE_MAP_YAML}" \
  --runner-template "${P6_REMOTE_SOURCE_RUNNER_TEMPLATE_YAML}" \
  --recipe-json "${P6_REMOTE_DERIVATION_ROOT}/recipe.json"
PYTHONPATH=. python tools/release/normalize_p6_history_root.py \
  --source-map "${P6_REMOTE_SOURCE_MAP_YAML}" \
  --history-root "${P6_REMOTE_SOURCE_HISTORY_ROOT}" \
  --private-dir "${P6_REMOTE_NORMALIZED_CODE_ROOT}" \
  --runner-template "${P6_REMOTE_SOURCE_RUNNER_TEMPLATE_YAML}"
PYTHONPATH=. python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "${P6_REMOTE_NORMALIZED_CODE_ROOT}/legacy.local.yaml" \
  --runner-template "${P6_REMOTE_NORMALIZED_CODE_ROOT}/runner-template.yaml" \
  --local-output-root "${P6_REMOTE_OUTPUT_ROOT}" \
  --binding-output "${P6_REMOTE_OUTPUT_ROOT}/binding.json" \
  --config-output "${P6_REMOTE_OUTPUT_ROOT}/local-config.json" \
  --source-wrapper-profile "${P6_REMOTE_NORMALIZED_CODE_ROOT}/source-wrapper-profile.yaml" \
  --external-training-binding "${P6_REMOTE_NORMALIZED_CODE_ROOT}/external-training-binding.yaml"
```

Expected: complete code/module/wrapper closure is self-contained; external values remain references with computed digests; all 11 outputs are still absent beneath the fresh output root; tracked state is unchanged.

- [ ] **Step 3: Run the zero-process preflight**

```bash
PYTHONPATH=. python tools/release/preflight_p6_materializer_training_bridge.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_REMOTE_OUTPUT_ROOT}/local-config.json" \
  --binding "${P6_REMOTE_OUTPUT_ROOT}/binding.json" \
  --runner-template "${P6_REMOTE_NORMALIZED_CODE_ROOT}/runner-template.yaml" \
  --source-wrapper-profile "${P6_REMOTE_NORMALIZED_CODE_ROOT}/source-wrapper-profile.yaml" \
  --external-training-binding "${P6_REMOTE_NORMALIZED_CODE_ROOT}/external-training-binding.yaml"
```

Expected public report is exactly the approved preflight schema with `status: accepted`, `validated_round_count: 4`, `training_required: true`, `historical_process_launch_count: 0`, and `gpu_probe_count: 0`. It validates external inputs in place and exact absence of Stage1, plan, registry, state, four public rounds, four binding-resolved private round layouts, metadata/context/receipt namespace, `materialized`, and every derived 11-path leaf. Stop without a controller launch if it fails.

- [ ] **Step 4: Launch exactly one controller process**

Run the following block once. Do not rerun it in the same root, even when it exits nonzero:

```bash
p6_controller_launch_count=0
p6_controller_launch_count=$((p6_controller_launch_count + 1))
test "${p6_controller_launch_count}" -eq 1
PYTHONPATH=. python tools/release/run_p6_h800_search.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_REMOTE_OUTPUT_ROOT}/local-config.json" \
  --code-revision "${P6_REMOTE_CODE_REVISION_LABEL}"
test "${p6_controller_launch_count}" -eq 1
```

Acceptance remains dynamic Stage1 → dynamic Stage2 candidate count → Gold176 cold start → four rounds of four selected rows. Candidate count is whatever the fresh valid plan contains, never a static 126/343/686 constant. Gold176 fit/refit counts remain 176/180/184/188 and Gold176 never enters measurement. The controller creates one context before round 0; first-use groups train/fine-tune from the exact bound values and write all 11 shared paths beneath the fresh output root; later same-group q-modes reuse only a valid receipt; every selected row runs quantization, TVM/performance, AP, and finalization. Exactly 16 unique row ids complete across 4×4; distinct group/receipt count remains dynamic.

- [ ] **Step 5: Run the exact verifier once**

```bash
PYTHONPATH=. python tools/release/verify_p6_materializer_training_run.py \
  --contract "${P6_PUBLIC_CONTRACT_JSON}" \
  --local-config "${P6_REMOTE_OUTPUT_ROOT}/local-config.json" \
  --binding "${P6_REMOTE_OUTPUT_ROOT}/binding.json"
```

Expected public output only:

```json
{
  "completed_rounds": 4,
  "gold176_remeasured_rows": 0,
  "schema_version": "p6_materializer_training_bridge_completion_v1",
  "selected_rows": 16,
  "status": "completed"
}
```

The verifier resolves exact paths without search and revalidates the external stable-file digests, one context, four requests, every current-run group receipt, every producer request/row, all artifact/marker digests, task-state/result/feedback/barrier, five finite metrics per row, 16 unique selected rows, and zero Gold176 overlap. Multiple rows may share a receipt; a producer may be in the same or an earlier round, never later.

- [ ] **Step 6: Stop safely on any failure**

Preserve the complete fresh ignored attempt. Do not repair/delete a context, receipt, marker, artifact, request, task-state, feedback, result, or barrier; do not synthesize feedback, substitute a candidate, switch to static registry, copy an external asset, or start a second controller in the same root. Do not update public docs. A later authorized attempt begins from another fresh derivation/code/output triple.

- [ ] **Step 7: Conditionally update public docs and run privacy gates**

Only after Step 5 succeeds, record these stable facts: external binding validated in place; normalized execution closure covered all runner roles; dynamic Pyramid/H800/TVM Stage1/Stage2 fed four rounds; Gold176 was cold-start evidence only; 16 unique rows completed; every row mapped to validated current-run source evidence before q-specific downstream stages. Do not disclose real paths/digests/parameters, group/row/q-mode identities, receipt count/mapping, producer, nonce, mtime, metrics, artifact details, GPU identity, host, logs, argv, or environment.

```bash
PYTHONPATH=. pytest tests/integration/test_anonymous_archive.py tests/release -q
python -m ruff check framework tools scripts tests
git diff --check
rg -n '/home/|/mnt/|GPU-|CUDA_VISIBLE_DEVICES=|Traceback|PRIVATE-EXTERNAL-TOKEN' \
  docs/AAAI27_RELEASE_AUDIT.md \
  docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md
```

Expected: tests/lint/diff pass and the `rg` privacy scan prints no matches.

- [ ] **Step 8: Review boundary and conditional commit**

Review the one controller-launch record, verifier output, and conditional public diff. Reject closure based on attempt 1–3 evidence, more than one controller launch, reused output root, copied external input, guessed default, static candidate count, Gold176 measurement, global group uniqueness, per-q retraining, bare markers, 16-receipt assumption, missing downstream row, or private disclosure.

```bash
git add \
  docs/AAAI27_RELEASE_AUDIT.md \
  docs/release-manifests/P6_V2_MATERIALIZER_TRAINING_BRIDGE.md
git commit -m "docs: record external P6 training completion"
```

Commit only when the verifier succeeded and those docs changed. Otherwise make no Task 10 commit.

---

## Task Ordering, Sequential Dependencies, and Ownership Conflicts

| Order | Owner | Depends on | Fresh review boundary | Shared-file rule |
| --- | --- | --- | --- | --- |
| Completed Task 1 | Training contract | None | Accepted in ledger | Historical content preserved |
| Completed Task 2 | Wrapper profile/provisioning | Task 1 | Accepted in ledger | Historical content preserved |
| Completed Task 3 | Projection/hash gates | Tasks 1–2 | Accepted in ledger | Historical content preserved |
| Completed Task 4 | Evidence schemas, resolved-root paths/hashes, digests, classification, receipt publication, measurement routing | Task 3 | Accepted in ledger | Historical content preserved; Task 8 adds only pre-GPU external revalidation routing |
| Completed Task 5 | Controller context creation, exact path planner/exported resolver, zero-process preflight, completion verifier, relaunch rejection | Task 4 | Accepted in ledger | Historical content preserved; Task 8 extends preflight inputs without changing context semantics |
| Completed Task 6 | Offline repeated-group lifecycle | Tasks 4–5 | Accepted in ledger | Historical content preserved; remains the 4×4/Gold176/reuse regression |
| Task 7 (A) | Exact external binding schema, loader, paths, digests, redaction, null-only example | Completed Tasks 1–6 | Fresh schema/security reviewer | Owns new binding module/example and training-validator delegation only |
| Task 8 (B) | All-role code/module/wrapper closure; normalize/provision/registry/request/runtime/preflight integration | Task 7 | Fresh implementation + compatibility reviewer | Owns closure module and integration; serialized because shared normalization/provision/runtime files change together |
| Task 9 (C) | Ignored artifact regeneration and mocked GPU-boundary validation | Task 8 | Fresh offline deployment reviewer | Test additions plus ignored artifacts only; no real controller |
| Task 10 (D) | Fresh remote preflight, exactly one controller, verifier, conditional docs | Task 9 and all full gates | Fresh real-run evidence/docs reviewer | No tracked code; docs only after verifier success |

Tasks 1–6 are immutable historical records. Execute Tasks 7–10 sequentially: Task 8 consumes Task 7's exact type and Task 9 consumes Task 8's normalized artifact names; Task 10 is forbidden until Task 9 proves zero process/GPU work. Task 9 must not patch production—a discovered production gap returns to Task 8—and Task 10 must not change tracked code. Existing files over 800 lines receive routing or fixture migration only; both new focused production modules and both new focused test modules start below 800 lines and must remain there.

## Acceptance Matrix

| Scenario | Classification | Source invocation | Downstream q-specific stages | Acceptance |
| --- | --- | ---: | ---: | --- |
| External dataset and stable files live in unrelated canonical readable roots; optional digests match or are null | Valid external binding | First-use groups only | Every selected row | Compute both stable-file digests and proceed |
| Null-only tracked example used for execution | Invalid external binding | Zero | Zero | Redacted `history_execution_invalid` |
| Exact program declaration is copied into ignored binding and validates | Valid external binding | First-use groups only | Every selected row | Proceed; runtime authority is the ignored binding |
| Basename/cwd/sibling/env/glob/registry default inference would be needed | Invalid external binding | Zero | Zero | Stop without guessing |
| External input copied/linked into normalized/output/result namespace | Invalid binding/layout | Zero | Zero | Redacted stop |
| Any of eight runner roles or a source import sibling is absent from closure | Invalid deployment | Zero | Zero | Redacted stop before provisioning/preflight |
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
| Attempts 1–3 preserved with controller count 0; attempt 4 uses fresh roots | Fresh-attempt prerequisite | Exactly one controller total in Task 10 | 16 rows if verifier passes | Prior attempts remain non-evidence |

## Failure Category and Public-Privacy Matrix

| Internal condition | Private stable category | Public category/output |
| --- | --- | --- |
| Request/row/q-mode/request hash drift | Existing request validator | `history_request_invalid` |
| Partial source evidence | `p6_source_reuse_partial` | `history_execution_invalid` only |
| Cross-run/root/task/revision/plan/registry evidence | `p6_source_reuse_stale` | `history_execution_invalid` only |
| Receipt/artifact/marker/producer/type/link/path mismatch | `p6_source_reuse_mismatch` | `history_execution_invalid` only |
| Invalid binding/context/source contract/projection | Existing stable validator | `history_execution_invalid` |
| External binding shape/type/path/readability/digest/overlap failure | `history_execution_invalid` | `history_execution_invalid` only |
| Code/module/wrapper closure missing, drifting, or containing a training asset | `history_normalization_invalid` | Stable normalization/provision/preflight failure only |
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
- **One private Git root for everything:** rejected because dataset/checkpoint/config are operator-owned read-only bindings independent of the normalized execution closure.
- **Training-asset normalization:** rejected because copying, hard-linking, vendoring, or rewriting external assets destroys the approved ownership boundary.
- **Default discovery:** rejected because basename/cwd/sibling/env/glob/registry inference is not exact program-declaration evidence.

## Spec Requirement Mapping

| Spec requirement | Task coverage |
| --- | --- |
| Fix deterministic private import failure without widening public env | Historical Task 2 plus Task 8 explicit all-role/module closure and black-box normalized import test |
| Exact `p6_external_training_binding_v1` schema and null-only public example | Task 7 loader/schema/example/privacy tests |
| Independent canonical/readable/no-symlink external paths | Task 7 path matrix; Task 8 pre-GPU revalidation; Task 9 private preflight |
| Optional author assertions become required computed checkpoint/config digests | Task 7 streaming digest normalization; Task 8 registry/request/runtime propagation |
| No dataset tree hashing; exact `dataset_split` only | Task 7 schema/digest tests |
| No same-Git-root requirement or training-asset copy | Task 8 migration/normalization tests |
| Self-contained code/module/wrapper/toolchain closure includes all runner roles | Task 8 closure schema, post-copy digest, normalized runner, and black-box import tests |
| Exact program declaration only; no default guessing | Task 8 migration tests; Tasks 9–10 operator gates |
| Enforce `training_required: true` and exact Pyramid training fields | Historical Task 1 superseded by Task 7 binding and Task 8 integration |
| Preserve external fields and computed digests through projection/hashes | Historical Task 3 plus Task 8 registry/projection tests |
| Reject incomplete recipe-v2 source contracts | Task 7 exact binding; Task 8 registry/provision/preflight integration |
| Render one shared source bundle per Pyramid group | Task 1 registry integration; Task 3 projection drift tests; Task 4 first-use receipt; Task 6 lifecycle |
| Keep all 11 generated leaves under the fresh output root and external assets outside | Task 7 reserved-path validation; Task 8 normalization/runtime alias tests; Tasks 9–10 preflight |
| Remove untrusted output aliases before canonical group contract hash | Task 1 registry integration; Task 3 projection tests |
| Immutable run context bound to task/revision/root/plan/registry/nonce | Task 4 schemas/persistence; Task 5 controller integration |
| Deterministic exact paths and no search | Tasks 4–5 path APIs/static scans |
| Lexical/resolved/symlink/type/hard-link safety | Task 4 path/digest matrix; Task 5 planned/runtime roots |
| Adapter-only exclusive receipt publication | Task 4 publication/wrapper-receipt tests; Task 6 real adapter fake |
| Current-run first-use/reuse classification | Task 4 unit/integration; Task 6 full lifecycle |
| Public/private boundary and no private leakage | Historical Tasks 1–6 plus Tasks 7–10 privacy gates |
| Source materializer direct argv once per first-use group | Tasks 2, 3, 4, 6 |
| Gate all downstream stages on validated current-run evidence | Tasks 4 and 6 |
| Same-round mixed q-mode and later-round reuse | Task 4 integration and Task 6 end-to-end |
| Zero-process preflight before Stage1/activation/GPU/history execution | Historical Task 5 extended by Task 8; Task 9 mocked boundary; Task 10 invocation order |
| Zero-GPU black-box dynamic lifecycle | Task 6 |
| Fresh Stage1/Stage2 through four rounds | Historical Task 6 fake lifecycle; Task 10 real H800 run |
| No Gold176 remeasurement | Historical Task 6 and Task 10 |
| Completion maps 16 rows to a dynamic number of receipts | Historical Task 5 verifier; historical Task 6 lifecycle; Task 10 real verifier |
| No in-place partial resume | Historical Tasks 4–5 and Tasks 9–10 |
| Preserve v1/static compatibility | Historical Tasks 1, 3, 5 plus Task 8 migration regression commands |
| Attempts 1–3 preserved/controller count 0; new roots for next run | Revision status, Task 9 ignored regeneration, Task 10 fresh-root gate |
| Trusted single-user scope and nonblocking root-swap limitation | Global constraints; Task 7 path semantics; Tasks 8–10 review boundaries |
| Final docs conditional on real completion | Task 10 only |

---

## Final Full Gates Before Task 10 or Merge Handoff

Acceptance proceeds in this order: (1) exact external schema/digests,
(2) self-contained all-role closure and runtime integration, (3) ignored
deployment regeneration plus mocked GPU boundary, then (4) one fresh real
Stage1→dynamic Stage2→Gold176→4×4 H800/TVM run and reproducible public
docs/config/results comparison. The Stage2 candidate count and the number of
distinct source receipts are observed dynamic values, never acceptance
constants. Existing-path canonical/no-symlink validation and private-data
redaction remain required. Adversarial TOCTOU/directory-root symlink-swap
resistance is the documented trusted-environment limitation, not a final gate.

Run from a clean tracked worktree after Task 9:

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
  framework/stage6/p6_external_training_binding_v1.py \
  framework/stage6/p6_source_reuse_evidence_v1.py \
  tools/release/preflight_p6_materializer_training_bridge.py \
  tools/release/verify_p6_materializer_training_run.py
python - <<'PY'
from pathlib import Path

limits = [
    Path("framework/stage6/p6_external_training_binding_v1.py"),
    Path("framework/stage6/p6_history_execution_closure_v1.py"),
    Path("tests/stage6/test_p6_external_training_binding.py"),
    Path("tests/stage6/test_p6_history_execution_closure.py"),
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
    Path("tests/release/test_p6_external_training_deployment.py"),
]
for path in limits:
    count = len(path.read_text(encoding="utf-8").splitlines())
    assert count < 800, f"{path} has {count} lines"
PY
```

Expected: all pytest/lint/compile/diff/privacy gates pass, coverage is at least 80%, the no-search `rg` has no matches, every focused file is below 800 lines, the tracked example is null-only, and ignored validation reports controller/process/GPU counts of zero.

## Plan Self-Review

- [x] Spec coverage: every external-binding key/type/path/digest/overlap rule maps to Task 7; normalization, all-role closure, migration, projection, pre-GPU revalidation, and compatibility map to Task 8; ignored regeneration and pure validation map to Task 9; fresh-root, one-controller, verifier, and conditional-doc rules map to Task 10. Historical source-reuse/context/4×4/Gold176 invariants remain mapped to completed Tasks 3–6.
- [x] Placeholder scan: the new executable tasks contain no unresolved implementation marker, deferred-fill instruction, guessed private value, ellipsis inside a concrete command, or unnamed file/function. Intentional `...` appears only in Python signature declarations, and nulls appear only in the exact tracked documentation example. Required operator paths are named environment variables guarded by `${VAR:?}` and are deliberately not real public values. Historical Tasks 1–6 remain byte-preserved and are explicitly superseded where contradictory.
- [x] Red-flag scan: literal unfinished-work tokens, private absolute paths, and hard-coded candidate/receipt counts are checked before the planning commit. The only fixed counts are approved protocol values: Gold176, four rounds, four rows, 16 selected rows, nine artifacts, two markers, and 11 shared paths.
- [x] Type consistency: Task 7 alone defines `P6ExternalTrainingBinding`; Task 8 imports that exact type in `validate_execution_closure_manifest()` and converts it only through `external_training_binding_to_mapping()`. Both `materialize_full_chain_binding()` and `preflight_materializer_training_bridge()` use the exact new keyword names shown in Task 8 and every Task 9–10 command uses the matching CLI flag `--external-training-binding`.
- [x] Existing-interface consistency: Task 4 remains owner of `P6FreshRunContext`, `P6GroupSourceReceipt`, `P6GroupReuseDecision`, path/context/classification/publication APIs, and Task 5 remains owner of preflight/verifier context semantics. Task 8 inserts external validation before those APIs without renaming them. `P6SourceReusePaths.local_output_root` and binding-owned `EXPECTED_HISTORY_ENV_KEYS` remain unchanged.
- [x] Sequential consistency: Tasks 1–6 are historical. Task 7 produces a detached computed-digest binding; Task 8 consumes it to build normalized closure/binding/registry/request state; Task 9 consumes Task 8's exact normalized artifact names without a real launch; Task 10 consumes only a green Task 9 deployment protocol and may launch one controller from fresh roots.
- [x] File-size check: new binding/closure modules and focused tests are split up front and gated below 800 lines; existing over-limit normalizer/measurement/controller/test modules receive validation/routing changes only.
- [x] Security/privacy check: the tracked example is null-only; real external values stay in ignored artifacts; no training bytes are normalized; public errors remain stable categories; docs/verifier output omit private paths, digests, parameters, metrics, identities, logs, argv, environment, and receipt mappings.
- [x] Reuse/search consistency: candidate count and receipt count remain dynamic; q-mode remains a row dimension; 4×4 downstream work and Gold176 cold-start semantics are unchanged; no task introduces global group uniqueness, per-q retraining, bare-marker reuse, static fallback, or dataset traversal.
- [x] Fresh-run consistency: attempts 1–3 are preserved with controller count 0. Task 9 uses new validation roots and zero real processes; Task 10 uses another fresh derivation/code/output triple, zero-process preflight, exactly one controller process, one verifier, and conditional docs only after 4/16/0 completion.

### Task interface matrix

| Producer | Exact output | Consumer | Consistency gate |
| --- | --- | --- | --- |
| Task 7 | `P6ExternalTrainingBinding` with canonical `Path` fields and two computed digests | Task 8 closure, provisioning, registry, runtime | Conversion only through `external_training_binding_to_mapping()`; exact keys and detached copies |
| Task 7 | `bind_external_training_contract(contract, binding) -> dict[str, Any]` | Task 8 binding/registry/request hashes | Called before source-contract, row, request, plan/registry-dependent context identities |
| Task 8 | `P6ValidatedExecutionClosure` with exactly eight roles | Task 8 normalizer/runner/profile and Task 9 deployment fixture | Source and post-copy tree digest equality; all normalized role paths beneath one code root |
| Task 8 | `paths["runner_template"]`, `paths["source_wrapper_profile"]`, `paths["external_training_binding"]` from `normalize_history_inputs()` | Tasks 9–10 provision/preflight commands | Exact normalized filenames under one code root; generated profile binds the copied source role |
| Task 8 | `materialize_full_chain_binding(..., source_wrapper_profile=..., external_training_binding=...)` | Provision CLI | Recipe-v2 requires both; v1/static accepts neither |
| Task 8 | `preflight_materializer_training_bridge(..., external_training_binding_path=...)` | Tasks 9–10 | Pure validation report retains controller/history/GPU counts at zero |
| Completed Tasks 4–5 | Context/receipt/path/classification/verifier APIs | Task 8 pre-GPU revalidation and Task 10 verifier | No renamed type/signature; external checks occur before context/GPU/process |
| Task 9 | Green ignored derive/normalize/provision/preflight protocol and mocked-GPU test | Task 10 | No real controller; new remote roots still required |
| Task 10 controller | One fresh context, four requests, 16 row results, dynamic group receipts | Task 10 verifier | Exactly one controller process; verifier resolves, never searches |
