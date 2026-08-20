# P6 程序式 Recipe Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从已验证的私有 Stage5 runner template 与 fresh private Git root 静态派生 `p6_history_dynamic_materialization_recipe_v1`，并让现有 normalizer/provision/Stage1/Stage2/registry-v2 离线链路在 measurement 前 fail-closed 验证通过。

**Architecture:** 抽出一个共享 pre-provision runner-template validator，bootstrap 与 recipe bridge 都消费同一份 canonical interface，避免桥接器依赖 provision 后才生成的 binding。新增版本化 profile registry、interface verifier 与 pure recipe renderer；source-map v2 在 explicit recipe 与 procedural profile 两路之间二选一，并把派生 recipe 送入既有 normalizer 与后续 post-provision canonical equality gate。

**Tech Stack:** Python 3.10+、PyYAML、标准库 JSON/pathlib/subprocess/tempfile/dataclasses、pytest、ruff。

**Spec:** `docs/superpowers/specs/2026-08-20-p6-procedural-recipe-bridge-design.md`

## Global Constraints

- `HEAD` baseline is `75771a3`; do not revert unrelated commits or overwrite user work.
- Create and commit only the implementation files listed inside each task; private H800 retry in Task 6 must not create tracked changes or a commit.
- No H800 access, SSH, remote execution, download, install, rebuild, TVM compilation, training, export, AP, latency, energy, or measurement is authorized until Task 6 explicitly reaches the private retry step after all offline gates pass.
- Public tracked files may contain profile names, profile schema versions, component role names, marker basenames, semantic argv names, placeholder names, capability names, recipe schema names, identity template rules, relative template rules, stable failure categories, and synthetic fixture values.
- Public tracked files must not contain private absolute paths, user directories, remote paths, real GPU index/UUID, host names, training data, model assets, source asset IDs, candidate IDs, checkpoint filenames, ONNX filenames, calibration data names, AP, latency, energy, objective, cost-model results, historical results, terminal status, logs, private source-map, binding, local YAML, Stage1 manifest, Stage2 plan, registry-v2, request, feedback, or normalized root.
- v1 `p6_history_normalization_source_v1` remains explicit-recipe only; v1 behavior must not require a profile and must not enter procedural inference.
- v2 `p6_history_normalization_source_v2` must enforce mutual exclusivity: `recipe_mode=explicit_dynamic_recipe` requires only `dynamic_materialization_recipe`; `recipe_mode=procedural_profile` requires only `procedural_recipe_source` and `procedural_recipe_profile`.
- All procedural derivation failures use stable category `history_recipe_derivation_invalid` and write no partial recipe, no source-map update, no normalized root, no YAML, no registry, and start no measurement.
- The bridge must consume only the ignored private source-map v2, validated pre-provision private runner template, fresh private Git root, selected procedural component refs, and public profile; it must not read the future binding as input.
- Actual argv, component roles, marker basenames, placeholders, output-layout fields, and capabilities must be read from the validated runner template; source-map v2 must not self-report semantic argv, capability, or output templates.
- Profile output templates describe the new P6 private output-root layout, not historical absolute paths.
- Use existing Pyramid canonical identity, source contract hash validation, Stage2 plan/registry identity gate, and contract validation only; do not add an asset-level hash ledger, directory Merkle tree, or per-checkpoint/ONNX SHA ledger.
- Candidate count is produced by the Stage2 dynamic plan; tests may assert `>= 16` and identity equality, but must not hard-code 343 or 686 as expected dynamic counts.
- Real four-round H800 execution is not an acceptance condition for this plan; it remains a later separately authorized stage.

---

## File responsibility map

- `framework/stage6/p6_runner_template_validator_v1.py` owns shared private runner-template loading, Git-ignore/root/symlink gates, role selection, executable argv rendering, marker basename extraction, and a frozen canonical `RunnerTemplateValidation` object. It is pre-provision only and imports neither registry nor binding creation.
- `framework/stage6/p6_full_chain_bootstrap_v1.py` keeps full-chain binding/provision responsibilities and delegates template parsing/path validation to `p6_runner_template_validator_v1.py`; its behavior must be byte-for-byte equivalent at public interfaces except for internal error detail text.
- `framework/stage6/p6_history_recipe_profiles_v1.py` owns public allowlisted profile data and profile schema validation. It contains no private paths, GPU identifiers, asset IDs, candidate IDs, metrics, or results.
- `framework/stage6/p6_history_recipe_bridge_v1.py` owns procedural source-map v2 profile selection validation, interface verification against the chosen profile, pure recipe rendering, output-template collision checks, and atomic private recipe JSON writing.
- `framework/stage6/p6_history_normalization_v1.py` owns v1/v2 source-map parsing and calls the bridge only when v2 is in `procedural_profile` mode. It preserves existing registry-v1/legacy locator output shape.
- `framework/stage6/p6_history_binding_v1.py` owns post-provision binding validation and must reject a binding whose `source_contract_template.dynamic_materialization_recipe` is not canonically equal to the pre-provision derived recipe carried through normalization.
- `tools/release/derive_p6_history_recipe.py` is the private-only CLI for derivation without normalization.
- `tools/release/normalize_p6_history_root.py` remains the normalizer CLI and gains v2 source-map support while keeping the existing v1 arguments compatible.
- `tools/release/provision_p6_full_chain_local_config.py` stays the provision CLI; if it receives a normalized root with a procedural derivation record, provision must enforce binding recipe consistency before writing the pair.
- `tests/stage6/test_p6_runner_template_validator.py` covers the shared validator and bootstrap regression seams.
- `tests/stage6/test_p6_history_recipe_bridge.py` covers profiles, interface verifier, renderer, collision/privacy gates, and atomic no-partial writes.
- `tests/release/test_derive_p6_history_recipe.py` covers CLI-only failure redaction, absolute ignored path gates, and atomic output behavior.
- `tests/stage6/test_p6_history_normalization.py` and `tests/release/test_normalize_p6_history_root.py` cover v1 compatibility, v2 explicit mode, v2 procedural mode, mutual exclusivity, source-contract consistency, and no normalized-root failures.
- `tests/stage6/test_p6_history_binding.py`, `tests/stage6/test_p6_full_chain_bootstrap.py`, and `tests/release/test_provision_p6_full_chain_local_config.py` cover post-provision recipe equality and bootstrap behavior after validator extraction.
- `tests/stage6/test_coptv2x_h800_search.py`, `tests/release/test_run_p6_h800_search.py`, and `tests/release/test_p6_history_execution_adapters.py` cover offline lifecycle derivation through registry-v2 with zero measurement.
- `docs/AAAI27_RELEASE_AUDIT.md` and this plan are the only public docs touched by implementation tasks; no public source-map, binding, local YAML, registry, manifest, plan, request, feedback, logs, metrics, or private output roots are added.

### Task 1: Shared pre-provision runner-template validator

**Files:**

- Create: `framework/stage6/p6_runner_template_validator_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Test: `tests/stage6/test_p6_runner_template_validator.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True)
class RunnerTemplateComponent:
    role: str
    marker_basename: str
    executable_path: Path
    argv: tuple[str, ...]
    required_placeholders: frozenset[str]
    template_node_ref: str

@dataclass(frozen=True)
class RunnerTemplateValidation:
    schema_version: str
    history_root: Path
    stage1_scan: Mapping[str, Any]
    execution_interface: Mapping[str, Any]
    components_by_role: Mapping[str, RunnerTemplateComponent]
    output_layout: Mapping[str, Any]
    environment: Mapping[str, Any]

def validate_pre_provision_runner_template(
    *,
    runner_template_path: Path,
    history_root: Path,
    expected_roles: Iterable[str],
    expected_marker_basenames: Mapping[str, str],
    role_selection: Mapping[str, str] | None = None,
    error_category: str = "execution_interface_unavailable",
) -> RunnerTemplateValidation
```

- Consumes from existing bootstrap: private `p6_history_runner_template_v1`, `history_root`, `COMPONENT_MARKERS`, `EXECUTION_INTERFACE_KEYS`, `STAGE1_PLACEHOLDERS`, and shell-token/path gates.
- Preserves for bootstrap: `materialize_full_chain_binding(legacy_local_config: Path, runner_template: Path, local_output_root: Path, binding_output: Path, config_output: Path, gpu_probe: GpuProbe) -> dict[str, Any]` and CLI stderr categories `execution_interface_unavailable`, `history_root_ambiguous`, `legacy_locator_invalid`, `private_destination`, `gpu_unavailable`, and `bootstrap_invalid`.
- Consumed by Task 2: `RunnerTemplateValidation.components_by_role`, `execution_interface`, `output_layout`, and `environment` without using binding.

- [ ] **Step 1: Write the validator acceptance test**

```python
def test_pre_provision_validator_returns_canonical_components_without_binding(tmp_path: Path) -> None:
    history_root = _private_history_git_root(tmp_path)
    template_path = _write_runner_template(history_root / "ignored" / "runner.yaml")
    validated = validate_pre_provision_runner_template(
        runner_template_path=template_path,
        history_root=history_root,
        expected_roles=("controller", "source_materializer", "performance_plan", "finalizer"),
        expected_marker_basenames={
            "controller": "stage5_task_round_controller_v3.sh",
            "source_materializer": "stage5_materialize_round_sources_v1.sh",
            "performance_plan": "stage5_build_performance_plan_v2.py",
            "finalizer": "stage5_finalize_feedback_v2.py",
        },
    )
    assert validated.schema_version == "p6_history_runner_template_v1"
    assert validated.history_root == history_root
    assert set(validated.components_by_role) == {
        "controller", "source_materializer", "performance_plan", "finalizer"
    }
    assert validated.components_by_role["source_materializer"].marker_basename == (
        "stage5_materialize_round_sources_v1.sh"
    )
    assert "{measurement_request}" in validated.components_by_role[
        "source_materializer"
    ].required_placeholders
```

- [ ] **Step 2: Write validator fail-closed tests for duplicate, extra, symlink, and out-of-root component refs**

```python
@pytest.mark.parametrize(
    "mutator",
    [
        lambda template, root, tmp: template["execution_interface"]["execution_chain"].append(
            copy.deepcopy(template["execution_interface"]["execution_chain"][0])
        ),
        lambda template, root, tmp: template["execution_interface"]["execution_chain"][0]["argv"].__setitem__(
            0, str(tmp / "outside" / "stage5_materialize_round_sources_v1.sh")
        ),
        lambda template, root, tmp: template["execution_interface"]["controller"]["argv"].__setitem__(
            0, "documented-stage5-chain/wrong_controller.sh"
        ),
    ],
)
def test_pre_provision_validator_rejects_unsafe_or_drifting_template(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any], Path, Path], None],
) -> None:
    history_root = _private_history_git_root(tmp_path)
    payload = _runner_template_payload()
    mutator(payload, history_root, tmp_path)
    template_path = _write_yaml(history_root / "ignored" / "runner.yaml", payload)
    with pytest.raises(RunnerTemplateValidationError, match=r"^execution_interface_unavailable:"):
        validate_pre_provision_runner_template(
            runner_template_path=template_path,
            history_root=history_root,
            expected_roles=("controller", "source_materializer", "performance_plan", "finalizer"),
            expected_marker_basenames=EXPECTED_MARKERS,
        )
```

- [ ] **Step 3: Run RED for the new validator tests**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_runner_template_validator.py -q`

Expected: FAIL because `framework.stage6.p6_runner_template_validator_v1` does not exist.

- [ ] **Step 4: Implement dataclasses and shared privacy/path helpers**

```python
class RunnerTemplateValidationError(ValueError):
    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")

def _template_component_for_ref(
    interface: Mapping[str, Any], role: str, template_node_ref: str
) -> RunnerTemplateComponent:
    node = _resolve_unique_interface_node(interface, template_node_ref)
    rendered_argv = _render_executable_argv(node["argv"], history_root)
    return RunnerTemplateComponent(
        role=role,
        marker_basename=Path(rendered_argv[0]).name,
        executable_path=Path(rendered_argv[0]),
        argv=tuple(rendered_argv),
        required_placeholders=frozenset(node.get("required_placeholders", [])),
        template_node_ref=template_node_ref,
    )
```

Copy the bootstrap gates into the new module first, then delete the duplicate bootstrap-only copies only after bootstrap tests are green.

- [ ] **Step 5: Refactor bootstrap to delegate parsing and path validation**

```python
validated = validate_pre_provision_runner_template(
    runner_template_path=runner_template,
    history_root=root,
    expected_roles=COMPONENT_MARKERS.keys(),
    expected_marker_basenames={
        role: marker for role, (marker, _version) in COMPONENT_MARKERS.items()
    },
)
component_paths = {
    role: str(component.executable_path)
    for role, component in validated.components_by_role.items()
}
interface = copy.deepcopy(dict(validated.execution_interface))
```

Keep `_load_legacy_local_locator`, `_unique_common_history_root`, `_render_full_chain_local_config`, GPU probing, binding write, and CLI behavior in `p6_full_chain_bootstrap_v1.py`.

- [ ] **Step 6: Run bootstrap regression GREEN**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_runner_template_validator.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py -q`

Expected: PASS, including private ignored runner-template tests and archived duplicate marker tests.

- [ ] **Step 7: Run lint and diff checks**

Run: `python -m ruff check framework/stage6/p6_runner_template_validator_v1.py framework/stage6/p6_full_chain_bootstrap_v1.py tests/stage6/test_p6_runner_template_validator.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py && git diff --check`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add framework/stage6/p6_runner_template_validator_v1.py framework/stage6/p6_full_chain_bootstrap_v1.py tests/stage6/test_p6_runner_template_validator.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py
git commit -m "refactor: share P6 runner template validation"
```

### Task 2: Versioned profile registry, verifier, and pure recipe renderer

**Files:**

- Create: `framework/stage6/p6_history_recipe_profiles_v1.py`
- Create: `framework/stage6/p6_history_recipe_bridge_v1.py`
- Test: `tests/stage6/test_p6_history_recipe_bridge.py`

**Interfaces:**

- Consumes: `validate_pre_provision_runner_template(runner_template_path: Path, history_root: Path, expected_roles: Iterable[str], expected_marker_basenames: Mapping[str, str], role_selection: Mapping[str, str] | None = None, error_category: str = "execution_interface_unavailable") -> RunnerTemplateValidation` from Task 1.
- Produces:

```python
RECIPE_DERIVATION_FAILURE = "history_recipe_derivation_invalid"
PROFILE_NAME = "p6_stage5_pyramid_h800_tvm_profile_v1"
PROFILE_SCHEMA_VERSION = "p6_history_recipe_profile_v1"

@dataclass(frozen=True)
class ProceduralRecipeProfile:
    profile_name: str
    profile_schema_version: str
    target: Mapping[str, str]
    component_roles: tuple[str, ...]
    marker_basenames_by_role: Mapping[str, frozenset[str]]
    required_semantic_argv_by_role: Mapping[str, tuple[SemanticArgvRequirement, ...]]
    stage_width_surface: tuple[str, str, str]
    precision_surface: Mapping[str, PrecisionRequirement]
    output_capabilities: Mapping[str, frozenset[str]]
    identity_policy: str
    output_templates_by_q_mode: Mapping[str, Mapping[str, str]]

def get_procedural_recipe_profile(
    profile_name: str,
    profile_schema_version: str,
) -> ProceduralRecipeProfile

def derive_dynamic_recipe_from_procedural_source(
    *,
    source_map: Mapping[str, Any],
    runner_template_path: Path,
    history_root: Path,
) -> dict[str, Any]

def write_derived_recipe_json(recipe: Mapping[str, Any], output_path: Path) -> None
```

- Produces stable exception: `P6HistoryRecipeDerivationError(category="history_recipe_derivation_invalid", detail=str)`.
- Later tasks depend on exact schema output:

```python
{
    "schema_version": "p6_history_dynamic_materialization_recipe_v1",
    "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
    "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
    "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
    "output_path_templates_by_q_mode": {
        "fp16": {
            "training_path_template": "materialized/{artifact_id}/fp16/training",
            "checkpoint_path_template": "materialized/{artifact_id}/fp16/checkpoint",
            "onnx_path_template": "materialized/{artifact_id}/fp16/onnx",
            "calibration_path_template": "materialized/{artifact_id}/fp16/calibration",
        },
        "int8": {
            "training_path_template": "materialized/{artifact_id}/int8/training",
            "checkpoint_path_template": "materialized/{artifact_id}/int8/checkpoint",
            "onnx_path_template": "materialized/{artifact_id}/int8/onnx",
            "calibration_path_template": "materialized/{artifact_id}/int8/calibration",
        },
    },
}
```

- [ ] **Step 1: Write known-profile acceptance test**

```python
def test_known_profile_derives_pure_dynamic_recipe_from_runner_template(
    tmp_path: Path,
) -> None:
    history_root = _private_history_git_root(tmp_path)
    runner_template = _write_recipe_capable_runner_template(history_root)
    source_map = _procedural_source_map_v2(
        history_root,
        profile_name="p6_stage5_pyramid_h800_tvm_profile_v1",
        profile_schema_version="p6_history_recipe_profile_v1",
    )

    recipe = derive_dynamic_recipe_from_procedural_source(
        source_map=source_map,
        runner_template_path=runner_template,
        history_root=history_root,
    )

    assert recipe["schema_version"] == "p6_history_dynamic_materialization_recipe_v1"
    assert recipe["stage_width_fields"] == ["stage1_width", "stage2_width", "stage3_width"]
    assert recipe["group_id_template"] == "pyramid|{stage1_width}x{stage2_width}x{stage3_width}"
    assert recipe["artifact_id_template"] == "pyramid-{stage1_width}-{stage2_width}-{stage3_width}"
    assert set(recipe["output_path_templates_by_q_mode"]) == {"fp16", "int8"}
    serialized = json.dumps(recipe, sort_keys=True)
    assert str(history_root) not in serialized
    assert "GPU-" not in serialized
    assert "candidate" not in serialized.lower()
```

- [ ] **Step 2: Write profile/role/marker/argv/placeholder failure tests**

```python
@pytest.mark.parametrize(
    ("mutator", "label"),
    [
        (lambda source, template: source["procedural_recipe_profile"].update(profile_name="missing"), "unknown profile"),
        (lambda source, template: template["execution_interface"]["execution_chain"][0]["stage"].__setitem__(slice(None), "source_drift"), "role drift"),
        (lambda source, template: template["execution_interface"]["execution_chain"][0]["argv"].__setitem__(0, "documented-stage5-chain/wrong.sh"), "marker drift"),
        (lambda source, template: template["execution_interface"]["execution_chain"][0]["argv"].remove("{stage1_width}"), "width placeholder drift"),
        (lambda source, template: template["execution_interface"]["output_layout"].pop("checkpoint"), "missing capability"),
    ],
)
def test_profile_interface_drift_fails_closed_without_recipe(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any], dict[str, Any]], None],
    label: str,
) -> None:
    history_root = _private_history_git_root(tmp_path)
    payload = _recipe_capable_runner_template_payload()
    source_map = _procedural_source_map_v2(history_root)
    mutator(source_map, payload)
    runner_template = _write_yaml(history_root / "ignored" / f"{label}.yaml", payload)
    with pytest.raises(
        P6HistoryRecipeDerivationError,
        match=r"^history_recipe_derivation_invalid:",
    ):
        derive_dynamic_recipe_from_procedural_source(
            source_map=source_map,
            runner_template_path=runner_template,
            history_root=history_root,
        )
```

- [ ] **Step 3: Write precision and renderer collision tests**

```python
def test_precision_surface_requires_fp16_and_int8_without_mixed_precision(tmp_path: Path) -> None:
    history_root = _private_history_git_root(tmp_path)
    payload = _recipe_capable_runner_template_payload()
    payload["execution_interface"]["output_layout"]["precision_modes"] = ["fp16"]
    runner_template = _write_yaml(history_root / "ignored" / "runner.yaml", payload)
    with pytest.raises(P6HistoryRecipeDerivationError, match="history_recipe_derivation_invalid"):
        derive_dynamic_recipe_from_procedural_source(
            source_map=_procedural_source_map_v2(history_root),
            runner_template_path=runner_template,
            history_root=history_root,
        )

def test_renderer_rejects_output_template_collisions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = dataclasses.replace(
        get_procedural_recipe_profile(
            "p6_stage5_pyramid_h800_tvm_profile_v1",
            "p6_history_recipe_profile_v1",
        ),
        output_templates_by_q_mode={
            "fp16": {key: "materialized/{artifact_id}/same" for key in OUTPUT_TEMPLATE_KEYS},
            "int8": {key: "materialized/{artifact_id}/same" for key in OUTPUT_TEMPLATE_KEYS},
        },
    )
    monkeypatch.setattr(profiles, "PROFILE_REGISTRY", {profile.profile_name: profile})
    with pytest.raises(P6HistoryRecipeDerivationError, match="history_recipe_derivation_invalid"):
        _render_recipe(profile)
```

- [ ] **Step 4: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_recipe_bridge.py -q`

Expected: FAIL because profile and bridge modules do not exist.

- [ ] **Step 5: Implement the profile registry as pure public constants**

```python
PROFILE_REGISTRY = MappingProxyType({
    "p6_stage5_pyramid_h800_tvm_profile_v1": ProceduralRecipeProfile(
        profile_name="p6_stage5_pyramid_h800_tvm_profile_v1",
        profile_schema_version="p6_history_recipe_profile_v1",
        target={"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"},
        component_roles=(
            "controller",
            "source_materializer",
            "training",
            "checkpoint_export",
            "onnx_export",
            "int8_calibration",
            "performance_plan",
            "finalizer",
        ),
        marker_basenames_by_role={
            "controller": frozenset({"stage5_task_round_controller_v3.sh"}),
            "source_materializer": frozenset({"stage5_materialize_round_sources_v1.sh"}),
            "training": frozenset({"stage5_train_pyramid_width_v1.py"}),
            "checkpoint_export": frozenset({"stage5_export_pyramid_checkpoint_v1.py"}),
            "onnx_export": frozenset({"stage5_export_pyramid_onnx_v1.py"}),
            "int8_calibration": frozenset({"stage5_calibrate_pyramid_int8_v1.py"}),
            "performance_plan": frozenset({"stage5_build_performance_plan_v2.py"}),
            "finalizer": frozenset({"stage5_finalize_feedback_v2.py"}),
        },
        required_semantic_argv_by_role={
            "source_materializer": (
                SemanticArgvRequirement(name="measurement_request", required_placeholder="{measurement_request}"),
                SemanticArgvRequirement(name="output_root", required_placeholder="{round_output_root}"),
            ),
            "training": (
                SemanticArgvRequirement(name="stage1_width", required_placeholder="{stage1_width}"),
                SemanticArgvRequirement(name="stage2_width", required_placeholder="{stage2_width}"),
                SemanticArgvRequirement(name="stage3_width", required_placeholder="{stage3_width}"),
                SemanticArgvRequirement(name="q_mode", required_placeholder="{q_mode}"),
                SemanticArgvRequirement(name="output_root", required_placeholder="{p6_private_output_root}"),
            ),
            "checkpoint_export": (
                SemanticArgvRequirement(name="artifact_id", required_placeholder="{artifact_id}"),
                SemanticArgvRequirement(name="checkpoint_output", required_placeholder="{checkpoint_output}"),
            ),
            "onnx_export": (
                SemanticArgvRequirement(name="artifact_id", required_placeholder="{artifact_id}"),
                SemanticArgvRequirement(name="onnx_output", required_placeholder="{onnx_output}"),
            ),
            "int8_calibration": (
                SemanticArgvRequirement(name="q_mode", required_placeholder="{q_mode}"),
                SemanticArgvRequirement(name="calibration_output", required_placeholder="{calibration_output}"),
            ),
            "performance_plan": (
                SemanticArgvRequirement(name="task_state", required_placeholder="{task_state}"),
                SemanticArgvRequirement(name="output_root", required_placeholder="{round_output_root}"),
            ),
            "finalizer": (
                SemanticArgvRequirement(name="actual_feedback", required_placeholder="{actual_feedback}"),
                SemanticArgvRequirement(name="actual_receipt", required_placeholder="{actual_receipt}"),
            ),
        },
        stage_width_surface=("stage1_width", "stage2_width", "stage3_width"),
        precision_surface={
            "fp16": PrecisionRequirement(q_mode="fp16", required_roles=frozenset({"training", "checkpoint_export", "onnx_export"})),
            "int8": PrecisionRequirement(q_mode="int8", required_roles=frozenset({"training", "checkpoint_export", "onnx_export", "int8_calibration"})),
        },
        output_capabilities={
            "training": frozenset({"fp16", "int8"}),
            "checkpoint": frozenset({"fp16", "int8"}),
            "onnx": frozenset({"fp16", "int8"}),
            "calibration": frozenset({"fp16", "int8"}),
        },
        identity_policy="pyramid_canonical_group_id_v1",
        output_templates_by_q_mode=PUBLIC_P6_OUTPUT_TEMPLATES,
    )
})
```

Use literal synthetic marker basenames and semantic argv names only; do not include real paths, GPU IDs, candidate IDs, checkpoint names, ONNX names, metrics, or results.

- [ ] **Step 6: Implement verifier and renderer**

```python
def derive_dynamic_recipe_from_procedural_source(*, source_map, runner_template_path, history_root):
    profile_ref = _validate_procedural_profile_ref(source_map.get("procedural_recipe_profile"))
    source_ref = _validate_procedural_recipe_source(source_map.get("procedural_recipe_source"))
    profile = get_procedural_recipe_profile(profile_ref.name, profile_ref.schema_version)
    validated = validate_pre_provision_runner_template(
        runner_template_path=runner_template_path,
        history_root=history_root,
        expected_roles=profile.component_roles,
        expected_marker_basenames={
            role: next(iter(markers))
            for role, markers in profile.marker_basenames_by_role.items()
        },
        role_selection=source_ref.role_selection,
        error_category=RECIPE_DERIVATION_FAILURE,
    )
    _verify_semantic_argv_surface(profile, validated)
    _verify_output_capabilities(profile, validated)
    return _render_recipe(profile)
```

`_render_recipe` must validate format fields, render sample widths `(16, 32, 64)` and `(24, 48, 96)` across both q modes, reject absolute paths, `..`, duplicates, and any collision with `source_registry.json`.

- [ ] **Step 7: Run GREEN, lint, and diff**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_recipe_bridge.py tests/stage6/test_p6_runner_template_validator.py -q && python -m ruff check framework/stage6/p6_history_recipe_profiles_v1.py framework/stage6/p6_history_recipe_bridge_v1.py tests/stage6/test_p6_history_recipe_bridge.py && git diff --check`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add framework/stage6/p6_history_recipe_profiles_v1.py framework/stage6/p6_history_recipe_bridge_v1.py tests/stage6/test_p6_history_recipe_bridge.py
git commit -m "feat: derive P6 history recipes from procedural profiles"
```

### Task 3: Private-only derive CLI with atomic no-partial output

**Files:**

- Create: `tools/release/derive_p6_history_recipe.py`
- Test: `tests/release/test_derive_p6_history_recipe.py`
- Test: `tests/stage6/test_p6_history_recipe_bridge.py`

**Interfaces:**

- Consumes: `derive_dynamic_recipe_from_procedural_source(source_map: Mapping[str, Any], runner_template_path: Path, history_root: Path) -> dict[str, Any]` and `write_derived_recipe_json(recipe: Mapping[str, Any], output_path: Path) -> None` from Task 2.
- Produces CLI:

```text
derive_p6_history_recipe
  --source-map <absolute ignored private source-map v2 path>
  --runner-template <absolute ignored p6_history_runner_template_v1 path>
  --recipe-json <absolute ignored output recipe path>
```

- Success stdout: `p6_history_recipe_derived\n`.
- Argument error stderr: `argument_error\n`, exit 2.
- Derivation/privacy/IO failure stderr: `history_recipe_derivation_invalid\n`, exit 1.
- Reads profile name/version only from source-map v2; CLI has no profile override flag.

- [ ] **Step 1: Write CLI success and redaction tests**

```python
def test_cli_derives_recipe_to_absolute_ignored_output_without_private_echo(
    tmp_path: Path,
) -> None:
    history_root = _private_history_git_root(tmp_path)
    source_map_path = _write_source_map_v2(history_root / "ignored" / "source-map.json", history_root)
    runner_template = _write_recipe_capable_runner_template(history_root)
    recipe_json = history_root / "ignored" / "derived-recipe.json"

    result = _run_cli(
        "--source-map", str(source_map_path),
        "--runner-template", str(runner_template),
        "--recipe-json", str(recipe_json),
    )

    assert result.returncode == 0
    assert result.stdout == "p6_history_recipe_derived\n"
    assert result.stderr == ""
    recipe = json.loads(recipe_json.read_text(encoding="utf-8"))
    assert recipe["schema_version"] == "p6_history_dynamic_materialization_recipe_v1"
    assert str(history_root) not in result.stdout + result.stderr + recipe_json.read_text(encoding="utf-8")
```

- [ ] **Step 2: Write absolute ignored path and no-partial failure tests**

```python
@pytest.mark.parametrize(
    "bad_args",
    [
        ("--source-map", "relative.json"),
        ("--runner-template", "relative.yaml"),
        ("--recipe-json", "relative-output.json"),
    ],
)
def test_cli_requires_absolute_paths(tmp_path: Path, bad_args: tuple[str, str]) -> None:
    result = _run_cli(*bad_args)
    assert result.returncode == 2
    assert result.stderr == "argument_error\n"

def test_cli_rejects_unignored_output_and_preserves_existing_file(tmp_path: Path) -> None:
    history_root = _private_history_git_root(tmp_path)
    recipe_json = REPOSITORY_ROOT / f"p6-unignored-recipe-{tmp_path.name}.json"
    recipe_json.write_text("preserve\n", encoding="utf-8")
    try:
        result = _run_cli(
            "--source-map", str(_write_source_map_v2(history_root / "ignored" / "source-map.json", history_root)),
            "--runner-template", str(_write_recipe_capable_runner_template(history_root)),
            "--recipe-json", str(recipe_json),
        )
    finally:
        preserved = recipe_json.read_text(encoding="utf-8")
        recipe_json.unlink(missing_ok=True)
    assert result.returncode == 1
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert preserved == "preserve\n"
```

- [ ] **Step 3: Run RED**

Run: `PYTHONPATH=. pytest tests/release/test_derive_p6_history_recipe.py -q`

Expected: FAIL because `tools/release/derive_p6_history_recipe.py` does not exist.

- [ ] **Step 4: Implement argv-only CLI and private JSON loaders**

```python
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--source-map", required=True, type=_absolute_path)
    parser.add_argument("--runner-template", required=True, type=_absolute_path)
    parser.add_argument("--recipe-json", required=True, type=_absolute_path)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        source_map = _load_private_json(args.source_map)
        history_root = _resolve_declared_history_root(source_map)
        recipe = derive_dynamic_recipe_from_procedural_source(
            source_map=source_map,
            runner_template_path=args.runner_template,
            history_root=history_root,
        )
        write_derived_recipe_json(recipe, args.recipe_json)
    except P6HistoryRecipeDerivationError:
        sys.stderr.write("history_recipe_derivation_invalid\n")
        return 1
    except Exception:
        sys.stderr.write("history_recipe_derivation_invalid\n")
        return 1
    sys.stdout.write("p6_history_recipe_derived\n")
    return 0
```

Use `subprocess.run(["git", "-C", repository, "check-ignore", "-q", "--", relative])` for in-repository private path checks; reject symlink components, existing output files, and destinations outside the declared ignored output root.

- [ ] **Step 5: Verify failure leaves no temp files or recipe file**

Run: `PYTHONPATH=. pytest tests/release/test_derive_p6_history_recipe.py::test_cli_rejects_unignored_output_and_preserves_existing_file tests/stage6/test_p6_history_recipe_bridge.py::test_profile_interface_drift_fails_closed_without_recipe -q`

Expected: PASS.

- [ ] **Step 6: Run GREEN, lint, and diff**

Run: `PYTHONPATH=. pytest tests/release/test_derive_p6_history_recipe.py tests/stage6/test_p6_history_recipe_bridge.py -q && python -m ruff check tools/release/derive_p6_history_recipe.py tests/release/test_derive_p6_history_recipe.py && git diff --check`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/release/derive_p6_history_recipe.py tests/release/test_derive_p6_history_recipe.py tests/stage6/test_p6_history_recipe_bridge.py
git commit -m "feat: add private P6 recipe derivation CLI"
```

### Task 4: Source-map v2 normalizer and post-provision recipe consistency

**Files:**

- Modify: `framework/stage6/p6_history_normalization_v1.py`
- Modify: `framework/stage6/p6_history_binding_v1.py`
- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py`
- Modify: `tools/release/normalize_p6_history_root.py`
- Modify: `tools/release/provision_p6_full_chain_local_config.py`
- Test: `tests/stage6/test_p6_history_normalization.py`
- Test: `tests/release/test_normalize_p6_history_root.py`
- Test: `tests/stage6/test_p6_history_binding.py`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces:**

- Consumes: Task 2 bridge derivation, Task 3 CLI path loaders, existing `_validate_recipe(raw_recipe: object) -> dict[str, Any]`, `_validate_source_group(raw_group: object, recipe: Mapping[str, Any]) -> dict[str, Any]`, and `normalize_history_inputs(source_map: Mapping[str, Any], history_root: Path, private_dir: Path) -> dict[str, Path]`.
- Produces:

```python
SOURCE_MAP_V2_SCHEMA_VERSION = "p6_history_normalization_source_v2"

def normalize_history_inputs(
    source_map: Mapping[str, Any],
    history_root: Path,
    private_dir: Path,
    *,
    runner_template_path: Path | None = None,
) -> dict[str, Path]

def validate_binding_recipe_consistency(
    binding: Mapping[str, Any],
    expected_recipe: Mapping[str, Any] | None,
) -> None
```

- v1 callers remain valid because `runner_template_path` defaults to `None`.
- For v2 procedural mode, `runner_template_path` is required and the derived recipe is placed into the in-memory canonical source-map before existing recipe/source-contract validation runs.
- Normalized private root gains an ignored `derivation/recipe.json` and `derivation/recipe-canonical.json` only for private operational trace; public code never tracks them.

- [ ] **Step 1: Write v1 compatibility regression**

```python
def test_v1_explicit_recipe_normalization_does_not_require_runner_template(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    private_dir = tmp_path / "private-normalized"
    paths = normalize_history_inputs(source_map, _history_root(source_map), private_dir)
    assert paths["registry"] == private_dir / "registry" / "candidate-source-registry.json"
    assert "derivation_recipe" not in paths
```

- [ ] **Step 2: Write v2 explicit compatibility and mutual exclusivity tests**

```python
def test_v2_explicit_recipe_matches_v1_behavior(tmp_path: Path) -> None:
    source_map = _source_map_v2_explicit_from_v1(valid_private_source_map(tmp_path))
    paths = normalize_history_inputs(source_map, _history_root(source_map), tmp_path / "private-normalized")
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    assert registry["groups"][0]["source_contract"]["dynamic_materialization_recipe"] == source_map["dynamic_materialization_recipe"]

@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update(procedural_recipe_source={"role_selection": {}}, procedural_recipe_profile={"profile_name": "x", "profile_schema_version": "y"}),
        lambda payload: payload.pop("dynamic_materialization_recipe"),
        lambda payload: payload.update(recipe_mode="procedural_profile"),
    ],
)
def test_v2_recipe_sources_are_mutually_exclusive_and_write_no_root(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    source_map = _source_map_v2_explicit_from_v1(valid_private_source_map(tmp_path))
    mutator(source_map)
    private_dir = tmp_path / "private-normalized"
    with pytest.raises(P6HistoryNormalizationError, match=r"^history_recipe_derivation_invalid:"):
        normalize_history_inputs(source_map, _history_root(source_map), private_dir)
    assert not private_dir.exists()
```

- [ ] **Step 3: Write v2 procedural derive-before-existing-validation tests**

```python
def test_v2_procedural_mode_derives_recipe_then_normalizes_private_root(
    tmp_path: Path,
) -> None:
    source_map = _procedural_source_map_with_contract_template(tmp_path)
    runner_template = _write_recipe_capable_runner_template(Path(source_map["history_root"]))
    paths = normalize_history_inputs(
        source_map,
        Path(source_map["history_root"]),
        tmp_path / "private-normalized",
        runner_template_path=runner_template,
    )
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    recipe = registry["groups"][0]["source_contract"]["dynamic_materialization_recipe"]
    assert recipe["schema_version"] == "p6_history_dynamic_materialization_recipe_v1"
    assert paths["derivation_recipe"].is_relative_to(tmp_path / "private-normalized")

def test_v2_procedural_source_contract_recipe_mismatch_writes_no_normalized_root(
    tmp_path: Path,
) -> None:
    source_map = _procedural_source_map_with_contract_template(tmp_path)
    source_map["source_contract"]["source_contract"]["dynamic_materialization_recipe"] = _recipe_with_collision_free_drift()
    private_dir = tmp_path / "private-normalized"
    with pytest.raises(P6HistoryNormalizationError, match=r"^history_recipe_derivation_invalid:"):
        normalize_history_inputs(
            source_map,
            Path(source_map["history_root"]),
            private_dir,
            runner_template_path=_write_recipe_capable_runner_template(Path(source_map["history_root"])),
        )
    assert not private_dir.exists()
```

- [ ] **Step 4: Write post-provision binding recipe canonical equality tests**

```python
def test_full_chain_provision_rejects_binding_recipe_drift_before_writing_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy, runner_template, output_root, binding_path, config_path = _valid_provision_inputs(tmp_path)
    expected_recipe = _derived_recipe()

    def drifting_build_history_binding(*args: Any, **kwargs: Any) -> dict[str, Any]:
        binding = _valid_binding_with_recipe(expected_recipe)
        binding["source_contract_template"]["dynamic_materialization_recipe"] = _recipe_with_collision_free_drift()
        return binding

    monkeypatch.setattr(bootstrap, "build_history_binding", drifting_build_history_binding)
    with pytest.raises(FullChainBootstrapError, match=r"^history_recipe_derivation_invalid:"):
        materialize_full_chain_binding(
            legacy,
            runner_template,
            output_root,
            binding_path,
            config_path,
            _gpu_probe(),
        )
    assert not binding_path.exists()
    assert not config_path.exists()
```

- [ ] **Step 5: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py -q`

Expected: FAIL on v2 schema handling, missing `runner_template_path`, missing v2 CLI flag, and missing post-provision recipe equality gate.

- [ ] **Step 6: Implement v2 parser and in-memory recipe derivation**

```python
def _validate_private_source_map(source_map, history_root, *, runner_template_path=None):
    if source_map.get("schema_version") == SOURCE_MAP_SCHEMA_VERSION:
        return _validate_v1_explicit_source_map(source_map, history_root)
    if source_map.get("schema_version") == SOURCE_MAP_V2_SCHEMA_VERSION:
        recipe = _recipe_from_v2_source_map(
            source_map,
            history_root,
            runner_template_path=runner_template_path,
        )
        canonical = _v2_to_v1_canonical_source_map(source_map, recipe)
        return _validate_v1_explicit_source_map(canonical, history_root)
    _invalid("source map contract is invalid")
```

For v2 procedural failures, raise `P6HistoryNormalizationError("history_recipe_derivation_invalid", detail)` so the CLI prints the new category and leaves no normalized root.

- [ ] **Step 7: Extend normalizer CLI with optional runner template**

```python
parser.add_argument("--runner-template", required=False, type=_absolute_path)
args = parser.parse_args(argv)
normalize_history_inputs(
    _load_source_map(args.source_map),
    args.history_root,
    args.private_dir,
    runner_template_path=args.runner_template,
)
```

When a v2 procedural source-map is detected and `--runner-template` is absent, stderr must be `history_recipe_derivation_invalid\n`.

- [ ] **Step 8: Implement post-provision canonical equality**

```python
def validate_binding_recipe_consistency(binding, expected_recipe):
    if expected_recipe is None:
        return
    actual = binding.get("source_contract_template", {}).get("dynamic_materialization_recipe")
    if _canonical_json(actual) != _canonical_json(expected_recipe):
        raise P6HistoryBindingError(
            "history_recipe_derivation_invalid",
            "binding recipe differs from derived recipe",
        )
```

Thread the expected recipe through normalized private metadata or legacy locator without exposing it publicly; provision must load it from the ignored normalized root before `write_private_binding_pair(binding: Mapping[str, Any], config: Mapping[str, Any], binding_path: Path, config_path: Path, repository_root: Path) -> None`.

- [ ] **Step 9: Run GREEN, lint, and diff**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py -q && python -m ruff check framework/stage6/p6_history_normalization_v1.py framework/stage6/p6_history_binding_v1.py framework/stage6/p6_full_chain_bootstrap_v1.py tools/release/normalize_p6_history_root.py tools/release/provision_p6_full_chain_local_config.py tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py && git diff --check`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add framework/stage6/p6_history_normalization_v1.py framework/stage6/p6_history_binding_v1.py framework/stage6/p6_full_chain_bootstrap_v1.py tools/release/normalize_p6_history_root.py tools/release/provision_p6_full_chain_local_config.py tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py
git commit -m "feat: support procedural P6 history source maps"
```

### Task 5: Offline lifecycle integration and public status docs

**Files:**

- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/release/test_run_p6_h800_search.py`
- Modify: `tests/release/test_p6_history_execution_adapters.py`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify: `docs/superpowers/plans/2026-08-20-p6-procedural-recipe-bridge.md`

**Interfaces:**

- Consumes: Task 3 CLI, Task 4 normalizer/provision integration, existing `run_p6_coptv2x_search(contract: PublicP6CoptV2XContract, local: LocalP6CoptV2XConfig, code_revision: str, command_runner: CommandRunner) -> P6CoptV2XSearchState`, Stage1 bridge, dynamic Stage2 plan generation, and `materialize_history_registry(plan: Mapping[str, Any], binding: Mapping[str, Any], local_output_root: Path, registry_output_path: Path | None = None) -> dict[str, Any]`.
- Produces: test proof of lifecycle `derive -> normalize -> provision -> Stage1 -> dynamic Stage2 -> registry-v2 -> stop before measurement`.
- Public docs status: `procedural recipe bridge implementation complete; H800 private preflight pending`.

- [ ] **Step 1: Write an end-to-end zero-measurement test starting with derive**

```python
def test_procedural_recipe_bridge_reaches_registry_v2_without_measurement(
    tmp_path: Path,
) -> None:
    history_root = _private_history_git_root(tmp_path)
    source_map_path = _write_source_map_v2(history_root / "ignored" / "source-map.json", history_root)
    runner_template = _write_recipe_capable_runner_template(history_root)
    recipe_json = history_root / "ignored" / "derived-recipe.json"
    assert derive_cli.main((
        "--source-map", str(source_map_path),
        "--runner-template", str(runner_template),
        "--recipe-json", str(recipe_json),
    )) == 0

    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    normalized_root = tmp_path / "normalized-private-root"
    normalized = normalize_history_inputs(
        source_map,
        history_root,
        normalized_root,
        runner_template_path=runner_template,
    )
    binding_path, local_config_path = _provision_from_normalized_root(normalized, runner_template, tmp_path)
    local = load_local_config(local_config_path, load_public_contract(_write_public_contract(tmp_path)))
    roles: list[str] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        if len(argv) > 1 and Path(argv[1]).name == "measure_p6_history_batch.py":
            roles.append("measurement-boundary")
            return 1
        roles.append("registry" if len(argv) > 1 and Path(argv[1]).name == "build_p6_history_registry.py" else "stage1")
        return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False).returncode

    state = run_p6_coptv2x_search(load_public_contract(_write_public_contract(tmp_path)), local, "test-revision", runner)
    assert state.status == "failed"
    assert state.failure_code == "command_failed"
    assert state.measured_candidate_count == 0
    assert roles == ["stage1", "registry", "measurement-boundary"]
```

- [ ] **Step 2: Assert dynamic candidate count and plan/registry identity equality without 343/686 expectations**

```python
plan = json.loads((local.local_output_root / "pyramid_candidate_plan.json").read_text(encoding="utf-8"))
registry = json.loads((local.local_output_root / "source_registry.json").read_text(encoding="utf-8"))
assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
assert plan["candidate_count"] == len(plan["candidates"])
assert plan["candidate_count"] >= 16
assert plan["candidate_count"] not in {343, 686}
plan_identities = {
    ((tuple(candidate["width"]), candidate["q_mode"]), tuple(candidate["source_point_ids"]))
    for candidate in plan["candidates"]
}
registry_identities = {
    ((tuple(group["width"]), q_mode), tuple(group["source_point_ids_by_q_mode"][q_mode]))
    for group in registry["groups"]
    for q_mode in group["available_q_modes"]
}
assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
assert registry_identities == plan_identities
```

- [ ] **Step 3: Write release CLI lifecycle regression**

```python
def test_run_cli_procedural_bridge_stops_before_measurement_adapter(
    tmp_path: Path,
) -> None:
    paths = _write_procedural_full_chain_fixture(tmp_path)
    result = _run_cli(
        "--contract", str(paths["contract"]),
        "--local-config", str(paths["local_config"]),
        "--code-revision", "test-revision",
    )
    assert result.returncode == 1
    assert result.stderr == "execution_failed\n"
    assert json.loads((paths["output_root"] / "pyramid_candidate_plan.json").read_text())["candidate_count"] >= 16
    assert json.loads((paths["output_root"] / "source_registry.json").read_text())["schema_version"] == "stage5_candidate_source_registry_v2"
    assert not (paths["output_root"] / "measurement_request_round_0.json").exists()
```

- [ ] **Step 4: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py::test_procedural_recipe_bridge_reaches_registry_v2_without_measurement tests/release/test_run_p6_h800_search.py::test_run_cli_procedural_bridge_stops_before_measurement_adapter -q`

Expected: FAIL until Tasks 3 and 4 wire derive/normalizer/provision into the fixture helpers.

- [ ] **Step 5: Implement lifecycle fixture wiring**

Use existing helper patterns in `tests/stage6/test_coptv2x_h800_search.py`:

```python
source_map_path = _write_source_map_v2(history_root / "ignored" / "source-map.json", history_root)
runner_template = _write_recipe_capable_runner_template(history_root)
recipe_path = history_root / "ignored" / "derived-recipe.json"
assert derive_recipe_cli.main((
    "--source-map", str(source_map_path),
    "--runner-template", str(runner_template),
    "--recipe-json", str(recipe_path),
)) == 0
normalized = normalize_history_inputs(
    json.loads(source_map_path.read_text(encoding="utf-8")),
    history_root,
    normalized_root,
    runner_template_path=runner_template,
)
materialize_full_chain_binding(
    normalized["legacy"],
    runner_template,
    provisioned_root,
    binding_path,
    local_config_path,
    _OfflineGpuProbe(),
)
```

The runner must return failure at measurement boundary and must not execute `measure_p6_history_batch.py`.

- [ ] **Step 6: Update public release audit language**

Edit the current P6 status in `docs/AAAI27_RELEASE_AUDIT.md` to say the procedural recipe bridge implementation is complete only after offline gates pass. Include these exact statements:

```markdown
procedural recipe bridge implementation complete; H800 private preflight pending
```

```markdown
This does not claim real H800 private preflight, real four-round closure, paper evidence, checkpoint/ONNX/calibration generation, AP, latency, energy, or result completion.
```

- [ ] **Step 7: Run full offline verification gates**

Run:

```bash
PYTHONPATH=. python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
PYTHONPATH=. pytest tests/release tests/integration/test_anonymous_archive.py -q
python -m ruff check framework tools scripts tests
git diff --check
```

Expected: PASS; Stage5 measurement adapter call count remains zero in the procedural bridge preflight tests.

- [ ] **Step 8: Run disclosure guard**

Run:

```bash
rg -n "/home/|/mnt/|ssh|GPU-[0-9A-Za-z-]+|candidate_id|checkpoint|onnx|calibration|latency_ms|energy_j|ap30|ap50|ap70|terminal_status|measured_success|source_registry.json|pyramid_candidate_plan.json" framework/stage6 tools/release tests/stage6 tests/release docs/AAAI27_RELEASE_AUDIT.md docs/superpowers/plans/2026-08-20-p6-procedural-recipe-bridge.md
```

Expected: only synthetic schema/field names and public failure-category/test assertions appear; no private path, real GPU, asset/candidate ID, checkpoint instance, ONNX instance, calibration instance, metric value, result context, or private output path appears. If a hit is a schema-field test such as `"terminal_status"` from existing Stage5 tests, record it in the commit message body as an existing synthetic fixture field, not a new private leak.

- [ ] **Step 9: Commit**

```bash
git add tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py tests/release/test_p6_history_execution_adapters.py docs/AAAI27_RELEASE_AUDIT.md docs/superpowers/plans/2026-08-20-p6-procedural-recipe-bridge.md
git commit -m "test: gate P6 procedural recipe lifecycle offline"
```

### Task 6: Private H800 preflight retry after offline gates

**Files:**

- No tracked file changes.
- Private ignored inputs only: source-map v2, runner template, normalized root, binding, local config, Stage1 manifest, Stage2 plan, registry-v2, and preflight diagnostics.

**Interfaces:**

- Consumes committed Tasks 1-5.
- Consumes only an already-approved existing SSH master connection, a new private root, and existing TVM activation argv from the private runner template.
- Produces no commit and no tracked files.
- This task stops after private preflight gates; real four-round H800 execution is not a completion condition for this plan.

- [ ] **Step 1: Verify public worktree is clean before private retry**

Run: `git status --short`

Expected: no output.

- [ ] **Step 2: Verify offline gates are still green**

Run:

```bash
PYTHONPATH=. pytest tests/stage6/test_p6_history_recipe_bridge.py tests/stage6/test_p6_history_normalization.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_derive_p6_history_recipe.py tests/release/test_normalize_p6_history_root.py tests/release/test_run_p6_h800_search.py -q
python -m ruff check framework/stage6 tools/release tests/stage6 tests/release
git diff --check
```

Expected: PASS and `git status --short` remains empty.

- [ ] **Step 3: Create a new private ignored root without recording its path**

Run this only in the private operator shell, not in public docs or commits:

```bash
P6_PRIVATE_RETRY_ROOT="$(mktemp -d)"
chmod 700 "$P6_PRIVATE_RETRY_ROOT"
```

Expected: a fresh private root exists outside tracked files or under a Git-ignored location.

- [ ] **Step 4: Run private derive with existing source-map v2 and runner template**

Use the existing approved SSH master and existing TVM activation only if the private operator has already established them. Do not download, install, rebuild, train, export, compile, evaluate, or measure.

Command shape:

```bash
PYTHONPATH=. python tools/release/derive_p6_history_recipe.py \
  --source-map "$P6_PRIVATE_SOURCE_MAP_V2" \
  --runner-template "$P6_PRIVATE_RUNNER_TEMPLATE" \
  --recipe-json "$P6_PRIVATE_RETRY_ROOT/derived-recipe.json"
```

Expected: stdout `p6_history_recipe_derived`; failure stderr `history_recipe_derivation_invalid` and no recipe JSON.

- [ ] **Step 5: Run private normalize and provision**

Command shape:

```bash
PYTHONPATH=. python tools/release/normalize_p6_history_root.py \
  --source-map "$P6_PRIVATE_SOURCE_MAP_V2" \
  --history-root "$P6_PRIVATE_HISTORY_ROOT" \
  --private-dir "$P6_PRIVATE_RETRY_ROOT/normalized" \
  --runner-template "$P6_PRIVATE_RUNNER_TEMPLATE"

PYTHONPATH=. python tools/release/provision_p6_full_chain_local_config.py \
  --legacy-local-config "$P6_PRIVATE_RETRY_ROOT/normalized/legacy.local.yaml" \
  --runner-template "$P6_PRIVATE_RUNNER_TEMPLATE" \
  --local-output-root "$P6_PRIVATE_RETRY_ROOT/provisioned" \
  --binding-output "$P6_PRIVATE_RETRY_ROOT/provisioned/binding.json" \
  --config-output "$P6_PRIVATE_RETRY_ROOT/provisioned/local.yaml"
```

Expected: stdout `p6_history_root_normalized` then `p6_full_chain_config_written`; failures use stable categories and do not write partial normalized/provisioned roots.

- [ ] **Step 6: Run private preflight intercept through registry-v2 and stop before measurement**

Use a private command runner wrapper that returns non-zero when argv points to `measure_p6_history_batch.py`, after Stage1 and registry have completed.

Expected:

- Stage1 manifest exists only in the ignored private root and has `stage1_partition_manifest_v1`.
- Stage2 dynamic plan exists only in the ignored private root and has `p6_pyramid_candidate_plan_v2`.
- Registry exists only in the ignored private root and has `stage5_candidate_source_registry_v2`.
- `candidate_count >= 16`.
- Registry identities equal Stage2 plan identities.
- Measurement adapter call count is zero.

- [ ] **Step 7: Verify no tracked modifications**

Run: `git status --short`

Expected: no output.

- [ ] **Step 8: Record private retry status outside the public repo**

Write only a private ignored note containing stable categories and zero-measurement status. Do not copy paths, hostnames, GPU IDs, candidate IDs, checkpoint/ONNX/calibration names, metrics, results, logs, or command output into tracked files.

- [ ] **Step 9: No commit**

Run: `git status --short`

Expected: no output. Do not run `git add` or `git commit` for Task 6.

## Self-review checklist for implementers

- Spec coverage: Tasks 1-4 cover profile registry, interface verifier, recipe renderer, v1/v2 source-map semantics, fail-closed categories, no-circular-data-flow, private path gates, output collision gates, and post-provision binding equality. Task 5 covers zero-measurement lifecycle. Task 6 covers private retry boundary without treating real four rounds as completion.
- Placeholder scan: Domain placeholder names such as `{stage1_width}`, `{q_mode}`, `{measurement_request}`, and `{round_output_root}` are intentional interface values. Do not leave unresolved implementation notes in code, tests, or docs.
- Type/signature consistency: `RunnerTemplateValidation` from Task 1 is the only pre-provision template interface consumed by Task 2; `derive_dynamic_recipe_from_procedural_source(source_map: Mapping[str, Any], runner_template_path: Path, history_root: Path) -> dict[str, Any]` from Task 2 is the only derivation API consumed by Tasks 3 and 4; `normalize_history_inputs(source_map: Mapping[str, Any], history_root: Path, private_dir: Path, runner_template_path: Path | None = None) -> dict[str, Path]` remains backward compatible for v1 callers.
- Ownership conflicts: Task 1 owns validator extraction and bootstrap regression; Task 2 owns bridge internals; Task 3 owns derive CLI; Task 4 owns normalizer/provision integration; Task 5 owns lifecycle tests/docs; Task 6 owns private retry with no tracked files.
- Consumer compatibility: Existing `build_p6_history_registry.py` and `run_p6_h800_search.py` continue to consume binding/local config only; they do not learn source-map v2 or profile internals.
- Disclosure guard: Every task that changes public files must run `git diff --check` and a disclosure scan before commit.
