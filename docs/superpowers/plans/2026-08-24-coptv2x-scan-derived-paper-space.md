# CoptV2X Scan-Derived Paper Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Stage1 → Stage2 formal space construction so the public framework reproduces the Pyramid, CoDriving, and F-Cooper CoptV2X paper spaces from scanner-derived evidence.

**Architecture:** Stage1 owns a `structural_axes` contract generated from graph/dataflow groups, materializer bindings, canonical base widths, and hardware/backend constraints. Stage2 and P6 consume a generic free-axis schema plus formally derived q modes; legacy B1 search groups and probe anchors remain diagnostic-only.

**Tech Stack:** Python 3, pytest, pytest-cov, PyYAML, existing Stage1/Stage2/Stage5/Stage6 modules.

**Spec:** `docs/superpowers/specs/2026-08-24-coptv2x-scan-derived-paper-space-design.md`

## Global Constraints

- Acceptance covers only Pyramid, CoDriving, and F-Cooper paper-model space reproduction.
- Do not add arbitrary synthetic four-axis fixtures or tests such as `6×8×10×12`.
- Do not start post-source adapters, SSH jobs, GPU training, or the four-round real search in this plan.
- Production code must not use `3`, `5`, `7`, `343`, `686`, `1792`, or `3584` as generation drivers.
- Legacy `view_b1_search_groups`, `software_candidates`, probe anchors, and cliff neighbors are diagnostics, never formal-axis fallbacks.
- Missing or invalid scanner-owned `structural_axes` must fail formal construction.
- q modes are `hardware precision ∩ backend support ∩ configured compression modes ∩ graph quant units`; an empty intersection fails.
- Over-base formal candidates are rejected before registry writes and are not normal `unavailable` rows.
- Each implementation task has one implementer, then strict serial review: SPEC reviewer first, QUALITY reviewer second.
- TDD is mandatory: commit only after RED evidence, GREEN evidence, and review fixes.
- Coverage for touched modules must be `>=80%`; run full Stage6 and release tests before completion.
- Keep production files under 800 lines and touched functions under 50 lines where practical.

## File Structure

- Create `framework/stage1/structural_axes.py`: scanner-owned immutable data shapes, validation, axis derivation, legal-width derivation, fixed-derived provenance.
- Modify `framework/stage1/graph_scan.py`: emit low-level prune groups plus materializer-binding/dataflow metadata needed by `structural_axes.py`; keep legacy diagnostic views.
- Modify `framework/stage1/hardware_scan.py`: expose hardware precision without silently defaulting unsupported modes.
- Modify `framework/stage1_bridge.py`: load scanner-owned axes, derive formal q modes from all four sources, and fail if formal axes are missing.
- Create `framework/stage2/formal_search_space.py`: generic planner that enumerates free structural axes times q modes and returns a model-agnostic candidate schema.
- Modify `framework/stage6/pyramid_search_space_adapter_v1.py`: thin P6 mapping from generic formal plan into legacy Pyramid row fields.
- Modify `framework/stage6/p6_history_registry_v1.py`: reject over-base rows before writes.
- Create `tests/fixtures/coptv2x_paper_space/pyramid_scanner_contract.yaml`: purified scanner-contract fixture from real Pyramid scan.
- Create `tests/fixtures/coptv2x_paper_space/codriving_scanner_contract.yaml`: purified scanner-contract fixture from real CoDriving scan; neck is fixed-derived.
- Create `tests/fixtures/coptv2x_paper_space/fcooper_scanner_contract.yaml`: purified scanner-contract fixture from real F-Cooper scan; two neck interface axes are free only because graph independence and materializer bindings prove independence.
- Create `tests/stage1/test_structural_axes.py`: unit tests for axis construction and validation.
- Modify `tests/stage1/test_hardware_scan.py`: q-mode source tests.
- Modify `tests/stage1/test_stage1_bridge.py`: Stage2 contract tests for formal axes and q modes.
- Create `tests/stage2/test_formal_search_space.py`: generic planner tests.
- Modify `tests/stage6/test_pyramid_search_space_adapter.py`: P6 thin-adapter tests without touching uncommitted user edits blindly.
- Modify `tests/stage6/test_p6_history_registry.py`: over-base rejection tests.
- Create `tests/integration/test_coptv2x_paper_space_reproduction.py`: three paper-model CPU-only integration gate.
- Create `scripts/reproduce_paper_search_space.py`: CLI summary for paper-space reproduction.

## Shared Test Helper Contracts

Use these helper names only in the test files that define them; do not import
private helpers across test modules.

- In `tests/stage1/test_stage1_bridge.py`, define:

```python
PYRAMID_WIDTHS = [
    [16, 24, 32, 40, 48, 56, 64],
    [32, 48, 64, 80, 96, 112, 128],
    [64, 96, 128, 160, 192, 224, 256],
]

FCOOPER_WIDTHS = [
    [32, 64],
    [32, 64, 96, 128],
    [32, 64, 96, 128, 160, 192, 224, 256],
    [32, 64, 96, 128],
    [64, 96, 128, 160, 192, 224, 256],
]

def _paper_fixture(name: str) -> Path:
    return Path("tests/fixtures/coptv2x_paper_space") / name

def _paper_axis(axis_id: str, base_width: int, legal_widths: list[int]) -> dict[str, object]:
    return {
        "axis_id": axis_id,
        "axis_kind": "free",
        "base_width": base_width,
        "legal_widths": legal_widths,
        "member_b1_groups": [
            {
                "b1_group_id": axis_id,
                "module_path": axis_id,
                "canonical_to_member_num": 1,
                "canonical_to_member_den": 1,
                "materializer_param": axis_id,
                "role": "output",
            }
        ],
        "round_to": 8,
        "provenance": {"source": "test"},
    }

def _legacy_search_group_with_widths() -> dict[str, object]:
    return {"group_id": "legacy.stage1", "widths": [16, 24, 32], "round_to": 8}

def _legacy_software_candidate() -> dict[str, object]:
    return {"id": "legacy.stage1", "software_points": [{"id": "p0", "width": 16}]}
```

- In `tests/stage1/test_structural_axes.py`, define compact raw-scan helpers:

```python
def _scan(prune_groups, dataflow_relations, materializer_bindings, base_widths, backend_constraints):
    return {
        "structural_axis_inputs": {
            "prune_groups": prune_groups,
            "dataflow_relations": dataflow_relations,
            "materializer_bindings": materializer_bindings,
            "base_widths": base_widths,
            "backend_constraints": backend_constraints,
        }
    }

def _pyramid_stage_with_output_c_and_internal_2c():
    return _scan(
        prune_groups=[
            {"id": "stage3.out", "module_path": "backbone.stage3.out", "cur_width": 256, "dataflow_group_id": "stage3"},
            {"id": "stage3.inner", "module_path": "backbone.stage3.inner", "cur_width": 512, "dataflow_group_id": "stage3"},
        ],
        dataflow_relations=[{"group_id": "stage3", "canonical_axis_id": "backbone.stage3"}],
        materializer_bindings=[
            {"b1_group_id": "stage3.out", "axis_id": "backbone.stage3", "param": "stage3", "role": "output"},
            {"b1_group_id": "stage3.inner", "axis_id": "backbone.stage3", "param": "stage3", "role": "internal"},
        ],
        base_widths=[{"axis_id": "backbone.stage3", "width": 256}],
        backend_constraints=[{"axis_id": "backbone.stage3", "round_to": 32, "min_width": 64}],
    )

def _stage_with_non_integral_member_width():
    bad = _pyramid_stage_with_output_c_and_internal_2c()
    bad["structural_axis_inputs"]["prune_groups"][1]["cur_width"] = 384
    bad["structural_axis_inputs"]["backend_constraints"][0]["min_width"] = 96
    return bad
```

```python
def _codriving_scan_with_neck_binding():
    return _scan(
        prune_groups=[
            {"id": "s1.out", "module_path": "backbone.stage1", "cur_width": 64, "dataflow_group_id": "s1"},
            {"id": "s2.out", "module_path": "backbone.stage2", "cur_width": 128, "dataflow_group_id": "s2"},
            {"id": "s3.out", "module_path": "backbone.stage3", "cur_width": 256, "dataflow_group_id": "s3"},
            {"id": "neck.out", "module_path": "neck.output", "cur_width": 256, "dataflow_group_id": "s3"},
        ],
        dataflow_relations=[
            {"group_id": "s1", "canonical_axis_id": "backbone.stage1"},
            {"group_id": "s2", "canonical_axis_id": "backbone.stage2"},
            {"group_id": "s3", "canonical_axis_id": "backbone.stage3"},
        ],
        materializer_bindings=[
            {"b1_group_id": "s1.out", "axis_id": "backbone.stage1", "param": "stage1", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "s2.out", "axis_id": "backbone.stage2", "param": "stage2", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "s3.out", "axis_id": "backbone.stage3", "param": "stage3", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "neck.out", "axis_id": "neck.output", "param": "stage3", "role": "derived_neck", "axis_kind": "fixed_derived", "derived_from": "backbone.stage3"},
        ],
        base_widths=[
            {"axis_id": "backbone.stage1", "width": 64},
            {"axis_id": "backbone.stage2", "width": 128},
            {"axis_id": "backbone.stage3", "width": 256},
            {"axis_id": "neck.output", "width": 256},
        ],
        backend_constraints=[
            {"axis_id": "backbone.stage1", "round_to": 8, "min_width": 16},
            {"axis_id": "backbone.stage2", "round_to": 16, "min_width": 32},
            {"axis_id": "backbone.stage3", "round_to": 32, "min_width": 64},
            {"axis_id": "neck.output", "round_to": 32, "min_width": 64},
        ],
    )

def _fcooper_scan_with_independent_neck_bindings():
    return _scan(
        prune_groups=[
            {"id": "b0.out", "module_path": "backbone.s0", "cur_width": 64, "dataflow_group_id": "b0"},
            {"id": "b1.out", "module_path": "backbone.s1", "cur_width": 128, "dataflow_group_id": "b1"},
            {"id": "b2.out", "module_path": "backbone.s2", "cur_width": 256, "dataflow_group_id": "b2"},
            {"id": "neck.deblock", "module_path": "neck.deblock", "cur_width": 128, "dataflow_group_id": "neck_deblock"},
            {"id": "neck.output", "module_path": "neck.output", "cur_width": 256, "dataflow_group_id": "neck_output"},
        ],
        dataflow_relations=[
            {"group_id": "b0", "canonical_axis_id": "backbone.s0"},
            {"group_id": "b1", "canonical_axis_id": "backbone.s1"},
            {"group_id": "b2", "canonical_axis_id": "backbone.s2"},
            {"group_id": "neck_deblock", "canonical_axis_id": "neck.deblock", "independent_interface": True},
            {"group_id": "neck_output", "canonical_axis_id": "neck.output", "independent_interface": True},
        ],
        materializer_bindings=[
            {"b1_group_id": "b0.out", "axis_id": "backbone.s0", "param": "backbone.s0", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "b1.out", "axis_id": "backbone.s1", "param": "backbone.s1", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "b2.out", "axis_id": "backbone.s2", "param": "backbone.s2", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "neck.deblock", "axis_id": "neck.deblock", "param": "neck.deblock", "role": "neck_interface", "axis_kind": "free"},
            {"b1_group_id": "neck.output", "axis_id": "neck.output", "param": "neck.output", "role": "neck_interface", "axis_kind": "free"},
        ],
        base_widths=[
            {"axis_id": "backbone.s0", "width": 64},
            {"axis_id": "backbone.s1", "width": 128},
            {"axis_id": "backbone.s2", "width": 256},
            {"axis_id": "neck.deblock", "width": 128},
            {"axis_id": "neck.output", "width": 256},
        ],
        backend_constraints=[
            {"axis_id": "backbone.s0", "round_to": 32, "min_width": 32},
            {"axis_id": "backbone.s1", "round_to": 32, "min_width": 32},
            {"axis_id": "backbone.s2", "round_to": 32, "min_width": 32},
            {"axis_id": "neck.deblock", "round_to": 32, "min_width": 32},
            {"axis_id": "neck.output", "round_to": 32, "min_width": 64},
        ],
    )
```

- In `tests/stage2/test_formal_search_space.py`, define `_space_with_axes`,
  `_space_with_fixed_axis`, and `_legacy_software_candidate` locally. Each
  returns a `stage2_search_space_v1` dictionary with `axis_schema`,
  `formal_q_modes`, and optional diagnostic `software_candidates`.

---

### Task 0: Correct Early Commits Into the Final Contract

**Implementer:** one fresh implementer only.

**Files:**
- Modify: `framework/stage1/hardware_scan.py`
- Modify: `framework/stage1_bridge.py`
- Modify: `tests/stage1/test_hardware_scan.py`
- Modify: `tests/stage1/test_stage1_bridge.py`

**Interfaces:**
- Consumes: existing commits `a9123df fix: derive q modes from hardware capability` and `9efa944 feat: expose scan-derived formal structural axes`.
- Produces: corrective behavior where q modes use all four sources, and legacy search groups cannot act as formal-axis fallback.

- [ ] **Step 1: Inspect the current early commits**

Run:

```bash
git show --stat --oneline a9123df
git show --stat --oneline 9efa944
git diff a9123df^..HEAD -- framework/stage1/hardware_scan.py framework/stage1_bridge.py
```

Expected: inspection confirms `a9123df` only intersects hardware/quant-width sources and `9efa944` still permits legacy fallback or fixture-injected `formal_axis` behavior.

- [ ] **Step 2: Write RED q-mode tests for backend/config/graph intersections**

Add these tests to `tests/stage1/test_stage1_bridge.py`:

```python
def test_formal_q_modes_require_backend_support_config_and_graph_units(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["hardware"]["ips"]["gpu"]["precisions"] = ["FP16", "INT8"]
    manifest["backend_support"] = {"precisions": ["FP16"]}
    manifest["compression_modes"] = ["fp16", "int8"]
    manifest["quant_units"] = [{"id": "all", "legal_precisions": ["FP16", "INT8"]}]
    manifest["structural_axes"] = [_paper_axis("stage1", 64, [16, 24, 32, 40, 48, 56, 64])]
    path = _write_manifest(tmp_path / "backend-fp16.yaml", manifest)
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]

def test_formal_q_modes_fail_when_intersection_is_empty(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["hardware"]["ips"]["gpu"]["precisions"] = ["FP16"]
    manifest["backend_support"] = {"precisions": ["INT8"]}
    manifest["compression_modes"] = ["int8"]
    manifest["quant_units"] = [{"id": "all", "legal_precisions": ["INT8"]}]
    manifest["structural_axes"] = [_paper_axis("stage1", 64, [16, 24, 32, 40, 48, 56, 64])]
    path = _write_manifest(tmp_path / "empty-q.yaml", manifest)
    with pytest.raises(ValueError, match="formal q modes"):
        load_stage2_search_space(path)
```

- [ ] **Step 3: Write RED missing-axis test**

Add this test to `tests/stage1/test_stage1_bridge.py`:

```python
def test_stage2_search_space_rejects_missing_scanner_structural_axes(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.pop("structural_axes", None)
    manifest["view_b1_search_groups"] = [_legacy_search_group_with_widths()]
    manifest["software_candidates"] = [_legacy_software_candidate()]
    path = _write_manifest(tmp_path / "legacy-only.yaml", manifest)
    with pytest.raises(ValueError, match="structural_axes"):
        load_stage2_search_space(path)
```

- [ ] **Step 4: Verify RED**

Run:

```bash
pytest tests/stage1/test_stage1_bridge.py::test_formal_q_modes_require_backend_support_config_and_graph_units \
       tests/stage1/test_stage1_bridge.py::test_formal_q_modes_fail_when_intersection_is_empty \
       tests/stage1/test_stage1_bridge.py::test_stage2_search_space_rejects_missing_scanner_structural_axes -q
```

Expected: tests fail because current behavior does not yet enforce all four q-mode sources and missing scanner axes.

- [ ] **Step 5: Implement the correction**

In `framework/stage1_bridge.py`, replace legacy fallback logic with a fail-closed loader:

```python
def _scanner_structural_axes(raw: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    axes = raw.get("structural_axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("stage2 formal construction requires scanner structural_axes")
    return tuple(_validated_structural_axis(axis) for axis in axes)
```

Then implement q-mode intersection:

```python
def _formal_q_modes(
    hw: HwCapability,
    backend_support: Mapping[str, Any],
    compression_modes: Sequence[str],
    quant_units: Sequence[QuantUnit],
) -> list[str]:
    hardware = _normalized_precisions(hw.gpu_precisions)
    backend = _normalized_precisions(backend_support.get("precisions", []))
    configured = _normalized_precisions(compression_modes)
    graph = _graph_quant_precisions(quant_units)
    modes = hardware & backend & configured & graph
    ordered = [mode for mode in ("fp16", "int8") if mode in modes]
    if not ordered:
        raise ValueError("formal q modes intersection is empty")
    return ordered
```

- [ ] **Step 6: Verify GREEN**

Run:

```bash
pytest tests/stage1/test_hardware_scan.py tests/stage1/test_stage1_bridge.py -q
```

- [ ] **Step 7: SPEC review**

Assign one SPEC reviewer. The reviewer must read the spec and diff, then write `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/task-0-spec-review.md` with either `PASS` or exact required fixes. Do not start QUALITY review until SPEC is PASS.

- [ ] **Step 8: QUALITY review**

Assign one QUALITY reviewer after SPEC PASS. The reviewer must inspect error handling, public compatibility, file/function size, and tests, then write `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/task-0-quality-review.md` with `PASS` or exact required fixes.

- [ ] **Step 9: Commit**

Run:

```bash
git add framework/stage1/hardware_scan.py framework/stage1_bridge.py tests/stage1/test_hardware_scan.py tests/stage1/test_stage1_bridge.py
git commit -m "fix: require scanner axes and complete q-mode intersection"
```

---

### Task 1: Scanner-Owned Structural Axes Module

**Implementer:** one fresh implementer only.

**Files:**
- Create: `framework/stage1/structural_axes.py`
- Create: `tests/stage1/test_structural_axes.py`
- Modify: `framework/stage1/graph_scan.py`

**Interfaces:**
- Consumes: low-level prune groups with module paths/current widths, graph dependency/dataflow IDs, materializer bindings, base config/checkpoint widths, and alignment constraints.
- Produces: `derive_structural_axes(raw_scan: Mapping[str, Any]) -> StructuralAxisBundle`.

- [ ] **Step 1: Write RED tests for canonical coordinate and member ratios**

Create `tests/stage1/test_structural_axes.py` with:

```python
def test_derive_structural_axis_uses_canonical_base_not_max_internal_width() -> None:
    bundle = derive_structural_axes(_pyramid_stage_with_output_c_and_internal_2c())
    axis = bundle.free_axes[0]
    assert axis.axis_id == "backbone.stage3"
    assert axis.base_width == 256
    assert axis.legal_widths == (64, 96, 128, 160, 192, 224, 256)
    assert [(m.canonical_to_member_num, m.canonical_to_member_den) for m in axis.member_b1_groups] == [(1, 1), (2, 1)]

def test_derive_structural_axes_rejects_non_integral_member_ratio() -> None:
    with pytest.raises(ValueError, match="integer ratio"):
        derive_structural_axes(_stage_with_non_integral_member_width())
```

- [ ] **Step 2: Write RED tests for CoDriving and F-Cooper neck provenance**

Add:

```python
def test_codriving_neck_is_fixed_derived_not_free_axis() -> None:
    bundle = derive_structural_axes(_codriving_scan_with_neck_binding())
    assert [axis.axis_id for axis in bundle.free_axes] == ["backbone.stage1", "backbone.stage2", "backbone.stage3"]
    assert [axis.axis_id for axis in bundle.fixed_axes] == ["neck.output"]

def test_fcooper_neck_interfaces_are_free_when_bindings_are_independent() -> None:
    bundle = derive_structural_axes(_fcooper_scan_with_independent_neck_bindings())
    assert [axis.axis_id for axis in bundle.free_axes] == [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ]
```

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest tests/stage1/test_structural_axes.py -q
```

Expected: import or assertions fail because `framework.stage1.structural_axes` is absent or incomplete.

- [ ] **Step 4: Implement immutable data shapes and validation**

Create frozen dataclasses:

```python
@dataclass(frozen=True)
class AxisMemberBinding:
    b1_group_id: str
    module_path: str
    canonical_to_member_num: int
    canonical_to_member_den: int
    materializer_param: str
    role: str

@dataclass(frozen=True)
class StructuralAxis:
    axis_id: str
    dense_stage: str | None
    axis_kind: Literal["free", "fixed_derived"]
    base_width: int
    legal_widths: tuple[int, ...]
    member_b1_groups: tuple[AxisMemberBinding, ...]
    round_to: int
    provenance: Mapping[str, object]

@dataclass(frozen=True)
class StructuralAxisBundle:
    axes: tuple[StructuralAxis, ...]
    diagnostics: Mapping[str, object]
```

Expose properties `free_axes` and `fixed_axes` that filter by `axis_kind`.

- [ ] **Step 5: Implement scanner-owned grouping**

In `derive_structural_axes`, group by `(dataflow_group_id, materializer_param)` from the raw scan. Compute member ratios using `member_current_width / canonical_base_width`; reduce with `math.gcd`; reject denominator/width combinations that are non-integral for every legal width.

- [ ] **Step 6: Implement legal-width derivation**

Generate legal widths from `round_to` increments up to `base_width`, keep only widths satisfying graph/materializer alignment, and validate sorted uniqueness:

```python
def _legal_widths(base_width: int, round_to: int, min_width: int) -> tuple[int, ...]:
    return tuple(width for width in range(min_width, base_width + 1, round_to) if width % round_to == 0)
```

Use model-specific raw scanner evidence only as input data; do not branch on model name to return fixed axis counts.

- [ ] **Step 7: Wire graph scan output**

Modify `framework/stage1/graph_scan.py` to include the required metadata in the Stage1 manifest:

```yaml
structural_axis_inputs:
  prune_groups: [...]
  dataflow_relations: [...]
  materializer_bindings: [...]
  base_widths: [...]
  backend_constraints: [...]
```

Keep `view_b1_search_groups` unchanged as a diagnostic field.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
pytest tests/stage1/test_structural_axes.py tests/stage1/test_auto_and_run_scan.py -q
```

- [ ] **Step 9: SPEC review, then QUALITY review**

Run strict serial review:

```bash
python -m compileall framework/stage1/structural_axes.py -q
pytest tests/stage1/test_structural_axes.py --cov=framework.stage1.structural_axes --cov-fail-under=80 -q
```

SPEC reviewer writes `task-1-spec-review.md`; QUALITY reviewer writes `task-1-quality-review.md`.

- [ ] **Step 10: Commit**

Run:

```bash
git add framework/stage1/structural_axes.py framework/stage1/graph_scan.py tests/stage1/test_structural_axes.py
git commit -m "feat: derive structural axes from scanner evidence"
```

---

### Task 2: Stage1 Bridge Formal Contract

**Implementer:** one fresh implementer only.

**Files:**
- Modify: `framework/stage1_bridge.py`
- Modify: `tests/stage1/test_stage1_bridge.py`
- Create: `tests/fixtures/coptv2x_paper_space/pyramid_scanner_contract.yaml`
- Create: `tests/fixtures/coptv2x_paper_space/codriving_scanner_contract.yaml`
- Create: `tests/fixtures/coptv2x_paper_space/fcooper_scanner_contract.yaml`

**Interfaces:**
- Consumes: `derive_structural_axes(raw_scan)` and `HwCapability`.
- Produces: `load_stage2_search_space(path)` with `structural_axes`, `axis_schema`, `formal_q_modes`, and diagnostic-only legacy fields.

- [ ] **Step 1: Build purified scanner-contract fixtures**

Create the three fixture files from real scanner contracts by retaining only:

```yaml
schema: stage1_scanner_contract_v1
model: <pyramid_lidar|codriving|fcooper>
hardware_target:
  name: h800
backend_support:
  precisions: [FP16, INT8]
compression_modes: [fp16, int8]
quant_units:
  - id: all_conv_units
    legal_precisions: [FP16, INT8]
structural_axis_inputs:
  prune_groups: [...]
  dataflow_relations: [...]
  materializer_bindings: [...]
  base_widths: [...]
  backend_constraints: [...]
view_b1_search_groups: [...]
software_candidates: []
```

Do not add synthetic networks and do not encode expected counts in fixture fields.

- [ ] **Step 2: Write RED paper fixture bridge tests**

Add to `tests/stage1/test_stage1_bridge.py`:

```python
@pytest.mark.parametrize(
    ("fixture_name", "expected_axis_ids", "expected_widths", "expected_q_modes"),
    [
        ("pyramid_scanner_contract.yaml", ["backbone.stage1", "backbone.stage2", "backbone.stage3"], PYRAMID_WIDTHS, ["fp16", "int8"]),
        ("codriving_scanner_contract.yaml", ["backbone.stage1", "backbone.stage2", "backbone.stage3"], PYRAMID_WIDTHS, ["fp16", "int8"]),
        ("fcooper_scanner_contract.yaml", ["backbone.s0", "backbone.s1", "backbone.s2", "neck.deblock", "neck.output"], FCOOPER_WIDTHS, ["fp16", "int8"]),
    ],
)
def test_stage1_bridge_emits_scanner_owned_paper_axes(fixture_name, expected_axis_ids, expected_widths, expected_q_modes):
    space = load_stage2_search_space(_paper_fixture(fixture_name))
    assert [axis["axis_id"] for axis in space["axis_schema"]["free_axes"]] == expected_axis_ids
    assert [axis["legal_widths"] for axis in space["axis_schema"]["free_axes"]] == expected_widths
    assert space["formal_q_modes"] == expected_q_modes
```

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest tests/stage1/test_stage1_bridge.py::test_stage1_bridge_emits_scanner_owned_paper_axes -q
```

- [ ] **Step 4: Implement bridge loading**

In `SpaceSpec.stage2_search_space()`, call `derive_structural_axes(self.raw)` and serialize:

```python
"structural_axes": [_axis_to_dict(axis) for axis in axis_bundle.axes],
"axis_schema": {
    "free_axes": [_axis_to_dict(axis) for axis in axis_bundle.free_axes],
    "fixed_derived_axes": [_axis_to_dict(axis) for axis in axis_bundle.fixed_axes],
},
"formal_candidate_policy": {
    "enumeration": "axis_schema.free_axes_x_formal_q_modes",
    "legacy_views": "diagnostic_only",
},
```

Keep `view_b1_prune_groups`, `view_b1_search_groups`, and `software_candidates` in the payload for compatibility.

- [ ] **Step 5: Implement bridge q modes**

Pass `backend_support`, `compression_modes`, and `quant_units` into the Task 0 `_formal_q_modes` intersection. Reject empty q modes before returning Stage2 space.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
pytest tests/stage1/test_stage1_bridge.py tests/stage1/test_structural_axes.py -q
```

- [ ] **Step 7: SPEC review, then QUALITY review**

SPEC reviewer checks that legacy fallback is removed. QUALITY reviewer checks fixture purity and compatibility-field preservation. Write `task-2-spec-review.md` then `task-2-quality-review.md`.

- [ ] **Step 8: Commit**

Run:

```bash
git add framework/stage1_bridge.py tests/stage1/test_stage1_bridge.py tests/fixtures/coptv2x_paper_space
git commit -m "feat: expose scanner-owned paper space contract"
```

---

### Task 3: Generic Formal Search-Space Planner

**Implementer:** one fresh implementer only.

**Files:**
- Create: `framework/stage2/formal_search_space.py`
- Create: `tests/stage2/test_formal_search_space.py`

**Interfaces:**
- Consumes: `axis_schema.free_axes` and `formal_q_modes`.
- Produces: `build_formal_search_plan(search_space: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write RED generic planner tests**

Create `tests/stage2/test_formal_search_space.py`:

```python
def test_formal_planner_enumerates_free_axes_and_q_modes() -> None:
    plan = build_formal_search_plan(_space_with_axes([[16, 24], [32, 64]], ["fp16", "int8"]))
    assert plan["structure_count"] == 4
    assert plan["candidate_count"] == 8
    assert plan["candidates"][0]["axis_values"] == {"a0": 16, "a1": 32}
    assert plan["candidates"][0]["q_mode"] == "fp16"

def test_formal_planner_ignores_fixed_derived_axes_for_structure_count() -> None:
    plan = build_formal_search_plan(_space_with_fixed_axis())
    assert plan["free_axis_count"] == 1
    assert plan["structure_count"] == 2

def test_formal_planner_rejects_missing_axes_even_when_legacy_candidates_exist() -> None:
    with pytest.raises(ValueError, match="axis_schema.free_axes"):
        build_formal_search_plan({"software_candidates": [_legacy_software_candidate()], "formal_q_modes": ["fp16"]})
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/stage2/test_formal_search_space.py -q
```

- [ ] **Step 3: Implement planner**

Implement sorted product enumeration:

```python
def build_formal_search_plan(search_space: Mapping[str, Any]) -> dict[str, Any]:
    free_axes = _validated_free_axes(search_space)
    q_modes = _validated_q_modes(search_space)
    structures = list(_structure_product(free_axes))
    candidates = [
        _candidate_row(index, axis_values, q_mode)
        for index, (axis_values, q_mode) in enumerate(product(structures, q_modes))
    ]
    return {
        "schema": "formal_search_plan_v1",
        "model": search_space.get("model"),
        "free_axis_count": len(free_axes),
        "structure_count": len(structures),
        "candidate_count": len(candidates),
        "axis_schema": deepcopy(search_space["axis_schema"]),
        "q_modes": list(q_modes),
        "candidates": candidates,
    }
```

Validation rejects duplicate axis IDs, non-positive widths, over-base legal widths, duplicate q modes, and empty products.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/stage2/test_formal_search_space.py -q
```

- [ ] **Step 5: SPEC review, then QUALITY review**

SPEC reviewer confirms no model-name branching. QUALITY reviewer confirms small pure functions and deterministic ordering. Write `task-3-spec-review.md` then `task-3-quality-review.md`.

- [ ] **Step 6: Commit**

Run:

```bash
git add framework/stage2/formal_search_space.py tests/stage2/test_formal_search_space.py
git commit -m "feat: add generic formal search-space planner"
```

---

### Task 4: P6 Pyramid Thin Adapter and Registry Guard

**Implementer:** one fresh implementer only.

**Files:**
- Modify: `framework/stage6/pyramid_search_space_adapter_v1.py`
- Modify: `framework/stage6/p6_history_registry_v1.py`
- Modify: `tests/stage6/test_pyramid_search_space_adapter.py`
- Modify: `tests/stage6/test_p6_history_registry.py`

**Interfaces:**
- Consumes: `build_formal_search_plan(search_space)`.
- Produces: P6-compatible rows and registry validation without legacy anchor enumeration.

- [ ] **Step 1: Preserve user edits before touching Stage6 tests**

Run:

```bash
git status --short tests/stage6/test_pyramid_search_space_adapter.py
git diff -- tests/stage6/test_pyramid_search_space_adapter.py
```

If the file has uncommitted user edits, append new tests without deleting or rewriting unrelated hunks.

- [ ] **Step 2: Write RED P6 formal adapter test**

Add to `tests/stage6/test_pyramid_search_space_adapter.py`:

```python
def test_build_pyramid_candidate_plan_uses_generic_formal_plan() -> None:
    space = _pyramid_space_with_axis_schema()
    plan = build_pyramid_candidate_plan(space)
    assert plan["structure_count"] == 343
    assert plan["candidate_count"] == 686
    assert plan["candidates"][0]["width"] == [16, 32, 64]
    assert plan["candidates"][-1]["width"] == [64, 128, 256]
    assert {row["q_mode"] for row in plan["candidates"]} == {"fp16", "int8"}
```

- [ ] **Step 3: Write RED registry over-base test**

Add to `tests/stage6/test_p6_history_registry.py`:

```python
def test_registry_rejects_over_base_formal_candidate_before_write(tmp_path: Path) -> None:
    plan = _formal_plan_with_candidate(width=[64, 128, 320], base_widths=[64, 128, 256])
    registry_path = tmp_path / "registry.json"
    with pytest.raises(P6HistoryRegistryError, match="over base"):
        write_p6_history_registry(plan, registry_path)
    assert not registry_path.exists()
```

- [ ] **Step 4: Verify RED**

Run:

```bash
pytest tests/stage6/test_pyramid_search_space_adapter.py::test_build_pyramid_candidate_plan_uses_generic_formal_plan \
       tests/stage6/test_p6_history_registry.py::test_registry_rejects_over_base_formal_candidate_before_write -q
```

- [ ] **Step 5: Implement P6 thin mapping**

In `build_pyramid_candidate_plan`, call `build_formal_search_plan(search_space)` first. Map axis IDs to legacy `width=[stage1, stage2, stage3]` only after generic enumeration validates the plan. Remove production use of `_common_stage_points()` and legacy `software_candidates` for formal enumeration.

- [ ] **Step 6: Implement registry fail-closed guard**

Before writing any registry file, validate every formal candidate width against the axis base widths from the plan. Raise `P6HistoryRegistryError("formal candidate width over base")` on the first violation.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
pytest tests/stage6/test_pyramid_search_space_adapter.py tests/stage6/test_p6_history_registry.py tests/stage2/test_formal_search_space.py -q
```

- [ ] **Step 8: SPEC review, then QUALITY review**

SPEC reviewer confirms P6 is a thin adapter and registry no longer records over-base formal candidates as `unavailable`. QUALITY reviewer checks deterministic plan rows and compatibility fields. Write `task-4-spec-review.md` then `task-4-quality-review.md`.

- [ ] **Step 9: Commit**

Run:

```bash
git add framework/stage6/pyramid_search_space_adapter_v1.py framework/stage6/p6_history_registry_v1.py tests/stage6/test_pyramid_search_space_adapter.py tests/stage6/test_p6_history_registry.py
git commit -m "fix: drive P6 from generic formal axes"
```

---

### Task 5: Three Paper-Model Reproduction Gate

**Implementer:** one fresh implementer only.

**Files:**
- Create: `tests/integration/test_coptv2x_paper_space_reproduction.py`
- Create: `scripts/reproduce_paper_search_space.py`
- Modify only if needed: `framework/stage5/fcooper_space_v1.py`
- Modify only if needed: `framework/stage5/source_registry_v1.py`

**Interfaces:**
- Consumes: `load_stage2_search_space`, `build_formal_search_plan`, and the P6 thin adapter.
- Produces: CPU-only proof that the scanner contract reproduces paper spaces.

- [ ] **Step 1: Write RED integration tests**

Create:

```python
@pytest.mark.parametrize(
    ("fixture_name", "expected_structures", "expected_candidates"),
    [
        ("pyramid_scanner_contract.yaml", 343, 686),
        ("codriving_scanner_contract.yaml", 343, 686),
        ("fcooper_scanner_contract.yaml", 1792, 3584),
    ],
)
def test_coptv2x_paper_space_reproduction(fixture_name, expected_structures, expected_candidates):
    space = load_stage2_search_space(_paper_fixture(fixture_name))
    plan = build_formal_search_plan(space)
    assert plan["structure_count"] == expected_structures
    assert plan["candidate_count"] == expected_candidates
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/integration/test_coptv2x_paper_space_reproduction.py -q
```

- [ ] **Step 3: Implement CLI**

Create `scripts/reproduce_paper_search_space.py` with:

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = [_summary(model) for model in _selected_models(args.model)]
    for row in rows:
        print(f"{row['model']}: structures={row['structures']} candidates={row['candidates']} q_modes={','.join(row['q_modes'])}")
    return 0
```

The CLI supports `--model {pyramid,codriving,fcooper,all}` and exits non-zero when any oracle count fails.

- [ ] **Step 4: Verify GREEN and CLI summaries**

Run:

```bash
pytest tests/integration/test_coptv2x_paper_space_reproduction.py -q
python scripts/reproduce_paper_search_space.py --model all
```

Expected CLI lines:

```text
pyramid: structures=343 candidates=686 q_modes=fp16,int8
codriving: structures=343 candidates=686 q_modes=fp16,int8
fcooper: structures=1792 candidates=3584 q_modes=fp16,int8
```

- [ ] **Step 5: SPEC review, then QUALITY review**

SPEC reviewer confirms counts are assertions only and not production drivers. QUALITY reviewer confirms CLI uses public APIs and no GPU/SSH. Write `task-5-spec-review.md` then `task-5-quality-review.md`.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests/integration/test_coptv2x_paper_space_reproduction.py scripts/reproduce_paper_search_space.py framework/stage5/fcooper_space_v1.py framework/stage5/source_registry_v1.py
git commit -m "test: reproduce CoptV2X paper spaces"
```

If the Stage5 files are unchanged, omit them from `git add`.

---

### Task 6: Full CPU Verification and Final Two-Stage Review

**Implementer:** one fresh implementer only for any fixes. Reviewers remain separate and serial.

**Files:**
- Modify only files required by failing verification.
- Write review records under `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/`.

**Interfaces:**
- Consumes: all prior committed tasks.
- Produces: verified branch ready for the later real Pyramid run restart.

- [ ] **Step 1: Run targeted Stage1/Stage2/Stage5/Stage6/integration tests**

Run:

```bash
pytest tests/stage1 tests/stage2 tests/stage5 tests/stage6 tests/integration/test_coptv2x_paper_space_reproduction.py -q
```

- [ ] **Step 2: Run full Stage6 and release tests explicitly**

Run:

```bash
pytest tests/stage6 -q
pytest tests/release -q
```

- [ ] **Step 3: Run compile and coverage gates**

Run:

```bash
python -m compileall framework scripts tests -q
pytest tests/stage1 tests/stage2 tests/stage6 tests/integration/test_coptv2x_paper_space_reproduction.py \
  --cov=framework.stage1.structural_axes \
  --cov=framework.stage1_bridge \
  --cov=framework.stage2.formal_search_space \
  --cov=framework.stage6.pyramid_search_space_adapter_v1 \
  --cov=framework.stage6.p6_history_registry_v1 \
  --cov-fail-under=80 -q
```

- [ ] **Step 4: Run final CLI proof**

Run:

```bash
python scripts/reproduce_paper_search_space.py --model all
```

Expected:

```text
pyramid: structures=343 candidates=686 q_modes=fp16,int8
codriving: structures=343 candidates=686 q_modes=fp16,int8
fcooper: structures=1792 candidates=3584 q_modes=fp16,int8
```

- [ ] **Step 5: Final SPEC review**

Assign one SPEC reviewer. Reviewer reads the spec and complete diff:

```bash
git diff --stat HEAD~5..HEAD
git diff HEAD~5..HEAD -- framework tests scripts
```

Reviewer writes `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/final-spec-review.md` with `PASS` only if:

- no arbitrary synthetic network test exists;
- formal axes are scanner-owned and missing axes fail;
- CoDriving neck is fixed-derived;
- F-Cooper neck axes are proven independent by graph/materializer evidence;
- q modes use all four sources;
- paper counts appear only in tests or CLI oracle summaries.

- [ ] **Step 6: Final QUALITY review**

After SPEC PASS, assign one QUALITY reviewer. Reviewer runs:

```bash
python -m compileall framework scripts tests -q
find framework tests scripts -name '*.py' -print0 | xargs -0 wc -l | sort -nr | sed -n '1,20p'
```

Reviewer writes `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/final-quality-review.md` with `PASS` only if:

- tests are deterministic and CPU-only;
- touched production files remain below 800 lines where practical;
- touched functions remain below 50 lines where practical;
- no public compatibility field is removed without replacement;
- no post-source adapter, SSH, or GPU job was started.

- [ ] **Step 7: Commit final verification fixes if any**

If reviews require fixes, run the smallest relevant tests again and commit:

```bash
git add framework tests scripts
git commit -m "fix: address paper space review findings"
```

- [ ] **Step 8: Handoff**

Report final commit SHA, verification commands and results, review PASS files, and whether the branch is ready to restart a new real Pyramid full-chain experiment. Do not launch the real experiment in this plan.
