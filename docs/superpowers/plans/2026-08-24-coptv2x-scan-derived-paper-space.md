# CoptV2X Scan-Derived Paper Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Stage1 → Stage2 formal space construction so the public framework reproduces the three CoptV2X paper model spaces.

**Architecture:** Add a formal structural-axis contract alongside existing diagnostic `software_candidates`. Stage2/P6 enumerate the formal axes times capability-derived q modes; legacy diagnostic anchors remain available but stop driving formal plans.

**Tech Stack:** Python 3, pytest, PyYAML, existing Stage1/Stage6 modules.

**Spec:** `docs/superpowers/specs/2026-08-24-coptv2x-scan-derived-paper-space-design.md`

## Global Constraints

- No arbitrary synthetic 4-axis test; acceptance is Pyramid, CoDriving, and F-Cooper only.
- Production code must not hardcode paper counts as generation drivers.
- Write RED tests before production changes.
- Keep production files below 800 lines where practical and functions below 50 lines where practical.
- Preserve existing public compatibility fields and legacy tests.

---

### Task 1: Capability-Derived Q Modes

**Files:**
- Modify: `framework/stage1/hardware_scan.py`
- Test: `tests/stage1/test_hardware_scan.py`

**Interfaces:**
- Consumes: `HwCapability.gpu_precisions`, `HwCapability.legal_bits`
- Produces: `HwCapability.legal_bits` that intersects GPU precision capability with quant constraints.

- [ ] **Step 1: Write the failing tests**

```python
def test_hardware_scan_intersects_gpu_precisions_with_quant_constraints(tmp_path: Path) -> None:
    raw = _raw_capability()
    raw["ips"]["gpu"]["precisions"] = ["FP16"]
    raw["quant_constraints"]["bit_widths_w"] = [8, 16]
    path = tmp_path / "fp16.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert HwCapability.from_yaml(path).legal_bits == ["FP16"]

def test_hardware_scan_rejects_int4_only_as_no_supported_search_bits(tmp_path: Path) -> None:
    raw = _raw_capability()
    raw["ips"]["gpu"]["precisions"] = ["INT4"]
    raw["quant_constraints"]["bit_widths_w"] = [4]
    path = tmp_path / "int4.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert HwCapability.from_yaml(path).legal_bits == []
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/stage1/test_hardware_scan.py::test_hardware_scan_intersects_gpu_precisions_with_quant_constraints tests/stage1/test_hardware_scan.py::test_hardware_scan_rejects_int4_only_as_no_supported_search_bits -q`

- [ ] **Step 3: Implement minimal intersection logic**

Map `8→INT8`, `16→FP16`; if `ips.gpu.precisions` is present, intersect with it; if no capability list is present, preserve the existing safe default.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/stage1/test_hardware_scan.py -q`

- [ ] **Step 5: Commit**

Run: `git add framework/stage1/hardware_scan.py tests/stage1/test_hardware_scan.py && git commit -m "fix: derive q modes from hardware capability"`

### Task 2: Formal Structural Axes in Stage1 Bridge

**Files:**
- Modify: `framework/stage1_bridge.py`
- Test: `tests/stage1/test_stage1_bridge.py`

**Interfaces:**
- Consumes: `view_b1_search_groups[*].formal_axis` when present; otherwise derives from `canonical_base_width` or legacy `widths`.
- Produces: `stage2_search_space()["structural_axes"]`, `formal_q_modes`, and `formal_candidate_policy`.

- [ ] **Step 1: Write failing tests for paper-model axis reproduction fixtures**

Create fixtures for Pyramid, CoDriving, and F-Cooper using `formal_axis.legal_widths` and assert axis counts, exact width sets, structure products, and q modes.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/stage1/test_stage1_bridge.py -q`

- [ ] **Step 3: Implement minimal formal-axis parser**

Add immutable dataclass fields for optional `canonical_base_width` and `formal_legal_widths`; validate positive integers, sorted unique widths, divisibility by `round_to`, and `max(widths) <= base_width`.

- [ ] **Step 4: Add q-mode derivation in `SpaceSpec.stage2_search_space()`**

Compute `formal_q_modes` from `HwCapability.legal_bits` and quant units, lowercased as `fp16`/`int8`.

- [ ] **Step 5: Verify GREEN**

Run: `pytest tests/stage1/test_stage1_bridge.py tests/stage6/test_pyramid_search_space_adapter.py -q`

- [ ] **Step 6: Commit**

Run: `git add framework/stage1_bridge.py tests/stage1/test_stage1_bridge.py && git commit -m "feat: expose scan-derived formal structural axes"`

### Task 3: P6 Formal Plan Enumeration

**Files:**
- Modify: `framework/stage6/pyramid_search_space_adapter_v1.py`
- Test: `tests/stage6/test_pyramid_search_space_adapter.py`

**Interfaces:**
- Consumes: `structural_axes` and `formal_q_modes` when present.
- Produces: P6 candidate plan with `structure_count`, `candidate_count`, and source provenance.

- [ ] **Step 1: Write failing tests for Pyramid paper plan**

Add a Stage2 payload with three formal axes and `formal_q_modes=["fp16","int8"]`; assert `structure_count == 343`, `candidate_count == 686`, exact first/last widths, and both q modes.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/stage6/test_pyramid_search_space_adapter.py::test_build_pyramid_candidate_plan_uses_formal_axes_for_paper_space -q`

- [ ] **Step 3: Implement formal-axis branch**

Before legacy `software_candidates` parsing, validate axes, enumerate product of `legal_widths`, and attach per-axis provenance IDs.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/stage6/test_pyramid_search_space_adapter.py -q`

- [ ] **Step 5: Commit**

Run: `git add framework/stage6/pyramid_search_space_adapter_v1.py tests/stage6/test_pyramid_search_space_adapter.py && git commit -m "fix: enumerate P6 candidates from formal axes"`

### Task 4: Scanner Canonical Base and Over-Base Guard

**Files:**
- Modify: `framework/stage1/graph_scan.py`
- Modify: `framework/stage6/p6_history_registry_v1.py`
- Test: `tests/stage6/test_p6_history_registry.py`

**Interfaces:**
- Consumes: consolidated member widths and recipe-v2 base widths.
- Produces: canonical base metadata and fail-closed registry behavior.

- [ ] **Step 1: Write failing registry test**

Assert a plan row with width above recipe-v2 base raises `P6HistoryRegistryError` and writes no registry.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/stage6/test_p6_history_registry.py::test_registry_rejects_plan_width_above_recipe_base_before_write -q`

- [ ] **Step 3: Implement over-base rejection**

Replace `source_status="unavailable"` for over-base rows with a validation failure before registry write.

- [ ] **Step 4: Add scanner canonical metadata**

Emit `canonical_base_width` and `formal_axis` from consolidated groups without removing legacy fields.

- [ ] **Step 5: Verify GREEN**

Run: `pytest tests/stage6/test_p6_history_registry.py tests/stage1/test_stage1_bridge.py -q`

- [ ] **Step 6: Commit**

Run: `git add framework/stage1/graph_scan.py framework/stage6/p6_history_registry_v1.py tests/stage6/test_p6_history_registry.py && git commit -m "fix: reject over-base formal candidates"`

### Task 5: Three Paper Model Integration Gate

**Files:**
- Create: `tests/integration/test_coptv2x_paper_space_reproduction.py`
- Create: `scripts/reproduce_paper_search_space.py`

**Interfaces:**
- Consumes: public Stage2 formal-axis contract.
- Produces: CPU-only reproduction proof for Pyramid, CoDriving, and F-Cooper.

- [ ] **Step 1: Write failing integration tests**

Assert Pyramid and CoDriving produce `343/686`, and F-Cooper produces `1792/3584`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/integration/test_coptv2x_paper_space_reproduction.py -q`

- [ ] **Step 3: Implement CPU reproduction CLI**

Add `scripts/reproduce_paper_search_space.py --model {pyramid,codriving,fcooper,all}` using public fixtures and bridge/adapter APIs.

- [ ] **Step 4: Verify GREEN and regression**

Run:

```bash
pytest tests/integration/test_coptv2x_paper_space_reproduction.py tests/stage1/test_stage1_bridge.py tests/stage6/test_pyramid_search_space_adapter.py -q
python scripts/reproduce_paper_search_space.py --model all
```

- [ ] **Step 5: Commit**

Run: `git add tests/integration/test_coptv2x_paper_space_reproduction.py scripts/reproduce_paper_search_space.py && git commit -m "test: reproduce CoptV2X paper search spaces"`

### Task 6: Full Public Regression and Two-Stage Review

**Files:**
- Modify only if prior test failures require fixes.

- [ ] **Step 1: Run targeted public regression**

Run: `pytest tests/stage1 tests/stage6 tests/stage5 tests/integration/test_coptv2x_paper_space_reproduction.py -q`

- [ ] **Step 2: Run compile check**

Run: `python -m compileall framework scripts tests -q`

- [ ] **Step 3: SPEC review**

Review diff against the spec and record any gaps in `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/spec-review.md`.

- [ ] **Step 4: QUALITY review**

Review code quality, file size, public compatibility, and test hygiene in `.superpowers/sdd/2026-08-24-coptv2x-scan-derived-paper-space/quality-review.md`.

- [ ] **Step 5: Commit any review fixes**

Use conventional commit messages for any fixes.
