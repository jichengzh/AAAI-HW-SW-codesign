# P6.2 Framework Search Space Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing `stage2_search_space_v1` framework output to the P6.1 CoptV2X Pyramid/H800/TVM search loop through a tested conversion layer, without falling back to the static 343x2 registry.

**Architecture:** P6.2 adds a pure adapter that turns framework search-space records into a local Pyramid structure plan, then extends the P6.1 local source-registry step to consume that plan. The P6.1 search loop remains the sole execution controller; P6.2 only changes how candidate sources are constructed before the loop starts. All real paths and generated registry artifacts remain local and Git-ignored.

**Tech Stack:** Python 3.10+, pytest, PyYAML, existing `framework.stage1_bridge.load_stage2_search_space`, P6.1 `framework.stage6.coptv2x_h800_search_v2`.

## Global Constraints

- P6.2 starts only after P6.1 offline implementation is complete and the first local P6.1 4×4 H800 validation run has been separately approved and completed successfully. An attempted or failed P6.1 run is not sufficient.
- Fixed downstream task remains Pyramid, H800, TVM only.
- Conversion output must represent complete Pyramid three-stage widths plus `q_mode`; `fp16` and `int8` remain independent candidates.
- The adapter must reject incomplete stage coverage, non-Pyramid models, non-H800 hardware, missing TVM tuned hardware candidate, non-buildable INT8 points, and unsupported quantization policies.
- No silent fallback to the static CoptV2X 343x2 grid is allowed in framework mode.
- Do not execute TensorRT, Orin, CPU, RTX 4090, network downloads, or asset discovery.
- Do not write private paths, commands, host info, candidate identifiers, raw metrics, checkpoints, logs, or run results into Git, anonymous archive outputs, or paper materials. P6 produces no public result summary.
- Use TDD for each task; commit after each task passes targeted tests.

---

## File Structure

- Create `framework/stage6/pyramid_search_space_adapter_v1.py`: pure conversion from `stage2_search_space_v1` dicts to a P6 local Pyramid structure plan.
- Create `tests/stage6/test_pyramid_search_space_adapter.py`: conversion, rejection, determinism, and no-fallback tests.
- Modify `framework/stage6/coptv2x_h800_search_v2.py`: add `candidate_source_mode` and optional framework search-space plan generation before `source_registry_step`.
- Modify `tests/stage6/test_coptv2x_h800_search.py`: controller integration tests for framework mode.
- Keep `candidate_source_mode` and the actual `stage2_search_space` path local-only in Git-ignored configuration; public files do not name either a path or a generated plan/candidate.
- Modify `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`: describe P6.1 and P6.2 as separate gates.
- Modify `docs/AAAI27_RELEASE_AUDIT.md`: keep P6 `进行中（本地）`, add P6.2 framework adapter status when implemented.
- Modify release/archive tests if they enforce explicit config/module lists.

## Task 1: Add The Pure Pyramid Structure Plan Adapter

**Files:**
- Create: `framework/stage6/pyramid_search_space_adapter_v1.py`
- Create: `tests/stage6/test_pyramid_search_space_adapter.py`

**Interfaces:**
- Produces:
  - `PyramidSearchSpaceAdapterError(ValueError)`
  - `build_pyramid_structure_plan(search_space: Mapping[str, Any]) -> dict[str, Any]`
- Output schema:

```python
{
    "schema_version": "p6_pyramid_structure_plan_v1",
    "source_schema": "stage2_search_space_v1",
    "target_model": "pyramid",
    "hardware_target": "h800",
    "execution_backend": "tvm_auto",
    "candidate_source_mode": "framework_stage2_search_space",
    "structure_count": len(structures),
    "structures": [
        {
            "width": [stage1_width, stage2_width, stage3_width],
            "q_mode": "fp16" | "int8",
            "source_point_ids": [stage1_point_id, stage2_point_id, stage3_point_id],
        },
    ],
}
```

- [ ] **Step 1: Write RED test for valid three-stage Pyramid conversion**

In `tests/stage6/test_pyramid_search_space_adapter.py`, add:

```python
from __future__ import annotations

from typing import Any

import pytest

from framework.stage6.pyramid_search_space_adapter_v1 import (
    PyramidSearchSpaceAdapterError,
    build_pyramid_structure_plan,
)


def _candidate(stage: str, widths: list[int], q_modes: list[str]) -> dict[str, Any]:
    points = []
    for width in widths:
        for q_mode in q_modes:
            points.append(
                {
                    "id": f"{stage}:w{width}:{q_mode}",
                    "width": width,
                    "quant_policy": q_mode,
                    "buildable": True,
                    "status": "active",
                }
            )
    return {
        "id": stage,
        "search_group_id": stage,
        "bucket": "pyramid_backbone",
        "dense_stage": stage,
        "width_anchors": [{"width": width} for width in widths],
        "software_points": points,
    }


def _space(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": "stage2_search_space_v1",
        "model": "pyramid_lidar",
        "hardware_target": {"name": "h800"},
        "software_candidates": [
            _candidate("stage1", [16, 32], ["fp16", "int8"]),
            _candidate("stage2", [32], ["fp16", "int8"]),
            _candidate("stage3", [64], ["fp16", "int8"]),
        ],
        "hardware_candidates": [
            {
                "id": "tvm_metaschedule_candidate",
                "backend_scope": "measured_h800_tvm",
                "hardware": "h800",
                "schedule_policy": "tuned",
            }
        ],
    }
    return {**payload, **overrides}


def test_build_pyramid_structure_plan_maps_complete_stage2_space() -> None:
    plan = build_pyramid_structure_plan(_space())

    assert plan["schema_version"] == "p6_pyramid_structure_plan_v1"
    assert plan["target_model"] == "pyramid"
    assert plan["hardware_target"] == "h800"
    assert plan["execution_backend"] == "tvm_auto"
    assert plan["structure_count"] == 4
    assert [item["width"] for item in plan["structures"]] == [
        [16, 32, 64],
        [16, 32, 64],
        [32, 32, 64],
        [32, 32, 64],
    ]
    assert [item["q_mode"] for item in plan["structures"]] == ["fp16", "int8", "fp16", "int8"]
```

- [ ] **Step 2: Write RED rejection tests**

Add:

```python
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"schema": "stage2_search_space_v0"}), "schema"),
        (lambda data: data.update({"model": "codriving"}), "model"),
        (lambda data: data.update({"hardware_target": {"name": "orin"}}), "H800"),
        (lambda data: data.update({"hardware_candidates": []}), "TVM"),
        (lambda data: data["software_candidates"].pop(), "stage"),
        (lambda data: data["software_candidates"][0]["software_points"][0].update({"status": "diagnostic_only"}), "active"),
        (lambda data: data["software_candidates"][0]["software_points"][1].update({"buildable": False}), "buildable"),
        (lambda data: data["software_candidates"][0]["software_points"][0].update({"quant_policy": "fp32"}), "quant"),
    ],
)
def test_build_pyramid_structure_plan_rejects_unmaterializable_spaces(
    mutate: Any, message: str
) -> None:
    payload = _space()
    mutate(payload)

    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_structure_plan(payload)


def test_build_pyramid_structure_plan_does_not_fall_back_to_static_grid() -> None:
    payload = _space(software_candidates=[])

    with pytest.raises(PyramidSearchSpaceAdapterError, match="stage"):
        build_pyramid_structure_plan(payload)
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest tests/stage6/test_pyramid_search_space_adapter.py -q
```

Expected: FAIL because the new module does not exist.

- [ ] **Step 4: Implement the adapter**

Create `framework/stage6/pyramid_search_space_adapter_v1.py` with:

```python
from __future__ import annotations

import copy
from collections.abc import Mapping
from itertools import product
from typing import Any


class PyramidSearchSpaceAdapterError(ValueError):
    """Raised when Stage2 search-space output cannot become Pyramid P6 candidates."""
```

Implementation rules:

1. Require `search_space["schema"] == "stage2_search_space_v1"`.
2. Require `search_space["model"] == "pyramid_lidar"` and normalize output model to `"pyramid"`.
3. Require `search_space["hardware_target"]["name"]` to be `"h800"` or begin with `"h800_"`; this accepts the bridge fixture label `h800_tvm_demo` but rejects non-H800 targets.
4. Require a hardware candidate whose `hardware` equals that same H800 label and whose remaining fields are:

```python
{
    "id": "tvm_metaschedule_candidate",
    "backend_scope": "measured_h800_tvm",
    "schedule_policy": "tuned",
}
```

5. Collect `software_candidates` for exactly `dense_stage` values `stage1`, `stage2`, and `stage3`.
6. For each stage and `q_mode` in `("fp16", "int8")`, keep points where:

```python
point["quant_policy"] == q_mode
point["buildable"] is True
point["status"] == "active"
```

7. Compute the intersection of q modes available from active/buildable points across all three stages. Reject only if that intersection is empty; this keeps `q_mode` as a whole-candidate feature while allowing a framework space that validly supports only FP16 or only INT8.
8. Build Cartesian products per q mode, sorted by `(width_tuple, q_mode)`.
9. Deep-copy only the point IDs into `source_point_ids`; do not copy private payloads.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
pytest tests/stage6/test_pyramid_search_space_adapter.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add framework/stage6/pyramid_search_space_adapter_v1.py tests/stage6/test_pyramid_search_space_adapter.py
git commit -m "feat: map Stage2 search space to Pyramid candidates"
```

## Task 2: Wire Framework Mode Into The P6.1 Controller

**Files:**
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`

**Interfaces:**
- Modifies `LocalP6CoptV2XConfig` with:
  - `candidate_source_mode: Literal["coptv2x_static_registry", "framework_stage2_search_space"]`
  - `stage2_search_space_path: Path | None`
- Extends its local input boundary:
  - for static mode: existing `gold176_rows`, `gold176_graph_features`, `capability_profiles`, `closure`
  - for framework mode: additionally require `stage2_search_space`
- In framework mode, `source_registry_step` additionally receives `{pyramid_structure_plan}`.

- [ ] **Step 1: Write RED tests for candidate source mode parsing**

Add:

```python
def test_local_contract_accepts_only_known_candidate_source_modes(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract()))
    static_contract = load_local_config(
        _write_yaml(tmp_path / "static.yaml", _local_config(candidate_source_mode="coptv2x_static_registry")), contract
    )
    framework_contract = load_local_config(
        _write_yaml(tmp_path / "framework.yaml", _local_config(candidate_source_mode="framework_stage2_search_space", stage2_search_space_path=str(tmp_path / "space.yaml"))), contract
    )

    assert static_contract.candidate_source_mode == "coptv2x_static_registry"
    assert framework_contract.candidate_source_mode == "framework_stage2_search_space"

    with pytest.raises(P6CoptV2XContractError, match="candidate_source_mode"):
        load_local_config(_write_yaml(tmp_path / "bad.yaml", _local_config(candidate_source_mode="static_fallback")), contract)
```

- [ ] **Step 2: Write RED controller integration test for framework mode**

Add:

```python
def test_run_p6_framework_mode_builds_structure_plan_before_source_registry(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        contract=contract,
        source_group_count=len(_framework_plan_structures()),
        include_stage2_search_space=True,
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        calls.append((argv[0], argv))
        if argv[1] == "local_build_registry.py":
            structure_plan_path = Path(argv[4])
            plan = json.loads(structure_plan_path.read_text(encoding="utf-8"))
            assert plan["schema_version"] == "p6_pyramid_structure_plan_v1"
            assert plan["candidate_source_mode"] == "framework_stage2_search_space"
            _write_source_registry_from_plan(Path(argv[3]), plan)
            return 0
        _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert calls[0][0] == "python"
```

`_loaded_local_config(..., contract=contract, include_stage2_search_space=True)` must use the provided framework-mode local configuration when calling `load_local_config()`, write a synthetic Stage1 partition manifest consumed through `load_stage2_search_space()`, set `stage2_search_space_path`, and change the source-registry argv from:

```python
["python", "local_build_registry.py", "{local_output_root}", "{source_registry_json}"]
```

to:

```python
[
    "python",
    "local_build_registry.py",
    "{local_output_root}",
    "{source_registry_json}",
    "{pyramid_structure_plan}",
]
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_local_contract_accepts_only_known_candidate_source_modes tests/stage6/test_coptv2x_h800_search.py::test_run_p6_framework_mode_builds_structure_plan_before_source_registry -q
```

Expected: FAIL because the local contract has no `candidate_source_mode` and the controller does not build a structure plan.

- [ ] **Step 4: Implement candidate source modes**

In `framework/stage6/coptv2x_h800_search_v2.py`:

1. Add `candidate_source_mode` and `stage2_search_space_path` to `LOCAL_KEYS` and `LocalP6CoptV2XConfig` only.
2. Accept only `"coptv2x_static_registry"` and `"framework_stage2_search_space"`.
3. In `load_local_config()`, require an absolute local `stage2_search_space_path` only when the local mode is `"framework_stage2_search_space"`; reject it in static mode.
4. Allow `{pyramid_structure_plan}` in `source_registry_step` only in framework mode.
5. Before source registry generation in framework mode:

```python
from framework.stage1_bridge import load_stage2_search_space
from framework.stage6.pyramid_search_space_adapter_v1 import build_pyramid_structure_plan

plan = build_pyramid_structure_plan(load_stage2_search_space(local.stage2_search_space_path))
plan_path = local.local_output_root / "pyramid_structure_plan.json"
plan_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

6. Pass `plan_path` as `{pyramid_structure_plan}` to `source_registry_step`; validate the returned registry against the exact plan `(width, q_mode)` identities before the Stage5 manifest expands it.
7. In static mode, reject any use of `{pyramid_structure_plan}` so static mode cannot pretend framework integration happened.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_local_contract_accepts_only_known_candidate_source_modes tests/stage6/test_coptv2x_h800_search.py::test_run_p6_framework_mode_builds_structure_plan_before_source_registry tests/stage6/test_pyramid_search_space_adapter.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_coptv2x_h800_search.py
git commit -m "feat: wire framework search space into P6 registry build"
```

## Task 3: Release Documentation And Verification For P6.2

**Files:**
- Modify: `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify: `tests/integration/test_anonymous_archive.py`
- Modify: `tests/release/test_project_handoff.py`
- Modify only if the archive test proves the pure public adapter is not included: `tools/release/anonymous_allowlist.txt`

**Interfaces:**
- Consumes:
  - P6.1 CLI and controller
  - `framework/stage6/pyramid_search_space_adapter_v1.py`
- Produces:
  - Public docs stating that P6.2 has an offline converter and still requires a separately approved real local validation run before P6 can close.

- [ ] **Step 1: Write RED archive/release tests for the P6.2 public files**

In `tests/integration/test_anonymous_archive.py`, add an expectation that the anonymous archive includes:

```python
"framework/stage6/pyramid_search_space_adapter_v1.py"
```

In `tests/release/test_project_handoff.py`, add a handoff assertion:

```python
def test_p6_framework_search_space_gate_is_documented() -> None:
    manifest = (REPOSITORY_ROOT / "docs/release-manifests/P6_H800_SEARCH_EXECUTION.md").read_text(encoding="utf-8")
    assert "framework_stage2" in manifest
    assert "不得静默回退" in manifest
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
pytest tests/integration/test_anonymous_archive.py::test_builder_includes_release_tools_required_by_archive_tests tests/release/test_project_handoff.py::test_p6_framework_search_space_gate_is_documented -q
```

Expected: FAIL until archive allowlist and docs are updated.

- [ ] **Step 3: Update docs**

In `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`, add a `P6.2 Framework Search Space Gate` section with:

```markdown
P6.2 does not replace the P6.1 search loop. It changes only the candidate-source construction step: `stage2_search_space_v1` is converted into `p6_pyramid_structure_plan_v1`, then a Git-ignored local adapter materializes that plan into the CoptV2X source registry consumed by P6.1.

Framework mode is selected only in the Git-ignored local P6 configuration; the public contract stays path-free and contains no search-space or candidate identifiers. The converter rejects non-Pyramid models, non-H800 hardware, missing tuned TVM candidates, incomplete stage1/stage2/stage3 coverage, unsupported quantization policies, and non-buildable active points. It must not silently fall back to the static 343x2 grid.
```

In `docs/AAAI27_RELEASE_AUDIT.md`, keep P6 as `进行中（本地）` and mention:

```markdown
P6.2 已提供离线框架搜索空间转换契约；真实框架来源闭环仍需在 P6.1 本地验证后单独授权执行。
```

- [ ] **Step 4: Update archive allowlist if required**

Run:

```bash
pytest tests/integration/test_anonymous_archive.py::test_builder_includes_release_tools_required_by_archive_tests -q
```

If it fails because the new pure module is excluded, add exactly this public-safe path to `tools/release/anonymous_allowlist.txt`:

```text
framework/stage6/pyramid_search_space_adapter_v1.py
```

- [ ] **Step 5: Run GREEN release tests**

Run:

```bash
pytest tests/integration/test_anonymous_archive.py::test_builder_includes_release_tools_required_by_archive_tests tests/release/test_project_handoff.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py tools/release/anonymous_allowlist.txt
git commit -m "docs: document P6 framework search-space gate"
```

## Task 4: Full P6.2 Verification Gate

**Files:**
- Modify only if targeted tests reveal a defect:
  - `framework/stage6/pyramid_search_space_adapter_v1.py`
  - `framework/stage6/coptv2x_h800_search_v2.py`
  - affected tests/docs

**Interfaces:**
- Consumes all P6.1 and P6.2 public interfaces.
- Produces a branch where both P6.1 static mode and P6.2 framework mode pass offline CI-quality tests.

- [ ] **Step 1: Run P6.2 targeted tests**

Run:

```bash
pytest tests/stage6/test_pyramid_search_space_adapter.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Stage1 bridge compatibility tests**

Run:

```bash
pytest tests/stage1/test_stage1_bridge.py tests/stage6/test_pyramid_search_space_adapter.py -q
```

Expected: PASS.

- [ ] **Step 3: Run release boundary tests**

Run:

```bash
pytest tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py tests/release/test_public_execution_surface.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full quality suite**

Run:

```bash
pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

Expected: PASS with total coverage at or above 80%.

- [ ] **Step 5: Run stale fallback and private-output scans**

Run:

```bash
rg -n "static fallback|fallback to static|public-summary|h800_search_execution_v1|trt_engine|TensorRT|RTX 4090|CPU" framework/stage6 tools/release configs/execution tests/stage6 tests/release docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md
rg -n "/home/|/mnt/|ssh|hostname|checkpoint|raw log|candidate-[0-9]" configs/execution docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md framework/stage6 tests/stage6
```

Expected:

- No executable P6.2 static fallback language.
- No public summary path.
- TensorRT/RTX/CPU only appear in exclusion language if present.
- No private absolute paths, SSH details, raw logs, or public candidate IDs in committed public files.

- [ ] **Step 6: Review diff**

Run:

```bash
git diff --check
git diff --stat
```

Expected: no whitespace errors; diff limited to P6.2 adapter, controller mode extension, docs, and tests.

- [ ] **Step 7: Commit any verification fixes**

If Step 1-6 required fixes, run:

```bash
git add framework/stage6/pyramid_search_space_adapter_v1.py framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_pyramid_search_space_adapter.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md
git commit -m "fix: close P6 framework search-space verification gaps"
```

If no fixes are needed, do not create an empty commit.

- [ ] **Step 8: Stop before a framework-mode real H800 run**

Do not run the real framework-mode H800 job automatically. After P6.1 has been run locally and the user explicitly approves P6.2 execution, use the same CLI with a Git-ignored local config that provides `stage2_search_space` and a source-registry adapter accepting `{pyramid_structure_plan}`:

```bash
python tools/release/run_p6_h800_search.py \
  --contract configs/execution/p6_h800_search.example.yaml \
  --local-config configs/local/p6_h800_search.local.yaml \
  --code-revision "$(git rev-parse --short=12 HEAD)"
```

Expected for an approved framework-mode run: `completed` after 4 rounds, with framework-derived Pyramid candidates passing through the same P6.1 loop and all outputs remaining Git-ignored.

## Self-Review Checklist

- Spec coverage: P6.2 converts `stage2_search_space_v1` into complete Pyramid width + `q_mode` candidates, rejects non-materializable inputs, does not fall back to static grid, and reuses the P6.1 loop.
- Placeholder scan: no `TBD`, no unqualified “add validation”, no “write tests for the above”, no undefined implementation task.
- Type consistency: the local `candidate_source_mode`, `p6_pyramid_structure_plan_v1`, and `{pyramid_structure_plan}` are defined before use and used consistently by tests, controller, local config, and docs.
