# P6 历史执行根归一桥接 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将分散的私有历史输入归一为可验证的 P6 私有执行根，并用真实 Stage1 扫描产物驱动动态 Stage2 候选空间。

**Architecture:** 一个通用 Stage1 CLI 在私有已激活环境中调用既有 `graph_scan`，只输出 P6 所需 JSON manifest。一个 normalizer 从 Git 忽略 source-map 复制四个小型输入、写唯一 source registry-v1 与 legacy locator；bootstrap 改为以私有 runner template 的显式组件路径消除历史快照重复标记的歧义。

**Tech Stack:** Python 3.10+、PyYAML、标准库 JSON/pathlib/subprocess、pytest、ruff。

**Spec:** `docs/superpowers/specs/2026-08-20-p6-history-normalization-bridge-design.md`

**Current status:** implementation complete; H800 private preflight pending. Task 4 added the offline normalizer→provision→Stage1 manifest→dynamic Stage2 plan→registry-v2 gate and stops before measurement adapter execution. Task 5 remains responsible for the real private H800 preflight.

## Global Constraints

- 不在 tracked 文件中写入真实设备编号、私有路径、资产 ID、候选 ID、checkpoint、指标或结果。
- P6 的公开口径保持 H800/TVM/Pyramid、Gold176、四轮、每轮四条、共 16 次实际测量。
- Stage1 由真实 `framework.stage1.graph_scan` 生成 `stage1_partition_manifest_v1` JSON；不得使用 demo manifest 或缓存候选伪装扫描。
- Stage2 自主从 Stage1 产物导出 active/buildable 的 FP16/INT8 点；不得固定候选数。
- 私有 source-map、runner template、binding、local YAML、registry、日志与结果必须在 Git 忽略或仓库外位置。
- 输入、组件、根目录与输出路径一律拒绝软链接、越界与不唯一选择；预检失败不得启动 Stage5 测量。
- 不下载、安装或重建 TVM；只能由私有 activation argv 启用已安装环境。

---

### Task 1: 真实 Stage1 JSON bridge

**Files:**

- Create: `framework/stage6/p6_stage1_bridge_v1.py`
- Create: `tools/release/build_p6_stage1_manifest.py`
- Test: `tests/stage6/test_p6_stage1_bridge.py`
- Test: `tests/release/test_build_p6_stage1_manifest.py`

**Interfaces:**

- Produces: `build_p6_stage1_partition_manifest(output_path: Path, hardware_path: Path, device: str, environment: Mapping[str, str], scanner: Callable[..., Mapping[str, Any]]) -> dict[str, Any]`.
- Produces: CLI `build_p6_stage1_manifest.py --hardware <absolute> --output <absolute> --device <cuda-device> --stage1-repo-root <absolute> --heal-root <absolute> --heal-checkpoint-root <absolute>`.
- Consumed by: private `scan-private` launcher, receiving `{stage1_partition_manifest}` and `{local_output_root}` from existing P6 bootstrap.

- [ ] **Step 1: Write failing bridge tests**

```python
def test_bridge_calls_scanner_and_writes_only_valid_json_manifest(tmp_path: Path) -> None:
    manifest = valid_stage1_manifest()
    result = build_p6_stage1_partition_manifest(
        tmp_path / "manifest.json", hardware_path, "cuda", {}, lambda *_: manifest
    )
    assert result == manifest
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest

def test_bridge_rejects_partial_or_wrong_model_without_output(tmp_path: Path) -> None:
    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(tmp_path / "manifest.json", hardware_path, "cuda", {}, lambda *_: {"scan_status": "partial"})
    assert not (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_stage1_bridge.py tests/release/test_build_p6_stage1_manifest.py -q`

Expected: FAIL because the bridge module and CLI do not exist.

- [ ] **Step 3: Implement the bridge and CLI**

```python
def build_p6_stage1_partition_manifest(...):
    manifest = scanner("pyramid_lidar", hardware_path, device, environment)
    if not _is_p6_stage1_manifest(manifest):
        raise P6Stage1BridgeError("stage1_scan_invalid")
    _atomic_write_json(output_path, manifest)
    return copy.deepcopy(manifest)
```

The production scanner adapter sets `STAGE1_REPO_ROOT`, `HEAL_ROOT`, and `HEAL_CKPT_ROOT` before importing existing adapters, loads the tracked H800 capability YAML, calls `graph_scan.scan(get_adapter("pyramid_lidar"), ...)`, and serializes JSON. It rejects non-absolute paths, symlinks and missing parents; all CLI failures print only `stage1_scan_invalid`.

- [ ] **Step 4: Run GREEN**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_stage1_bridge.py tests/release/test_build_p6_stage1_manifest.py -q && python -m ruff check framework/stage6/p6_stage1_bridge_v1.py tools/release/build_p6_stage1_manifest.py tests/stage6/test_p6_stage1_bridge.py tests/release/test_build_p6_stage1_manifest.py && git diff --check`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add framework/stage6/p6_stage1_bridge_v1.py tools/release/build_p6_stage1_manifest.py tests/stage6/test_p6_stage1_bridge.py tests/release/test_build_p6_stage1_manifest.py
git commit -m "feat: bridge P6 Stage1 scan to JSON manifest"
```

### Task 2: 私有历史输入与基线 source-contract normalizer

**Files:**

- Create: `framework/stage6/p6_history_normalization_v1.py`
- Create: `tools/release/normalize_p6_history_root.py`
- Test: `tests/stage6/test_p6_history_normalization.py`
- Test: `tests/release/test_normalize_p6_history_root.py`

**Interfaces:**

- Produces: `normalize_history_inputs(source_map: Mapping[str, Any], history_root: Path, private_dir: Path) -> dict[str, Path]`.
- Private source-map schema: `p6_history_normalization_source_v1`, with exact `asset_paths`, `input_sources`, `source_contract`, `dynamic_materialization_recipe`, and `history_root` keys.
- Produces canonical input filenames, one `stage5_candidate_source_registry_v1`, and a legacy `p6_h800_coptv2x_local_v2` locator.
- Consumed by: existing `provision_p6_full_chain_local_config.py`.

- [ ] **Step 1: Write failing normalizer tests**

```python
def test_normalizer_copies_four_json_inputs_and_writes_one_registry(tmp_path: Path) -> None:
    paths = normalize_history_inputs(valid_private_source_map(tmp_path), history_root, private_dir)
    assert set(paths) == {"gold176_rows", "gold176_graph_features", "capability_profiles", "closure", "registry", "legacy"}
    assert json.loads(paths["registry"].read_text())["schema_version"] == "stage5_candidate_source_registry_v1"

@pytest.mark.parametrize("mutator", [outside_root, symlink_input, wrong_recipe, colliding_template])
def test_normalizer_fails_without_writing_partial_private_root(...):
    with pytest.raises(P6HistoryNormalizationError, match="history_normalization_invalid"):
        normalize_history_inputs(mutator(valid_private_source_map(tmp_path)), history_root, private_dir)
    assert not private_dir.exists()
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py -q`

Expected: FAIL because the normalizer module and CLI do not exist.

- [ ] **Step 3: Implement fail-closed source-map normalization**

```python
def normalize_history_inputs(source_map, history_root, private_dir):
    canonical = _validate_private_source_map(source_map, history_root)
    staged = _make_private_staging_directory(private_dir)
    _copy_json_inputs(canonical["input_sources"], staged / "inputs")
    registry = _build_registry_v1(canonical["source_contract"], canonical["dynamic_materialization_recipe"])
    _atomic_write_json(staged / "registry" / "candidate-source-registry.json", registry)
    _atomic_write_yaml(staged / "legacy.local.yaml", _legacy_locator(canonical, staged))
    return _publish_staging_directory(staged, private_dir)
```

Validation requires an existing Git root; source paths below it without symlink components; all four JSON inputs; the three asset labels; one Pyramid/TVM source contract; a complete v1 dynamic recipe; and relative, non-colliding output templates. Contract data cannot contain metric, terminal, cache or result fields. The CLI reads only an absolute private map and prints `history_normalization_invalid` on failure.

- [ ] **Step 4: Run GREEN**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py tests/stage6/test_p6_history_registry.py -q && python -m ruff check framework/stage6/p6_history_normalization_v1.py tools/release/normalize_p6_history_root.py tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py && git diff --check`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/stage6/p6_history_normalization_v1.py tools/release/normalize_p6_history_root.py tests/stage6/test_p6_history_normalization.py tests/release/test_normalize_p6_history_root.py
git commit -m "feat: normalize private P6 history inputs"
```

### Task 3: bootstrap 的显式历史组件绑定

**Files:**

- Modify: `framework/stage6/p6_full_chain_bootstrap_v1.py:125-191,335-379`
- Test: `tests/stage6/test_p6_full_chain_bootstrap.py`
- Test: `tests/release/test_provision_p6_full_chain_local_config.py`

**Interfaces:**

- Consumes: runner template `execution_interface` from existing `p6_history_runner_template_v1`.
- Produces: explicit `component_paths` passed to unchanged `build_history_binding()`.
- Preserves: `discover_history_binding()` retains strict auto-discovery for callers without a runner template.

- [ ] **Step 1: Write failing bootstrap tests**

```python
def test_full_chain_provision_uses_template_component_paths_when_history_contains_archived_duplicates(...):
    create_duplicate_marker(history_root / "archived")
    result = run_provision(valid_args_with_template_paths(history_root))
    assert result.returncode == 0

def test_full_chain_provision_rejects_template_component_outside_history_root(...):
    template = valid_template(history_root)
    template["execution_interface"]["controller"]["argv"] = [str(tmp_path / "outside" / MARKER)]
    assert run_provision(args).stderr == "history_root_ambiguous\n"
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py -q`

Expected: the first test fails with ambiguous marker selection.

- [ ] **Step 3: Implement explicit component resolution**

```python
template = _load_runner_template(runner_template)
root = _unique_common_history_root(locator)
component_paths = _template_component_paths(root, template["execution_interface"])
interface = _render_runner_interface(root, template["execution_interface"])
binding = build_history_binding(root, component_paths=component_paths, execution_interface=interface, ...)
```

`_template_component_paths` requires the exact four marker basenames, absolute resolved non-symlink executable files below `root`, and equality with rendered controller/materializer/performance/finalizer argv. It never silently falls back to recursive discovery.

- [ ] **Step 4: Run GREEN**

Run: `PYTHONPATH=. pytest tests/stage6/test_p6_history_binding.py tests/stage6/test_p6_full_chain_bootstrap.py tests/stage6/test_p6_history_normalization.py tests/release/test_provision_p6_full_chain_local_config.py -q && python -m ruff check framework/stage6/p6_full_chain_bootstrap_v1.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py && git diff --check`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/stage6/p6_full_chain_bootstrap_v1.py tests/stage6/test_p6_full_chain_bootstrap.py tests/release/test_provision_p6_full_chain_local_config.py
git commit -m "feat: bind P6 history components from private template"
```

### Task 4: lifecycle integration、公开文档与离线门禁

**Files:**

- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/release/test_run_p6_h800_search.py`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify: `docs/superpowers/specs/2026-08-20-p6-history-normalization-bridge-design.md`
- Modify: `docs/superpowers/plans/2026-08-20-p6-history-normalization-bridge.md`

**Interfaces:**

- Consumes: Task 1 Stage1 bridge, Task 2 normalized private root, Task 3 explicit bootstrap components.
- Produces: offline proof that provision → Stage1 manifest → Stage2 dynamic plan → registry-v2 succeeds before the measurement adapter is called.

- [x] **Step 1: Write the failing no-measurement full-chain test**

```python
def test_normalized_private_root_reaches_dynamic_stage2_and_registry_without_measurement(...):
    state = run_p6_preflight(normalized_private_root, fake_stage1_scanner, fake_registry_adapter)
    assert state.stage1_schema == "stage1_partition_manifest_v1"
    assert state.eligible_candidate_count >= 16
    assert state.measurement_calls == 0
```

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=. pytest tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -q`

Expected: FAIL before Tasks 1–3 integration is wired.

- [x] **Step 3: Wire existing test seams and update documentation**

Use the existing command-runner injection; do not introduce a second controller. Document P6 as `implementation complete; H800 private preflight pending` until the remote preflight succeeds. Do not record private sources, GPU identities, paths or outputs.

- [x] **Step 4: Run complete offline gates**

Run:

```bash
PYTHONPATH=. python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
PYTHONPATH=. pytest tests/release tests/integration/test_anonymous_archive.py -q
python -m ruff check framework tools scripts tests
git diff --check
```

Expected: all tests pass, coverage is at least 80%, and the tracked tree contains no private runtime detail.

- [ ] **Step 5: Commit**

```bash
git add tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py docs/AAAI27_RELEASE_AUDIT.md docs/superpowers/specs/2026-08-20-p6-history-normalization-bridge-design.md docs/superpowers/plans/2026-08-20-p6-history-normalization-bridge.md
git commit -m "test: cover P6 history normalization preflight"
```

### Task 5: H800 私有根构建与真实预检

**Files:**

- Create only in a Git-ignored or repository-external H800 location: private source-map, private runner template, normalized root, binding/local YAML, output directory.
- Do not modify tracked files.

**Interfaces:**

- Consumes: Tasks 1–3, `tools/release/provision_p6_full_chain_local_config.py`, and the tracked public contract.
- Produces: ignored binding/config pair and a Stage1/Stage2/registry preflight record with zero measurement calls.

- [ ] **Step 1: Deploy the reviewed branch without altering existing historical sources**

Create a clean H800 checkout of the reviewed branch under an approved private working location. Verify revision, clean status, installed Python and existing TVM activation command. Do not install packages or rebuild TVM.

- [ ] **Step 2: Build the private source-map from explicit structured historical evidence**

Select the frozen source recorded by the existing deployment manifest; select the Gold176 file with exactly 176 rows; pair it with compatible graph/profile/Stage4 closure inputs; point to the unique Stage7 baseline registry and dynamic recipe. Reject ambiguity rather than selecting by newest filename. Store this map outside the public checkout.

- [ ] **Step 3: Materialize the normalized private root and private runner**

Run `git init` only in the new private root, never in the existing historical source tree. Use the normalizer to copy only four JSON inputs and write registry/legacy locator. Create private runner wrappers that invoke the already-installed conda/TVM environment and Task 1 CLI; do not copy model, training data, checkpoint, or result directories.

- [ ] **Step 4: Generate pair and run preflight only**

Run `provision_p6_full_chain_local_config.py`, then run the public controller in preflight mode that executes Stage1 and registry materialization but intercepts measurement. Require a valid Stage1 JSON manifest, dynamically generated Stage2 plan with at least 16 eligible rows, registry-v2/plan identity equality and zero training/measurement processes.

- [ ] **Step 5: Write an ignored redacted report**

Record only statuses, schema versions, candidate count threshold, revision label and whether measurement was zero. Scan the tracked repository to confirm no private file or value entered Git.

### Task 6: H800 四轮真实闭环

**Files:**

- Create only Git-ignored/external H800 output artifacts.
- Modify: `docs/AAAI27_RELEASE_AUDIT.md` only after all four rounds succeed and the user approves the public status update.

**Interfaces:**

- Consumes: Task 5 validated binding/config and current private three-H800 policy.
- Produces: four completed rounds, 16 actual feedback rows, four cost-model refits and a redacted private run summary.

- [ ] **Step 1: Recheck private GPU admission and runtime environment**

Use the binding policy, not hard-coded indices, to check the same three H800 UUIDs/model/occupancy immediately before launch. Confirm the private activation argv exposes the existing TVM environment. Any mismatch stops before measurement.

- [ ] **Step 2: Start exactly one controller run**

Invoke `tools/release/run_p6_h800_search.py` with the tracked public contract, generated ignored local config and explicit code revision. Do not run multiple controllers or a manual Stage5 replacement.

- [ ] **Step 3: Monitor redacted state transitions**

Verify Gold176 cold start → selection of four candidates → actual feedback → cost-model refit, repeated through rounds 0–3. On failure preserve ignored diagnostics and stop; do not manually rerun a partial round.

- [ ] **Step 4: Verify completion**

Require `completed_rounds == 4`, `measured_candidate_count == 16`, valid pre/post GPU admission and all feedback receipt/barrier checks. Compare only aggregate metrics against the paper baseline in a Git-ignored report. Do not claim paper reproduction in public documentation until evidence has been reviewed.
