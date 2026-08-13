# P6.1 CoptV2X TVM Search Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current four-arm H800 runner with a Pyramid/H800/TVM CoptV2X single-genome search controller that executes 4 rounds, 4 candidates per round, with Gold176 cold-start refitting and atomic feedback.

**Architecture:** Public code owns only the fixed P6.1 contract, offline state machine, Stage5 single-target API calls, request/result validation, and synthetic tests. Local ignored configuration owns asset paths and argv-only adapters for source registry generation and real batch measurement. No public summary is produced; true H800 outputs remain under the ignored local output root.

**Tech Stack:** Python 3.10+, pytest, PyYAML, existing `framework.stage5.single_target_search_v2`, existing `framework.stage5.production_search_v1`, `subprocess.run(..., shell=False)` in the CLI.

## Global Constraints

- Fixed task: Pyramid, H800, TVM only.
- Candidate genome: full Pyramid three-stage width plus `q_mode`; `fp16` and `int8` are independent candidates.
- Budget: exactly 4 rounds, 4 candidates per round, 16 total measured slots.
- Cold start: reuse local Gold176 rows and graph features; retrain the cost model; do not remeasure Gold176.
- Each selected candidate must run pruning, fine-tuning, checkpoint selection, ONNX export, TVM compile/run, and measure `latency_ms`, `energy_j`, `ap30`, `ap50`, `ap70`.
- Cost model selection targets are `latency_ms`, `energy_j`, and `ap70`; `ap30` and `ap50` are required evidence but not optimization heads.
- Do not execute TensorRT, Orin, CPU, RTX 4090, network downloads, or asset discovery.
- Do not write private paths, commands, host info, candidate identifiers, raw metrics, checkpoints, logs, or run results into Git, public summaries, anonymous archive outputs, or paper materials.
- P6 remains `进行中（本地）` after this local run until a later closure review.
- Use TDD for each task; commit after each task passes targeted tests.
- P6.2’s framework-search-space adapter is explicitly out of this implementation sequence. It begins only after a separately approved P6.1 local 4×4 H800 run; see `2026-08-13-p6-2-framework-search-space-integration.md`.

---

## File Structure

- Rename and rewrite `framework/stage6/h800_search_execution_v1.py` to `framework/stage6/coptv2x_h800_search_v2.py`: P6.1 contract parser, local config parser, search loop, request/result validation, local record writes, and stable failure codes.
- Modify `framework/stage6/__init__.py` only if it exports a stale four-arm controller symbol; otherwise leave it unchanged.
- Modify `tools/release/run_p6_h800_search.py`: keep the public CLI path but switch it to the v2 controller and remove `--public-summary`.
- Modify `configs/execution/p6_h800_search.example.yaml`: public-safe v2 contract for Pyramid/H800/TVM and fixed 4x4 budget.
- Modify `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`: document P6.1 local-only execution semantics and no public summary.
- Modify `docs/AAAI27_RELEASE_AUDIT.md`: align the P6 row with the approved P6.1 scope and “进行中（本地）” status.
- Modify `docs/superpowers/plans/2026-08-11-p6-h800-search-execution.md`: add a one-line supersession notice pointing to this plan.
- Rename and rewrite `tests/stage6/test_h800_search_execution.py` to `tests/stage6/test_coptv2x_h800_search.py`: offline unit/integration coverage for contract, local config, and full four-round loop.
- Modify `tests/release/test_run_p6_h800_search.py`: black-box CLI tests using `subprocess.run()`.
- Modify `tests/release/test_project_handoff.py` and `tests/integration/test_anonymous_archive.py`; modify any additional release-surface test only if it names the old module or public summary.

## Task 1: Replace Public And Local P6 Contracts

**Files:**
- Rename: `framework/stage6/h800_search_execution_v1.py` -> `framework/stage6/coptv2x_h800_search_v2.py`
- Rename: `tests/stage6/test_h800_search_execution.py` -> `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `framework/stage6/__init__.py`
- Modify: `configs/execution/p6_h800_search.example.yaml`

**Interfaces:**
- Produces:
  - `P6CoptV2XContractError(ValueError)`
  - `RegisteredAsset(label: str, version: str, license_status: str)`
  - `PublicP6CoptV2XContract(search_id: str, target: str, target_model: str, execution_backend: str, seed: int, sample_budget: int, batch_size: int, round_count: int, configuration_label: str, candidate_space_label: str, assets: tuple[RegisteredAsset, ...], metric_names: tuple[str, ...])`
  - `LocalExecutionStep(name: str, argv: tuple[str, ...])`
  - `LocalP6CoptV2XConfig(asset_paths: Mapping[str, Path], local_input_paths: Mapping[str, Path], source_registry_step: LocalExecutionStep, measurement_step: LocalExecutionStep, local_output_root: Path)`
  - `load_public_contract(path: Path) -> PublicP6CoptV2XContract`
  - `load_local_config(path: Path, contract: PublicP6CoptV2XContract) -> LocalP6CoptV2XConfig`

- Consumes:
  - YAML public config with `schema_version: p6_h800_coptv2x_search_contract_v2`
  - YAML local config with `schema_version: p6_h800_coptv2x_local_v2`

- [ ] **Step 1: Rename files and update imports in the renamed test to point at the future v2 module**

Run:

```bash
git mv framework/stage6/h800_search_execution_v1.py framework/stage6/coptv2x_h800_search_v2.py
git mv tests/stage6/test_h800_search_execution.py tests/stage6/test_coptv2x_h800_search.py
```

In `tests/stage6/test_coptv2x_h800_search.py`, replace imports from `framework.stage6.h800_search_execution_v1` with:

```python
from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import (
    LocalP6CoptV2XConfig,
    P6CoptV2XContractError,
    PublicP6CoptV2XContract,
    load_local_config,
    load_public_contract,
)
```

- [ ] **Step 2: Write RED tests for the fixed v2 public contract**

Replace the old `_public_contract()` helper with:

```python
def _public_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_coptv2x_search_contract_v2",
        "search_id": "p6-pyramid-h800-tvm",
        "target": "h800",
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": "p6-pyramid-h800-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    return {**contract, **overrides}
```

Add these tests:

```python
def test_load_public_contract_requires_fixed_pyramid_h800_tvm_budget(tmp_path: Path) -> None:
    invalid_contracts = [
        _public_contract(target="orin"),
        _public_contract(target_model="codriving"),
        _public_contract(execution_backend="trt_engine"),
        _public_contract(sample_budget=15),
        _public_contract(batch_size=2),
        _public_contract(round_count=5),
    ]

    for index, payload in enumerate(invalid_contracts):
        with pytest.raises(P6CoptV2XContractError):
            load_public_contract(_write_yaml(tmp_path / f"contract-{index}.yaml", payload))


def test_load_public_contract_accepts_the_public_example() -> None:
    contract = load_public_contract(
        REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
    )

    assert isinstance(contract, PublicP6CoptV2XContract)
    assert contract.target == "h800"
    assert contract.target_model == "pyramid"
    assert contract.execution_backend == "tvm_auto"
    assert (contract.round_count, contract.batch_size, contract.sample_budget) == (4, 4, 16)
    assert contract.metric_names == ("latency_ms", "energy_j", "ap30", "ap50", "ap70")


@pytest.mark.parametrize("field", ["max_rounds", "public_summary", "summary_path"])
def test_load_public_contract_rejects_old_public_summary_and_round_keys(
    tmp_path: Path, field: str
) -> None:
    payload = _public_contract()
    payload[field] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", payload))
```

- [ ] **Step 3: Write RED tests for the v2 local config boundary**

Add a new local helper:

```python
def _local_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    output_root = tmp_path / "private-output"
    payload = {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            "training-data": str(tmp_path / "training-data"),
            "model-init": str(tmp_path / "model-init"),
            "toolchain": str(tmp_path / "toolchain"),
        },
        "local_input_paths": {
            "gold176_rows": str(tmp_path / "gold176_rows.json"),
            "gold176_graph_features": str(tmp_path / "gold176_graph_features.json"),
            "capability_profiles": str(tmp_path / "capability_profiles.json"),
            "closure": str(tmp_path / "closure.json"),
        },
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                "python",
                "local_build_registry.py",
                "{local_output_root}",
                "{source_registry_json}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                "python",
                "local_measure.py",
                "{measurement_request}",
                "{feedback_json}",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(output_root),
    }
    return {**payload, **overrides}
```

Add:

```python
def test_load_local_config_requires_source_and_measurement_steps(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()

    loaded = load_local_config(_write_yaml(tmp_path / "local.yaml", _local_config(tmp_path)), contract)

    assert isinstance(loaded, LocalP6CoptV2XConfig)
    assert loaded.source_registry_step.name == "build_source_registry"
    assert loaded.measurement_step.name == "measure_batch"
    assert loaded.local_output_root == tmp_path / "private-output"


@pytest.mark.parametrize(
    "key",
    ["candidate_registry", "measurements", "result_path_template", "steps", "result_step"],
)
def test_load_local_config_rejects_legacy_stage5_input_and_result_keys(
    tmp_path: Path, key: str
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    payload = _local_config(tmp_path)
    payload[key] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_local_config(_write_yaml(tmp_path / "local.yaml", payload), contract)


def test_load_local_config_rejects_shell_or_unknown_template_tokens(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    shell_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["bash", "-c", "private", "{measurement_request}"],
        },
    )
    token_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["python", "local_measure.py", "{candidate_id}"],
        },
    )

    with pytest.raises(P6CoptV2XContractError, match="shell executable"):
        load_local_config(_write_yaml(tmp_path / "shell.yaml", shell_payload), contract)
    with pytest.raises(P6CoptV2XContractError, match="template"):
        load_local_config(_write_yaml(tmp_path / "token.yaml", token_payload), contract)
```

- [ ] **Step 4: Run RED tests**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_load_public_contract_requires_fixed_pyramid_h800_tvm_budget tests/stage6/test_coptv2x_h800_search.py::test_load_local_config_requires_source_and_measurement_steps -q
```

Expected: FAIL because the renamed module still exposes old schema names, old `max_rounds`, old four-arm fields, and `H800SearchContractError` instead of v2 names.

- [ ] **Step 5: Implement minimal v2 contract parsing**

In `framework/stage6/coptv2x_h800_search_v2.py`, delete old summary dataclasses and four-arm constants. Define:

```python
PUBLIC_SCHEMA_VERSION = "p6_h800_coptv2x_search_contract_v2"
LOCAL_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
METRIC_NAMES = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
FIXED_TARGET = "h800"
FIXED_MODEL = "pyramid"
FIXED_BACKEND = "tvm_auto"
FIXED_SAMPLE_BUDGET = 16
FIXED_BATCH_SIZE = 4
FIXED_ROUND_COUNT = 4
LOCAL_INPUT_NAMES = frozenset(
    {"gold176_rows", "gold176_graph_features", "capability_profiles", "closure"}
)
ALLOWED_TEMPLATE_TOKENS = frozenset(
    {
        "{local_output_root}",
        "{source_registry_json}",
        "{measurement_request}",
        "{feedback_json}",
        "{round_output_root}",
    }
)
PUBLIC_KEYS = frozenset(
    {
        "schema_version",
        "search_id",
        "target",
        "target_model",
        "execution_backend",
        "seed",
        "sample_budget",
        "batch_size",
        "round_count",
        "configuration_label",
        "candidate_space_label",
        "metric_names",
        "assets",
    }
)
LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "local_input_paths",
        "source_registry_step",
        "measurement_step",
        "local_output_root",
    }
)
```

Keep the existing public string validation helpers, shell executable rejection, `MappingProxyType` conversion, YAML loading, path normalization, and UTF-8/YAML failure handling, but update names to raise `P6CoptV2XContractError`.

Implement local step validation as:

```python
def _load_step(payload: Mapping[str, object], *, expected_name: str, required_tokens: set[str]) -> LocalExecutionStep:
    if set(payload) != {"name", "argv"}:
        raise P6CoptV2XContractError("step keys invalid")
    if payload["name"] != expected_name:
        raise P6CoptV2XContractError("step name invalid")
    argv = _validate_argv(payload["argv"])
    present = {item for arg in argv for item in ALLOWED_TEMPLATE_TOKENS if item in arg}
    if present != required_tokens:
        raise P6CoptV2XContractError("template tokens invalid")
    return LocalExecutionStep(name=expected_name, argv=argv)
```

For `source_registry_step`, require `{"{local_output_root}", "{source_registry_json}"}`. For `measurement_step`, require `{"{measurement_request}", "{feedback_json}", "{round_output_root}"}`.

- [ ] **Step 6: Update the public example YAML**

Set `configs/execution/p6_h800_search.example.yaml` to exactly:

```yaml
schema_version: p6_h800_coptv2x_search_contract_v2
search_id: p6-pyramid-h800-tvm
target: h800
target_model: pyramid
execution_backend: tvm_auto
seed: 73
sample_budget: 16
batch_size: 4
round_count: 4
configuration_label: p6-pyramid-h800-tvm
candidate_space_label: coptv2x-pyramid-width-grid-v1
metric_names:
  - latency_ms
  - energy_j
  - ap30
  - ap50
  - ap70
assets:
  - label: training-data
    version: local-cleared-v1
    license_status: cleared
  - label: model-init
    version: local-cleared-v1
    license_status: cleared
  - label: toolchain
    version: h800-tvm-local-v1
    license_status: cleared
```

- [ ] **Step 7: Run GREEN tests for contract parsing**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_load_public_contract_requires_fixed_pyramid_h800_tvm_budget tests/stage6/test_coptv2x_h800_search.py::test_load_public_contract_accepts_the_public_example tests/stage6/test_coptv2x_h800_search.py::test_load_local_config_requires_source_and_measurement_steps tests/stage6/test_coptv2x_h800_search.py::test_load_local_config_rejects_legacy_stage5_input_and_result_keys tests/stage6/test_coptv2x_h800_search.py::test_load_local_config_rejects_shell_or_unknown_template_tokens -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add framework/stage6/coptv2x_h800_search_v2.py framework/stage6/__init__.py tests/stage6/test_coptv2x_h800_search.py configs/execution/p6_h800_search.example.yaml
git add -u framework/stage6/h800_search_execution_v1.py tests/stage6/test_h800_search_execution.py
git commit -m "feat: replace P6 H800 contract with CoptV2X TVM task"
```

## Task 2: Build The Gold176 Cold-Start And Four-Round State Machine

**Files:**
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`

**Interfaces:**
- Consumes from Task 1:
  - `PublicP6CoptV2XContract`
  - `LocalP6CoptV2XConfig`
  - `LocalExecutionStep`
- Produces:
  - `P6CoptV2XRunState(schema_version: str, status: Literal["completed", "failed"], completed_rounds: int, measured_candidate_count: int, failure_code: str | None, local_state_path: Path)`
  - `run_p6_coptv2x_search(contract: PublicP6CoptV2XContract, local: LocalP6CoptV2XConfig, code_revision: str, command_runner: CommandRunner) -> P6CoptV2XRunState`
  - Internal `_build_search_task(contract: PublicP6CoptV2XContract, profile: Mapping[str, Any]) -> SearchTask`

- [ ] **Step 1: Write RED synthetic fixtures for Gold176 and the source registry**

In `tests/stage6/test_coptv2x_h800_search.py`, create helpers:

```python
def _profile() -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )


def _graph(group_id: str, width: list[int]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "conv_count": 27,
        "conv_macs": float(width[0] * width[1] * width[2]),
        "group_conv_count": 3,
    }


def _gold176() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(176):
        width = [16 + (index % 7) * 8, 32 + (index % 8) * 8, 64 + (index % 9) * 8]
        group_id = f"gold-{index:03d}"
        q_mode = "int8" if index % 2 else "fp16"
        graphs.append(_graph(group_id, width))
        rows.append(
            {
                "manifest_job_id": f"{group_id}|q={q_mode}|profile=h800-tvm-auto",
                "row_id": f"{group_id}|q={q_mode}|profile=h800-tvm-auto",
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "dispatch_key": "tvm_auto",
                "capability_profile_id": "h800-tvm-auto",
                "q_mode": q_mode,
                "latency_ms": 2.0 + index * 0.01,
                "energy_j": 0.5 + index * 0.005,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70 - index * 0.0001,
                "terminal_status": "measured_success_gold",
                "training_source": "initial_coldstart",
            }
        )
    return rows, graphs
```

For source registry lineage, define:

```python
def _source_group(group_id: str, width: list[int]) -> dict[str, Any]:
    evidence_sha = hashlib.sha256(f"source:{group_id}".encode()).hexdigest()
    source_contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "artifact_id": f"fixture-{group_id}",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
    }
    contract_sha = hashlib.sha256(
        json.dumps(source_contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "source_contract": source_contract,
        "source_contract_sha256": contract_sha,
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "graph_features": _graph(group_id, width),
    }
```

- [ ] **Step 1b: Add complete local fixture helpers used by the loop tests**

Add these helpers after `_source_group()`:

```python
def _closure() -> dict[str, Any]:
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 176},
        "frozen_holdout": {"groups": []},
    }


def _write_source_registry(path: Path, *, count: int) -> None:
    groups = [
        _source_group(
            f"candidate-{index:03d}",
            [16 + (index // 49) * 8, 32 + (index // 7 % 7) * 8, 64 + (index % 7) * 8],
        )
        for index in range(count)
    ]
    path.write_text(
        json.dumps({"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}),
        encoding="utf-8",
    )


def _write_feedback_from_request(request_path: Path, feedback_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": request["measurement_request_sha256"],
        "rows": [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in request["rows"]
        ],
    }
    feedback_path.write_text(json.dumps(payload), encoding="utf-8")


def _loaded_local_config(
    tmp_path: Path,
    *,
    source_group_count: int = 343,
) -> LocalP6CoptV2XConfig:
    del source_group_count
    gold_rows, gold_graphs = _gold176()
    payloads = {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [_profile()],
        "closure": _closure(),
    }
    for name, payload in payloads.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    return load_local_config(_write_yaml(tmp_path / "local.yaml", _local_config(tmp_path)), contract)
```

The `source_group_count` parameter is intentionally accepted for tests that vary registry size through `_write_source_registry()` while sharing one local-config helper.

- [ ] **Step 2: Write RED state-machine test that verifies source registry build, Gold176 refit, 4 rounds, and 16 unique single genomes**

Add:

```python
def test_run_p6_builds_registry_refits_gold176_and_runs_four_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    requests: list[dict[str, Any]] = []
    production_fit_input_counts: list[int] = []
    online_fit_input_counts: list[int] = []
    real_production_fit = execution.fit_production_bundle
    real_online_fit = execution.fit_online_bundle

    def recording_production_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any):
        production_fit_input_counts.append(len(rows))
        assert kwargs["training_view_policy"] == "initial_coldstart_only"
        return real_production_fit(rows, *args, **kwargs)

    def recording_online_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any):
        online_fit_input_counts.append(len(rows))
        return real_online_fit(rows, *args, **kwargs)

    monkeypatch.setattr(execution, "fit_production_bundle", recording_production_fit)
    monkeypatch.setattr(execution, "fit_online_bundle", recording_online_fit)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            source_registry_path = Path(argv[3])
            _write_source_registry(source_registry_path, count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        requests.append(request)
        feedback = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": [
                {
                    "row_id": row["row_id"],
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 3.0 + len(requests),
                    "energy_j": 0.7 + len(requests) / 10,
                    "ap30": 0.91,
                    "ap50": 0.82,
                    "ap70": 0.73,
                }
                for row in request["rows"]
            ],
        }
        Path(argv[3]).write_text(json.dumps(feedback), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    assert len(requests) == 4
    assert [request["round_index"] for request in requests] == [0, 1, 2, 3]
    assert all(request["required_metrics"] == ["latency_ms", "energy_j", "ap30", "ap50", "ap70"] for request in requests)
    selected = [row["row_id"] for request in requests for row in request["rows"]]
    assert len(selected) == len(set(selected)) == 16
    assert {row["q_mode"] for request in requests for row in request["rows"]} <= {"fp16", "int8"}
    assert {row["dispatch_key"] for request in requests for row in request["rows"]} == {"tvm_auto"}
    assert production_fit_input_counts == [176]
    assert online_fit_input_counts == [180, 184, 188]
```

- [ ] **Step 3: Run RED test**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_builds_registry_refits_gold176_and_runs_four_rounds -q
```

Expected: FAIL because `run_p6_coptv2x_search` and v2 feedback parsing do not exist.

- [ ] **Step 4: Implement local input loading and source-registry generation**

In `framework/stage6/coptv2x_h800_search_v2.py`, import:

```python
from framework.stage2.canonical_search_v3 import validate_capability_profile
from framework.stage5.production_search_v1 import fit_production_bundle, predict_candidate_rows
from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_measurement_request,
    build_task_candidate_manifest,
    fit_online_bundle,
    freeze_initial_coldstart,
    select_task_batch,
    validate_task_feedback_history,
)
```

Add:

```python
def _read_json_object_or_list(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P6CoptV2XContractError("local input invalid") from exc


def _run_step(step: LocalExecutionStep, replacements: Mapping[str, Path], *, cwd: Path, runner: CommandRunner) -> None:
    argv = tuple(
        _replace_exact_token(arg, replacements)
        for arg in step.argv
    )
    if runner(argv, cwd) != 0:
        raise P6CoptV2XExecutionError("command_failed", "local command failed")
```

`_replace_exact_token()` must only replace full-argument tokens, not substrings. If an argument is `--out={feedback_json}`, reject it during config loading; the tests in Task 1 cover this.

Write source registry to:

```python
    source_registry_path = local.local_output_root / "source_registry.json"
```

Call `source_registry_step` before fitting the first model. The registry step receives:

```python
{
    "{local_output_root}": local.local_output_root,
    "{source_registry_json}": source_registry_path,
}
```

- [ ] **Step 5: Implement the four-round search loop**

Inside `run_p6_coptv2x_search()`:

1. Create `local.local_output_root` with `parents=True, exist_ok=True`.
2. Load `gold176_rows`, `gold176_graph_features`, `capability_profiles`, and `closure`.
3. Call `freeze_initial_coldstart(gold_rows)` and reject anything other than exactly 176 `initial_coldstart` rows.
4. Select exactly one capability profile whose `hardware_target` is `h800` and `dispatch_key` is `tvm_auto`; reject if none or more than one.
5. Build `SearchTask(task_id=contract.search_id, target_model="pyramid", hardware_id="h800", capability_profile=profile)`.
6. For round zero, call `fit_production_bundle(freeze_initial_coldstart(gold_rows), gold_graph_features, [profile], closure, seed=contract.seed, training_view_policy="initial_coldstart_only")`. For rounds 1–3, call `fit_online_bundle(current_training_rows, gold_graph_features, [profile], seed=contract.seed)`. This intentionally retrains from Gold176 plus accepted online feedback while preserving CoptV2X’s initial-only cold-start fit.
7. Build the manifest using `build_task_candidate_manifest(source_registry, task=task, measured_row_ids=measured_row_ids)`.
8. Predict with `predict_candidate_rows(bundle, manifest["rows"], [profile])`.
9. Select with `select_task_batch(predicted_rows, current_training_rows, gold_graph_features, task=task)`.
10. Build the measurement request with `build_measurement_request(task=task, selected_rows=selection["selected_rows"], round_index=round_index)`.
11. Write request to `local.local_output_root / f"round-{round_index:02d}" / "measurement_request.json"`.
12. Run `measurement_step` with `{measurement_request}`, `{feedback_json}`, and `{round_output_root}`.
13. Parse and validate feedback with the controller-local `_release_feedback_rows()`, append only released feedback rows to `online_feedback_rows`.
14. After rounds 1, 2, and 3, call `validate_task_feedback_history(online_feedback_rows, task=task, completed_rounds=completed_rounds)`.
15. Return `P6CoptV2XRunState(status="completed", completed_rounds=4, measured_candidate_count=16, failure_code=None, local_state_path=local.local_output_root / "state.json")`.

- [ ] **Step 6: Store only local ignored state**

Write `state.json` under `local.local_output_root` with this shape:

```python
{
    "schema_version": "p6_h800_coptv2x_local_state_v2",
    "status": state.status,
    "code_revision": code_revision,
    "completed_rounds": completed_rounds,
    "measured_candidate_count": len(measured_row_ids),
    "failure_code": failure_code,
}
```

Do not include selected row IDs, private paths, command argv, raw metrics, host name, or predictions.

- [ ] **Step 7: Run GREEN test**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_builds_registry_refits_gold176_and_runs_four_rounds -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add framework/stage6/coptv2x_h800_search_v2.py tests/stage6/test_coptv2x_h800_search.py
git commit -m "feat: run P6 CoptV2X TVM search loop"
```

## Task 3: Enforce Atomic Feedback And Candidate-Level Failure Semantics

**Files:**
- Modify: `framework/stage6/coptv2x_h800_search_v2.py`
- Modify: `framework/stage5/single_target_search_v2.py`
- Modify: `tests/stage6/test_coptv2x_h800_search.py`
- Modify: `tests/stage5/test_single_target_search.py`

**Interfaces:**
- Consumes:
  - `stage5_measurement_request_v2`
- Produces:
  - Accepted feedback rows using `training_source: "online_feedback"`, `task_id`, `task_sha256`, `model`, `hardware_id`, `capability_profile_id`, `dispatch_key`, and all five required metrics for `measured_success_gold`.
  - Stable batch failure codes: `source_registry_missing`, `source_registry_invalid`, `command_failed`, `feedback_missing`, `feedback_invalid_json`, `feedback_request_mismatch`, `feedback_candidate_mismatch`, `feedback_metrics_invalid`, `feedback_terminal_status_invalid`, `local_input_invalid`.

- [ ] **Step 1: Write RED tests for candidate-level true failures**

Add:

```python
@pytest.mark.parametrize("terminal_status", ["feasibility_failure", "numerical_feasibility_failure"])
def test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path, source_group_count=343)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        feedback = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": [
                {
                    "row_id": row["row_id"],
                    "terminal_status": terminal_status,
                    "failure_reason": "synthetic_feasibility",
                }
                for row in request["rows"]
            ],
        }
        Path(argv[3]).write_text(json.dumps(feedback), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.measured_candidate_count == 16
```

- [ ] **Step 2: Write RED tests for atomic batch isolation**

Add:

```python
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_row", "feedback_candidate_mismatch"),
        ("extra_row", "feedback_candidate_mismatch"),
        ("wrong_request_sha", "feedback_request_mismatch"),
        ("wrong_row_id", "feedback_candidate_mismatch"),
        ("missing_metric", "feedback_metrics_invalid"),
        ("nan_metric", "feedback_metrics_invalid"),
        ("public_runner_failure", "feedback_terminal_status_invalid"),
    ],
)
def test_run_p6_quarantines_invalid_feedback_batches(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path, source_group_count=343)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        rows = [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in request["rows"]
        ]
        if mutation == "missing_row":
            rows.pop()
        elif mutation == "extra_row":
            rows.append(dict(rows[0]))
        elif mutation == "wrong_row_id":
            rows[0]["row_id"] = "wrong"
        elif mutation == "missing_metric":
            del rows[0]["ap50"]
        elif mutation == "nan_metric":
            rows[0]["latency_ms"] = float("nan")
        elif mutation == "public_runner_failure":
            rows[0] = {"row_id": rows[0]["row_id"], "terminal_status": "public_runner_failure"}
        payload = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": "wrong" if mutation == "wrong_request_sha" else request["measurement_request_sha256"],
            "rows": rows,
        }
        Path(argv[3]).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "failed"
    assert state.failure_code == expected_code
    assert state.completed_rounds == 0
    assert state.measured_candidate_count == 0
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback tests/stage6/test_coptv2x_h800_search.py::test_run_p6_quarantines_invalid_feedback_batches -q
```

Expected: FAIL because feedback validation is not complete.

- [ ] **Step 4: Implement feedback validation**

Implement the P6 controller-local `_release_feedback_rows()` function below. The repository copy of `framework.stage5.single_target_search_v2` currently provides `build_measurement_request()` and `validate_task_feedback_history()`, but it does not expose a feedback finalizer, so Stage6 owns the request/feedback identity check and batch-release decision.

Add:

```python
SUCCESS_STATUS = "measured_success_gold"
TRUE_FAILURE_STATUSES = frozenset({"feasibility_failure", "numerical_feasibility_failure"})
FEEDBACK_SCHEMA_VERSION = "p6_h800_coptv2x_feedback_v2"
```

Implement:

```python
def _release_feedback_rows(
    feedback: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    task: SearchTask,
) -> list[dict[str, Any]]:
    if feedback.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
        raise P6CoptV2XExecutionError("feedback_invalid_json", "feedback schema invalid")
    if feedback.get("measurement_request_sha256") != request.get("measurement_request_sha256"):
        raise P6CoptV2XExecutionError("feedback_request_mismatch", "feedback request mismatch")
    rows = feedback.get("rows")
    if not isinstance(rows, list):
        raise P6CoptV2XExecutionError("feedback_candidate_mismatch", "feedback rows invalid")
    by_request = {str(row["row_id"]): row for row in request["rows"]}
    by_feedback = {str(row.get("row_id") or ""): row for row in rows if isinstance(row, Mapping)}
    if set(by_feedback) != set(by_request) or len(rows) != len(by_request):
        raise P6CoptV2XExecutionError("feedback_candidate_mismatch", "feedback candidate mismatch")
    released = []
    for row_id, request_row in by_request.items():
        row = dict(by_feedback[row_id])
        status = str(row.get("terminal_status") or "")
        base = {
            **copy.deepcopy(request_row),
            "training_source": "online_feedback",
            "terminal_status": status,
        }
        if status == SUCCESS_STATUS:
            for metric in METRIC_NAMES:
                if not _finite(row.get(metric)):
                    raise P6CoptV2XExecutionError("feedback_metrics_invalid", "feedback metrics invalid")
            released.append({**base, **{metric: float(row[metric]) for metric in METRIC_NAMES}})
        elif status in TRUE_FAILURE_STATUSES:
            reason = str(row.get("failure_reason") or "")
            if not reason:
                raise P6CoptV2XExecutionError("feedback_terminal_status_invalid", "failure reason missing")
            released.append({**base, "failure_reason": _redacted_reason(reason)})
        else:
            raise P6CoptV2XExecutionError("feedback_terminal_status_invalid", "feedback status invalid")
    return released
```

`_redacted_reason()` must accept only lower-case letters, digits, `_`, and `-`; return `"unspecified"` if the incoming reason is not public-safe.

- [ ] **Step 5: Ensure failed batches do not advance or consume budget**

Wrap each round in a `try/except P6CoptV2XExecutionError`. On error:

1. Write `failure.json` under the current `round-XX` directory with `schema_version`, `failure_code`, and `completed_rounds`.
2. Return a failed state with no newly released rows from that round.
3. Do not append partial feedback and do not increment `completed_rounds`.

- [ ] **Step 6: Run GREEN tests**

Run:

```bash
pytest tests/stage6/test_coptv2x_h800_search.py::test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback tests/stage6/test_coptv2x_h800_search.py::test_run_p6_quarantines_invalid_feedback_batches -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add framework/stage6/coptv2x_h800_search_v2.py framework/stage5/single_target_search_v2.py tests/stage6/test_coptv2x_h800_search.py tests/stage5/test_single_target_search.py
git commit -m "fix: enforce P6 CoptV2X atomic feedback"
```

## Task 4: Update The CLI And Release Surface

**Files:**
- Modify: `tools/release/run_p6_h800_search.py`
- Modify: `tests/release/test_run_p6_h800_search.py`
- Modify: `tests/release/test_project_handoff.py`
- Modify: `tests/integration/test_anonymous_archive.py`
- Modify: `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify: `docs/superpowers/plans/2026-08-11-p6-h800-search-execution.md`

**Interfaces:**
- CLI consumes:
  - `--contract PATH`
  - `--local-config PATH`
  - `--code-revision PUBLIC_LABEL`
- CLI returns:
  - `0` and `completed\n` only when `run_p6_coptv2x_search()` completes all 4 rounds.
  - `1` and `execution_failed\n` when the local loop fails after valid contracts.
  - `2` and `contract_error\n` for bad contracts or unsafe output boundaries.

- [ ] **Step 1: Write RED black-box CLI test for no public summary**

In `tests/release/test_run_p6_h800_search.py`, remove `PUBLIC_SUMMARY_KEYS` and `--public-summary`. Add:

```python
def test_cli_runs_v2_loop_without_public_summary_and_keeps_outputs_local(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    assert "public-summary" not in result.stdout + result.stderr
    state = json.loads((paths["output_root"] / "state.json").read_text(encoding="utf-8"))
    serialized_state = json.dumps(state, sort_keys=True)
    assert state["status"] == "completed"
    assert state["completed_rounds"] == 4
    assert state["measured_candidate_count"] == 16
    assert str(tmp_path) not in serialized_state
    assert "candidate-" not in serialized_state
```

Update `_run_cli()` to call:

```python
[
    sys.executable,
    str(CLI),
    "--contract",
    str(paths["contract"]),
    "--local-config",
    str(paths["local"]),
    "--code-revision",
    "test-revision",
]
```

- [ ] **Step 2: Write RED CLI argument test that rejects the old `--public-summary` flag**

Add:

```python
def test_cli_rejects_legacy_public_summary_argument(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths, "--public-summary", str(tmp_path / "summary.json"))

    assert result.returncode == 2
    assert result.stderr == "argument_error\n"
```

- [ ] **Step 3: Run RED CLI tests**

Run:

```bash
pytest tests/release/test_run_p6_h800_search.py::test_cli_runs_v2_loop_without_public_summary_and_keeps_outputs_local tests/release/test_run_p6_h800_search.py::test_cli_rejects_legacy_public_summary_argument -q
```

Expected: FAIL because the CLI still requires `--public-summary` and imports the old controller names.

- [ ] **Step 4: Implement CLI v2**

In `tools/release/run_p6_h800_search.py`:

1. Import from `framework.stage6.coptv2x_h800_search_v2`.
2. Remove `--public-summary` and `_paths_conflict()`.
3. Keep `subprocess.run(..., shell=False, text=True, capture_output=True, check=False)`.
4. Call `run_p6_coptv2x_search(contract, local, args.code_revision, _run_command)`.
5. Return `0`, `1`, or `2` as defined above.

- [ ] **Step 5: Update release and archive tests**

Update tests that hard-code old names:

```bash
rg -n "h800_search_execution_v1|H800SearchSummary|public-summary|p6_h800_search_summary_v1|trt_engine|max_rounds" tests tools docs configs
```

Expected after edits:

- `h800_search_execution_v1` absent.
- `H800SearchSummary` absent.
- `p6_h800_search_summary_v1` absent.
- `public-summary` absent except in a test name/assertion rejecting the legacy argument.
- `trt_engine` absent from P6 code/tests except historical docs outside the v2 release manifest.
- `max_rounds` absent from P6 code/tests/config.

For `tests/integration/test_anonymous_archive.py`, ensure the anonymous archive still includes:

```python
"framework/stage6/coptv2x_h800_search_v2.py"
"tools/release/run_p6_h800_search.py"
"configs/execution/p6_h800_search.example.yaml"
```

and excludes:

```python
"configs/local/"
"outputs/"
"results/"
```

- [ ] **Step 6: Update docs**

In `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`, document:

- P6.1 fixed Pyramid/H800/TVM scope.
- 343 structures expanded by `q_mode` into 686 possible TVM candidates.
- Gold176 is reused for cost-model fitting and not remeasured.
- The loop is 4 rounds x 4 candidates.
- Each selected candidate must train/fine-tune and measure five metrics.
- No public summary is generated.
  - P6.2 framework search-space integration is separate and starts only after an approved P6.1 local H800 run; no P6.2 public configuration/example is added in P6.1.

In `docs/AAAI27_RELEASE_AUDIT.md`, update the P6 row to state:

```markdown
| P6 | 进行中（本地） | 第一阶段验证 Pyramid/H800/TVM 的 CoptV2X 完整搜索闭环：复用本地 Gold176 重新训练 cost model，基于 343×2 候选注册表每轮选择 4 个候选，并通过本地适配器完成剪枝、微调、导出、TVM 编译、AP/能耗/时延实测和反馈重训；Orin 与 TensorRT 是后续独立线。 | 本地忽略配置保存资产路径、命令和原始输出；公开面只保留固定契约、脱敏资产标签、版本/许可状态和离线验证器，不生成公开 P6 结果摘要。不下载资产，不公开路径、命令、主机信息、候选 ID、原始日志、checkpoint、编译产物或逐轮原始结果；共享资产/命令/结果结构失败必须整批隔离，候选真实可行性失败才消耗预算。 |
```

In `docs/superpowers/plans/2026-08-11-p6-h800-search-execution.md`, add at the top:

```markdown
> Superseded for implementation by `docs/superpowers/plans/2026-08-13-p6-1-coptv2x-tvm-search-loop.md` and `docs/superpowers/plans/2026-08-13-p6-2-framework-search-space-integration.md`; this file records the earlier design before the P6.1/P6.2 split.
```

- [ ] **Step 7: Run GREEN release tests**

Run:

```bash
pytest tests/release/test_run_p6_h800_search.py tests/integration/test_anonymous_archive.py::test_builder_includes_release_tools_required_by_archive_tests tests/integration/test_anonymous_archive.py::test_builder_allows_only_the_anonymous_ci_hidden_path tests/release/test_project_handoff.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add tools/release/run_p6_h800_search.py tests/release/test_run_p6_h800_search.py tests/release/test_project_handoff.py tests/integration/test_anonymous_archive.py docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md docs/superpowers/plans/2026-08-11-p6-h800-search-execution.md
git commit -m "docs: align P6 release surface with CoptV2X TVM loop"
```

## Task 5: Full Local Verification Gate

**Files:**
- Modify only if targeted tests reveal a defect:
  - `framework/stage6/coptv2x_h800_search_v2.py`
  - `tools/release/run_p6_h800_search.py`
  - affected tests

**Interfaces:**
- Consumes all P6.1 public and local interfaces from Tasks 1-4.
- Produces a clean branch with passing offline tests.

- [ ] **Step 1: Run the P6.1 targeted suite**

Run:

```bash
pytest tests/stage5/test_single_target_search.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py -q
```

Expected: PASS.

- [ ] **Step 2: Run release boundary tests**

Run:

```bash
pytest tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py tests/release/test_public_execution_surface.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full CI-equivalent quality suite locally**

Run:

```bash
pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

Expected: PASS with total coverage at or above 80%.

- [ ] **Step 4: Run static scans for stale P6 semantics**

Run:

```bash
rg -n "h800_search_execution_v1|H800SearchSummary|p6_h800_search_summary_v1|public-summary|max_rounds|trt_engine|TensorRT|RTX 4090|CPU" framework/stage6 tools/release configs/execution tests/stage6 tests/release docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md
```

Expected:

- No hits for `h800_search_execution_v1`, `H800SearchSummary`, `p6_h800_search_summary_v1`, `public-summary`, or `max_rounds`.
- `TensorRT`, `RTX 4090`, and `CPU` may appear only in exclusion language in docs, not as P6.1 executable scope.

- [ ] **Step 5: Review diff before final P6.1 implementation commit**

Run:

```bash
git diff --check
git diff --stat
git diff -- framework/stage6/coptv2x_h800_search_v2.py tools/release/run_p6_h800_search.py tests/stage6/test_coptv2x_h800_search.py tests/release/test_run_p6_h800_search.py docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md
```

Expected: no whitespace errors; diff shows only P6.1 migration and docs/tests.

- [ ] **Step 6: Commit any verification fixes**

If Step 1-5 required fixes, run:

```bash
git add framework/stage6/coptv2x_h800_search_v2.py framework/stage5/single_target_search_v2.py tools/release/run_p6_h800_search.py tests/stage6/test_coptv2x_h800_search.py tests/stage5/test_single_target_search.py tests/release/test_run_p6_h800_search.py tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py tests/release/test_public_execution_surface.py docs/release-manifests/P6_H800_SEARCH_EXECUTION.md docs/AAAI27_RELEASE_AUDIT.md
git commit -m "fix: close P6 CoptV2X verification gaps"
```

If no fixes are needed, do not create an empty commit.

- [ ] **Step 7: Stop before real H800 execution**

Do not run the real H800/Pyramid/TVM job as part of implementation completion. Real execution requires a fresh explicit launch approval because it consumes local assets and GPU time. When approved, use the same CLI with a Git-ignored local config:

```bash
python tools/release/run_p6_h800_search.py \
  --contract configs/execution/p6_h800_search.example.yaml \
  --local-config configs/local/p6_h800_search.local.yaml \
  --code-revision "$(git rev-parse --short=12 HEAD)"
```

Expected for a real approved run: `completed` after 4 rounds, with `outputs/` or the configured local root containing only Git-ignored local state, requests, feedback, logs, checkpoints, ONNX, TVM artifacts, and run outputs.

## Self-Review Checklist

- Spec coverage: P6.1 fixed Pyramid/H800/TVM, single-genome `q_mode`, Gold176 cold-start retraining, 4x4 budget, full measurement metrics, atomic feedback, no public summary, local-only outputs, and real-run gate are covered in Tasks 1-5.
- Placeholder scan: no `TBD`, no unqualified “add validation”, no “write tests for the above”, no undefined implementation task.
- Type consistency: contract/config/state names use the `P6CoptV2X` v2 prefix throughout; CLI and tests import `coptv2x_h800_search_v2`; Stage5 interfaces match existing `SearchTask`, `build_task_candidate_manifest`, `predict_candidate_rows`, `select_task_batch`, `fit_online_bundle`, and `build_measurement_request`.
