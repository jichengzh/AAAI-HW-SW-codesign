# AAAI-27 Dual-Artifact Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Stage1/Stage2 repository into a tested, paper-aligned reproducibility repository that produces both a public GitHub release and an anonymous AAAI-27 Code and Data ZIP from the same reviewed commit.

**Architecture:** Preserve the existing Stage1/Stage2 public runtime, then add the minimum production Stage4--Stage7 selection and analysis modules used by the paper. Hardware materialization, TVM/TensorRT execution, full AP evaluation, private checkpoints, and cluster orchestration remain explicit external boundaries. An allowlist-based exporter builds the anonymous ZIP and rejects identity, local-path, secret, unverified-result, and oversized-file leakage.

**Tech Stack:** Python 3.10; PyTorch and torch-pruning for optional Stage1 scanning; NumPy, pandas, SciPy, scikit-learn, and LightGBM for Stage4--Stage7 reproduction; pytest, pytest-cov, Ruff, and GitHub Actions for quality gates; Apache-2.0 license.

## Global Constraints

- The working branch is `release/aaai27-reproducibility`; `main` is not modified directly.
- Existing user changes on `snapshot-20260624-current-github` are preserved and must not be reset, overwritten, or silently reformatted.
- Public and anonymous artifacts must be generated from the same Git commit.
- The anonymous archive must not contain `.git`, GitHub owner/repository URLs, author names, email addresses, local absolute paths, host addresses, credentials, private checkpoints, or third-party source trees.
- Demo data must be labeled as non-paper evidence and must never be used to populate paper result tables.
- Only artifacts with an explicit schema, SHA256, provenance record, and verified status may enter `artifacts/verified/`.
- Stage7 numerical results must not be published as paper evidence until all 12 formal trajectories and 192 selected events have a final aggregate manifest.
- Timing warm-ups, iterations, and in-process repeats must not be reported as independent algorithm runs.
- New and migrated deterministic core modules must reach at least 80% line coverage; hardware-only adapters are excluded from the CI coverage denominator and tested through contract mocks.
- All user-controlled paths and JSON/YAML inputs must be validated before use.
- No command in this plan pushes to GitHub. Push and pull-request creation require a final explicit approval after local verification.

---

## 1. Current State and Protected Baseline

The target checkout is the current repository root:

```text
$REPO_ROOT
```

Its collaboration remote is intentionally not recorded in this public-safe plan. Verify the configured remote locally before any authorized push; do not copy a personal account URL into release documentation.

```text
<private-collaboration-remote>
```

The repository currently contains:

- remote `main` at `67d468f`, publishing the Stage1 scanner;
- remote `snapshot-20260624-current-github` at `e1351d6`;
- uncommitted Stage2 contracts, search drivers, documentation, and Stage1 bridge changes;
- no dependency metadata, license file, test suite, CI workflow, or anonymous release exporter.

Before implementation, save a safety copy outside the repository:

```bash
repo=$(git rev-parse --show-toplevel)
snapshot="$(dirname "$repo")/AAAI-HW-SW-codesign-pre-release-20260729"
mkdir -p "$snapshot"
git -C "$repo" status --short > "$snapshot/status.txt"
git -C "$repo" diff --binary > "$snapshot/tracked-changes.patch"
git -C "$repo" ls-files --others --exclude-standard > "$snapshot/untracked-files.txt"
tar -C "$repo" -czf "$snapshot/untracked-files.tar.gz" \
  --files-from "$snapshot/untracked-files.txt"
sha256sum "$snapshot"/* > "$snapshot/SHA256SUMS"
```

The snapshot stays outside both release artifacts.

---

## 2. Target Repository Layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── README.zh-CN.md
├── README.anonymous.md
├── REPRODUCIBILITY.md
├── ARTIFACTS.md
├── pyproject.toml
├── framework/
│   ├── capability_schema.py
│   ├── stage1/
│   ├── stage1_bridge.py
│   ├── stage2/
│   │   ├── canonical_search_v3.py
│   │   ├── contracts.py
│   │   └── __init__.py
│   ├── stage4/
│   │   ├── cost_model_selection_v1.py
│   │   └── __init__.py
│   ├── stage5/
│   │   ├── genome_contract_v1.py
│   │   ├── production_search_v1.py
│   │   ├── single_target_search_v2.py
│   │   └── __init__.py
│   ├── stage6/
│   │   ├── paper_table_v1.py
│   │   └── __init__.py
│   └── stage7/
│       ├── ablation_statistics_v2.py
│       ├── online_component_ablation_v1.py
│       ├── search_policy_v1.py
│       └── __init__.py
├── scripts/
│   ├── prepare_stage2_demo_data.py
│   ├── reproduce/
│   │   ├── cost_model_selection.py
│   │   ├── stage5_selection.py
│   │   ├── stage6_table.py
│   │   ├── stage7_selection.py
│   │   └── reproduce_all.py
│   ├── stage1_classify_models.py
│   ├── stage2_optimize_model.py
│   └── stage2_update_evidence.py
├── data/
│   ├── README.md
│   └── demo/
│       ├── capability_profiles.json
│       ├── coldstart_graph_features.jsonl
│       ├── coldstart_measurements.jsonl
│       ├── candidate_pool.jsonl
│       └── manifest.json
├── artifacts/
│   ├── README.md
│   └── verified/
│       ├── manifest.json
│       └── stage4/
│           ├── cost_model_selection_report.json
│           └── nested_cv_folds.csv
├── tools/
│   └── release/
│       ├── anonymous_allowlist.txt
│       ├── forbidden_patterns.txt
│       ├── build_anonymous_archive.py
│       └── verify_archive.py
└── tests/
    ├── conftest.py
    ├── integration/
    │   ├── test_anonymous_archive.py
    │   └── test_reproduce_all.py
    ├── stage1/
    │   └── test_stage1_bridge.py
    ├── stage2/
    │   ├── test_contracts.py
    │   └── test_demo_pipeline.py
    ├── stage4/
    │   └── test_cost_model_selection.py
    ├── stage5/
    │   ├── test_genome_contract.py
    │   ├── test_production_search.py
    │   └── test_single_target_search.py
    ├── stage6/
    │   └── test_paper_table.py
    ├── stage7/
    │   ├── test_ablation_contracts.py
    │   ├── test_search_policy.py
    │   └── test_statistics.py
    └── release/
        └── test_identity_scan.py
```

---

## 3. Frozen Source Modules

The following current production files are the migration sources. Their SHA256 values freeze the reviewed starting point; implementation must fail rather than silently copy a different source.

| Target file | Source SHA256 |
|---|---|
| `framework/stage2/canonical_search_v3.py` | `bab5ddf6f2427d168c4785050f1963c9d99991bdf64f53ce531ced0a539ac183` |
| `framework/stage4/cost_model_selection_v1.py` | `e93b42b0d659f1ecaa7ecbf6ba7e847c21da37922eec15e312f07074c414c784` |
| `framework/stage5/genome_contract_v1.py` | `1aa183c576510cf4612d3a96028a56e37b8fd4281058867b7d3b4a08660fbe7b` |
| `framework/stage5/production_search_v1.py` | `6d373c312e236222d7630d91deb8cb122d5c928b464e5d1b60f7070ea33db2e2` |
| `framework/stage5/single_target_search_v2.py` | `7d3694a82cee4e9f9265471fd32371d1012279a71a6c150bf8443240b870df35` |
| `framework/stage6/paper_table_v1.py` | `a929e6bf58637f5ee74a7232f2a923474b9ed64c8623d0f53a61bd5d967e7b58` |
| `framework/stage7/online_component_ablation_v1.py` | `ac2785f86f9cbeeac8d623eeb47bce762966e58f99b4637f134b6971e2af8960` |
| `framework/stage7/search_policy_v1.py` | `23207c57f609f8f127b2b0fa4ba3d44d16bee817e973f596380fcf8597b0ba7c` |
| `framework/stage7/ablation_statistics_v2.py` | `27d2f7436205675ce79d676db387def813994e96814b42f7482a2dadff4252a0` |

Migration preserves the source algorithms. Repository-specific changes are limited to package-relative paths, explicit input arguments, public error messages, comments linking modules to paper sections, and removal of private runtime assumptions. Algorithmic changes require a separate equivalence test and explicit review.

---

## 4. Task-by-Task Implementation

### Task 1: Packaging, License, and Development Baseline

**Files:**

- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `CITATION.cff`
- Create: `tests/conftest.py`
- Create: `tests/release/test_identity_scan.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**

- Produces the `gear_codesign` project metadata and dependency extras.
- Exposes the commands `ruff check .`, `pytest`, and `pytest --cov=framework`.
- Defines Apache-2.0 copyright as `The GEAR Authors`.

- [ ] **Step 1: Write metadata tests**

  Create a test in `tests/release/test_identity_scan.py` that parses `pyproject.toml`, verifies Python `>=3.10`, checks that `LICENSE` contains `Apache License`, and asserts that public package metadata contains no local path.

- [ ] **Step 2: Verify the tests fail**

  Run:

  ```bash
  pytest tests/release/test_identity_scan.py -q
  ```

  Expected result: failure because packaging files do not exist.

- [ ] **Step 3: Add package metadata**

  Define these dependency groups in `pyproject.toml`:

  ```toml
  dependencies = ["numpy>=1.24,<3", "pyyaml>=6,<7", "pydantic>=2,<3"]
  scan = ["torch>=2.0", "torch-pruning>=1.4"]
  repro = [
    "pandas>=2.0,<3",
    "scipy>=1.10,<2",
    "scikit-learn>=1.5,<2",
    "lightgbm>=4.0,<5",
  ]
  dev = [
    "build>=1.2,<2",
    "pytest>=8,<9",
    "pytest-cov>=5,<7",
    "ruff>=0.6,<1",
  ]
  ```

  Configure Ruff for Python 3.10 with line length 100. Configure pytest to search `tests/` and require strict markers.

- [ ] **Step 4: Add CI**

  `.github/workflows/ci.yml` runs on Python 3.10 and 3.11:

  ```bash
  pip install -e '.[repro,dev]'
  ruff check framework scripts tests tools
  pytest -q --cov=framework --cov=scripts/reproduce --cov-fail-under=80
  python tools/release/build_anonymous_archive.py --check-only
  ```

- [ ] **Step 5: Verify**

  Run metadata tests and `python -m build`. Both must pass.

- [ ] **Step 6: Commit**

  ```bash
  git add pyproject.toml LICENSE .gitignore CITATION.cff tests/conftest.py \
    tests/release/test_identity_scan.py .github/workflows/ci.yml
  git commit -m "chore: establish reproducible package baseline"
  ```

### Task 2: Preserve and Close the Existing Stage2 Work

**Files:**

- Modify: `framework/stage1_bridge.py`
- Modify: `framework/stage1/model_classifier.py`
- Add existing: `framework/stage2/contracts.py`
- Add existing: `framework/stage2/__init__.py`
- Add existing: `framework/search_three_arm.py`
- Add existing: `framework/run_b4_ablation.py`
- Add existing: `framework/run_pqs_ablation.py`
- Add existing: `framework/run_pqs_codriving.py`
- Add existing: `scripts/prepare_stage2_demo_data.py`
- Add existing: `scripts/stage2_optimize_model.py`
- Add existing: `scripts/stage2_update_evidence.py`
- Create: `tests/stage1/test_stage1_bridge.py`
- Create: `tests/stage2/test_contracts.py`
- Create: `tests/stage2/test_demo_pipeline.py`

**Interfaces:**

- `load_stage2_search_space(manifest_path)` returns a model-level legal search space.
- `build_stage2_output(Stage2Input)` returns the frozen Stage2 output schema.
- The demo pipeline writes only under a caller-provided temporary directory.

- [ ] **Step 1: Write contract tests**

  Cover invalid manifest paths, missing classification files, mismatched model identity, illegal INT8 widths, classifier policy gating, deterministic search under a fixed seed, and immutable evidence-delta creation.

- [ ] **Step 2: Verify failure**

  ```bash
  pytest tests/stage1 tests/stage2 -q
  ```

  Expected result: failures for missing fixtures and non-parameterized output paths.

- [ ] **Step 3: Make demo paths explicit**

  Add `--output-root` to `scripts/prepare_stage2_demo_data.py`; it must reject `/`, the repository root, and an existing non-demo directory. Remove repository-relative write defaults from tests.

- [ ] **Step 4: Close the Stage2 tests**

  The end-to-end test runs:

  ```bash
  python scripts/prepare_stage2_demo_data.py --output-root "$tmp/demo"
  python scripts/stage2_optimize_model.py \
    --manifest "$tmp/demo/framework/partitions/pyramid_lidar_partition.yaml" \
    --classification "$tmp/demo/results/model_classifier.json" \
    --out-json "$tmp/out/stage2.json"
  ```

  It asserts stable schema, candidate count, policy, and output SHA across two runs.

- [ ] **Step 5: Commit**

  Stage only the reviewed Stage2 files and their tests:

  ```bash
  git add README.md README.zh-CN.md \
    docs/stage2-evidence-delta.zh-CN.md docs/stage2-new-hardware.zh-CN.md \
    framework/stage1/model_classifier.py framework/stage1_bridge.py \
    framework/stage2 framework/search_three_arm.py \
    framework/run_b4_ablation.py framework/run_pqs_ablation.py \
    framework/run_pqs_codriving.py \
    scripts/prepare_stage2_demo_data.py scripts/stage2_*.py \
    scripts/phase2/b4_integrate.py scripts/phase2/b5_verify_convergence.py \
    scripts/phase2/closedloop_objective_query.py tests/stage1 tests/stage2
  git commit -m "feat: publish deterministic stage2 search contracts"
  ```

### Task 3: Add the Stage4 Cost-Model Selection Core

**Files:**

- Create: `framework/stage4/__init__.py`
- Create: `framework/stage4/cost_model_selection_v1.py`
- Create: `scripts/reproduce/cost_model_selection.py`
- Create: `tests/stage4/test_cost_model_selection.py`

**Interfaces:**

- `run_nested_selection(rows, graph_features, outer_splits=5, inner_splits=3, seed=20260716)` returns the report payload.
- CLI input is JSONL measurements plus JSONL graph features.
- CLI output is `report.json`, `folds.csv`, and `manifest.json`.

- [ ] **Step 1: Verify the frozen source SHA**

  ```bash
  source_root=${GEAR_SOURCE_ROOT:?set GEAR_SOURCE_ROOT to the reviewed source checkout}
  test "$(sha256sum "$source_root/framework/stage4/cost_model_selection_v1.py" | cut -d' ' -f1)" \
    = "e93b42b0d659f1ecaa7ecbf6ba7e847c21da37922eec15e312f07074c414c784"
  ```

- [ ] **Step 2: Write tests first**

  Test 5-by-3 fold construction, group non-leakage, seed derivation, label-feature exclusion, deterministic selected predictor, and rejection when fewer than five groups are available.

- [ ] **Step 3: Verify failure**

  ```bash
  pytest tests/stage4/test_cost_model_selection.py -q
  ```

- [ ] **Step 4: Migrate the source and add the CLI**

  Keep the production model definitions unchanged. The CLI records input SHA256, seed, row/group counts, dependency versions, and output SHA256.

- [ ] **Step 5: Verify**

  ```bash
  pytest tests/stage4/test_cost_model_selection.py -q
  python scripts/reproduce/cost_model_selection.py --help
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add framework/stage4 scripts/reproduce/cost_model_selection.py tests/stage4
  git commit -m "feat: add reproducible cost-model selection"
  ```

### Task 4: Add the Stage5 Production Search Core

**Files:**

- Create: `framework/stage2/canonical_search_v3.py`
- Create: `framework/stage5/__init__.py`
- Create: `framework/stage5/genome_contract_v1.py`
- Create: `framework/stage5/production_search_v1.py`
- Create: `framework/stage5/single_target_search_v2.py`
- Create: `scripts/reproduce/stage5_selection.py`
- Create: `tests/stage5/test_genome_contract.py`
- Create: `tests/stage5/test_production_search.py`
- Create: `tests/stage5/test_single_target_search.py`

**Interfaces:**

- `fit_production_bundle(...)` returns the frozen AP/latency/energy predictor bundle.
- `select_predicted_frontier_diversity(...)` returns ordered selected candidate IDs.
- `SearchTask` and `build_measurement_request(...)` define the hardware-execution boundary.

- [ ] **Step 1: Verify all four frozen source SHAs**

  Check `canonical_search_v3.py`, `genome_contract_v1.py`, `production_search_v1.py`, and `single_target_search_v2.py` against Section 3.

- [ ] **Step 2: Write tests**

  Cover:

  - Pyramid, CoDriving, and F-Cooper genome identity;
  - capability-profile validation;
  - seed-bound ExtraTrees/LightGBM fitting;
  - candidate-label hiding before selection;
  - deterministic ordered IDs;
  - diversity tie-breaking;
  - no duplicate candidate selection;
  - exact measurement-request identity;
  - rejection of non-finite metrics and malformed graph features.

- [ ] **Step 3: Verify failure**

  ```bash
  pytest tests/stage5 -q
  ```

- [ ] **Step 4: Migrate source without algorithm changes**

  Preserve `predicted_frontier_diversity`, model heads, target transforms, uncertainty logic, and seed rules. Replace only private default paths with required CLI arguments.

- [ ] **Step 5: Add a selection-only CLI**

  `scripts/reproduce/stage5_selection.py` reads public JSON/JSONL files, trains the bundle, freezes ordered selected IDs, and writes a request manifest. It does not run TVM, TensorRT, AP evaluation, latency, or energy measurement.

- [ ] **Step 6: Verify**

  ```bash
  pytest tests/stage5 -q
  python scripts/reproduce/stage5_selection.py --help
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add framework/stage2/canonical_search_v3.py framework/stage5 \
    scripts/reproduce/stage5_selection.py tests/stage5
  git commit -m "feat: publish production search selection core"
  ```

### Task 5: Add Stage6 Representative-Point Selection

**Files:**

- Create: `framework/stage6/__init__.py`
- Create: `framework/stage6/paper_table_v1.py`
- Create: `scripts/reproduce/stage6_table.py`
- Create: `tests/stage6/test_paper_table.py`

**Interfaces:**

- Input rows contain AP70, latency, energy, terminal status, validation status, and evidence-SHA status.
- Output contains one representative point per model/backend/method cell or an explicit failure status.

- [ ] **Step 1: Write tests**

  Verify:

  - $\Delta\mathrm{AP}_{70}\leq0.10$ feasibility;
  - rejection of non-gold or SHA-unverified points;
  - 1% latency window;
  - energy tie-break inside the window;
  - no fallback to a different model/backend/method cell;
  - explicit empty-cell reporting.

- [ ] **Step 2: Verify failure**

  ```bash
  pytest tests/stage6/test_paper_table.py -q
  ```

- [ ] **Step 3: Migrate the frozen source and add CSV/Markdown output**

  The CLI writes raw JSON, CSV, Markdown, and a manifest binding all output rows to their input evidence IDs.

- [ ] **Step 4: Verify and commit**

  ```bash
  pytest tests/stage6/test_paper_table.py -q
  git add framework/stage6 scripts/reproduce/stage6_table.py tests/stage6
  git commit -m "feat: add evidence-conservative paper table selection"
  ```

### Task 6: Add the Stage7 Four-Variant Selection and Statistics Core

**Files:**

- Create: `framework/stage7/__init__.py`
- Create: `framework/stage7/online_component_ablation_v1.py`
- Create: `framework/stage7/search_policy_v1.py`
- Create: `framework/stage7/ablation_statistics_v2.py`
- Create: `scripts/reproduce/stage7_selection.py`
- Create: `tests/stage7/test_ablation_contracts.py`
- Create: `tests/stage7/test_search_policy.py`
- Create: `tests/stage7/test_statistics.py`

**Interfaces:**

- Variants are `full`, `without_surrogate`, `without_measured_feedback`, and `backend_blind`.
- Seeds are exactly `20260718`, `20260719`, and `20260720`.
- Each trajectory uses four rounds, four candidates per round, and a total budget of 16.
- The public CLI is selection/statistics only and consumes explicit feedback JSONL between rounds.

- [ ] **Step 1: Write contract tests**

  Assert the four single-variable definitions, frozen seeds/budget, local `random.Random(seed)` for uniform sampling, no measured-feedback refit for A2, removal of backend/capability/profile-derived features for backend-blind, and label/cache hiding before selection.

- [ ] **Step 2: Write policy tests**

  Test round-zero and later-round selection, selected-ID exclusion, A2 frozen-bundle reuse, backend-blind schema diff, identical-trajectory acceptance, exact request identity, and invalid feedback rejection.

- [ ] **Step 3: Write statistics tests**

  Use four budget checkpoints and verify $\Delta$HV-AUC, mean, sample standard deviation, paired deltas, median/range, and direction consistency. Do not implement significance tests.

- [ ] **Step 4: Verify failure**

  ```bash
  pytest tests/stage7 -q
  ```

- [ ] **Step 5: Migrate the frozen sources**

  Preserve the current `predicted_frontier_diversity` implementation and do not rename it NSGA-II. Exclude GPU leasing, cache execution, TVM build, AP evaluation, and cluster orchestration modules.

- [ ] **Step 6: Add the selection/statistics CLI**

  The CLI supports:

  ```bash
  python scripts/reproduce/stage7_selection.py select --variant full --seed 20260718 --round 0 ...
  python scripts/reproduce/stage7_selection.py summarize \
    --trajectory-root results/stage7_fixture \
    --output-json results/stage7_fixture/summary.json
  ```

  `summarize` exits non-zero unless all expected trajectories and terminal events are present.

- [ ] **Step 7: Verify and commit**

  ```bash
  pytest tests/stage7 -q
  git add framework/stage7 scripts/reproduce/stage7_selection.py tests/stage7
  git commit -m "feat: publish online ablation selection contracts"
  ```

### Task 7: Add Demo Data and Verified Artifact Boundaries

**Files:**

- Create: `data/README.md`
- Create: `data/demo/capability_profiles.json`
- Create: `data/demo/coldstart_graph_features.jsonl`
- Create: `data/demo/coldstart_measurements.jsonl`
- Create: `data/demo/candidate_pool.jsonl`
- Create: `data/demo/manifest.json`
- Create: `artifacts/README.md`
- Create: `artifacts/verified/manifest.json`
- Create: `artifacts/verified/stage4/cost_model_selection_report.json`
- Create: `artifacts/verified/stage4/nested_cv_folds.csv`

**Interfaces:**

- Demo manifest has `"paper_evidence": false`.
- Verified manifest entries require path, SHA256, schema, provenance, paper mapping, and verification status.

- [ ] **Step 1: Write manifest validation tests**

  Add assertions to `tests/integration/test_reproduce_all.py` that reject missing SHA, absolute source paths, duplicate artifact IDs, and demo records marked as paper evidence.

- [ ] **Step 2: Sanitize the Stage4 verified artifacts**

  Remove local paths and user identities while preserving the seed, row count, group count, folds, candidate names, selected models, metrics, and source-data content hash.

- [ ] **Step 3: Keep Stage7 results absent until formal closure**

  `artifacts/verified/manifest.json` must not contain a Stage7 paper-table entry unless the source aggregate states all 12 trajectories and 192 events are complete. The absence is documented in `artifacts/README.md`, not replaced by fabricated values.

- [ ] **Step 4: Verify and commit**

  ```bash
  pytest tests/integration/test_reproduce_all.py -q
  git add data artifacts tests/integration/test_reproduce_all.py
  git commit -m "data: add audited demo and verified artifact manifests"
  ```

### Task 8: Add Unified Reproduction Commands

**Files:**

- Create: `scripts/reproduce/reproduce_all.py`
- Create: `REPRODUCIBILITY.md`
- Create: `tests/integration/test_reproduce_all.py`

**Interfaces:**

- `reproduce_all.py --mode smoke --output-root results/repro_smoke` runs without a GPU.
- `reproduce_all.py --mode verified --output-root results/repro_verified` reproduces only artifact-backed paper analyses and fails if required verified inputs are absent.

- [ ] **Step 1: Write the failing end-to-end test**

  Run the smoke mode twice in separate temporary directories and assert identical selected IDs, schemas, and content hashes.

- [ ] **Step 2: Implement the orchestrator**

  The command executes Stage2 demo generation, Stage4 model selection, Stage5 selection, Stage6 representative selection, and a Stage7 selection-only round. It writes `run_manifest.json` with component versions, seeds, input/output SHA256 values, start/end times, and status.

- [ ] **Step 3: Verify**

  ```bash
  pytest tests/integration/test_reproduce_all.py -q
  python scripts/reproduce/reproduce_all.py --mode smoke --output-root /tmp/gear-aaai27-smoke
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add scripts/reproduce/reproduce_all.py REPRODUCIBILITY.md \
    tests/integration/test_reproduce_all.py
  git commit -m "feat: add end-to-end reproducibility entrypoint"
  ```

### Task 9: Build the Anonymous AAAI ZIP

**Files:**

- Create: `README.anonymous.md`
- Create: `tools/release/anonymous_allowlist.txt`
- Create: `tools/release/forbidden_patterns.txt`
- Create: `tools/release/build_anonymous_archive.py`
- Create: `tools/release/verify_archive.py`
- Create: `tests/integration/test_anonymous_archive.py`
- Modify: `tests/release/test_identity_scan.py`

**Interfaces:**

- Builder input is the repository root and an output directory.
- Builder output is `aaai27_code_data_anonymous.zip`, `aaai27_code_data_anonymous.zip.sha256`, and `archive_manifest.json`.
- Builder copies only allowlisted paths and renames `README.anonymous.md` to `README.md` inside the archive.

- [ ] **Step 1: Write failure tests**

  Fixtures contain a local path, GitHub owner URL, email, IP address, token-shaped string, symlink escaping the repository, and a file over 20 MiB. Each must make archive creation fail with a non-secret diagnostic.

- [ ] **Step 2: Define the allowlist**

  Include:

  ```text
  LICENSE
  README.anonymous.md
  REPRODUCIBILITY.md
  ARTIFACTS.md
  pyproject.toml
  framework/**
  scripts/reproduce/**
  scripts/prepare_stage2_demo_data.py
  scripts/stage1_classify_models.py
  scripts/stage2_optimize_model.py
  data/**
  artifacts/**
  tests/**
  ```

  Exclude `.git`, `.github`, `CITATION.cff`, public README files, caches, build outputs, checkpoints, engines, ONNX models, result roots, and all files not explicitly listed.

- [ ] **Step 3: Implement scanning**

  Inspect file names, symlink targets, and UTF-8 text content. Scan the completed ZIP again after extraction into a temporary directory.

- [ ] **Step 4: Verify**

  ```bash
  pytest tests/integration/test_anonymous_archive.py tests/release/test_identity_scan.py -q
  python tools/release/build_anonymous_archive.py --output-dir dist
  python tools/release/verify_archive.py dist/aaai27_code_data_anonymous.zip
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add README.anonymous.md tools/release tests/integration/test_anonymous_archive.py \
    tests/release/test_identity_scan.py
  git commit -m "feat: add anonymous AAAI code archive exporter"
  ```

### Task 10: Complete Public Documentation

**Files:**

- Rewrite: `README.md`
- Rewrite: `README.zh-CN.md`
- Modify: `REPRODUCIBILITY.md`
- Create: `ARTIFACTS.md`
- Update: `CITATION.cff`

**Interfaces:**

- README quick start completes a CPU-only smoke run.
- REPRODUCIBILITY maps each paper table/figure to code, input artifact, seed, run count, and evidence status.
- ARTIFACTS distinguishes demo, verified, external, and unavailable artifacts.

- [ ] **Step 1: Add documentation tests**

  Verify every documented command resolves to an existing file, every relative link exists, and anonymous README contains no public repository URL or author identity.

- [ ] **Step 2: Write the public documentation**

  Include installation, quick start, repository layout, supported scope, external hardware boundary, dataset access, seed rules, algorithm-run definitions, artifact provenance, known limitations, license, and citation.

- [ ] **Step 3: Verify**

  ```bash
  pytest tests/release/test_identity_scan.py -q
  python -m compileall -q framework scripts tools
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add README.md README.zh-CN.md REPRODUCIBILITY.md ARTIFACTS.md CITATION.cff
  git commit -m "docs: document AAAI reproducibility workflow"
  ```

### Task 11: Final Verification and Release Preparation

**Files:**

- Generated, not committed: `dist/aaai27_code_data_anonymous.zip`
- Generated, not committed: `dist/aaai27_code_data_anonymous.zip.sha256`
- Generated, not committed: `dist/archive_manifest.json`
- Create: `docs/AAAI27_RELEASE_AUDIT.md`

- [ ] **Step 1: Run build and static checks**

  ```bash
  pip install -e '.[repro,dev]'
  ruff check framework scripts tests tools
  python -m compileall -q framework scripts tools
  ```

- [ ] **Step 2: Run tests and coverage**

  ```bash
  pytest -q --cov=framework --cov=scripts/reproduce \
    --cov-report=term-missing --cov-report=xml --cov-fail-under=80
  ```

- [ ] **Step 3: Run security checks**

  Scan tracked files and the archive for private keys, credential patterns, personal paths, account names, emails, host addresses, and symlink escapes. Inspect `git diff --check` and all new dependencies.

- [ ] **Step 4: Reproduce from the anonymous archive**

  Extract into a fresh temporary directory, create a new Python environment, install `.[repro,dev]`, run smoke reproduction, and run the complete archive test suite.

- [ ] **Step 5: Write the audit**

  `docs/AAAI27_RELEASE_AUDIT.md` records:

  - source commit SHA;
  - archive SHA256;
  - test and coverage totals;
  - Python and dependency versions;
  - included/excluded artifact categories;
  - identity-scan result;
  - known external hardware boundaries;
  - Table1/Table2/Figure evidence status.

- [ ] **Step 6: Review before external action**

  Run:

  ```bash
  git status --short
  git log --oneline --decorate main..HEAD
  git diff --stat main...HEAD
  ```

  Request explicit approval before pushing `release/aaai27-reproducibility`, opening a pull request, merging into `main`, or publishing a GitHub release.

---

## 5. Required Test Matrix

| Area | Unit | Integration | End-to-end |
|---|---:|---:|---:|
| Stage1 bridge/classifier | Yes | Stage1-to-Stage2 contract | Included in smoke |
| Stage2 contracts/demo | Yes | Demo optimization | Included in smoke |
| Stage4 model selection | Yes | JSONL-to-report | Included in smoke |
| Stage5 production search | Yes | Bundle-to-request | Included in smoke |
| Stage6 table selection | Yes | Evidence-to-table | Included in verified mode |
| Stage7 four variants | Yes | Multi-round fixture | Selection-only smoke |
| Anonymous exporter | Yes | Build/extract/rescan | Fresh-environment smoke |

Hardware execution is not silently mocked as a paper result. Contract mocks prove interface behavior only, while actual latency, energy, AP, TVM, and TensorRT evidence remains external and must be bound through verified manifests.

---

## 6. Release Outputs

After all tasks pass, the same commit produces:

### Public GitHub artifact

- complete repository history;
- public English and Chinese README files;
- `CITATION.cff`;
- Apache-2.0 license;
- GitHub Actions results;
- verified small artifacts and explicit external boundaries.

### Anonymous OpenReview artifact

- `aaai27_code_data_anonymous.zip`;
- no Git history or author identity;
- anonymous README;
- code, representative data, tests, artifact manifests, and reproducibility commands;
- SHA256 receipt and archive manifest stored outside the uploaded ZIP.

The public repository is not linked from the anonymous paper or anonymous supplementary materials during double-blind review.
