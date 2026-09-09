# GEAR Co-design reproducibility package

This repository is a scoped reproducibility artifact for the submission. It
provides a deterministic CPU-only smoke workflow, a small verified cost-model
selection audit, and public interfaces for candidate selection, evidence
validation, result aggregation, release inspection, and an external private
hardware-run interface. It does not bundle datasets, model checkpoints, ONNX
files, compiled engines, TVM caches, raw logs, private measurement outputs, or
private source trees. The public CPU workflows below do not execute hardware
measurements, AP evaluation, TVM, or TensorRT.

For evidence boundaries, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[ARTIFACTS.md](ARTIFACTS.md). The reviewer-facing anonymous entry point is
`README.anonymous.md`; it intentionally is not the public project README.
The maintained release handoff, current state, and path to complete open source
are recorded in [docs/AAAI27_RELEASE_AUDIT.md](docs/AAAI27_RELEASE_AUDIT.md).

## Quick start: clean clone and CPU-only smoke

Use Python 3.10--3.13. During anonymous review, obtain the HTTPS clone URL from
the repository page and set it locally; keeping the URL outside this source
tree preserves the review boundary. A shallow, partial clone avoids downloading
unrelated history. `feat/runtime-gpu-pool` is the current full-chain candidate
branch; the stable anonymous CPU-only release branch remains
`release/aaai27-reproducibility`:

```bash
export GEAR_REPOSITORY_URL='<HTTPS clone URL from the repository page>'
git clone --depth 1 --filter=blob:none --single-branch \
  --branch feat/runtime-gpu-pool "$GEAR_REPOSITORY_URL" gear-codesign
cd gear-codesign
```

Create a CPU-only environment and run the deterministic smoke workflow:

```bash
python --version
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python scripts/reproduce/reproduce_all.py --mode smoke --output-root ./repro-smoke-output
```

Smoke is deterministic, offline, CPU-only, and uses only the small fixtures in
`data/demo/`. Its output is a contract exercise, not paper evidence: every
source record is explicitly marked `paper_evidence: false`. The resulting
`run_manifest.json` records input and output SHA-256 identities, source hashes,
frozen seeds, and execution boundaries. Re-running a matching completed output
directory is a no-op; a directory containing different data is rejected.

To reproduce the public three-model formal search spaces from the checked-in
purified scanner evidence, run:

```bash
python scripts/reproduce_paper_search_space.py --model all
```

This command is CPU-only and offline. It validates the Pyramid, CoDriving, and
F-Cooper scanner-contract fixtures through the public Stage1 bridge and generic
Stage2 formal planner, then prints derived structure/candidate counts. It does
not train, measure latency/energy/AP, invoke TVM/TensorRT, or access datasets,
checkpoints, GPUs, SSH, or private paths.

To independently check that a fresh clone, new virtual environment, and smoke
path work together, run this from any existing checkout:

```bash
bash scripts/reproduce/smoke_clean_clone.sh \
  --repo-url "$GEAR_REPOSITORY_URL" \
  --ref feat/runtime-gpu-pool
```

This command does not download datasets, checkpoints, ONNX files, compiled
engines, TVM/TensorRT artifacts, or hardware measurements. It retains its
temporary work directory for inspection.

The checked-in verified bundle can be audited with the same entry point:

```bash
python scripts/reproduce/reproduce_all.py --mode verified --output-root ./repro-verified-output
```

This command is intentionally expected to exit non-zero today. It verifies the
available cost-model audit bytes, then reports `unavailable` because the
hardware-backed representative results and formal online-ablation aggregate are
not in the package. Treat that non-zero result as an auditable availability
check, not a successful paper run.

## External private H800 / RTX4090 full-chain run

H800 and RTX4090 use the same Stage1, Stage2, GPU-admission, controller, and
verifier path. Only the hardware profile, CUDA/TVM architecture, and external
assets differ. This is not the CPU smoke path, and the repository does not
download or fabricate datasets, checkpoints, model sources, ONNX/calibration
inputs, TVM caches, or hardware measurements.

Material ownership is explicit:

- **User-provided private inputs:** the private history root (the exact Git
  top-level), a
  completed v5 private source-map, a pre-normalization runner-template,
  licensed datasets, model sources/checkpoints, any required ONNX or calibration inputs,
  a profile-compatible CUDA/TVM toolchain (a CUDA/TVM sm89 toolchain
  for RTX4090), a not-yet-created normalized directory, and a fresh output root. The file formerly
  described as an RTX local config or locator is no longer a user input.
- **Repository-provided public references:** the four-round contracts are
  `configs/execution/p6_h800_search.example.yaml` and
  `configs/execution/p6_rtx4090_search.example.yaml`; the single-command
  manifests are `configs/execution/p6_h800_full_chain.example.yaml` and
  `configs/execution/p6_rtx4090_full_chain.example.yaml`. Complete schema shapes
  are shown by `configs/execution/p6_history_source_map.example.yaml`,
  `configs/execution/p6_history_runner_template.example.yaml`, and the
  null-only schema example
  `configs/execution/p6_external_training_binding.example.yaml`. The four
  source JSON payloads have separate format references:
  `p6_gold176_rows.example.json`, `p6_gold176_graph_features.example.json`,
  `p6_closure.example.json`, and either
  `p6_h800_capability_profiles.example.json` or
  `p6_rtx4090_capability_context.example.json`, all in `configs/execution/`.
  All examples are redacted. A manifest with `template_only: true` is
  intentionally rejected.
- **`normalize`-generated private outputs:**
  `<abs-normalized-private-dir>/legacy.local.yaml`,
  `<abs-normalized-private-dir>/runner-template.yaml`,
  `<abs-normalized-private-dir>/source-wrapper-profile.yaml`,
  `<abs-normalized-private-dir>/external-training-binding.yaml`,
  `<abs-normalized-private-dir>/post-source-adapter-profile.yaml`, and the
  derived recipe. Do not hand-author or mix these authorities. The files
  `configs/execution/p6_h800_local_locator.example.yaml` and
  `configs/execution/p6_rtx4090_local_locator.example.yaml` document only the
  generated shape; they cannot replace normalize output. An explicit
  `hardware_profile: h800` uses the parallel v3 locator; legacy H800 authority
  that omits the profile remains compatible with v2.

Copy the chosen full-chain manifest outside the repository or to a Git-ignored
location, set `template_only: false`, replace every path with an absolute path,
and ensure the manifest, source-map, and contract select the same profile. The
static check parses the four subordinate JSON payloads, verifies their
cardinalities, identities, closure, profiles, and self-consistent digests. It
does not probe a GPU and creates neither normalized nor run outputs:

```bash
python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-full-chain-manifest.yaml> \
  --check-inputs
```

After that check passes, use the same recommended entry point for either
profile. `N` is only the requested GPU count:

```bash
# H800: manifest copied from p6_h800_full_chain.example.yaml
GPU_POOL=N python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-h800-full-chain-manifest.yaml>

# RTX4090: manifest copied from p6_rtx4090_full_chain.example.yaml
GPU_POOL=N python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-rtx4090-full-chain-manifest.yaml>
```

Once launched, the command runs derive, normalize, fresh provision, the four
controller rounds, and independent verification in order. No intermediate file
editing, device selection, or copying is required. A failed stage stops the
sequence with a stage-specific error and is never promoted to completion.

The JSON examples are **format references, not runnable data**. The Gold input
must contain exactly 176 real rows and one matching graph-feature row per
`group_id`; its two H800 capability profiles must retain their exact historical
digests. The RTX4090 capability context must be generated from the measured
probe/rebuild workflow and its embedded bytes and digests, not assembled by
editing the example. Dataset, checkpoint, config, model source, ONNX, and
calibration files keep their native upstream formats; the v5 source-map binds
their paths and the supported checkpoint/config SHA-256 identities.

On success, `<fresh_output_root>/state.json` is the controller completion state;
`binding.json` and `local-config.yaml` bind the admitted run; and
`round-00/feedback.json` through `round-03/feedback.json` contain the 16 released
rows and their `latency_ms`, `energy_j`, `ap30`, `ap50`, and `ap70` values. The
normalized runner's `execution_interface.actual_feedback` paths retain the
corresponding private native evidence and receipts. Treat metrics as final only
after the independent verifier has printed its `completed` report. That report
proves four rounds and 16 distinct measurements but deliberately does not pick
a “best” row or publish private metric values. Dataset identity is currently a
path-bound external input rather than a snapshot digest, so this supports
full-chain mechanism validation, not an exact paper-number claim.

### Historical four-step interface (debug only)

The single-command entry point internally invokes `derive_p6_history_recipe.py`,
`normalize_p6_history_root.py`, the historically named
`run_p6_h800_search.py`, and `verify_p6_materializer_training_run.py`. Run them
separately only to diagnose a stage failure. The third-step shape is retained
below for old-log diagnosis:

```bash
GPU_POOL=7 python tools/release/run_p6_h800_search.py \
  --contract configs/execution/p6_rtx4090_search.example.yaml \
  --code-revision "$(git rev-parse --short=12 HEAD)" \
  --legacy-local-config <abs-normalized-private-dir>/legacy.local.yaml \
  --runner-template <abs-normalized-private-dir>/runner-template.yaml \
  --local-output-root <abs-fresh-output-root> \
  --binding-output <abs-fresh-output-root>/binding.json \
  --config-output <abs-fresh-output-root>/local-config.yaml \
  --source-wrapper-profile <abs-normalized-private-dir>/source-wrapper-profile.yaml \
  --external-training-binding <abs-normalized-private-dir>/external-training-binding.yaml \
  --post-source-adapter-profile <abs-normalized-private-dir>/post-source-adapter-profile.yaml
```

`--legacy-local-config` and `run_p6_h800_search.py` are historical names. The
manifest-selected v3 public contract is the profile authority, and the locator
always comes from the current normalize operation.

`GPU_POOL` is a strict positive GPU count and the only selector in fresh-run
mode. `GPU_POOL=1` requests one available GPU, `GPU_POOL=3` requests three, and
`GPU_POOL=7` requests seven. The caller does not name physical GPU indices and
no persistent UUID is configured in advance. Live admission automatically
selects a stable idle subset that matches the hardware profile, then records
the selected indices and UUIDs only in that fresh run's private binding. The controller derives the leaf
binding from the admitted ordered policy instead of assuming every native leaf
receives the same pool. Some leaves consume the full pool, including native
performance planning/execution through `gpu_pool`; source materialization,
quantization, and AP shards receive single-card assignments where the adapter
contract requires it. Within a round, candidate/source work may run
concurrently across distinct GPUs, while work assigned to the same GPU is
serialized. The four search rounds still execute in order because each later
round consumes the previous round's verified feedback.

The legacy `--local-config` mode remains available. When `GPU_POOL` is supplied
with an existing local config, its count must match that run's existing
binding; use fresh-run mode for automatic device selection.

H800 and RTX4090 measurements are hardware-specific evidence. They must never
be relabeled, merged, or numerically adjusted across profiles. RTX4090 can
validate the sm89 end-to-end mechanism but cannot stand in for H800 paper data.
The maintained implementation has completed one private RTX4090 mechanism
validation with `GPU_POOL=3`, four rounds, 16 selected/measured rows, and an
independent verifier pass. This structural fact does not publish the private
result bundle or expand the repository's paper-evidence claim.

## What is included

```text
data/demo/                     Deterministic smoke-only fixtures
artifacts/verified/            Small, sanitized cost-model audit and SHA-256 manifest
framework/                     Scanning, selection, validation, and aggregation modules
framework/reproduction/        Public CPU-only paper search-space reproduction gate
scripts/reproduce/             CPU-only smoke and verified-boundary entry points
tools/release/                 Public release, private-configuration, and verifier CLIs
tests/                         Unit, integration, and release checks
```

The smoke workflow exercises cost-model selection, a candidate measurement
request, representative-result validation, and one fixed online-ablation
selection request. It never runs a device backend or claims a measured result.

## Scope and external boundary

The included modules are released as input-validation, selection-interface,
aggregation, and audit logic. A hardware latency, energy, AP, TVM, TensorRT,
checkpoint, ONNX, or engine result becomes evidence only when separately
supplied with its own provenance, immutable inputs, and verified hashes. This
repository neither downloads those materials nor substitutes synthetic data
for them.

The bundled demo data is not a dataset access path. Access to full datasets,
model source trees, trained checkpoints, and hardware systems is external and
subject to the relevant provider's terms. Attach such inputs only through the
explicit command interfaces and keep their provenance separate from this
anonymous/public package.

## Reproducibility rules

The frozen cost-model selection seed is `20260716`; the deterministic
selection/request exercise uses `20260717`; formal online-ablation trajectories
use `20260718`, `20260719`, and `20260720`. A seed makes a selection or analysis
deterministic, but it does not create a hardware run. An algorithm run is one
complete execution of a specified selection/analysis procedure on fixed inputs.
Timing iterations, compiler retries, cache probes, and repeated device
measurements are observations within an external execution, not additional
algorithm runs. See the evidence matrix for the precise counts and status in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Development checks

The release-oriented documentation and identity checks are runnable locally:

```bash
pytest tests/release/test_identity_scan.py -q
```

Use `python scripts/reproduce/reproduce_all.py --help` to inspect the supported
reproduction arguments. The fixed `requirements.txt` environment supports the
CPU smoke and the current test suite; optional extras in `pyproject.toml` remain
machine-readable dependency groups rather than the recommended release install.

## Scope of the released implementation

This repository is not the complete training and hardware-search implementation
used for every result in the paper. The released code is intended to let
reviewers inspect data contracts, grouped cost-model selection, deterministic
candidate-request construction, evidence boundaries, result aggregation, and
the public side of the private hardware-run contract. The included lightweight
deterministic selection policy exercises the public candidate and feedback
interfaces; it is not a replacement for, or an equivalence claim about, the
complete evolutionary candidate generator described in the paper.

The CPU smoke workflow validates interfaces and provenance handling but does
not reproduce the paper's hardware tables or full online-ablation results. Any
full hardware run requires external private assets and separately verified
provenance.

## License and citation

The package is distributed under the [Apache-2.0 license](LICENSE). If you use
the software, cite the collective metadata in [CITATION.cff](CITATION.cff).
