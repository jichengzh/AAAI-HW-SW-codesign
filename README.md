# GEAR Co-design reproducibility package

This repository is a scoped reproducibility artifact for the submission. It
provides a deterministic CPU-only smoke workflow, a small verified cost-model
selection audit, and public interfaces for candidate selection, evidence
validation, result aggregation, and release inspection. It does not bundle or
execute external hardware measurements, AP evaluation, model checkpoints, ONNX
files, compiled engines, TVM, or TensorRT.

For evidence boundaries, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[ARTIFACTS.md](ARTIFACTS.md). The reviewer-facing anonymous entry point is
`README.anonymous.md`; it intentionally is not the public project README.
The maintained release handoff, current state, and path to complete open source
are recorded in [docs/AAAI27_RELEASE_AUDIT.md](docs/AAAI27_RELEASE_AUDIT.md).

## Quick start: clean clone and CPU-only smoke

Use Python 3.10--3.13. During anonymous review, obtain the HTTPS clone URL from
the repository page and set it locally; keeping the URL outside this source
tree preserves the review boundary. A shallow, partial clone avoids downloading
unrelated history:

```bash
export GEAR_REPOSITORY_URL='<HTTPS clone URL from the repository page>'
git clone --depth 1 --filter=blob:none --single-branch \
  --branch release/aaai27-reproducibility "$GEAR_REPOSITORY_URL" gear-codesign
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

To independently check that a fresh clone, new virtual environment, and smoke
path work together, run this from any existing checkout:

```bash
bash scripts/reproduce/smoke_clean_clone.sh \
  --repo-url "$GEAR_REPOSITORY_URL" \
  --ref release/aaai27-reproducibility
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

## What is included

```text
data/demo/                     Deterministic smoke-only fixtures
artifacts/verified/            Small, sanitized cost-model audit and SHA-256 manifest
framework/                     Scanning, selection, validation, and aggregation modules
scripts/reproduce/             CPU-only smoke and verified-boundary entry points
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
candidate-request construction, evidence boundaries, and result aggregation.
The included lightweight deterministic selection policy exercises the public
candidate and feedback interfaces; it is not a replacement for, or an
equivalence claim about, the complete evolutionary candidate generator
described in the paper.

The current artifact also omits the full model materialization and training
stack, hardware scheduling and measurement executors, paper-specific online
ablation pipelines, and their complete trajectory and terminal-evidence
manifests. Consequently, the smoke workflow validates interfaces and
provenance handling but does not reproduce the paper's hardware tables or full
online-ablation results. We plan to release the complete implementation and the
corresponding experiment manifests upon publication.

## License and citation

The package is distributed under the [Apache-2.0 license](LICENSE). If you use
the software, cite the collective metadata in [CITATION.cff](CITATION.cff).
