# GEAR Co-design reproducibility package

This repository provides a CPU-only smoke workflow, a small verified Stage4
artifact set, and the Stage1–7 selection and analysis contracts used to audit
the submission. It does not bundle or execute external hardware measurements,
AP evaluation, model checkpoints, ONNX files, compiled engines, TVM, or TensorRT.

For evidence boundaries, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[ARTIFACTS.md](ARTIFACTS.md). The reviewer-facing anonymous entry point is
`README.anonymous.md`; it intentionally is not the public project README.

## Quick start: CPU-only smoke

Use Python 3.10 or later. From a source checkout, install the reproducibility
and development extras, then choose a new or empty output directory:

```bash
python --version
pip install -e '.[repro,dev]'
python scripts/reproduce/reproduce_all.py --mode smoke --output-root ./repro-smoke-output
```

Smoke is deterministic, offline, CPU-only, and uses only the small fixtures in
`data/demo/`. Its output is a contract exercise, not paper evidence: every
source record is explicitly marked `paper_evidence: false`. The resulting
`run_manifest.json` records input and output SHA-256 identities, source hashes,
frozen seeds, and execution boundaries. Re-running a matching completed output
directory is a no-op; a directory containing different data is rejected.

The checked-in verified bundle can be audited with the same entry point:

```bash
python scripts/reproduce/reproduce_all.py --mode verified --output-root ./repro-verified-output
```

This command is intentionally expected to exit non-zero today. It verifies the
available Stage4 artifact bytes, then reports `unavailable` because the required
Stage6 evidence and Stage7 formal aggregate are not in the package. Treat that
non-zero result as an auditable availability check, not a successful paper run.

## What is included

```text
data/demo/                     Deterministic smoke-only fixtures
artifacts/verified/            Small, sanitized Stage4 audit artifacts and SHA-256 manifest
framework/stage1/              Model scanning and classification contracts
framework/stage4/              Nested grouped cost-model selection
framework/stage5/              Selection and measurement-request contracts
framework/stage6/              External-evidence paper-table adapter
framework/stage7/              Selection-only online-ablation contracts and statistics
scripts/reproduce/             CPU-only smoke and verified-boundary entry points
tests/                         Unit, integration, and release checks
```

The smoke workflow invokes the public Stage4 selector, Stage5 selection request,
Stage6 representative-selection adapter, and one fixed Stage7 selection-only
round. It never runs a device backend or claims a measured result.

## Scope and external boundary

Stage1–7 code is released as input-validation, selection, aggregation, and
audit logic. A hardware latency, energy, AP, TVM, TensorRT, checkpoint, ONNX,
or engine result becomes evidence only when separately supplied with its own
provenance, immutable inputs, and verified hashes. This repository neither
downloads those materials nor substitutes synthetic data for them.

The bundled demo data is not a dataset access path. Access to full datasets,
model source trees, trained checkpoints, and hardware systems is external and
subject to the relevant provider's terms. Attach such inputs only through the
explicit command interfaces and keep their provenance separate from this
anonymous/public package.

## Reproducibility rules

The frozen Stage4 selection seed is `20260716`; Stage5 uses `20260717`; formal
Stage7 trajectories use `20260718`, `20260719`, and `20260720`. A seed makes a
selection or analysis deterministic, but it does not create a hardware run.
An algorithm run is one complete execution of a specified selection/analysis
procedure on fixed inputs. Timing iterations, compiler retries, cache probes,
and repeated device measurements are observations within an external execution,
not additional algorithm runs. See the evidence matrix for the precise counts
and status in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Development checks

The release-oriented documentation and identity checks are runnable locally:

```bash
pytest tests/release/test_identity_scan.py -q
```

Use `python scripts/reproduce/reproduce_all.py --help` to inspect the supported
reproduction arguments. The complete test suite may exercise additional Python
dependencies; the smoke quick start above needs only the declared `repro` and
`dev` extras.

## License and citation

The package is distributed under the [Apache-2.0 license](LICENSE). If you use
the software, cite the collective metadata in [CITATION.cff](CITATION.cff).
