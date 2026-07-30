# Reproducibility entrypoint

This repository separates a CPU-only smoke exercise from artifact-backed
analysis. Neither command performs GPU work, hardware measurement, cache
execution, TVM compilation, or AP evaluation.

## Smoke

Install the declared local reproduction dependencies, then provide a new or
empty output directory:

```bash
pip install -e '.[repro,dev]'
python scripts/reproduce/reproduce_all.py --mode smoke --output-root results/repro_smoke
```

Smoke runs the public Stage2 demo generator and materializes deterministic
selection contracts solely from `data/demo/`. It verifies that every source
record is marked `paper_evidence: false`, then calls the existing public Stage4
selector, Stage5 selection-only request publisher, Stage6 representative
selector, and one fixed `full` Stage7 selection-only round. The Stage7 round
publishes a request; it never measures a device or runs a backend.

Smoke output is deliberately not paper evidence and must not be interpreted as
reproducing reported experimental results. The output root contains
`run_manifest.json`, input/output SHA-256 identities, component source hashes,
frozen seeds, selection-stage status, and UTC start/end times. The manifest
does not hash itself. Re-running a completed matching root is a no-op; a root
with different content is refused.

## Verified

```bash
python scripts/reproduce/reproduce_all.py --mode verified --output-root results/repro_verified
```

Verified mode reads only `artifacts/verified/manifest.json`, validates every
declared path, SHA-256, schema label, and verification status, and runs only
analysis supported by those checked-in inputs. It never falls back to demo
data or synthetic substitutes.

The current public verified artifact set contains Stage4 audit artifacts but
does not include the required Stage6 evidence or Stage7 formal aggregate.
Accordingly, this command currently exits non-zero with an explicit
`unavailable` state and atomically records a failure manifest. That failure is
intentional and must not be treated as a successful paper reproduction.

## External measurement and paper evidence

Hardware measurement, model compilation, caches, and unpublished evidence are
outside this entrypoint. They require separately supplied, auditable inputs and
their own release approval; this command will not discover or download them.
