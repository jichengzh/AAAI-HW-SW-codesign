# Artifact inventory and acquisition boundary

This page classifies every reproducibility input by evidence status. A class is
not a quality ranking: it states what the package can support today. No category
authorizes a paper claim beyond its recorded provenance.

| Class | Manifest or path | SHA-256 and validation | Acquisition / connection rule |
| --- | --- | --- | --- |
| **demo** | `data/demo/manifest.json` and its four files under `data/demo/` | The demo manifest declares `paper_evidence: false`. A smoke `run_manifest.json` records the SHA-256 identity of each consumed input and output. | Bundled only for deterministic CPU-only smoke. Never report it as a benchmark, measurement, or paper result. |
| **verified** | `artifacts/verified/manifest.json`; `artifacts/verified/stage4/cost_model_selection_report.json`; `artifacts/verified/stage4/nested_cv_folds.csv` | The verified manifest lists each path, schema, verification status, source-data content hashes, and SHA-256. `python scripts/reproduce/reproduce_all.py --mode verified --output-root ./repro-verified-output` recomputes and checks every listed digest. | Checked into this package. It currently supports only the small Stage4 audit; it is not a complete experimental bundle. |
| **external** | Hardware latency/energy/AP evidence, TVM/TensorRT outputs, checkpoints, ONNX exports, compiled engines, full datasets, and device execution logs | No bytes or manifest are bundled here. A supplied bundle must carry immutable input identities, SHA-256 values, non-path provenance labels, seed/configuration, and independent-validation status. | Obtain from the applicable data/model/hardware provider under its terms. Supply only through explicit analysis inputs; scripts do not discover or download it. |
| **unavailable** | Stage6 representative-selection evidence and Stage7 formal aggregate | No checked-in manifest entry can validate absent bytes. Verified mode records `unavailable` after it validates the available Stage4 subset. | Do not replace with demo, zero-filled, synthetic, or surrogate records. Release a reviewed external bundle before promoting this status. |

## Current verified scope

The only current **verified** artifacts are the two sanitized Stage4 files in
`artifacts/verified/stage4/`. Their manifest labels the first as
`stage4_cost_model_selection_v1` and the second as `stage4_nested_cv_folds_v1`.
The selection report records seed `20260716`, 176 measurements, 44 groups, 5
outer splits, and 3 inner splits. The fold CSV is an audit trace, not a count of
independent algorithm runs.

`artifacts/README.md` records the related Stage7 boundary: no Stage7 artifact is
listed until a formal aggregate confirms all 12 trajectories and 192 selected
events. The verified mode therefore has an expected non-zero outcome today; it
does not turn Stage4 bytes into a verified Stage6 or Stage7 result.

P6.1 has completed a separate Git-ignored local execution closure. That fact
does not add a checked-in artifact, a public result summary, or a verified
Stage6/Stage7 entry: all inputs and outputs of that run remain external under
the rules above.

## How to inspect an artifact

Read the manifest before consuming a file. For a **verified** entry, compare the
path, schema, status, and SHA-256 from `artifacts/verified/manifest.json` with
the file bytes; the verified reproduction mode performs this check and refuses
mismatches. For **demo** inputs, inspect `data/demo/manifest.json` and the smoke
run manifest, which binds the exact input hashes used for that run. For an
**external** bundle, do not attach it until its provenance and hashes meet the
requirements in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## What is deliberately absent

The package contains no full dataset, trained model, private model source,
checkpoint, ONNX file, compiled engine, device cache, TVM/TensorRT build output,
latency/energy trace, AP evaluation output, P6.1 execution result bundle, or
final Stage6/Stage7 paper aggregate. It also does not contain the local model
materialization and training assets, hardware measurement inputs, complete
online-ablation outputs, formal trajectories, or terminal-evidence manifests.
Those absences are intentional release boundaries, not implicit permissions to
reconstruct or infer missing numbers. Public paper evidence requires a
separately reviewed artifact bundle.

## License and citation

The source package is licensed under [Apache-2.0](LICENSE). Cite it using the
collective metadata in [CITATION.cff](CITATION.cff).
