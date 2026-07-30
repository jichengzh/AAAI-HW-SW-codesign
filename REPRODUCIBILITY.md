# Reproducibility and evidence boundary

This package separates a runnable CPU-only smoke exercise from verified small
artifacts and from external experimental evidence. It is intentionally
fail-closed: a missing paper-evidence input produces `unavailable`, never a
demo substitute or a fabricated aggregate. Artifact inventory and checksums are
in [ARTIFACTS.md](ARTIFACTS.md).

## Environment and entry points

Use Python 3.10 or later and install the declared extras:

```bash
python --version
pip install -e '.[repro,dev]'
python scripts/reproduce/reproduce_all.py --mode smoke --output-root ./repro-smoke-output
```

The smoke invocation is offline and disables GPU, hardware, cache, TVM, and AP
execution. It materializes deterministic selection contracts from `data/demo/`
and records input/output SHA-256 values, package versions, source hashes, seeds,
and execution flags in `run_manifest.json`. Every demo record has
`paper_evidence: false`; smoke output is therefore **demo**, not paper evidence.

To audit the checked-in verified manifest, run:

```bash
python scripts/reproduce/reproduce_all.py --mode verified --output-root ./repro-verified-output
```

The current command deliberately exits non-zero after auditing Stage4, because
verified Stage6 evidence and the Stage7 formal aggregate are absent. Its failure
manifest retains the completed audit and marks the missing stages `unavailable`.

## Evidence matrix

The manuscript's exact table/figure labels are not part of this archive. The
following conservative mapping uses descriptive analysis items rather than
inventing a table number or a completed result. “Algorithm runs” counts complete
selection/analysis procedures only; it does not count timing iterations,
compiler retries, cache probes, or individual device observations.

| Paper table/figure or analysis item | Code | Input artifact | Seed | Algorithm runs | Evidence status |
| --- | --- | --- | --- | --- | --- |
| Stage4 cost-model selection audit (supporting analysis; manuscript figure/table label unavailable) | `framework/stage4/cost_model_selection_v1.py`; `scripts/reproduce/cost_model_selection.py` | `artifacts/verified/stage4/cost_model_selection_report.json`; `artifacts/verified/stage4/nested_cv_folds.csv`, enumerated by `artifacts/verified/manifest.json` | `20260716` | One nested grouped selection analysis on 176 measurements in 44 groups. It uses 5 outer folds and up to 3 inner folds per outer split; the 15 CSV rows are audit records, **not** 15 independent algorithm runs. | **verified** small, sanitized Stage4 artifact |
| Stage5 selection-request analysis (not a manuscript result) | `framework/stage5/production_search_v1.py`; `scripts/reproduce/stage5_selection.py` | `data/demo/` only in the public smoke path | `20260717` | One selection/request construction in smoke; no external execution | **demo**; `paper_evidence: false` |
| Main paper-table adapter (exact table label unavailable in this archive) | `framework/stage6/paper_table_v1.py`; `scripts/reproduce/stage6_table.py` | External terminal measurement, AP, energy, and independent-validation records | External execution seed/log required | No public completed algorithm run; adapter validates supplied records before table selection | **unavailable**; required inputs are **external** |
| Online-ablation table/figure aggregate (exact label unavailable in this archive) | `framework/stage7/online_component_ablation_v1.py`; `framework/stage7/ablation_statistics_v2.py`; `scripts/reproduce/stage7_selection.py` | External terminal trajectory records and formal aggregate; no Stage7 aggregate is in `artifacts/verified/` | `20260718`, `20260719`, `20260720` | Required formal design: 12 trajectories across four variants and three seeds, with 192 selected events total. The 192 selected events are terminal observations, not 192 algorithm runs. | **unavailable**; formal aggregate is **external** pending closure |
| Any remaining manuscript table or figure | No unique mapping can be established from package contents | Not supplied in the package | Not supplied | Not established | **unavailable** |

The Stage7 smoke step is narrower than the formal design: it produces only one
fixed `full` selection-only request using seed `20260718`. It does not perform a
hardware measurement or create a Stage7 result. In particular, the existence of
selection contracts, 12 trajectories, or 192 selected events must not be read as
evidence that a formal aggregate has been verified.

## Seed and repetition rules

- Stage4 uses seed `20260716`. The nested procedure has 5 outer grouped folds
  and up to 3 inner grouped folds; folds are validation partitions within one
  analysis, not independent repetitions.
- Stage5 uses seed `20260717` for deterministic selection/request construction.
- Stage7's frozen experimental seeds are `20260718`, `20260719`, and
  `20260720`. A formal repetition is a complete trajectory for one
  variant–seed pair; each trajectory has four rounds of four selections.
- A timing iteration, device retry, compilation retry, cache warm-up, or AP
  evaluation repeat is not an algorithm run. It belongs in external execution
  provenance and must be reported separately if it supports a paper claim.

## Artifact provenance and dataset access

The demo fixture is synthetic and deterministic. It is not a benchmark,
measurement corpus, or access route to a full dataset. The verified Stage4
files retain source-data content hashes while removing local source locations.
Verified mode recomputes each declared SHA-256, checks schema and verification
status, and refuses a mismatch.

Full datasets, model code, checkpoints, ONNX exports, compiled engines, device
telemetry, TVM/TensorRT outputs, and AP/energy/latency measurements are external
inputs. They are not downloaded or discovered by the scripts. Before connecting
them to an analysis, retain an immutable input copy or digest, a non-path
provenance label, execution configuration, seed, and independent-validation
record. External hardware results are outside this package's reproducibility
claim until such a reviewed artifact bundle is released.

## Known limitations

Only the small Stage4 artifact set is currently verified. Stage6 representative
selection and the Stage7 formal aggregate are unavailable, so this archive
cannot reproduce a hardware/AP/energy paper table or an ablation aggregate.
The smoke workflow demonstrates interfaces, not numerical conclusions. It does
not validate any device, backend, compiler, cache, dataset, checkpoint, ONNX,
or engine result.

## License and citation

This package is licensed under [Apache-2.0](LICENSE). Cite the collective public
metadata in [CITATION.cff](CITATION.cff).
