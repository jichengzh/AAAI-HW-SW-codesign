# Reproducibility and evidence boundary

This package separates a runnable CPU-only smoke exercise from verified small
artifacts and from external experimental evidence. It is intentionally
fail-closed: a missing paper-evidence input produces `unavailable`, never a
demo substitute or a fabricated aggregate. Artifact inventory and checksums are
in [ARTIFACTS.md](ARTIFACTS.md).

## Environment and entry points

Use Python 3.10--3.13. For the public CPU-only smoke, install the pinned
requirements file and the local package without resolving a second dependency
set:

```bash
python --version
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
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

To reproduce the public formal search spaces for the three CoptV2X paper
models, run:

```bash
python scripts/reproduce_paper_search_space.py --model all
```

This gate is deterministic, offline, and CPU-only. It reads only the three
checked-in purified scanner-evidence fixtures under `tests/fixtures/paper_spaces/`,
derives scanner-owned structural axes, validates them through the public Stage1
bridge, and enumerates the generic Stage2 formal software plan. The printed
counts are a search-space contract check, not measured accuracy, latency,
energy, compiler, training, checkpoint, dataset, GPU, SSH, TVM, or TensorRT
evidence.

## External H800 / RTX4090 full-chain contract

The preferred external-hardware entry point is the manifest-driven orchestrator:

```bash
python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-full-chain-manifest.yaml> \
  --check-inputs
GPU_POOL=N python tools/release/run_p6_full_chain.py \
  --manifest <abs-private-full-chain-manifest.yaml>
```

The static check is read-only, parses and validates all four subordinate JSON
payloads, and does not require `GPU_POOL`. It proves content consistency but
does not replace live GPU/runtime admission. A full invocation
requires a positive GPU count and, after it starts, runs derive, normalize,
fresh provision, four sequential feedback-dependent search rounds, and the
independent verifier without intermediate operator edits. Redacted H800 and
RTX4090 manifests and input-shape examples live under `configs/execution/`.
They document schemas but do not supply licensed bytes or measured evidence.
The JSON references cover Gold rows, matching graph features, the closure audit,
the two-profile H800 authority, and the measured RTX4090 capability-context
shape. They are deliberately non-executable: cardinalities, content hashes,
embedded probe bytes, and measured values must come from the approved source or
probe workflow.
In particular, locator, normalized runner, wrapper profiles, and normalized
training binding are outputs of the same normalize operation and must not be
hand-authored or mixed across runs.

After a successful verifier pass, the controller state is
`<fresh_output_root>/state.json`; released per-candidate metrics are in the four
`round-XX/feedback.json` files. Native evidence and receipts remain under the
paths declared by the normalized runner's `execution_interface.actual_feedback`
contract. The verifier completion JSON is a structural/provenance report, not a
best-candidate metric summary. The current external-training contract does not
bind a dataset snapshot digest, so a completed private run is mechanism evidence
unless a separately reviewed dataset/result bundle supplies that missing
identity.

The hardware profile is an evidence invariant. An H800/sm90 observation is not
RTX4090/sm89 evidence, and an RTX4090/sm89 observation is not H800/sm90
evidence. Selection seeds and a shared controller do not authorize relabeling,
merging, or numerically adjusting results across profiles.

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
| CoptV2X formal paper-model search-space contract | `framework/reproduction/coptv2x_paper_space_v1.py`; `scripts/reproduce_paper_search_space.py` | `tests/fixtures/paper_spaces/{pyramid,codriving,fcooper}_scanner_evidence.yaml` | Not randomized | One CPU-only derivation/enumeration per model; no training or measurement | **verified contract** for formal search-space size only; not numerical paper evidence |
| Main paper-table adapter (exact table label unavailable in this archive) | `framework/stage6/paper_table_v1.py`; `scripts/reproduce/stage6_table.py` | External terminal measurement, AP, energy, and independent-validation records | External execution seed/log required | No public completed algorithm run; adapter validates supplied records before table selection | **unavailable**; required inputs are **external** |
| Online-ablation table/figure aggregate (exact label unavailable in this archive) | `framework/stage7/online_component_ablation_v1.py`; `framework/stage7/ablation_statistics_v2.py`; `scripts/reproduce/stage7_selection.py` | External terminal trajectory records and formal aggregate; no Stage7 aggregate is in `artifacts/verified/` | `20260718`, `20260719`, `20260720` | Required formal design: 12 trajectories across four variants and three seeds, with 192 selected events total. The 192 selected events are terminal observations, not 192 algorithm runs. | **unavailable**; formal aggregate is **external** pending closure |
| Any remaining manuscript table or figure | No unique mapping can be established from package contents | Not supplied in the package | Not supplied | Not established | **unavailable** |

The Stage7 smoke step is narrower than the formal design: it produces only one
fixed `full` selection-only request using seed `20260718`. It does not perform a
hardware measurement or create a Stage7 result. In particular, the existence of
selection contracts, 12 trajectories, or 192 selected events must not be read as
evidence that a formal aggregate has been verified.

## Method-to-artifact mapping

The public smoke path uses `predicted_frontier_diversity` as a deterministic,
lightweight exercise of candidate-selection and feedback interfaces. It is not
the complete NSGA-II candidate generator described in the manuscript and does
not establish algorithmic or numerical equivalence with that generator.

Likewise, the included online-ablation modules expose selection-only contracts,
variant input projections, request identities, and descriptive aggregation
logic. They are not the complete paper-specific ablation pipelines and cannot
reproduce the manuscript's online-ablation table without the missing formal
trajectories, terminal evidence, model bundles, and hardware execution records.
The presence of these interfaces does not expand the verified evidence scope
beyond the small Stage4 audit listed above.

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

The public full-chain templates improve the bring-your-own-assets handoff but
do not change that evidence status. `--check-inputs` can establish that the
declared local files and profile agree; it cannot create missing private assets
or confer permission to redistribute them.

## Known limitations

The checked-in verified artifact set remains the small Stage4 subset. P6.1 has
completed a separate, Git-ignored local execution closure, and an
RTX4090 three-card four-round local mechanism validation has completed with
16 selected rows accepted by the closeout verifier. Neither local closure
contributes a checked-in result bundle, formal execution manifest, or public
numerical claim.
Stage6 representative selection and the Stage7 formal aggregate remain
unavailable, so this archive cannot reproduce a hardware/AP/energy paper table
or an ablation aggregate. The smoke workflow demonstrates interfaces, not
numerical conclusions. It does not validate any device, backend, compiler,
cache, dataset, checkpoint, ONNX, or engine result. Any future public paper
evidence requires a separately reviewed artifact bundle.

## License and citation

This package is licensed under [Apache-2.0](LICENSE). Cite the collective public
metadata in [CITATION.cff](CITATION.cff).
