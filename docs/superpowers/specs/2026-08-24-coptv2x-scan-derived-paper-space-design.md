# CoptV2X Scan-Derived Paper Space Design

## Goal

Repair Stage1 → Stage2 formal search-space construction so the public
framework reproduces the three CoptV2X paper model spaces, while the
production implementation remains driven by graph scanning, dependency
relations, materializer bindings, base checkpoint widths, and backend
hardware constraints instead of paper constants.

## Non-goals

- Do not add an arbitrary four-axis synthetic network such as `6×8×10×12`.
  Acceptance is limited to the three paper models: Pyramid, CoDriving, and
  F-Cooper.
- Do not start post-source adapters, SSH jobs, H800 training, or the four-round
  real search in this repair plan. Those resume only after the CPU-only formal
  space contract is correct.
- Do not use `3`, `5`, `7`, `343`, `686`, `1792`, or `3584` as production
  generation drivers. These values are allowed only as test or CLI summary
  oracles for the three paper models.

## Binding requirements

- Formal production enumeration consumes `structural_axes` only. Legacy
  `view_b1_search_groups` and probe anchors are diagnostic views; they are not
  fallback production axes.
- If scanner-owned `structural_axes` are absent, malformed, or incomplete,
  Stage2 formal construction fails closed with a clear validation error.
- The scanner-owned structural-axis builder consumes low-level prune groups,
  graph dependency/dataflow relations, adapter materialization bindings, base
  config/checkpoint widths, and hardware/backend constraints. It must not return
  fixed axis counts or fixed axis sets from model name.
- Adapter code does not return final axes, paper width sets, or paper counts.
  It returns a frozen trace context plus selector declarations. The scanner
  resolves those selectors against the traced network, loaded config, loaded
  checkpoint/module tensors, and DepGraph groups.
- Each axis records canonical base width, legal widths, member B1 groups,
  integer-ratio transforms from canonical coordinate to member coordinate,
  provenance, and whether the axis is free or fixed-derived.
- Probe anchors, cliff-neighbor diagnostics, and low/mid/high samples remain
  available only as diagnostic metadata and never truncate the formal universe.
- q modes are derived as:
  `hardware precision ∩ backend support ∩ configured compression modes ∩ graph quant units`.
  An empty intersection is an error.
- A formal candidate must never require a canonical axis width above that
  axis's base checkpoint/config width. Over-base rows are rejected before
  registry writes; they are not retained as normal `unavailable` candidates.
- The formal planner consumes a generic `axis_schema`; P6 Pyramid code is only a
  thin adapter over that schema.

## Trace adapter and scenario contract

Add an adapter module for model-specific trace-context extraction:

- `framework/stage1/adapters.py`

The adapter interface is explicit and narrow. Define a `TraceAdapter`
protocol, or an equivalent interface with the same responsibilities:

```python
class TraceAdapter(Protocol):
    def build_trace_context(
        self,
        net: Any,
        example_inputs: tuple[Any, ...],
        loaded_config: Mapping[str, Any],
        checkpoint_evidence: Mapping[str, Any],
    ) -> TraceContext: ...

    def materialization_axis_bindings(
        self,
        context: TraceContext,
        prune_groups: Sequence[Mapping[str, Any]],
    ) -> tuple[MaterializerParameterSource, ...]: ...

    def canonical_axis_base_widths(
        self,
        context: TraceContext,
    ) -> tuple[Mapping[str, Any], ...]: ...
```

Equivalent method names are acceptable, but the ownership split is mandatory:
adapter methods declare selectors and evidence locations; the scanner resolves
the selectors and derives widths/ratios.

`TraceContext` is frozen and contains only evidence needed by the generic
scanner:

- traced `net`
- `example_inputs`
- full model/module root
- parsed loaded config
- checkpoint-loaded evidence
- materializer source selectors
- trace module/dataflow metadata

`MaterializerParameterSource` is frozen and declares selectors only:

- `axis_id: str`
- `config_selector: str`
- `mutation_kind: str`
- `module_root_selector: str`
- `allowed_roles: tuple[str, ...]`
- `provenance: dict[str, object]`

The adapter must not branch on model name to return paper axis counts, paper
width sets, `343/686`, or `1792/3584`. The three paper models still need
adapter definitions, but those definitions expose selector paths, not oracle
spaces.

Backend and quantization constraints come from an explicit `ScanScenario`
input, not from adapter/model-name logic:

- `hardware_precisions`
- `backend_precisions`
- `compression_modes`
- `graph_quant_unit_policy`
- alignment/packing constraints

The scanner combines `TraceContext`, DepGraph groups, checkpoint-loaded module
tensors, and `ScanScenario` to create formal structural axes.

## Scanner-owned structural-axis contract

Add a Stage1 module responsible for the formal axis contract:

- `framework/stage1/structural_axes.py`

The module owns these immutable data shapes:

- `AxisMemberBinding`
  - `b1_group_id: str`
  - `module_path: str`
  - `canonical_to_member_num: int`
  - `canonical_to_member_den: int`
  - `materializer_param: str`
  - `role: str`
- `StructuralAxis`
  - `axis_id: str`
  - `dense_stage: str | None`
  - `axis_kind: "free" | "fixed_derived"`
  - `base_width: int`
  - `legal_widths: tuple[int, ...]`
  - `member_b1_groups: tuple[AxisMemberBinding, ...]`
  - `round_to: int`
  - `provenance: dict[str, object]`
- `StructuralAxisBundle`
  - `axes: tuple[StructuralAxis, ...]`
  - `diagnostics: dict[str, object]`

Builder responsibilities:

1. Consume `TraceContext`, DepGraph prune groups, and `ScanScenario`.
2. Resolve materializer source selectors into config paths and module roots.
3. Fail closed when a selector is missing, ambiguous, or resolves to multiple
   config paths for one axis.
4. Group low-level prune groups by graph dependency/dataflow and resolved
   materializer parameter bindings.
5. Choose the canonical coordinate from the unique loaded config selector, then
   cross-check it against checkpoint-loaded module tensors and DepGraph group
   width relations.
6. Preserve member relationships as exact integer ratios. For example, an
   internal bottleneck tensor at `2C` is recorded as a `2/1` member transform,
   not as a separate legal canonical width.
7. Generate `legal_widths` by pruning from canonical base width under graph
   validity, materializer feasibility, and alignment/packing constraints.
8. Mark axes that are graph/materializer-derived but not independent knobs as
   `fixed_derived`. Fixed-derived axes keep provenance but do not multiply the
   formal structure count.
9. Fail closed when a free axis has no legal widths, when a member transform is
   non-integral for any legal canonical width, when a binding points to no
   DepGraph group, or when required provenance/digest data is missing.

The existing `view_b1_prune_groups`, `view_b1_search_groups`, and
`software_candidates` fields remain public compatibility and diagnostic fields.
They are not a source of formal axes.

## Graph scan and bridge ownership

`framework/stage1/graph_scan.py` owns the production flow:

1. Build `TraceContext` from the model adapter.
2. Run tracing/DepGraph extraction.
3. Resolve materializer selectors into `structural_axis_inputs`.
4. Call `derive_structural_axes(...)`.
5. Emit scanner-owned derived axes with compact source provenance and digest.

`framework/stage1_bridge.py` consumes only scanner-owned derived axes. It must
reject top-level handwritten `formal_axis`, `structural_axes`, or equivalent
oracle injection that lacks scanner provenance/digest. The bridge may serialize
the scanner-owned axes into Stage2 output as `structural_axes`, but the input
must be trace/scanner-derived.

## q-mode contract

Stage1 exposes `formal_q_modes` from four sources:

1. `HwCapability.gpu_precisions` or equivalent hardware profile.
2. Backend implementation support, such as TVM/TensorRT precision support.
3. Configured compression/search modes for the run.
4. Graph quant-unit legality emitted by the scanner.

The mapping is explicit and case-normalized:

- `FP16` → `fp16`
- `INT8` → `int8`

Unsupported entries such as INT4 are ignored unless all configured modes become
unsupported, in which case construction fails.

## Three paper-model reproduction oracles

These values are test oracles only.

Pyramid:

- free axes:
  - `[16,24,32,40,48,56,64]`
  - `[32,48,64,80,96,112,128]`
  - `[64,96,128,160,192,224,256]`
- structures: `343`
- H800/TVM q modes: `fp16`, `int8`
- candidates: `686`

CoDriving:

- free axes:
  - `[16,24,32,40,48,56,64]`
  - `[32,48,64,80,96,112,128]`
  - `[64,96,128,160,192,224,256]`
- structures: `343`
- H800/TVM q modes: `fp16`, `int8`
- candidates: `686`
- neck components are preserved as fixed-derived provenance, not independent
  free axes.

F-Cooper:

- free axes:
  - `[32,64]`
  - `[32,64,96,128]`
  - `[32,64,96,128,160,192,224,256]`
  - `[32,64,96,128]`
  - `[64,96,128,160,192,224,256]`
- structures: `1792`
- H800/TVM q modes: `fp16`, `int8`
- candidates: `3584`
- the two neck axes are independent interface axes only when graph
  independence and materializer bindings prove they are not fixed-derived from
  a backbone axis.

## Planner and registry contract

- Generic planning code enumerates `axis_schema.free_axes × formal_q_modes`.
- P6 Pyramid adapter maps generic axis IDs into legacy P6 row fields after
  formal enumeration; it does not rediscover Pyramid stages from legacy anchors.
- Registry validation rejects over-base canonical widths before writing any
  plan or registry output.
- Registry records may keep rejected diagnostics outside the formal plan, but
  rejected rows must never be counted as formal search candidates.

## Acceptance gates

- Three purified real scanner-contract fixtures are the only integration
  acceptance fixtures:
  - Pyramid fixture purified from real scanner output plus adapter contract.
  - CoDriving fixture purified from real scanner output plus adapter contract,
    with neck fixed-derived provenance.
  - F-Cooper fixture purified from real scanner output plus adapter contract,
    with two independent neck interface axes.
- Fixtures retain source provenance/digest and contain no `expected_counts`,
  `expected_widths`, or handwritten formal-axis oracle fields.
- Unit tests must include RED evidence for missing adapter binding/base,
  binding that resolves to no DepGraph group, non-unique config path, fixture
  handwritten axes, missing provenance/digest, malformed member ratios, empty
  q-mode intersections, and over-base registry rejection.
- Integration tests and CLI summaries assert exact paper counts:
  - Pyramid `343/686`
  - CoDriving `343/686`
  - F-Cooper `1792/3584`
- Full verification for this plan is CPU-only:
  - targeted Stage1/Stage5/Stage6/integration pytest
  - release tests
  - `python -m compileall framework scripts tests -q`
  - coverage `>=80%` for touched modules
- Each implementation task has one implementer, followed by strict serial
  two-stage review: SPEC reviewer first, QUALITY reviewer second.
