# CoptV2X Scan-Derived Paper Space Design

## Goal

Make Stage1 → Stage2 space construction reproduce the CoptV2X paper spaces for Pyramid, CoDriving, and F-Cooper without using the paper counts as production drivers.

## Binding requirements

- The formal search space is generated from scanned graph structure, dependency provenance, base checkpoint widths, materializer feasibility, and hardware/backend precision capability.
- Probe anchors, cliff-neighbor diagnostics, and low/mid/high samples are diagnostic-only. They must not replace or truncate the formal candidate universe.
- A candidate must never require a width above the canonical base checkpoint width for its structural axis. Over-base widths are construction errors, not normal `unavailable` search candidates.
- q modes are the intersection of hardware capability, backend/search implementation support, and quant-unit legality. For H800/TVM paper reproduction this yields `fp16` and `int8`.
- Production logic must not be driven by constants such as `3`, `7`, `343`, `686`, `1792`, or `3584`. Tests may assert those counts as paper-model reproduction oracles.
- Acceptance covers only the three paper models. Do not add arbitrary synthetic four-axis networks such as `6×8×10×12`.

## Formal model-space contract

Stage2 output keeps existing compatibility fields:

- `view_b1_prune_groups`: low-level dependency truth.
- `view_b1_search_groups`: legacy search-group compatibility and diagnostic anchors.
- `software_candidates`: existing public candidate/diagnostic view.

It additionally exposes a formal product-space view:

- `structural_axes`: ordered canonical axes. Each axis contains `axis_id`, `dense_stage`, `base_width`, `legal_widths`, `member_b1_groups`, and provenance.
- `formal_q_modes`: ordered q modes derived from capability and implementation support.
- `formal_candidate_policy`: states that formal enumeration is `structural_axes × formal_q_modes`, while probes are diagnostic-only.

P6 consumes `structural_axes` when present. The legacy `software_candidates` parser remains as a fallback for old tests and archives.

## Paper-model reproduction oracles

Pyramid and CoDriving:

- axes: `[16,24,32,40,48,56,64]`, `[32,48,64,80,96,112,128]`, `[64,96,128,160,192,224,256]`
- structures: `343`
- H800/TVM q modes: `fp16`, `int8`
- candidates: `686`
- CoDriving neck provenance may be preserved, but neck is not an independent free axis.

F-Cooper:

- axes: `[32,64]`, `[32,64,96,128]`, `[32,64,96,128,160,192,224,256]`, `[32,64,96,128]`, `[64,96,128,160,192,224,256]`
- structures: `1792`
- H800/TVM q modes: `fp16`, `int8`
- candidates: `3584`

## Implementation boundaries

- Stage1 may emit explicit `formal_axis` metadata when the scanner can derive a canonical axis directly.
- If `formal_axis` is absent, the bridge derives the axis from the legacy search group using canonical base width and prune constraints, preserving old fixture behavior.
- Hardware precision parsing must honor `ips.gpu.precisions` and `quant_constraints.bit_widths_w`; an INT4-only device must not silently produce INT8/FP16.
- P6 registry materialization rejects over-base plan rows before writing instead of publishing `unavailable` rows.
