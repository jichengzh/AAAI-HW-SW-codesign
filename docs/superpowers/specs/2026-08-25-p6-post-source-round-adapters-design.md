# P6 Post-Source Round Adapters Design

## Status

Approved architecture for completing the real CoptV2X post-source measurement
chain. This document defines the implementation boundary; it does not claim
that the adapters, private deployment, or H800 experiment are complete.

The approved approach keeps the public five-stage runner contract and exact
five-key environment. Four narrow adapters translate the public round-stage
argv into the historical leaf CLIs used by the CoptV2X Pyramid/H800/TVM flow.
The historical leaves remain the authority for quantization, performance,
AP, and Stage5 finalization. Public code owns only orchestration, validation
of their native outputs, and the final P6 completion-set projection.

## Goal

Complete the existing search pipeline after source materialization so that a
fresh run can:

1. build the scanner-derived Pyramid candidate space;
2. select four candidates per round for four rounds;
3. materialize and train first-use structural groups;
4. run row-specific quantization, TVM performance measurement, AP evaluation,
   and feedback finalization;
5. feed 16 real terminal measurements back into the CoptV2X search loop; and
6. satisfy the existing private completion verifier.

The objective is semantic reproduction of the CoptV2X experiment, not a new
general execution framework.

## Scope ruling: minimum validation, no over-defense

Validation is required only where an error would change CoptV2X semantics,
execute the wrong private component, corrupt row identity, consume a stale
stage output, or publish an incomplete P6 completion set.

This work does not add:

- separate P6 quantization, performance, or AP receipt schemas;
- a generic workflow engine, retry service, cache, resume protocol, or plugin
  architecture;
- proxy latency, proxy energy, proxy AP, or synthetic replacement rows;
- new candidate filtering, q-mode inference, search-space constants, or
  alternate device/backend routing;
- adversarial filesystem race protection beyond the repository's existing
  canonical path, regular-file, no-symlink, and ignored-output conventions;
- new public environment keys or private values in tracked files.

The native Stage3/Stage5 outputs are the intermediate evidence. Only the
existing P6 task state, actual feedback, receipt, and finalization barrier form
the final completion set.

## Existing public contract remains unchanged

The public measurement layer continues to execute these stages in order:

```text
source_materialization
quantization
performance
ap
finalization
```

The stage argv surfaces remain:

```text
quantization: {task_state} {round_output_root}
performance:  {task_state} {round_output_root}
ap:           {task_state} {round_output_root}
finalization: {measurement_request} {task_state} {actual_feedback}
              {actual_receipt} {finalization_barrier} {round_output_root}
```

Every stage still receives exactly:

```text
CUDA_VISIBLE_DEVICES
P6_HISTORY_RUN_MODE
P6_HISTORY_PRIVATE_ROOT
P6_HISTORY_TASK_STATE
P6_HISTORY_ROUND_OUTPUT_ROOT
```

The measurement request, public feedback schema, search state, round count,
batch size, metrics, source-reuse receipts, and GPU admission policy are not
versioned or widened by this work.

## Private deployment contract

### Existing execution closure and private leaf binding

`p6_execution_code_closure_v1` remains unchanged. The four post-source role
entrypoints are tracked adapter implementations:

```text
quantization
performance
ap
finalization
```

For private deployment, the exact reviewed public-code archive is first placed
under the private source-history Git root. The existing closure mechanism can
therefore copy the tracked adapter implementations without accepting code
outside its declared source root.

A new source-map-only `p6_post_source_leaf_binding_v1` contains exactly seven
historical Python leaves:

```text
quant_contract
performance_plan
performance_execute
ap_plan
ap_execute
feedback_finalize
feedback_promote
```

Every leaf is declared by `closure_id` and `entrypoint_relative_path`, and
must resolve beneath one of the already declared closure roots. Normalization
copies that closure root once, then derives the normalized leaf paths and
digests. No absolute historical leaf path is tracked.

The private packager gives raw historical leaf files dependency-only names
that do not collide with the generated runner marker basenames. Renaming does
not alter their bytes or digest. Legacy source maps without the leaf binding
retain their existing behavior but cannot pass the real post-source adapter
preflight.

### Post-source adapter profile

Normalization writes one ignored profile beneath the normalized private root:

```yaml
schema_version: p6_post_source_adapter_profile_v1
target:
  model: pyramid
  hardware: h800
  backend: tvm_auto
runner_interface_schema_version: p6_history_runner_interface_v1
project_python: <canonical private executable>
adapters:
  quantization:
    implementation_relative_path: <normalized relative path>
    implementation_cwd_relative_path: <normalized relative root>
  performance: {...}
  ap: {...}
  finalization: {...}
leaves:
  quant_contract:
    implementation_relative_path: <normalized relative path>
    implementation_cwd_relative_path: <normalized relative root>
    sha256: <copied file sha256>
  performance_plan: {...}
  performance_execute: {...}
  ap_plan: {...}
  ap_execute: {...}
  feedback_finalize: {...}
  feedback_promote: {...}
```

Unknown keys are rejected. Relative paths must resolve beneath the normalized
private root. The declared copied leaf bytes must match the profile digest.
`project_python` is the same canonical interpreter extracted from the
historical source-materializer implementation's unique literal
`PY=${PY:-...}` assignment.

The profile does not contain row IDs, candidate IDs, source paths, GPU UUIDs,
metrics, requests, output paths, or round-specific state.

### Generated wrappers and existing runner interface v1

Normalization renders four small private wrappers. A wrapper:

1. verifies the exact five incoming environment keys;
2. invokes the copied tracked adapter implementation with the canonical
   project Python, the ignored profile, and the unchanged stage argv;
3. supplies deterministic `PATH` and `PYTHONPATH` derived from the profile;
4. uses direct argv and returns the adapter's exact exit code.

`p6_history_runner_interface_v1` remains unchanged. The four post-source
`argv[0]` entries are generated adapter wrappers rather than raw historical
leaf CLIs. The performance and finalization wrapper destinations retain the
required marker basenames `stage5_build_performance_plan_v2.py` and
`stage5_finalize_feedback_v2.py`; binding `component_paths` point to those
wrappers. The corresponding historical raw leaves are private profile
dependencies under non-conflicting names.

## Shared adapter rules

Each adapter:

- resolves the canonical request at
  `<round_output_root>/measurement-request.json`;
- validates its request SHA, four row hashes, task-state row identities, and
  the expected preceding task-state stage;
- preserves request order and never merges rows by structural group;
- parses the ordered GPU list from `CUDA_VISIBLE_DEVICES` without sorting;
- invokes only profile-bound leaves using direct argv and a deterministic
  private environment;
- writes diagnostics only beneath the private round root;
- advances task state atomically only after its historical native outputs are
  complete and valid; and
- returns nonzero on adapter, CLI, schema, identity, or missing-output errors.

The wrapper's incoming process boundary still has exactly the five keys. The
deterministic `PATH` and `PYTHONPATH` exist only in the adapter/leaf child
environment constructed from the private profile.

Historical leaf nonzero is a round execution failure. Candidate-level
`feasibility_failure` and `numerical_feasibility_failure` are accepted only
when represented by the historical performance/AP/finalization state and
the historical finalizer completes successfully.

There is no in-place retry or resume after a failed real attempt. A failed
root is frozen and a later experiment uses a wholly fresh root.

## Quantization adapter

Input argv:

```text
<task_state> <round_output_root>
```

Behavior:

1. Read the unchanged four-row measurement request.
2. For every FP16 row, make zero quant-contract leaf calls and create no fake
   quant contract.
3. For every INT8 row, invoke `quant_contract` once with:

   ```text
   --onnx
   --calibration-npz
   --calibration-summary
   --output-json
   ```

4. Use the historical output layout:
   `quant_contracts/<width>/tensor_quant_params.json`.
5. Assign INT8 rows to the ordered private GPU pool in request order by
   setting that leaf process's `CUDA_VISIBLE_DEVICES` to one bound index; the
   historical quant CLI has no GPU argv flag.
6. After all INT8 calls complete, require every requested INT8 contract to be
   a nonempty native `stage3_tvm_int8_quant_contract_v3` JSON artifact.
7. Atomically set task state to `quantization`, retaining four pending rows.

Source reuse never skips this stage. Same-group FP16 and INT8 rows remain two
distinct row identities.

## Performance adapter

Input argv:

```text
<task_state> <round_output_root>
```

Behavior:

1. Require task state `quantization` and the exact quant outputs required by
   the request's INT8 rows.
2. Invoke `performance_plan` once with the unchanged request, private
   `performance_execution` artifact root, `performance` output directory,
   quant-contract root, and ordered GPU CSV.
3. Require native `performance_manifest.json` and
   `performance_jobs.jsonl`, each covering the exact four request rows.
4. Invoke `performance_execute` once with the native jobs, native state path,
   ordered GPU CSV, and `max_workers` equal to the number of bound GPUs.
5. Require the native performance state to contain a terminal performance
   outcome for every request row. Confirmed historical failures remain native
   inputs to finalization; missing or ready rows are not terminal.
6. Atomically set task state to `performance`, retaining the four rows.

The adapter does not calculate latency or energy and does not change the
historical TVM trial policy.

## AP adapter

Input argv:

```text
<task_state> <round_output_root>
```

Behavior:

1. Require task state `performance` and exact four-row terminal performance
   evidence.
2. Invoke `ap_plan` once to create native `ap_plan.json` and
   `ap_plan.jsonl` covering the exact four rows.
3. Deterministically partition the four plan rows across the ordered private
   GPUs. With fewer GPUs than rows, one GPU's shard contains multiple rows
   and executes them sequentially.
4. Invoke `ap_execute --stage sanity` once per nonempty GPU shard, in parallel
   across distinct GPUs.
5. Invoke `ap_execute --stage full` for the same shards. The historical AP
   executor decides whether a row proceeds from sanity to full and writes the
   native numerical-feasibility skip when required.
6. Concatenate shard states deterministically. Every AP-ready row must have a
   native terminal full-stage outcome or numerical-feasibility skip; rows
   already blocked by terminal performance evidence remain represented by the
   AP plan and are resolved by the historical finalizer.
7. Atomically set task state to `ap`, retaining the four rows.

The adapter does not implement AP, reinterpret AP reports, or invent a new
sanity/full policy.

## Finalization adapter

Input argv:

```text
<measurement_request> <task_state> <actual_feedback> <actual_receipt>
<finalization_barrier> <round_output_root>
```

Behavior:

1. Require task state `ap` and exact native quant/performance/AP artifacts.
2. Invoke `feedback_finalize` once with the native manifest, request, AP plan,
   performance state, AP state, and private final output directory.
3. Require a terminal historical four-row `stage5_feedback_batch_v2` output
   with request-row identities and source evidence matching the request.
4. Invoke `feedback_promote` once with the request and historical feedback.
   Require four promoted rows and the native promotion audit.
5. Project only the existing P6 fields:
   - success: row identity, `measured_success_gold`, latency, energy, and AP30,
     AP50, AP70;
   - true failure: row identity, one allowed failure status, and a stable
     path-free failure reason.
6. Construct the exact existing receipt and barrier maps from the canonical
   request.
7. Publish in this order: actual feedback, actual receipt, final task state,
   then finalization barrier. Each leaf is a sibling-temp fsync/replace write;
   the barrier is the completion commit point.

No post-source adapter metadata is added to public feedback or the exact P6
completion files.

## Preflight and deployment

Before a real role runs, preflight additionally proves:

- runner interface v1 and all four generated wrappers match their expected
  bytes;
- the ignored adapter profile is present and internally consistent;
- the four copied adapter implementations and seven copied historical leaf
  scripts match the normalized execution closure;
- the canonical project Python exists and is executable;
- round adapter output roots and final completion leaves are absent; and
- historical process and GPU counters remain zero during preflight.

Preflight does not execute quantization, performance, AP, finalization,
Stage1, source training, or the controller.

## TDD acceptance

### Unit RED/GREEN cycles

- profile rejects a missing/extra leaf, path outside the normalized root,
  digest mismatch, or wrong target;
- generated wrappers preserve stage argv, exact incoming environment, cwd,
  interpreter, and return code;
- FP16 makes zero quant leaf calls; INT8 makes exactly one call per row;
- mixed same-group FP16/INT8 keeps both row identities and only the INT8 row
  receives a quant contract;
- performance planner and executor each run once and cover four rows;
- AP planner runs once, sanity/full use ordered GPU shards, and numerical
  feasibility skips are preserved;
- finalization invokes finalize then promote and emits the exact four P6
  completion leaves;
- any historical leaf nonzero prevents task-state advancement and prevents
  final completion publication.

### Integration gates

- one synthetic four-row round executes all four adapters through the public
  measurement CLI and produces valid public feedback;
- source reuse skips only group materialization; all four downstream row
  identities still execute;
- four synthetic rounds yield 16 unique terminal rows with zero Gold176
  overlap;
- recipe-v2 normalization, provisioning, preflight, and verifier all consume
  the new profile and generated v1 interface; legacy non-training fixtures retain their
  existing behavior;
- changed production modules achieve at least 80% branch coverage;
- Stage6, release, Ruff, compile, diff, privacy, file-size, and function-size
  gates pass with no skip or xfail added.

## Real experiment gate

After implementation and sequential SPEC, QUALITY, MLE, and security review:

1. rebuild normalization, deployment, and output roots from the exact reviewed
   commit;
2. run official derive, normalize, real-policy provision, and zero-process
   preflight;
3. verify the ControlMaster, module origins, canonical GPU double snapshots,
   runtime, disk, process, and fresh-output gates;
4. launch the controller exactly once under a persistent parent;
5. monitor Stage1, formal Stage2, registry, each round's source and downstream
   stages, four feedback rows, and online update without interrupting healthy
   work; and
6. accept official paper-run completion only when the exact verifier reports
   four completed rounds, 16 selected rows, 16 `measured_success_gold` real
   measurements, and zero Gold176 remeasurement. Allowed true-failure rows are
   valid round feedback but do not satisfy this success-only completion gate.

A failed real root is preserved for read-only diagnosis and is never repaired
or relaunched in place.
