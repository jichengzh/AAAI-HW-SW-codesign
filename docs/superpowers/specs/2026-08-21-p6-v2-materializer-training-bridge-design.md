# P6 V2 Materializer Training Bridge Design

## Status and root cause evidence

Status: approved design for the `p6-materializer-training-bridge` branch. This document specifies the smallest architecture change that keeps P6 faithful to CoptV2X while fixing the observed real-run failure.

Observed run state:

- Stage1/Stage2 dynamic Pyramid/H800/TVM candidate generation completed and produced a fresh dynamic candidate pool, including the observed 126 FP16 candidates.
- Gold176 loaded and was used as cold-start cost-model data.
- Round 0 selected four rows and emitted a `stage5_measurement_request_v2` request.
- The measurement command failed before any selected candidate was materialized, trained, fine-tuned, exported, compiled, measured, or evaluated.

Root cause:

- The public activation wrapper path and invocation shape are correct: the adapter invokes the validated private activation executable with the `private-bound` argument.
- The failing historical V2 source materializer imports a validator from its own repository as if the process cwd or `PYTHONPATH` were the historical repo root.
- The production adapter intentionally runs from the private round cwd with direct argv, `shell=False`, and the exact five-key private environment. It does not inherit ambient environment and does not set `PYTHONPATH`.
- Therefore the import failure is deterministic and is a private wrapper/import packaging defect, not a Stage1/Stage2 candidate defect or a Gold176 cost-model defect.

Latent correctness failure:

- Recipe-v2 projection currently provides shared materialization output paths, but the selected V2 source contracts can lack an explicit real-training mandate and Pyramid static training inputs.
- If `training_required` is missing or false, the historical V2 defaults can skip training, assume nonexistent checkpoints, or treat output paths as already satisfied.
- That would violate the P6 claim even after the import failure is fixed.

## Goals

- Keep CoptV2X lifecycle semantics: dynamic Stage1/Stage2 candidates, Gold176 cold-start cost-model retraining, four rounds, four selected candidates per round, and 16 total selected measurement slots.
- Ensure every selected candidate is really trained or fine-tuned before TVM compilation, AP evaluation, latency measurement, and energy measurement.
- Keep Gold176 as cost-model cold-start evidence only. Gold176 is never remeasured by this run.
- Keep private paths, commands, raw logs, checkpoints, datasets, GPU UUIDs, and host details out of the public contract and public artifacts.
- Make historical imports deterministic without weakening the public adapter environment boundary.
- Fail closed before process launch when a private wrapper, binding, recipe-v2 contract, output layout, or static training contract is incomplete.

## Non-goals

- Do not change the public P6 target: Pyramid, H800, TVM, Gold176, four rounds, batch size four, sample budget 16, metrics `latency_ms`, `energy_j`, `ap30`, `ap50`, `ap70`.
- Do not add Orin, TensorRT, CPU, RTX 4090, mixed backends, static-registry fallback, or a new acquisition policy.
- Do not publish private absolute paths, checkpoints, model code, dataset locations, raw historical logs, or per-host execution details.
- Do not make Gold176 a measurement batch.
- Do not rely on ambient cwd, shell startup files, inherited environment, or manual relaunch of a partial round.

## CoptV2X lifecycle invariants

The run is valid only if this sequence holds:

1. Stage1 scans the real Pyramid structure for H800.
2. Stage2 produces the dynamic Pyramid/H800/TVM candidate space.
3. The P6 adapter converts the Stage2 space into the dynamic candidate plan and then into a registry-v2 source space without reverting to the static 343/686 P6.1 registry.
4. Round 0 fits the cost model from frozen Gold176 rows, Gold176 graph features, and the matching capability profiles. Gold176 is not remeasured.
5. Each round selects exactly four unmeasured rows from the current dynamic source space.
6. For every selected row, the private source materializer must execute real Pyramid candidate materialization and training/fine-tuning before downstream TVM/AP/latency/energy stages.
7. A round is released only after the request hash, row hashes, source evidence hashes, receipt, finalization barrier, terminal status, and five metric fields validate.
8. Rounds 1-3 refit online from Gold176 plus accepted prior feedback. They must not use failed or fabricated metrics as successful measurements.
9. Any bridge, GPU, source, stage, feedback, or hash failure stops the run. No synthetic replacement row is allowed.

## Architecture and data flow

```text
public contract v2
  -> local private config + private binding
  -> Stage1 scan step
  -> Stage2 search-space loader
  -> dynamic Pyramid candidate plan
  -> registry-v2 materialization from private source_contract_template
  -> Gold176 cold-start cost-model fit
  -> select 4 rows
  -> project recipe-v2 request
  -> write private round request/state
  -> private activation wrapper
  -> private source materializer wrapper per selected group
  -> quantization wrapper
  -> TVM/performance wrapper
  -> AP wrapper
  -> finalizer wrapper
  -> validated P6 feedback
  -> online cost-model refit for next round
```

The public search loop remains responsible for selection, cost-model fitting, request identity, feedback validation, and run-state accounting. The private historical chain remains responsible for real selected-candidate training/fine-tuning, checkpoint production, ONNX/export artifacts, TVM compilation, AP evaluation, latency measurement, energy measurement, and private diagnostics.

The historical controller is not invoked as the round owner. The adapter keeps direct stage execution because it already validates stage order, task-state initialization, feedback receipt, finalization barrier, and GPU policy. If the historical controller contains useful implementation logic, it may be called from inside private wrappers, but it must not replace the public adapter's validated stage boundary.

## Private self-contained wrapper contract

Every private executable referenced by the runner template must be self-contained. In particular, the source materializer executable named by the validated `source_materialization` stage must be a wrapper with this contract:

- It is an executable regular file beneath the validated private history Git root.
- It may be named with the documented marker, for example `stage5_materialize_round_sources_v1.sh`, but its internal implementation is private.
- It accepts the direct argv emitted by the public adapter:
  - `--request <absolute-private-request-json>`
  - `--model pyramid`
  - `--group-id <canonical-pyramid-group-id>`
  - `--gpu <validated-binding-gpu-index>`
- It must succeed from the private round cwd. It must not require the process cwd to be the historical repository root.
- It must succeed with only the exact environment rendered by the adapter:
  - `CUDA_VISIBLE_DEVICES`
  - `P6_HISTORY_RUN_MODE`
  - `P6_HISTORY_PRIVATE_ROOT`
  - `P6_HISTORY_TASK_STATE`
  - `P6_HISTORY_ROUND_OUTPUT_ROOT`
- It must resolve its own private repository/import root internally, for example by using its script location or `P6_HISTORY_PRIVATE_ROOT`, then changing cwd or setting private import paths before execing the real historical materializer.
- It must not require the public adapter to set `PYTHONPATH`, add environment keys, call a shell wrapper, or run from the history root.
- It must write only to declared private output paths under the local output root and update task-state according to the validated interface.
- It must return nonzero on any import, missing-training-input, skipped-training, failed-training, missing-marker, checkpoint, export, TVM, AP, latency, or energy failure.

The activation wrapper remains a validation/admission stage. It cannot be the only place where imports are fixed, because subprocess environment changes from activation do not persist unless the private wrappers independently consume an activation artifact or resolve their own imports.

## Recipe-v2 and static training contract

Recipe-v2 remains the public-safe description of shared output path templates. It does not carry real private paths or hyperparameter values.

Every recipe-v2 source contract generated for a selected Pyramid group must contain these exact semantic fields after registry materialization and before request projection:

- `training_required`: exactly `true`.
- `training_source_kind`: a public-safe label whose semantics are "selected candidate must be trained/fine-tuned for this run", not "checkpoint already exists".
- `base_checkpoint_path`: private absolute path to the existing Pyramid initialization checkpoint used for fine-tuning.
- `dataset_root`: private absolute path to the existing CoptV2X training/evaluation dataset root.
- `pyramid_config_path`: private absolute path to the existing base Pyramid training configuration.
- `training_parameters`: a mapping with the following required semantic keys:
  - `training_mode`: selected-candidate fine-tuning mode.
  - `epochs`: positive integer epoch count or equivalent bounded training duration.
  - `seed`: deterministic private training seed.
  - `optimizer`: optimizer family label.
  - `learning_rate`: finite positive learning-rate value or schedule reference.
  - `batch_size`: positive integer batch size.
  - `dataset_split`: train/validation split identity.
  - `checkpoint_selection`: rule for selecting the produced checkpoint.
  - `freeze_policy`: Pyramid layer-freezing or full-fine-tune policy.
- `stage_widths`: the canonical Pyramid width identity for the group.
- `shared_source_paths`: the recipe-v2 rendered output map with exactly the existing shared keys:
  - `checkpoint_path`
  - `checkpoint_dir`
  - `config_path`
  - `training_done_marker`
  - `onnx_path`
  - `onnx_report_path`
  - `calibration_root`
  - `calibration_npz`
  - `calibration_summary`
  - `trt_calibration_dir`
  - `source_done_marker`

No real values for these fields belong in this design document or in public docs. In the private binding and local generated registry, all path values must be absolute, symlink-safe, and under the validated private history root or local private output root as appropriate.

The bridge must reject recipe-v2 source contracts that omit the static training fields, set `training_required` to false, provide empty `training_parameters`, point training inputs outside the private root, render output paths outside the local output root, or collide output paths across groups.

## Registry, projection, and hash behavior

Registry materialization:

- `materialize_history_registry()` validates the dynamic candidate plan identity against the Stage2-derived plan.
- For recipe-v2, it renders one shared source bundle per Pyramid group, not one source bundle per q-mode.
- It copies all required static training fields from the private `source_contract_template`.
- It removes untrusted self-reported output paths and legacy q-mode-specific output aliases before writing the canonical group contract.
- It renders shared output paths beneath the local private output root.
- It recomputes `source_contract_sha256` after inserting group identity, static training fields, and rendered shared paths.
- It writes the registry atomically to an ignored private destination.

Request projection:

- `project_source_materialization_request()` flattens recipe-v2 `shared_source_paths` into the row source contract consumed by the private materializer.
- It preserves the required static training fields.
- It rejects mixed legacy and recipe-v2 rows in one request.
- It rejects same-group static training drift across q-modes.
- It rejects shared-path mismatch, cross-group path collision, missing training fields, false `training_required`, or stale hashes.
- It recomputes every row hash and the `measurement_request_sha256` after projection.

Runtime behavior:

- The source materializer receives one projected request path and one canonical group id.
- The materializer may train once per group and share the trained bundle across selected q-modes, but the selected row is not eligible for downstream TVM/AP/latency/energy until its group-level training marker and source marker are produced and validated.
- Downstream stages must consume the produced checkpoint/export artifacts, not assume a preexisting checkpoint.

## Public/private boundary

Public surface:

- Public P6 contract labels, metric names, target, model, backend, round count, batch size, sample budget, asset labels, license/status labels.
- Public-safe component versions and binding status.
- Public-safe failure categories.
- Hash identities for requests, rows, source contracts, and feedback validation.

Private surface:

- History root, local output root, binding JSON, local config JSON.
- Runner template, executable paths, wrapper internals, import paths, real cwd behavior, raw stderr/stdout.
- Dataset paths, checkpoint paths, model source paths, GPU UUIDs, host details, raw AP/TVM/latency/energy logs.
- Static training values and materialized artifacts.

Boundary rules:

- The public adapter must not add private import knowledge, `PYTHONPATH`, host-specific cwd, or shell execution.
- The public binding projection must not copy static training values, GPU policy, paths, commands, or raw candidate identities.
- Private generated registries and round outputs must stay under ignored private destinations.
- Error messages crossing into public state must be stable categories only.

## Versioning and compatibility

No public top-level schema version change is required for this design:

- Keep `p6_h800_coptv2x_search_contract_v2`.
- Keep `p6_h800_coptv2x_local_v2`.
- Keep `stage5_measurement_request_v2`.
- Keep `p6_h800_coptv2x_feedback_v2`.
- Keep `p6_history_dynamic_materialization_recipe_v2`.

The required static training fields are a stricter P6 recipe-v2 source-contract validation layer, not a public contract expansion. Because `stage5_source_contract_v1` already permits additional hashed private fields, the bridge can enforce these fields for recipe-v2 P6 without breaking legacy static P6.1 source registries.

If implementation later needs a machine-readable distinction, add a private-only recipe-v2 profile revision or binding capability flag. Do not change the public P6 search schema unless a public user must supply new non-private data, which this design avoids.

## Fail-closed errors

Use existing stable categories where possible:

- `history_request_invalid`: malformed measurement request, row hash drift, request hash drift, wrong batch/budget/metrics, bad q-mode, or identity mismatch.
- `history_execution_invalid`: invalid private binding, runner template, environment shape, output layout, source contract, recipe-v2 static training contract, path collision, unsafe symlink, missing marker, or untrusted projection.
- `history_gpu_admission_failed`: selected H800 indices unavailable, UUID drift, non-H800 device, or occupancy above policy before or after execution.
- `history_execution_failed`: nonzero or raised private stage execution after validation, including deterministic import failure, training failure, export failure, TVM failure, AP failure, latency failure, or energy failure.
- `unsafe_destination`: adapter output destination escapes its round root, is not ignored, already exists, or is symlinked.
- `command_failed`: outer release runner reports the measurement adapter failed without exposing private stderr.

The implementation may preserve private diagnostics under ignored private logs, but public run state and failure JSON must contain only public-safe categories.

## TDD test matrix

Add or update tests before implementation.

| Area | Required failing test |
| --- | --- |
| Wrapper determinism | Fake source wrapper imports a sibling private module successfully while adapter cwd is the round root and env has exactly the five allowed keys. A wrapper that requires repo cwd fails closed. |
| No public env widening | Source invocation never receives `PYTHONPATH`, ambient `PATH` expansion beyond the wrapper's own logic, shell execution, or inherited environment. |
| Recipe-v2 training mandate | Registry rejects recipe-v2 source templates missing `training_required`, setting it false, or omitting any required static training field. |
| Static training path safety | Registry rejects static training input paths outside the private root, symlinked paths, missing base checkpoint/config/dataset anchors, or path values in public projection. |
| Projection preservation | Projection preserves `training_required` and static training fields, strips legacy dynamic output aliases, flattens shared paths, and recomputes row/request hashes. |
| Same-group consistency | Mixed q-mode rows for one group with different training fields or shared paths are rejected before source execution. |
| Dynamic lifecycle | Full fake run proves Stage1 scan, dynamic candidate plan, registry-v2, Gold176 initial fit count 176, online fit counts 180/184/188, four 4-row batches, and 16 unique selected rows. |
| Gold176 boundary | No test path sends Gold176 rows to measurement; Gold176 is cold-start cost-model data only. |
| Real training gate | Fake materializer that skips writing `training_done_marker` or `source_done_marker` causes downstream stages not to run. |
| Failure category | Import failure or nonzero materializer exit returns a stable redacted category and does not leak private paths, argv, traceback, or environment. |
| Relaunch safety | A partial or failed round cannot be resumed by silently reusing stale request, feedback, or Stage1 manifest files. |

## Migration and provisioning

Provisioning must produce a new private binding/config pair from already-cleared private inputs:

1. Validate the private history root and runner template.
2. Require the source materializer marker to resolve to a self-contained wrapper.
3. Validate that the wrapper executable is under the private Git root, regular, executable, non-symlinked, and referenced by the source stage.
4. Validate the private `source_contract_template` has recipe-v2 plus the required static training fields.
5. Validate private static input anchors exist and are under the private root.
6. Render the local config so the public adapter still calls only the release adapters and private binding path.
7. Write the binding/config atomically to ignored private destinations.

Existing generated bindings without `training_required: true` and the required static training fields are invalid for recipe-v2 real runs. They must be regenerated, not patched in public output.

## Zero-process preflight

Before launching activation, GPU probing, or any historical executable, the adapter/provisioning layer must be able to validate:

- Public and local contract schemas.
- Private binding shape and target.
- Runner interface stage order, placeholders, executable markers, and exact environment keys.
- Source materializer wrapper path safety and executable bit.
- Recipe-v2 source contract static training fields.
- Static training path lexical safety and private-root containment.
- Shared output path renderability, uniqueness, and local-output-root containment.
- Measurement request identity, row hashes, q-mode identity, graph-feature identity, and source-contract hashes.
- Round output root safety and absence of preexisting request/feedback/task-state/barrier files.

This preflight must not import historical Python modules, execute the source materializer, call the historical controller, invoke TVM/AP code, run training, or read private raw logs.

## Fresh-run and relaunch criteria

Fresh run criteria:

- Use a newly prepared ignored private local output root or an output root with no completed/failed P6 state.
- Stage1 manifest path is reserved and removed before the Stage1 scan step; a no-op Stage1 command must not reuse a stale manifest.
- Source registry destination does not preexist as a directory or symlink and is rewritten atomically by the registry adapter.
- Each round creates a fresh round directory and fresh request/feedback/task-state/receipt/barrier leaves.

Relaunch criteria after failure:

- Do not relaunch a partial round in place.
- Preserve failed private diagnostics under the ignored output root.
- Fix the private wrapper, binding, or source contract cause.
- Start a new local output root or explicitly clean only the scoped ignored output root after reviewing what will be removed.
- Do not replace failed feedback with synthetic success rows, do not skip candidates, and do not switch to the static registry.

## Acceptance and closure

The bridge is accepted when:

- TDD tests in the matrix are implemented and pass.
- `git diff --check` passes.
- A private preflight rejects the old V2 binding that lacks `training_required` and static Pyramid training fields.
- A private preflight accepts a regenerated binding with self-contained wrappers and complete recipe-v2 training contract.
- A fresh real run reaches Stage1/Stage2 dynamic generation, Gold176 cold-start cost-model fit, four selected round-0 rows, and source materialization without the deterministic import failure.
- For each selected candidate, private evidence shows real training/fine-tuning completed before TVM/AP/latency/energy stages run.
- Completed closure requires four rounds, 16 selected rows, validated feedback, online refits after rounds 0-2, and a completed local state with no public private-path leakage.

If any selected candidate cannot be trained/fine-tuned or cannot produce validated downstream metrics, the run is not closed as a successful measurement run. A real feasibility failure may consume budget only if it is emitted by the validated private chain after the required attempted materialization/training path and is represented by an allowed terminal failure status.

## Self-review

This specification intentionally keeps the public adapter environment strict and moves import determinism into private wrappers. It does not require a public schema version change. It treats recipe-v2 static training fields as private hashed source-contract semantics. It explicitly states that Gold176 is cold-start cost-model data only and that every selected candidate must be really trained/fine-tuned before downstream TVM/AP/latency/energy work.
