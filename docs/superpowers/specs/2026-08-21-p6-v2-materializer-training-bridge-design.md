# P6 V2 Materializer Training Bridge Design

## Status and approved architecture ruling

Status: approved architecture correction for the `p6-materializer-training-bridge` branch. This document is the design authority; it does not claim that the
runtime, preflight, verifier, or private H800 run is complete.

The observed failure remains a deterministic private source-wrapper/import
packaging defect. The public adapter correctly invokes direct argv from the
private round cwd with `shell=False` and the exact five-key environment. The
private wrapper must make its own imports deterministic. Recipe-v2 must also
require real selected-candidate training/fine-tuning and complete static
Pyramid training inputs.

The approved correction is:

- Materialize and train a canonical Pyramid group on its first use in one
  fresh four-round run.
- Reuse that q-independent trained source bundle for a later selected
  `q_mode` of the same group only after cryptographic and structural
  validation of an adapter-owned receipt bound to the current fresh run.
- Keep `q_mode` as a row-level search dimension. The 16 selected row
  identities remain `(group_id, q_mode)` genomes even when fewer than 16
  group-level bundles are trained.
- Run every q-specific downstream stage for every selected row. Reuse skips
  only the group-level source-materialization/training invocation.

This design rejects global group uniqueness, per-q-mode or per-row retraining,
and reuse based only on marker presence.

## Goals

- Preserve the CoptV2X lifecycle: fresh Stage1/Stage2 dynamic candidates,
  Gold176 cold-start cost-model evidence, four rounds, four rows per round,
  and 16 unique selected row ids.
- Require real Pyramid materialization and training/fine-tuning before a
  group's first downstream use.
- Permit a q-independent source bundle to support FP16 and INT8 rows selected
  in the same or later rounds of the same fresh run.
- Make reuse provenance immutable, deterministic, fail-closed, and independent
  of filesystem mtimes alone.
- Keep the exact five-key private process environment and the existing direct
  argv, cwd, binding, request, feedback, and round-layout boundaries.
- Keep private paths, nonce values, artifact names, digests, logs, GPU
  identities, and host details out of public output and errors.
- Reject stale, partial, cross-run, mismatched, symlinked, or wrapper-authored
  evidence before any downstream q-specific stage runs.

## Non-goals

- No change to Pyramid/H800/TVM, Gold176, metrics, batch size, sample budget,
  round count, or acquisition policy.
- No static-registry fallback, Orin, TensorRT execution target, CPU, RTX 4090,
  mixed backend, or Gold176 remeasurement.
- No global rule that a canonical Pyramid group may be selected only once.
- No retraining merely because a different `q_mode` is selected.
- No reuse across fresh-run roots, task identities, revisions, candidate
  plans, registries, or nonces.
- No public schema or environment-key expansion.
- No in-place resume after a failed or partial run.

## CoptV2X and source-bundle invariants

1. Stage1 scans the real Pyramid structure for H800, and Stage2 produces the
   dynamic Pyramid/H800/TVM candidate plan.
2. The plan and registry preserve a separate row for every available
   `(group_id, q_mode)`. Neither selection nor completion collapses rows by
   group.
3. Recipe-v2 renders exactly one shared source contract and one set of 11
   shared output paths per canonical Pyramid group. The contract is
   q-independent and is byte-for-byte canonical across that group's q-modes.
4. Round 0 fits only frozen Gold176 evidence. Rounds 1-3 refit from Gold176 plus
   accepted earlier feedback. Gold176 rows never enter a measurement request.
5. A group's first selected use must execute the source wrapper and produce a
   validated immutable source bundle plus an adapter-owned current-run
   receipt, including the complete 11-path q-independent superset even when
   the producer is FP16, so later INT8 never requires source retraining.
6. A later selected row for the same group may skip only that source wrapper,
   and only while its current-run receipt and every bound source artifact still
   validate.
7. Quantization, performance/TVM, AP, and finalization process all four selected
   row ids in every round, including both q-modes of a reused group.
8. Once a receipt is published, the 11 shared source paths are immutable.
   Downstream q-specific stages may read them but must write q-specific results
   only under their validated round/row outputs.
9. A round is accepted only after existing request, row, source-contract,
   source-evidence, task-state, feedback, receipt, barrier, terminal-status,
   and metric validation succeeds.
10. Any invalid source-reuse state, source failure, GPU failure, feedback
    failure, or hash drift stops the run. No synthetic replacement row or
    partial in-place retry is permitted.

## Existing API boundaries

The correction fits the current call graph:

```text
run_p6_coptv2x_search(contract, local, code_revision, command_runner)
  -> Stage1 manifest
  -> build_pyramid_candidate_plan()
  -> materialize_history_registry()
  -> validate_search_task()
  -> create_fresh_run_context() exactly once
  -> four calls through the configured measurement step

run_history_measurement_batch(request, binding, round_output_root, runner, gpu_probe)
  -> project_source_materialization_request()
  -> load and validate the fixed run context
  -> classify each distinct selected group
  -> invoke source wrapper only for UNSEEN groups
  -> validate artifacts and publish adapter-owned receipts
  -> revalidate all selected groups as READY_CURRENT_RUN
  -> quantization -> performance -> AP -> finalization for every row
```

`run_p6_coptv2x_search()` already owns the code-revision label, the validated
`SearchTask`, the Stage2 plan, the materialized registry, and the local output
root. It is therefore the only run-context creator. The context is created
after plan/registry identity validation and before the first measurement
request is launched.

`run_history_measurement_batch()` already receives the supplied public round
root. After the existing runtime round-root validation, its safe parent is the
same local output root used by registry rendering. Measurement resolves the
run context and receipts from that root. Nothing is passed to a private wrapper
through a new argv flag or environment key.

`project_source_materialization_request()` remains the canonical request and
hash gate. `q_mode` stays on the row; it is not inserted into the shared
source contract, run-context path, or receipt path.

## Stable lifecycle states and categories

Each distinct selected group is classified into exactly one immutable result:

```python
P6SourceReuseState = Literal[
    "UNSEEN",
    "READY_CURRENT_RUN",
    "INVALID_PARTIAL",
    "INVALID_STALE",
    "INVALID_MISMATCH",
]
```

- `UNSEEN`: the expected receipt leaf and all 11 declared shared source
  outputs are absent, with no symlink at any leaf.
- `READY_CURRENT_RUN`: the receipt and all outputs are present as the exact
  expected types; the receipt binds the current context, group, contract,
  producer request and row; and all artifact and marker digests recompute.
- `INVALID_PARTIAL`: only a subset exists, including bare markers, artifacts
  without a receipt, a receipt with a missing artifact, or one marker without
  the other.
- `INVALID_STALE`: a complete-looking receipt/output set binds a different
  run context, nonce, task, revision, local-root fingerprint, plan, or registry.
- `INVALID_MISMATCH`: current-run binding exists but receipt shape/hash,
  group identity, producer binding, source-contract/source-evidence identity,
  output type, artifact digest, marker digest, or immutable-bundle validation
  fails.

Only `UNSEEN` and `READY_CURRENT_RUN` are executable states. Invalid states
are never persisted as success. Their stable private diagnostic categories are
`p6_source_reuse_partial`, `p6_source_reuse_stale`, and
`p6_source_reuse_mismatch`. All three map to the public
`history_execution_invalid` category without diagnostic values.

## Canonical hashing

All JSON identities introduced by this design use one shared function:

```python
def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Hashes are 64 lowercase hexadecimal characters. Unknown keys, duplicate JSON
keys, booleans where integers are required, non-finite numbers, non-canonical
paths, and non-exact key sets are rejected.

- `run_context_sha256` hashes the exact run-context object excluding only
  `run_context_sha256`.
- `receipt_sha256` hashes the exact receipt object excluding only
  `receipt_sha256`.
- `candidate_plan_sha256` hashes the complete validated
  `p6_pyramid_candidate_plan_v2` object.
- `source_registry_sha256` hashes the complete validated
  `stage5_candidate_source_registry_v2` object.
- `local_output_root_sha256` is
  `SHA256(b"p6-local-output-root-v1\0" + str(resolved_root).encode("utf-8"))`.
- `group_key_sha256` is
  `SHA256(b"p6-group-receipt-v1\0" + group_id.encode("utf-8"))`.
- `run_nonce` is 32 bytes from `secrets.token_bytes(32)`, serialized as 64
  lowercase hexadecimal characters and generated exactly once.

Regular-file digests stream the raw bytes through SHA-256. A directory-tree
digest traverses only the exact declared directory root, never searches for a
receipt, and hashes a canonical JSON list sorted by POSIX relative path:

- directory entry: `{"kind":"directory","path":"relative/path"}`
- file entry:
  `{"kind":"regular_file","path":"relative/path","sha256":"...","size":N}`

The declared root itself is not an entry; an empty directory hashes the empty
list. Symlinks, sockets, devices, FIFOs, hard-linked regular files, path escape,
and traversal outside the validated local root are rejected.

## Deterministic private layout

These root-relative constants are fixed:

```python
RUN_METADATA_RELATIVE_ROOT = Path(".p6-materializer-training-bridge-v1")
RUN_CONTEXT_RELATIVE_PATH = (
    RUN_METADATA_RELATIVE_ROOT / "run-context.json"
)
GROUP_RECEIPT_RELATIVE_ROOT = (
    RUN_METADATA_RELATIVE_ROOT / "group-receipts"
)
SOURCE_OUTPUT_RELATIVE_ROOT = Path("materialized")
```

For canonical `group_id`:

```python
receipt_path = (
    local_output_root
    / GROUP_RECEIPT_RELATIVE_ROOT
    / f"{group_key_sha256(group_id)}.json"
)
```

The run context is always
`local_output_root / RUN_CONTEXT_RELATIVE_PATH`. Source output paths remain
the 11 exact absolute leaves rendered by the existing recipe profile under
`materialized/{artifact_id}/...`.

No code may use `glob`, `rglob`, directory scanning, basename search,
marker-parent inference, or filename guessing to locate a context or receipt.
The bounded directory traversal described above is allowed only to compute the
digest of an already-declared directory artifact.

## Filesystem and persistence rules

- The local output root must be absolute, existing, Git-ignored when inside
  the repository, a real directory, and resolved component by component with
  `lstat`. No root component may be a symlink.
- Metadata and receipt paths must remain lexically and resolved beneath that
  root. Existing parents must be real directories; absent parents are planned
  lexically and created mode `0700`.
- Declared source leaves must remain beneath the same root, have canonical
  lexical spelling, and contain no symlink component. Expected files must be
  single-link regular files; expected directories must be real directories.
- Context and receipt publication is create-only. The adapter writes canonical
  JSON to a same-directory temporary regular file opened with
  `O_CREAT|O_EXCL|O_NOFOLLOW` and mode `0600`, flushes and `fsync`s it,
  then publishes to the absent final leaf with an atomic no-replace operation.
  `renameat2(RENAME_NOREPLACE)` or a same-filesystem hard-link publication
  with equivalent no-replace semantics is acceptable. The parent directory is
  `fsync`ed. No ordinary replace/rename is acceptable.
- If the final leaf appears before publication, publication fails. Contexts
  and receipts are never updated, repaired, truncated, or overwritten.
- A private wrapper does not receive the receipt path. Immediately before
  source invocation the adapter requires the receipt absent; immediately after
  the wrapper returns it requires the receipt still absent. Only then may the
  adapter publish it. A wrapper-created receipt is
  `INVALID_MISMATCH`, even if its bytes would otherwise validate.

The path APIs keep planned absence separate from runtime existence:

```python
@dataclass(frozen=True)
class P6SourceReusePaths:
    local_output_root: Path
    metadata_root: Path
    run_context: Path
    receipt_root: Path

def plan_source_reuse_paths(
    local_output_root: Path,
) -> P6SourceReusePaths: ...

def resolve_existing_source_reuse_paths(
    local_output_root: Path,
) -> P6SourceReusePaths: ...

def receipt_path_for_group(
    paths: P6SourceReusePaths,
    group_id: str,
) -> Path: ...
```

`plan_source_reuse_paths()` permits the metadata root and final leaves to be
absent while validating all existing ancestors. It is used by zero-process
preflight and context creation. `resolve_existing_source_reuse_paths()`
requires a valid published context and safe receipt directory and is used by
measurement and completion. The existing
`plan_validated_history_round_paths()` versus
`resolve_validated_history_round_paths()` distinction remains unchanged.

## Immutable run-context schema

The in-memory and on-disk contract is:

```python
@dataclass(frozen=True)
class P6FreshRunContext:
    schema_version: Literal["p6_materializer_fresh_run_context_v1"]
    run_nonce: str
    task_id: str
    task_sha256: str
    code_revision: str
    local_output_root_sha256: str
    candidate_plan_schema_version: Literal["p6_pyramid_candidate_plan_v2"]
    candidate_plan_sha256: str
    source_registry_schema_version: Literal[
        "stage5_candidate_source_registry_v2"
    ]
    source_registry_sha256: str
    created_before_round_index: Literal[0]
    run_context_sha256: str
```

The JSON object has exactly those keys. `task_id` and `task_sha256` come
from `validate_search_task(task)`; `code_revision` is the already-validated
public-safe label. The plan and registry hashes are computed from the exact
validated objects also persisted as `pyramid_candidate_plan.json` and
`source_registry.json`.

```python
def create_fresh_run_context(
    *,
    local_output_root: Path,
    task_contract: Mapping[str, Any],
    code_revision: str,
    candidate_plan: Mapping[str, Any],
    source_registry: Mapping[str, Any],
) -> P6FreshRunContext: ...

def load_fresh_run_context(
    *,
    local_output_root: Path,
    expected_task_id: str,
    expected_task_sha256: str,
) -> P6FreshRunContext: ...
```

`create_fresh_run_context()` is called once in
`run_p6_coptv2x_search()`, after dynamic plan/registry validation and before
the round loop can launch measurement. Before generating the nonce it requires
the context, receipt namespace, all deterministic group receipt leaves, and
all 11 source leaves for every registry group to be absent. It then publishes
the context exclusively and creates empty `group-receipts` as a real
mode-`0700` directory. A crash between creates invalidates that root.

`load_fresh_run_context()` recomputes the context hash, root fingerprint,
plan hash from the exact plan leaf, and registry hash from the exact registry
leaf. It validates the current measurement request's task identity. It never
reads environment state or private logs.

## Immutable per-group receipt schema

The receipt records q-independent source provenance, not authorization to skip
q-specific work:

```python
ArtifactKind = Literal["regular_file", "directory_tree"]

@dataclass(frozen=True)
class P6ArtifactDigest:
    kind: ArtifactKind
    sha256: str

@dataclass(frozen=True)
class P6GroupSourceReceipt:
    schema_version: Literal["p6_group_source_reuse_receipt_v1"]
    status: Literal["READY_CURRENT_RUN"]
    run_context_sha256: str
    run_nonce: str
    task_id: str
    task_sha256: str
    group_id: str
    group_key_sha256: str
    source_contract_sha256: str
    source_evidence_sha256: str
    producer_round_index: int
    producer_measurement_request_sha256: str
    producer_row_id: str
    producer_row_sha256: str
    producer_q_mode: Literal["fp16", "int8"]
    artifact_digests: tuple[
        tuple[str, P6ArtifactDigest], ...
    ]
    marker_digests: tuple[tuple[str, str], ...]
    receipt_sha256: str
```

The JSON object has exactly the receipt fields above, with
`artifact_digests` serialized as an exact-key object and
`marker_digests` serialized as an exact-key object.

`artifact_digests` has exactly these nine keys:

- `checkpoint_path`: `regular_file`
- `checkpoint_dir`: `directory_tree`
- `config_path`: `regular_file`
- `onnx_path`: `regular_file`
- `onnx_report_path`: `regular_file`
- `calibration_root`: `directory_tree`
- `calibration_npz`: `regular_file`
- `calibration_summary`: `regular_file`
- `trt_calibration_dir`: `directory_tree`

`marker_digests` has exactly `training_done_marker` and
`source_done_marker`, each mapped directly to its raw-byte SHA-256.
Duplicated content between a directory-tree digest and a declared file digest
is intentional: the receipt proves both the declared leaf and the containing
tree.

The producer row is the lexicographically smallest `row_id` for the group in
the first-use request. Its row hash is taken from the request's
`row_sha256`; its `q_mode` is evidence only and does not restrict later
reuse. The producer request must be the canonical projected request persisted
at the exact binding-resolved private round request path. Later validation
requires:

- producer round is in `0..3` and no later than the consumer round;
- producer request hash recomputes and equals the receipt;
- producer row exists in that request with the bound row hash, group id,
  q-mode, source-contract hash, and source-evidence hash;
- the current consumer row has the same group-level source-contract and
  source-evidence hashes, while its row id and q-mode may differ.

## Classification and receipt APIs

```python
@dataclass(frozen=True)
class P6GroupReuseDecision:
    group_id: str
    state: P6SourceReuseState
    receipt_path: Path

def classify_selected_group_sources(
    request: Mapping[str, Any],
    *,
    run_context: P6FreshRunContext,
    local_output_root: Path,
    interface: Mapping[str, Any],
    private_root: Path,
) -> tuple[P6GroupReuseDecision, ...]: ...

def validate_and_publish_group_receipt(
    request: Mapping[str, Any],
    *,
    group_id: str,
    run_context: P6FreshRunContext,
    local_output_root: Path,
    interface: Mapping[str, Any],
    private_root: Path,
) -> P6GroupSourceReceipt: ...

def require_selected_groups_ready_current_run(
    request: Mapping[str, Any],
    *,
    run_context: P6FreshRunContext,
    local_output_root: Path,
    interface: Mapping[str, Any],
    private_root: Path,
) -> tuple[P6GroupSourceReceipt, ...]: ...
```

All functions consume a detached canonical projected request. They validate
same-group contract identity already enforced by
`validate_projected_training_marker_pairs()`, derive paths only through the
fixed layout, and return groups sorted by canonical `group_id`.

## First-use and reuse execution flow

`run_history_measurement_batch()` performs this flow:

1. Validate the binding and projected request. Resolve the public/private round
   paths with existing exact-template APIs.
2. Load and validate the one run context from the local root before GPU probe,
   activation, source execution, or any downstream wrapper.
3. Classify every distinct selected group. Any invalid state stops with a
   redacted `history_execution_invalid` before process launch.
4. Run GPU admission, initialize the exact private round request/task-state
   leaves, and run the existing activation boundary with the exact five-key
   environment.
5. Build source invocations only for `UNSEEN` group ids. If every selected
   group is `READY_CURRENT_RUN`, the invocation tuple is empty and no source
   wrapper runs. `build_source_invocations()` need not accept an empty group
   list; the caller simply does not call it.
6. Immediately before each first-use batch, require each unseen group's
   receipt and all 11 outputs absent. Run one direct-argv source invocation per
   unseen group.
7. After all source invocations return zero, require the receipts still absent.
   For each unseen group, validate all 11 outputs, require
   `producer_request_mtime <= training_marker_mtime <= source_marker_mtime`
   as supplemental first-use ordering evidence, compute digests, and publish
   the adapter-owned receipt exclusively.
8. Reclassify every selected group and require `READY_CURRENT_RUN`. This
   revalidates artifact and marker content immediately before downstream work.
9. Execute quantization, performance/TVM, AP, and finalization in their existing
   order. Each stage's task-state/result evidence must cover all four selected
   row ids, including every q-mode row that shared a receipt.

For two q-modes of one group selected in the same request, classification sees
one `UNSEEN` group, the source wrapper runs once, the canonical producer row
is recorded, and both rows continue through all downstream stages.

For FP16 selected in round 0 and INT8 of the same group selected later, round 0
creates the receipt. The later round validates that earlier producer request
through the exact binding template, skips only source materialization, and
runs the INT8 downstream path. The reverse q-mode order is equally valid.

Marker mtimes never authorize later reuse: a later request is expected to have
a newer mtime than the original markers. The current-run receipt and recomputed
content digests replace the contradictory “markers must be absent before every
invocation” rule.

## Exact process environment and wrapper boundary

The private wrapper contract remains direct argv:

- `--request <exact private measurement request>`
- `--model pyramid`
- `--group-id <canonical group id>`
- `--gpu <validated binding index>`

The environment remains exactly:

- `CUDA_VISIBLE_DEVICES`
- `P6_HISTORY_RUN_MODE`
- `P6_HISTORY_PRIVATE_ROOT`
- `P6_HISTORY_TASK_STATE`
- `P6_HISTORY_ROUND_OUTPUT_ROOT`

There is no `PYTHONPATH`, run-context, nonce, receipt, plan, registry, or local
root environment key. The self-contained wrapper resolves its private imports
from its own script/private-root contract. It writes only declared source
outputs and task-state. It cannot publish, update, or bless a receipt.

## Preflight, fresh-run, and relaunch policy

Zero-process operator preflight occurs before Stage1, GPU probing, activation,
or historical execution. It uses `plan_source_reuse_paths()` and requires:

- the exact run-context leaf absent;
- the entire exact `.p6-materializer-training-bridge-v1` namespace absent;
- the fixed `materialized` source-output root absent;
- Stage1 manifest, candidate-plan, source-registry, `state.json`, and public
  round directories absent;
- every binding-resolved private round root/request/task-state/result/receipt/
  barrier destination absent through planned-path validation.

It does not scan or delete the local root. Binding/config inputs intentionally
present in the root are validated separately.

After Stage1 and registry construction, but before context creation, the
controller performs a second no-process freshness gate over the exact registry:
every derived receipt leaf and all 11 declared source leaves for every group
must be absent. This closes the gap between preflight templates and the fresh
dynamic plan.

Runtime permits existing source output only through a
`READY_CURRENT_RUN` receipt. A preexisting marker, artifact, or receipt that
cannot validate against the newly created context is not a cache hit; it is an
invalid state.

After any failure, preserve ignored private diagnostics and start with a new
scoped ignored local output root after fixing the cause. Do not delete or
repair individual markers, outputs, receipts, contexts, requests, feedback,
task-state, or barriers and then resume in place.

## Completion verification

The completion verifier resolves, never searches:

- the one run context through `RUN_CONTEXT_RELATIVE_PATH`;
- each of four public/private requests through the existing exact round
  templates;
- a group's receipt through `receipt_path_for_group()`;
- task-state, result, actual feedback receipt, and finalization barrier through
  the validated binding templates.

It validates four completed rounds and exactly 16 unique selected row ids.
Each row must map to one valid `READY_CURRENT_RUN` group receipt. Several
rows may map to the same receipt; receipt count is therefore the number of
distinct first-used groups, not a required 16. A receipt producer may be in
the same round or an earlier round, never a later round.

For every row the verifier rechecks the producer request/row binding, current
source-contract/source-evidence identity, current artifact/marker digests,
per-row terminal status, and all five metric fields. It separately proves the
16 selected row ids and measurement identities are disjoint from frozen
Gold176. Gold176 remains cost-model evidence only.

The public completion report remains limited to stable counts and status. It
must not emit number of distinct receipts, group ids, producer rows, q-modes,
nonce, hashes, paths, mtimes, metrics, or artifact details.

## Public/private boundary and failures

Public-safe surfaces remain the existing contract versions, target labels,
metric names, round/sample counts, binding status, and stable failure
categories. No new public top-level schema is required:

- `p6_h800_coptv2x_search_contract_v2`
- `p6_h800_coptv2x_local_v2`
- `stage5_measurement_request_v2`
- `p6_h800_coptv2x_feedback_v2`
- `p6_history_dynamic_materialization_recipe_v2`

`p6_materializer_fresh_run_context_v1` and
`p6_group_source_reuse_receipt_v1` are ignored private schemas.

Existing public categories remain authoritative:

- `history_request_invalid`: request, row, q-mode, or request identity drift.
- `history_execution_invalid`: invalid binding, context, receipt, artifact,
  marker, path, source contract, projection, or reuse state.
- `history_gpu_admission_failed`: H800 identity/occupancy admission failure.
- `history_execution_failed`: nonzero private execution after validation.
- `unsafe_destination`: unsafe or preexisting planned destination.
- `command_failed`: outer measurement command failure.

Public errors and state contain only the category. Private diagnostics may
record a stable private subcategory under ignored output, but never an absolute
path, argv, environment, traceback, nonce, digest, group id, row id, GPU UUID,
hostname, or raw stderr/stdout. Tests must inject recognizable private tokens
and prove none crosses the adapter/CLI boundary.

## Task ownership corrections

The approved architecture changes the responsibilities of Tasks 4–7 while
leaving completed Task 3 projection and hash semantics intact.

This supersedes only contradictory Task 4–7 clauses: marker absence applies to
`UNSEEN` first use, source-call counting spans the whole run, and completion
counts 16 q-level rows mapped to receipts. Other Task 3–7 constraints remain.

### Task 4 — runtime source reuse contract

Task 4 owns the focused source-reuse module, immutable schemas, canonical
hash/digest helpers, deterministic paths, state classification, receipt
publication/validation, source-invocation filtering, exact environment
assertion, and downstream gate. It may route through
`p6_history_measurement_v1.py` and
`p6_history_source_materialization_v1.py`; focused reuse tests belong in a
new small test module rather than enlarging an already oversized mixed module.

Task 4 tests must cover:

- round-0 FP16 first use followed by later INT8 reuse of the same group;
- later FP16 after an INT8 producer;
- FP16 and INT8 of one group in the same round;
- one source invocation per first-use group and zero source invocation for a
  ready group;
- all downstream row ids executing in both first-use and reuse cases;
- bare markers, receipt-only, one-marker, one-artifact-missing, wrapper-created
  receipt, malformed receipt, artifact tamper, marker tamper, producer-request
  tamper, cross-run receipt copy, root change, task/revision/plan/registry
  drift, symlinks, hard links, and path escape;
- exact five-key env with no receipt/context key and redacted failures.

### Task 5 — fresh-run integration, preflight, and verifier

Task 5 owns controller integration of
`create_fresh_run_context()`, planned-absent versus runtime-existing path
APIs, zero-process preflight freshness checks, exact-layout producer-request
resolution, completion mapping from 16 rows to receipts, and relaunch safety.
It must not weaken the existing runtime requirement that a supplied round root
already exists.

Preflight tests prove context/metadata/materialized namespaces and every
binding-resolved round leaf are rejected when preexisting, without historical
process launch or GPU probe. Completion tests prove a producer may be from an
earlier round, multiple rows may share one receipt, all 16 row ids remain
unique, and Gold176 overlap is zero.

### Task 6 — zero-GPU lifecycle gate

The fake source wrapper writes complete declared source artifacts and markers;
the public adapter, not the fake wrapper, publishes receipts. The lifecycle
fixture must deliberately select:

- one group first as FP16 and later as INT8; and
- one same-round mixed-q group.

Expected source-call count is the number of distinct groups at their first use
across the whole run, not the sum of distinct groups in each round. Expected
downstream row count remains 16. The fake must never launch GPU, training, TVM,
AP, latency, energy, or historical processes, and Gold176 must never appear in
a request.

### Task 7 — private H800 run and conditional closure

Private preflight must pass on an absent context/receipt/source namespace.
The fresh controller creates one context before round 0. Real evidence must
show first-use training, valid current-run receipts, later same-group reuse
when selected, downstream q-specific processing for all 16 rows, four accepted
rounds, and no Gold176 remeasurement. The completion verifier, not marker
counts, decides closure.

Final public docs remain conditional on verifier success and may state only
that every selected row mapped to validated current-run source evidence.
They must not disclose which rows shared a receipt or any private receipt
field. Failure means no final-doc update and no in-place relaunch.

## Acceptance matrix

| Scenario | Classification | Source invocation | Downstream q-specific stages | Result |
| --- | --- | ---: | ---: | --- |
| New group; no outputs or receipt | `UNSEEN` | Once for group | Every selected row | Publish receipt, then proceed |
| FP16 producer; later INT8 same group/current run | `READY_CURRENT_RUN` | Zero later | INT8 row runs | Accept |
| FP16 and INT8 same group/same round | One `UNSEEN` group | Once | Both rows run | Accept |
| Four ready groups in later request | All `READY_CURRENT_RUN` | Zero | All four rows run | Accept |
| Markers/artifacts without receipt | `INVALID_PARTIAL` | Zero | Zero | Redacted stop |
| Receipt with missing output | `INVALID_PARTIAL` | Zero | Zero | Redacted stop |
| Receipt copied from another fresh run/root | `INVALID_STALE` | Zero | Zero | Redacted stop |
| Current receipt with artifact/marker tamper | `INVALID_MISMATCH` | Zero | Zero | Redacted stop |
| Producer request/row/contract binding drifts | `INVALID_MISMATCH` | Zero | Zero | Redacted stop |
| Wrapper writes receipt | `INVALID_MISMATCH` | Source has run | Zero | Redacted stop; new root required |
| Symlink/hard-link/path escape | Invalid | Zero | Zero | Redacted stop |
| Four rounds, 16 unique q-level rows, shared receipts allowed | All valid | First-use groups only | 16 rows | Completion may pass |
| Any Gold176 row enters measurement | Irrelevant | Stop | Stop | Completion fails |

## Stop conditions

Stop before private execution when:

- zero-process preflight finds any context, receipt namespace, source-output
  namespace, public round, private round leaf, Stage1 manifest, plan, registry,
  or state from an earlier attempt;
- a fresh context cannot be exclusively and atomically created from the exact
  task/revision/root/plan/registry identities;
- projected same-group source contracts drift across q-modes;
- group classification is any invalid state;
- a new environment key, ambient environment, shell, guessed path, directory
  search, or marker-parent receipt inference would be required.

Stop after source execution and before downstream when:

- the wrapper returns nonzero, creates a receipt, omits any declared source
  artifact, produces unsafe types/links, reverses first-use marker order, or
  fails receipt publication;
- any selected group fails immediate `READY_CURRENT_RUN` revalidation.

Stop closure and do not update final docs when:

- a downstream stage does not process every selected row or mutates an
  immutable shared source bundle;
- four rounds, 16 unique selected rows, five metrics, feedback hashes,
  producer bindings, receipt mappings, or zero Gold176 overlap do not verify;
- success would depend on global group uniqueness, per-row retraining, bare
  markers, synthetic feedback, static fallback, or an in-place retry.

## Rejected alternatives

### Global group uniqueness

Rejecting every later q-mode of a previously selected group changes the
row-level search space and acquisition behavior. It is not a source-integrity
rule and can make a valid 16-row dynamic run unavailable.

### Per-row or per-q-mode retraining

Recipe-v2 intentionally defines one trained bundle per canonical Pyramid
group. Retraining an identical q-independent bundle wastes private compute and
creates competing ownership of shared paths without adding q-specific
evidence.

### Bare-marker reuse

Marker existence and mtime do not bind a task, revision, root, plan, registry,
request, row, contract, artifact content, or fresh nonce. Markers alone cannot
distinguish current, stale, partial, copied, or tampered output and therefore
never authorize reuse.

## Self-review

- No unresolved placeholder, illustrative private absolute path, unresolved
  schema key, or alternate receipt location remains.
- One q-independent bundle per group is consistent with row-level q-mode
  selection, same-round mixed q, later-round reuse, and 16-row completion.
- The exact five-key environment is unchanged; context and receipt discovery
  use the existing round-root caller boundary.
- Planned-absent preflight and runtime-existing validation are separate.
- Markers are first-use ordering evidence only; adapter-owned receipts are the
  sole reuse authority.
- Completion counts selected rows, not receipts, and never remeasures Gold176.
- This document defines architecture and acceptance only; it is not an
  implementation-completion claim or an execution checklist.
