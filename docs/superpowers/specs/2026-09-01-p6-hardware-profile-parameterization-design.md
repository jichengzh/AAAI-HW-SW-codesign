# P6 Hardware-Profile Parameterization Design

## Status and goal

Approved design for running the existing P6 four-round experiment on either
H800 or RTX4090 without adding a second controller, verifier, or measurement
chain.  It replaces only hardware-bearing literals with a selected immutable
profile.  It does not claim an RTX result exists.

The goal is one scanner-generated, real-measurement P6 run on four RTX4090
cards: four rounds, four candidates per round, 16 distinct candidates,
`Gold176` remeasurement count zero, and the existing five real metrics.  H800
remains the default for all existing public v2 contracts and CLI invocations.

## Non-goals

- No parallel `rtx4090` controller, verifier, four-round state machine, or
  copied round adapters.
- No proxy/synthetic training, quantization, TVM, AP, latency, energy, or
  feedback rows.
- No change to candidate-space mechanics, 4x4/16 cardinality, Gold holdout,
  or the five metrics.
- No broad security, historical-source, deployment, or cache-cleanup work.
- No TensorRT engine is accepted as TVM evidence.

## Architecture

Add `framework/stage6/hardware_execution_profile_v1.py`.  It owns the two
authoritative immutable `HardwareExecutionProfile` values and the only
selection API:

```python
@dataclass(frozen=True)
class HardwareExecutionProfile:
    profile_id: Literal["h800", "rtx4090"]
    target_hardware_id: str
    hardware_capability_path: PurePosixPath
    environment_contract_path: PurePosixPath
    allowed_normalized_gpu_models: frozenset[str]
    backend_scope: frozenset[str]
    tvm_arch: str
    tvm_cache_namespace: str
    required_gpu_count: int | None
    maximum_occupancy: float

def load_hardware_execution_profile(profile_id: str) -> HardwareExecutionProfile: ...
def default_hardware_execution_profile() -> HardwareExecutionProfile: ...
def validate_profile_gpu_policy(profile: HardwareExecutionProfile,
                                indices: tuple[int, ...]) -> tuple[int, ...]: ...
```

`h800` preserves today’s admission compatibility with `required_gpu_count=None`,
the present H800 model normalizations, `tvm_auto`, `sm90`, and the existing
occupancy threshold.  `rtx4090` binds `configs/hardware/rtx4090.yaml`,
`configs/environment/rtx4090.yaml`, exact normalized RTX4090 names,
`tvm_auto`, `sm89`, a distinct `rtx4090-sm89` TVM cache namespace, four cards,
and the same occupancy threshold.  The registry exposes no paths, GPU UUIDs,
commands, results, or private roots.

Public contract v2 remains valid and selects the default H800 profile.  A new
public contract v3 adds exactly `hardware_profile`; its `target` and
`execution_backend` must agree with the registry.  Local config v3 similarly
adds `hardware_profile` and must agree with its public contract.  Existing
`p6_h800_coptv2x_search_contract_v2` / `p6_h800_coptv2x_local_v2` names,
files, and CLI arguments remain accepted.  RTX uses a tracked v3 public
example plus private ignored local inputs.

## Data flow and consistency boundary

```text
public v2/v3 contract ─┐
                       ├─> selected HardwareExecutionProfile
local v2/v3 config ────┘          │
                                  ├─ Stage1 hardware/environment consistency
private GPU policy + live probe ──┼─ binding and runtime admission
candidate plan / source contract ─┼─ target/backend consistency
post-source profile / TVM leaf ───┴─ hardware, arch, cache-namespace evidence
```

No downstream component accepts a free-form hardware string once a profile is
selected.  It receives the `HardwareExecutionProfile`, or validates a persisted
profile id against it.  This prevents an RTX policy with H800 source artifacts,
an H800 model under an RTX contract, or a TVM cache/architecture mismatch.

The existing 4-row request contract stays unchanged.  RTX’s four-card policy
is ordered and stable; source materialization and quantization assign work by
existing order, performance uses four workers, and AP creates four shards.
Thus each candidate in a round receives one designated card without creating a
new scheduling layer.

## Production surfaces

1. **Profile registry and contracts.** Add the registry; update
   `coptv2x_h800_search_v2.py` loaders and public/local dataclasses to retain
   the selected profile.  The module filename and its CLI import remain for
   compatibility; a later rename is out of scope.
2. **Stage1 and candidate plan.** Parameterize the Stage1 bridge’s hardware
   validation and `p6_formal_plan_contract_v1.py` envelope.  Add a formal RTX
   scan config derived from the existing scanner input, not a hand-written
   candidate plan.  Stage2 output must declare the selected target.
3. **Private binding and runtime.** Replace H800 model/occupancy literals in
   `p6_history_binding_v1.py`, `p6_history_measurement_v1.py`, and
   `p6_history_source_materialization_v1.py` with profile consistency checks.
   Binding persists the selected profile id and validates exactly two live
   snapshots as it does today.
4. **Normalized profiles and adapters.** Make the post-source profile and
   source/registry/feedback target fields profile-consistent.  Adapters retain
   their current mechanics; performance must pass profile-bound TVM arch and
   cache namespace to the historical TVM leaf through its existing bound
   invocation surface.
5. **Release entrypoints.** Provision, full-chain bootstrap, preflight,
   `run_p6_h800_search.py`, and the verifier load the same selected profile.
   The legacy executable name remains an alias; no new RTX controller or
   verifier executable is introduced.

`tools/release/provision_p6_history_local_config.py` is a legacy H800-only,
non-materializing diagnostic.  It is not an RTX entrypoint.  The only RTX
provision boundary is `tools/release/provision_p6_full_chain_local_config.py`.

## TVM evidence and cache policy

The measured native result must record a manifest with the selected profile id,
TVM arch, cache namespace, compiler/toolchain identity, candidate identity,
and source digest.  Reuse is permitted only when every manifest field exactly
matches the active profile and candidate/source identity.  A missing or
mismatched manifest is a cache miss and invokes the normal TVM compile path.
Artifacts identifying TensorRT, or lacking the required TVM manifest, fail
before they can become a performance result.  The cache namespace is distinct
per profile, so H800/SM90 artifacts cannot satisfy RTX/SM89 work.

## Acceptance gates

- v2 H800 config and current CLI tests remain byte-compatible in behavior;
  profile selection is implicit only for that legacy path.
- v3 contract/local/binding/profile/plan values agree on one registry profile;
  mismatch, unknown profile, wrong backend, wrong hardware YAML, and wrong
  GPU model fail closed before process launch.
- RTX binding admits exactly four ordered cards whose two live snapshots have
  stable identity, exact allowed model normalization, and threshold occupancy.
- Stage1/Stage2 candidate plan is scanner-generated for RTX; no manual plan,
  proxy space, or static fallback is accepted.
- Each RTX round executes real training, quantization, TVM performance, AP,
  and finalization, produces finite five-metric rows, and uses the selected
  SM89 TVM manifest/cache policy.
- Existing controller/verifier mechanics prove 4 completed rounds, 16 unique
  measured candidates, and zero Gold176 overlap; verifier output includes the
  selected public profile id but no private values.

## Reporting and scientific comparability

Each completed result has a hardware provenance manifest: profile id, hardware
model family, GPU count, CUDA/driver/framework versions, TVM arch/cache
namespace, code/source digests, data split/checkpoint identity, seed, and
metric protocol.  Private UUIDs, paths, raw commands, logs, and private asset
values remain ignored.

AP is reportable across hardware only when data split, checkpoint/initial
state, seed, and metric protocol match.  Latency and energy remain
hardware-specific; report them in separate tables/series, never pool them with
H800 values or label RTX measurements as H800.  Candidate-space membership,
ranking, and Pareto frontier are also hardware-specific.  A cross-hardware
analysis may show the scanner-generated intersection and per-profile rankings,
with every conclusion labeled as cross-hardware rather than same-device.

## Risks and containment

Historical leaves may contain unobserved H800/CUDA assumptions.  A profile
mismatch must stop before a real launch, while an actual leaf incompatibility
is a real failed RTX attempt, not a reason to synthesize a result.  TVM’s
historical CLI may not currently expose arch/cache arguments; then the minimal
change is a profile-bound argument mapping in the existing adapter, not a new
compiler abstraction.  RTX driver/toolchain differences may make absolute
latency or energy incomparable; provenance and separated reporting contain
that risk without weakening the real-measurement requirement.
