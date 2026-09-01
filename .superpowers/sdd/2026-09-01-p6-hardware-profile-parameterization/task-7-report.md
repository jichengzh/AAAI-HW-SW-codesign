# Task 7 Report: Shared Controller Runner and Verifier

## Outcome

- The existing `run_p6_h800_search.py` entrypoint remains the single runner and
  now describes its v2 H800 / v3 selected-profile compatibility explicitly.
- `P6MaterializerCompletionReport` publishes one additional allowlisted field:
  `hardware_profile`, containing only the canonical registry profile id.
- Completion verification now binds the public/local pair, private binding,
  post-source adapter profile, persisted candidate plan, and every projected
  round request to the same selected `HardwareExecutionProfile` before the
  unchanged four-round, 16-unique-row, five-metric, and zero-Gold176 checks.
- Legacy H800 completion remains valid without a post-source profile file. An
  explicit non-H800 profile requires a canonical post-source v4 profile.
- Post-source evidence is read without following symlinks; missing/dangling or
  non-regular profile leaves fail through the existing sanitized verifier error.

## Strict RED to GREEN evidence

The required profile slice was first run before production changes:

```bash
PYTHONPATH=. pytest -q \
  tests/release/test_run_p6_h800_search.py \
  tests/release/test_verify_p6_materializer_training_run.py \
  -k 'hardware_profile or rtx'
```

Corrected RED result: 2 intended failures, 7 passes. The report dataclass lacked
`hardware_profile`, and a valid completed RTX evidence tree failed because the
verifier projected requests with the legacy H800 default.

After the minimal verifier change, the same slice passed:

```text
9 passed, 52 deselected
```

The security review added one further regression for a dangling private
post-source profile symlink:

- RED: `1 failed, 45 deselected` because the dangling link was treated as an
  absent legacy H800 profile.
- GREEN: `1 passed, 45 deselected` after switching to no-follow `lstat` type
  validation.

## Coverage added

- v3 RTX contract/local profile forwarding through the existing runner entrypoint.
- runner rejection of public/local profile mismatch before adapter launch.
- byte-compatible legacy H800 CLI/state behavior.
- a locally completed fake RTX four-round tree with 16 distinct selections and
  zero Gold176 overlap.
- rejection of otherwise completed 4/16 RTX evidence when the binding,
  post-source profile, or persisted candidate plan carries H800 evidence.
- exact public report allowlist and legacy H800 report profile provenance.
- rejection of a dangling post-source profile symlink without private leakage.

The fake RTX completion uses the existing executable, controller, round
iteration, request projection, measurement CLI subprocess, selection, feedback,
and completion mechanics. Only the external leaf executables and GPU probe are
local deterministic fixtures; no GPU, remote, or private production runtime is
launched.

## Verification

Required full suites:

```text
62 passed in 93.19s
```

Required branch coverage gate:

```text
46 passed in 56.88s
tools/release/verify_p6_materializer_training_run.py: 82% branch coverage
Required test coverage of 80% reached
```

Static checks:

```text
python -m ruff check ... : passed
git diff --check        : passed
python -m py_compile ...: passed
```

The passing pytest commands emitted pre-existing temporary cleanup warnings
from normalization fixture teardown; both commands exited zero.

## Review gate

The public JSON surface is exactly the completion dataclass fields. The only new
field is the registry id string; no binding mappings, GPU indices/UUIDs, paths,
commands, receipts, logs, manifests, or private error details are serialized.
All invalid evidence continues to collapse to `history_execution_invalid` in the
API and `verification_failed` in the CLI.

## Fix round 1: real RTX execution bridge

The former scoped concern is resolved. A new real-path regression first removed
both test-only profile injections and reproduced `execution_failed` from the v3
RTX executable before round completion. The focused measurement bridge tests
also reproduced a missing `profile` argument and proved that unknown or
target-mismatched private binding profiles reached runner/probe construction.

The controller now projects each request with `local.hardware_profile`. The
measurement CLI derives a canonical registry profile from the private binding's
validated public projection, validates the complete private execution binding
against that exact profile, and only then constructs the runner/probe and calls
`run_history_measurement_batch(profile=...)`. No public option, GPU override, or
free-form environment selector was added.

Fix-round GREEN evidence:

```text
real v3 RTX executable: 1 passed in 19.91s
binding bridge/rejection slice: 3 passed in 1.38s
owned release suites: 94 passed in 171.30s
controller/fresh-run suites: 151 passed in 49.39s
verifier branch coverage: 46 passed, 82.03%
```

The release suite initially exposed two unrelated fixture failures: the copied
private test runtime omitted the registry-declared `configs/hardware` and
`configs/environment` YAMLs. The same failures reproduced with the measurement
CLI fix removed. The focused adapter-chain fixture now copies those declared
configuration directories, restoring legacy H800 execution without changing
production runtime behavior.

## Scoped concern

No functional concern remains in the Task 7 scope. Coverage collection for the
measurement CLI reports 49% because most CLI cases execute it in subprocesses;
the new in-process profile selection and pre-execution rejection branches are
covered directly. The verifier retains its enforced 80% branch gate at 82.03%.
