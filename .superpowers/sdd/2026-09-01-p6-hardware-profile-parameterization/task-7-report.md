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

The fake RTX completion uses the existing controller, round iteration,
selection, feedback, and completion mechanics with a local injected fake
measurement boundary. It launches no GPU, remote, or private production runtime.

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

## Scoped concern

Task 7 did not modify unowned controller or measurement modules. The fake RTX
fixture has to inject the selected profile at two existing optional-profile
call sites: request projection in `coptv2x_h800_search_v2.py` and the release
measurement adapter's `run_history_measurement_batch` call. Without that
injection, those call sites select the legacy H800 default. The named runner
does load and forward v3 contract/local profiles correctly, but a real RTX
subprocess execution requires those upstream/downstream call sites to forward
the selected profile before Task 10.
