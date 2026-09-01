# Task 6 Report: Profile-Aware Bootstrap, Provision, and Preflight

## Outcome

Task 6 is implemented without a GPU/profile override flag or a new controller.
The full-chain bootstrap now resolves the public example selected by the legacy
local contract, loads the canonical registry profile from that validated public
contract, and carries that same object through post-source validation, GPU
admission, binding construction, local-config rendering, and pair validation.

RTX4090 fixtures use four ordered fake cards and exactly two provision
snapshots. Legacy H800 CLI behavior, ignored/atomic pair publication, the
four-round preflight plan, and preflight process/GPU counts of zero remain
unchanged.

## TDD evidence

Baseline before edits:

```text
92 passed in 92.06s
```

The required RED selector was run before any production edit:

```text
5 failed, 92 deselected in 3.99s
```

The failures named the missing behavior:

- v3 RTX local locators were rejected by the H800-only bootstrap;
- the legacy provision route validated RTX records with the H800 profile;
- RTX preflight inputs could not be materialized;
- the post-source mismatch did not yet reach the intended profile boundary.

After the minimal production changes, the same selector passed:

```text
5 passed, 92 deselected in 5.81s
```

## Implementation

- `PUBLIC_CONTRACT_PATH` is now the selected-example template
  `p6_{hardware_profile}_search.example.yaml`. The local profile id is accepted
  only through the immutable profile registry, and the resulting public
  contract must resolve to that exact canonical profile before downstream use.
- Legacy local v2 remains implicit H800. Local v3 requires an explicit profile
  and target matching the validated public contract. Rendered RTX local config
  is v3 and retains the same canonical profile id.
- Full-chain post-source v4 profiles must be identical to the public-contract
  profile. Legacy post-source v3 remains implicit H800. RTX requires both the
  source-wrapper and post-source private profiles.
- Post-source profile agreement is checked before source-wrapper creation and
  before the GPU probe. The binding builder receives the same canonical profile
  and still owns exactly two snapshots.
- The rendered binding is revalidated against that profile before the local
  validation tempfile or atomic pair writer is reached.
- Legacy history provisioning now passes the public-contract profile into the
  existing discovery/binding boundary. Its obsolete-route terminal category is
  unchanged.
- Preflight binds the local config, private binding, and post-source profile to
  the public-contract profile before runner and freshness validation. It adds no
  process or GPU call.

`tools/release/provision_p6_full_chain_local_config.py` required no CLI change;
its existing call already delegates all profile selection to the bootstrap.

## Independent no-side-effect gate

The reviewed call order is:

```text
validated public contract/local selection
  -> source/post-source profile agreement
  -> source-wrapper publication
  -> runner validation
  -> canonical-profile binding construction (two snapshots)
  -> binding/profile validation
  -> rendered-pair validation
  -> atomic pair publication
```

The RTX mismatch regression removes the source wrapper before invoking the
bootstrap and proves that a post-source mismatch leaves the wrapper and both
pair destinations absent with zero probe calls. The preflight mismatch
regression proves that a binding/profile mismatch stops before post-source or
runner validation. AST/source review still finds no preflight `snapshot`,
`subprocess.run`, `Popen`, or historical execution call.

## Verification

Required Task 6 suite:

```text
97 passed in 100.21s
```

Directly adjacent bootstrap/deployment regression:

```text
50 passed in 14.82s
```

Specified Ruff and diff gates:

```text
All checks passed!
git diff --check: clean
```

All seven owned source/test files also pass Ruff together. Python compilation
of the changed production modules succeeded.

Direct Coverage.py branch run completed with all 97 tests passing. Whole-file
coverage was 71% for the existing 900-line bootstrap, 89% for preflight, and
45% for the intentionally obsolete legacy provision module (67% aggregate).
The requested Task 6 brief has no coverage threshold, and expanding unreachable
legacy code/tests is outside this profile-plumbing task. Pytest-cov itself still
fails during collection under ambient Python 3.13 because NumPy is loaded twice;
this is the same pre-existing tool issue recorded by Tasks 1, 2, and 4.

## Security and scope review

- No secrets, credentials, private runtime paths, real UUIDs, or real GPU data
  were added.
- Profile-derived path formatting is whitelist-bound by the immutable registry;
  arbitrary path segments are not accepted.
- No shell invocation, profile/GPU CLI override, remote action, GPU access, or
  private runtime was used.
- Stable redacted CLI categories and atomic ignored-destination enforcement are
  unchanged.
- No files outside the Task 6 list and this required report were modified.

## Concerns

- Whole-file branch coverage remains below the repository-wide 80% preference
  because these long pre-existing modules contain obsolete and defensive paths;
  Task 6 behavior and adjacent regressions are green, but this debt remains.
- Pytest continues to emit a pre-existing temporary-directory cleanup warning
  for `.execution-closure.locked.tmp`; it does not change test exit status.
