# P4 external-input contract final-fix report

## Scope

Fixed the P4 contract so registry records can reference only inputs beneath the
caller-supplied `--asset-root`.  The parser now permits only a non-empty,
non-absolute `relative_path` with no `..` path segment.  No symlink, special
file, size, TOCTOU, GPU, network, or external-asset behavior was added.

## RED

Added parametrized regression coverage for `../outside.txt` and `/outside.txt`
in `tests/release/test_external_input_registry.py`.

Command:

```text
python -m pytest -q tests/release/test_external_input_registry.py
```

Output before the implementation:

```text
..FF.                                                                    [100%]
2 failed, 3 passed in 0.07s
```

Both failures reported that `load_registry()` did not raise `RegistryError`.

## GREEN

`tools/release/validate_external_inputs.py` now validates `relative_path`
during registry parsing through `_relative_asset_path` before an
`ExternalInput` is constructed.

Commands:

```text
python -m pytest -q tests/release/test_external_input_registry.py tests/integration/test_external_input_validation.py
python -m ruff check tools/release/validate_external_inputs.py
```

Output:

```text
........                                                                 [100%]
8 passed in 0.47s
All checks passed!
```

## Files

- `tests/release/test_external_input_registry.py` — regression coverage for
  parent-directory and absolute paths.
- `tools/release/validate_external_inputs.py` — parser-level path contract.

## Commit

Implementation commit SHA: `bd2044a68bfbd9acb3eb546a2c1a78854960a040`

## Concern

None within the requested scope.  This deliberately does not attempt symlink,
special-file, size, TOCTOU, or network hardening.
