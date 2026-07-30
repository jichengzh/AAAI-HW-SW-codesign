# Anonymous AAAI Submission: Code and Data

This archive contains a scoped subset of validation, selection-interface, and
aggregation code, together with public demonstration data, a small verified
audit artifact, and tests. It is intended to support anonymous inspection
without identifying the authors or development location.

## Quick start

Create a Python 3.10 or later environment, then install the reproducibility
dependencies:

```bash
pip install -e '.[repro,dev]'
python scripts/reproduce/reproduce_all.py --mode smoke --output-root ./run-output
```

The smoke mode uses only the bundled demonstration data.  Its generated
manifests record the source inputs and checksums needed for inspection.

The smoke workflow is a deterministic CPU-only interface exercise. It does not
run the complete evolutionary candidate generator, model training or
materialization stack, hardware scheduling and measurement executors, or the
full online-ablation experiment. The included lightweight selection policy
checks the public candidate and feedback contracts only; it is not an
equivalence claim or a numerical reproduction of the complete paper method.
The complete implementation and corresponding experiment manifests are
planned for release upon publication.

## Included material

- `framework/` contains the released validation, selection-interface, and
  aggregation modules.
- `scripts/reproduce/` contains the reproducibility entry points.
- `data/` contains public demonstration inputs.
- `artifacts/` contains checked-in verified outputs.
- `tests/` contains the automated checks.

Run the test suite with `pytest -q` after installing the development extras.
