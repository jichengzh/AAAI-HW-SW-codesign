# Anonymous AAAI Submission: Code and Data

This archive contains the code, public demonstration data, verified artifacts,
and tests needed to evaluate the submitted method without identifying its
authors or development location.

## Quick start

Create a Python 3.10 or later environment, then install the reproducibility
dependencies:

```bash
pip install -e '.[repro,dev]'
python scripts/reproduce/reproduce_all.py --mode smoke --output-root ./run-output
```

The smoke mode uses only the bundled demonstration data.  Its generated
manifests record the source inputs and checksums needed for inspection.

## Included material

- `framework/` contains the implementation modules.
- `scripts/reproduce/` contains the reproducibility entry points.
- `data/` contains public demonstration inputs.
- `artifacts/` contains checked-in verified outputs.
- `tests/` contains the automated checks.

Run the test suite with `pytest -q` after installing the development extras.
