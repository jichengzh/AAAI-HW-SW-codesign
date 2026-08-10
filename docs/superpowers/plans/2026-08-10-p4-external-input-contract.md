# P4 External Input Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small public registry and offline command that reports whether explicitly supplied P4 inputs are available, without downloading or executing external assets.

**Architecture:** `tools/release/validate_external_inputs.py` owns lightweight registry parsing, local presence checks, optional checksum comparison, result rendering, and the CLI. The initial checked-in registry contains only the two known Stage6/Stage7 bundle boundaries, both honestly marked unavailable; it does not pretend to be the complete 506-item P3 handoff.

**Tech Stack:** Python 3.10--3.13 standard library (`argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`) and pytest.

## Global Constraints

- Each record requires source, license, version, intended use, relative path, and availability; `sha256` is optional and is checked only when present.
- The command requires explicit `--registry`, `--asset-root`, and `--output` arguments. It does not download, run models, use GPU/hardware, or execute Stage6/Stage7.
- Missing input remains `unavailable`; `data/demo/` is never substituted.
- Do not add real datasets, models, checkpoints, ONNX files, engines, device logs, credentials, or paper results.
- Keep P4 as `进行中（本地）`; factual registry population for the remaining P3 handoff work is a later P4 task.
- Do not add mandatory file-size fields, mandatory checksums, TOCTOU handling, or special-file hardening.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `tools/release/validate_external_inputs.py` | Parser, local presence check, optional checksum comparison, result rendering, and CLI. |
| `tests/release/test_external_input_registry.py` | Unit tests for registry records and optional checksums. |
| `tests/integration/test_external_input_validation.py` | CLI tests for verified and unavailable outputs. |
| `artifacts/external/registry.json` | Initial public registry for the known Stage6/Stage7 unavailable bundles. |
| `docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md` | Format, command, and current P4 boundary. |
| `docs/AAAI27_RELEASE_AUDIT.md:157` | Records P4 as locally in progress and links the contract. |

### Task 1: Registry parser and unit tests

**Files:**

- Create: `tools/release/validate_external_inputs.py`
- Create: `tests/release/test_external_input_registry.py`

**Interfaces:**

- Consumes: a JSON document with `format`, `registry_version`, and `inputs`.
- Produces: `ExternalInput`, `InputResult`, `RegistryError`, `load_registry(path)`, and `validate_inputs(inputs, asset_root)`.

- [ ] **Step 1: Write failing tests for lightweight available and declared unavailable records**

Create `tests/release/test_external_input_registry.py` with this test setup and behavior:

```python
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/validate_external_inputs.py"


def _module():
    spec = importlib.util.spec_from_file_location("external_inputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(**overrides: object) -> dict[str, object]:
    base = {
        "input_id": "stage6-terminal-evidence",
        "asset_kind": "evidence_bundle",
        "source": "publication-pending",
        "license": {"status": "unconfirmed", "reference": "provider-terms-required"},
        "version": "unreleased",
        "relative_path": "stage6/terminal-evidence.jsonl",
        "intended_use": "stage6_representative_selection",
        "consumer_ids": [],
        "availability": "available_for_verification",
    }
    return {**base, **overrides}


def _write_registry(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"format": "aaai27_external_input_registry_v1", "registry_version": 1, "inputs": records}), encoding="utf-8")


def test_load_registry_accepts_record_without_checksum(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_record()])
    item = _module().load_registry(registry)[0]
    assert item.sha256 is None
    assert item.availability == "available_for_verification"


def test_declared_unavailable_keeps_its_reason(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_record(availability="unavailable", unavailable_reason="bundle_not_published")])
    module = _module()
    result = module.validate_inputs(module.load_registry(registry), tmp_path)
    assert result[0].status == "unavailable"
    assert result[0].reason == "bundle_not_published"
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
python -m pytest -q tests/release/test_external_input_registry.py
```

Expected: FAIL because `tools/release/validate_external_inputs.py` does not exist.

- [ ] **Step 3: Implement the smallest parser and validator**

Create `tools/release/validate_external_inputs.py` with these public types:

```python
REGISTRY_FORMAT = "aaai27_external_input_registry_v1"
RESULT_FORMAT = "aaai27_external_input_validation_v1"


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalInput:
    input_id: str
    asset_kind: str
    source: str
    license_status: str
    license_reference: str
    version: str
    relative_path: str
    intended_use: str
    consumer_ids: tuple[str, ...]
    availability: str
    unavailable_reason: str | None
    sha256: str | None


@dataclass(frozen=True)
class InputResult:
    input_id: str
    relative_path: str
    status: str
    reason: str | None
```

`load_registry(path: Path) -> tuple[ExternalInput, ...]` must require non-empty strings for `input_id`, `asset_kind`, `source`, `version`, `relative_path`, and `intended_use`; unique input IDs; a `license` object with status `confirmed`, `unconfirmed`, or `not_redistributable`; a list of string `consumer_ids`; and availability `available_for_verification` or `unavailable`. Require `unavailable_reason` only for unavailable records. Accept an optional lowercase 64-character hexadecimal `sha256`. Do not reject additional public fields.

Implement this validator behavior:

```python
def validate_inputs(inputs: tuple[ExternalInput, ...], asset_root: Path) -> tuple[InputResult, ...]:
    results: list[InputResult] = []
    for item in inputs:
        if item.availability == "unavailable":
            results.append(InputResult(item.input_id, item.relative_path, "unavailable", item.unavailable_reason))
            continue
        target = asset_root / item.relative_path
        if not target.is_file():
            results.append(InputResult(item.input_id, item.relative_path, "unavailable", "asset_missing"))
        elif item.sha256 is not None and hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256:
            results.append(InputResult(item.input_id, item.relative_path, "unavailable", "asset_sha256_mismatch"))
        else:
            results.append(InputResult(item.input_id, item.relative_path, "verified", None))
    return tuple(results)
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```bash
python -m pytest -q tests/release/test_external_input_registry.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add tools/release/validate_external_inputs.py tests/release/test_external_input_registry.py
git commit -m "feat: add P4 external input registry"
```

### Task 2: Offline CLI and black-box validation tests

**Files:**

- Modify: `tools/release/validate_external_inputs.py`
- Create: `tests/integration/test_external_input_validation.py`

**Interfaces:**

- Consumes: `python tools/release/validate_external_inputs.py --registry REGISTRY --asset-root ROOT --output OUTPUT`.
- Produces: `{"format": "aaai27_external_input_validation_v1", "status": "verified" | "unavailable", "inputs": [...]}`.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/integration/test_external_input_validation.py` with these imports and helpers:

```python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/validate_external_inputs.py"


def _record(**overrides: object) -> dict[str, object]:
    base = {
        "input_id": "stage7-formal-aggregate",
        "asset_kind": "evidence_bundle",
        "source": "publication-pending",
        "license": {"status": "unconfirmed", "reference": "provider-terms-required"},
        "version": "unreleased",
        "relative_path": "stage7/formal-aggregate.json",
        "intended_use": "stage7_formal_aggregate",
        "consumer_ids": [],
        "availability": "available_for_verification",
    }
    return {**base, **overrides}


def _write_registry(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps({"format": "aaai27_external_input_registry_v1", "registry_version": 1, "inputs": [record]}), encoding="utf-8")


def _run(registry: Path, asset_root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry), "--asset-root", str(asset_root), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
```

Then add these three subprocess tests:

```python
def test_cli_reports_verified_file_without_checksum(tmp_path: Path) -> None:
    registry, assets, output = tmp_path / "registry.json", tmp_path / "assets", tmp_path / "result.json"
    target = assets / "stage7/formal-aggregate.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"paper_evidence": false}\n', encoding="utf-8")
    _write_registry(registry, _record())
    result = _run(registry, assets, output)
    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "verified"


def test_cli_marks_missing_file_unavailable(tmp_path: Path) -> None:
    registry, assets, output = tmp_path / "registry.json", tmp_path / "assets", tmp_path / "result.json"
    assets.mkdir()
    _write_registry(registry, _record())
    result = _run(registry, assets, output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert manifest["inputs"][0]["reason"] == "asset_missing"


def test_cli_compares_checksum_only_when_given(tmp_path: Path) -> None:
    registry, assets, output = tmp_path / "registry.json", tmp_path / "assets", tmp_path / "result.json"
    target = assets / "stage7/formal-aggregate.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bundle")
    _write_registry(registry, _record(sha256=hashlib.sha256(b"different").hexdigest()))
    result = _run(registry, assets, output)
    assert result.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["inputs"][0]["reason"] == "asset_sha256_mismatch"
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
python -m pytest -q tests/integration/test_external_input_validation.py
```

Expected: FAIL because Task 1 has no CLI or result output.

- [ ] **Step 3: Implement the CLI and result renderer**

Add these functions to `tools/release/validate_external_inputs.py`:

```python
def render_result(results: tuple[InputResult, ...]) -> dict[str, object]:
    status = "verified" if all(item.status == "verified" for item in results) else "unavailable"
    return {
        "format": RESULT_FORMAT,
        "status": status,
        "inputs": [
            {"input_id": item.input_id, "relative_path": item.relative_path, "status": item.status, "reason": item.reason}
            for item in results
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = render_result(validate_inputs(load_registry(args.registry), args.asset_root))
    except RegistryError:
        payload = {"format": RESULT_FORMAT, "status": "unavailable", "inputs": [], "reason": "registry_invalid"}
    _write_json(args.output, payload)
    return 0 if payload["status"] == "verified" else 1
```

Implement `_parse_args` with required `Path` arguments `--registry`, `--asset-root`, and `--output`. Implement `_write_json` as `output.parent.mkdir(parents=True, exist_ok=True)` followed by `output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")`. End the module with `raise SystemExit(main())`.

- [ ] **Step 4: Run unit and integration tests to verify GREEN**

Run:

```bash
python -m pytest -q tests/release/test_external_input_registry.py tests/integration/test_external_input_validation.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/release/validate_external_inputs.py tests/integration/test_external_input_validation.py
git commit -m "feat: validate P4 external inputs offline"
```

### Task 3: Initial unavailable registry, documentation, and release gate

**Files:**

- Create: `artifacts/external/registry.json`
- Create: `docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md:157`
- Modify: `tests/release/test_project_handoff.py`

**Interfaces:**

- Consumes: Task 1 registry schema and Task 2 CLI.
- Produces: two explicit unavailable records and a discoverable P4 operating contract.

- [ ] **Step 1: Write a failing handoff test**

Add this test to `tests/release/test_project_handoff.py`, moving `import json` to the existing import block:

```python
def test_p4_contract_is_discoverable_and_starts_unavailable() -> None:
    registry = json.loads((REPOSITORY_ROOT / "artifacts/external/registry.json").read_text(encoding="utf-8"))
    assert (REPOSITORY_ROOT / "docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md").is_file()
    assert registry["format"] == "aaai27_external_input_registry_v1"
    assert {item["input_id"] for item in registry["inputs"]} == {"stage6-terminal-evidence", "stage7-formal-aggregate"}
    assert all(item["availability"] == "unavailable" for item in registry["inputs"])
    handoff = HANDOFF.read_text(encoding="utf-8")
    assert "P4_EXTERNAL_INPUT_CONTRACT.md" in handoff
    assert "进行中（本地）" in handoff
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest -q tests/release/test_project_handoff.py::test_p4_contract_is_discoverable_and_starts_unavailable
```

Expected: FAIL because the registry and contract document do not exist.

- [ ] **Step 3: Add the exact initial registry and documentation**

Create `artifacts/external/registry.json` with these two records:

```json
{
  "format": "aaai27_external_input_registry_v1",
  "registry_version": 1,
  "inputs": [
    {"input_id": "stage6-terminal-evidence", "asset_kind": "evidence_bundle", "source": "publication-pending", "license": {"status": "unconfirmed", "reference": "provider-terms-required"}, "version": "unreleased", "relative_path": "stage6/terminal-evidence.jsonl", "intended_use": "stage6_representative_selection", "consumer_ids": [], "availability": "unavailable", "unavailable_reason": "bundle_not_published"},
    {"input_id": "stage7-formal-aggregate", "asset_kind": "evidence_bundle", "source": "publication-pending", "license": {"status": "unconfirmed", "reference": "provider-terms-required"}, "version": "unreleased", "relative_path": "stage7/formal-aggregate.json", "intended_use": "stage7_formal_aggregate", "consumer_ids": [], "availability": "unavailable", "unavailable_reason": "bundle_not_published"}
  ]
}
```

Create `docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md` documenting the format, the exact CLI command, the two current unavailable records, and these facts: a checksum is optional; later records must use factual source/license/version information; the validator downloads nothing and runs neither Stage6 nor Stage7.

Update the P4 row in `docs/AAAI27_RELEASE_AUDIT.md` to `进行中（本地）`, link `release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md`, and state that this change establishes only the initial unavailable registry and offline checker. Do not mark P4 complete.

- [ ] **Step 4: Run focused and release regressions**

Run:

```bash
python -m pytest -q tests/release/test_external_input_registry.py tests/integration/test_external_input_validation.py tests/release/test_project_handoff.py
python -m pytest -q tests/release tests/integration/test_public_clean_clone.py tests/integration/test_anonymous_archive.py
python -m ruff check .
python -m compileall -q tools/release
git diff --check
```

Expected: all commands exit 0. The checked-in registry is allowed and expected to report both records `unavailable` when passed to the CLI.

- [ ] **Step 5: Commit Task 3**

```bash
git add artifacts/external/registry.json docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md docs/AAAI27_RELEASE_AUDIT.md tests/release/test_project_handoff.py
git commit -m "docs: start P4 external input contract"
```

## Plan Self-Review

- Spec coverage: Tasks 1--2 provide the lightweight registry and offline validator; Task 3 publishes the honest initial unavailable boundary, usage docs, handoff state, and release evidence.
- Scope: this is the P4 foundation subproject. It does not invent the actual provider, license, or version details needed to populate the remaining P3 handoff items.
- Consistency: the field names `input_id`, `relative_path`, `availability`, `unavailable_reason`, and optional `sha256` are identical in the schema, tests, CLI result, and docs.
- No execution task downloads an asset, performs external work, or changes Stage6/Stage7 behavior.
