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
    path.write_text(
        json.dumps(
            {
                "format": "aaai27_external_input_registry_v1",
                "registry_version": 1,
                "inputs": records,
            }
        ),
        encoding="utf-8",
    )


def test_load_registry_accepts_record_without_checksum(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_record()])
    item = _module().load_registry(registry)[0]
    assert item.sha256 is None
    assert item.availability == "available_for_verification"


def test_declared_unavailable_keeps_its_reason(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(
        registry,
        [_record(availability="unavailable", unavailable_reason="bundle_not_published")],
    )
    module = _module()
    result = module.validate_inputs(module.load_registry(registry), tmp_path)
    assert result[0].status == "unavailable"
    assert result[0].reason == "bundle_not_published"
