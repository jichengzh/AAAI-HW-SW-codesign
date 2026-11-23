from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stage5_initialize_fcooper_actual_v2 import (
    validate_formal_registry_provenance,
)

ROOT = Path(__file__).resolve().parents[2]


def test_formal_initializer_hard_gates_pilot_probe_and_recovery_evidence() -> None:
    source = (ROOT / "scripts/stage5_initialize_fcooper_actual_v2.py").read_text()

    assert "pilot_online_labels_loaded" in source
    assert "probe_metrics_loaded_as_labels" in source
    assert "numeric-gate-summary-json" in source
    assert "probe-isolation-audit-json" in source
    assert "scanner-execution-json" in source
    assert 'training_view_policy="initial_coldstart_only"' in source
    assert "S5-FCO-TRT-V2" in source


def test_formal_initializer_rejects_registry_content_from_pilot() -> None:
    registry = {
        "groups": [
            {
                "source_contract": {
                    "checkpoint": (
                        "/tmp/fcooper_workpackage_a_20260723/source/checkpoint.pth"
                    )
                }
            }
        ]
    }

    with pytest.raises(ValueError, match="pilot"):
        validate_formal_registry_provenance(registry)
