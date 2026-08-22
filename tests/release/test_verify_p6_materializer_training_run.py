"""Public completion report surface for P6 materializer training."""

from __future__ import annotations

from tools.release.verify_p6_materializer_training_run import (
    P6MaterializerCompletionReport,
    verify_materializer_training_run,
)
from tests.release.test_run_p6_h800_search import _history_cli_fixture, _run_cli


def test_completion_report_does_not_publish_receipt_mapping() -> None:
    """Receipt reuse is internal evidence, never part of the public report."""
    assert tuple(P6MaterializerCompletionReport.__dataclass_fields__) == (
        "schema_version",
        "status",
        "completed_rounds",
        "selected_rows",
        "gold176_remeasured_rows",
    )


def test_completion_accepts_four_round_current_run_with_shared_receipts(tmp_path) -> None:
    """Four completed batches prove rows, not distinct receipt count, are selected."""
    paths = _history_cli_fixture(tmp_path)
    assert _run_cli(paths, env=paths["env"]).returncode == 0
    report = verify_materializer_training_run(
        public_contract_path=paths["contract"],
        local_config_path=paths["local"],
        private_binding_path=paths["binding"],
    )
    assert (report.status, report.completed_rounds, report.selected_rows) == (
        "completed", 4, 16
    )
