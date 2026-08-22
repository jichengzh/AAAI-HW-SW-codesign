"""Public completion report surface for P6 materializer training."""

from __future__ import annotations

from tools.release.verify_p6_materializer_training_run import (
    P6MaterializerCompletionReport,
)


def test_completion_report_does_not_publish_receipt_mapping() -> None:
    """Receipt reuse is internal evidence, never part of the public report."""
    assert tuple(P6MaterializerCompletionReport.__dataclass_fields__) == (
        "schema_version",
        "status",
        "completed_rounds",
        "selected_rows",
        "gold176_remeasured_rows",
    )
