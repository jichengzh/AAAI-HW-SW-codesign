"""Exact-layout, read-only completion check for a P6 materializer run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Literal

from framework.stage5.single_target_search_v2 import validate_search_task
from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XExecutionError,
    _build_search_task,
    _load_search_inputs,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_history_binding_v1 import validate_history_execution_binding
from framework.stage6.p6_history_feedback_validation_v1 import translate_history_feedback
from framework.stage6.p6_history_measurement_v1 import (
    resolve_validated_history_round_paths,
)
from framework.stage6.p6_history_source_materialization_v1 import (
    project_source_materialization_request,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    canonical_json_sha256,
    load_fresh_run_context,
    require_selected_groups_ready_current_run,
)


@dataclass(frozen=True)
class P6MaterializerCompletionReport:
    schema_version: Literal["p6_materializer_training_bridge_completion_v1"]
    status: Literal["completed"]
    completed_rounds: int
    selected_rows: int
    gold176_remeasured_rows: int


def _read_mapping(path: Path, *, root: Path) -> dict[str, Any]:
    """Read one canonical regular JSON leaf without following private links."""
    relative = path.relative_to(root)
    current = root
    if stat.S_ISLNK(current.lstat().st_mode) or not current.is_dir():
        raise ValueError
    for index, component in enumerate(relative.parts):
        current /= component
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or (
            index < len(relative.parts) - 1 and not stat.S_ISDIR(mode)
        ) or (index == len(relative.parts) - 1 and not stat.S_ISREG(mode)):
            raise ValueError
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError
    return payload


def _identity(row: Mapping[str, Any]) -> str:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError
    return str(row.get("manifest_job_id") or row_id)


def _successful_feedback(rows: object) -> bool:
    if not isinstance(rows, list) or len(rows) != 4:
        return False
    for row in rows:
        if not isinstance(row, Mapping) or row.get("terminal_status") != "measured_success_gold":
            return False
        metrics = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
        if set(row) != {"row_id", "terminal_status", *metrics}:
            return False
        values = {key: row[key] for key in metrics}
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values.values()):
            return False
        if float(values["latency_ms"]) <= 0 or float(values["energy_j"]) <= 0:
            return False
        if any(not 0 <= float(values[key]) <= 1 for key in ("ap30", "ap50", "ap70")):
            return False
    return True


def verify_materializer_training_run(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
) -> P6MaterializerCompletionReport:
    """Resolve canonical evidence paths and prove all four rounds completed."""
    try:
        contract = load_public_contract(public_contract_path)
        local = load_local_config(local_config_path, contract)
        binding = _read_mapping(
            private_binding_path, root=private_binding_path.parent.resolve(strict=True)
        )
        interface = validate_history_execution_binding(binding)
        private_root_value = binding.get("private_root")
        if not isinstance(private_root_value, str):
            raise ValueError
        private_root = Path(private_root_value)
        frozen_gold, _, _, profile = _load_search_inputs(local)
        task = _build_search_task(contract, profile)
        task_contract = validate_search_task(task)
        context = load_fresh_run_context(
            local_output_root=local.local_output_root,
            expected_task_id=task_contract["task_id"],
            expected_task_sha256=task_contract["task_sha256"],
        )
        state = _read_mapping(
            local.local_output_root / "state.json", root=local.local_output_root
        )
        if (
            state.get("status") != "completed"
            or state.get("completed_rounds") != 4
            or state.get("measured_candidate_count") != 16
        ):
            raise ValueError
        row_ids: set[str] = set()
        measurement_ids: set[str] = set()
        for round_index in range(4):
            public_round = local.local_output_root / f"round-{round_index:02d}"
            paths = resolve_validated_history_round_paths(
                interface, private_root, public_round, round_index
            )
            public_request = _read_mapping(
                public_round / "measurement_request.json", root=local.local_output_root
            )
            private_request = _read_mapping(
                paths["measurement_request"], root=paths["history_root"]
            )
            public_projected = project_source_materialization_request(public_request).request
            private_projected = project_source_materialization_request(private_request).request
            if public_projected != private_projected:
                raise ValueError
            request = public_projected
            if request.get("measurement_request_sha256") != canonical_json_sha256(
                {key: value for key, value in request.items() if key != "measurement_request_sha256"}
            ):
                raise ValueError
            require_selected_groups_ready_current_run(
                request,
                run_context=context,
                local_output_root=local.local_output_root,
                interface=interface,
                private_root=private_root,
            )
            feedback = translate_history_feedback(
                request,
                interface,
                paths,
                read_json=lambda path: _read_mapping(path, root=paths["history_root"]),
            )
            if not _successful_feedback(feedback.get("rows")):
                raise ValueError
            rows = request.get("rows")
            if not isinstance(rows, list) or len(rows) != 4:
                raise ValueError
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError
                row_id = row.get("row_id")
                if not isinstance(row_id, str) or not row_id or row_id in row_ids:
                    raise ValueError
                identity = _identity(row)
                if identity in measurement_ids:
                    raise ValueError
                row_ids.add(row_id)
                measurement_ids.add(identity)
        gold_ids = {_identity(row) for row in frozen_gold}
        if len(row_ids) != 16 or len(measurement_ids) != 16 or measurement_ids & gold_ids:
            raise ValueError
        return P6MaterializerCompletionReport(
            schema_version="p6_materializer_training_bridge_completion_v1",
            status="completed",
            completed_rounds=4,
            selected_rows=16,
            gold176_remeasured_rows=0,
        )
    except P6CoptV2XExecutionError:
        raise
    except Exception:
        raise P6CoptV2XExecutionError(
            "history_execution_invalid", "history execution invalid"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        report = verify_materializer_training_run(
            public_contract_path=args.contract,
            local_config_path=args.local_config,
            private_binding_path=args.binding,
        )
    except Exception:
        sys.stderr.write("verification_failed\n")
        return 1
    sys.stdout.write(json.dumps(asdict(report), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
