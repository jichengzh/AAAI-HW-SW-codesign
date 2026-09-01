"""Exact-layout, read-only completion check for a P6 materializer run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Literal


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage5.single_target_search_v2 import validate_search_task  # noqa: E402
from framework.stage6.coptv2x_h800_search_v2 import (  # noqa: E402
    P6CoptV2XExecutionError,
    PublicP6CoptV2XContract,
    _build_search_task,
    _load_search_inputs,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_formal_plan_contract_v1 import (  # noqa: E402
    validate_p6_candidate_plan,
)
from framework.stage6.hardware_execution_profile_v1 import (  # noqa: E402
    HardwareExecutionProfile,
)
from framework.stage6.p6_history_binding_v1 import (  # noqa: E402
    validate_history_execution_binding,
)
from framework.stage6.p6_external_training_binding_v1 import (  # noqa: E402
    external_training_binding_from_contract,
)
from framework.stage6.p6_history_feedback_validation_v1 import (  # noqa: E402
    translate_history_feedback,
)
from framework.stage6.p6_history_measurement_v1 import (  # noqa: E402
    resolve_validated_history_round_paths,
)
from framework.stage6.p6_history_source_materialization_v1 import (  # noqa: E402
    project_source_materialization_request,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (  # noqa: E402
    PROFILE_SCHEMA_VERSION_V4,
    ValidatedPostSourceAdapterProfile,
    load_post_source_adapter_profile,
    require_post_source_adapter_profile_v4,
)
from framework.stage6.p6_performance_round_adapter_v1 import (  # noqa: E402
    ValidatedPerformanceEvidence,
    validate_completed_performance_evidence,
)
from framework.stage6.p6_public_report_v1 import (  # noqa: E402
    P6HardwareSpecificReportProvenance,
    validate_hardware_specific_report_provenance,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (  # noqa: E402
    canonical_json_sha256,
    load_fresh_run_context,
    require_selected_groups_ready_current_run,
)


@dataclass(frozen=True)
class P6MaterializerCompletionReport:
    schema_version: Literal["p6_materializer_training_bridge_completion_v1"]
    status: Literal["completed"]
    hardware_profile: str
    provenance: P6HardwareSpecificReportProvenance
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


def _read_json_value(path: Path, *, root: Path) -> Any:
    """Read one regular private JSON leaf without following links."""
    relative = path.relative_to(root)
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or (
            index < len(relative.parts) - 1 and not stat.S_ISDIR(mode)
        ) or (index == len(relative.parts) - 1 and not stat.S_ISREG(mode)):
            raise ValueError
    return json.loads(path.read_text(encoding="utf-8"))


def _require_native_finalization_leaves(paths: Mapping[str, Path]) -> None:
    root = paths["history_root"]
    round_root = paths["round_root"]
    finalized = _read_json_value(
        round_root / "final/stage5_feedback_v2_final.json", root=root
    )
    final_audit = _read_json_value(
        round_root / "final/stage5_feedback_v2_audit.json", root=root
    )
    atomic = _read_json_value(round_root / "final/atomic_batch_audit.json", root=root)
    promoted = _read_json_value(
        round_root / "actual_feedback/stage5_feedback_v3_actual.json", root=root
    )
    promotion_audit = _read_json_value(
        round_root / "actual_feedback/actual_feedback_batch_audit_v3.json", root=root
    )
    if (
        not isinstance(finalized, list)
        or len(finalized) != 4
        or not isinstance(final_audit, Mapping)
        or final_audit.get("schema_version") != "stage5_feedback_batch_v2"
        or not isinstance(promoted, list)
        or len(promoted) != 4
        or not isinstance(atomic, Mapping)
        or atomic.get("schema_version") != "stage5_atomic_batch_audit_v2"
        or not isinstance(promotion_audit, Mapping)
        or promotion_audit.get("schema_version")
        != "stage5_actual_feedback_batch_audit_v3"
    ):
        raise ValueError


P6_AP_METRIC_PROTOCOL = "coptv2x-ap30-ap50-ap70-v1"


def _load_verification_context(
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
) -> tuple[
    PublicP6CoptV2XContract,
    Any,
    Mapping[str, Any],
    Path,
    list[Any],
    Any,
    Mapping[str, Any],
    ValidatedPostSourceAdapterProfile | None,
    tuple[int, ...],
]:
    contract = load_public_contract(public_contract_path)
    local = load_local_config(local_config_path, contract)
    binding = _read_mapping(
        private_binding_path, root=private_binding_path.parent.resolve(strict=True)
    )
    profile = contract.hardware_profile
    interface = validate_history_execution_binding(binding, profile)
    private_root_value = binding.get("private_root")
    if not isinstance(private_root_value, str):
        raise ValueError
    private_root = Path(private_root_value).resolve(strict=True)
    post_source = _require_post_source_profile(private_root, profile)
    frozen_gold, context = _load_output_context(local, contract, profile)
    indices = tuple(int(index) for index in binding["gpu_policy"]["indices"])
    return (
        contract,
        local,
        interface,
        private_root,
        frozen_gold,
        context,
        binding,
        post_source,
        indices,
    )


def _load_output_context(
    local: Any,
    contract: PublicP6CoptV2XContract,
    profile: HardwareExecutionProfile,
) -> tuple[list[Any], Any]:
    frozen_gold, _, _, capability_profile = _load_search_inputs(local, contract)
    task_contract = validate_search_task(
        _build_search_task(contract, capability_profile)
    )
    candidate_plan = _read_mapping(
        local.local_output_root / "pyramid_candidate_plan.json",
        root=local.local_output_root,
    )
    validate_p6_candidate_plan(candidate_plan, profile=profile)
    return frozen_gold, load_fresh_run_context(
        local_output_root=local.local_output_root,
        expected_task_id=task_contract["task_id"],
        expected_task_sha256=task_contract["task_sha256"],
    )


def _completion_provenance(
    contract: PublicP6CoptV2XContract,
    binding: Mapping[str, Any],
    state: Mapping[str, Any],
    source_digests: list[str],
    performance_evidence: list[ValidatedPerformanceEvidence],
    gpu_count: int,
) -> P6HardwareSpecificReportProvenance:
    profile = contract.hardware_profile
    training = external_training_binding_from_contract(
        binding["source_contract_template"]
    )
    parameters = training["training_parameters"]
    _require_performance_training_identity(training, performance_evidence)
    return validate_hardware_specific_report_provenance(
        {
            "schema_version": "p6_hardware_specific_report_provenance_v3",
            "comparison_scope": "hardware_specific",
            "hardware_profile": profile.profile_id,
            "target": contract.target,
            "target_model": contract.target_model,
            "hardware_model_family": profile.target_hardware_id,
            "gpu_count": gpu_count,
            "execution_backend": contract.execution_backend,
            "tvm_arch": profile.tvm_arch,
            "tvm_cache_namespace": profile.tvm_cache_namespace,
            "declared_environment_contract_digest": _sha256_file(
                REPOSITORY_ROOT / profile.environment_contract_path
            ),
            "runtime_observation_status": (
                "observed" if performance_evidence else "unavailable_legacy_profile"
            ),
            "observed_runtime_digest": (
                _aggregate_digest(
                    "observed_runtime",
                    [
                        digest
                        for summary in performance_evidence
                        for digest in summary.observed_runtime_digests
                    ],
                )
                if performance_evidence
                else None
            ),
            "code_revision": state["code_revision"],
            "source_digest": _aggregate_digest("sources", source_digests),
            "compiler_toolchain_digest": (
                _compiler_digest(state, performance_evidence)
                if performance_evidence
                else None
            ),
            "latency_energy_hardware_profile": profile.profile_id,
            "pareto_hardware_profile": profile.profile_id,
            "ap_provenance": {
                "data_split": parameters["dataset_split"],
                "checkpoint_initial_state": training["base_checkpoint_sha256"],
                "training_config_digest": training["pyramid_config_sha256"],
                "seed": parameters["seed"],
                "metric_protocol": P6_AP_METRIC_PROTOCOL,
                "dataset_snapshot_digest": None,
                "evaluation_snapshot_digest": None,
                "cross_hardware_comparison_status": (
                    "unavailable_unverified_dataset_identity"
                ),
            },
        }
    )


def _require_performance_training_identity(
    training: Mapping[str, Any], evidence: list[ValidatedPerformanceEvidence]
) -> None:
    for summary in evidence:
        if set(summary.configuration_digests) != {training["pyramid_config_sha256"]}:
            raise ValueError
        if set(summary.checkpoint_digests) != {training["base_checkpoint_sha256"]}:
            raise ValueError


def _compiler_digest(
    state: Mapping[str, Any], evidence: list[ValidatedPerformanceEvidence]
) -> str:
    return canonical_json_sha256(
        {
            "code_revision": state["code_revision"],
            "code_digests": sorted(
                digest for summary in evidence for digest in summary.code_digests
            ),
            "toolchain_ids": sorted(
                toolchain for summary in evidence for toolchain in summary.toolchain_ids
            )
            or ["tvm_auto"],
            "observed_runtime_digests": sorted(
                digest
                for summary in evidence
                for digest in summary.observed_runtime_digests
            ),
        }
    )


def _aggregate_digest(label: str, values: list[str]) -> str:
    if not values:
        raise ValueError
    return canonical_json_sha256({label: sorted(values)})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_post_source_profile(
    private_root: Path, profile: HardwareExecutionProfile
) -> ValidatedPostSourceAdapterProfile | None:
    profile_path = private_root / "post-source-adapter-profile.yaml"
    try:
        profile_mode = profile_path.lstat().st_mode
    except FileNotFoundError:
        if profile.profile_id != "h800":
            raise ValueError
        return None
    if stat.S_ISLNK(profile_mode) or not stat.S_ISREG(profile_mode):
        raise ValueError
    post_source = load_post_source_adapter_profile(
        profile_path,
        private_root=private_root,
    )
    if profile.profile_id != "h800":
        require_post_source_adapter_profile_v4(post_source)
    if post_source.hardware_profile is not profile:
        raise ValueError
    return post_source


def _requires_native_performance_attestation(
    profile: ValidatedPostSourceAdapterProfile | None,
) -> bool:
    return profile is not None and profile.schema_version == PROFILE_SCHEMA_VERSION_V4


def _require_completed_state(local_output_root: Path) -> Mapping[str, Any]:
    state = _read_mapping(
        local_output_root / "state.json", root=local_output_root
    )
    if (
        state.get("status") != "completed"
        or state.get("completed_rounds") != 4
        or state.get("measured_candidate_count") != 16
    ):
        raise ValueError
    return state


def _validated_round_request(
    *,
    public_round: Path,
    paths: Mapping[str, Path],
    local_output_root: Path,
    context: Any,
    interface: Mapping[str, Any],
    private_root: Path,
    hardware_profile: HardwareExecutionProfile,
) -> Mapping[str, Any]:
    public_request = _read_mapping(
        public_round / "measurement_request.json", root=local_output_root
    )
    private_request = _read_mapping(
        paths["measurement_request"], root=paths["history_root"]
    )
    public_projected = project_source_materialization_request(
        public_request, hardware_profile
    ).request
    private_projected = project_source_materialization_request(
        private_request, hardware_profile
    ).request
    if public_projected != private_projected:
        raise ValueError
    request = public_projected
    payload = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    if request.get("measurement_request_sha256") != canonical_json_sha256(payload):
        raise ValueError
    require_selected_groups_ready_current_run(
        request,
        run_context=context,
        local_output_root=local_output_root,
        interface=interface,
        private_root=private_root,
    )
    return request


def _validated_round_ids(
    rows: object,
    row_ids: set[str],
    measurement_ids: set[str],
) -> tuple[set[str], set[str]]:
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError
    next_row_ids = set(row_ids)
    next_measurement_ids = set(measurement_ids)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in next_row_ids:
            raise ValueError
        identity = _identity(row)
        if identity in next_measurement_ids:
            raise ValueError
        next_row_ids.add(row_id)
        next_measurement_ids.add(identity)
    return next_row_ids, next_measurement_ids


def _verify_completed_round(
    *,
    round_index: int,
    local: Any,
    interface: Mapping[str, Any],
    private_root: Path,
    context: Any,
    row_ids: set[str],
    measurement_ids: set[str],
    post_source_profile: ValidatedPostSourceAdapterProfile | None,
    gpu_indices: tuple[int, ...],
) -> tuple[
    set[str], set[str], ValidatedPerformanceEvidence | None, tuple[str, ...]
]:
    public_round = local.local_output_root / f"round-{round_index:02d}"
    paths = resolve_validated_history_round_paths(
        interface, private_root, public_round, round_index
    )
    request = _validated_round_request(
        public_round=public_round,
        paths=paths,
        local_output_root=local.local_output_root,
        context=context,
        interface=interface,
        private_root=private_root,
        hardware_profile=local.hardware_profile,
    )
    performance_evidence = (
        validate_completed_performance_evidence(
            post_source_profile, request, paths["round_root"], gpu_indices
        )
        if _requires_native_performance_attestation(post_source_profile)
        else None
    )
    _require_native_finalization_leaves(paths)
    feedback = translate_history_feedback(
        request,
        interface,
        paths,
        read_json=lambda path: _read_mapping(path, root=paths["history_root"]),
    )
    if not _successful_feedback(feedback.get("rows")):
        raise ValueError
    next_rows, next_measurements = _validated_round_ids(
        request.get("rows"), row_ids, measurement_ids
    )
    source_digests = tuple(str(row["source_evidence_sha256"]) for row in request["rows"])
    return next_rows, next_measurements, performance_evidence, source_digests


def verify_materializer_training_run(
    *,
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
) -> P6MaterializerCompletionReport:
    """Resolve canonical evidence paths and prove all four rounds completed."""
    try:
        return _verified_completion_report(
            public_contract_path, local_config_path, private_binding_path
        )
    except P6CoptV2XExecutionError:
        raise
    except Exception:
        raise P6CoptV2XExecutionError(
            "history_execution_invalid", "history execution invalid"
        ) from None


def _verified_completion_report(
    public_contract_path: Path,
    local_config_path: Path,
    private_binding_path: Path,
) -> P6MaterializerCompletionReport:
    (
        contract,
        local,
        interface,
        private_root,
        frozen_gold,
        context,
        binding,
        post_source_profile,
        gpu_indices,
    ) = _load_verification_context(
        public_contract_path, local_config_path, private_binding_path
    )
    state = _require_completed_state(local.local_output_root)
    row_ids, measurement_ids, evidence, source_digests = _verify_all_rounds(
        local, interface, private_root, context, post_source_profile, gpu_indices
    )
    gold_ids = {_identity(row) for row in frozen_gold}
    if len(row_ids) != 16 or len(measurement_ids) != 16 or measurement_ids & gold_ids:
        raise ValueError
    provenance = _completion_provenance(
        contract, binding, state, source_digests, evidence, len(gpu_indices)
    )
    return P6MaterializerCompletionReport(
        schema_version="p6_materializer_training_bridge_completion_v1",
        status="completed",
        hardware_profile=local.hardware_profile.profile_id,
        provenance=provenance,
        completed_rounds=4,
        selected_rows=16,
        gold176_remeasured_rows=0,
    )


def _verify_all_rounds(
    local: Any,
    interface: Mapping[str, Any],
    private_root: Path,
    context: Any,
    post_source_profile: ValidatedPostSourceAdapterProfile | None,
    gpu_indices: tuple[int, ...],
) -> tuple[set[str], set[str], list[ValidatedPerformanceEvidence], list[str]]:
    row_ids: set[str] = set()
    measurement_ids: set[str] = set()
    evidence: list[ValidatedPerformanceEvidence] = []
    source_digests: list[str] = []
    for round_index in range(4):
        row_ids, measurement_ids, round_evidence, round_sources = (
            _verify_completed_round(
                round_index=round_index,
                local=local,
                interface=interface,
                private_root=private_root,
                context=context,
                row_ids=row_ids,
                measurement_ids=measurement_ids,
                post_source_profile=post_source_profile,
                gpu_indices=gpu_indices,
            )
        )
        if round_evidence is not None:
            evidence.append(round_evidence)
        source_digests.extend(round_sources)
    return row_ids, measurement_ids, evidence, source_digests


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
