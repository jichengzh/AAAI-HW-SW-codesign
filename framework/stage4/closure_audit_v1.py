"""Pure, fail-closed closure audit for the public Stage4 protocol.

The audit consumes already-materialized in-memory summaries.  It intentionally
does not read experiment directories, invoke hardware, or emit input paths.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence


TARGETS = ("latency_ms", "energy_j", "ap70")
LABEL_FIELDS = {"latency_ms", "energy_j", "ap30", "ap50", "ap70", "terminal_status"}
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
REQUIRED_REPRO_ARTIFACTS = {
    "gold176_baseline_completion",
    "feedback_cost_model",
    "feedback_update_eval",
    "feedback_completion_ranking_uncertainty_pareto_replay",
}
PUBLIC_VALUE_HEADS = {
    "extra_trees_raw",
    "extra_trees_log",
    "lgbm_l1_raw",
    "lgbm_l1_log",
    "lgbm_huber_raw",
    "lgbm_huber_log",
    "lgbm_quantile_raw",
    "lgbm_quantile_log",
    "extra_trees_residual",
    "lgbm_l1_residual",
    "lgbm_huber_residual",
    "lgbm_quantile_residual",
}
PUBLIC_ACQUISITION_POLICIES = {"random", "predicted_frontier_diversity"}
PUBLIC_TRAINING_SOURCES = {"initial_coldstart", "online_feedback"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping only when it is safe to inspect as a report object."""
    return value if isinstance(value, Mapping) else None


def _record_list(value: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        return None
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unit_interval(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _canonical_heads(report: Mapping[str, Any]) -> dict[str, str]:
    targets = _mapping(report.get("targets"))
    if targets is None:
        raise ValueError("missing candidate selection evidence")
    output = {}
    for target in TARGETS:
        target_report = _mapping(targets.get(target))
        counts = _mapping(target_report.get("selected_candidate_counts")) if target_report else None
        if not counts:
            raise ValueError(f"missing candidate selection evidence for {target}")
        if any(
            not isinstance(name, str) or name not in PUBLIC_VALUE_HEADS for name in counts
        ):
            raise ValueError("candidate selection contains an unapproved public value head")
        if not all(_positive_integer(value) for value in counts.values()):
            raise ValueError("candidate selection counts must be positive integers")
        maximum = max(counts.values())
        output[target] = min(
            str(name) for name, value in counts.items() if value == maximum
        )
    return output


def _summary(completion: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _mapping(completion.get("summary"))
    if payload is None:
        raise ValueError("completion report lacks summary")
    return payload


def _policy_reports(completion: Mapping[str, Any]) -> Mapping[str, Any] | None:
    replay = _mapping(completion.get("replay"))
    return _mapping(replay.get("policies")) if replay else None


def _eligible_acquisitions(
    baseline: Mapping[str, Any], feedback: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    baseline_policies = _policy_reports(baseline)
    feedback_policies = _policy_reports(feedback)
    if baseline_policies is None or feedback_policies is None:
        raise ValueError("completion report lacks replay policies")
    if any(
        not isinstance(policy, str) or policy not in PUBLIC_ACQUISITION_POLICIES
        for policy in (*baseline_policies, *feedback_policies)
    ):
        return [], {}
    random_baseline = _mapping(baseline_policies.get("random"))
    random_feedback = _mapping(feedback_policies.get("random"))
    if random_baseline is None or random_feedback is None:
        raise ValueError("completion report lacks random replay")
    random_medians = (
        random_baseline.get("groups_to_95pct_oracle_HV_median"),
        random_feedback.get("groups_to_95pct_oracle_HV_median"),
    )
    random_success_rates = (
        random_baseline.get("success_rate"),
        random_feedback.get("success_rate"),
    )
    if not all(_positive_integer(value) for value in random_medians) or not all(
        _unit_interval(value) for value in random_success_rates
    ):
        return [], {}

    comparison: dict[str, Any] = {}
    eligible: list[str] = []
    for policy in sorted((set(baseline_policies) & set(feedback_policies)) - {"random"}):
        before = _mapping(baseline_policies[policy])
        after = _mapping(feedback_policies[policy])
        if before is None or after is None:
            continue
        medians = (
            before.get("groups_to_95pct_oracle_HV_median"),
            after.get("groups_to_95pct_oracle_HV_median"),
        )
        metrics_valid = all(_positive_integer(value) for value in medians) and all(
            _unit_interval(value)
            for value in (before.get("success_rate"), after.get("success_rate"))
        )
        passes = (
            metrics_valid
            and float(medians[0]) <= float(random_medians[0])
            and float(medians[1]) <= float(random_medians[1])
            and float(before["success_rate"]) >= float(random_success_rates[0])
            and float(after["success_rate"]) >= float(random_success_rates[1])
        )
        comparison[str(policy)] = (
            {
                "gold176_median_groups": medians[0],
                "gold176_random_median_groups": random_medians[0],
                "feedback_view_median_groups": medians[1],
                "feedback_view_random_median_groups": random_medians[1],
                "passes": bool(passes),
            }
            if metrics_valid
            else {"metric_contract_valid": False, "passes": False}
        )
        if passes:
            eligible.append(str(policy))
    return eligible, comparison


def _holdout_gate(
    holdout: Mapping[str, Any], merged_rows: Sequence[Mapping[str, Any]], *, rows_valid: bool
) -> tuple[bool, dict[str, Any]]:
    groups = _record_list(holdout.get("groups"))
    if groups is None or not groups:
        return False, {
            "group_count": 0,
            "expected_models_observed": False,
            "overlap_with_training_or_feedback_count": 0,
            "frozen_status_observed": False,
            "contract_valid": False,
        }
    for group in groups:
        leaked = sorted(LABEL_FIELDS & set(group))
        if leaked:
            raise ValueError(f"independent holdout contains label fields: {leaked}")

    holdout_id_list = [str(group.get("group_id") or "") for group in groups]
    holdout_ids = set(holdout_id_list)
    used_ids = {str(row.get("group_id") or "") for row in merged_rows}
    overlap = sorted(holdout_ids & used_ids)
    models = sorted({str(group.get("model") or "") for group in groups})
    width_keys = {
        "x".join(map(str, group.get("width") or []))
        for group in groups
        if isinstance(group.get("width") or [], list)
    }
    widths_valid = all(
        isinstance(group.get("width"), list) and bool(group.get("width")) for group in groups
    )
    freeze_audit = _mapping(holdout.get("freeze_audit"))
    prior_references = (
        _mapping(freeze_audit.get("pre_freeze_results_or_script_reference_count_by_width"))
        if freeze_audit
        else None
    )
    required_arms = holdout.get("required_arm_product")
    arms = {
        (str(item[0]), str(item[1]))
        for item in required_arms if isinstance(required_arms, list) and isinstance(item, list) and len(item) == 2
    } if isinstance(required_arms, list) else set()
    forbidden = freeze_audit.get("forbidden_until_measurement", []) if freeze_audit else []
    try:
        zero_references = prior_references is not None and all(
            int(value) == 0 for value in prior_references.values()
        )
    except (TypeError, ValueError):
        zero_references = False
    passes = (
        rows_valid
        and holdout.get("schema_version") == "stage4_independent_holdout_v1"
        and holdout.get("status") == "frozen_before_source_materialization_and_labels"
        and holdout.get("group_count") == 4
        and len(groups) == 4
        and len(holdout_ids) == 4
        and holdout.get("row_count_after_four_arm_measurement") == 16
        and arms == EXPECTED_ARMS
        and not overlap
        and models == ["codriving", "pyramid"]
        and all(holdout_id_list)
        and widths_valid
        and all(group.get("label_state_at_freeze") == "unavailable" for group in groups)
        and freeze_audit is not None
        and freeze_audit.get("overlap_with_gold176_or_feedback16_groups") == []
        and prior_references is not None
        and set(prior_references) == width_keys
        and zero_references
        and isinstance(forbidden, list)
        and LABEL_FIELDS <= set(forbidden)
    )
    return bool(passes), {
        "group_count": len(groups),
        "expected_models_observed": models == ["codriving", "pyramid"],
        "overlap_with_training_or_feedback_count": len(overlap),
        "frozen_status_observed": holdout.get("status")
        == "frozen_before_source_materialization_and_labels",
        "contract_valid": bool(passes),
    }


def _feedback_protocol_valid(
    feedback_update: Mapping[str, Any],
    canonical_heads: Mapping[str, str],
    merged_rows: Sequence[Mapping[str, Any]],
    *,
    rows_valid: bool,
) -> bool:
    folds = _record_list(feedback_update.get("folds"))
    if not rows_valid or folds is None or len(folds) != 4:
        return False
    heldouts = [str(fold.get("heldout_feedback_group") or "") for fold in folds]
    feedback_groups = set(heldouts)
    if len(feedback_groups) != 4 or not all(heldouts):
        return False
    if not all(
        isinstance(fold.get("baseline_fit_groups"), list)
        and isinstance(fold.get("calibration_groups"), list)
        for fold in folds
    ):
        return False
    try:
        baseline_sets = {
            tuple(sorted(map(str, fold.get("baseline_fit_groups", [])))) for fold in folds
        }
        calibration_sets = {
            tuple(sorted(map(str, fold.get("calibration_groups", [])))) for fold in folds
        }
    except TypeError:
        return False
    if len(baseline_sets) != 1 or len(calibration_sets) != 1:
        return False
    baseline = set(next(iter(baseline_sets)))
    calibration = set(next(iter(calibration_sets)))
    if not baseline or not calibration or baseline & calibration:
        return False
    for fold, heldout in zip(folds, heldouts):
        added_raw = fold.get("added_feedback_groups", [])
        updated_raw = fold.get("updated_fit_groups", [])
        if not isinstance(added_raw, list) or not isinstance(updated_raw, list):
            return False
        added = set(map(str, added_raw))
        updated = set(map(str, updated_raw))
        if (
            added != feedback_groups - {heldout}
            or updated != baseline | added
            or heldout in updated
            or updated & calibration
        ):
            return False

    rows_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in merged_rows:
        rows_by_group.setdefault(str(row.get("group_id") or ""), []).append(row)
    valid_groups = {
        group_id
        for group_id, group_rows in rows_by_group.items()
        if len(group_rows) == 4
        and all(all(_finite_number(row.get(target)) for target in TARGETS) for row in group_rows)
    }
    expected_baseline = {
        group_id
        for group_id in valid_groups
        if {
            (str(row.get("training_source") or ""), str(row.get("split") or ""))
            for row in rows_by_group[group_id]
        } == {("initial_coldstart", "train")}
    }
    expected_calibration = {
        group_id
        for group_id in valid_groups
        if {
            (str(row.get("training_source") or ""), str(row.get("split") or ""))
            for row in rows_by_group[group_id]
        } == {("initial_coldstart", "locked_holdout")}
    }
    expected_feedback = {
        group_id
        for group_id in valid_groups
        if {
            (str(row.get("training_source") or ""), str(row.get("split") or ""))
            for row in rows_by_group[group_id]
        } == {("online_feedback", "online_feedback")}
    }
    if (
        baseline != expected_baseline
        or calibration != expected_calibration
        or feedback_groups != expected_feedback
    ):
        return False

    before = _mapping(feedback_update.get("before_feedback"))
    after = _mapping(feedback_update.get("after_feedback"))
    if before is None or after is None or before.get("row_count") != 16 or after.get("row_count") != 16:
        return False
    feedback_rows = [
        row for group_id in expected_feedback for row in rows_by_group[group_id]
    ]
    manifest_groups = {
        str(row.get("manifest_job_id") or ""): str(row.get("group_id") or "")
        for row in feedback_rows
    }
    expected_feedback_ids = set(manifest_groups)
    if (
        len(feedback_rows) != 16
        or len(manifest_groups) != 16
        or not all(expected_feedback_ids)
        or not all(manifest_groups.values())
    ):
        return False
    before_records_by_target = _mapping(before.get("records"))
    after_records_by_target = _mapping(after.get("records"))
    if before_records_by_target is None or after_records_by_target is None:
        return False
    for target in TARGETS:
        before_records = _record_list(before_records_by_target.get(target))
        after_records = _record_list(after_records_by_target.get(target))
        if before_records is None or after_records is None:
            return False
        before_ids = [str(item.get("manifest_job_id") or "") for item in before_records]
        after_ids = [str(item.get("manifest_job_id") or "") for item in after_records]
        if (
            len(before_ids) != 16
            or len(after_ids) != 16
            or len(set(before_ids)) != 16
            or len(set(after_ids)) != 16
            or set(before_ids) != expected_feedback_ids
            or set(before_ids) != set(after_ids)
            or any(
                str(item.get("group_id") or "") != manifest_groups[item_id]
                for item, item_id in zip(before_records, before_ids)
            )
            or any(
                str(item.get("group_id") or "") != manifest_groups[item_id]
                for item, item_id in zip(after_records, after_ids)
            )
        ):
            return False
    return (
        feedback_update.get("schema_version") == "stage4_feedback_update_eval_v1"
        and feedback_update.get("canonical_value_heads") == canonical_heads
    )


def _reproducibility_valid(audit: Mapping[str, Any]) -> bool:
    artifacts = _mapping(audit.get("artifacts"))
    if (
        audit.get("schema_version") != "stage4_p1_p3_reproducibility_audit_v1"
        or audit.get("all_exact_match") is not True
        or artifacts is None
        or not REQUIRED_REPRO_ARTIFACTS <= set(artifacts)
    ):
        return False
    return all(
        isinstance(artifacts[name], Mapping)
        and artifacts[name].get("byte_exact") is True
        and isinstance(artifacts[name].get("original_sha256"), str)
        and SHA256.fullmatch(artifacts[name]["original_sha256"]) is not None
        and artifacts[name].get("original_sha256") == artifacts[name].get("rerun_sha256")
        for name in REQUIRED_REPRO_ARTIFACTS
    )


def _target_scoped_feedback_gain(feedback_update: Mapping[str, Any]) -> list[str]:
    before_feedback = _mapping(feedback_update.get("before_feedback"))
    after_feedback = _mapping(feedback_update.get("after_feedback"))
    before = _mapping(before_feedback.get("targets")) if before_feedback else None
    after = _mapping(after_feedback.get("targets")) if after_feedback else None
    if before is None or after is None:
        return []
    improved = []
    for target in TARGETS:
        old = _mapping(before.get(target))
        new = _mapping(after.get(target))
        old_metrics = _mapping(old.get("point_metrics")) if old else None
        new_metrics = _mapping(new.get("point_metrics")) if new else None
        old_interval = _mapping(old.get("interval")) if old else None
        new_interval = _mapping(new.get("interval")) if new else None
        if not all((old_metrics, new_metrics, old_interval, new_interval)):
            continue
        values = (
            old_metrics.get("mae"),
            new_metrics.get("mae"),
            old_interval.get("mean_width"),
            new_interval.get("mean_width"),
            old_interval.get("row_coverage"),
            new_interval.get("row_coverage"),
        )
        if all(_finite_number(value) for value in values) and (
            float(new_metrics["mae"]) < float(old_metrics["mae"])
            and float(new_interval["mean_width"]) <= float(old_interval["mean_width"])
            and float(new_interval["row_coverage"]) >= float(old_interval["row_coverage"])
        ):
            improved.append(target)
    return improved


def _validated_rows(value: Any) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return (), False
    rows = tuple(item for item in value if isinstance(item, Mapping))
    return rows, len(rows) == len(value)


def build_stage4_closure_audit(
    cold_cost_report: Mapping[str, Any],
    baseline_completion: Mapping[str, Any],
    feedback_completion: Mapping[str, Any],
    feedback_update: Mapping[str, Any],
    independent_holdout: Mapping[str, Any],
    *,
    reproducibility_audit: Mapping[str, Any],
    merged_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate Stage4 using summaries, identities, and hashes only.

    A valid result is deliberately a proof of protocol closure, not a statement
    that every target improved or that any private experiment asset is public.
    """
    for name, report in {
        "cold_cost_report": cold_cost_report,
        "baseline_completion": baseline_completion,
        "feedback_completion": feedback_completion,
        "feedback_update": feedback_update,
        "independent_holdout": independent_holdout,
        "reproducibility_audit": reproducibility_audit,
    }.items():
        if not isinstance(report, Mapping):
            raise ValueError(f"{name} must be a report object")
    rows, rows_valid = _validated_rows(merged_rows)
    canonical_heads = _canonical_heads(cold_cost_report)
    baseline_summary = _summary(baseline_completion)
    feedback_summary = _summary(feedback_completion)
    eligible, acquisition_comparison = _eligible_acquisitions(
        baseline_completion, feedback_completion
    )
    selected_policy = None
    if eligible:
        selected_policy = min(
            eligible,
            key=lambda policy: (
                float(acquisition_comparison[policy]["gold176_median_groups"])
                / float(acquisition_comparison[policy]["gold176_random_median_groups"])
                + float(acquisition_comparison[policy]["feedback_view_median_groups"])
                / float(acquisition_comparison[policy]["feedback_view_random_median_groups"]),
                policy,
            ),
        )
    feedback_protocol_valid = _feedback_protocol_valid(
        feedback_update, canonical_heads, rows, rows_valid=rows_valid
    )
    reproducibility_valid = _reproducibility_valid(reproducibility_audit)
    improved_feedback_targets = _target_scoped_feedback_gain(feedback_update)
    sources = Counter(str(row.get("training_source") or "") for row in rows)
    source_roles_valid = (
        rows_valid
        and set(sources) == PUBLIC_TRAINING_SOURCES
        and all(count > 0 for count in sources.values())
        and all(
            (row.get("training_source"), row.get("split"))
            in {
                ("initial_coldstart", "train"),
                ("initial_coldstart", "locked_holdout"),
                ("online_feedback", "online_feedback"),
            }
            for row in rows
        )
    )
    holdout_valid, holdout_audit = _holdout_gate(
        independent_holdout, rows, rows_valid=rows_valid
    )
    calibrator = baseline_summary.get("selected_uncertainty_method")
    gates = {
        "canonical_value_heads_frozen": feedback_update.get("canonical_value_heads")
        == canonical_heads,
        "ranker_decision_evidenced": (
            baseline_summary.get("retain_ranker") is False
            and feedback_summary.get("retain_ranker") is False
        ),
        "acquisition_not_worse_than_random": bool(eligible),
        "feedback_update_reproducible": bool(
            feedback_protocol_valid and reproducibility_valid
        ),
        "feedback_has_target_scoped_gain": bool(improved_feedback_targets),
        "training_source_roles_explicit": bool(source_roles_valid),
        "independent_holdout_frozen_before_labels": bool(holdout_valid),
        "uncertainty_calibrator_frozen": (
            calibrator == "lgbm_quantile"
            and feedback_summary.get("selected_uncertainty_method") == calibrator
            and feedback_update.get("uncertainty_method")
            == "lgbm_quantile_plus_group_conformal"
        ),
    }
    closed = all(gates.values())
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": closed,
        "stage5_search_ready": closed,
        "gates": gates,
        "canonical_value_heads": canonical_heads,
        "ranker_policy": "rejected_use_value_heads_only",
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": selected_policy,
        "eligible_acquisition_policies": eligible,
        "acquisition_comparison": acquisition_comparison,
        "feedback_update_scope": (
            "target_specific_update_evidence_not_a_claim_of_uniform_all_target_improvement"
        ),
        "reproducibility_exact_match": reproducibility_valid,
        "feedback_improved_targets": improved_feedback_targets,
        "training_source_rows": {
            source: sources.get(source, 0) for source in sorted(PUBLIC_TRAINING_SOURCES)
        },
        "independent_holdout": holdout_audit,
    }


__all__ = ["build_stage4_closure_audit"]
