"""Private helpers shared by the public Stage1 trace-plan facade."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


TRACE_PLAN_SCHEMA = "stage1_trace_plan_v1"
HETER_BASELINE_MODELS = {
    "fcooper", "attfuse", "where2comm", "v2vnet", "disconet",
}
_DENSE_NAME_HINTS = (
    "backbone", "resnet", "base_bev_backbone", "pyramid_backbone",
    "neck", "deblock", "shrink", "shrinker",
)
_HEAD_NAME_HINTS = (
    "cls_head", "reg_head", "dir_head", "single_head", "occ_head",
    "occupancy", "aux_head",
)
_SPARSE_HINTS = (
    "pillar_vfe", "vfe", "voxel", "scatter", "sparse", "quickcumsum",
    "quicksum", "cumsum", "lift", "splat", "geometry",
)
_FUSION_HINTS = (
    "fusion", "fuse", "warp", "affine", "pairwise_t_matrix",
    "record_len", "collab", "maxfusion",
)
_ATTENTION_HINTS = (
    "attention", "attfusion", "transformer", "hmsa", "mswin", "attfuse",
)
_ROUTING_HINTS = (
    "where2comm", "v2v", "v2vnet", "disco", "communication", "routing",
    "message_passing",
)
_POSTPROCESS_HINTS = (
    "postprocess", "post_process", "nms", "decode", "box_coder", "proposal",
)
_TRAINING_HINTS = ("loss", "assigner", "target", "metric", "eval")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _module_path(module_path: str) -> str:
    return module_path or "<root>"


def _text_for(path: str, type_name: str, class_name: str = "") -> str:
    return f"{path} {type_name} {class_name}".lower()


def _has_any(text: str, hints: Iterable[str]) -> bool:
    return any(hint in text for hint in hints)


def _param_count(module: nn.Module, *, recurse: bool) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters(recurse=recurse)))


def _is_descendant(path: str, parent: str) -> bool:
    return path != parent and path.startswith(parent + ".")


def _is_under_any(path: str, parents: Iterable[str]) -> bool:
    return any(_is_descendant(path, parent) for parent in parents)


def _sort_paths(paths: Iterable[str]) -> list[str]:
    return sorted(set(paths), key=lambda path: (path.count("."), path))


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    if not path:
        return model
    module: nn.Module = model
    for part in path.split("."):
        if isinstance(module, nn.ModuleDict) and part in module:
            module = module[part]
        elif part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


@dataclass(frozen=True, eq=False)
class _FrozenModuleKeyMap(Mapping[str, str]):
    """Deepcopy-safe immutable wrapper-name to checkpoint-name mapping."""

    entries: tuple[tuple[str, str], ...]

    def __getitem__(self, key: str) -> str:
        for item_key, value in self.entries:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _extend_checkpoint_module_key_map(
    mapping: dict[str, str],
    *,
    wrapper_root: str,
    source_root: str,
    source_module: nn.Module,
) -> None:
    """Record exact source names for a generated wrapper subtree."""
    used_sources = set(mapping.values())
    for relative_name, _module in source_module.named_modules():
        wrapper_name = wrapper_root
        source_name = source_root
        if relative_name:
            wrapper_name = f"{wrapper_root}.{relative_name}"
            source_name = f"{source_root}.{relative_name}"
        if wrapper_name in mapping:
            raise ValueError("duplicate generated checkpoint module mapping")
        if source_name in used_sources:
            continue
        mapping[wrapper_name] = source_name
        used_sources.add(source_name)


def _tensor_shapes(value: Any) -> list[list[int]]:
    if torch.is_tensor(value):
        return [list(value.shape)]
    if isinstance(value, dict):
        return _nested_tensor_shapes(value.values())
    if isinstance(value, (list, tuple)):
        return _nested_tensor_shapes(value)
    return []


def _nested_tensor_shapes(values: Iterable[Any]) -> list[list[int]]:
    shapes: list[list[int]] = []
    for value in values:
        shapes.extend(_tensor_shapes(value))
    return shapes


def tag_record(record: Any, record_factory: Callable[..., Any]) -> Any:
    text = _text_for(record.path, record.type_name, record.class_name)
    tags = set(record.tags)
    reasons = list(record.reasons)
    _tag_dense(record, text, tags, reasons)
    _tag_skipped(text, tags, reasons)
    if not tags and record.n_children == 0 and record.params_recursive > 0:
        tags.add("unknown_param_leaf")
        reasons.append("parameterized leaf did not match known Stage1 boundary hints")
    return record_factory(
        path=record.path,
        type_name=record.type_name,
        class_name=record.class_name,
        depth=record.depth,
        n_children=record.n_children,
        params_direct=record.params_direct,
        params_recursive=record.params_recursive,
        tags=sorted(tags),
        reasons=sorted(set(reasons)),
    )


def _tag_dense(record: Any, text: str, tags: set[str], reasons: list[str]) -> None:
    dense_ops = ("Conv2d", "ConvTranspose2d", "Linear", "BatchNorm2d")
    if any(operator in record.class_name for operator in dense_ops):
        tags.add("dense_op")
        reasons.append("module type is a dense torch op")
    if _has_any(text, _DENSE_NAME_HINTS):
        tags.add("dense_path_candidate")
        reasons.append("name matches dense backbone/neck/shrinker hints")
    if _has_any(text, _HEAD_NAME_HINTS):
        tags.update(("head_candidate", "ignored_candidate", "dense_path_candidate"))
        reasons.append("name matches output head hints")


def _tag_skipped(text: str, tags: set[str], reasons: list[str]) -> None:
    matches = (
        (_SPARSE_HINTS, "sparse_or_geometry_preprocess", "sparse or geometry preprocess"),
        (_ATTENTION_HINTS, "attention_or_routing_fusion", "attention or transformer"),
        (_ROUTING_HINTS, "attention_or_routing_fusion", "routing or message-passing"),
        (_FUSION_HINTS, "fusion_or_alignment", "fusion or alignment"),
        (_POSTPROCESS_HINTS, "postprocess_or_decode", "postprocess/decode"),
        (_TRAINING_HINTS, "training_or_eval_only", "training/eval-only"),
    )
    for hints, tag, reason in matches:
        if _has_any(text, hints):
            tags.update(("skipped_candidate", tag))
            reasons.append(f"name/type matches {reason} hints")


def validate_boundary(
    wrapper: nn.Module,
    example_input: torch.Tensor,
    *,
    ignored_layers: list[nn.Module] | None,
    run_prune: bool,
    pruning_module: Any,
) -> dict[str, Any]:
    result = _initial_boundary_result(run_prune)
    with torch.no_grad():
        outputs = wrapper(example_input)
    shapes = _tensor_shapes(outputs)
    _record_forward_result(result, shapes)
    if pruning_module is None:
        result.update(
            depgraph_build="unavailable_torch_pruning",
            prune_dryrun="unavailable_torch_pruning",
        )
        return result
    ignored = ignored_layers or []
    groups = _dependency_groups(pruning_module, wrapper, example_input, ignored)
    result.update(depgraph_build="ok", n_prunable_groups=len(groups))
    if run_prune:
        _record_prune_result(
            result, pruning_module, wrapper, example_input, ignored
        )
    return result


def _initial_boundary_result(run_prune: bool) -> dict[str, Any]:
    return {
        "full_model_load_sanity": "not_applicable_wrapper_only",
        "wrapper_forward_dryrun": "pending",
        "output_shape_sanity": "pending",
        "depgraph_build": "pending",
        "prune_dryrun": "pending" if run_prune else "skipped",
        "interface_invariant_check": "pending",
        "latency_coverage_annotation": "trace_net_only",
    }


def _record_forward_result(result: dict[str, Any], shapes: list[list[int]]) -> None:
    result["wrapper_forward_dryrun"] = "ok"
    result["output_shape_sanity"] = "ok" if shapes else "no_tensor_output"
    result["out_shapes"] = shapes
    result["interface_invariant_check"] = "ok" if shapes else "needs_review"


def _dependency_groups(
    pruning_module: Any,
    wrapper: nn.Module,
    example_input: torch.Tensor,
    ignored_layers: list[nn.Module],
) -> list[Any]:
    graph = pruning_module.DependencyGraph().build_dependency(
        wrapper, example_inputs=example_input
    )
    return list(graph.get_all_groups(
        root_module_types=[nn.Conv2d, nn.ConvTranspose2d, nn.Linear],
        ignored_layers=ignored_layers,
    ))


def _record_prune_result(
    result: dict[str, Any],
    pruning_module: Any,
    wrapper: nn.Module,
    example_input: torch.Tensor,
    ignored_layers: list[nn.Module],
) -> None:
    wrapper_copy = copy.deepcopy(wrapper)
    copied_input = example_input.detach().clone()
    ignored_copy = _copied_ignored_layers(wrapper, wrapper_copy, ignored_layers)
    try:
        pruner = pruning_module.pruner.MetaPruner(
            wrapper_copy,
            copied_input,
            importance=pruning_module.importance.MagnitudeImportance(p=1),
            pruning_ratio=0.5,
            round_to=32,
            global_pruning=False,
            iterative_steps=1,
            ignored_layers=ignored_copy,
        )
        pruner.step()
        with torch.no_grad():
            pruned_outputs = wrapper_copy(copied_input)
        result["prune_dryrun"] = "ok"
        result["pruned_out_shapes"] = _tensor_shapes(pruned_outputs)
    except Exception as error:  # noqa: BLE001
        result["prune_dryrun"] = "fail"
        result["prune_error"] = f"{type(error).__name__}: {str(error)[:200]}"


def _copied_ignored_layers(
    wrapper: nn.Module,
    wrapper_copy: nn.Module,
    ignored_layers: list[nn.Module],
) -> list[nn.Module]:
    return [
        _get_module_by_path(wrapper_copy, name)
        for name, module in wrapper.named_modules()
        if any(module is ignored for ignored in ignored_layers)
    ]


def heter_rejections(
    backbone: str,
    modality: str,
    included_paths: list[str],
    records_by_path: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rejected = []
    if backbone not in records_by_path:
        rejected.append({
            "candidate_id": f"heter_baseline_dense_{modality}",
            "status": "rejected",
            "failed_at": "module_tree_scan",
            "error": f"missing {backbone}",
            "suggested_override": (
                "provide a manual dense entry or model-specific boundary override"
            ),
        })
    if not included_paths:
        rejected.append({
            "candidate_id": "heter_baseline_dense_core",
            "status": "rejected",
            "failed_at": "candidate_selection",
            "error": "no backbone/shrinker/head dense path was found",
            "suggested_override": "review module names or add a detector plugin",
        })
    return rejected


def heter_candidates(
    candidate_factory: Callable[..., Any],
    *,
    backbone: str,
    shrinker: str,
    modality: str,
    input_shape: list[int] | None,
    included_paths: list[str],
    ignored_paths: list[str],
    skipped: list[dict[str, Any]],
) -> list[Any]:
    if not included_paths:
        return []
    return [candidate_factory(
        candidate_id=f"heter_baseline_dense_{modality}",
        entry="post_scatter_bev",
        input_shape=input_shape,
        included_modules=included_paths,
        ignored_layers=ignored_paths,
        skipped_subgraphs=[item["name"] for item in skipped],
        confidence="medium",
        selection_reason=(
            "HeterModelBaseline dense path selected from module tree: "
            f"{backbone} -> {shrinker if shrinker in included_paths else 'heads'}."
        ),
        wrapper_kind="heter_baseline_dense_path",
    )]


def generic_candidates(
    candidate_factory: Callable[..., Any],
    *,
    input_shape: list[int] | None,
    included_paths: list[str],
    ignored_paths: list[str],
    skipped: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    if included_paths:
        return [candidate_factory(
            candidate_id="generic_dense_path",
            entry="unknown_dense_entry",
            input_shape=input_shape,
            included_modules=_sort_paths(included_paths),
            ignored_layers=_sort_paths(ignored_paths),
            skipped_subgraphs=[item["name"] for item in skipped],
            confidence="medium",
            selection_reason="generic dense path hints from module names",
            wrapper_kind="module_path",
        )], []
    return [], [{
        "candidate_id": "generic_dense_path",
        "status": "rejected",
        "failed_at": "candidate_selection",
        "error": "generic detector found no dense path candidate",
        "suggested_override": (
            "add a detector plugin or provide a manual TraceAdapter fallback"
        ),
    }]


def plan_from_candidates(
    *,
    records: list[Any],
    records_by_path: dict[str, Any],
    full_model: nn.Module,
    model_name: str,
    config_path: str,
    ckpt_path: str,
    ckpt_status: str,
    manual_override_used: bool,
    detector_name: str,
    candidates: list[Any],
    skipped: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    module_ref: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    review_reasons = _candidate_review_reasons(skipped, rejected, ckpt_status)
    selected, included, ignored, confidence = _selected_candidate(
        candidates, skipped, rejected
    )
    return {
        "schema": TRACE_PLAN_SCHEMA,
        "model": model_name,
        "model_class": type(full_model).__name__,
        "config_path": config_path,
        "ckpt_path": ckpt_path,
        "ckpt_status": ckpt_status,
        "detector": detector_name,
        "manual_override_used": bool(manual_override_used),
        "trace_confidence": confidence,
        "coverage_scope": "dense_core_only" if skipped else "full_dense_path_candidate",
        "review_required": bool(review_reasons),
        "review_reasons": sorted(set(review_reasons)),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected_candidate": selected,
        "included_modules": _module_refs(
            module_ref, records_by_path, included, "included"
        ),
        "ignored_layers": _module_refs(
            module_ref, records_by_path, ignored, "ignored"
        ),
        "skipped_subgraphs": skipped,
        "rejected_candidates": rejected,
        "module_inventory": [record.to_dict() for record in records],
    }


def _candidate_review_reasons(
    skipped: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    ckpt_status: str,
) -> list[str]:
    reasons = []
    if skipped:
        reasons.append("skipped_subgraphs_require_user_review")
    if any(item.get("full_model_verdict_blocker") for item in skipped):
        reasons.append("fusion_attention_or_routing_boundary_not_closed")
    if "missing" in str(ckpt_status):
        reasons.append("missing_checkpoint_architecture_only")
    if rejected:
        reasons.append("candidate_rejection_present")
    return reasons


def _selected_candidate(
    candidates: list[Any],
    skipped: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str], str]:
    if not candidates:
        return _no_candidate(skipped), [], [], "low"
    selected = candidates[0]
    payload = selected.to_dict()
    payload["status"] = "selected" if not rejected else "selected_with_rejections"
    payload["validation"] = {
        "full_model_module_tree_scan": "ok",
        "wrapper_forward_dryrun": "pending_graph_scan",
        "depgraph_build": "pending_graph_scan",
        "prune_dryrun": "pending_graph_scan",
    }
    confidence = selected.confidence if not rejected else "low"
    return (
        payload,
        list(selected.included_modules),
        list(selected.ignored_layers),
        confidence,
    )


def _no_candidate(skipped: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_id": "no_dense_candidate",
        "status": "no_candidate",
        "entry": "unknown",
        "input_shape": None,
        "included_modules": [],
        "ignored_layers": [],
        "skipped_subgraphs": [item["name"] for item in skipped],
        "selection_reason": "no candidate generated by DensePathFinder",
        "wrapper_kind": "none",
        "validation": {
            "full_model_module_tree_scan": "ok",
            "wrapper_forward_dryrun": "not_run_no_candidate",
            "depgraph_build": "not_run_no_candidate",
            "prune_dryrun": "not_run_no_candidate",
        },
    }


def _module_refs(
    module_ref: Callable[..., dict[str, Any]],
    records_by_path: dict[str, Any],
    paths: list[str],
    role: str,
) -> list[dict[str, Any]]:
    reason = (
        "selected dense trace path"
        if role == "included"
        else "head/interface layer kept in forward but ignored by pruning"
    )
    return [module_ref(records_by_path, path, role, reason) for path in paths]


def legacy_trace_plan_from_manifest_impl(manifest: dict[str, Any]) -> dict[str, Any]:
    trace = dict(manifest.get("trace", {}) or {})
    skipped = _legacy_skipped_subgraphs(trace)
    ckpt_status = str(manifest.get("ckpt_status") or "unknown")
    review_reasons = ["legacy_adapter_boundary_requires_review"]
    if "missing" in ckpt_status:
        review_reasons.append("missing_checkpoint_architecture_only")
    if any(item.get("full_model_verdict_blocker") for item in skipped):
        review_reasons.append("fusion_attention_or_routing_boundary_not_closed")
    return {
        "schema": TRACE_PLAN_SCHEMA,
        "model": str(manifest.get("model") or "unknown"),
        "model_class": manifest.get("model_class") or "",
        "config_path": manifest.get("config") or manifest.get("config_path") or "",
        "ckpt_path": manifest.get("ckpt") or manifest.get("ckpt_path") or "",
        "ckpt_status": ckpt_status,
        "detector": "legacy_trace_adapter_manifest_normalizer",
        "manual_override_used": True,
        "trace_confidence": "low" if "missing" in ckpt_status else "medium",
        "coverage_scope": "dense_core_only" if skipped else "trace_adapter_scope",
        "review_required": True,
        "review_reasons": sorted(set(review_reasons)),
        "selected_candidate": _legacy_selected_candidate(manifest, trace, skipped),
        "included_modules": [],
        "ignored_layers": [],
        "skipped_subgraphs": skipped,
        "rejected_candidates": [],
        "module_inventory": [],
    }


def _legacy_skipped_subgraphs(trace: dict[str, Any]) -> list[dict[str, Any]]:
    skipped = [
        dict(item)
        for item in _as_list(trace.get("skipped_subgraphs"))
        if isinstance(item, dict)
    ]
    for index, raw in enumerate(_as_list(trace.get("skipped_modules"))):
        text = str(raw)
        skip_type, blocker, gate = _legacy_skip_type(text.lower())
        skipped.append({
            "name": text.split("(", 1)[0].strip() or f"skipped_{index}",
            "type": skip_type,
            "description": text,
            "full_model_verdict_blocker": blocker,
            "blocker_gate": gate,
            "source": "legacy_trace_adapter_manifest",
        })
    return skipped


def _legacy_skip_type(text: str) -> tuple[str, bool, str]:
    if any(key in text for key in _ATTENTION_HINTS + _ROUTING_HINTS):
        gate = "attention_fusion_coverage_anchor"
        if any(key in text for key in _ROUTING_HINTS):
            gate = "routing_fusion_coverage_anchor"
        return "attention_or_routing_fusion", True, gate
    if any(key in text for key in _FUSION_HINTS):
        if "maxfusion" in text or "max pooling" in text:
            return "fusion_or_alignment", False, "maxfusion_coverage_anchor"
        return "fusion_or_alignment", True, "routing_fusion_coverage_anchor"
    if any(key in text for key in _SPARSE_HINTS):
        return "sparse_or_geometry_preprocess", False, "trace_sparse_frontend_boundary"
    return "custom_untraced_subgraph", True, "custom_subgraph_coverage_gate"


def _legacy_selected_candidate(
    manifest: dict[str, Any],
    trace: dict[str, Any],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    has_scan = bool(manifest.get("scan_status"))
    has_groups = bool(manifest.get("view_b1_prune_groups"))
    checks = manifest.get("checks", {}) or {}
    prune_status = (
        str(checks.get("dryrun_prune05", {}).get("status"))
        if isinstance(checks, dict)
        else "unknown"
    )
    return {
        "candidate_id": "legacy_trace_adapter_boundary",
        "status": "selected_legacy",
        "entry": "trace_adapter_declared_entry",
        "input_shape": trace.get("entry_shape"),
        "included_modules": [],
        "ignored_layers": [],
        "skipped_subgraphs": [item.get("name") for item in skipped],
        "selection_reason": "normalized from existing TraceAdapter manifest fields",
        "validation": {
            "full_model_module_tree_scan": "not_available_legacy_manifest",
            "wrapper_forward_dryrun": "already_run_by_graph_scan" if has_scan else "unknown",
            "depgraph_build": "already_run_by_graph_scan" if has_groups else "unknown",
            "prune_dryrun": prune_status,
        },
    }
