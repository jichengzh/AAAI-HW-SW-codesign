"""Trace-boundary planning primitives for Stage1.

This module is the first autonomy layer above the old AutoTraceAdapter
registry.  It inspects a full model module tree, tags modules with conservative
heuristics, and emits a TracePlan that can be embedded in Stage1 manifests.
The plan is an auditable boundary proposal; it is not a model-level
separability verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

try:
    import torch_pruning as tp
except Exception:  # noqa: BLE001
    tp = None

from framework.stage1.trace_plan_support import (
    HETER_BASELINE_MODELS,
    TRACE_PLAN_SCHEMA,
    _ATTENTION_HINTS,
    _FUSION_HINTS,
    _FrozenModuleKeyMap,
    _POSTPROCESS_HINTS,
    _ROUTING_HINTS,
    _SPARSE_HINTS,
    _TRAINING_HINTS,
    _as_list,
    _extend_checkpoint_module_key_map,
    _get_module_by_path,
    _has_any,
    _is_under_any,
    _module_path,
    _param_count,
    _sort_paths,
    _text_for,
    generic_candidates,
    heter_candidates,
    heter_rejections,
    legacy_trace_plan_from_manifest_impl,
    plan_from_candidates,
    tag_record,
    validate_boundary,
)


@dataclass
class ModuleRecord:
    """One module-tree inventory row."""

    path: str
    type_name: str
    class_name: str
    depth: int
    n_children: int
    params_direct: int
    params_recursive: int
    tags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = _module_path(self.path)
        return data


@dataclass
class TraceCandidate:
    """A candidate dense trace boundary before wrapper validation."""

    candidate_id: str
    entry: str
    input_shape: list[int] | None
    included_modules: list[str]
    ignored_layers: list[str]
    skipped_subgraphs: list[str]
    confidence: str
    selection_reason: str
    wrapper_kind: str = "module_path"
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModuleTreeScanner:
    """Collect module inventory from a full nn.Module."""

    def scan(self, model: nn.Module) -> list[ModuleRecord]:
        records: list[ModuleRecord] = []
        for path, module in model.named_modules():
            class_name = type(module).__name__
            type_name = f"{type(module).__module__}.{class_name}"
            records.append(
                ModuleRecord(
                    path=path,
                    type_name=type_name,
                    class_name=class_name,
                    depth=0 if not path else path.count(".") + 1,
                    n_children=len(list(module.children())),
                    params_direct=_param_count(module, recurse=False),
                    params_recursive=_param_count(module, recurse=True),
                )
            )
        return records


class HeuristicTagger:
    """Conservative module tags used by TraceBoundaryDetector."""

    def tag(self, record: ModuleRecord) -> ModuleRecord:
        return tag_record(record, ModuleRecord)

    def tag_many(self, records: Iterable[ModuleRecord]) -> list[ModuleRecord]:
        return [self.tag(record) for record in records]

    def skipped_type(self, record: ModuleRecord, *, model_name: str = "") -> tuple[str, bool, str | None]:
        text = _text_for(record.path, record.type_name, record.class_name)
        model = str(model_name).lower()
        is_fusion_boundary = _has_any(text, _FUSION_HINTS) or record.path.endswith("fusion_net")

        if _has_any(text, _SPARSE_HINTS):
            return "sparse_or_geometry_preprocess", False, "trace_sparse_frontend_boundary"
        if _has_any(text, _POSTPROCESS_HINTS):
            return "postprocess_or_decode", True, "postprocess_coverage_gate"
        if _has_any(text, _TRAINING_HINTS):
            return "training_or_eval_only", False, None
        if _has_any(text, _ATTENTION_HINTS):
            return "attention_or_routing_fusion", True, "attention_fusion_coverage_anchor"
        if _has_any(text, _ROUTING_HINTS):
            return "attention_or_routing_fusion", True, "routing_fusion_coverage_anchor"
        if "maxfusion" in text or "max_fusion" in text or (is_fusion_boundary and model == "fcooper"):
            return "fusion_or_alignment", False, "maxfusion_coverage_anchor"
        if is_fusion_boundary and model in {"where2comm", "v2vnet", "disconet"}:
            return "attention_or_routing_fusion", True, "routing_fusion_coverage_anchor"
        if is_fusion_boundary and model == "attfuse":
            return "attention_or_routing_fusion", True, "attention_fusion_coverage_anchor"
        if _has_any(text, _FUSION_HINTS):
            return "fusion_or_alignment", True, "routing_fusion_coverage_anchor"
        return "custom_untraced_subgraph", True, "custom_subgraph_coverage_gate"


class DensePathFinder:
    """Generate dense trace candidates from tagged module inventory."""

    def find(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        tagger: HeuristicTagger,
        model_name: str,
        input_shape: Iterable[int] | None = None,
    ) -> tuple[list[TraceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
        modality = _infer_modality(records_by_path)
        if modality:
            return self._find_heter_baseline(
                records=records,
                records_by_path=records_by_path,
                tagger=tagger,
                model_name=model_name,
                modality=modality,
                input_shape=list(input_shape) if input_shape is not None else None,
            )
        return self._find_generic(
            records=records,
            records_by_path=records_by_path,
            tagger=tagger,
            model_name=model_name,
            input_shape=list(input_shape) if input_shape is not None else None,
        )

    def _top_skipped_paths(
        self,
        *,
        records: list[ModuleRecord],
        included_paths: list[str],
    ) -> list[str]:
        skip_records = [
            record
            for record in records
            if record.path
            and "skipped_candidate" in record.tags
            and not _is_under_any(record.path, included_paths)
        ]
        top_skip_paths: list[str] = []
        for path in _sort_paths(record.path for record in skip_records):
            if not _is_under_any(path, top_skip_paths):
                top_skip_paths.append(path)
        return top_skip_paths

    def _find_heter_baseline(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        tagger: HeuristicTagger,
        model_name: str,
        modality: str,
        input_shape: list[int] | None,
    ) -> tuple[list[TraceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
        backbone = f"backbone_{modality}"
        shrinker = f"shrinker_{modality}"
        included_paths = [
            path
            for path in (backbone, shrinker, "cls_head", "reg_head", "dir_head")
            if path in records_by_path
        ]
        ignored_paths = [
            path
            for path in ("cls_head", "reg_head", "dir_head")
            if path in records_by_path
        ]
        top_skip_paths = self._top_skipped_paths(records=records, included_paths=included_paths)
        skipped = [
            _skip_ref(tagger, records_by_path[path], model_name=model_name)
            for path in top_skip_paths
        ]
        rejected = heter_rejections(
            backbone, modality, included_paths, records_by_path
        )
        candidates = heter_candidates(
            TraceCandidate,
            backbone=backbone,
            shrinker=shrinker,
            modality=modality,
            input_shape=input_shape,
            included_paths=included_paths,
            ignored_paths=ignored_paths,
            skipped=skipped,
        )
        return candidates, skipped, rejected

    def _find_generic(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        tagger: HeuristicTagger,
        model_name: str,
        input_shape: list[int] | None,
    ) -> tuple[list[TraceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
        skipped_paths = self._top_skipped_paths(records=records, included_paths=[])
        skipped = [
            _skip_ref(tagger, records_by_path[path], model_name=model_name)
            for path in skipped_paths
        ]
        included_paths = [
            record.path
            for record in records
            if record.path
            and "dense_path_candidate" in record.tags
            and record.depth <= 2
            and not _is_under_any(record.path, skipped_paths)
        ]
        ignored_paths = [
            record.path
            for record in records
            if record.path
            and "ignored_candidate" in record.tags
            and not _is_under_any(record.path, skipped_paths)
        ]
        candidates, rejected = generic_candidates(
            TraceCandidate,
            input_shape=input_shape,
            included_paths=included_paths,
            ignored_paths=ignored_paths,
            skipped=skipped,
        )
        return candidates, skipped, rejected


class GeneratedTraceWrapper(nn.Module):
    """Wrapper generated from a TraceCandidate.

    First version supports the common BEV dense path:
    backbone(dict I/O) -> optional shrinker -> output heads.
    """

    def __init__(self, full_model: nn.Module, candidate: TraceCandidate | dict[str, Any]):
        super().__init__()
        data = candidate.to_dict() if isinstance(candidate, TraceCandidate) else dict(candidate)
        self.candidate = data
        self.wrapper_kind = str(data.get("wrapper_kind") or "module_path")
        included = [str(item) for item in _as_list(data.get("included_modules"))]
        ignored = set(str(item) for item in _as_list(data.get("ignored_layers")))
        body_paths = [
            path
            for path in included
            if path not in ignored and "head" not in path.lower()
        ]
        head_paths = [
            path
            for path in included
            if path in ignored or "head" in path.lower()
        ]
        self.body_path_names = list(body_paths)
        self.head_path_names = list(head_paths)
        self.body = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        checkpoint_module_key_map: dict[str, str] = {}
        for path in body_paths:
            alias = self._alias(path)
            module = _get_module_by_path(full_model, path)
            self.body[alias] = module
            _extend_checkpoint_module_key_map(
                checkpoint_module_key_map,
                wrapper_root=f"body.{alias}",
                source_root=path,
                source_module=module,
            )
        for path in head_paths:
            alias = self._alias(path)
            module = _get_module_by_path(full_model, path)
            self.heads[alias] = module
            _extend_checkpoint_module_key_map(
                checkpoint_module_key_map,
                wrapper_root=f"heads.{alias}",
                source_root=path,
                source_module=module,
            )
        self.checkpoint_module_key_map = _FrozenModuleKeyMap(
            tuple(checkpoint_module_key_map.items())
        )

    @staticmethod
    def _alias(path: str) -> str:
        return path.replace(".", "__")

    @staticmethod
    def _extract_dense_feature(value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("spatial_features_2d", "spatial_features", "bev_feature", "features"):
                if key in value:
                    return value[key]
        return value

    def _run_module(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        try:
            y = module({"spatial_features": x})
        except Exception:
            y = module(x)
        return self._extract_dense_feature(y)

    def forward(self, spatial_features: torch.Tensor):
        feat = spatial_features
        for path in self.body_path_names:
            feat = self._run_module(self.body[self._alias(path)], feat)
        outs = []
        for path in self.head_path_names:
            outs.append(self.heads[self._alias(path)](feat))
        if outs:
            return tuple(outs)
        return feat


class WrapperSynthesizer:
    """Build executable wrappers from trace candidates."""

    def synthesize(
        self,
        full_model: nn.Module,
        candidate: TraceCandidate | dict[str, Any],
    ) -> nn.Module:
        return GeneratedTraceWrapper(full_model, candidate)


class BoundaryValidator:
    """Validate generated wrapper candidates with Stage1 dry-run gates."""

    def validate(
        self,
        wrapper: nn.Module,
        example_input: torch.Tensor,
        *,
        ignored_layers: list[nn.Module] | None = None,
        run_prune: bool = True,
    ) -> dict[str, Any]:
        return validate_boundary(
            wrapper,
            example_input,
            ignored_layers=ignored_layers,
            run_prune=run_prune,
            pruning_module=tp,
        )


def _module_ref(records_by_path: dict[str, ModuleRecord], path: str, role: str, reason: str) -> dict[str, Any]:
    record = records_by_path.get(path)
    if record is None:
        return {"name": path, "role": role, "reason": reason}
    return {
        "name": _module_path(path),
        "type": record.class_name,
        "role": role,
        "param_count": record.params_recursive,
        "tags": list(record.tags),
        "reason": reason,
    }


def _skip_ref(tagger: HeuristicTagger, record: ModuleRecord, *, model_name: str) -> dict[str, Any]:
    skip_type, blocker, gate = tagger.skipped_type(record, model_name=model_name)
    return {
        "name": _module_path(record.path),
        "type": skip_type,
        "description": f"{_module_path(record.path)} ({record.class_name}) auto-skipped by module-tree heuristic",
        "full_model_verdict_blocker": bool(blocker),
        "blocker_gate": gate,
        "source": "trace_boundary_detector.module_tree",
        "tags": list(record.tags),
        "reason": "; ".join(record.reasons) or "matched Stage1 skipped boundary heuristic",
    }


def _infer_modality(records_by_path: dict[str, ModuleRecord]) -> str | None:
    for modality in ("m1", "m2", "m3", "m4"):
        if f"backbone_{modality}" in records_by_path:
            return modality
    for path in records_by_path:
        if path.startswith("backbone_") and len(path) >= len("backbone_m1"):
            return path.split("_", 1)[1].split(".", 1)[0]
    return None


def _infer_model_key(model_name: str, config_path: str = "", full_model: nn.Module | None = None) -> str:
    text = f"{model_name} {config_path} {type(full_model).__name__ if full_model is not None else ''}".lower()
    for key in ("where2comm", "v2vnet", "disconet", "attfuse", "fcooper"):
        if key in text:
            return key
    if "att" in text and "fusion" in text:
        return "attfuse"
    return model_name or "unknown"


def _ckpt_status_from_path(ckpt_path: str | Path | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if not ckpt_path:
        return "missing_architecture_scan_only"
    return "ok" if Path(ckpt_path).is_file() else "missing_architecture_scan_only"


class TraceBoundaryDetector:
    """Produce TracePlan dictionaries from full-model module trees."""

    def __init__(
        self,
        scanner: ModuleTreeScanner | None = None,
        tagger: HeuristicTagger | None = None,
        path_finder: DensePathFinder | None = None,
    ):
        self.scanner = scanner or ModuleTreeScanner()
        self.tagger = tagger or HeuristicTagger()
        self.path_finder = path_finder or DensePathFinder()

    def detect(
        self,
        full_model: nn.Module,
        *,
        model_name: str,
        config_path: str = "",
        ckpt_path: str = "",
        ckpt_status: str | None = None,
        input_shape: Iterable[int] | None = None,
        manual_override_used: bool = False,
    ) -> dict[str, Any]:
        records = self.tagger.tag_many(self.scanner.scan(full_model))
        records_by_path = {record.path: record for record in records}
        model_key = _infer_model_key(model_name, config_path, full_model)
        candidates, skipped, rejected = self.path_finder.find(
            records=records,
            records_by_path=records_by_path,
            tagger=self.tagger,
            model_name=model_key,
            input_shape=input_shape,
        )
        is_heter = model_key in HETER_BASELINE_MODELS or _infer_modality(
            records_by_path
        )
        detector_kind = "heter_baseline" if is_heter else "generic"
        return self._plan_from_candidates(
            records=records,
            records_by_path=records_by_path,
            full_model=full_model,
            model_name=model_key,
            config_path=config_path,
            ckpt_path=ckpt_path,
            ckpt_status=_ckpt_status_from_path(ckpt_path, ckpt_status),
            manual_override_used=manual_override_used,
            detector_name=f"TraceBoundaryDetector.{detector_kind}_v1",
            candidates=candidates,
            skipped=skipped,
            rejected=rejected,
        )

    def _plan_from_candidates(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        full_model: nn.Module,
        model_name: str,
        config_path: str,
        ckpt_path: str,
        ckpt_status: str,
        manual_override_used: bool,
        detector_name: str,
        candidates: list[TraceCandidate],
        skipped: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return plan_from_candidates(
            records=records,
            records_by_path=records_by_path,
            full_model=full_model,
            model_name=model_name,
            config_path=config_path,
            ckpt_path=ckpt_path,
            ckpt_status=ckpt_status,
            manual_override_used=manual_override_used,
            detector_name=detector_name,
            candidates=candidates,
            skipped=skipped,
            rejected=rejected,
            module_ref=_module_ref,
        )


def attach_runtime_validation(
    trace_plan: dict[str, Any] | None,
    *,
    forward_status: str | None = None,
    depgraph_status: str | None = None,
    prune_status: str | None = None,
    n_prunable_groups: int | None = None,
    output_shapes: list[list[int]] | None = None,
) -> dict[str, Any] | None:
    """Add graph_scan validation results to a TracePlan dict."""

    if not isinstance(trace_plan, dict):
        return trace_plan
    plan = dict(trace_plan)
    candidate = dict(plan.get("selected_candidate", {}) or {})
    validation = dict(candidate.get("validation", {}) or {})
    if forward_status is not None:
        validation["wrapper_forward_dryrun"] = forward_status
        validation["boundary_validator"] = "BoundaryValidator.graph_scan_integrated_v1"
    if depgraph_status is not None:
        validation["depgraph_build"] = depgraph_status
    if prune_status is not None:
        validation["prune_dryrun"] = prune_status
        validation["interface_invariant_check"] = "ok" if prune_status == "ok" else "needs_review"
    if n_prunable_groups is not None:
        validation["n_prunable_groups"] = int(n_prunable_groups)
    if output_shapes is not None:
        validation["output_shape_sanity"] = "ok" if output_shapes else "no_tensor_output"
        validation["out_shapes"] = output_shapes
    validation.setdefault("latency_coverage_annotation", "trace_net_only")
    candidate["validation"] = validation
    plan["selected_candidate"] = candidate
    return plan


def legacy_trace_plan_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize old adapter manifests into the TracePlan schema."""
    return legacy_trace_plan_from_manifest_impl(manifest)


__all__ = [
    "TRACE_PLAN_SCHEMA",
    "ModuleRecord",
    "TraceCandidate",
    "ModuleTreeScanner",
    "HeuristicTagger",
    "DensePathFinder",
    "GeneratedTraceWrapper",
    "WrapperSynthesizer",
    "BoundaryValidator",
    "TraceBoundaryDetector",
    "attach_runtime_validation",
    "legacy_trace_plan_from_manifest",
]
