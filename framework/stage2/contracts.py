"""Stage2 integration contracts.

The public Stage2 entrypoint is intentionally thin: a Stage1 partition manifest
and a Stage1 model-classification report. Hardware context, search-space shape,
and claim boundaries are derived from those Stage1 artifacts.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from framework.stage1_bridge import load_stage2_search_space


STAGE2_OUTPUT_SCHEMA = "stage2_output_v1"
STAGE2_EVIDENCE_DELTA_SCHEMA = "stage2_evidence_delta_v1"
ALLOWED_NEW_MEASURED_BACKEND = "h800_tvm"
VALID_EVIDENCE_KINDS = {"measured", "demo", "proxy", "historical", "estimated"}


def _freeze_json_value(value: Any) -> Any:
    """Make nested evidence data immutable without changing exported JSON."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return deepcopy(value)


def _json_copy(value: Any) -> Any:
    """Return a detached JSON-compatible copy of frozen evidence data."""
    if isinstance(value, MappingProxyType):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_copy(item) for item in value]
    return deepcopy(value)


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid classification JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"classification JSON must decode to an object: {path}")
    return data


def _load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid manifest: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest must decode to an object: {path}")
    return data


def _manifest_digest(path: str | Path) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid manifest path: {path}") from exc
    return hashlib.sha256(data).hexdigest()


def _manifest_model(path: str | Path) -> str:
    data = _load_manifest(path)
    return str(data.get("model") or Path(path).stem.replace("_partition", ""))


def _hardware_context_from_manifest(path: str | Path) -> dict[str, Any]:
    hw = _load_manifest(path).get("hw_capability", {}) or {}
    return {
        "source": "manifest.hw_capability",
        "read_only": True,
        "path": hw.get("path"),
        "name": hw.get("name", "unknown"),
        "arch": hw.get("arch"),
        "int8_align": hw.get("int8_align"),
        "fp16_align": hw.get("fp16_align"),
        "int8_pack_factor": hw.get("int8_pack_factor"),
        "alignment_enforcement": hw.get("alignment_enforcement"),
        "legal_bits": hw.get("legal_bits", []),
    }


def _compact_public_input(manifest_path: Path, classification_path: Path | None) -> list[str]:
    fields = ["manifest_path"]
    if classification_path is not None:
        fields.append("model_classification_path")
    return fields


def _find_model_record(
    manifest_path: str | Path,
    classification: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    model = _manifest_model(manifest_path)
    manifest_name = Path(manifest_path).name
    records = classification.get("models", [])
    if not isinstance(records, list):
        raise ValueError("classification models must be a list")
    identity_mismatch = False
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("classification model records must be objects")
        item_manifest = item.get("manifest")
        same_model = str(item.get("model")) == model
        same_manifest = bool(item_manifest) and Path(str(item_manifest)).name == manifest_name
        if same_model and (not item_manifest or same_manifest):
            return deepcopy(item), False
        if same_model or same_manifest:
            identity_mismatch = True
    return None, identity_mismatch


@dataclass(frozen=True)
class Stage2Input:
    manifest_path: str | Path
    model_classification_path: str | Path | None = None

    def to_runtime_context(self) -> dict[str, Any]:
        manifest_path = Path(self.manifest_path)
        classification_path = _path_or_none(self.model_classification_path)
        return {
            "public_input": _compact_public_input(manifest_path, classification_path),
            "manifest_path": str(manifest_path),
            "model_classification_path": (
                None if classification_path is None else str(classification_path)
            ),
            "hardware_context": _hardware_context_from_manifest(manifest_path),
        }


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    runtime_mode: str
    reason: str
    model: str
    source_classification: dict[str, Any] = field(default_factory=dict)

    def to_optimization_status(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "mode": self.runtime_mode,
            "reason": self.reason,
            "source_classification": {
                "model": self.source_classification.get("model", self.model),
                "acceleration_class": self.source_classification.get("acceleration_class"),
                "classification": self.source_classification.get("classification"),
                "ckpt_status": self.source_classification.get("ckpt_status"),
                "evidence_level": self.source_classification.get("evidence_level"),
                "scope": self.source_classification.get("scope"),
            },
        }


def apply_stage1_gate(
    manifest_path: str | Path,
    model_classification_path: str | Path | None,
) -> GateDecision:
    model = _manifest_model(manifest_path)
    classification_path = _path_or_none(model_classification_path)
    if classification_path is None or not classification_path.exists():
        return GateDecision(
            allowed=False,
            runtime_mode="fail_closed",
            reason="missing_classification",
            model=model,
        )

    classification = _load_json(classification_path)
    record, identity_mismatch = _find_model_record(manifest_path, classification)
    if record is None:
        return GateDecision(
            allowed=False,
            runtime_mode="fail_closed",
            reason=(
                "model_identity_mismatch" if identity_mismatch else "classification_record_missing"
            ),
            model=model,
        )

    acceleration_class = str(record.get("acceleration_class") or record.get("verdict"))
    ckpt_status = str(record.get("ckpt_status") or "")
    if acceleration_class == "SCAN_FAILED":
        return GateDecision(
            allowed=False,
            runtime_mode="fail_closed",
            reason="scan_failed",
            model=model,
            source_classification=record,
        )
    if "missing" in ckpt_status or "architecture" in ckpt_status:
        return GateDecision(
            allowed=False,
            runtime_mode="fail_closed",
            reason="checkpoint_optimization_blocked",
            model=model,
            source_classification=record,
        )
    if acceleration_class == "SEPARABLE_ACCELERATION":
        return GateDecision(
            allowed=True,
            runtime_mode="serial",
            reason="separable_acceleration",
            model=model,
            source_classification=record,
        )
    if acceleration_class == "CO_ACCELERATION_REQUIRED":
        return GateDecision(
            allowed=True,
            runtime_mode="joint",
            reason="co_acceleration_required",
            model=model,
            source_classification=record,
        )
    return GateDecision(
        allowed=False,
        runtime_mode="fail_closed",
        reason="unknown_acceleration_class",
        model=model,
        source_classification=record,
    )


@dataclass(frozen=True)
class Stage2EvidenceRecord:
    backend: str
    hardware: str
    scope: str
    evidence_kind: str
    provenance: str
    candidate_config: dict[str, Any]
    metric: dict[str, Any]

    def __post_init__(self) -> None:
        if self.evidence_kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence_kind: {self.evidence_kind}")
        if self.evidence_kind == "measured" and self.backend != ALLOWED_NEW_MEASURED_BACKEND:
            raise ValueError("Only h800_tvm measured evidence may be new measured evidence")
        if not isinstance(self.candidate_config, dict) or not self.candidate_config:
            raise ValueError("candidate_config must be a non-empty dict")
        if not isinstance(self.metric, dict) or not self.metric:
            raise ValueError("metric must be a non-empty dict")
        object.__setattr__(self, "candidate_config", _freeze_json_value(self.candidate_config))
        object.__setattr__(self, "metric", _freeze_json_value(self.metric))

    @property
    def promotable_to_classifier(self) -> bool:
        return self.evidence_kind == "measured" and self.backend == ALLOWED_NEW_MEASURED_BACKEND

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "hardware": self.hardware,
            "scope": self.scope,
            "evidence_kind": self.evidence_kind,
            "provenance": self.provenance,
            "candidate_config": _json_copy(self.candidate_config),
            "metric": _json_copy(self.metric),
            "promotable_to_classifier": self.promotable_to_classifier,
        }


@dataclass(frozen=True)
class Stage2EvidenceDelta:
    model: str
    records: tuple[Stage2EvidenceRecord, ...] | list[Stage2EvidenceRecord] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        if not all(isinstance(record, Stage2EvidenceRecord) for record in self.records):
            raise ValueError("records must contain Stage2EvidenceRecord values")
        object.__setattr__(self, "records", tuple(self.records))

    def to_dict(self) -> dict[str, Any]:
        records = [record.to_dict() for record in self.records]
        promotable = [record for record in records if record["promotable_to_classifier"]]
        return {
            "schema": STAGE2_EVIDENCE_DELTA_SCHEMA,
            "model": self.model,
            "records": records,
            "promotable_record_count": len(promotable),
            "no_overpromotion": True,
            "notes": [
                "demo/proxy/historical evidence is not promoted to pass conclusions",
                "only h800_tvm measured records are promotable into classifier refresh",
            ],
        }


@dataclass(frozen=True)
class Stage2Output:
    schema: str
    model: str
    manifest_digest: str
    public_input: list[str]
    hardware_context: dict[str, Any]
    optimization_status: dict[str, Any]
    search_space_summary: dict[str, Any]
    arms: list[str]
    pareto_front: list[dict[str, Any]]
    recommended_configs: list[dict[str, Any]]
    optimized_scope: str
    claim_boundaries: dict[str, Any]
    evidence_delta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model": self.model,
            "manifest_digest": self.manifest_digest,
            "public_input": self.public_input,
            "hardware_context": self.hardware_context,
            "optimization_status": self.optimization_status,
            "search_space_summary": self.search_space_summary,
            "arms": self.arms,
            "pareto_front": self.pareto_front,
            "recommended_configs": self.recommended_configs,
            "optimized_scope": self.optimized_scope,
            "claim_boundaries": self.claim_boundaries,
            "evidence_delta": self.evidence_delta,
        }


def _summarize_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": search_space["schema"],
        "model": search_space["model"],
        "model_search_policy": search_space["model_search_policy"],
        "n_software_candidates": len(search_space.get("software_candidates", [])),
        "n_hardware_candidates": len(search_space.get("hardware_candidates", [])),
        "n_hierarchical_blocks": len(search_space.get("hierarchical_blocks", [])),
        "quant_axis_statuses": sorted(
            {
                candidate.get("quant_axis_status")
                for candidate in search_space.get("software_candidates", [])
            }
        ),
        "claim_scope": search_space.get("claim_scope", {}),
    }


def _claim_boundaries(
    search_space: dict[str, Any],
    gate: GateDecision,
) -> dict[str, Any]:
    claim_scope = search_space.get("claim_scope", {}) or {}
    unsupported = set(claim_scope.get("unsupported_conclusions", []))
    unsupported.update(search_space.get("unsupported_conclusions", []))
    unsupported.update(gate.source_classification.get("unsupported_conclusions", []))
    if gate.reason == "missing_classification":
        unsupported.add("automatic_optimization_without_stage1_classification")
    return {
        "optimized_scope": claim_scope.get("optimized_scope", "rsu_dense_core"),
        "trace_dense_core_latency_scope": claim_scope.get(
            "trace_dense_core_latency_scope",
            "trace_net_only",
        ),
        "full_model_claim_allowed": bool(claim_scope.get("full_model_claim_allowed", False)),
        "full_model_blockers": sorted(set(claim_scope.get("full_model_blockers", []))),
        "future_blocks": claim_scope.get("future_blocks", []),
        "unsupported_conclusions": sorted(unsupported),
    }


def _recommendations(
    search_space: dict[str, Any],
    gate: GateDecision,
) -> list[dict[str, Any]]:
    if not gate.allowed:
        return []
    candidates = search_space.get("software_candidates", [])
    if not candidates:
        return []
    candidate = candidates[0]
    points = candidate.get("software_points", [])
    active_points = [point for point in points if point.get("status") == "active"]
    selected_point = active_points[0] if active_points else (points[0] if points else {})
    return [
        {
            "id": f"{gate.runtime_mode}:{candidate['id']}",
            "runtime_mode": gate.runtime_mode,
            "decision_level": "model",
            "software_candidate": candidate["id"],
            "software_point": selected_point,
            "source": "stage1_gate_plus_stage2_search_space_smoke",
            "claim_scope": "dense_core_only",
        }
    ]


def _demo_evidence_delta(
    model: str,
    hardware_context: dict[str, Any],
    recommended_configs: list[dict[str, Any]],
) -> Stage2EvidenceDelta:
    if not recommended_configs:
        return Stage2EvidenceDelta(model=model, records=[])
    record = Stage2EvidenceRecord(
        backend=ALLOWED_NEW_MEASURED_BACKEND,
        hardware="H800 Hopper",
        scope="rsu_dense_core",
        evidence_kind="demo",
        provenance="stage2_optimize_model_demo",
        candidate_config=recommended_configs[0],
        metric={
            "latency_us": None,
            "measurement_status": "not_measured_demo_only",
            "manifest_hardware_context": hardware_context.get("name"),
        },
    )
    return Stage2EvidenceDelta(model=model, records=[record])


def build_stage2_output(stage2_input: Stage2Input) -> Stage2Output:
    manifest_path = Path(stage2_input.manifest_path)
    context = stage2_input.to_runtime_context()
    gate = apply_stage1_gate(manifest_path, stage2_input.model_classification_path)
    search_space = load_stage2_search_space(manifest_path)
    claim_boundaries = _claim_boundaries(search_space, gate)
    recommendations = _recommendations(search_space, gate)
    delta = _demo_evidence_delta(
        model=search_space["model"],
        hardware_context=context["hardware_context"],
        recommended_configs=recommendations,
    )
    return Stage2Output(
        schema=STAGE2_OUTPUT_SCHEMA,
        model=search_space["model"],
        manifest_digest=_manifest_digest(manifest_path),
        public_input=context["public_input"],
        hardware_context=context["hardware_context"],
        optimization_status=gate.to_optimization_status(),
        search_space_summary=_summarize_search_space(search_space),
        arms=["joint", "serial", "noS/default"],
        pareto_front=[] if not gate.allowed else recommendations,
        recommended_configs=recommendations,
        optimized_scope=claim_boundaries["optimized_scope"],
        claim_boundaries=claim_boundaries,
        evidence_delta=delta.to_dict(),
    )
