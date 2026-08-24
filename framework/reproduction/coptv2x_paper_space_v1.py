"""CPU-only public reproduction for the three CoptV2X paper search spaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from framework.stage1.adapters import ScanScenario, TraceContext, get_adapter
from framework.stage1.structural_axes import axis_to_scanner_dict
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from framework.stage1_bridge import load_stage2_search_space
from framework.stage2.formal_software_space_v1 import build_formal_software_plan


SUPPORTED_MODELS = ("pyramid", "codriving", "fcooper")
_MODEL_TO_EVIDENCE_MODEL = {
    "pyramid": "pyramid_lidar",
    "codriving": "codriving",
    "fcooper": "fcooper",
}
_EVIDENCE_FIELDS = frozenset(
    {
        "schema", "model", "source_provenance", "loaded_config",
        "checkpoint_evidence", "prune_groups", "dataflow_relations",
        "scan_scenario", "diagnostics",
    }
)
_SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "scan_manifest_digest", "source_group_count",
        "declared_relevant_group_ids", "relevant_groups_digest",
        "purification",
    }
)
_CHECKPOINT_EVIDENCE_FIELDS = frozenset({"digest", "module_widths"})
_SCAN_SCENARIO_FIELDS = frozenset(
    {
        "hardware_precisions", "backend_precisions", "compression_modes",
        "graph_quant_unit_policy", "alignment",
    }
)


@dataclass(frozen=True)
class PaperSpaceReproduction:
    """Derived public artifacts for one paper-model search-space reproduction."""

    model: str
    evidence_path: Path
    _evidence: Mapping[str, Any] = field(repr=False)
    _scanner_manifest: Mapping[str, Any] = field(repr=False)
    _search_space: Mapping[str, Any] = field(repr=False)
    _formal_plan: Mapping[str, Any] = field(repr=False)

    @property
    def evidence(self) -> dict[str, Any]:
        """Return a fresh mutable evidence artifact copy."""
        return _deep_thaw(self._evidence)

    @property
    def scanner_manifest(self) -> dict[str, Any]:
        """Return a fresh mutable Stage1 scanner manifest copy."""
        return _deep_thaw(self._scanner_manifest)

    @property
    def search_space(self) -> dict[str, Any]:
        """Return a fresh mutable Stage2 search-space copy."""
        return _deep_thaw(self._search_space)

    @property
    def formal_plan(self) -> dict[str, Any]:
        """Return a fresh mutable formal software-plan copy."""
        return _deep_thaw(self._formal_plan)

    def mutable_evidence(self) -> dict[str, Any]:
        """Compatibility alias for the fresh public evidence property."""
        return self.evidence

    def mutable_scanner_manifest(self) -> dict[str, Any]:
        """Compatibility alias for the fresh public scanner-manifest property."""
        return self.scanner_manifest

    def mutable_search_space(self) -> dict[str, Any]:
        """Compatibility alias for the fresh public search-space property."""
        return self.search_space

    def mutable_formal_plan(self) -> dict[str, Any]:
        """Compatibility alias for the fresh public formal-plan property."""
        return self.formal_plan


@dataclass(frozen=True)
class _WidthModule:
    out_channels: int


def repository_root() -> Path:
    """Return the repository root containing the checked-in paper fixtures."""
    return Path(__file__).resolve().parents[2]


def selected_model_names(selector: str) -> tuple[str, ...]:
    """Expand the CLI selector into the supported public model sequence."""
    if selector == "all":
        return SUPPORTED_MODELS
    if selector in SUPPORTED_MODELS:
        return (selector,)
    raise ValueError(f"unsupported model selector: {selector}")


def paper_scanner_evidence_path(
    model: str, repo_root: Path | None = None
) -> Path:
    """Return the repository-relative purified scanner-evidence fixture path."""
    _require_supported_model(model)
    root = repo_root or repository_root()
    return root / "tests" / "fixtures" / "paper_spaces" / (
        f"{model}_scanner_evidence.yaml"
    )


def load_paper_scanner_evidence(
    model: str, repo_root: Path | None = None
) -> dict[str, Any]:
    """Load one checked-in purified scanner-evidence fixture."""
    path = paper_scanner_evidence_path(model, repo_root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _validated_evidence(model, payload)


def derive_paper_axis_bundle(
    model: str, evidence: Mapping[str, Any] | None = None
) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    """Derive sealed structural axes from purified scanner evidence."""
    evidence_payload = (
        load_paper_scanner_evidence(model)
        if evidence is None
        else _validated_evidence(model, evidence)
    )
    adapter = get_adapter(_evidence_model(evidence_payload))
    loaded_config = _mapping(evidence_payload["loaded_config"], "loaded_config")
    context = TraceContext(
        net=object(),
        example_inputs=(),
        full_model=object(),
        loaded_config=loaded_config,
        checkpoint_evidence=_mapping(
            evidence_payload["checkpoint_evidence"], "checkpoint_evidence"
        ),
        materializer_sources=adapter.materializer_parameter_sources(loaded_config),
        trace_modules=_trace_modules(evidence_payload),
        dataflow_relations=tuple(evidence_payload["dataflow_relations"]),
    )
    inputs = _build_axis_inputs(context, evidence_payload)
    bundle = _derive_axes(inputs)
    return evidence_payload, inputs, bundle


def build_stage1_scanner_manifest(
    model: str, evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a public Stage1 scanner manifest from purified evidence."""
    evidence_payload, inputs, bundle = derive_paper_axis_bundle(model, evidence)
    scenario = _mapping(evidence_payload["scan_scenario"], "scan_scenario")
    scanner_axes = [axis_to_scanner_dict(axis, inputs) for axis in bundle.axes]
    return _stage1_manifest(evidence_payload, scenario, scanner_axes)


def build_paper_space_reproduction(model: str) -> PaperSpaceReproduction:
    """Run evidence → public bridge → formal planner for one supported model."""
    _require_supported_model(model)
    evidence_path = paper_scanner_evidence_path(model)
    evidence = load_paper_scanner_evidence(model)
    manifest = build_stage1_scanner_manifest(model, evidence)
    with TemporaryDirectory(prefix=f"paper-space-{model}-") as tmp:
        manifest_path = Path(tmp) / f"{model}_scanner_manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        search_space = load_stage2_search_space(manifest_path)
    formal_plan = build_formal_software_plan(search_space)
    return PaperSpaceReproduction(
        model=model,
        evidence_path=evidence_path,
        _evidence=_deep_freeze(evidence),
        _scanner_manifest=_deep_freeze(manifest),
        _search_space=_deep_freeze(search_space),
        _formal_plan=_deep_freeze(formal_plan),
    )


def build_paper_space_summaries(selector: str) -> tuple[dict[str, Any], ...]:
    """Return deterministic summary rows for the selected public model(s)."""
    return tuple(
        _summary_row(build_paper_space_reproduction(model))
        for model in selected_model_names(selector)
    )


def _build_axis_inputs(context: TraceContext, evidence: Mapping[str, Any]) -> dict[str, Any]:
    from framework.stage1.structural_axes import build_structural_axis_inputs

    return build_structural_axis_inputs(
        trace_context=context,
        prune_groups=evidence["prune_groups"],
        scenario=ScanScenario(**evidence["scan_scenario"]),
        group_manifest=evidence["source_provenance"],
    )


def _derive_axes(inputs: Mapping[str, Any]) -> Any:
    from framework.stage1.structural_axes import derive_structural_axes

    return derive_structural_axes({"structural_axis_inputs": inputs})


def _stage1_manifest(
    evidence: Mapping[str, Any],
    scenario: Mapping[str, Any],
    scanner_axes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "stage1_scanner_contract_v1",
        "model": evidence["model"],
        "scan_status": "ok",
        "hw_capability": _hardware_capability(scenario),
        "backend_support": {"precisions": list(scenario["backend_precisions"])},
        "compression_modes": list(scenario["compression_modes"]),
        "quant_units": _quant_units(scenario),
        "scanner_structural_axes": list(scanner_axes),
        "scanner_structural_axes_digest": scanner_structural_axes_digest(
            scanner_axes
        ),
        "view_b1_search_groups": [],
        "view_b2_quant_units": [],
        "view_d_routing_segments": {"segments": []},
    }


def _hardware_capability(scenario: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": "h800",
        "ips": {"gpu": {"precisions": list(scenario["hardware_precisions"])}},
    }


def _quant_units(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = _mapping(
        scenario["graph_quant_unit_policy"], "scan_scenario.graph_quant_unit_policy"
    )
    return [
        {"id": unit, "legal_precisions": list(precisions)}
        for unit, precisions in policy.items()
    ]


def _summary_row(result: PaperSpaceReproduction) -> dict[str, Any]:
    plan = result.formal_plan
    return {
        "model": result.model,
        "structures": plan["structure_count"],
        "candidates": plan["candidate_count"],
        "q_modes": list(plan["q_modes"]),
    }


def _trace_modules(evidence: Mapping[str, Any]) -> dict[str, _WidthModule]:
    checkpoint = _mapping(evidence["checkpoint_evidence"], "checkpoint_evidence")
    widths = _mapping(checkpoint["module_widths"], "checkpoint_evidence.module_widths")
    return {str(path): _WidthModule(int(width)) for path, width in widths.items()}


def _validated_evidence(model: str, evidence: object) -> dict[str, Any]:
    _require_supported_model(model)
    payload = _mapping(evidence, "paper scanner evidence")
    _exact_fields(payload, _EVIDENCE_FIELDS, "paper scanner evidence")
    if payload.get("schema") != "paper_scanner_evidence_v1":
        raise ValueError("unsupported paper scanner evidence schema")
    expected = _MODEL_TO_EVIDENCE_MODEL[model]
    if payload.get("model") != expected:
        raise ValueError(
            f"paper scanner evidence model does not match requested model: "
            f"{payload.get('model')!r} != {expected!r}"
        )
    _exact_fields(
        _mapping(payload["source_provenance"], "source_provenance"),
        _SOURCE_PROVENANCE_FIELDS,
        "source_provenance",
    )
    _exact_fields(
        _mapping(payload["checkpoint_evidence"], "checkpoint_evidence"),
        _CHECKPOINT_EVIDENCE_FIELDS,
        "checkpoint_evidence",
    )
    _mapping(payload["loaded_config"], "loaded_config")
    _mapping(payload["checkpoint_evidence"]["module_widths"], "module_widths")
    _mapping(payload["scan_scenario"], "scan_scenario")
    _exact_fields(payload["scan_scenario"], _SCAN_SCENARIO_FIELDS, "scan_scenario")
    _mapping(payload["scan_scenario"]["graph_quant_unit_policy"], "graph_quant_unit_policy")
    _mapping(payload["scan_scenario"]["alignment"], "alignment")
    _required_sequence(payload["prune_groups"], "prune_groups")
    _required_sequence(payload["dataflow_relations"], "dataflow_relations")
    _mapping(payload["diagnostics"], "diagnostics")
    return deepcopy(dict(payload))


def _evidence_model(evidence: Mapping[str, Any]) -> str:
    model = evidence.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("paper scanner evidence model must be a non-empty string")
    return model


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any], fields: frozenset[str], field_name: str
) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"{field_name} has unknown fields: {unknown}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"{field_name} missing required fields: {missing}")


def _required_sequence(value: object, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{field_name} must be a non-empty sequence")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field_name} entries must be objects")
    return value


def _require_supported_model(model: str) -> None:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported paper model: {model}")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_deep_thaw(child) for child in value)
    return value


__all__ = [
    "PaperSpaceReproduction",
    "SUPPORTED_MODELS",
    "build_paper_space_reproduction",
    "build_paper_space_summaries",
    "build_stage1_scanner_manifest",
    "derive_paper_axis_bundle",
    "load_paper_scanner_evidence",
    "paper_scanner_evidence_path",
    "repository_root",
    "selected_model_names",
]
