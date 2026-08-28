"""Small real-torch workflows for Stage1 boundaries, adapters, and graph scans."""

from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pytest

from framework.stage1.adapters import (
    MaterializerParameterSource,
    ScanScenario,
    TraceContext,
    _first_conv_in_channels,
    _generic_bucket,
    get_adapter,
)
from framework.stage1.graph_scan import _output_shapes, _resolve_profile, scan
from framework.stage1.formal_checkpoint_snapshot import FormalCheckpointSnapshot
from framework.stage1.formal_config_snapshot import FormalConfigSnapshot
from framework.stage1 import formal_trace_build
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from framework.stage1_bridge import load_stage2_search_space
from framework.stage1.trace_plan import (
    BoundaryValidator,
    GeneratedTraceWrapper,
    TraceBoundaryDetector,
    attach_runtime_validation,
    legacy_trace_plan_from_manifest,
)
from tests.stage1.trace_graph_test_support import (
    FullModel,
    ToyAdapter,
    formal_evidence,
    hardware_capability,
)


class _DiagnosticOnlyAdapter(ToyAdapter):
    build_trace_context = None  # type: ignore[assignment]


class _ForgingAdapter(ToyAdapter):
    def build_trace_context(
        self,
        net: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        loaded_config: dict,
        checkpoint_evidence: dict,
    ) -> TraceContext:
        return super().build_trace_context(
            net,
            example_inputs,
            {"model": {"backbone": {"width": 16}}},
            checkpoint_evidence,
        )


class _MutatingAdapter(ToyAdapter):
    def materializer_parameter_sources(
        self, loaded_config: dict
    ) -> tuple[MaterializerParameterSource, ...]:
        loaded_config["model"]["backbone"]["width"] = 16
        return super().materializer_parameter_sources(loaded_config)


class _ScannerOwnedEvidenceAdapter(ToyAdapter):
    def __init__(
        self, config_path: Path, checkpoint_path: Path, trusted_root: Path
    ) -> None:
        self.config_path = str(config_path)
        self.ckpt_path = str(checkpoint_path)
        self.formal_config_root = trusted_root
        self.formal_checkpoint_root = trusted_root
        self.model_config_import_root = trusted_root
        self.resolve_calls = 0
        self.loaded_checkpoint_authorities: list[FormalCheckpointSnapshot] = []
        self.formal_build_calls = 0
        self.native_load_calls = 0
        self.formal_config_identities: list[int] = []
        self.loaded_config_authorities: list[FormalConfigSnapshot] = []
        self.native_model_config = OrderedDict(
            [
                (
                    "model",
                    {"backbone": {"width": np.int64(32)}},
                )
            ]
        )

    def resolved_checkpoint_path(self) -> Path:
        self.resolve_calls += 1
        return Path(self.ckpt_path)

    def build_trace_net(
        self,
        device: str,
        checkpoint_authority: FormalCheckpointSnapshot | None = None,
    ) -> tuple[nn.Module, torch.Tensor]:
        if checkpoint_authority is not None:
            self.loaded_checkpoint_authorities.append(checkpoint_authority)
        return super().build_trace_net(device)

    def build_formal_trace_net(
        self,
        device: str,
        checkpoint_authority: FormalCheckpointSnapshot,
        config_authority: FormalConfigSnapshot,
        loaded_model_config: object | None = None,
    ) -> tuple[nn.Module, torch.Tensor, object]:
        self.formal_build_calls += 1
        self.loaded_config_authorities.append(config_authority)
        if loaded_model_config is None:
            self.native_load_calls += 1
            loaded_model_config = self.native_model_config
        self.formal_config_identities.append(id(loaded_model_config))
        net, inputs = self.build_trace_net(device, checkpoint_authority)
        return net, inputs, loaded_model_config


class _ChangingCheckpointAdapter(_ScannerOwnedEvidenceAdapter):
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        trusted_root: Path,
        *,
        replacement_path: Path,
        change_kind: str,
    ) -> None:
        super().__init__(config_path, checkpoint_path, trusted_root)
        self.replacement_path = replacement_path
        self.change_kind = change_kind
        self.changed = False

    def build_trace_net(self, device, checkpoint_authority=None):
        result = super().build_trace_net(device, checkpoint_authority)
        if not self.changed:
            checkpoint = Path(self.ckpt_path)
            if self.change_kind == "replace":
                self.replacement_path.replace(checkpoint)
            else:
                checkpoint.write_bytes(self.replacement_path.read_bytes())
            self.changed = True
        return result


class _CrossRootFormalAdapter(_ScannerOwnedEvidenceAdapter):
    build_formal_trace_net = formal_trace_build.FormalTraceBuildMixin.build_formal_trace_net

    def __init__(
        self,
        *args,
        import_root: Path,
        build_cwd: Path,
        fail_on_build: int | None,
    ) -> None:
        super().__init__(*args)
        self.model_config_import_root = import_root
        self.build_cwd = build_cwd
        self.fail_on_build = fail_on_build
        self.build_roots: list[str] = []
        self.config_roots: list[str] = []

    def build_trace_net(
        self,
        device: str,
        checkpoint_authority: FormalCheckpointSnapshot | None = None,
        loaded_model_config: object | None = None,
    ) -> tuple[nn.Module, torch.Tensor]:
        os.chdir(self.build_cwd)
        identity = importlib.import_module("opencood.model_identity").ROOT_ID
        self.build_roots.append(identity)
        if isinstance(loaded_model_config, dict):
            self.config_roots.append(str(loaded_model_config["native_root"]))
        if self.fail_on_build == len(self.build_roots):
            raise RuntimeError("formal model build failed")
        return super().build_trace_net(device, checkpoint_authority)


def _identified_native_root(root: Path, identity: str) -> Path:
    package = root / "opencood" / "hypes_yaml"
    package.mkdir(parents=True)
    (root / "opencood" / "__init__.py").write_text("", encoding="utf-8")
    (root / "opencood" / "model_identity.py").write_text(
        f"ROOT_ID = {identity!r}\n", encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "yaml_utils.py").write_text(
        "import yaml\n"
        "from opencood.model_identity import ROOT_ID\n"
        "def load_yaml(path):\n"
        "    with open(path, encoding='utf-8') as stream:\n"
        "        value = yaml.safe_load(stream)\n"
        "    value['native_root'] = ROOT_ID\n"
        "    return value\n",
        encoding="utf-8",
    )
    return root


def _restore_opencood_modules(modules: dict[str, object]) -> None:
    for name in tuple(sys.modules):
        if name == "opencood" or name.startswith("opencood."):
            sys.modules.pop(name, None)
    sys.modules.update(modules)


def _absolute_manifest_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            item
            for child in value.values()
            for item in _absolute_manifest_strings(child)
        ]
    if isinstance(value, (list, tuple)):
        return [
            item for child in value for item in _absolute_manifest_strings(child)
        ]
    if isinstance(value, str) and Path(value).is_absolute():
        return [value]
    return []


def _manifest_contains(value: object, expected: object) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_manifest_contains(child, expected) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_manifest_contains(child, expected) for child in value)
    return False


def test_trace_detector_and_generated_wrapper_preserve_dense_head_interface() -> None:
    """The detector identifies fusion as a gate and emits an executable dense wrapper."""
    full_model = FullModel().eval()
    plan = TraceBoundaryDetector().detect(
        full_model, model_name="attfuse", ckpt_status="loaded", input_shape=[1, 32, 8, 8]
    )
    wrapper = GeneratedTraceWrapper(full_model, plan["selected_candidate"])
    validation = BoundaryValidator().validate(wrapper, torch.ones(1, 32, 8, 8), run_prune=False)

    assert plan["review_required"] is True
    assert plan["selected_candidate"]["status"] == "selected"
    assert any(item["full_model_verdict_blocker"] for item in plan["skipped_subgraphs"])
    assert [list(tensor.shape) for tensor in wrapper(torch.ones(1, 32, 8, 8))] == [[1, 2, 8, 8], [1, 4, 8, 8]]
    assert validation["wrapper_forward_dryrun"] == "ok"
    assert validation["depgraph_build"] == "ok"
    assert validation["prune_dryrun"] == "skipped"


def test_legacy_trace_plan_and_runtime_validation_keep_coverage_boundaries() -> None:
    """Old manifest skips convert to typed blockers and gain non-mutating runtime evidence."""
    legacy = {"model": "v2xvit", "trace": {"skipped_modules": ["fusion attention", "sparse vfe"]}}
    plan = legacy_trace_plan_from_manifest(legacy)
    updated = attach_runtime_validation(
        plan, forward_status="ok", depgraph_status="ok", prune_status="fail", n_prunable_groups=3, output_shapes=[[1, 2]]
    )

    assert plan["coverage_scope"] == "dense_core_only"
    assert updated is not plan
    validation = updated["selected_candidate"]["validation"]
    assert validation["interface_invariant_check"] == "needs_review"
    assert validation["n_prunable_groups"] == 3


def test_toy_adapter_executes_full_cpu_graph_scan_without_hardware_measurement() -> None:
    """A real Conv graph reaches Stage1's B1/B2/routing/validation contracts on CPU."""
    manifest = scan(ToyAdapter(), hardware_capability(), device="cpu", profile_latency_mode="off")

    assert manifest["scan_status"] == "ok"
    assert manifest["view_b1_prune_groups"]
    assert manifest["view_b1_search_groups"]
    assert manifest["view_b2_quant_units"]
    assert manifest["view_d_routing_segments"]["n_segments"] >= 1
    assert manifest["view_latency"]["status"] == "skipped"
    assert manifest["trace_plan"] is None
    assert manifest["formal_scan"]["status"] == "not_requested"
    assert "scanner_structural_axes" not in manifest


def test_formal_scan_requires_trace_context_instead_of_legacy_width_fallback() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )

    loaded_config, checkpoint = formal_evidence()
    with pytest.raises(ValueError, match="TraceContext"):
        scan(
            _DiagnosticOnlyAdapter(),
            hardware_capability(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )


def test_formal_scan_rejects_adapter_without_real_scanner_evidence_inputs() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )

    with pytest.raises(ValueError, match="formal checkpoint authority"):
        scan(
            ToyAdapter(),
            hardware_capability(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
        )


def test_public_scan_rejects_non_scenario_objects_at_the_api_boundary() -> None:
    loaded_config, checkpoint = formal_evidence()

    with pytest.raises(TypeError, match="ScanScenario"):
        scan(
            ToyAdapter(),
            hardware_capability(),
            device="cpu",
            profile_latency_mode="off",
            scenario={"hardware_precisions": ["FP16"]},  # type: ignore[arg-type]
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )


def test_formal_scan_rejects_adapter_forged_scanner_evidence() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )
    loaded_config, checkpoint = formal_evidence()

    with pytest.raises(ValueError, match="scanner-owned loaded_config"):
        scan(
            _ForgingAdapter(),
            hardware_capability(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )


def test_formal_scan_prevents_adapter_mutating_scanner_evidence() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )
    loaded_config, checkpoint = formal_evidence()

    with pytest.raises(TypeError):
        scan(
            _MutatingAdapter(),
            hardware_capability(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
            loaded_config=loaded_config,
            checkpoint_evidence=checkpoint,
        )
    assert loaded_config["model"]["backbone"]["width"] == 32


def test_toy_scan_marks_real_partition_manifest_schema() -> None:
    manifest = scan(ToyAdapter(), hardware_capability(), device="cpu", profile_latency_mode="off")

    assert manifest["schema"] == "stage1_partition_manifest_v1"
    assert manifest["stage"] == "stage1_partition"


def test_scan_scenario_drives_backend_and_compression_q_sources(tmp_path) -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16",),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={"backbone": ("FP16", "INT8"), "heads": ("FP16",)},
        alignment={
            "default_round_to": 32,
            "default_max_rate_numerator": 3,
            "default_max_rate_denominator": 4,
        },
    )
    loaded_config, checkpoint = formal_evidence()
    adapter = ToyAdapter()
    adapter.config_path = str(tmp_path / "private" / "config.yaml")
    adapter.ckpt_path = str(tmp_path / "private" / "checkpoint.pth")
    adapter.trace_plan = {
        "config_path": adapter.config_path,
        "ckpt_path": adapter.ckpt_path,
    }
    hardware = hardware_capability()
    hardware.path = str(tmp_path / "private" / "hardware.yaml")
    manifest = scan(
        adapter,
        hardware,
        device="cpu",
        profile_latency_mode="off",
        scenario=scenario,
        loaded_config=loaded_config,
        checkpoint_evidence=checkpoint,
    )
    path = tmp_path / "toy-scenario.yaml"
    import yaml

    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    assert manifest["scan_scenario"]["backend_precisions"] == ["FP16"]
    assert manifest["scanner_structural_axes"]
    assert len(manifest["scanner_structural_axes_digest"]) == 64
    assert manifest["scanner_structural_axes"][0]["provenance"]["input_digest"] == (
        manifest["formal_scan"]["structural_axis_inputs_digest"]
    )
    assert manifest["scanner_structural_axes_digest"] == (
        scanner_structural_axes_digest(manifest["scanner_structural_axes"])
    )
    assert "structural_axis_inputs" not in manifest
    assert _absolute_manifest_strings(manifest) == []
    assert manifest["view_latency"]["status"] == "skipped"
    assert "full_detail_sidecar" not in manifest["view_latency"]
    assert _manifest_contains(manifest["scanner_structural_axes"], "model.backbone.width")
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_scanner_reuses_one_resolved_checkpoint_for_load_and_digest(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  backbone:\n    width: 32\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": FullModel().state_dict()}, checkpoint)
    adapter = _ScannerOwnedEvidenceAdapter(config, checkpoint, tmp_path)
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"all": ("FP16",)},
        alignment={
            "default_round_to": 16,
            "default_hardware_alignment": 16,
            "default_max_rate_numerator": 50,
            "default_max_rate_denominator": 100,
        },
    )

    manifest = scan(
        adapter,
        hardware_capability(),
        device="cpu",
        profile_latency_mode="off",
        scenario=scenario,
    )

    assert manifest["scanner_structural_axes"]
    assert adapter.formal_build_calls == 3
    assert adapter.native_load_calls == 1
    assert len(set(adapter.formal_config_identities)) == 1
    assert len({id(item) for item in adapter.loaded_config_authorities}) == 1
    config_authority = adapter.loaded_config_authorities[0]
    assert isinstance(config_authority, FormalConfigSnapshot)
    assert adapter.resolve_calls == 1
    assert adapter.loaded_checkpoint_authorities
    assert len({id(item) for item in adapter.loaded_checkpoint_authorities}) == 1
    authority = adapter.loaded_checkpoint_authorities[0]
    assert isinstance(authority, FormalCheckpointSnapshot)
    with pytest.raises(ValueError, match="snapshot is unavailable"):
        _ = authority.loader_path
    with pytest.raises(ValueError, match="snapshot is unavailable"):
        _ = config_authority.loader_path

    config.write_text(
        "model:\n  backbone:\n    width: 32\n# byte-provenance drift\n",
        encoding="utf-8",
    )
    drifted_manifest = scan(
        _ScannerOwnedEvidenceAdapter(config, checkpoint, tmp_path),
        hardware_capability(),
        device="cpu",
        profile_latency_mode="off",
        scenario=scenario,
    )
    original_provenance = manifest["scanner_structural_axes"][0]["provenance"]
    drifted_provenance = drifted_manifest["scanner_structural_axes"][0]["provenance"]
    assert original_provenance["config_digest"] == drifted_provenance[
        "config_digest"
    ]
    assert original_provenance["scanner_input_digest"] != drifted_provenance[
        "scanner_input_digest"
    ]
    assert manifest["scanner_structural_axes_digest"] != drifted_manifest[
        "scanner_structural_axes_digest"
    ]


@pytest.mark.parametrize("fail_on_build", [None, 2])
def test_formal_scan_keeps_model_b_root_for_native_load_and_every_build(
    tmp_path: Path, fail_on_build: int | None
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  backbone:\n    width: 32\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": FullModel().state_dict()}, checkpoint)
    root_a = _identified_native_root(tmp_path / "root-a", "A")
    root_b = _identified_native_root(tmp_path / "root-b", "B")
    build_cwd = tmp_path / "fcooper-build-cwd"
    build_cwd.mkdir()
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "opencood" or name.startswith("opencood.")
    }
    _restore_opencood_modules({})
    sys.path.insert(0, str(root_a))
    importlib.import_module("opencood.model_identity")
    sys.path[:] = original_path
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "opencood" or name.startswith("opencood.")
    }
    adapter = _CrossRootFormalAdapter(
        config,
        checkpoint,
        tmp_path,
        import_root=root_b,
        build_cwd=build_cwd,
        fail_on_build=fail_on_build,
    )
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"all": ("FP16",)},
        alignment={
            "default_round_to": 16,
            "default_hardware_alignment": 16,
            "default_max_rate_numerator": 50,
            "default_max_rate_denominator": 100,
        },
    )

    try:
        if fail_on_build is None:
            manifest = scan(
                adapter,
                hardware_capability(),
                device="cpu",
                profile_latency_mode="off",
                scenario=scenario,
            )
            assert manifest["scanner_structural_axes"]
            expected_builds = 3
        else:
            with pytest.raises(RuntimeError, match="formal model build failed"):
                scan(
                    adapter,
                    hardware_capability(),
                    device="cpu",
                    profile_latency_mode="off",
                    scenario=scenario,
                )
            expected_builds = fail_on_build
        assert adapter.build_roots == ["B"] * expected_builds
        assert adapter.config_roots == ["B"] * expected_builds
        assert Path.cwd() == original_cwd
        assert sys.path == original_path
        assert {
            name: module
            for name, module in sys.modules.items()
            if name == "opencood" or name.startswith("opencood.")
        } == before_modules
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path
        _restore_opencood_modules(original_modules)


def test_formal_import_transaction_cleans_all_state_when_cwd_is_unrestorable(
    tmp_path: Path,
) -> None:
    safe_cwd = Path.cwd()
    doomed_cwd = tmp_path / "private-doomed-cwd"
    doomed_cwd.mkdir()
    root_a = _identified_native_root(tmp_path / "root-a", "A")
    root_b = _identified_native_root(tmp_path / "root-b", "B")
    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "opencood" or name.startswith("opencood.")
    }
    _restore_opencood_modules({})
    sys.path.insert(0, str(root_a))
    importlib.import_module("opencood.model_identity")
    sys.path[:] = original_path
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "opencood" or name.startswith("opencood.")
    }

    class _Adapter:
        model_config_import_root = root_b

    caught: Exception | None = None
    os.chdir(doomed_cwd)
    try:
        try:
            with formal_trace_build.formal_model_import_transaction(_Adapter()):
                importlib.import_module("opencood.model_identity")
                os.chdir(safe_cwd)
                doomed_cwd.rmdir()
        except Exception as error:  # noqa: BLE001 - assert stable public boundary.
            caught = error

        after_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "opencood" or name.startswith("opencood.")
        }
        assert (
            type(caught),
            str(caught),
            str(tmp_path) in str(caught),
            isinstance(caught.__cause__, FileNotFoundError),
            sys.path,
            after_modules,
            formal_trace_build._ACTIVE_ROOT.get(),
        ) == (
            ValueError,
            "formal model import state restoration failed",
            False,
            True,
            original_path,
            before_modules,
            None,
        )
    finally:
        os.chdir(safe_cwd)
        sys.path[:] = original_path
        _restore_opencood_modules(original_modules)


@pytest.mark.parametrize("change_kind", ["replace", "in_place"])
def test_formal_scan_rejects_checkpoint_change_after_first_model_load(
    tmp_path: Path,
    change_kind: str,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  backbone:\n    width: 32\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    replacement = tmp_path / "replacement.pth"
    torch.save({"model_state_dict": FullModel().state_dict()}, checkpoint)
    torch.save({"model_state_dict": FullModel().state_dict()}, replacement)
    adapter = _ChangingCheckpointAdapter(
        config,
        checkpoint,
        tmp_path,
        replacement_path=replacement,
        change_kind=change_kind,
    )
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"all": ("FP16",)},
        alignment={
            "default_round_to": 16,
            "default_hardware_alignment": 16,
            "default_max_rate_numerator": 50,
            "default_max_rate_denominator": 100,
        },
    )

    with pytest.raises(ValueError, match="checkpoint authority changed"):
        scan(
            adapter,
            hardware_capability(),
            device="cpu",
            profile_latency_mode="off",
            scenario=scenario,
        )


def test_adapter_helpers_and_graph_shape_modes_are_explicit() -> None:
    """Small helpers keep naming buckets and CPU profiling policy deterministic."""
    adapter = ToyAdapter()

    assert _first_conv_in_channels(FullModel()) == 32
    assert _generic_bucket("shrinker.block") == "neck"
    assert _generic_bucket("cls_head") == "heads"
    assert adapter.typed_skipped_subgraphs()[0]["type"] == "attention_or_routing_fusion"
    assert _resolve_profile("auto", "cpu") is False
    assert _resolve_profile("on", "cpu") is True
    assert _output_shapes({"one": torch.ones(1, 2), "two": (torch.ones(1),)}) == [[1, 2], [1]]
    with pytest.raises(KeyError, match="not-a-model"):
        get_adapter("not-a-model")
