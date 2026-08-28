from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from types import MappingProxyType
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from framework.stage1.adapters import (
    MaterializerParameterSource,
    PyramidLidarAdapter,
    ScanScenario,
)
from framework.stage1.formal_config_evidence import canonicalize_model_config
from framework.stage1.formal_scan_evidence import (
    load_formal_scan_evidence,
    load_scan_scenario,
    resolve_formal_checkpoint_path,
    validate_scenario_hardware,
)
from framework.stage1.formal_checkpoint_snapshot import (
    scanner_owned_checkpoint_snapshot,
)
from framework.stage1.formal_config_snapshot import scanner_owned_config_snapshot
from framework.stage1 import formal_trace_build
from framework.stage1.trace_plan import GeneratedTraceWrapper
from framework.stage1.structural_axis_digest import canonical_digest


class _Adapter:
    def __init__(
        self, config_path: Path, checkpoint_path: Path, trusted_root: Path
    ) -> None:
        self.config_path = str(config_path)
        self.ckpt_path = str(checkpoint_path)
        self.formal_config_root = trusted_root
        self.formal_checkpoint_root = trusted_root
        self.formal_scenario_root = trusted_root

    def materializer_parameter_sources(self, loaded_config):
        assert loaded_config["model"]["width"] == 12
        return (
            MaterializerParameterSource(
                axis_id="backbone.s0",
                config_selector="model.width",
                mutation_kind="out_channels",
                module_root_selector="0",
                allowed_roles=("output",),
                provenance={"adapter": "test"},
            ),
        )


def _native_loader_root(
    root: Path, *, raises: bool = False, invocation_marker: Path | None = None
) -> Path:
    package = root / "opencood" / "hypes_yaml"
    package.mkdir(parents=True)
    (root / "opencood" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    marker = (
        ""
        if invocation_marker is None
        else (
            "from pathlib import Path\n"
            f"Path({str(invocation_marker)!r}).write_text('called')\n"
        )
    )
    body = marker + (
        "def load_yaml(_path):\n    raise RuntimeError('native failure')\n"
        if raises
        else (
            "import yaml\n"
            "def load_yaml(path):\n"
            "    with open(path, encoding='utf-8') as stream:\n"
            "        return yaml.load(stream, Loader=yaml.Loader)\n"
        )
    )
    (package / "yaml_utils.py").write_text(body, encoding="utf-8")
    return root


def _opencood_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "opencood" or name.startswith("opencood.")
    }


def _load_evidence(adapter: _Adapter, net: nn.Module):
    config_path = Path(adapter.config_path)
    loaded_model_config = (
        yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
        if config_path.is_file()
        else {"model": {"width": 12}}
    )
    with (
        scanner_owned_checkpoint_snapshot(adapter) as authority,
        scanner_owned_config_snapshot(adapter) as config_authority,
    ):
        return load_formal_scan_evidence(
            adapter,
            net,
            checkpoint_authority=authority,
            config_authority=config_authority,
            loaded_model_config=loaded_model_config,
        )


def _scenario(path: Path) -> Path:
    path.write_text(
        """hardware_precisions: [FP16, INT8]
backend_precisions: [FP16, INT8]
compression_modes: [fp16, int8]
graph_quant_unit_policy:
  backbone: [FP16, INT8]
alignment:
  default_round_to: 4
""",
        encoding="utf-8",
    )
    return path


def test_scenario_loader_returns_explicit_scan_scenario(tmp_path: Path) -> None:
    scenario = load_scan_scenario(
        _scenario(tmp_path / "scenario.yaml"), trusted_root=tmp_path
    )

    assert isinstance(scenario, ScanScenario)
    assert scenario.backend_precisions == ("FP16", "INT8")
    assert scenario.alignment["default_round_to"] == 4


def test_tracked_p6_scenario_is_count_free_and_exact_schema() -> None:
    path = Path("configs/stage1/p6_h800_formal_scan.yaml").resolve()
    scenario = load_scan_scenario(path, trusted_root=path.parent)
    tokens = path.read_text(encoding="utf-8").replace(":", " ").split()

    assert isinstance(scenario, ScanScenario)
    assert not {"3", "5", "7", "343", "686"}.intersection(tokens)


def test_formal_evidence_uses_real_checkpoint_digest_and_trace_module_widths(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    net = nn.Sequential(nn.Conv2d(3, 12, 1))
    torch.save({"model_state_dict": net.state_dict()}, checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()

    loaded_config, evidence = _load_evidence(
        _Adapter(config, checkpoint, tmp_path), net
    )

    assert loaded_config == {"model": {"width": 12}}
    assert evidence == {
        "digest": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "config_file_digest": hashlib.sha256(config.read_bytes()).hexdigest(),
        "module_widths": {"0": 12},
    }


def test_formal_evidence_accepts_model_native_pyramid_config_tags(
    tmp_path: Path,
) -> None:
    fixture = Path("tests/stage1/fixtures/pyramid_native_tags_config.yaml").resolve()
    config = tmp_path / "config.yaml"
    config.write_bytes(fixture.read_bytes())
    checkpoint = tmp_path / "checkpoint.pth"
    net = nn.Sequential(nn.Conv2d(3, 12, 1))
    torch.save({"model_state_dict": net.state_dict()}, checkpoint)
    native = yaml.load(config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    assert isinstance(native["noise_setting"], OrderedDict)
    assert isinstance(
        native["model"]["args"]["m1"]["encoder_args"]
        ["point_pillar_scatter"]["grid_size"],
        np.ndarray,
    )
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load(config.read_text(encoding="utf-8"))

    loaded_config, evidence = _load_evidence(
        _Adapter(config, checkpoint, tmp_path), net
    )

    assert loaded_config["model"]["args"]["fusion_backbone"]["num_filters"] == (
        64,
        128,
        256,
    )
    assert loaded_config["model"]["args"]["m1"]["encoder_args"][
        "point_pillar_scatter"
    ]["grid_size"] == (512, 256, 1)
    assert loaded_config["noise_setting"] == {"add_noise": False}
    assert evidence["config_file_digest"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert len(canonical_digest(loaded_config)) == 64
    assert tuple(
        source.axis_id
        for source in PyramidLidarAdapter().materializer_parameter_sources(
            loaded_config
        )
    ) == ("backbone.s0", "backbone.s1", "backbone.s2", "neck.output")
    with pytest.raises(TypeError):
        loaded_config["model"] = {}  # type: ignore[index]


def test_formal_trace_uses_model_native_loader_on_scanner_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path("tests/stage1/fixtures/pyramid_native_tags_config.yaml").resolve()
    config = tmp_path / "config.yaml"
    config.write_bytes(fixture.read_bytes())
    adapter = _Adapter(config, tmp_path / "unused.pth", tmp_path)
    adapter.model_config_import_root = _native_loader_root(tmp_path / "native")

    with scanner_owned_config_snapshot(adapter) as authority:
        native = formal_trace_build._load_model_native_config(adapter, authority)

    assert isinstance(native["noise_setting"], OrderedDict)
    assert isinstance(
        native["model"]["args"]["m1"]["encoder_args"]
        ["point_pillar_scatter"]["grid_size"],
        np.ndarray,
    )


@pytest.mark.parametrize("tag_kind", ["object_apply", "python_name"])
def test_formal_trace_rejects_unknown_python_tag_before_native_side_effect(
    tmp_path: Path, tag_kind: str,
) -> None:
    marker = tmp_path / "compromised"
    native_marker = tmp_path / "native-called"
    config = tmp_path / "config.yaml"
    payload = (
        f"payload: !!python/object/apply:os.system ['touch {marker}']\n"
        if tag_kind == "object_apply"
        else "payload: !!python/name:builtins.eval ''\n"
    )
    config.write_text(payload, encoding="utf-8")
    adapter = _Adapter(config, tmp_path / "unused.pth", tmp_path)
    adapter.model_config_import_root = _native_loader_root(
        tmp_path / "native", invocation_marker=native_marker
    )

    with scanner_owned_config_snapshot(adapter) as authority:
        with pytest.raises(ValueError, match="^formal model config load is invalid$"):
            formal_trace_build._load_model_native_config(adapter, authority)

    assert not native_marker.exists()
    assert not marker.exists()


@pytest.mark.parametrize("native_failure", [False, True])
def test_formal_trace_restores_exact_import_state_after_native_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_failure: bool,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    adapter = _Adapter(config, tmp_path / "unused.pth", tmp_path)
    adapter.model_config_import_root = _native_loader_root(
        tmp_path / "native", raises=native_failure
    )
    sentinel = ModuleType("opencood")
    prior_child = ModuleType("opencood.prior")
    monkeypatch.setitem(sys.modules, "opencood", sentinel)
    monkeypatch.setitem(sys.modules, "opencood.prior", prior_child)
    before_path = list(sys.path)
    before_modules = _opencood_modules()

    try:
        with scanner_owned_config_snapshot(adapter) as authority:
            if native_failure:
                with pytest.raises(
                    ValueError, match="^formal model config load is invalid$"
                ):
                    formal_trace_build._load_model_native_config(adapter, authority)
            else:
                native = formal_trace_build._load_model_native_config(
                    adapter, authority
                )
                assert native == {"model": {"width": 12}}
        assert sys.path == before_path
        assert _opencood_modules() == before_modules
        assert sys.modules["opencood"] is sentinel
        assert sys.modules["opencood.prior"] is prior_child
    finally:
        sys.path[:] = before_path
        for name in tuple(_opencood_modules()):
            sys.modules.pop(name, None)
        sys.modules.update(before_modules)


def test_model_native_config_canonicalization_is_json_safe_and_immutable() -> None:
    native = OrderedDict(
        [
            ("z_array", np.array([3, 2, 1], dtype=np.int64)),
            ("a_scalar", np.float32(1.25)),
            ("nested", OrderedDict([("enabled", np.bool_(True))])),
        ]
    )

    canonical = canonicalize_model_config(native)

    assert isinstance(canonical, MappingProxyType)
    assert tuple(canonical) == ("a_scalar", "nested", "z_array")
    assert canonical["a_scalar"] == 1.25
    assert canonical["nested"] == {"enabled": True}
    assert canonical["z_array"] == (3, 2, 1)
    assert len(canonical_digest(canonical)) == 64
    with pytest.raises(TypeError):
        canonical["a_scalar"] = 2.5  # type: ignore[index]


@pytest.mark.parametrize(
    "native",
    [
        {"unknown": object()},
        {"non_finite": float("nan")},
        {"non_finite": np.float64("inf")},
        {1: "non-string-key"},
    ],
)
def test_model_native_config_canonicalization_rejects_unknown_values(
    native: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="model config"):
        canonicalize_model_config(native)


def test_formal_evidence_covers_internal_and_transposed_named_modules(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    net = nn.Sequential(
        nn.Conv2d(3, 12, 1),
        nn.Sequential(nn.Conv2d(12, 24, 1), nn.ConvTranspose2d(24, 16, 2)),
    )
    torch.save({"model_state_dict": net.state_dict()}, checkpoint)

    _, evidence = _load_evidence(_Adapter(config, checkpoint, tmp_path), net)

    assert evidence["module_widths"] == {"0": 12, "1.0": 24, "1.1": 16}


def test_fcooper_generated_wrapper_maps_alias_to_source_checkpoint_key(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"

    class _FullModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_m1 = nn.Module()
            self.backbone_m1.blocks = nn.Sequential(nn.Conv2d(3, 12, 1))

    full_model = _FullModel()
    torch.save({"model_state_dict": full_model.state_dict()}, checkpoint)
    wrapper = GeneratedTraceWrapper(
        full_model,
        {
            "included_modules": ["backbone_m1.blocks"],
            "ignored_layers": [],
            "wrapper_kind": "heter_baseline_dense_path",
        },
    )

    _, evidence = _load_evidence(
        _Adapter(config, checkpoint, tmp_path), wrapper
    )

    assert wrapper.checkpoint_module_key_map == {
        "body.backbone_m1__blocks": "backbone_m1.blocks",
        "body.backbone_m1__blocks.0": "backbone_m1.blocks.0",
    }
    copied_wrapper = copy.deepcopy(wrapper)
    assert copied_wrapper.checkpoint_module_key_map == (
        wrapper.checkpoint_module_key_map
    )
    with pytest.raises(TypeError):
        wrapper.checkpoint_module_key_map["forged"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        copied_wrapper.checkpoint_module_key_map["forged"] = "forged"  # type: ignore[index]
    assert evidence["module_widths"] == {"body.backbone_m1__blocks.0": 12}


@pytest.mark.parametrize(
    "checkpoint_module_key_map",
    [
        MappingProxyType({}),
        MappingProxyType({"0": "source", "1": "source"}),
        MappingProxyType({"0": "0", "1": "1", "ghost": "ghost"}),
    ],
    ids=["missing", "duplicate", "unknown"],
)
def test_formal_evidence_rejects_invalid_explicit_checkpoint_module_mapping(
    tmp_path: Path,
    checkpoint_module_key_map: MappingProxyType,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    net = nn.Sequential(nn.Conv2d(3, 12, 1), nn.Conv2d(12, 12, 1))
    net.checkpoint_module_key_map = checkpoint_module_key_map  # type: ignore[attr-defined]
    torch.save({"model_state_dict": net.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="checkpoint module mapping"):
        _load_evidence(_Adapter(config, checkpoint, tmp_path), net)


def test_formal_evidence_rejects_empty_or_missing_selected_checkpoint_tensor(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="selected checkpoint tensor"):
        _load_evidence(
            _Adapter(config, checkpoint, tmp_path),
            nn.Sequential(nn.Conv2d(3, 12, 1)),
        )


def test_formal_evidence_rejects_shape_incompatible_selected_tensor(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": {"0.weight": torch.ones(11, 3, 1, 1)}}, checkpoint)

    with pytest.raises(ValueError, match="selected checkpoint tensor shape"):
        _load_evidence(
            _Adapter(config, checkpoint, tmp_path),
            nn.Sequential(nn.Conv2d(3, 12, 1)),
        )


def test_scenario_hardware_rejects_precision_not_supported_by_capability(
    tmp_path: Path,
) -> None:
    scenario = load_scan_scenario(
        _scenario(tmp_path / "scenario.yaml"), trusted_root=tmp_path
    )

    class _Hardware:
        legal_bits = ["FP16"]

    with pytest.raises(ValueError, match="hardware precisions"):
        validate_scenario_hardware(scenario, _Hardware())


@pytest.mark.parametrize(
    "unsafe_kind", ["symlink_file", "symlink_directory", "outside_root"]
)
def test_scenario_loader_rejects_untrusted_paths_without_path_leak(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    trusted_root = tmp_path / "trusted"
    outside_root = tmp_path / "outside"
    trusted_root.mkdir()
    outside_root.mkdir()
    outside_scenario = _scenario(outside_root / "scenario.yaml")
    if unsafe_kind == "symlink_file":
        scenario_path = trusted_root / "scenario.yaml"
        scenario_path.symlink_to(outside_scenario)
    elif unsafe_kind == "symlink_directory":
        linked_directory = trusted_root / "linked"
        linked_directory.symlink_to(outside_root, target_is_directory=True)
        scenario_path = linked_directory / "scenario.yaml"
    else:
        scenario_path = outside_scenario

    with pytest.raises(ValueError) as caught:
        load_scan_scenario(scenario_path, trusted_root=trusted_root)

    assert str(caught.value) == "formal scan scenario is invalid"
    assert str(tmp_path) not in str(caught.value)


def test_formal_evidence_rejects_symlinked_config_under_declared_root(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    real_config = trusted_root / "real-config.yaml"
    real_config.write_text("model:\n  width: 12\n", encoding="utf-8")
    linked_config = trusted_root / "config.yaml"
    linked_config.symlink_to(real_config)
    checkpoint = trusted_root / "checkpoint.pth"
    net = nn.Sequential(nn.Conv2d(3, 12, 1))
    torch.save({"model_state_dict": net.state_dict()}, checkpoint)
    adapter = _Adapter(linked_config, checkpoint, trusted_root)

    with pytest.raises(ValueError) as caught:
        _load_evidence(adapter, net)

    assert str(caught.value) == "formal config authority is invalid"
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("unsafe_kind", ["symlink_directory", "outside_root"])
def test_checkpoint_resolver_rejects_path_outside_declared_root(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    trusted_root = tmp_path / "trusted"
    outside_root = tmp_path / "outside"
    trusted_root.mkdir()
    outside_root.mkdir()
    checkpoint = outside_root / "checkpoint.pth"
    torch.save({"model_state_dict": {}}, checkpoint)
    if unsafe_kind == "symlink_directory":
        linked_directory = trusted_root / "linked"
        linked_directory.symlink_to(outside_root, target_is_directory=True)
        selected_checkpoint = linked_directory / "checkpoint.pth"
    else:
        selected_checkpoint = checkpoint
    config = trusted_root / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    adapter = _Adapter(config, selected_checkpoint, trusted_root)

    with pytest.raises(ValueError) as caught:
        resolve_formal_checkpoint_path(adapter)

    assert str(caught.value) == "formal scan checkpoint identity is invalid"
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("config_path", "formal config authority"),
        ("ckpt_path", "formal checkpoint authority"),
    ],
)
def test_formal_evidence_rejects_missing_real_input(
    tmp_path: Path,
    field: str,
    error: str,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": nn.Sequential(nn.Conv2d(3, 12, 1)).state_dict()}, checkpoint)
    adapter = _Adapter(config, checkpoint, tmp_path)
    setattr(adapter, field, str(tmp_path / "missing"))

    with pytest.raises(ValueError, match=error):
        _load_evidence(adapter, nn.Sequential(nn.Conv2d(3, 12, 1)))
