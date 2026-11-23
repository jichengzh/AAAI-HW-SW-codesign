from __future__ import print_function

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import yaml


class HilArtifactError(ValueError):
    """Raised when the frozen HIL artifact contract is inconsistent."""


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class HilArtifactSpec:
    arm_id: str
    model: str
    widths: Tuple[int, int, int]
    precision: str
    runtime: str
    config: Path
    checkpoint: Path
    engine: Optional[Path]
    artifact_sha256: Mapping[str, str]
    input_contract: Tuple[TensorContract, ...]
    output_contract: Tuple[TensorContract, ...]


HIL_ARM_IDS = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "joint_shcosearch",
)

DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent_config"
    / "coptv2x_hil"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "/home/jichengzhi/V2X/results/lane_c_orin_stage6_five_config_20260724"
)

_DOCUMENT_FIELDS = frozenset(
    (
        "arm_id",
        "model",
        "widths",
        "precision",
        "runtime",
        "config",
        "checkpoint",
        "engine",
        "artifact_sha256",
        "input_contract",
        "output_contract",
    )
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ARM_RULES = {
    "original_default": {
        "model": "codriving",
        "widths": (64, 128, 256),
        "precision": "fp32",
        "runtime": "native_pytorch",
        "config": "source/codriving/064x128x256/config.yaml",
        "checkpoint": "source/codriving/064x128x256/net_epoch29.pth",
        "engine": None,
    },
    "compression_only": {
        "model": "codriving",
        "widths": (32, 64, 96),
        "precision": "fp16",
        "runtime": "tensorrt",
        "config": "source/codriving/032x064x096/config.yaml",
        "checkpoint": "source/codriving/032x064x096/net_epoch_bestval_at11.pth",
        "engine": "codriving/compression_only/backbone.engine",
    },
    "schedule_only": {
        "model": "codriving",
        "widths": (64, 128, 256),
        "precision": "fp32",
        "runtime": "tensorrt",
        "config": "source/codriving/064x128x256/config.yaml",
        "checkpoint": "source/codriving/064x128x256/net_epoch29.pth",
        "engine": "codriving/schedule_only/backbone.engine",
    },
    "compress_then_tune": {
        "model": "codriving",
        "widths": (32, 80, 128),
        "precision": "fp16",
        "runtime": "tensorrt",
        "config": "source/codriving/032x080x128/config.yaml",
        "checkpoint": "source/codriving/032x080x128/net_epoch_bestval_at5.pth",
        "engine": "codriving/compress_then_tune/backbone.engine",
    },
    "joint_shcosearch": {
        "model": "codriving",
        "widths": (16, 32, 64),
        "precision": "fp16",
        "runtime": "tensorrt",
        "config": "source/codriving/016x032x064/config.yaml",
        "checkpoint": "source/codriving/016x032x064/net_epoch_bestval_at9.pth",
        "engine": "codriving/joint_shcosearch_fp16/backbone.engine",
    },
}


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise HilArtifactError("%s must be a mapping" % label)
    return value


def _relative_path(value, label):
    if not isinstance(value, str) or not value:
        raise HilArtifactError("%s must be a non-empty relative path" % label)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise HilArtifactError("%s must stay below artifact root" % label)
    return path


def _tensor_contracts(raw_contract, label):
    contract = _require_mapping(raw_contract, label)
    if not contract:
        raise HilArtifactError("%s must not be empty" % label)
    tensors = []
    for name, raw_tensor in contract.items():
        if not isinstance(name, str) or not name:
            raise HilArtifactError("%s tensor names must be non-empty strings" % label)
        tensor = _require_mapping(raw_tensor, "%s.%s" % (label, name))
        if set(tensor) != {"shape", "dtype"}:
            raise HilArtifactError(
                "%s.%s requires exactly shape and dtype" % (label, name)
            )
        raw_shape = tensor["shape"]
        if (
            not isinstance(raw_shape, list)
            or not raw_shape
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in raw_shape
            )
        ):
            raise HilArtifactError("%s.%s shape is invalid" % (label, name))
        dtype = tensor["dtype"]
        if dtype not in ("float16", "float32"):
            raise HilArtifactError("%s.%s dtype is invalid" % (label, name))
        tensors.append(TensorContract(name, tuple(raw_shape), dtype))
    return tuple(tensors)


def _hash_contract(raw_hashes, runtime):
    hashes = _require_mapping(raw_hashes, "artifact_sha256")
    expected_keys = {"config", "checkpoint"}
    if runtime == "tensorrt":
        expected_keys.add("engine")
    if set(hashes) != expected_keys:
        raise HilArtifactError(
            "artifact_sha256 keys must be %s" % sorted(expected_keys)
        )
    normalized = {}
    for name in sorted(hashes):
        value = hashes[name]
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value.lower()):
            raise HilArtifactError("artifact_sha256.%s must be SHA-256" % name)
        normalized[name] = value.lower()
    return MappingProxyType(normalized)


def _validate_frozen_contract(spec):
    rule = _ARM_RULES[spec.arm_id]
    actual = {
        "model": spec.model,
        "widths": spec.widths,
        "precision": spec.precision,
        "runtime": spec.runtime,
        "config": str(spec.config),
        "checkpoint": str(spec.checkpoint),
        "engine": None if spec.engine is None else str(spec.engine),
    }
    for name, expected in rule.items():
        if actual[name] != expected:
            detail = ""
            if spec.arm_id == "joint_shcosearch" and name == "engine":
                detail = "; required joint_shcosearch_fp16 engine"
            raise HilArtifactError(
                "%s %s must be %r, got %r%s"
                % (spec.arm_id, name, expected, actual[name], detail)
            )

    expected_input = (
        TensorContract("spatial_features", (2, 64, 256, 512), "float32"),
    )
    expected_output = tuple(
        TensorContract(name, shape, "float32")
        for name, shape in (
            ("res0", (2, spec.widths[0], 128, 256)),
            ("res1", (2, spec.widths[1], 64, 128)),
            ("res2", (2, spec.widths[2], 32, 64)),
        )
    )
    if spec.input_contract != expected_input:
        raise HilArtifactError("%s input_contract is not frozen" % spec.arm_id)
    if spec.output_contract != expected_output:
        raise HilArtifactError("%s output_contract is not frozen" % spec.arm_id)


def _load_spec(path, expected_arm):
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise HilArtifactError("cannot read %s: %s" % (path, error)) from error
    document = _require_mapping(raw, path.name)
    if set(document) != _DOCUMENT_FIELDS:
        missing = sorted(_DOCUMENT_FIELDS - set(document))
        unknown = sorted(set(document) - _DOCUMENT_FIELDS, key=repr)
        raise HilArtifactError(
            "%s fields mismatch; missing=%s unknown=%s"
            % (path.name, missing, unknown)
        )
    if document["arm_id"] != expected_arm:
        raise HilArtifactError(
            "%s arm_id must be %s" % (path.name, expected_arm)
        )
    widths = document["widths"]
    if (
        not isinstance(widths, list)
        or len(widths) != 3
        or any(type(width) is not int or width <= 0 for width in widths)
    ):
        raise HilArtifactError(
            "%s widths must contain three positive integers" % expected_arm
        )
    runtime = document["runtime"]
    engine = document["engine"]
    if engine is not None:
        engine = _relative_path(engine, "%s.engine" % expected_arm)
    spec = HilArtifactSpec(
        arm_id=expected_arm,
        model=document["model"],
        widths=tuple(widths),
        precision=document["precision"],
        runtime=runtime,
        config=_relative_path(document["config"], "%s.config" % expected_arm),
        checkpoint=_relative_path(
            document["checkpoint"], "%s.checkpoint" % expected_arm
        ),
        engine=engine,
        artifact_sha256=_hash_contract(document["artifact_sha256"], runtime),
        input_contract=_tensor_contracts(
            document["input_contract"], "%s.input_contract" % expected_arm
        ),
        output_contract=_tensor_contracts(
            document["output_contract"], "%s.output_contract" % expected_arm
        ),
    )
    _validate_frozen_contract(spec)
    return spec


def load_hil_artifact_registry(config_dir=DEFAULT_CONFIG_DIR):
    config_dir = Path(config_dir).expanduser().resolve()
    if not config_dir.is_dir():
        raise HilArtifactError("artifact config directory is missing: %s" % config_dir)
    expected_names = {"artifact_%s.yaml" % arm_id for arm_id in HIL_ARM_IDS}
    actual_names = {
        path.name for path in config_dir.glob("artifact_*.yaml") if path.is_file()
    }
    if actual_names != expected_names:
        raise HilArtifactError(
            "artifact YAML set mismatch; missing=%s unknown=%s"
            % (
                sorted(expected_names - actual_names),
                sorted(actual_names - expected_names),
            )
        )
    registry = {
        arm_id: _load_spec(config_dir / ("artifact_%s.yaml" % arm_id), arm_id)
        for arm_id in HIL_ARM_IDS
    }
    return MappingProxyType(registry)


def get_hil_artifact(arm_id, registry=None):
    if arm_id not in HIL_ARM_IDS:
        raise HilArtifactError(
            "unknown HIL arm %r; expected one of %s" % (arm_id, HIL_ARM_IDS)
        )
    if registry is None:
        registry = load_hil_artifact_registry()
    try:
        return registry[arm_id]
    except KeyError as error:
        raise HilArtifactError("registry is missing HIL arm %s" % arm_id) from error


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root, relative_path, arm_id, artifact_name):
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise HilArtifactError(
            "%s %s escapes artifact root" % (arm_id, artifact_name)
        ) from error
    if not path.is_file():
        raise HilArtifactError(
            "%s missing %s: %s" % (arm_id, artifact_name, path)
        )
    return path


def _contract_records(contract):
    return [
        {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}
        for tensor in contract
    ]


def preflight_hil_artifacts(
    config_dir=DEFAULT_CONFIG_DIR, artifact_root=DEFAULT_ARTIFACT_ROOT
):
    registry = load_hil_artifact_registry(config_dir)
    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise HilArtifactError("artifact root is missing: %s" % root)
    records = []
    for arm_id in HIL_ARM_IDS:
        spec = registry[arm_id]
        paths = {"config": spec.config, "checkpoint": spec.checkpoint}
        if spec.engine is not None:
            paths["engine"] = spec.engine
        verified_hashes = {}
        for artifact_name in sorted(paths):
            path = _artifact_path(root, paths[artifact_name], arm_id, artifact_name)
            actual_sha256 = _sha256(path)
            expected_sha256 = spec.artifact_sha256[artifact_name]
            if actual_sha256 != expected_sha256:
                raise HilArtifactError(
                    "%s %s SHA-256 mismatch: expected %s, got %s"
                    % (arm_id, artifact_name, expected_sha256, actual_sha256)
                )
            verified_hashes[artifact_name] = actual_sha256
        records.append(
            {
                "arm_id": arm_id,
                "artifact_id": "%s:%s" % (spec.model, arm_id),
                "runtime": spec.runtime,
                "precision": spec.precision,
                "artifact_sha256": verified_hashes,
                "input_contract": _contract_records(spec.input_contract),
                "output_contract": _contract_records(spec.output_contract),
            }
        )
    return tuple(records)


__all__ = (
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_CONFIG_DIR",
    "HIL_ARM_IDS",
    "HilArtifactError",
    "HilArtifactSpec",
    "TensorContract",
    "get_hil_artifact",
    "load_hil_artifact_registry",
    "preflight_hil_artifacts",
)
