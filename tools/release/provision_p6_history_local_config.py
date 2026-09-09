"""Run the legacy H800-only, non-materializing diagnostic provision check.

RTX provisioning is supported only by provision_p6_full_chain_local_config.py.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage1_bridge import load_stage2_search_space  # noqa: E402
from framework.stage6.p6_gpu_policy_v1 import canonical_gpu_indices  # noqa: E402
from framework.stage6.coptv2x_h800_search_v2 import (  # noqa: E402
    P6CoptV2XContractError,
    PublicP6CoptV2XContract,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_history_binding_v1 import (  # noqa: E402
    GpuRecord,
    P6HistoryBindingError,
    discover_history_binding,
    prevalidate_private_binding_pair_destinations,
)


PUBLIC_CONTRACT_PATH = REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
SOURCE_ADAPTER_PATH = REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"
MEASUREMENT_ADAPTER_PATH = REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"
EXPECTED_ASSET_LABELS = ("training-data", "model-init", "toolchain")
EXPECTED_LOCAL_INPUTS = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)
class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


class NvidiaSmiGpuProbe:
    """Read supplied private GPU admission fields through one direct command."""

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        records = self._query(_gpu_query_argv(indices))
        by_index = {record.index: record for record in records}
        return tuple(by_index[index] for index in indices if index in by_index)

    def snapshot_all(self) -> tuple[GpuRecord, ...]:
        """Return one immutable live snapshot for every visible NVIDIA GPU."""
        return self._query(_all_gpu_query_argv())

    @staticmethod
    def _query(argv: tuple[str, ...]) -> tuple[GpuRecord, ...]:
        completed = subprocess.run(
            argv,
            shell=False,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if completed.returncode != 0:
            raise RuntimeError("GPU query failed")
        return _parse_gpu_records(completed.stdout)


def _all_gpu_query_argv() -> tuple[str, ...]:
    return (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )


def _gpu_query_argv(indices: tuple[int, ...]) -> tuple[str, ...]:
    _validate_indices(indices)
    return (
        "nvidia-smi",
        "--id=" + ",".join(str(index) for index in indices),
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )


def _validate_indices(indices: tuple[int, ...]) -> None:
    if not isinstance(indices, tuple):
        raise ValueError("canonical GPU indices required")
    canonical_gpu_indices(indices)


def _parse_gpu_records(output: str) -> tuple[GpuRecord, ...]:
    records: list[GpuRecord] = []
    seen_indices: set[int] = set()
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 5:
            raise ValueError("invalid GPU query row")
        index_text, uuid, model_name, used_text, total_text = fields
        index = int(index_text)
        if index in seen_indices:
            raise ValueError("duplicate GPU index")
        seen_indices.add(index)
        used = float(used_text)
        total = float(total_text)
        if total <= 0.0:
            raise ValueError("invalid GPU memory total")
        records.append(
            GpuRecord(
                index=index,
                uuid=uuid,
                model_name=model_name,
                occupancy=used / total,
            )
        )
    return tuple(records)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--history-root", required=True, type=_absolute_path)
    parser.add_argument("--local-output-root", required=True, type=_absolute_path)
    parser.add_argument("--binding-output", required=True, type=_absolute_path)
    parser.add_argument("--config-output", required=True, type=_absolute_path)
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_private_output_root(path: Path) -> Path:
    if path.is_symlink():
        raise P6HistoryBindingError(
            "unsafe_destination", "local output root cannot be a symlink"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError(
            "unsafe_destination", "local output root does not exist"
        ) from error
    if not resolved.is_dir():
        raise P6HistoryBindingError(
            "unsafe_destination", "local output root is not a directory"
        )
    repository = REPOSITORY_ROOT.resolve(strict=True)
    if _is_relative_to(resolved, repository) and not _git_check_ignored(
        repository, resolved
    ):
        raise P6HistoryBindingError(
            "unsafe_destination", "repository output root is not git-ignored"
        )
    return resolved


def _git_check_ignored(repository: Path, path: Path) -> bool:
    relative = path.relative_to(repository)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _discover_stage2_search_space(binding: Mapping[str, Any]) -> Path:
    root = _binding_root(binding)
    candidates: list[Path] = []
    for raw_path in root.rglob("*.json"):
        if raw_path.is_symlink() or not raw_path.is_file():
            continue
        try:
            path = raw_path.resolve(strict=True)
        except OSError:
            continue
        if not _is_relative_to(path, root):
            continue
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_payload, Mapping):
                continue
            source_schema = raw_payload.get("schema")
            if not isinstance(source_schema, str) or not source_schema.startswith(
                "stage1_partition_manifest_"
            ):
                continue
            search_space = load_stage2_search_space(path)
        except (
            AttributeError,
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ):
            continue
        if search_space.get("schema") == "stage2_search_space_v1":
            candidates.append(path)
    if len(candidates) != 1:
        raise P6HistoryBindingError(
            "stage2_search_space_unavailable",
            "expected exactly one valid Stage2 search-space JSON",
        )
    return candidates[0]


def _binding_root(binding: Mapping[str, Any]) -> Path:
    value = binding.get("private_root")
    if not isinstance(value, str):
        raise P6HistoryBindingError("history_root", "binding root is invalid")
    try:
        root = Path(value).resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError("history_root", "binding root is unavailable") from error
    if not root.is_dir():
        raise P6HistoryBindingError("history_root", "binding root is not a directory")
    return root


def _private_anchor(value: object, root: Path, label: str) -> Path:
    if not isinstance(value, str):
        raise P6HistoryBindingError("history_binding", f"{label} anchor is invalid")
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError(
            "history_binding", f"{label} anchor is unavailable"
        ) from error
    if not _is_relative_to(path, root):
        raise P6HistoryBindingError(
            "history_binding", f"{label} anchor escapes history root"
        )
    return path


def _render_local_config(
    binding: Mapping[str, Any],
    contract: PublicP6CoptV2XContract,
    local_output_root: Path,
    binding_output: Path,
) -> dict[str, Any]:
    asset_labels = tuple(asset.label for asset in contract.assets)
    if asset_labels != EXPECTED_ASSET_LABELS:
        raise P6CoptV2XContractError("public asset labels are invalid")
    root = _binding_root(binding)
    raw_inputs = binding.get("local_input_paths")
    raw_components = binding.get("component_paths")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(EXPECTED_LOCAL_INPUTS):
        raise P6HistoryBindingError("history_binding", "local input mapping is invalid")
    if not isinstance(raw_components, Mapping):
        raise P6HistoryBindingError("history_binding", "component mapping is invalid")
    input_paths = {
        name: str(_private_anchor(raw_inputs[name], root, name))
        for name in EXPECTED_LOCAL_INPUTS
    }
    data_root = _private_anchor(
        raw_inputs["gold176_rows"], root, "training-data"
    ).parent
    model_init = _private_anchor(
        binding.get("source_registry_path"), root, "model-init"
    )
    toolchain = _private_anchor(
        raw_components.get("performance_plan"), root, "toolchain"
    )
    stage2_search_space = _discover_stage2_search_space(binding)
    python_executable = str(Path(sys.executable).absolute())
    binding_path = str(binding_output.absolute())
    return {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            "training-data": str(data_root),
            "model-init": str(model_init),
            "toolchain": str(toolchain),
        },
        "local_input_paths": input_paths,
        "candidate_source_mode": "framework_stage2_search_space",
        "stage2_search_space_path": str(stage2_search_space),
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                python_executable,
                str(SOURCE_ADAPTER_PATH),
                "--binding",
                binding_path,
                "--pyramid-candidate-plan",
                "{pyramid_candidate_plan}",
                "--source-registry-json",
                "{source_registry_json}",
                "--local-output-root",
                "{local_output_root}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                python_executable,
                str(MEASUREMENT_ADAPTER_PATH),
                "--binding",
                binding_path,
                "--measurement-request",
                "{measurement_request}",
                "--feedback-json",
                "{feedback_json}",
                "--round-output-root",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(local_output_root),
    }


def _validate_local_config(
    config: Mapping[str, Any],
    contract: PublicP6CoptV2XContract,
    local_output_root: Path,
) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        dir=local_output_root,
        prefix=".p6-history-local-config.",
        suffix=".json",
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=True, allow_nan=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        load_local_config(path, contract)
    finally:
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Provision the fixed private pair and return only stable status categories."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        _resolve_private_output_root(args.local_output_root)
        prevalidate_private_binding_pair_destinations(
            args.binding_output,
            args.config_output,
            REPOSITORY_ROOT,
        )
        contract = load_public_contract(PUBLIC_CONTRACT_PATH)
        if contract.hardware_profile.profile_id != "h800":
            raise P6HistoryBindingError(
                "legacy_h800_only",
                "RTX requires provision_p6_full_chain_local_config.py",
            )
        discover_history_binding(
            args.history_root,
            NvidiaSmiGpuProbe(),
            profile=contract.hardware_profile,
        )
        raise P6HistoryBindingError(
            "stage1_scan_unavailable",
            "legacy provisioning cannot supply the required Stage1 scan step",
        )
    except P6HistoryBindingError as error:
        sys.stderr.write(f"{error.category}\n")
        return 1
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2
    except (OSError, TypeError, ValueError):
        sys.stderr.write("provisioning_error\n")
        return 1

    sys.stdout.write(
        f"provisioned {args.binding_output.name} {args.config_output.name}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
