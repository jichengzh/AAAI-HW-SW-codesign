#!/usr/bin/env python3
"""Produce one immutable real-TVM RTX4090 capability context."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.hardware_execution_profile_v1 import (  # noqa: E402
    load_hardware_execution_profile,
)
from framework.stage6.p6_capability_context_v1 import (  # noqa: E402
    build_rtx_capability_context,
    canonical_probe_code_sha256,
    capability_context_to_mapping,
)
from framework.stage6.p6_capability_probe_models_v1 import (  # noqa: E402
    build_probe_onnx,
)
from framework.stage6.p6_capability_probe_producer_v1 import (  # noqa: E402
    build_probe_evidence,
    validate_live_gpu_snapshots,
)
from framework.stage6.p6_capability_probe_specs_v1 import (  # noqa: E402
    NEUTRAL_PROBE_IDS,
    PRUNING_PROBE_IDS,
)
from framework.stage6.p6_capability_probe_worker_v1 import (  # noqa: E402
    run_probe_family,
)
from framework.stage6.p6_capability_tvm_v1 import (  # noqa: E402
    collect_tvm_runtime_identity,
    compile_tvm_probe,
    require_cuda_sm89,
)
from framework.stage6.p6_gpu_policy_v1 import canonical_gpu_indices  # noqa: E402
from framework.stage6.p6_history_binding_v1 import FORMAL_TVM_ENV_KEYS  # noqa: E402
from framework.stage6.p6_post_source_adapter_profile_v1 import (  # noqa: E402
    load_post_source_adapter_profile,
    require_post_source_adapter_profile_v4,
)
from framework.stage6.p6_python_runtime_v1 import validate_adapter_python  # noqa: E402
from framework.stage6.p6_runner_template_validator_v1 import (  # noqa: E402
    validate_pre_provision_runner_template,
)
from tools.release.provision_p6_history_local_config import (  # noqa: E402
    NvidiaSmiGpuProbe,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _read_json(path: Path) -> Any:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
            raise OSError
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("capability probe input invalid") from error


def _private_root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path != resolved or not path.is_dir() or not (path / ".git").is_dir():
        raise ValueError("capability probe root invalid")
    return resolved


def _runtime_values(
    private_root: Path, profile_path: Path, runner_path: Path
) -> tuple[Any, dict[str, str]]:
    post_source = load_post_source_adapter_profile(profile_path, private_root=private_root)
    require_post_source_adapter_profile_v4(post_source)
    runner = validate_pre_provision_runner_template(
        runner_path, private_root, require_exact_history_environment=True
    )
    values = runner.execution_interface["environment"]["values"]
    runtime: dict[str, str] = {}
    kinds = {
        "P6_TVM_PYTHON": "external_executable",
        "P6_TVM_SITE": "external_directory",
        "P6_TVM_NVLIBS_FILE": "external_file",
        "P6_TVM_SUPPORT_ROOT": "private_path",
        "P6_TVM_SUPPORT_ROOT_SHA256": "literal",
    }
    for key in FORMAL_TVM_ENV_KEYS:
        entry = values.get(key)
        if not isinstance(entry, Mapping) or set(entry) != {"kind", "value"}:
            raise ValueError("capability probe runtime invalid")
        value = entry.get("value")
        if (
            entry.get("kind") != kinds[key]
            or not isinstance(value, str)
            or not value
            or os.environ.get(key) != value
        ):
            raise ValueError("capability probe runtime invalid")
        runtime[key] = value
    if (
        post_source.hardware_profile.profile_id != "rtx4090"
        or post_source.adapter_python is None
        or validate_adapter_python(runtime["P6_TVM_PYTHON"]) != post_source.adapter_python
        or Path(sys.executable).resolve(strict=True) != post_source.adapter_python
        or Path(runtime["P6_TVM_SUPPORT_ROOT"]) != post_source.tvm_support_root
        or runtime["P6_TVM_SUPPORT_ROOT_SHA256"] != post_source.tvm_support_root_sha256
    ):
        raise ValueError("capability probe runtime invalid")
    _require_external_path(Path(runtime["P6_TVM_SITE"]), want_dir=True)
    _require_external_path(Path(runtime["P6_TVM_NVLIBS_FILE"]), want_dir=False)
    return post_source, runtime


def _require_external_path(path: Path, *, want_dir: bool) -> None:
    if not path.is_absolute():
        raise ValueError("capability probe runtime invalid")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("capability probe runtime invalid")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError("capability probe runtime invalid") from error
    valid = stat.S_ISDIR(info.st_mode) if want_dir else stat.S_ISREG(info.st_mode)
    if not valid or not want_dir and info.st_nlink != 1:
        raise ValueError("capability probe runtime invalid")


def _gpu_indices(policy_path: Path) -> tuple[int, ...]:
    payload = _read_json(policy_path)
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "indices"}
        or payload.get("schema_version") != "p6_private_gpu_policy_v1"
    ):
        raise ValueError("capability probe GPU policy invalid")
    indices = canonical_gpu_indices(payload["indices"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, indices)):
        raise ValueError("capability probe GPU policy invalid")
    return indices


def _output_path(path: Path) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("capability probe output invalid")
    parent = path.parent.resolve(strict=True)
    info = parent.stat()
    if not parent.is_dir() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("capability probe output invalid")
    return path


def _write_context(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("capability probe output invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _run_probe_families() -> tuple[Any, Any]:
    require_cuda_sm89()
    neutral = run_probe_family(
        family="neutral",
        probe_ids=NEUTRAL_PROBE_IDS,
        model_builder=lambda probe_id, q_mode: build_probe_onnx("neutral", probe_id, q_mode),
        compiler=compile_tvm_probe,
    )
    pruning = run_probe_family(
        family="pruning",
        probe_ids=PRUNING_PROBE_IDS,
        model_builder=lambda probe_id, q_mode: build_probe_onnx("pruning", probe_id, q_mode),
        compiler=compile_tvm_probe,
    )
    return neutral, pruning


def _build_context(
    *,
    profile: Any,
    historical: object,
    runtime: Mapping[str, str],
    verified_count: int,
    neutral: Any,
    pruning: Any,
) -> dict[str, Any]:
    runtime_identity = collect_tvm_runtime_identity(
        tvm_site=Path(runtime["P6_TVM_SITE"]),
        nvlibs_file=Path(runtime["P6_TVM_NVLIBS_FILE"]),
        support_root_sha256=runtime["P6_TVM_SUPPORT_ROOT_SHA256"],
    )
    code_sha = canonical_probe_code_sha256(REPOSITORY_ROOT)
    evidence = build_probe_evidence(
        profile=profile,
        repository_root=REPOSITORY_ROOT,
        verified_gpu_count=verified_count,
        runtime_identity=runtime_identity,
        probe_code_sha256=code_sha,
        neutral=neutral,
        pruning=pruning,
    )
    return capability_context_to_mapping(
        build_rtx_capability_context(
            historical_profiles=historical,
            evidence=evidence,
            profile=profile,
            repository_root=REPOSITORY_ROOT,
            expected_probe_code_sha256=code_sha,
            expected_support_root_sha256=runtime["P6_TVM_SUPPORT_ROOT_SHA256"],
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _private_root(args.private_root)
    _, runtime = _runtime_values(root, args.profile, args.runner_template)
    indices = _gpu_indices(args.gpu_policy)
    output = _output_path(args.output)
    profile = load_hardware_execution_profile("rtx4090")
    probe = NvidiaSmiGpuProbe()
    verified_count = validate_live_gpu_snapshots(
        profile=profile,
        indices=indices,
        first=probe.snapshot(indices),
        second=probe.snapshot(indices),
    )
    neutral, pruning = _run_probe_families()
    context = _build_context(
        profile=profile,
        historical=_read_json(args.historical_profiles),
        runtime=runtime,
        verified_count=verified_count,
        neutral=neutral,
        pruning=pruning,
    )
    _write_context(output, context)
    return {
        "schema_version": context["schema_version"],
        "verified_gpu_count": verified_count,
        "neutral_cells": len(neutral.records),
        "pruning_cells": len(pruning.records),
        "feature_count": len(context["active_profile"]["features"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--runner-template", type=Path, required=True)
    parser.add_argument("--historical-profiles", type=Path, required=True)
    parser.add_argument("--gpu-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception:
        print("capability_probe_failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
