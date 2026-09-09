from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.stage6.p6_capability_runtime_authority_v1 import (
    P6CapabilityRuntimeAuthorityError,
    _helper_environment,
    _normalized_runtime_inputs,
    _parse_authority_output,
    _run_authority_helper,
    _validated_context_path,
    probe_normalized_capability_authority,
)
from tests.python_runtime_fixture import write_current_python_launcher
from tests.stage6.test_p6_capability_context import probe_evidence
from tests.stage6.test_p6_rtx_capability_search_context import (
    _write_rtx_context_source,
)


def _output() -> dict[str, object]:
    evidence = probe_evidence()
    return {
        "schema_version": "p6_normalized_capability_authority_v1",
        "runtime_identity": evidence["runtime_identity"],
        "probe_records": {
            "neutral": evidence["neutral_records"],
            "pruning": evidence["pruning_records"],
        },
    }


def test_authority_output_requires_exact_runtime_and_probe_partitions() -> None:
    payload = _output()
    parsed = _parse_authority_output(json.dumps(payload))
    assert parsed.runtime_identity == payload["runtime_identity"]
    assert len(parsed.probe_records["neutral"]) == 12

    for mutation in ("extra", "missing", "duplicate"):
        changed = _output()
        neutral = changed["probe_records"]["neutral"]  # type: ignore[index]
        if mutation == "extra":
            changed["extra"] = True
        elif mutation == "missing":
            neutral.pop()  # type: ignore[union-attr]
        else:
            neutral[-1] = neutral[0]  # type: ignore[index]
        with pytest.raises(P6CapabilityRuntimeAuthorityError):
            _parse_authority_output(json.dumps(changed))


def test_normalized_authority_discards_helper_diagnostics_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = SimpleNamespace(
        python=tmp_path / "python",
        tvm_site=tmp_path / "site",
        nvlibs_file=tmp_path / "nvlibs",
        support_root=tmp_path / "support",
        support_root_sha256="a" * 64,
        dependency_root=tmp_path / "dependency",
        gpu_indices=(0, 1, 2, 3),
    )
    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1._normalized_runtime_inputs",
        lambda **kwargs: inputs,
    )
    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1._run_authority_helper",
        lambda **kwargs: json.dumps(_output()),
    )
    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir()
    context = inputs_root / "context.json"
    context.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1._validated_context_path",
        lambda path, root: path,
    )

    result = probe_normalized_capability_authority(
        private_root=tmp_path,
        capability_context_path=context,
        expected_gpu_indices=(0, 1, 2, 3),
    )
    assert result.runtime_identity["target"] == "cuda -arch=sm_89"

    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1._run_authority_helper",
        lambda **kwargs: (_ for _ in ()).throw(OSError("/private/diagnostic")),
    )
    with pytest.raises(
        P6CapabilityRuntimeAuthorityError, match="capability_runtime_authority_invalid"
    ):
        probe_normalized_capability_authority(
            private_root=tmp_path,
            capability_context_path=context,
            expected_gpu_indices=(0, 1, 2, 3),
        )


def test_runtime_inputs_are_locked_to_normalized_profile_and_runner(
    tmp_path: Path,
) -> None:
    normalized, _ = _write_rtx_context_source(tmp_path)
    inputs = _normalized_runtime_inputs(
        private_root=normalized, expected_gpu_indices=None
    )
    assert inputs.support_root.is_relative_to(normalized)
    assert len(inputs.gpu_indices) == 4

    runtime_indices = (7,)
    rebound = _normalized_runtime_inputs(
        private_root=normalized,
        expected_gpu_indices=runtime_indices,
    )
    assert rebound.gpu_indices == runtime_indices


@pytest.mark.parametrize("runtime_indices", [(7,), (3, 1), (6, 2, 5)])
def test_runtime_gpu_indices_override_historical_runner_literal(
    tmp_path: Path,
    runtime_indices: tuple[int, ...],
) -> None:
    normalized, _ = _write_rtx_context_source(tmp_path)

    inputs = _normalized_runtime_inputs(
        private_root=normalized,
        expected_gpu_indices=runtime_indices,
    )

    assert inputs.gpu_indices == runtime_indices
    assert _helper_environment(inputs)["CUDA_VISIBLE_DEVICES"] == ",".join(
        map(str, runtime_indices)
    )


def test_runtime_inputs_keep_adapter_and_formal_tvm_python_roles_distinct(
    tmp_path: Path,
) -> None:
    tvm_python = tmp_path / "tvm-runtime/bin/python3.10"
    write_current_python_launcher(tvm_python)
    normalized, _ = _write_rtx_context_source(
        tmp_path, tvm_python=str(tvm_python)
    )

    inputs = _normalized_runtime_inputs(
        private_root=normalized, expected_gpu_indices=None
    )

    assert inputs.python == tvm_python


def test_context_path_and_helper_environment_fail_closed(
    tmp_path: Path,
) -> None:
    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir()
    context = inputs_root / "context.json"
    context.write_text("{}", encoding="utf-8")
    assert _validated_context_path(context, tmp_path) == context

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(P6CapabilityRuntimeAuthorityError):
        _validated_context_path(outside, tmp_path)

    tvm_site = tmp_path / "site"
    tvm_site.mkdir()
    nvlibs = tmp_path / "nvlibs"
    nvlibs.write_text("/runtime/lib\n", encoding="utf-8")
    inputs = SimpleNamespace(
        tvm_site=tvm_site,
        nvlibs_file=nvlibs,
        gpu_indices=(2, 3, 4, 5),
    )
    environment = _helper_environment(inputs)
    assert environment["CUDA_VISIBLE_DEVICES"] == "2,3,4,5"
    assert environment["LD_LIBRARY_PATH"].endswith("/runtime/lib")

    inputs.nvlibs_file = tmp_path / "missing"
    with pytest.raises(P6CapabilityRuntimeAuthorityError):
        _helper_environment(inputs)


def test_helper_launch_is_exact_and_rejects_process_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = SimpleNamespace(
        python=tmp_path / "python",
        tvm_site=tmp_path / "site",
        nvlibs_file=tmp_path / "nvlibs",
        support_root=tmp_path / "support",
        support_root_sha256="a" * 64,
        dependency_root=tmp_path / "dependency",
        gpu_indices=(0, 1, 2, 3),
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1._regular_file",
        lambda path: True,
    )
    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1._helper_environment",
        lambda value: {"LOCKED": "1"},
    )

    def successful_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps(_output()))

    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1.subprocess.run",
        successful_run,
    )
    assert json.loads(
        _run_authority_helper(inputs=inputs, capability_context_path=tmp_path / "context")
    )["schema_version"] == "p6_normalized_capability_authority_v1"
    assert observed["kwargs"]["env"] == {"LOCKED": "1"}  # type: ignore[index]
    assert observed["kwargs"]["shell"] is False  # type: ignore[index]

    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="diagnostic"),
    )
    with pytest.raises(P6CapabilityRuntimeAuthorityError):
        _run_authority_helper(inputs=inputs, capability_context_path=tmp_path / "context")

    monkeypatch.setattr(
        "framework.stage6.p6_capability_runtime_authority_v1.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private detail")),
    )
    with pytest.raises(P6CapabilityRuntimeAuthorityError):
        _run_authority_helper(inputs=inputs, capability_context_path=tmp_path / "context")


def test_authority_parser_rejects_resealed_identity_and_schema_drift() -> None:
    for mutation in ("fingerprint", "records", "schema"):
        payload = _output()
        if mutation == "fingerprint":
            payload["runtime_identity"]["compiler_fingerprint"] = "b" * 64  # type: ignore[index]
        elif mutation == "records":
            payload["probe_records"] = []
        else:
            payload["schema_version"] = "alternate"
        with pytest.raises(P6CapabilityRuntimeAuthorityError):
            _parse_authority_output(json.dumps(payload))
