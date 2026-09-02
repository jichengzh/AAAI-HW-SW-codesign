from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.stage6.p6_capability_context_v1 import canonical_probe_code_sha256
from framework.stage6.p6_capability_probe_specs_v1 import (
    NEUTRAL_PROBE_IDS,
    PRUNING_PROBE_IDS,
)
from framework.stage6.p6_capability_probe_worker_v1 import run_probe_family
from tests.stage6.test_p6_capability_context import historical_profiles, runtime_identity
from tools.release import probe_p6_rtx_capability as cli


def _counts(q_mode: str) -> dict:
    return {
        "int8_propagated_ops": 1 if q_mode == "int8" else None,
        "precision_eligible_ops": 2 if q_mode == "int8" else None,
        "qdq_folded_pairs": 1 if q_mode == "int8" else None,
        "qdq_pairs": 1 if q_mode == "int8" else None,
        "reformat_ops": 0,
        "total_ops": 2,
        "fused_ops": 0,
        "fusible_ops": 1,
    }


def _family(family: str, probe_ids: tuple[str, ...]):
    return run_probe_family(
        family=family,
        probe_ids=probe_ids,
        model_builder=lambda probe_id, q_mode: f"onnx:{probe_id}:{q_mode}".encode(),
        compiler=lambda payload, q_mode: (_counts(q_mode), b"compiler-ir:" + payload),
    )


def test_cli_path_json_gpu_and_atomic_output_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / ".git").mkdir()
    assert cli._private_root(root) == root
    payload_path = root / "policy.json"
    payload_path.write_text(
        json.dumps({"schema_version": "p6_private_gpu_policy_v1", "indices": [0, 1, 2, 3]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert cli._gpu_indices(payload_path) == (0, 1, 2, 3)
    output = root / "context.json"
    assert cli._output_path(output) == output
    cli._write_context(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="output"):
        cli._output_path(output)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1,0")
    with pytest.raises(ValueError, match="GPU policy"):
        cli._gpu_indices(payload_path)


def test_cli_builds_both_exact_probe_families(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "require_cuda_sm89", lambda: calls.append("cuda"))
    monkeypatch.setattr(
        cli,
        "build_probe_onnx",
        lambda family, probe_id, q_mode: f"{family}:{probe_id}:{q_mode}".encode(),
    )
    monkeypatch.setattr(
        cli, "compile_tvm_probe", lambda payload, q_mode: (_counts(q_mode), payload)
    )

    neutral, pruning = cli._run_probe_families()

    assert calls == ["cuda"]
    assert len(neutral.records) == len(pruning.records) == 12


def test_cli_build_context_recomputes_all_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_identity()
    monkeypatch.setattr(cli, "collect_tvm_runtime_identity", lambda **kwargs: runtime)
    context = cli._build_context(
        profile=cli.load_hardware_execution_profile("rtx4090"),
        historical=historical_profiles(),
        runtime={
            "P6_TVM_SITE": "/unused/site",
            "P6_TVM_NVLIBS_FILE": "/unused/nvlibs",
            "P6_TVM_SUPPORT_ROOT_SHA256": runtime["support_root_sha256"],
        },
        verified_count=4,
        neutral=_family("neutral", NEUTRAL_PROBE_IDS),
        pruning=_family("pruning", PRUNING_PROBE_IDS),
    )
    assert context["measurement_evidence"]["probe_code_sha256"] == (
        canonical_probe_code_sha256(cli.REPOSITORY_ROOT)
    )
    assert context["active_profile"]["hardware_target"] == "rtx4090"


def test_cli_run_and_main_have_sanitized_aggregate_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps(historical_profiles()), encoding="utf-8")
    output = tmp_path / "output.json"
    neutral = _family("neutral", NEUTRAL_PROBE_IDS)
    pruning = _family("pruning", PRUNING_PROBE_IDS)
    monkeypatch.setattr(cli, "_private_root", lambda value: value)
    monkeypatch.setattr(cli, "_runtime_values", lambda *args: (None, {}))
    monkeypatch.setattr(cli, "_gpu_indices", lambda value: (0, 1, 2, 3))
    monkeypatch.setattr(cli, "_output_path", lambda value: value)
    monkeypatch.setattr(
        cli, "NvidiaSmiGpuProbe", lambda: SimpleNamespace(snapshot=lambda indices: ())
    )
    monkeypatch.setattr(cli, "validate_live_gpu_snapshots", lambda **kwargs: 4)
    monkeypatch.setattr(cli, "_run_probe_families", lambda: (neutral, pruning))
    monkeypatch.setattr(
        cli,
        "_build_context",
        lambda **kwargs: {
            "schema_version": "p6_rtx_capability_context_v1",
            "active_profile": {"features": {str(index): 1.0 for index in range(23)}},
        },
    )
    args = argparse.Namespace(
        private_root=tmp_path,
        profile=tmp_path / "profile",
        runner_template=tmp_path / "runner",
        historical_profiles=historical,
        gpu_policy=tmp_path / "gpu",
        output=output,
    )

    result = cli.run(args)
    assert result == {
        "schema_version": "p6_rtx_capability_context_v1",
        "verified_gpu_count": 4,
        "neutral_cells": 12,
        "pruning_cells": 12,
        "feature_count": 23,
    }
    monkeypatch.setattr(cli, "run", lambda args: result)
    assert (
        cli.main(
            [
                "--private-root",
                str(tmp_path),
                "--profile",
                "p",
                "--runner-template",
                "r",
                "--historical-profiles",
                "h",
                "--gpu-policy",
                "g",
                "--output",
                "o",
            ]
        )
        == 0
    )
    assert "verified_gpu_count" in capsys.readouterr().out
    monkeypatch.setattr(cli, "run", lambda args: (_ for _ in ()).throw(ValueError()))
    assert (
        cli.main(
            [
                "--private-root",
                str(tmp_path),
                "--profile",
                "p",
                "--runner-template",
                "r",
                "--historical-profiles",
                "h",
                "--gpu-policy",
                "g",
                "--output",
                "o",
            ]
        )
        == 1
    )
    assert capsys.readouterr().err == "capability_probe_failed\n"
