from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.stage6.test_p6_capability_context import probe_evidence
from tools.release import rebuild_p6_rtx_capability_authority as cli


def _arguments(tmp_path: Path) -> SimpleNamespace:
    paths = {
        name: tmp_path / name
        for name in ("tvm-site", "support-root", "dependency-root")
    }
    for path in paths.values():
        path.mkdir()
    nvlibs = tmp_path / "nvlibs.path"
    nvlibs.write_text("/runtime/lib\n", encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps({"measurement_evidence": probe_evidence()}), encoding="utf-8"
    )
    return SimpleNamespace(
        context=context,
        tvm_site=paths["tvm-site"],
        nvlibs_file=nvlibs,
        support_root=paths["support-root"],
        support_root_sha256="a" * 64,
        dependency_root=paths["dependency-root"],
    )


def test_helper_rebuilds_runtime_and_both_partitions_from_raw_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    evidence = probe_evidence()
    observed: list[str] = []
    monkeypatch.setattr(
        cli, "canonical_tvm_support_tree_sha256", lambda root: "a" * 64
    )
    monkeypatch.setattr(
        cli, "collect_tvm_runtime_identity", lambda **kwargs: evidence["runtime_identity"]
    )

    def rebuild(blobs, *, family, probe_ids):
        assert len(blobs) == len(probe_ids) * 2
        observed.append(family)
        return evidence[f"{family}_records"]

    monkeypatch.setattr(cli, "rebuild_probe_records", rebuild)

    result = cli.run(args)

    assert observed == ["neutral", "pruning"]
    assert result["runtime_identity"] == evidence["runtime_identity"]
    assert set(result["probe_records"]) == {"neutral", "pruning"}
    assert str(tmp_path) not in json.dumps(result)


def test_helper_main_reports_only_sanitized_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run",
        lambda args: (_ for _ in ()).throw(OSError("/private/runtime-path")),
    )
    monkeypatch.setattr(cli, "parse_args", lambda argv: SimpleNamespace())
    assert cli.main([]) == 1
    assert capsys.readouterr().err == "capability_runtime_authority_failed\n"


def test_helper_paths_and_context_reject_aliases_and_invalid_payloads(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular.json"
    regular.write_text("{}", encoding="utf-8")
    assert cli._path(regular, want_dir=False) == regular

    alias = tmp_path / "alias.json"
    alias.symlink_to(regular)
    with pytest.raises(ValueError, match="authority input invalid"):
        cli._path(alias, want_dir=False)
    with pytest.raises(ValueError, match="authority input invalid"):
        cli._path(tmp_path / "missing", want_dir=False)

    regular.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="authority input invalid"):
        cli._read_context(regular)
    regular.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="authority input invalid"):
        cli._read_context(regular)


def test_helper_rejects_support_and_runtime_target_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    monkeypatch.setattr(
        cli, "canonical_tvm_support_tree_sha256", lambda root: "b" * 64
    )
    with pytest.raises(ValueError, match="authority input invalid"):
        cli.run(args)

    evidence = probe_evidence()
    monkeypatch.setattr(
        cli, "canonical_tvm_support_tree_sha256", lambda root: "a" * 64
    )
    monkeypatch.setattr(
        cli,
        "rebuild_probe_records",
        lambda blobs, *, family, probe_ids: evidence[f"{family}_records"],
    )
    changed_runtime = dict(evidence["runtime_identity"])
    changed_runtime["target"] = "cuda -arch=sm_80"
    monkeypatch.setattr(cli, "collect_tvm_runtime_identity", lambda **kwargs: changed_runtime)
    with pytest.raises(ValueError, match="authority input invalid"):
        cli.run(args)


def test_helper_main_emits_canonical_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"schema_version": "test"}
    monkeypatch.setattr(cli, "parse_args", lambda argv: SimpleNamespace())
    monkeypatch.setattr(cli, "run", lambda args: payload)
    assert cli.main([]) == 0
    assert json.loads(capsys.readouterr().out) == payload
