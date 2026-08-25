from __future__ import annotations

from pathlib import Path

import pytest

from tools.release import run_p6_finalization_round_adapter as cli


def test_cli_requires_six_absolute_stage_paths(tmp_path: Path) -> None:
    assert cli.main(["--profile", str(tmp_path / "profile.yaml"), "relative"]) == 2


def test_cli_loads_profile_and_calls_finalization_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("private", encoding="utf-8")
    paths = [tmp_path / name for name in ("request", "state", "feedback", "receipt", "barrier", "round")]
    sentinel = object()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "load_post_source_adapter_profile", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        cli,
        "run_finalization_round",
        lambda *args: calls.append(args),
    )

    result = cli.main(["--profile", str(profile_path), *(str(path) for path in paths)])

    output = capsys.readouterr()
    assert result == 0
    assert output.out == "finalization_complete\n"
    assert output.err == ""
    assert calls and calls[0][:-1] == (sentinel, *paths)
    assert isinstance(calls[0][-1], cli.SubprocessLeafRunner)


def test_cli_redacts_adapter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("private", encoding="utf-8")
    paths = [tmp_path / name for name in ("request", "state", "feedback", "receipt", "barrier", "round")]
    monkeypatch.setattr(cli, "load_post_source_adapter_profile", lambda *args, **kwargs: object())

    def fail(*args: object) -> None:
        del args
        raise cli.P6FinalizationRoundAdapterError()

    monkeypatch.setattr(cli, "run_finalization_round", fail)

    assert cli.main(["--profile", str(profile_path), *(str(path) for path in paths)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "history_execution_invalid\n"
