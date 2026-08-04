"""Public-contract checks for maintainer-authorized synthetic smoke fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "framework" / "tests" / "fixtures"
SMOKE_ROOT = FIXTURE_ROOT / "stage12_smoke"


def _load_yaml(relative_path: str) -> dict[str, object]:
    value = yaml.safe_load((FIXTURE_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_json(relative_path: str) -> dict[str, object]:
    value = json.loads((FIXTURE_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_release_asset_fixture_is_placeholder_only_and_checksum_bound() -> None:
    manifest = _load_yaml("release_assets/manifest.yaml")
    assert manifest["schema"] == "univ2x_release_assets_v1"
    assets = manifest["assets"]
    assert isinstance(assets, list) and len(assets) == 2
    for asset in assets:
        assert isinstance(asset, dict)
        assert str(asset["source"]).startswith("https://example.invalid/")
        checksum = asset["checksum"]
        assert isinstance(checksum, dict)
        assert checksum["algorithm"] == "sha256"
        assert len(str(checksum["value"])) == 64
        assert checksum["required_before_formal_run"] is True


def test_stage1_manifests_are_synthetic_and_never_full_model_evidence() -> None:
    for name in ("codriving.yaml", "pyramid_lidar.yaml", "where2comm.yaml"):
        manifest = _load_yaml(f"stage12_smoke/manifests/{name}")
        assert manifest["stage"] == "stage1_partition"
        assert manifest["trace_plan"]["coverage_scope"] == "dense_core_only"
        assert manifest["view_latency"]["coverage"]["coverage_scope"] == "trace_net_only"
        assert manifest["view_latency"]["coverage"]["full_model_latency_pct"] is None
        assert "Synthetic dense-core fixture" in str(manifest["view_latency"]["coverage"]["note"])
        assert "results/" not in json.dumps(manifest, sort_keys=True)


def test_stage2_inputs_are_explicit_synthetic_fixtures() -> None:
    anchors = _load_json("stage12_smoke/stage2/ap_anchors.json")
    latency = _load_json("stage12_smoke/stage2/latency_lut.json")
    classification = _load_json("stage12_smoke/stage2/classification.json")
    assert anchors["_source"] == "synthetic_fixture"
    assert latency["_source"] == "synthetic_fixture"
    assert all(item["evidence_level"] == "fixture_only" for item in classification["models"])


def test_anchor_evidence_is_scoped_fixture_data_not_paper_evidence() -> None:
    anchor = _load_json("stage12_smoke/stage1_evidence/standard_conv_anchor_measured_results_v1.json")
    assert anchor["_source"] == "synthetic_fixture"
    assert anchor["evidence_level"] == "fixture_only"
    assert anchor["paper_evidence"] is False
    assert isinstance(anchor["anchors"], dict)


def test_fixture_tree_contains_only_declared_small_text_inputs() -> None:
    files = sorted(path for path in FIXTURE_ROOT.rglob("*") if path.is_file())
    assert len(files) == 8
    assert {path.suffix for path in files} == {".json", ".yaml"}
    assert all(path.stat().st_size < 4096 for path in files)
    digest = hashlib.sha256(
        b"".join(path.relative_to(FIXTURE_ROOT).as_posix().encode("utf-8") + path.read_bytes() for path in files)
    ).hexdigest()
    assert len(digest) == 64
