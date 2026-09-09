"""Public, redacted P6 full-chain template and hardware-contract gates."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.stage6.coptv2x_h800_search_v2 import METRIC_NAMES, load_public_contract


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_CONFIG = ROOT / "configs/execution"

SEARCH_CONTRACTS = {
    "h800": EXECUTION_CONFIG / "p6_h800_search.example.yaml",
    "rtx4090": EXECUTION_CONFIG / "p6_rtx4090_search.example.yaml",
}
FULL_CHAIN_MANIFESTS = {
    "h800": EXECUTION_CONFIG / "p6_h800_full_chain.example.yaml",
    "rtx4090": EXECUTION_CONFIG / "p6_rtx4090_full_chain.example.yaml",
}
USER_INPUT_TEMPLATES = (
    EXECUTION_CONFIG / "p6_history_source_map.example.yaml",
    EXECUTION_CONFIG / "p6_history_runner_template.example.yaml",
    EXECUTION_CONFIG / "p6_h800_local_locator.example.yaml",
    EXECUTION_CONFIG / "p6_rtx4090_local_locator.example.yaml",
    EXECUTION_CONFIG / "p6_external_training_binding.example.yaml",
)
PRIVATE_JSON_FORMAT_TEMPLATES = {
    "gold176_rows": EXECUTION_CONFIG / "p6_gold176_rows.example.json",
    "gold176_graph_features": EXECUTION_CONFIG
    / "p6_gold176_graph_features.example.json",
    "closure": EXECUTION_CONFIG / "p6_closure.example.json",
    "h800_capability_profiles": EXECUTION_CONFIG
    / "p6_h800_capability_profiles.example.json",
    "rtx4090_capability_context": EXECUTION_CONFIG
    / "p6_rtx4090_capability_context.example.json",
}
PUBLIC_YAML_EXAMPLES = (
    *SEARCH_CONTRACTS.values(),
    *FULL_CHAIN_MANIFESTS.values(),
    *USER_INPUT_TEMPLATES,
)

MANIFEST_KEYS = {
    "schema_version",
    "template_only",
    "hardware_profile",
    "contract",
    "source_map",
    "history_root",
    "pre_normalization_runner_template",
    "normalized_private_dir",
    "fresh_output_root",
}
FORBIDDEN_RUNTIME_SELECTOR_KEYS = {
    "gpu_count",
    "gpu_ids",
    "gpu_indices",
}
SENSITIVE_KEYS = {"token", "password", "api_key", "private_key", "access_key"}
PLACEHOLDER = re.compile(r"^(?:\$\{[A-Z][A-Z0-9_]*\}|<[A-Z][A-Z0-9_-]*>)$")
HOME_PATH_MARKER = "/" + "home/"
EXDATA_PATH_MARKER = "/" + "exdata/"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    return payload


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(value: object) -> list[tuple[str | None, object]]:
    found: list[tuple[str | None, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.append((str(key).lower(), child))
            found.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk(child))
    return found


def test_every_public_full_chain_example_exists_is_tracked_and_is_yaml_mapping() -> None:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )

    for path in PUBLIC_YAML_EXAMPLES:
        assert path.is_file(), path.relative_to(ROOT)
        assert path.relative_to(ROOT).as_posix() in tracked
        _read_yaml(path)


def test_every_private_json_format_template_exists_is_tracked_and_parseable() -> None:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )

    for path in PRIVATE_JSON_FORMAT_TEMPLATES.values():
        assert path.is_file(), path.relative_to(ROOT)
        assert path.relative_to(ROOT).as_posix() in tracked
        _read_json(path)


def test_private_json_templates_publish_each_required_payload_shape() -> None:
    rows = _read_json(PRIVATE_JSON_FORMAT_TEMPLATES["gold176_rows"])
    graphs = _read_json(PRIVATE_JSON_FORMAT_TEMPLATES["gold176_graph_features"])
    closure = _read_json(PRIVATE_JSON_FORMAT_TEMPLATES["closure"])
    h800 = _read_json(PRIVATE_JSON_FORMAT_TEMPLATES["h800_capability_profiles"])
    rtx4090 = _read_json(
        PRIVATE_JSON_FORMAT_TEMPLATES["rtx4090_capability_context"]
    )

    assert isinstance(rows, list) and len(rows) == 1
    assert {
        "manifest_job_id",
        "row_id",
        "group_id",
        "model",
        "width",
        "dispatch_key",
        "capability_profile_id",
        "q_mode",
        "latency_ms",
        "energy_j",
        "ap30",
        "ap50",
        "ap70",
        "terminal_status",
        "training_source",
    } == set(rows[0])
    assert isinstance(graphs, list) and len(graphs) == 1
    assert {"group_id", "model", "width"} < set(graphs[0])
    assert isinstance(closure, dict)
    assert closure["schema_version"] == "stage4_p1_p3_closure_audit_v1"
    assert isinstance(h800, list) and len(h800) == 2
    assert {item["dispatch_key"] for item in h800} == {"tvm_auto", "trt_engine"}
    assert all(item["hardware_target"] == "h800" for item in h800)
    assert isinstance(rtx4090, dict)
    assert set(rtx4090) == {
        "schema_version",
        "historical_source_base64",
        "historical_profiles",
        "active_profile",
        "measurement_evidence",
        "context_digest",
    }
    assert rtx4090["schema_version"] == "p6_rtx_capability_context_v1"
    assert rtx4090["active_profile"]["hardware_target"] == "rtx4090"


def test_private_json_format_templates_are_redacted_and_do_not_publish_metrics() -> None:
    forbidden_text = re.compile(
        rf"(?:{re.escape(HOME_PATH_MARKER)}|{re.escape(EXDATA_PATH_MARKER)}|"
        r"https?://|ssh://|GPU-[0-9a-f-]{8,})",
        re.IGNORECASE,
    )

    for name, path in PRIVATE_JSON_FORMAT_TEMPLATES.items():
        text = path.read_text(encoding="utf-8")
        assert not forbidden_text.search(text), path.relative_to(ROOT)
        if name in {"h800_capability_profiles", "rtx4090_capability_context"}:
            keys = {key for key, _ in _walk(_read_json(path)) if key is not None}
            assert not {"latency_ms", "energy_j", "ap30", "ap50", "ap70"} & keys


@pytest.mark.parametrize("profile", ["h800", "rtx4090"])
def test_full_chain_manifest_is_an_explicit_non_executable_template(profile: str) -> None:
    payload = _read_yaml(FULL_CHAIN_MANIFESTS[profile])

    assert set(payload) == MANIFEST_KEYS
    assert payload["template_only"] is True
    assert payload["hardware_profile"] == profile
    assert payload["contract"] == (
        f"configs/execution/p6_{profile}_search.example.yaml"
    )
    assert FORBIDDEN_RUNTIME_SELECTOR_KEYS.isdisjoint(
        {str(key).lower() for key, _ in _walk(payload)}
    )
    assert "cuda_visible_devices" not in {
        str(key).lower() for key, _ in _walk(payload)
    }
    for key in (
        "source_map",
        "history_root",
        "pre_normalization_runner_template",
        "normalized_private_dir",
        "fresh_output_root",
    ):
        assert isinstance(payload[key], str)
        assert PLACEHOLDER.fullmatch(payload[key]), (key, payload[key])


def test_h800_and_rtx4090_manifests_have_the_same_shape() -> None:
    h800 = _read_yaml(FULL_CHAIN_MANIFESTS["h800"])
    rtx4090 = _read_yaml(FULL_CHAIN_MANIFESTS["rtx4090"])

    assert set(h800) == set(rtx4090) == MANIFEST_KEYS
    differing = {key for key in MANIFEST_KEYS if h800[key] != rtx4090[key]}
    assert differing <= {"hardware_profile", "contract"}


def test_user_input_templates_publish_the_current_schema_shapes() -> None:
    source_map = _read_yaml(EXECUTION_CONFIG / "p6_history_source_map.example.yaml")
    runner = _read_yaml(EXECUTION_CONFIG / "p6_history_runner_template.example.yaml")
    external_training = _read_yaml(
        EXECUTION_CONFIG / "p6_external_training_binding.example.yaml"
    )
    h800_locator = _read_yaml(EXECUTION_CONFIG / "p6_h800_local_locator.example.yaml")
    rtx_locator = _read_yaml(
        EXECUTION_CONFIG / "p6_rtx4090_local_locator.example.yaml"
    )

    assert source_map["schema_version"] == "p6_history_normalization_source_v5"
    assert {
        "history_root",
        "asset_paths",
        "input_sources",
        "source_contract",
        "recipe_mode",
        "procedural_recipe_profile",
        "procedural_recipe_source",
        "external_training_binding",
        "execution_code_closure",
        "post_source_leaf_binding",
        "adapter_python",
        "adapter_dependency_closure_id",
    } <= set(source_map)
    assert runner["schema_version"] == "p6_history_runner_template_v1"
    assert set(runner) == {"schema_version", "stage1_scan", "execution_interface"}
    assert external_training["schema_version"] == "p6_external_training_binding_v1"
    assert h800_locator["hardware_profile"] == h800_locator["target"] == "h800"
    assert rtx_locator["hardware_profile"] == rtx_locator["target"] == "rtx4090"
    assert set(h800_locator) == set(rtx_locator)


def test_all_new_templates_are_redacted_and_contain_no_fixed_gpu_binding() -> None:
    forbidden_text = re.compile(
        rf"(?:{re.escape(HOME_PATH_MARKER)}|{re.escape(EXDATA_PATH_MARKER)}|"
        r"https?://|ssh://|GPU-[0-9a-f-]{8,})",
        re.IGNORECASE,
    )

    for path in (*FULL_CHAIN_MANIFESTS.values(), *USER_INPUT_TEMPLATES):
        text = path.read_text(encoding="utf-8")
        payload = _read_yaml(path)
        assert not forbidden_text.search(text), path.relative_to(ROOT)
        assert FORBIDDEN_RUNTIME_SELECTOR_KEYS.isdisjoint(
            {str(key).lower() for key, _ in _walk(payload)}
        )
        for key, value in _walk(payload):
            if key == "cuda_visible_devices":
                assert isinstance(value, dict)
                rendered = yaml.safe_dump(value)
                assert "${" in rendered or "<" in rendered
                assert not re.search(r"value:\s*['\"]?\d+(?:,\d+)*", rendered)
            if key in SENSITIVE_KEYS:
                assert value is None or (
                    isinstance(value, str) and PLACEHOLDER.fullmatch(value)
                ), (path.relative_to(ROOT), key)


@pytest.mark.parametrize(
    ("profile", "expected_arch"),
    [("h800", "sm90"), ("rtx4090", "sm89")],
)
def test_hardware_search_examples_use_the_shared_v3_four_round_contract(
    profile: str, expected_arch: str
) -> None:
    payload = _read_yaml(SEARCH_CONTRACTS[profile])
    contract = load_public_contract(SEARCH_CONTRACTS[profile])

    assert payload["schema_version"] == "p6_coptv2x_search_contract_v3"
    assert payload["hardware_profile"] == payload["target"] == profile
    assert contract.hardware_profile.profile_id == profile
    assert contract.hardware_profile.tvm_arch == expected_arch
    assert (contract.sample_budget, contract.batch_size, contract.round_count) == (
        16,
        4,
        4,
    )
    assert contract.metric_names == METRIC_NAMES


def test_h800_and_rtx4090_search_examples_are_structurally_parallel() -> None:
    h800 = _read_yaml(SEARCH_CONTRACTS["h800"])
    rtx4090 = _read_yaml(SEARCH_CONTRACTS["rtx4090"])

    assert set(h800) == set(rtx4090)
    for invariant in (
        "schema_version",
        "target_model",
        "execution_backend",
        "seed",
        "sample_budget",
        "batch_size",
        "round_count",
        "candidate_space_label",
        "metric_names",
    ):
        assert h800[invariant] == rtx4090[invariant]
