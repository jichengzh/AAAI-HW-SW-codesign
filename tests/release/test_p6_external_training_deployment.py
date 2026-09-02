"""Black-box release gate for regenerated external P6 training deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pytest
import yaml

from framework.stage5.single_target_search_v2 import (
    build_measurement_request,
    build_task_candidate_manifest,
    freeze_initial_coldstart,
)
from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XExecutionError,
    load_public_contract,
)
from framework.stage6.p6_full_chain_bootstrap_v1 import (
    materialize_full_chain_binding,
)
from framework.stage6.p6_history_binding_v1 import GpuRecord
from framework.stage6.p6_history_measurement_v1 import (
    P6HistoryMeasurementError,
    run_history_measurement_batch,
)
from framework.stage6.p6_history_normalization_v1 import (
    P6HistoryNormalizationError,
    normalize_history_inputs,
)
from framework.stage6.p6_history_registry_v1 import materialize_history_registry
from framework.stage6.p6_history_source_materialization_v1 import (
    project_source_materialization_request,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_source_reuse_evidence_v1 import create_fresh_run_context
from tests.release.p6_source_reuse_lifecycle_fixture import _stage1_manifest
from tests.release.scanner_owned_stage1_fixture import (
    scanner_owned_pyramid_stage1_manifest,
)
from tests.stage6.test_coptv2x_h800_search import (
    _OfflineGpuProbe,
    _gold176,
    _profile,
    _public_contract,
    _write_yaml,
)
from tests.stage6.test_p6_history_normalization import (
    _as_v2_procedural,
    valid_private_source_map,
)
from tools.release.derive_p6_history_recipe import main as derive_recipe_main
from tools.release.preflight_p6_materializer_training_bridge import (
    preflight_materializer_training_bridge,
)
import tools.release.render_p6_stage1_launcher as stage1_launcher_renderer
from tools.release.render_p6_stage1_launcher import (
    P6Stage1LauncherRenderError,
    main as render_stage1_launcher_main,
    render_stage1_launcher,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_TOKEN = "PRIVATE-EXTERNAL-DEPLOYMENT-TOKEN"


@dataclass(frozen=True)
class _Result:
    returncode: int = 0


@dataclass
class _FailIfCalledRunner:
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        del cwd, env, shell
        self.calls.append(tuple(argv))
        raise AssertionError("historical process boundary must not be called")


@dataclass
class _RejectingGpuProbe:
    calls: int = 0

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        self.calls += 1
        return tuple(
            GpuRecord(
                index=index,
                uuid=f"GPU-synthetic-{index}",
                model_name="NVIDIA H800 80GB HBM3",
                occupancy=1.0,
            )
            for index in indices
        )


@dataclass(frozen=True)
class _MockedMeasurementBoundary:
    projected_request: Mapping[str, Any]
    binding_mapping: Mapping[str, Any]
    public_round_root: Path
    candidate_count: int
    candidate_rows: int
    frozen_gold_row_ids: frozenset[str]


@dataclass(frozen=True)
class _RegeneratedDeploymentFixture:
    root: Path
    source_map: Path
    source_history_root: Path
    source_runner: Path
    derivation_root: Path
    normalized_root: Path
    local_output_root: Path
    public_contract: Path
    provision_probe: _OfflineGpuProbe
    rejecting_gpu_probe: _RejectingGpuProbe
    fail_if_called_runner: _FailIfCalledRunner

    @property
    def normalized_runner(self) -> Path:
        return self.normalized_root / "runner-template.yaml"

    @property
    def normalized_wrapper_profile(self) -> Path:
        return self.normalized_root / "source-wrapper-profile.yaml"

    @property
    def normalized_external_binding(self) -> Path:
        return self.normalized_root / "external-training-binding.yaml"

    @property
    def binding_path(self) -> Path:
        return self.local_output_root / "binding.json"

    @property
    def local_config_path(self) -> Path:
        return self.local_output_root / "local-config.json"

    @property
    def binding_mapping(self) -> dict[str, Any]:
        return json.loads(self.binding_path.read_text(encoding="utf-8"))

    def derive_normalize_and_provision(self) -> tuple[Path, Path]:
        self.derivation_root.mkdir(mode=0o700)
        assert (
            derive_recipe_main(
                [
                    "--source-map",
                    str(self.source_map),
                    "--runner-template",
                    str(self.source_runner),
                    "--recipe-json",
                    str(self.derivation_root / "recipe.json"),
                ]
            )
            == 0
        )
        normalized = normalize_history_inputs(
            yaml.safe_load(self.source_map.read_text(encoding="utf-8")),
            self.source_history_root,
            self.normalized_root,
            runner_template_path=self.source_runner,
        )
        self.local_output_root.mkdir(mode=0o700)
        materialize_full_chain_binding(
            normalized["legacy"],
            normalized["runner_template"],
            self.local_output_root,
            self.binding_path,
            self.local_config_path,
            self.provision_probe,
            source_wrapper_profile=normalized["source_wrapper_profile"],
            external_training_binding=normalized["external_training_binding"],
        )
        return self.binding_path, self.local_config_path

    def prepare_mocked_measurement(self) -> _MockedMeasurementBoundary:
        plan = _dynamic_candidate_plan(self.root)
        _write_json(self.local_output_root / "pyramid_candidate_plan.json", plan)
        binding = self.binding_mapping
        registry = materialize_history_registry(
            plan,
            binding,
            self.local_output_root,
        )
        projected, candidate_rows = _projected_dynamic_request(
            self.public_contract, registry
        )
        context_registry = execution._normalize_recipe_v2_registry_paths(registry)
        _write_json(
            self.local_output_root / "source_registry.json", context_registry
        )
        create_fresh_run_context(
            local_output_root=self.local_output_root,
            task_contract={
                "task_id": projected["task_id"],
                "task_sha256": projected["task_sha256"],
            },
            code_revision="external-deployment-test-v1",
            candidate_plan=plan,
            source_registry=context_registry,
        )
        public_round_root = self.local_output_root / "round-00"
        public_round_root.mkdir()
        return _MockedMeasurementBoundary(
            projected_request=projected,
            binding_mapping=binding,
            public_round_root=public_round_root,
            candidate_count=int(plan["candidate_count"]),
            candidate_rows=candidate_rows,
            frozen_gold_row_ids=_frozen_gold_row_ids(),
        )


def _dynamic_candidate_plan(root: Path) -> dict[str, Any]:
    stage1_seed = root / "stage1-seed.yaml"
    stage1_seed.write_text(
        yaml.safe_dump(
            scanner_owned_pyramid_stage1_manifest(_stage1_manifest()),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return execution.build_pyramid_candidate_plan(
        execution.load_stage2_search_space(stage1_seed)
    )


def test_scanner_owned_seed_builder_deep_copies_nested_manifest() -> None:
    original = _stage1_manifest()
    migrated = scanner_owned_pyramid_stage1_manifest(original)

    migrated["view_b1_search_groups"][0]["widths"].append(96)
    original["view_b2_quant_units"][0]["legal_bits"].append("FP8")

    assert original["view_b1_search_groups"][0]["widths"] == [128]
    assert migrated["view_b2_quant_units"][0]["legal_bits"] == ["FP16", "INT8"]


def _projected_dynamic_request(
    public_contract: Path, registry: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    contract = load_public_contract(public_contract)
    task = execution._build_search_task(contract, _profile())
    manifest = build_task_candidate_manifest(
        registry,
        task=task,
        measured_row_ids=set(),
    )
    selected_rows: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    for row in manifest["rows"]:
        group_id = str(row["group_id"])
        if group_id not in selected_groups:
            selected_groups.add(group_id)
            selected_rows.append(dict(row))
        if len(selected_rows) == 4:
            break
    request = build_measurement_request(
        task=task,
        selected_rows=selected_rows,
        round_index=0,
    )
    projected = dict(project_source_materialization_request(request).request)
    return projected, len(manifest["rows"])


def _frozen_gold_row_ids() -> frozenset[str]:
    gold_rows, _graphs = _gold176(include_non_target_backend=True)
    frozen = freeze_initial_coldstart(gold_rows)
    return frozenset(str(row["row_id"]) for row in frozen)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _write_synthetic_real_stage1_launcher(
    root: Path,
    *,
    schema_key: str = "schema",
    hardware_via_symlink: bool = False,
    mapper_accepts_scenario: bool = True,
    mapper_splits_scenario_contract: bool = False,
    mapper_uses_foreign_scenario: bool = False,
    mapper_imports_then_shadows_scan: bool = False,
    scenario_path: Path | None = None,
) -> Path:
    private_bin = root / "private-runner" / "bin"
    private_bin.mkdir(parents=True, exist_ok=True)
    mapper = private_bin / "stage1-map-real-private.py"
    scenario_argument = (
        'parser.add_argument("--scenario", required=True)'
        if mapper_accepts_scenario
        else ""
    )
    shadow_scan = (
        "def real_scan(*args, **kwargs):\n"
        "    del args, kwargs\n"
        "    raise SystemExit(\"shadow scanner called\")\n"
        if mapper_imports_then_shadows_scan
        else ""
    )
    build_call = (
        "build_manifest(\n"
        "    Path(args.output), Path(args.hardware), args.device,\n"
        "    {{\"STAGE1_REPO_ROOT\": args.stage1_repo_root,\n"
        "     \"HEAL_ROOT\": args.heal_root,\n"
        "     \"HEAL_CKPT_ROOT\": args.heal_checkpoint_root}},\n"
        "    real_scan,\n"
        "    scenario_path={scenario},\n"
        ")"
    )
    if mapper_uses_foreign_scenario:
        mapper_invocation = (
            "class StaticScenario:\n"
            "    scenario = expected_scenario\n\n"
            + build_call.format(scenario="StaticScenario.scenario")
        )
    elif mapper_splits_scenario_contract:
        mapper_invocation = (
            "consume_scenario(scenario_path=Path(args.scenario))\n"
            + build_call.format(scenario="expected_scenario")
        )
    else:
        mapper_invocation = build_call.format(scenario="Path(args.scenario)")
    mapper.write_text(
        f"""#!{sys.executable}
import argparse
from pathlib import Path
from framework.stage6.p6_stage1_bridge_v1 import (
    build_p6_stage1_partition_manifest as build_manifest,
    run_real_stage1_scan as real_scan,
)

class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\\n")


parser = _ArgumentParser()
parser.add_argument("--hardware", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", required=True)
{scenario_argument}
parser.add_argument("--stage1-repo-root", required=True)
parser.add_argument("--heal-root", required=True)
parser.add_argument("--heal-checkpoint-root", required=True)
args = parser.parse_args()
expected_scenario = (
    Path(__file__).resolve().parents[2]
    / "public-code"
    / "configs"
    / "stage1"
    / "p6_h800_formal_scan.yaml"
)
if args.device != "cuda:0":
    raise SystemExit("invalid device")
if Path(args.scenario) != expected_scenario:
    raise SystemExit("invalid scenario path")
if expected_scenario.read_text(encoding="utf-8") != "scenario-version: original\\n":
    raise SystemExit("invalid scenario content")


def consume_scenario(*, scenario_path: Path) -> None:
    del scenario_path


{shadow_scan}
{mapper_invocation}
""",
        encoding="utf-8",
    )
    mapper.chmod(0o700)
    public_code = root / "public-code"
    if hardware_via_symlink:
        hardware_root = root / "outside-hardware"
        hardware = hardware_root / "hardware" / "h800.yaml"
        hardware.parent.mkdir(parents=True, exist_ok=True)
        public_code.mkdir()
        (public_code / "configs").symlink_to(hardware_root, target_is_directory=True)
    else:
        hardware = public_code / "configs" / "hardware" / "h800.yaml"
        hardware.parent.mkdir(parents=True, exist_ok=True)
    hardware.write_text("name: NVIDIA H800\n", encoding="utf-8")
    tracked_scenario = (
        public_code / "configs" / "stage1" / "p6_h800_formal_scan.yaml"
    )
    tracked_scenario.parent.mkdir(parents=True, exist_ok=True)
    tracked_scenario.write_text("scenario-version: original\n", encoding="utf-8")
    bridge = public_code / "framework" / "stage6" / "p6_stage1_bridge_v1.py"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    (public_code / "framework" / "__init__.py").write_text("", encoding="utf-8")
    (bridge.parent / "__init__.py").write_text("", encoding="utf-8")
    bridge.write_text(
        f"""from pathlib import Path
import json


def run_real_stage1_scan(
    model_name, hardware_path, device, environment, scenario_path,
):
    del hardware_path, environment
    expected = (
        Path(__file__).resolve().parents[2]
        / "configs" / "stage1" / "p6_h800_formal_scan.yaml"
    )
    if model_name != "pyramid_lidar" or device != "cuda:0":
        raise RuntimeError("invalid scan request")
    if scenario_path != expected or scenario_path.read_text() != (
        "scenario-version: original\\n"
    ):
        raise RuntimeError("formal scenario not consumed")
    return {{
        {schema_key!r}: "stage1_partition_manifest_v1",
        "stage": "stage1_partition",
        "model": "pyramid_lidar",
        "scan_status": "ok",
    }}


def build_p6_stage1_partition_manifest(
    output_path, hardware_path, device, environment, scanner, *, scenario_path,
):
    payload = scanner(
        "pyramid_lidar", hardware_path, device, environment, scenario_path,
    )
    Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
""",
        encoding="utf-8",
    )
    (root / "dependency-overlay").mkdir(exist_ok=True)
    for name in ("heal", "checkpoints"):
        (root / name).mkdir(exist_ok=True)
    return render_stage1_launcher(
        output_path=private_bin / "stage1-launch-real-private.sh",
        tooling_python=Path(sys.executable).resolve(strict=True),
        heal_root=root / "heal",
        heal_checkpoint_root=root / "checkpoints",
        scenario_path=scenario_path or tracked_scenario,
    )


def _regenerated_deployment_fixture(root: Path) -> _RegeneratedDeploymentFixture:
    source_root = root / "source"
    source_map, source_runner = _as_v2_procedural(
        valid_private_source_map(source_root), source_root
    )
    external = source_map["external_training_binding"]
    synthetic_caps = (128, 128, 128)
    external["training_parameters"]["base_stage_widths"] = list(synthetic_caps)
    dataset_file = Path(external["dataset_root"]) / "fixture-record.bin"
    dataset_file.write_bytes(b"SYNTHETIC-EXTERNAL-DATASET-RECORD-UNIQUE")
    Path(external["base_checkpoint_path"]).write_bytes(
        b"SYNTHETIC-EXTERNAL-CHECKPOINT-UNIQUE"
    )
    Path(external["pyramid_config_path"]).write_text(
        "model:\n  args:\n    fusion_backbone:\n"
        f"      num_filters: {list(synthetic_caps)!r}\n",
        encoding="utf-8",
    )
    source_map_path = root / "ignored-source-map.yaml"
    source_map_path.write_text(
        yaml.safe_dump(source_map, sort_keys=False), encoding="utf-8"
    )
    return _RegeneratedDeploymentFixture(
        root=root,
        source_map=source_map_path,
        source_history_root=Path(source_map["history_root"]),
        source_runner=source_runner,
        derivation_root=root / "derivation",
        normalized_root=root / "normalized",
        local_output_root=root / "validation-output",
        public_contract=_write_yaml(root / "public.yaml", _public_contract()),
        provision_probe=_OfflineGpuProbe(),
        rejecting_gpu_probe=_RejectingGpuProbe(),
        fail_if_called_runner=_FailIfCalledRunner(),
    )


def _preflight(fixture: _RegeneratedDeploymentFixture) -> None:
    preflight_materializer_training_bridge(
        public_contract_path=fixture.public_contract,
        local_config_path=fixture.local_config_path,
        private_binding_path=fixture.binding_path,
        runner_template_path=fixture.normalized_runner,
        source_wrapper_profile_path=fixture.normalized_wrapper_profile,
        external_training_binding_path=fixture.normalized_external_binding,
    )


def _assert_mocked_boundary_untouched(fixture: _RegeneratedDeploymentFixture) -> None:
    assert fixture.rejecting_gpu_probe.calls == 0
    assert fixture.fail_if_called_runner.calls == []


def _external_asset_paths(fixture: _RegeneratedDeploymentFixture) -> tuple[Path, ...]:
    external = yaml.safe_load(
        fixture.normalized_external_binding.read_text(encoding="utf-8")
    )
    dataset_root = Path(external["dataset_root"])
    return (
        *tuple(path for path in dataset_root.rglob("*") if path.is_file()),
        Path(external["base_checkpoint_path"]),
        Path(external["pyramid_config_path"]),
    )


def _assert_external_assets_not_copied(fixture: _RegeneratedDeploymentFixture) -> None:
    external_assets = _external_asset_paths(fixture)
    external_inodes = {
        (path.stat().st_dev, path.stat().st_ino) for path in external_assets
    }
    external_digests = {
        hashlib.sha256(path.read_bytes()).hexdigest() for path in external_assets
    }
    private_files = tuple(
        path
        for root in (fixture.normalized_root, fixture.local_output_root)
        for path in root.rglob("*")
        if path.is_file()
    )
    assert external_inodes.isdisjoint(
        (path.stat().st_dev, path.stat().st_ino) for path in private_files
    )
    assert external_digests.isdisjoint(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in private_files
    )


def test_regenerated_external_deployment_reaches_mocked_gpu_boundary_without_process(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    binding, local_config = fixture.derive_normalize_and_provision()
    report = preflight_materializer_training_bridge(
        public_contract_path=fixture.public_contract,
        local_config_path=local_config,
        private_binding_path=binding,
        runner_template_path=fixture.normalized_runner,
        source_wrapper_profile_path=fixture.normalized_wrapper_profile,
        external_training_binding_path=fixture.normalized_external_binding,
    )

    assert report.historical_process_launch_count == 0
    assert report.gpu_probe_count == 0
    assert report.validated_round_count == 4
    _assert_external_assets_not_copied(fixture)
    boundary = fixture.prepare_mocked_measurement()
    assert boundary.candidate_count == boundary.candidate_rows
    assert boundary.candidate_count not in {343, 686}
    selected_ids = {
        str(row["row_id"]) for row in boundary.projected_request["rows"]
    }
    assert len(selected_ids) == 4
    assert selected_ids.isdisjoint(boundary.frozen_gold_row_ids)
    with pytest.raises(P6HistoryMeasurementError) as captured:
        run_history_measurement_batch(
            boundary.projected_request,
            boundary.binding_mapping,
            boundary.public_round_root,
            fixture.fail_if_called_runner,
            fixture.rejecting_gpu_probe,
        )
    assert captured.value.category == "history_gpu_admission_failed"
    assert fixture.rejecting_gpu_probe.calls == 1
    assert fixture.fail_if_called_runner.calls == []


def test_null_only_public_example_stops_before_mocked_gpu_boundary(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    fixture.derive_normalize_and_provision()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        preflight_materializer_training_bridge(
            public_contract_path=fixture.public_contract,
            local_config_path=fixture.local_config_path,
            private_binding_path=fixture.binding_path,
            runner_template_path=fixture.normalized_runner,
            source_wrapper_profile_path=fixture.normalized_wrapper_profile,
            external_training_binding_path=(
                REPOSITORY_ROOT
                / "configs/execution/p6_external_training_binding.example.yaml"
            ),
        )

    assert captured.value.failure_code == "history_execution_invalid"
    _assert_mocked_boundary_untouched(fixture)


def test_external_digest_drift_stops_before_mocked_gpu_boundary(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    fixture.derive_normalize_and_provision()
    external = yaml.safe_load(
        fixture.normalized_external_binding.read_text(encoding="utf-8")
    )
    Path(external["base_checkpoint_path"]).write_bytes(b"digest drift")

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        _preflight(fixture)

    assert captured.value.failure_code == "history_execution_invalid"
    _assert_mocked_boundary_untouched(fixture)


def test_missing_execution_closure_role_stops_before_provision_or_gpu(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    source_map = yaml.safe_load(fixture.source_map.read_text(encoding="utf-8"))
    source_map["execution_code_closure"]["roles"].pop("ap")
    fixture.source_map.write_text(
        yaml.safe_dump(source_map, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(P6HistoryNormalizationError):
        fixture.derive_normalize_and_provision()

    assert fixture.provision_probe.calls == []
    _assert_mocked_boundary_untouched(fixture)


def test_copied_training_asset_stops_before_mocked_gpu_boundary(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    fixture.derive_normalize_and_provision()
    external = yaml.safe_load(
        fixture.normalized_external_binding.read_text(encoding="utf-8")
    )
    copied_checkpoint = fixture.normalized_root / "copied-training.ckpt"
    shutil.copy2(Path(external["base_checkpoint_path"]), copied_checkpoint)
    external["base_checkpoint_path"] = str(copied_checkpoint)
    external["base_checkpoint_sha256"] = hashlib.sha256(
        copied_checkpoint.read_bytes()
    ).hexdigest()
    fixture.normalized_external_binding.write_text(
        yaml.safe_dump(external, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        _preflight(fixture)

    assert captured.value.failure_code == "history_execution_invalid"
    _assert_mocked_boundary_untouched(fixture)


def test_preexisting_output_leaf_stops_before_mocked_gpu_boundary(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    fixture.derive_normalize_and_provision()
    (fixture.local_output_root / "materialized").mkdir()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        _preflight(fixture)

    assert captured.value.failure_code == "unsafe_destination"
    _assert_mocked_boundary_untouched(fixture)


def test_private_token_is_redacted_before_mocked_gpu_boundary(
    tmp_path: Path,
) -> None:
    fixture = _regenerated_deployment_fixture(tmp_path)
    fixture.derive_normalize_and_provision()
    external = yaml.safe_load(
        fixture.normalized_external_binding.read_text(encoding="utf-8")
    )
    external["dataset_root"] = str(tmp_path / PRIVATE_TOKEN)
    fixture.normalized_external_binding.write_text(
        yaml.safe_dump(external, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        _preflight(fixture)

    assert captured.value.failure_code == "history_execution_invalid"
    assert PRIVATE_TOKEN not in str(captured.value)
    _assert_mocked_boundary_untouched(fixture)


def test_real_stage1_launcher_accepts_mapper_schema_without_external_execution(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture"
    source_map, runner_path = _as_v2_procedural(
        valid_private_source_map(fixture_root), fixture_root
    )
    history_root = Path(source_map["history_root"])
    launcher = _write_synthetic_real_stage1_launcher(history_root)
    runner_payload = yaml.safe_load(runner_path.read_text(encoding="utf-8"))
    runner_payload["stage1_scan"]["argv"][0] = str(launcher)
    runner_path.write_text(
        yaml.safe_dump(runner_payload, sort_keys=False), encoding="utf-8"
    )
    validated = validate_pre_provision_runner_template(
        runner_path,
        history_root,
        require_exact_history_environment=True,
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest = output_root / "stage1_partition_manifest.json"
    rendered_argv = [
        validated.stage1_argv[0],
        str(manifest),
        str(output_root),
    ]

    completed = subprocess.run(
        rendered_argv,
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert validated.stage1_argv[1:] == (
        "{stage1_partition_manifest}",
        "{local_output_root}",
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "schema": "stage1_partition_manifest_v1",
        "stage": "stage1_partition",
        "model": "pyramid_lidar",
        "scan_status": "ok",
    }


def test_stage1_launcher_uses_tracked_public_code_scenario(
    tmp_path: Path,
) -> None:
    launcher = _write_synthetic_real_stage1_launcher(tmp_path / "history")
    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest = output_root / "stage1_partition_manifest.json"

    completed = subprocess.run(
        [str(launcher), str(manifest), str(output_root)],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    tracked_scenario = (
        tmp_path
        / "history"
        / "public-code"
        / "configs"
        / "stage1"
        / "p6_h800_formal_scan.yaml"
    )
    assert tracked_scenario.read_text(encoding="utf-8") == "scenario-version: original\n"
    assert completed.returncode == 0, completed.stderr


def test_stage1_launcher_rejects_mapper_without_scenario_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(
            tmp_path / "history",
            mapper_accepts_scenario=False,
        )

    assert not (
        tmp_path
        / "history"
        / "private-runner"
        / "bin"
        / "stage1-launch-real-private.sh"
    ).exists()


def test_stage1_launcher_rejects_scenario_consumed_by_unrelated_call(
    tmp_path: Path,
) -> None:
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(
            tmp_path / "history",
            mapper_splits_scenario_contract=True,
        )


def test_stage1_launcher_rejects_foreign_scenario_attribute(
    tmp_path: Path,
) -> None:
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(
            tmp_path / "history",
            mapper_uses_foreign_scenario=True,
        )


def test_stage1_launcher_rejects_locally_shadowed_real_scan_import(
    tmp_path: Path,
) -> None:
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(
            tmp_path / "history",
            mapper_imports_then_shadows_scan=True,
        )


def test_stage1_launcher_rolls_back_launcher_when_temporary_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_root = tmp_path / "history"
    real_mkstemp = stage1_launcher_renderer.tempfile.mkstemp
    calls = 0

    def fail_launcher_temporary(*args: Any, **kwargs: Any) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic launcher write failure")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(
        stage1_launcher_renderer.tempfile, "mkstemp", fail_launcher_temporary
    )
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(history_root)

    private_runner = history_root / "private-runner"
    assert not (private_runner / "bin" / "stage1-launch-real-private.sh").exists()
    assert not tuple(private_runner.rglob("*.tmp"))
    assert (
        history_root
        / "public-code"
        / "configs"
        / "stage1"
        / "p6_h800_formal_scan.yaml"
    ).is_file()

    monkeypatch.setattr(stage1_launcher_renderer.tempfile, "mkstemp", real_mkstemp)
    assert _write_synthetic_real_stage1_launcher(history_root).is_file()


def test_stage1_launcher_rejects_scenario_outside_public_code_closure(
    tmp_path: Path,
) -> None:
    external_scenario = tmp_path / "external-scenario.yaml"
    external_scenario.write_text("scenario-version: original\n", encoding="utf-8")

    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(
            tmp_path / "history", scenario_path=external_scenario
        )


def test_stage1_launcher_renderer_cli_reports_stable_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert render_stage1_launcher_main(
        [
            "--output",
            "relative-output",
            "--tooling-python",
            str(tmp_path / "python"),
            "--heal-root",
            str(tmp_path / "heal"),
            "--heal-checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--scenario",
            str(tmp_path / "scenario.yaml"),
        ]
    ) == 1
    assert "stage1_launcher_render_invalid" in capsys.readouterr().err

    monkeypatch.setattr(
        stage1_launcher_renderer,
        "render_stage1_launcher",
        lambda **_kwargs: tmp_path / "launcher.sh",
    )
    assert render_stage1_launcher_main(
        [
            "--output",
            str(tmp_path / "output"),
            "--tooling-python",
            str(tmp_path / "python"),
            "--heal-root",
            str(tmp_path / "heal"),
            "--heal-checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--scenario",
            str(tmp_path / "scenario.yaml"),
        ]
    ) == 0
    assert capsys.readouterr().out == "p6_stage1_launcher_rendered\n"


def test_stage1_launcher_never_overwrites_destination_created_after_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_root = tmp_path / "history"
    private_bin = history_root / "private-runner" / "bin"
    destination = private_bin / "stage1-launch-real-private.sh"
    real_mkstemp = stage1_launcher_renderer.tempfile.mkstemp
    calls = 0

    def create_destination_after_planning(
        *args: Any, **kwargs: Any
    ) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        result = real_mkstemp(*args, **kwargs)
        if calls == 1:
            destination.write_text("attacker-owned\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        stage1_launcher_renderer.tempfile,
        "mkstemp",
        create_destination_after_planning,
    )
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(history_root)

    assert destination.read_text(encoding="utf-8") == "attacker-owned\n"
    assert not (history_root / "private-runner" / "stage1-scenario.yaml").exists()


def test_stage1_launcher_removes_linked_destination_when_publication_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_root = tmp_path / "history"
    destination = (
        history_root / "private-runner" / "bin" / "stage1-launch-real-private.sh"
    )
    real_open = stage1_launcher_renderer.os.open

    def fail_linked_destination_open(
        path: str | bytes | os.PathLike[str], *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == destination:
            raise OSError("synthetic publication open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(stage1_launcher_renderer.os, "open", fail_linked_destination_open)
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(history_root)

    assert not destination.exists()
    monkeypatch.setattr(stage1_launcher_renderer.os, "open", real_open)
    assert _write_synthetic_real_stage1_launcher(history_root).is_file()


def test_stage1_launcher_closes_mismatched_publication_fd_and_removes_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_root = tmp_path / "history"
    destination = (
        history_root / "private-runner" / "bin" / "stage1-launch-real-private.sh"
    )
    replacement = tmp_path / "unrelated-publication-target"
    replacement.write_text("unrelated\n", encoding="utf-8")
    real_open = stage1_launcher_renderer.os.open

    def open_unrelated_file(
        path: str | bytes | os.PathLike[str], *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == destination:
            return real_open(replacement, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    before_descriptors = len(tuple(Path("/proc/self/fd").iterdir()))
    monkeypatch.setattr(stage1_launcher_renderer.os, "open", open_unrelated_file)
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(history_root)
    after_descriptors = len(tuple(Path("/proc/self/fd").iterdir()))

    assert after_descriptors == before_descriptors
    assert not destination.exists()
    monkeypatch.setattr(stage1_launcher_renderer.os, "open", real_open)
    assert _write_synthetic_real_stage1_launcher(history_root).is_file()


def test_real_stage1_launcher_rejects_schema_version_when_python_is_optimized(
    tmp_path: Path,
) -> None:
    launcher = _write_synthetic_real_stage1_launcher(
        tmp_path / "history", schema_key="schema_version"
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest = output_root / "stage1_partition_manifest.json"

    completed = subprocess.run(
        [str(launcher), str(manifest), str(output_root)],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "PYTHONOPTIMIZE": "1"},
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0


def test_real_stage1_launcher_rejects_manifest_outside_local_output_root(
    tmp_path: Path,
) -> None:
    launcher = _write_synthetic_real_stage1_launcher(tmp_path / "history")
    output_root = tmp_path / "output"
    output_root.mkdir()
    outside_manifest = output_root / ".." / "outside.json"

    completed = subprocess.run(
        [str(launcher), str(outside_manifest), str(output_root)],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "outside.json").exists()


def test_stage1_launcher_renderer_rejects_symlinked_hardware_component(
    tmp_path: Path,
) -> None:
    with pytest.raises(P6Stage1LauncherRenderError):
        _write_synthetic_real_stage1_launcher(
            tmp_path / "history", hardware_via_symlink=True
        )
