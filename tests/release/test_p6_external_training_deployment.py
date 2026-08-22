"""Black-box release gate for regenerated external P6 training deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
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
from framework.stage6.p6_source_reuse_evidence_v1 import create_fresh_run_context
from tests.release.p6_source_reuse_lifecycle_fixture import _stage1_manifest
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
        yaml.safe_dump(_stage1_manifest(), sort_keys=False), encoding="utf-8"
    )
    return execution.build_pyramid_candidate_plan(
        execution.load_stage2_search_space(stage1_seed)
    )


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


def _regenerated_deployment_fixture(root: Path) -> _RegeneratedDeploymentFixture:
    source_root = root / "source"
    source_map, source_runner = _as_v2_procedural(
        valid_private_source_map(source_root), source_root
    )
    external = source_map["external_training_binding"]
    dataset_file = Path(external["dataset_root"]) / "fixture-record.bin"
    dataset_file.write_bytes(b"SYNTHETIC-EXTERNAL-DATASET-RECORD-UNIQUE")
    Path(external["base_checkpoint_path"]).write_bytes(
        b"SYNTHETIC-EXTERNAL-CHECKPOINT-UNIQUE"
    )
    Path(external["pyramid_config_path"]).write_bytes(
        b"SYNTHETIC-EXTERNAL-PYRAMID-CONFIG-UNIQUE"
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
