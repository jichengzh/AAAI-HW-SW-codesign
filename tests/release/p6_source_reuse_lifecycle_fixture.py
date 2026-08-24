"""Injected boundaries for the P6 dynamic source-reuse release lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pytest
import yaml

from framework.stage5.single_target_search_v2 import validate_search_task
from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6 import p6_source_reuse_evidence_v1 as reuse_evidence
from framework.stage6.coptv2x_h800_search_v2 import (
    LocalExecutionStep,
    LocalP6CoptV2XConfig,
    PublicP6CoptV2XContract,
    RegisteredAsset,
)
from framework.stage6.p6_history_measurement_v1 import run_history_measurement_batch
from framework.stage6.p6_source_reuse_evidence_v1 import (
    SOURCE_ARTIFACT_KEYS,
    SOURCE_MARKER_KEYS,
)
from tests.release.scanner_owned_stage1_fixture import (
    scanner_owned_pyramid_stage1_manifest,
)
from tests.stage6.test_coptv2x_h800_search import (
    _closure,
    _gold176,
    _non_target_profile,
    _profile,
    _write_recipe_v2_training_registry_from_plan,
)
from tests.stage6.test_p6_history_measurement import FakeProbe, _binding


FILE_KEYS = frozenset(
    {
        "checkpoint_path",
        "config_path",
        "onnx_path",
        "onnx_report_path",
        "calibration_npz",
        "calibration_summary",
    }
)
DIRECTORY_KEYS = frozenset(
    {"checkpoint_dir", "calibration_root", "trt_calibration_dir"}
)
MARKER_KEYS = frozenset(SOURCE_MARKER_KEYS)
if FILE_KEYS | DIRECTORY_KEYS != frozenset(SOURCE_ARTIFACT_KEYS):
    raise AssertionError("fixture source bundle keys drifted")


@dataclass(frozen=True)
class OfflineReuseLifecycle:
    call_kwargs: Mapping[str, Any]
    local_output_root: Path
    private_root: Path
    gold176_row_ids: frozenset[str]
    selected_rows_by_round: tuple[tuple[tuple[str, str], ...], ...]
    process_records: list[Mapping[str, Any]]
    source_calls: list[Mapping[str, Any]]
    downstream_rows: dict[str, list[tuple[str, ...]]]
    binding: Mapping[str, Any]
    task_sha256: str
    fit_input_counts: list[int]
    receipt_bytes_before_reuse: Mapping[str, bytes]


@dataclass(frozen=True)
class _Result:
    returncode: int = 0


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _stage1_manifest() -> dict[str, Any]:
    return {
        "schema": "stage1_partition_manifest_v1",
        "stage": "stage1_partition",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": "h800"},
        "view_b1_search_groups": [
            {
                "search_group_id": f"pyramid_group.{suffix}",
                "bucket": "pyramid_backbone",
                "widths": [128],
                "round_to": 32,
                "int8_buildable_align": 32,
                "max_rate": 0.5,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": [f"pyramid_group.{suffix}"],
            }
            for suffix in ("s0", "s1", "s2")
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": [
                    "pyramid_group.s0",
                    "pyramid_group.s1",
                    "pyramid_group.s2",
                ],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _selected_schedule(plan: Mapping[str, Any]) -> tuple[tuple[tuple[str, str], ...], ...]:
    groups_by_width: dict[tuple[int, ...], str] = {}
    for candidate in plan["candidates"]:
        width = tuple(candidate["width"])
        groups_by_width[width] = "pyramid|" + "x".join(str(item) for item in width)
    groups = tuple(groups_by_width[width] for width in sorted(groups_by_width))
    if len(groups) < 14:
        raise AssertionError("fixture requires fourteen dynamic Stage2 groups")
    return (
        ((groups[0], "fp16"), (groups[1], "int8"), (groups[2], "fp16"), (groups[3], "int8")),
        ((groups[4], "fp16"), (groups[4], "int8"), (groups[5], "fp16"), (groups[6], "int8")),
        ((groups[0], "int8"), (groups[7], "fp16"), (groups[8], "int8"), (groups[9], "fp16")),
        ((groups[10], "fp16"), (groups[11], "int8"), (groups[12], "fp16"), (groups[13], "int8")),
    )


class _OfflineAdapterRunner:
    """A no-subprocess private boundary retaining real adapter semantics."""

    def __init__(
        self,
        *,
        process_records: list[Mapping[str, Any]],
        source_calls: list[Mapping[str, Any]],
        downstream_rows: dict[str, list[tuple[str, ...]]],
    ) -> None:
        self._process_records = process_records
        self._source_calls = source_calls
        self._downstream_rows = downstream_rows

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        name = Path(argv[0]).name
        self._process_records.append(
            MappingProxyType(
                {"kind": "injected_boundary", "stage": name, "launched": False}
            )
        )
        if shell:
            raise AssertionError("the adapter must use direct argv")
        if name == "stage5_materialize_round_sources_v1.sh":
            self._write_source_outputs(argv, cwd, env)
        elif name in {"quantize", "stage5_build_performance_plan_v2.py", "measure-ap"}:
            stage = {
                "quantize": "quantization",
                "stage5_build_performance_plan_v2.py": "performance",
                "measure-ap": "ap",
            }[name]
            self._record_downstream(stage, cwd)
        elif name == "stage5_finalize_feedback_v2.py":
            self._record_downstream("finalization", cwd)
            self._write_finalization(argv)
        elif name != "activate":
            raise AssertionError(f"unexpected private argv: {argv!r}")
        return _Result()

    def _request(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_source_outputs(
        self, argv: Sequence[str], cwd: Path, env: Mapping[str, str]
    ) -> None:
        request_path = Path(argv[2])
        request = self._request(request_path)
        group_id = argv[6]
        rows = [row for row in request["rows"] if row["group_id"] == group_id]
        if not rows:
            raise AssertionError("source invocation must name one requested group")
        contract = rows[0]["source_contract"]
        self._source_calls.append(
            MappingProxyType(
                {
                    "group_id": group_id,
                    "round": request["round_index"],
                    "argv": tuple(argv),
                    "env": MappingProxyType(dict(env)),
                }
            )
        )
        if set(env) != {
            "CUDA_VISIBLE_DEVICES",
            "P6_HISTORY_RUN_MODE",
            "P6_HISTORY_PRIVATE_ROOT",
            "P6_HISTORY_TASK_STATE",
            "P6_HISTORY_ROUND_OUTPUT_ROOT",
        }:
            raise AssertionError("adapter environment drift")
        for key in ("trt_calibration_dir", "calibration_root", "checkpoint_dir"):
            directory = Path(contract[key])
            directory.mkdir(parents=True)
            (directory / "z-created-first.bin").write_bytes(
                f"{group_id}:{key}:z\n".encode("ascii")
            )
            (directory / "a-created-second.bin").write_bytes(
                f"{group_id}:{key}:a\n".encode("ascii")
            )
        for key in FILE_KEYS:
            path = Path(contract[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{group_id}:{key}\n".encode("ascii"))
        marker_payloads = {
            "training_done_marker": b"trained\n",
            "source_done_marker": b"source-ready\n",
        }
        for key in ("training_done_marker", "source_done_marker"):
            if key not in MARKER_KEYS:
                raise AssertionError("source marker key drift")
            path = Path(contract[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(marker_payloads[key])

    def _record_downstream(self, stage: str, cwd: Path) -> None:
        request = self._request(cwd / "measurement-request.json")
        self._downstream_rows[stage].append(
            tuple(row["row_id"] for row in request["rows"])
        )

    def _write_finalization(self, argv: Sequence[str]) -> None:
        request_path, state_path, result_path, receipt_path, barrier_path = map(
            Path, argv[1:6]
        )
        request = self._request(request_path)
        row_hashes = request["row_sha256"]
        evidence = {
            row["row_id"]: row["source_evidence_sha256"] for row in request["rows"]
        }
        state_rows = [
            {
                "row_id": row["row_id"],
                "row_sha256": row_hashes[row["row_id"]],
                "source_evidence_sha256": row["source_evidence_sha256"],
                "terminal_status": "measured_success_gold",
            }
            for row in request["rows"]
        ]
        validation = {
            "measurement_request_sha256": request["measurement_request_sha256"],
            "row_sha256": row_hashes,
            "source_evidence_sha256": evidence,
        }
        outputs: tuple[tuple[Path, Mapping[str, Any]], ...] = (
            (state_path, {"stage": "finalization", "rows": state_rows}),
            (
                result_path,
                {
                    "measurement_request_sha256": request["measurement_request_sha256"],
                    "rows": [
                        {
                            **row,
                            "latency_ms": 2.0 + index,
                            "energy_j": 0.5 + index / 10,
                            "ap30": 0.91,
                            "ap50": 0.82,
                            "ap70": 0.73,
                        }
                        for index, row in enumerate(state_rows)
                    ],
                },
            ),
            (receipt_path, validation),
            (barrier_path, validation),
        )
        for path, payload in outputs:
            _write_json(path, payload)


def build_offline_reuse_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> OfflineReuseLifecycle:
    """Build a fresh real-controller lifecycle with every external boundary injected."""
    local_output_root = tmp_path / "offline-public"
    private_root = tmp_path / "PRIVATE-OFFLINE-REUSE-TOKEN"
    private_root.mkdir()
    gold_rows, gold_graphs = _gold176(include_non_target_backend=True)
    local_inputs = {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [_profile(), _non_target_profile()],
        "closure": _closure(),
    }
    for name, payload in local_inputs.items():
        _write_json(tmp_path / f"{name}.json", payload)
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()
    contract = PublicP6CoptV2XContract(
        search_id="p6-offline-reuse",
        target="h800",
        target_model="pyramid",
        execution_backend="tvm_auto",
        seed=73,
        sample_budget=16,
        batch_size=4,
        round_count=4,
        configuration_label="p6-offline-reuse",
        candidate_space_label="dynamic-stage2-fixture",
        assets=(
            RegisteredAsset("training-data", "v1", "cleared"),
            RegisteredAsset("model-init", "v2", "cleared"),
            RegisteredAsset("toolchain", "v3", "cleared"),
        ),
        metric_names=("latency_ms", "energy_j", "ap30", "ap50", "ap70"),
    )
    local = LocalP6CoptV2XConfig(
        asset_paths=MappingProxyType(
            {label: tmp_path / label for label in ("training-data", "model-init", "toolchain")}
        ),
        local_input_paths=MappingProxyType(
            {name: tmp_path / f"{name}.json" for name in local_inputs}
        ),
        candidate_source_mode="framework_stage2_search_space",
        stage2_search_space_path=local_output_root / "stage1-partition.yaml",
        stage1_scan_step=LocalExecutionStep(
            "offline_stage1", ("offline-stage1", "{stage1_partition_manifest}")
        ),
        source_registry_step=LocalExecutionStep(
            "offline_registry",
            ("offline-registry", "{source_registry_json}", "{pyramid_candidate_plan}"),
        ),
        measurement_step=LocalExecutionStep(
            "offline_measurement",
            ("offline-measurement", "{measurement_request}", "{feedback_json}", "{round_output_root}"),
        ),
        local_output_root=local_output_root,
    )
    process_records: list[Mapping[str, Any]] = []
    source_calls: list[Mapping[str, Any]] = []
    downstream_rows = {stage: [] for stage in ("quantization", "performance", "ap", "finalization")}
    binding = _binding(private_root)
    adapter_runner = _OfflineAdapterRunner(
        process_records=process_records,
        source_calls=source_calls,
        downstream_rows=downstream_rows,
    )
    selected_rows_by_round: list[tuple[tuple[str, str], ...]] = []
    fit_input_counts: list[int] = []
    receipt_bytes_before_reuse: dict[str, bytes] = {}
    task_sha256 = validate_search_task(execution._build_search_task(contract, _profile()))["task_sha256"]

    real_create_context = execution.create_fresh_run_context

    def record_context(**kwargs: Any) -> Any:
        process_records.append(
            MappingProxyType({"kind": "injected_boundary", "event": "context_publish", "launched": False})
        )
        return real_create_context(**kwargs)

    def initial_fit(rows: Sequence[Mapping[str, Any]], *_args: Any, **_kwargs: Any) -> object:
        fit_input_counts.append(len(rows))
        return object()

    def online_fit(rows: Sequence[Mapping[str, Any]], *_args: Any, **_kwargs: Any) -> object:
        fit_input_counts.append(len(rows))
        return object()

    def select_rows(predicted: Sequence[Mapping[str, Any]], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        round_index = len(selected_rows_by_round)
        by_identity = {(row["group_id"], row["q_mode"]): row for row in predicted}
        scheduled = schedule[round_index]
        selected = [dict(by_identity[identity]) for identity in scheduled]
        selected_rows_by_round.append(scheduled)
        return {"selected_rows": selected}

    def controller_runner(argv: tuple[str, ...], cwd: Path) -> int:
        command = argv[0]
        process_records.append(
            MappingProxyType({"kind": "injected_boundary", "stage": command, "launched": False})
        )
        if command == "offline-stage1":
            path = Path(argv[1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(
                    scanner_owned_pyramid_stage1_manifest(_stage1_manifest()),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        elif command == "offline-registry":
            registry_path, plan_path = map(Path, argv[1:3])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_recipe_v2_training_registry_from_plan(registry_path, plan, local_output_root)
        elif command == "offline-measurement":
            request_path, feedback_path, public_round_root = map(Path, argv[1:4])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            feedback = run_history_measurement_batch(
                request, binding, public_round_root, adapter_runner, FakeProbe()
            )
            _write_json(feedback_path, feedback)
            if request["round_index"] == 0:
                repeated_group = schedule[0][0][0]
                receipt_path = reuse_evidence.receipt_path_for_group(
                    reuse_evidence.plan_source_reuse_paths(local_output_root), repeated_group
                )
                receipt_bytes_before_reuse[repeated_group] = receipt_path.read_bytes()
        else:
            raise AssertionError(f"unexpected controller argv: {argv!r} from {cwd}")
        return 0

    seed_plan = execution.build_pyramid_candidate_plan(
        execution.load_stage2_search_space(_write_stage1_seed(tmp_path))
    )
    schedule = _selected_schedule(seed_plan)
    monkeypatch.setattr(execution, "create_fresh_run_context", record_context)
    monkeypatch.setattr(execution, "fit_initial_coldstart_bundle", initial_fit)
    monkeypatch.setattr(execution, "fit_online_bundle", online_fit)
    monkeypatch.setattr(execution, "predict_candidate_rows", lambda _bundle, rows, _profiles: list(rows))
    monkeypatch.setattr(execution, "select_task_batch", select_rows)
    return OfflineReuseLifecycle(
        call_kwargs=MappingProxyType(
            {
                "contract": contract,
                "local": local,
                "code_revision": "offline-reuse-v1",
                "command_runner": controller_runner,
            }
        ),
        local_output_root=local_output_root,
        private_root=private_root,
        gold176_row_ids=frozenset(row["row_id"] for row in gold_rows),
        selected_rows_by_round=tuple(schedule),
        process_records=process_records,
        source_calls=source_calls,
        downstream_rows=downstream_rows,
        binding=binding,
        task_sha256=task_sha256,
        fit_input_counts=fit_input_counts,
        receipt_bytes_before_reuse=MappingProxyType(receipt_bytes_before_reuse),
    )


def _write_stage1_seed(tmp_path: Path) -> Path:
    path = tmp_path / "stage1-seed.yaml"
    path.write_text(
        yaml.safe_dump(
            scanner_owned_pyramid_stage1_manifest(_stage1_manifest()),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
