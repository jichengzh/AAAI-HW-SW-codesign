import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage6_finalize_fcooper_table1_v2 import (
    TASK_ID,
    _median_performance,
    _validate_independent_ap,
    normalize_terminal_row,
    repeat_paths_for_row,
    select_gear_candidate,
    verify_success_evidence,
)


def success(row_id: str, *, task_id: str = TASK_ID) -> dict:
    return {
        "row_id": row_id,
        "task_id": task_id,
        "training_source": "online_feedback",
        "terminal_status": "measured_success_gold",
        "width": [32, 64, 128, 64, 128],
        "q_mode": "int8",
        "ap30": 0.7,
        "ap50": 0.65,
        "ap70": 0.60,
        "latency_ms": 1.0,
        "energy_j": 0.2,
    }


class FCooperTableFinalizerV2Test(unittest.TestCase):
    def test_gear_requires_exactly_16_formal_online_rows(self) -> None:
        rows = [success(f"online-{index:02d}") for index in range(16)]
        rows.append(success("probe", task_id="S5-FCO-PROBE"))

        with self.assertRaisesRegex(ValueError, "exactly 16"):
            select_gear_candidate(rows, ap70_ref=0.63, max_ap_drop=0.10)

        selected = select_gear_candidate(
            rows[:16], ap70_ref=0.63, max_ap_drop=0.10
        )
        self.assertIn(selected["row_id"], {row["row_id"] for row in rows[:16]})

    def test_repeat_paths_are_derived_from_selected_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = "formal-row-7"
            for index in range(3):
                path = (
                    root
                    / selected
                    / f"same_gpu7_repeat_{index}"
                    / "performance.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            unrelated = (
                root / "old-probe" / "same_gpu7_repeat_0" / "performance.json"
            )
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text("{}")

            paths = repeat_paths_for_row(root, selected)

            self.assertEqual(len(paths), 3)
            self.assertTrue(all(selected in str(path) for path in paths))

    def test_three_repeats_must_all_be_bound_to_gpu7(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(3):
                path = Path(tmp) / f"repeat-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "gpu_abs": 6 if index == 2 else 7,
                            "precision": "int8",
                            "builder_optimization_level": 5,
                            "lat_p50_ms": 1.0,
                            "energy_j": 0.2,
                            "artifact_sha256": {
                                "source_onnx": "a" * 64,
                                "compiled_engine": "b" * 64,
                            },
                        }
                    )
                )
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "GPU7"):
                _median_performance(
                    paths,
                    expected_onnx_sha256="a" * 64,
                    expected_precision="int8",
                    expected_builder_level=5,
                )

    def test_success_evidence_verifies_bytes_and_pruned_recovery_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            for field in ("performance", "ap", "source", "engine", "checkpoint"):
                path = root / field
                path.write_bytes(field.encode())
                files[field] = path
            files["contract"] = root / "contract.json"
            files["config"] = root / "config.yaml"
            files["initial"] = root / "initial.pth"
            files["config"].write_text("model: fcooper\n")
            files["initial"].write_bytes(b"initial")
            files["contract"].write_text(
                json.dumps(
                    {
                        "seed": 20260723,
                        "start_epoch": 23,
                        "minimum_epochs": 4,
                        "recovery_epochs": 8,
                        "amp_fp16": True,
                    }
                )
            )
            epoch_records = []
            for epoch in range(24, 28):
                epoch_checkpoint = root / f"net_epoch{epoch}.pth"
                epoch_checkpoint.write_bytes(f"epoch-{epoch}".encode())
                epoch_records.append(
                    {
                        "epoch": epoch,
                        "train_loss": 1.0,
                        "validation_loss": 0.9,
                        "elapsed_seconds": 1.0,
                        "checkpoint_path": str(epoch_checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            epoch_checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                )
            files["recovery"] = root / "recovery"
            files["recovery"].write_text(
                json.dumps(
                    {
                        "schema_version": "fcooper_recovery_training_report_v2",
                        "status": "success",
                        "initialization_policy": "scanner_dependency_l1_v2",
                        "recovery_contract_path": str(files["contract"]),
                        "recovery_contract_sha256": hashlib.sha256(
                            files["contract"].read_bytes()
                        ).hexdigest(),
                        "config_path": str(files["config"]),
                        "config_sha256": hashlib.sha256(
                            files["config"].read_bytes()
                        ).hexdigest(),
                        "initial_checkpoint_path": str(files["initial"]),
                        "initial_checkpoint_sha256": hashlib.sha256(
                            files["initial"].read_bytes()
                        ).hexdigest(),
                        "recovered_checkpoint_path": str(files["checkpoint"]),
                        "epochs_completed": 4,
                        "recovered_checkpoint_sha256": hashlib.sha256(
                            files["checkpoint"].read_bytes()
                        ).hexdigest(),
                        "seed": 20260723,
                        "amp_fp16": True,
                        "dataset": {
                            "train_root": "/data/train",
                            "validation_root": "/data/val",
                            "train_samples": 10,
                            "validation_samples": 5,
                            "full_train_split": True,
                            "full_validation_split": True,
                        },
                        "epoch_records": epoch_records,
                        "elapsed_seconds": 4.0,
                    }
                )
            )
            source_payload = {
                "checkpoint_path": str(files["checkpoint"]),
                "checkpoint_sha256": hashlib.sha256(
                    files["checkpoint"].read_bytes()
                ).hexdigest(),
                "recovery_training_report_path": str(files["recovery"]),
                "recovery_training_report_sha256": hashlib.sha256(
                    files["recovery"].read_bytes()
                ).hexdigest(),
            }
            files["source"].write_text(json.dumps(source_payload))
            row = {
                **success("pruned"),
                "performance_result_json": str(files["performance"]),
                "performance_result_sha256": hashlib.sha256(
                    files["performance"].read_bytes()
                ).hexdigest(),
                "ap_report_path": str(files["ap"]),
                "ap_report_sha256": hashlib.sha256(files["ap"].read_bytes()).hexdigest(),
                "materialized_source_evidence_path": str(files["source"]),
                "materialized_source_evidence_sha256": hashlib.sha256(
                    files["source"].read_bytes()
                ).hexdigest(),
                "engine_path": str(files["engine"]),
                "engine_sha256": hashlib.sha256(
                    files["engine"].read_bytes()
                ).hexdigest(),
                "checkpoint_sha256": source_payload["checkpoint_sha256"],
                "recovery_training_report_sha256": source_payload[
                    "recovery_training_report_sha256"
                ],
                "graph_features": {"group_id": "g"},
            }
            row["materialized_graph_features_sha256"] = hashlib.sha256(
                json.dumps(
                    row["graph_features"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()

            audit = verify_success_evidence(row, root=root)

            self.assertTrue(audit["verified"])
            files["recovery"].write_text("drift")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                verify_success_evidence(row, root=root)

    def test_credible_failure_has_no_invented_metrics(self) -> None:
        failure = {
            "row_id": "failed",
            "terminal_status": "feasibility_failure",
            "failure_reason": "TensorRT rejected the materialized graph",
        }

        normalized = normalize_terminal_row(failure)

        self.assertEqual(normalized["terminal_status"], "feasibility_failure")
        for metric in ("ap30", "ap50", "ap70", "latency_ms", "energy_j"):
            self.assertIsNone(normalized[metric])

    def test_independent_ap_requires_full_2170_and_matching_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ap.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "success_full",
                        "dataset_samples": 2170,
                        "processed_samples": 2170,
                        "fallback_samples": 0,
                        "ap30": 0.7,
                        "ap50": 0.65,
                        "ap70": 0.6,
                    }
                )
            )
            audit = _validate_independent_ap(
                path, expected={"ap30": 0.7, "ap50": 0.65, "ap70": 0.6}
            )
            self.assertEqual(audit["processed_samples"], 2170)
            path.write_text(
                json.dumps(
                    {
                        "status": "success_full",
                        "dataset_samples": 2170,
                        "processed_samples": 2170,
                        "fallback_samples": 0,
                        "ap30": 0.7,
                        "ap50": 0.65,
                        "ap70": 0.5,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "metric drift"):
                _validate_independent_ap(
                    path, expected={"ap30": 0.7, "ap50": 0.65, "ap70": 0.6}
                )


if __name__ == "__main__":
    unittest.main()
