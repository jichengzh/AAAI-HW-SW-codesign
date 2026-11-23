#!/usr/bin/env python3
"""Export the frozen MetaSchedule FP32 Pyramid artifact used by schedule-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stage2_h800_run_measurement_job import configure_tvm_env, load_relax_module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-database", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_tvm_env(str(args.gpu))
    import tvm
    from tvm import relax
    import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

    workload = args.work_dir / "database_workload.json"
    records = args.work_dir / "database_tuning_record.json"
    for path in (args.onnx, workload, records):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = tvm.cuda(0)
    target = tvm.target.Target.from_device(device)
    module, shapes, dtypes = load_relax_module(args.onnx)
    pipeline = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    started = time.monotonic()
    with target, tvm.transform.PassContext(opt_level=3):
        fused = pipeline(module)
        scheduled = (
            fused
            if args.skip_database
            else relax.transform.MetaScheduleApplyDatabase(work_dir=str(args.work_dir))(fused)
        )
        executable = tvm.compile(scheduled, target=target)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    executable.export_library(str(args.artifact))
    report = {
        "schema_version": "stage6_tvm_fp32_schedule_artifact_v1",
        "status": "success",
        "precision": "fp32",
        "onnx_path": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "work_dir": str(args.work_dir),
        "database_workload_sha256": sha256_file(workload),
        "database_tuning_record_sha256": sha256_file(records),
        "artifact_path": str(args.artifact),
        "artifact_sha256": sha256_file(args.artifact),
        "input_shapes": {name: list(shape) for name, shape in shapes.items()},
        "input_dtypes": dtypes,
        "compile_s": time.monotonic() - started,
        "retuned": False,
        "schedule_database_applied": not args.skip_database,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
