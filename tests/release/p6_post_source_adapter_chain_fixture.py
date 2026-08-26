"""Private full-chain fixture using generated wrappers and tracked adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml

from framework.stage6.p6_post_source_adapter_profile_v1 import (
    POST_SOURCE_ADAPTER_STAGES,
    POST_SOURCE_LEAF_NAMES,
    PostSourceAdapter,
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
    post_source_adapter_profile_to_mapping,
)
from framework.stage6.p6_post_source_wrapper_template_v1 import (
    render_post_source_adapter_wrappers,
)
from framework.stage6.p6_history_source_materialization_v1 import (
    project_source_materialization_request,
)
from tests.release.test_run_p6_ap_round_adapter import (
    _execute_leaf_body as _ap_execute_body,
)
from tests.release.test_run_p6_ap_round_adapter import (
    _plan_leaf_body as _ap_plan_body,
)
from tests.release.test_run_p6_performance_round_adapter import (
    _execute_leaf_body as _performance_execute_body,
)
from tests.release.test_run_p6_performance_round_adapter import (
    _plan_leaf_body as _performance_plan_body,
)
from tests.stage6 import test_p6_history_measurement as history_fixtures
from tests.stage6.test_p6_source_reuse_measurement import _create_context


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SOURCES: Mapping[str, Path] = {
    stage: REPOSITORY_ROOT / f"tools/release/run_p6_{stage}_round_adapter.py"
    for stage in POST_SOURCE_ADAPTER_STAGES
}


def install_adapter_chain(private_root: Path) -> Mapping[str, Path]:
    """Install reviewed adapter bytes, seven native fakes, profile, and wrappers."""
    _initialize_private_git_root(private_root)
    _copy_runtime(private_root)
    adapters = tuple(_copy_adapter(private_root, stage) for stage in POST_SOURCE_ADAPTER_STAGES)
    leaves = tuple(_write_leaf(private_root, name) for name in POST_SOURCE_LEAF_NAMES)
    project_python = private_root / "historical-env/bin/python3.9"
    project_python.parent.mkdir(parents=True)
    project_python.write_text(
        "#!/bin/sh\nexec /usr/bin/python3.10 \"$@\"\n", encoding="utf-8"
    )
    project_python.chmod(0o700)
    dependency_root = private_root / "execution-closure/dependency-overlay"
    dependency_root.mkdir(parents=True)
    profile = ValidatedPostSourceAdapterProfile(
        "p6_post_source_adapter_profile_v3",
        private_root.resolve(strict=True),
        project_python,
        adapters,
        leaves,
        adapter_python=Path(sys.executable).resolve(strict=True),
        adapter_dependency_root=dependency_root,
    )
    profile_path = private_root / "post-source-adapter-profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(post_source_adapter_profile_to_mapping(profile), sort_keys=False),
        encoding="utf-8",
    )
    return render_post_source_adapter_wrappers(profile, private_root=private_root)


def build_adapter_measurement_request(local_output_root: Path) -> dict[str, Any]:
    unprojected = history_fixtures._unprojected_recipe_v2_request(local_output_root)
    projected = dict(project_source_materialization_request(unprojected).request)
    _create_context(local_output_root, projected)
    return projected


def write_source_materializer(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n{_source_materializer_body()}", encoding="utf-8"
    )
    path.chmod(0o700)
    return str(path)


def assert_adapter_leaf_chain(
    history_round: Path,
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    stages = binding["execution_interface"]["execution_chain"]
    assert [stage["stage"] for stage in stages] == [
        "source_materialization", "quantization", "performance", "ap", "finalization"
    ]
    quant = _read_jsonl(history_round / "quant-leaf.log")
    assert len(quant) == len([row for row in request["rows"] if row["q_mode"] == "int8"]) == 2
    assert [row["cuda"] for row in quant] == ["101", "103"]
    performance = _read_jsonl(history_round / "performance-leaves.log")
    assert [row["leaf"] for row in performance] == ["plan", "execute"]
    ap = _read_jsonl(history_round / "ap-leaves.log")
    assert [row["leaf"] for row in ap].count("plan") == 1
    execute_stages = _ap_execute_stages(ap)
    assert execute_stages.count("sanity") == execute_stages.count("full") == 3
    assert execute_stages.index("full") > max(
        index for index, stage in enumerate(execute_stages) if stage == "sanity"
    )
    finalization = _read_jsonl(history_round / "finalization-leaves.log")
    assert [row["leaf"] for row in finalization] == ["finalize", "promote"]
    leaf_runtime = _read_jsonl(history_round / "leaf-runtime.log")
    assert {row["leaf"] for row in leaf_runtime} == set(POST_SOURCE_LEAF_NAMES)
    private_root = Path(binding["private_root"])
    expected_path = f"{private_root / 'historical-env/bin'}:/usr/bin:/bin"
    expected_pythonpath = f"{private_root}:{private_root / 'tools/release'}"
    assert all(row["path"] == expected_path for row in leaf_runtime)
    assert all(row["pythonpath"] == expected_pythonpath for row in leaf_runtime)
    assert all("dependency-overlay" not in row["pythonpath"] for row in leaf_runtime)
    feedback = _read_json(history_round / "actual-feedback.json")
    assert len(feedback["rows"]) == 4
    metrics = {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
    assert all(metrics <= set(row) for row in feedback["rows"])
    assert _read_json(history_round / "barrier.json") == _read_json(
        history_round / "receipt.json"
    )
    assert _read_json(history_round / "state/task-state.json")["stage"] == "finalization"


def _initialize_private_git_root(private_root: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(private_root)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _copy_runtime(private_root: Path) -> None:
    shutil.copytree(
        REPOSITORY_ROOT / "framework",
        private_root / "framework",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (private_root / "tools/release").mkdir(parents=True)


def _copy_adapter(private_root: Path, stage: str) -> PostSourceAdapter:
    source = ADAPTER_SOURCES[stage]
    destination = private_root / "tools/release" / source.name
    shutil.copy2(source, destination)
    destination.chmod(0o700)
    return PostSourceAdapter(stage, destination, destination.parent)


def _write_leaf(private_root: Path, name: str) -> PostSourceLeaf:
    leaf_root = private_root / "historical-leaves"
    leaf_root.mkdir(exist_ok=True)
    path = leaf_root / f"{name}.py"
    path.write_text(
        f"#!{sys.executable}\n{_leaf_body(name)}{_leaf_runtime_audit_body(name)}",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return PostSourceLeaf(name, path, leaf_root, hashlib.sha256(path.read_bytes()).hexdigest())


def _leaf_runtime_audit_body(name: str) -> str:
    return f'''\n
import json as _runtime_json
import os as _runtime_os
from pathlib import Path as _RuntimePath
import sys as _runtime_sys
with (_RuntimePath(_runtime_os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"]) / "leaf-runtime.log").open("a", encoding="utf-8") as _runtime_handle:
    _runtime_handle.write(_runtime_json.dumps({{
        "leaf": {name!r},
        "path": _runtime_os.environ["PATH"],
        "pythonpath": _runtime_os.environ["PYTHONPATH"],
    }}, sort_keys=True) + "\\n")
'''


def _leaf_body(name: str) -> str:
    bodies = {
        "quant_contract": _native_quant_leaf_body(),
        "performance_plan": _performance_plan_body(),
        "performance_execute": _performance_execute_body(),
        "ap_plan": _ap_plan_body(),
        "ap_execute": _ap_execute_body(),
        "feedback_finalize": _finalize_body(),
        "feedback_promote": _promote_body(),
    }
    return bodies[name]


def _native_quant_leaf_body() -> str:
    return r'''
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

onnx = Path(sys.argv[sys.argv.index("--onnx") + 1])
calibration_npz = Path(sys.argv[sys.argv.index("--calibration-npz") + 1])
calibration_summary = Path(sys.argv[sys.argv.index("--calibration-summary") + 1])
output_path = Path(sys.argv[sys.argv.index("--output-json") + 1])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps({
    "schema": "stage3_tvm_int8_quant_contract_v3",
    "onnx_path": str(onnx.resolve()),
    "onnx_sha256": sha256(onnx),
    "calibration_npz": str(calibration_npz.resolve()),
    "calibration_npz_sha256": sha256(calibration_npz),
    "calibration_summary": str(calibration_summary.resolve()),
    "calibration_summary_sha256": sha256(calibration_summary),
    "params": {"spatial_features": {"scale": 0.5}},
}, sort_keys=True), encoding="utf-8")
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "quant-leaf.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
        "round_root": str(round_root),
    }, sort_keys=True) + "\n")
'''


def _finalize_body() -> str:
    return r'''
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

request = json.loads(Path(sys.argv[sys.argv.index("--measurement-request-json") + 1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[sys.argv.index("--manifest-json") + 1]).read_text(encoding="utf-8"))
if manifest.get("source_request_sha256") != request.get("measurement_request_sha256"):
    raise ValueError("legacy request/manifest mismatch")
if any(row.get("schema_version") != "stage5_feedback_row_v2" for row in request["rows"]):
    raise ValueError("native finalizer schema mismatch")
rows = []
for row in request["rows"]:
    rows.append({
        **row,
        "measurement_request_row_sha256": request["row_sha256"][row["row_id"]],
        "terminal_status": "measured_success_gold",
        "latency_ms": 2.0,
        "energy_j": 0.5,
        "ap30": 0.3,
        "ap50": 0.5,
        "ap70": 0.7,
        "failure_reason": None,
    })
output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
output.mkdir(parents=True, exist_ok=True)
(output / "stage5_feedback_v2_final.json").write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
(output / "stage5_feedback_v2_final.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
(output / "stage5_feedback_v2_audit.json").write_text(json.dumps({"schema_version": "stage5_feedback_batch_v2", "summary": {"measured": 4, "failure": 0, "pending": 0, "total": 4}}, sort_keys=True), encoding="utf-8")
(output / "atomic_batch_audit.json").write_text(json.dumps({"schema_version": "stage5_atomic_batch_audit_v2", "feedback_released": True, "batch_quarantined": False, "budget_consumed": 4, "released_feedback_rows": rows}, sort_keys=True), encoding="utf-8")
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "finalization-leaves.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"leaf": "finalize", "argv": sys.argv[1:]}, sort_keys=True) + "\n")
'''


def _promote_body() -> str:
    return r'''
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

request = json.loads(Path(sys.argv[sys.argv.index("--measurement-request-json") + 1]).read_text(encoding="utf-8"))
feedback = json.loads(Path(sys.argv[sys.argv.index("--feedback-json") + 1]).read_text(encoding="utf-8"))
rows = []
audit_rows = []
for requested, historical in zip(request["rows"], feedback, strict=True):
    candidate = requested["graph_features"]
    actual = {**candidate, "graph_feature_provenance": "materialized_onnx_extracted_v1"}
    promoted = {
        **historical,
        "candidate_graph_features": candidate,
        "candidate_graph_features_sha256": sha(candidate),
        "graph_features": actual,
        "materialized_graph_features_sha256": sha(actual),
        "historical_feedback_row_sha256": sha(historical),
        "feedback_feature_contract": "actual_feedback_v3",
        "graph_feature_promotion_schema": "stage5_actual_feedback_promotion_v3",
    }
    promoted["actual_feedback_row_sha256"] = sha(promoted)
    rows.append(promoted)
    audit_rows.append({key: promoted[key] for key in ("manifest_job_id", "candidate_graph_features_sha256", "materialized_graph_features_sha256", "actual_feedback_row_sha256")})
output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
output.mkdir(parents=True, exist_ok=True)
(output / "stage5_feedback_v3_actual.json").write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
(output / "actual_feedback_batch_audit_v3.json").write_text(json.dumps({"schema_version": "stage5_actual_feedback_batch_audit_v3", "promoted_row_count": 4, "silent_surrogate_fallback_count": 0, "rows": audit_rows}, sort_keys=True), encoding="utf-8")
round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"])
with (round_root / "finalization-leaves.log").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"leaf": "promote", "argv": sys.argv[1:]}, sort_keys=True) + "\n")
'''


def _source_materializer_body() -> str:
    return r'''
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

stage_log = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"]) / "executed-stages.log"
with stage_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(dict(stage=Path(__file__).name, argv=sys.argv[1:], cwd=str(Path.cwd()), round_output_root=os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"], task_state=os.environ["P6_HISTORY_TASK_STATE"]), sort_keys=True) + "\n")
request_path = Path(sys.argv[sys.argv.index("--request") + 1])
group_id = sys.argv[sys.argv.index("--group-id") + 1]
request = json.loads(request_path.read_text(encoding="utf-8"))
row = next(row for row in request["rows"] if row["group_id"] == group_id)
contract = row["source_contract"]
directory_keys = {"checkpoint_dir", "calibration_root", "trt_calibration_dir"}
artifact_keys = ("checkpoint_path", "checkpoint_dir", "config_path", "onnx_path", "onnx_report_path", "calibration_root", "calibration_npz", "calibration_summary", "trt_calibration_dir")
for key in artifact_keys:
    target = Path(contract[key])
    if key in directory_keys:
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.bin").write_bytes(key.encode("utf-8"))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(key.encode("utf-8"))
request_mtime = request_path.stat().st_mtime_ns
for offset, key in enumerate(("training_done_marker", "source_done_marker"), start=1):
    marker = Path(contract[key])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(key.encode("utf-8"))
    os.utime(marker, ns=(request_mtime + offset, request_mtime + offset))
'''


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ap_execute_stages(rows: list[dict[str, Any]]) -> list[str]:
    return [
        row["argv"][row["argv"].index("--stage") + 1]
        for row in rows
        if row["leaf"] == "execute"
    ]
