#!/usr/bin/env python3
"""Audit a completed P4 bundle without rewriting live manifests or run state.

This proves complete frozen input *mappings* and five integration episodes,
not paper performance or that an asynchronous worker consumed every input.
Actual Orin NPY byte verification is a separate deployment SHA-256 gate.
"""

import argparse
import json
from pathlib import Path
import sys

from coptv2x_hil_manifest import build_manifest, select_pilot
from coptv2x_hil_result import atomic_json, file_sha256, validate_episode_audit, validate_task_artifacts
from coptv2x_hil_runner import _manifest_sha256, _task_ids_digest, _read_checkpoint, classify_episode
from coptv2x_hil_trigger import parse_hil_trigger


SCOPE = {"schema":"coptv2x.hil-p4-bundle-audit.v1", "integration_smoke_only":True,
         "performance_conclusion_allowed":False,
         "scope":"single-route five-arm asynchronous HIL integration smoke"}


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence JSON must be an object: %s" % path)
    return payload


def _sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("input SHA-256 must be a 64-character hexadecimal string")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError("input SHA-256 must be hexadecimal")
    return value.lower()


def _frame_map(frames, remote=False):
    if not isinstance(frames, list):
        raise ValueError("input frames must be a list")
    result = {}
    for frame in frames:
        source = frame["source_frame"]
        sample = frame["sample_id"]
        if type(source) is not int or source < 0 or sample != "carla-frame-%d" % source:
            raise ValueError("input frame identity is invalid")
        key = (source, sample)
        if key in result:
            raise ValueError("duplicate input frame identity")
        feature = frame["inputs"]["spatial_features"] if remote else frame
        if feature["shape"] != [2, 64, 256, 512] or feature["dtype"] != "float32":
            raise ValueError("input tensor shape/dtype contract is invalid")
        result[key] = _sha256(feature["sha256"] if remote else feature["spatial_features_sha256"])
    return result


def _input_authority(manifest, state, host_path, expected_sha, orin_path, expected_count):
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected frame count must be positive")
    expected_sha = _sha256(expected_sha)
    if file_sha256(host_path) != expected_sha:
        raise ValueError("host input index SHA does not match pinned historical authority")
    host, orin = _read_json(host_path), _read_json(orin_path)
    if host.get("schema") != "coptv2x.route-input-host-index.v1" or host.get("status") != "HOST_INDEX_VALID":
        raise ValueError("host input index schema/status is invalid")
    cells = manifest["route_scenarios"]
    if len(cells) != 1 or set(host["route_scenarios"]) != set(cells):
        raise ValueError("P4 requires one exact frozen pilot cell")
    if host["route_scenarios_sha256"] != manifest["route_scenarios_sha256"]:
        raise ValueError("host input index route authority differs from cohort")
    reference = manifest.get("host_input_index")
    if (not isinstance(reference, dict) or reference.get("sha256") != expected_sha
            or state.get("host_input_index") != reference):
        raise ValueError("run state does not reference the pinned host input index")
    host_frames = _frame_map(host["route_scenarios"][cells[0]])
    if len(host_frames) != expected_count or reference.get("verified_feature_files") != expected_count:
        raise ValueError("frozen input frame count is not the expected complete count")
    first_task = manifest["tasks"][0]
    if (orin.get("schema_version") != "coptv2x.local-input-manifest.v1"
            or orin.get("batch_order") != ["ego", "rsu"]
            or orin.get("route_id") != "Town05-route-%d" % first_task["route"]
            or orin.get("scenario_id") != first_task["scenario_id"]):
        raise ValueError("Orin input manifest route/scenario/schema contract differs")
    if _frame_map(orin["samples"], remote=True) != host_frames:
        raise ValueError("Orin input map differs from exact frozen frame identity/SHA set")
    return host_frames, {"host_input_index_path":str(host_path),
                         "host_input_index_sha256":expected_sha,
                         "orin_input_manifest_path":str(orin_path),
                         "orin_input_manifest_sha256":file_sha256(orin_path),
                         "route_scenario_id":cells[0], "expected_frame_count":expected_count,
                         "verified_frame_identities":len(host_frames),
                         "exact_source_sample_sha256_mapping":True,
                         "npy_bytes_rehashed_by_finalizer":False}


def _selected_audits(task, record, cohort_id, inputs):
    path = Path(record["fusion_audit_path"])
    selected = []
    sources = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("selected_result_sequence_no") is None:
            continue
        entry = validate_episode_audit({"batch_order":row.get("batch_order"),
                "trigger_identity":row.get("trigger_identity"),
                "result_identity":row.get("selected_result_trigger_identity"),
                "artifact_sha256":row.get("artifact_sha256"),
                "aoi_frames":row.get("rsu_aoi_frames"),
                "remote_input_tensor_bytes":row.get("remote_input_tensor_bytes", 0)})
        validate_task_artifacts(task, entry)
        expected = {"run_id":cohort_id, "route_id":"Town05-route-%d" % task["route"],
                    "scenario_id":task["scenario_id"], "arm_id":task["arm_id"]}
        for identity in (entry["trigger_identity"], entry["result_identity"]):
            parse_hil_trigger(identity)
            if any(identity.get(key) != value for key, value in expected.items()):
                raise ValueError("selected fusion task identity differs from frozen episode")
        result = entry["result_identity"]
        key = (result["source_frame"], result["sample_id"])
        if key not in inputs:
            raise ValueError("selected result is outside exact frozen input mapping")
        if row["selected_result_sequence_no"] != result["sequence_no"]:
            raise ValueError("selected fusion sequence differs from result identity")
        sources.add(key)
        selected.append(entry)
    if not selected:
        raise ValueError("episode has no selected fusion evidence")
    if selected[-1] != validate_episode_audit(record["hil_audit"]):
        raise ValueError("stored HIL audit does not match actual final selected fusion evidence")
    return {"selected_fusion_frames":len(selected), "unique_selected_source_frames":len(sources),
            "fusion_audit_path":str(path), "fusion_audit_sha256":file_sha256(path)}


def _checkpoint_path_for_result(path, expected_status, expected_result):
    """Resolve the same first terminal candidate used by runner._read_checkpoint."""
    for candidate in (path, path.parent / "ego_vehicle_0" / path.name):
        if not candidate.is_file():
            continue
        try:
            payload = _read_json(candidate)
            data = payload.get("_checkpoint", payload)
            global_record = data.get("global_record", {})
            records = data.get("records", [])
            status = global_record.get("status")
            if status == "Completed":
                parsed_status = "completed"
            elif status == "Failed" and records:
                route_record = records[0]
                infractions = route_record.get("infractions", {})
                parsed_status = ("route_timeout" if infractions.get("route_timeout")
                                 else route_record.get("status"))
            else:
                continue
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if parsed_status != expected_status or global_record != expected_result:
            raise ValueError("actual CARLA evidence selection differs from parsed result")
        return candidate
    raise ValueError("actual CARLA terminal checkpoint evidence is missing")


def _episode(task, record, cohort_id, inputs):
    if record.get("classification") not in ("COMPLETED", "DRIVING_TERMINAL"):
        raise ValueError("episode is not a valid driving terminal: %s" % task["arm_id"])
    if record.get("arm_id") != task["arm_id"] or record.get("artifact_sha256") != task["artifact_sha256"]:
        raise ValueError("terminal arm/artifact identity differs from frozen task")
    for hook in ("episode_start", "episode_stop"):
        if hook in record and record[hook].get("return_code") != 0:
            raise ValueError("episode service hook failed")
    checkpoint = Path(record["carla_result_path"])
    status, carla_result = _read_checkpoint(checkpoint)
    actual_classification = classify_episode(status, record.get("return_code", 127))
    if actual_classification != record["classification"] or carla_result != record.get("carla_result"):
        raise ValueError("actual CARLA result does not support stored terminal classification")
    checkpoint = _checkpoint_path_for_result(checkpoint, status, carla_result)
    fusion = _selected_audits(task, record, cohort_id, inputs)
    return dict(fusion, task_id=task["task_id"], arm_id=task["arm_id"],
                classification=actual_classification, carla_status=status,
                carla_result_path=str(checkpoint), carla_result_sha256=file_sha256(checkpoint),
                integration_smoke_only=True, performance_conclusion_allowed=False)


def audit_p4_bundle(cohort_dir, host_input_index, expected_input_index_sha256,
                    orin_input_manifest, expected_input_frames=2225):
    cohort = Path(cohort_dir).expanduser().resolve()
    manifest = _read_json(cohort / "manifest.json")
    state = _read_json(cohort / "state.json")
    tasks = manifest["tasks"]
    cells = manifest["route_scenarios"]
    frozen = select_pilot(build_manifest(), cells)
    if len(cells) != 1 or tasks != frozen["tasks"] or manifest["protocol"] != frozen["protocol"]:
        raise ValueError("P4 manifest must be the exact single-cell frozen five-arm protocol")
    task_ids = [task["task_id"] for task in tasks]
    if (state.get("schema") != "coptv2x.hil-run-state.v1"
            or state.get("manifest_sha256") != _manifest_sha256(manifest)
            or state.get("manifest_task_ids_sha256") != _task_ids_digest(task_ids)
            or state.get("task_ids") != task_ids):
        raise ValueError("run state manifest identity mismatch")
    if not isinstance(state.get("results"), dict) or set(state["results"]) != set(task_ids):
        raise ValueError("five valid driving terminals are required; P4 still incomplete")
    inputs, authority = _input_authority(manifest, state,
                Path(host_input_index).expanduser().resolve(), expected_input_index_sha256,
                Path(orin_input_manifest).expanduser().resolve(), expected_input_frames)
    episodes = [_episode(task, state["results"][task["task_id"]], cohort.name, inputs) for task in tasks]
    return dict(SCOPE, status="P4_PASS", cohort_dir=str(cohort), input_authority=authority,
                manifest_sha256=file_sha256(cohort / "manifest.json"),
                state_sha256=file_sha256(cohort / "state.json"), episodes=episodes,
                asynchronous_all_inputs_consumed_required=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--host-input-index", type=Path, required=True)
    parser.add_argument("--expected-input-index-sha256", required=True)
    parser.add_argument("--orin-input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-frames", type=int, default=2225)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_p4_bundle(args.cohort_dir, args.host_input_index,
                                args.expected_input_index_sha256, args.orin_input_manifest,
                                args.expected_input_frames)
        code = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(SCOPE, status="P4_NOT_COMPLETE", error=str(error))
        code = 2
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
