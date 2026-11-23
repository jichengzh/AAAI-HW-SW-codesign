"""Target-local consumer of authenticated RN50 transfer bundles."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import remote
from .remote import CODE_FILES, RESULT_FILES, digest, sha, transport_config, write_json
from .runtime import measure_candidate
from .trt_runtime import TensorRTRuntime, build_engine


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "transfer_manifest.json").read_text())
    body = {key: value for key, value in manifest.items() if key != "transport_id"}
    if manifest.get("schema_version") != "resnet_compare_transfer_v1" or digest(body) != manifest["transport_id"]:
        raise ValueError("Transfer manifest SHA identity mismatch")
    if digest(manifest["samples"]) != manifest["sample_rows_sha256"]:
        raise ValueError("Sample row SHA identity mismatch")
    if sha(root / "candidate.onnx") != manifest["export_sha256"] or sha(root / "samples.npz") != manifest["sample_pack_sha256"]:
        raise ValueError("Export or sample pack SHA differs")
    if set(manifest["code_sha256"]) != set(CODE_FILES):
        raise ValueError("Executable SHA inventory differs")
    for name in CODE_FILES:
        if sha(root / "framework/resnet_compare" / name) != manifest["code_sha256"][name]:
            raise ValueError("Executable SHA differs: " + name)
    return manifest


def unpack_samples(root, manifest):
    with np.load(Path(root) / "samples.npz", allow_pickle=False) as pack:
        inputs, logits = pack["inputs"], pack["source_logits"]
    rows = manifest["samples"]
    if inputs.shape != (len(rows), 3, 224, 224) or logits.shape != (len(rows), 1000):
        raise ValueError("Sample pack shape differs")
    if not np.isfinite(inputs).all() or not np.isfinite(logits).all() or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Nonfinite or duplicate target samples")
    samples = []
    for i, row in enumerate(rows):
        if hashlib.sha256(inputs[i].tobytes()).hexdigest() != row["input_sha256"] or hashlib.sha256(logits[i].tobytes()).hexdigest() != row["source_logits_sha256"]:
            raise ValueError("Per-sample input/logit SHA differs: " + row["sample_id"])
        samples.append({"sample_id": row["sample_id"], "input": torch.from_numpy(inputs[i:i+1].copy()),
                        "source_logits": torch.from_numpy(logits[i:i+1].copy()), "label": row["label"],
                        "is_synthetic": row["is_synthetic"]})
    return samples


@contextmanager
def serial_target(root):
    """One persistent inode for all experiment jobs, released by kernel on exit."""
    transport_config({"remote_root": str(root)})
    with (Path(remote.APPROVED_ROOT) / ".orin-runtime.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_power_reader(spec, approved_root):
    if not spec:
        return None, "No approved readable integrating power provider; energy unavailable"
    path = Path(spec["path"]).resolve()
    if Path(approved_root) not in path.parents or sha(path) != spec["sha256"]:
        raise ValueError("Power provider path/SHA outside approved experiment identity")
    module_spec = importlib.util.spec_from_file_location("resnet_approved_power_reader", str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    reader = module.open_reader()
    probe = reader.probe()
    return (reader, probe.get("reason", "Approved sensor available")) if probe.get("available") is True else (None, probe.get("reason", "Sensor unavailable"))


class PowerRuntime:
    """Optional approved sensor-integrator enclosure; never energy=power×latency."""
    def __init__(self, adapter, reader):
        self.adapter, self.reader = adapter, reader
        for key in ("backend", "hardware", "layer_precision", "target_graph_verified"):
            setattr(self, key, getattr(adapter, key))

    def infer(self, tensor):
        self.reader.start_request()
        started = time.monotonic()
        observation = self.adapter.infer(tensor)
        completed = time.monotonic()
        evidence = self.reader.finish_request()
        valid = (observation.get("completion") == "cuda_event_synchronized"
                 and evidence.get("method") == "sensor_integral" and evidence.get("sensor_id")
                 and type(evidence.get("sample_count")) is int and evidence["sample_count"] >= 2
                 and all(math.isfinite(evidence[k]) for k in ("energy_j", "start_monotonic_s", "end_monotonic_s"))
                 and evidence["energy_j"] > 0 and evidence["start_monotonic_s"] <= started
                 and evidence["end_monotonic_s"] >= completed)
        if not valid:
            raise ValueError("Power provider must supply actual sensor integral enclosing completed GPU request")
        return {**observation, "energy_j": evidence["energy_j"], "energy_evidence": evidence}


def _engine(root, manifest):
    identity = {key: manifest[key] for key in ("export_sha256", "precision", "code_sha256")}
    identity["candidate"] = manifest["candidate"]
    identity_path, engine = root / "engine_identity.json", root / "candidate.engine"
    if engine.exists():
        saved = json.loads(identity_path.read_text())
        if saved["identity"] != identity or saved["engine_sha256"] != sha(engine):
            raise ValueError("Existing engine identity/SHA differs; preserving it")
        return engine, {**saved["build"], "reused_same_identity": True}
    build = build_engine(root / "candidate.onnx", engine, manifest["precision"])
    write_json(identity_path, {"identity": identity, "engine_sha256": sha(engine), "build": build})
    return engine, {**build, "reused_same_identity": False}


def _reuse_result(root, manifest):
    receipt = json.loads((root / "result/receipt.json").read_text())
    if receipt["transport_id"] != manifest["transport_id"] or receipt["manifest_sha256"] != sha(root / "transfer_manifest.json"):
        raise ValueError("Existing result transport identity differs")
    if set(receipt["files_sha256"]) != set(RESULT_FILES):
        raise ValueError("Existing result evidence inventory differs")
    for name in RESULT_FILES:
        if sha(root / "result" / name) != receipt["files_sha256"][name]:
            raise ValueError("Existing result SHA differs; preserving artifact")
    return json.loads((root / "result/baseline_runtime.json").read_text())


def run_transfer(root):
    root = Path(root).resolve()
    manifest = verify_bundle(root)
    config = transport_config(manifest["transport"])
    expected_root = Path(config["remote_root"]) / manifest["candidate"]["run_id"] / manifest["transport_id"]
    if root != expected_root:
        raise ValueError("Target job path differs from authenticated run/candidate identity")
    with serial_target(config["remote_root"]):
        if (root / "result/receipt.json").exists():
            return _reuse_result(root, manifest)
        if (root / "result").exists():
            raise FileExistsError("Incomplete target result preserved; use a new run identity")
        samples = unpack_samples(root, manifest)
        engine, build = _engine(root, manifest)
        adapter = TensorRTRuntime(engine)
        try:
            reader, power_reason = load_power_reader(config.get("power_reader"), config["remote_root"])
        except (OSError, ValueError, ImportError) as exc:
            reader, power_reason = None, "Approved power provider unavailable: " + str(exc)
        measured = PowerRuntime(adapter, reader) if reader is not None else adapter
        result = measure_candidate(manifest["candidate"], {"target": "orin", "precision": manifest["precision"],
            "adapter": measured, **{key: config[key] for key in ("warmup", "windows", "requests_per_window")}}, samples, root / "result")
        rows = [json.loads(line) for line in (root / "result/timings.jsonl").read_text().splitlines()]
        energy = [row.get("energy_j") for row in rows]
        if rows and all(value is not None for value in energy) and result["status"] != "FAILED":
            result = {**result, "energy_j": float(np.mean(energy)),
                      "energy_reason": "Actual approved sensor integrals; mean per completed request",
                      "energy_boundary": "sensor-integral request enclosure; includes reader overhead; GPU complete before end"}
        else:
            result = {**result, "energy_reason": power_reason, "energy_boundary": None}
        if result["status"] == "READY" and result["energy_j"] is None:
            result = {**result, "status": "PARTIAL", "reason": "NEEDS_POWER: target timing/numerics available, three-objective feedback blocked"}
        # Existing file is this invocation's runtime result, not historical data.
        (root / "result/baseline_runtime.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_json(root / "result/build.json", build)
        write_json(root / "result/receipt.json", {"transport_id": manifest["transport_id"],
            "manifest_sha256": sha(root / "transfer_manifest.json"),
            "files_sha256": {name: sha(root / "result" / name) for name in RESULT_FILES}})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run_transfer(args.job_root), allow_nan=False))
