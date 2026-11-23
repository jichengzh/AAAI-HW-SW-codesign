"""Authenticated, bounded source-to-Orin experiment transport."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import torch

from .contracts import REFERENCE_SHA256, validate_result
from .runtime import _base

APPROVED_ROOT = Path("/home/jichengzhi/resnet_compare_s1_20261013")
HOST = "jichengzhi@172.16.62.222"
PYTHON = "/home/jichengzhi/uniad_venv/bin/python"
CODE_FILES = ("__init__.py", "contracts.py", "runtime.py", "trt_runtime.py", "remote.py", "target_entry.py")
RESULT_FILES = ("baseline_runtime.json", "timings.jsonl", "layer_precision.json", "numerical.json", "build.json")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def transport_config(value=None):
    config = {"host": HOST, "remote_root": str(APPROVED_ROOT), "python": PYTHON,
              "timeout_s": 1800, "warmup": 50, "windows": 3, "requests_per_window": 200,
              **(value or {})}
    if set(config) - {"host", "remote_root", "python", "timeout_s", "warmup", "windows", "requests_per_window", "power_reader"}:
        raise ValueError("Unknown transport configuration")
    root = Path(config["remote_root"]).resolve()
    if config["host"] != HOST or config["python"] != PYTHON or (root != APPROVED_ROOT and APPROVED_ROOT not in root.parents):
        raise ValueError("Transport must stay inside approved Orin experiment scope")
    for name, maximum in (("timeout_s", 1800), ("warmup", 100), ("windows", 3), ("requests_per_window", 200)):
        if type(config[name]) is not int or not 1 <= config[name] <= maximum:
            raise ValueError("Transport budget must be positive and bounded: " + name)
    config["remote_root"] = str(root)
    provider = config.get("power_reader")
    if provider is not None:
        if not isinstance(provider, dict) or set(provider) != {"path", "sha256"}:
            raise ValueError("Power provider needs explicit approved path and SHA")
        if root not in Path(provider["path"]).resolve().parents or not re.fullmatch(r"[0-9a-f]{64}", provider["sha256"]):
            raise ValueError("Power provider outside approved experiment root/SHA")
    return config


def _run(command, timeout, log=None):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if log is not None:
        with Path(log).open("x") as stream:
            stream.write(result.stdout + "\nSTDERR\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Remote command failed ({result.returncode}): {result.stderr[-2000:] or result.stdout[-2000:]}")
    return result.stdout.strip()


PREFLIGHT = '''import json,os,sys,hashlib,importlib.util,torch,tensorrt as trt
from pathlib import Path
config=json.loads(sys.argv[1]); platform=Path('/proc/device-tree/model').read_text().strip('\\x00\\n')
cap=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
name=torch.cuda.get_device_name(0) if cap else 'unavailable'
verified=cap==[8,7] and 'orin' in platform.lower() and 'orin' in name.lower()
power=False; reason='No approved readable integrating sensor provider configured; INA3221 permissions cannot be bypassed'
spec=config.get('power_reader')
if spec:
 try:
  path=Path(spec['path']).resolve()
  assert Path(config['remote_root']) in path.parents
  assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
  module_spec=importlib.util.spec_from_file_location('approved_reader',str(path))
  module=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
  probe=module.open_reader().probe(); power=probe.get('available') is True; reason=probe.get('reason','Sensor probe completed')
 except Exception as exc: reason='Approved sensor unavailable: '+str(exc)
print(json.dumps(dict(target_verified=verified,hardware=dict(name=name,capability=cap,platform=platform),
 runtime=dict(torch=torch.__version__,tensorrt=trt.__version__,python=sys.version.split()[0]),
 power_available=power,power_reason=reason)))
'''


def preflight_remote(transport=None):
    config = transport_config(transport)
    command = shlex.join([config["python"], "-c", PREFLIGHT, json.dumps(config)])
    try:
        result = json.loads(_run(["ssh", *SSH_OPTIONS, config["host"], command], 30))
        verified = result.get("target_verified") is True and result.get("hardware", {}).get("capability") == [8, 7]
        return {**result, "target_verified": verified, "status": "READY" if verified else "FAILED",
                "power_available": result.get("power_available") is True}
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"status": "FAILED", "target_verified": False, "power_available": False,
                "power_reason": "Power unverified", "reason": str(exc), "hardware": {}, "runtime": {}}


def _validate_source(candidate, model, samples, precision):
    from .model import structure_digest
    from .candidate_artifacts import model_state_sha256
    if candidate.get("initial_weight_sha256") != REFERENCE_SHA256 or candidate.get("seed") != 17:
        raise ValueError("Common source weight/seed identity differs")
    if structure_digest(model) != candidate.get("structure_digest"):
        raise ValueError("Actual source model structure identity differs")
    if candidate.get("method_id") != "copt" or candidate.get("source_model_state_sha256") is not None:
        state_sha = candidate.get("source_model_state_sha256")
        artifact_sha = candidate.get("candidate_artifact_sha256")
        artifact = candidate.get("candidate_artifact")
        if (not isinstance(state_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", state_sha)
                or not isinstance(artifact_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha)
                or not isinstance(artifact, str) or not Path(artifact).is_file()
                or sha(artifact) != artifact_sha):
            raise ValueError("Physical source artifact SHA256 is missing or differs")
        if model_state_sha256(model) != state_sha:
            raise ValueError("Actual source model state/parameter SHA256 differs from artifact")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", candidate.get("run_id", "")):
        raise ValueError("Unsafe run_id")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate.get("sample_manifest_sha256", "")):
        raise ValueError("Authenticated Task2 sample manifest SHA required")
    if precision not in ("FP32", "FP16"):
        raise ValueError("Transport currently supports explicit FP32/FP16, not uncalibrated INT8")
    if candidate.get("precision_semantics") != precision:
        raise ValueError("Candidate precision semantics differ from requested transport precision")
    if not 1 <= len(samples) <= 256 or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Expected 1..256 distinct sample IDs")
    for sample in samples:
        tensor = sample["input"]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (1, 3, 224, 224) or not torch.isfinite(tensor).all():
            raise ValueError("Finite preprocessed RGB224 CPU sample required")
        if not sample.get("is_synthetic", False) and (type(sample.get("label")) is not int or not 0 <= sample["label"] < 1000):
            raise ValueError("Real sample requires class label 0..999")


def prepare_transfer(candidate, model, samples, output_root, *, precision="FP16", transport=None):
    """Clone source graph, compute its own logits, export ONNX13 and freeze bytes."""
    config = transport_config(transport)
    _validate_source(candidate, model, samples, precision)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".transfer-reserved").open("x"):
        pass
    upload = root / "upload"
    code = upload / "framework/resnet_compare"
    code.mkdir(parents=True)
    source = copy.deepcopy(model).cpu().eval()
    inputs = [s["input"].detach().cpu().float().contiguous() for s in samples]
    with torch.no_grad():
        logits = [source(t).detach().cpu().float().numpy() for t in inputs]
    if any(x.shape != (1, 1000) or not np.isfinite(x).all() for x in logits):
        raise ValueError("Candidate source must produce finite 1000 logits")
    torch.onnx.export(source, inputs[0], str(upload / "candidate.onnx"), opset_version=13,
                      input_names=["input"], output_names=["logits"], do_constant_folding=True)
    tensors = np.concatenate([x.numpy() for x in inputs])
    scores = np.concatenate(logits)
    np.savez(upload / "samples.npz", inputs=tensors, source_logits=scores)
    rows = [{"sample_id": sample["sample_id"], "label": sample.get("label"),
             "is_synthetic": sample.get("is_synthetic", False),
             "input_sha256": hashlib.sha256(tensors[i].tobytes()).hexdigest(),
             "source_logits_sha256": hashlib.sha256(scores[i].tobytes()).hexdigest()}
            for i, sample in enumerate(samples)]
    for name in CODE_FILES:
        (code / name).write_bytes((Path(__file__).parent / name).read_bytes())
    body = {"schema_version": "resnet_compare_transfer_v1", "candidate": copy.deepcopy(candidate),
            "precision": precision, "samples": rows, "sample_rows_sha256": digest(rows),
            "export_sha256": sha(upload / "candidate.onnx"), "sample_pack_sha256": sha(upload / "samples.npz"),
            "code_sha256": {name: sha(code / name) for name in CODE_FILES}, "transport": config}
    manifest = {**body, "transport_id": digest(body)}
    write_json(upload / "transfer_manifest.json", manifest)
    write_json(root / "transfer_manifest.json", manifest)
    return manifest


def _dispatch(manifest, root):
    config = manifest["transport"]
    job = str(Path(config["remote_root"]) / manifest["candidate"]["run_id"] / manifest["transport_id"])
    manifest_sha = sha(root / "upload/transfer_manifest.json")
    check = ("from pathlib import Path; import hashlib; p=Path(" + repr(job) + "); "
             "exists=p.exists(); "
             "assert not exists or hashlib.sha256((p/'transfer_manifest.json').read_bytes()).hexdigest()==" + repr(manifest_sha) + "; "
             "p.mkdir(parents=True,exist_ok=True); print('EXISTS' if exists else 'NEW')")
    state = _run(["ssh", *SSH_OPTIONS, config["host"], shlex.join([config["python"], "-c", check])], 30, root / "reserve.log")
    if state == "NEW":
        files = [root / "upload" / name for name in ("framework", "transfer_manifest.json", "candidate.onnx", "samples.npz")]
        _run(["scp", *SSH_OPTIONS, "-r", *map(str, files), config["host"] + ":" + job + "/"], config["timeout_s"], root / "upload.log")
    elif state != "EXISTS":
        raise ValueError("Unexpected target reservation response")
    command = "cd " + shlex.quote(job) + " && " + shlex.join([
        "timeout", "--signal=TERM", "--kill-after=15s", str(config["timeout_s"]), config["python"],
        "-m", "framework.resnet_compare.target_entry", job])
    _run(["ssh", *SSH_OPTIONS, config["host"], command], config["timeout_s"] + 30, root / "execute.log")
    _run(["scp", *SSH_OPTIONS, "-r", config["host"] + ":" + job + "/result", str(root / "target")], 120, root / "fetch.log")
    return job


def collect_result(manifest, root, remote_job):
    root = Path(root).resolve()
    target = root / "target"
    receipt = json.loads((target / "receipt.json").read_text())
    if receipt["transport_id"] != manifest["transport_id"] or receipt["manifest_sha256"] != sha(root / "upload/transfer_manifest.json"):
        raise ValueError("Fetched transport identity mismatch")
    if set(receipt["files_sha256"]) != set(RESULT_FILES):
        raise ValueError("Fetched evidence file set mismatch")
    for name in RESULT_FILES:
        if sha(target / name) != receipt["files_sha256"][name]:
            raise ValueError("Fetched evidence SHA differs: " + name)
    result = validate_result(json.loads((target / "baseline_runtime.json").read_text()))
    for key in ("run_id", "method_id", "seed", "candidate_id", "source_commit", "initial_weight_sha256", "structure_digest", "sample_manifest_sha256", "precision_semantics"):
        if result[key] != manifest["candidate"].get(key):
            raise ValueError("Fetched candidate identity mismatch: " + key)
    for key in ("candidate_artifact_sha256", "source_model_state_sha256", "source_bundle_sha256"):
        if key in result and result[key] != manifest["candidate"].get(key):
            raise ValueError("Fetched source parameter/artifact state differs: " + key)
    if result.get("actual_backend") is not None and result.get("requested_precision") != manifest["precision"]:
        raise ValueError("Fetched requested precision differs from authenticated build request")
    target_accepted = result["status"] == "READY" and result["energy_j"] is not None
    result = {**result, "transport_id": manifest["transport_id"], "export_sha256": manifest["export_sha256"],
              "sample_pack_sha256": manifest["sample_pack_sha256"], "remote_job": remote_job,
              **{key: manifest["candidate"][key] for key in
                 ("candidate_artifact_sha256", "source_model_state_sha256", "source_bundle_sha256")
                 if key in manifest["candidate"]},
              "provenance_verified": False, "target_measurement_accepted": target_accepted,
              "evidence_paths": [str(root / "baseline_runtime.json"), *[str(target / name) for name in RESULT_FILES],
                                 str(target / "receipt.json"), str(root / "transfer_manifest.json")]}
    if target_accepted:
        result = {**result, "status": "PARTIAL", "precondition": "NEEDS_PROVENANCE",
                  "reason": "NEEDS_PROVENANCE: target measurement accepted, reference/sample claims require runner attestation"}
    elif result["energy_j"] is None and result["status"] == "READY":
        result = {**result, "status": "PARTIAL", "reason": "Target graph/numerics measured; NEEDS_POWER for three-objective feedback"}
    result = validate_result(result)
    write_json(root / "baseline_runtime.json", result)
    return result


def measure_remote_candidate(candidate, model, samples, output_root, *, precision="FP16", transport=None):
    """Real export→SSH→target-local build/complete→fetch; no queued fake hook."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "baseline_runtime.json").exists():
        raise FileExistsError("Existing local result preserved")
    if not samples:
        result = {**_base(candidate), "reason": "NEEDS_DATA: no ImageNet samples supplied", "precondition": "NEEDS_DATA",
                  "provenance_verified": False, "target_measurement_accepted": False,
                  "evidence_paths": [str(root / "baseline_runtime.json")]}
        write_json(root / "baseline_runtime.json", validate_result(result))
        return result
    try:
        manifest = prepare_transfer(candidate, model, samples, root, precision=precision, transport=transport)
        preflight = preflight_remote(transport)
        write_json(root / "preflight.json", preflight)
        if not preflight["target_verified"]:
            raise RuntimeError("Orin SSH preflight failed: " + preflight.get("reason", "target identity differs"))
        job = _dispatch(manifest, root)
        return collect_result(manifest, root, job)
    except FileExistsError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result = {**_base(candidate), "status": "FAILED", "reason": f"Transport {type(exc).__name__}: {exc}",
                  "provenance_verified": False, "target_measurement_accepted": False,
                  "evidence_paths": [str(root / "baseline_runtime.json")]}
        write_json(root / "baseline_runtime.json", validate_result(result))
        return result
