"""Local measurement of the frozen Pruner TVM graph, without tuning/build/RPC.

Run only on the authorized Orin, in its existing native TVM runtime environment.
All executable graph artifacts and real source samples must be SHA-bound in a
new job_manifest.json. Results are exclusive-create and never reused/overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


APPROVED_ROOT = Path("/home/jichengzhi/resnet_compare_s1_20261013")
ARTIFACTS = frozenset(("graph_json", "params", "library", "samples", "samples_manifest"))
RTOL, ATOL = 1e-3, 1e-4


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def graph_input_contract(path):
    """Resolve the external argument entry, not an internal layout-transform output."""
    graph = json.loads(Path(path).read_text())
    try:
        indices = [i for i, node in enumerate(graph["nodes"]) if node["name"] == "input0"]
        if len(indices) != 1:
            raise ValueError("Frozen graph needs one named input0")
        index = indices[0]
        node = graph["nodes"][index]
        row_ptr = graph["node_row_ptr"]
        if (index not in graph["arg_nodes"] or node["op"] != "null" or node["inputs"]
                or row_ptr[index+1] - row_ptr[index] != 1):
            raise ValueError("Frozen graph input0 must be a single external argument")
        entry = row_ptr[index]
        contract = {"name": "input0", "shape": graph["attrs"]["shape"][1][entry],
                    "dtype": graph["attrs"]["dltype"][1][entry]}
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Malformed frozen graph input contract") from exc
    if contract != {"name": "input0", "shape": [1, 3, 224, 224], "dtype": "float32"}:
        raise ValueError("Frozen graph external input must be float32 NCHW [1,3,224,224]")
    return contract


def verify_job(root):
    root = Path(root).resolve()
    if APPROVED_ROOT not in root.parents:
        raise ValueError("Job path outside approved root")
    manifest = json.loads((root / "job_manifest.json").read_text())
    expected = {"schema_version": "pruner_local_measure_v1", "job_id": root.name,
                "approved_root": str(APPROVED_ROOT), "precision": "float32",
                "input_name": "input0", "input_layout": "NCHW", "sample_layout": "NCHW",
                "runner_sha256": sha(__file__)}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen job identity/precision/layout/runner SHA differs")
    if set(manifest.get("artifacts", {})) != ARTIFACTS:
        raise ValueError("Frozen artifact inventory differs")
    paths = {}
    for key, spec in manifest["artifacts"].items():
        path = Path(spec["path"])
        if not path.is_absolute() or APPROVED_ROOT not in path.resolve().parents:
            raise ValueError("Artifact outside approved root: " + key)
        if sha(path) != spec.get("sha256"):
            raise ValueError("Artifact SHA differs: " + key)
        paths[key] = path.resolve()
    graph_input_contract(paths["graph_json"])
    power = manifest.get("power_reader")
    if power is not None:
        path = Path(power["path"]).resolve()
        if APPROVED_ROOT not in path.parents or sha(path) != power.get("sha256"):
            raise ValueError("Power reader path/SHA differs")
    return manifest, paths


def unpack_samples(paths):
    rows = json.loads(paths["samples_manifest"].read_text())["samples"]
    if len(rows) != 256 or len({r["sample_id"] for r in rows}) != 256:
        raise ValueError("Need exactly 256 unique real samples")
    if any(not isinstance(r["sample_id"], str) or not r["sample_id"]
           or r.get("is_synthetic") is not False for r in rows):
        raise ValueError("Samples must be identified real, nonsynthetic source inputs")
    with np.load(paths["samples"], allow_pickle=False) as pack:
        inputs, logits = pack["inputs"], pack["source_logits"]
    if (inputs.shape != (256, 3, 224, 224) or logits.shape != (256, 1000)
            or inputs.dtype != np.float32 or logits.dtype != np.float32
            or not np.isfinite(inputs).all() or not np.isfinite(logits).all()):
        raise ValueError("Source sample shape/dtype/finite contract differs")
    samples = []
    for index, row in enumerate(rows):
        if (hashlib.sha256(inputs[index].tobytes()).hexdigest() != row["input_sha256"]
                or hashlib.sha256(logits[index].tobytes()).hexdigest() != row["source_logits_sha256"]):
            raise ValueError("Per-sample input/source logit SHA differs: " + row["sample_id"])
        samples.append({**row, "input": np.ascontiguousarray(inputs[index:index+1]),
                        "source_logits": logits[index:index+1].copy()})
    return samples


@contextmanager
def serial_target():
    with (APPROVED_ROOT / ".orin-runtime.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_native_graph(paths):
    if os.environ.get("TVM_USE_RUNTIME_LIB") != "1":
        raise ValueError("Use the existing TVM runtime only: TVM_USE_RUNTIME_LIB=1 required")
    import tvm
    from tvm.contrib import graph_runtime

    ctx = tvm.gpu(0)  # The frozen Pruner fork predates the tvm.cuda alias.
    if not ctx.exist:
        raise RuntimeError("Authorized local CUDA device unavailable")
    library = tvm.runtime.load_module(str(paths["library"]))
    graph = graph_runtime.create(paths["graph_json"].read_text(), library, ctx)
    graph.load_params(bytearray(paths["params"].read_bytes()))
    ctx.sync()
    return graph, ctx


def load_power(spec):
    try:
        from framework.resnet_compare.target_entry import load_power_reader

        return load_power_reader(spec, str(APPROVED_ROOT))
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        return None, "Approved named-rail reader unavailable: " + str(exc)


def infer(graph, sample):
    graph.set_input("input0", sample["input"])
    graph.run()
    return graph.get_output(0).asnumpy()


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(np.mean(data)), "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)), "min": float(np.min(data)), "max": float(np.max(data))}


def numerical_check(graph, samples, output_root):
    outputs, passed = [], True
    with (output_root / "numerical.jsonl").open("x") as stream:
        for sample in samples:
            output = infer(graph, sample)
            finite = bool(np.isfinite(output).all())
            same_shape = output.shape == (1, 1000)
            matches = same_shape and finite and bool(np.allclose(
                output, sample["source_logits"], rtol=RTOL, atol=ATOL))
            error = float(np.max(np.abs(output - sample["source_logits"]))) if same_shape and finite else None
            row = {"sample_id": sample["sample_id"], "shape": list(output.shape),
                   "finite": finite, "allclose": matches, "max_abs_error": error,
                   "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                   "source_logits_sha256": sample["source_logits_sha256"]}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            outputs.append(output)
            passed = passed and matches
    with (output_root / "outputs.npz").open("xb") as stream:
        if all(value.shape == (1, 1000) for value in outputs):
            np.savez_compressed(stream, outputs=np.concatenate(outputs))
        else:
            np.savez_compressed(stream, **{"sample_%03d" % i: value for i, value in enumerate(outputs)})
    return passed


def sensor_evidence(evidence, started, completed):
    valid = (isinstance(evidence, dict) and evidence.get("method") == "sensor_integral"
             and evidence.get("rail_label") == "VIN_SYS_5V0" and evidence.get("sensor_id")
             and type(evidence.get("sample_count")) is int and evidence["sample_count"] >= 2
             and all(isinstance(evidence.get(k), (float, int)) and math.isfinite(evidence[k])
                     for k in ("energy_j", "start_monotonic_s", "end_monotonic_s"))
             and evidence["energy_j"] > 0 and evidence["start_monotonic_s"] <= started
             and evidence["end_monotonic_s"] >= completed)
    if not valid:
        raise ValueError("Need actual VIN_SYS_5V0 sensor integral enclosing completed CPU output")
    return evidence


def service_measure(graph, ctx, samples, reader, reason, stream):
    latencies, energies = [], []
    for window in range(3):
        for index in range(200):
            sample = samples[index]
            ctx.sync()
            if reader is not None:
                try:
                    reader.start_request()
                except (OSError, ValueError, RuntimeError) as exc:
                    reader, reason = None, "Named-rail read failed: " + str(exc)
            started = time.monotonic()
            output = infer(graph, sample)
            completed = time.monotonic()
            if output.shape != (1, 1000) or not np.isfinite(output).all():
                raise ValueError("Service output invalid")
            evidence = None
            if reader is not None:
                try:
                    evidence = sensor_evidence(reader.finish_request(), started, completed)
                except (OSError, ValueError, RuntimeError) as exc:
                    reader, reason = None, "Named-rail read failed: " + str(exc)
            latency = (completed - started) * 1000
            energy = evidence["energy_j"] if evidence else None
            row = {"window": window, "request": index, "sample_id": sample["sample_id"],
                   "latency_ms": latency, "energy_j": energy, "energy_evidence": evidence,
                   "start_monotonic_s": started, "end_monotonic_s": completed,
                   "completion": "cpu_output_materialized", "output_sha256": hashlib.sha256(output.tobytes()).hexdigest()}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            latencies.append(latency)
            energies.append(energy)
    return latencies, energies, reason


def forward_measure(graph, ctx, sample, stream):
    graph.set_input("input0", sample["input"])
    all_ms = []
    for window in range(3):
        ctx.sync()
        timer = graph.module.time_evaluator("run", ctx, number=1, repeat=200, min_repeat_ms=0)
        seconds = timer().results
        ctx.sync()
        if len(seconds) != 200 or any(not math.isfinite(v) or v <= 0 for v in seconds):
            raise ValueError("TVM event timing must return 200 positive real observations")
        for index, value in enumerate(seconds):
            stream.write(json.dumps({"window": window, "repeat": index, "latency_ms": value * 1000,
                                     "sample_id": sample["sample_id"]}) + "\n")
            all_ms.append(value * 1000)
        stream.flush()
    return {"latency_ms": distribution(all_ms), "implicit_warmup_runs": 3,
            "paired_with_service_requests": False, "sample_id": sample["sample_id"],
            "boundary": "TVM CUDA event time_evaluator(run), excludes set_input/get_output/CPU/RPC",
            "number": 1, "repeat": 200, "windows": 3, "min_repeat_ms": 0}


def run_job(root):
    root = Path(root).resolve()
    manifest, paths = verify_job(root)
    with serial_target():
        output_root = root / "result"
        output_root.mkdir()  # Refuse even incomplete earlier output directories.
        result = {"job_id": manifest["job_id"], "actual_backend": "native_tvm_cuda",
                  "precision": "float32", "rtol": RTOL, "atol": ATOL, "energy_j": None,
                  "numerical_passed": False, "manifest_sha256": sha(root / "job_manifest.json"),
                  "artifacts": manifest["artifacts"], "runner_sha256": manifest["runner_sha256"],
                  "input_layout": "NCHW", "graph_input_contract": graph_input_contract(paths["graph_json"]),
                  "service_boundary": "CPU wall set_input(NCHW) -> run(includes graph-internal layout transforms) -> get_output(0).asnumpy; excludes decode/preprocess/network/build",
                  "energy_scope": "VIN_SYS_5V0 named rail only, not GPU-only or whole-board",
                  "warmup_requests": 50, "windows": 3, "requests_per_window": 200,
                  "provenance_ready": False}
        try:
            samples = unpack_samples(paths)
            graph, ctx = load_native_graph(paths)
            passed = numerical_check(graph, samples, output_root)
            result = {**result, "numerical_passed": passed}
            with (output_root / "timings.jsonl").open("x") as service, (output_root / "forward_timings.jsonl").open("x") as forward:
                if passed:
                    for index in range(50):
                        infer(graph, samples[index])
                    ctx.sync()
                    reader, reason = load_power(manifest.get("power_reader"))
                    times, energies, reason = service_measure(graph, ctx, samples, reader, reason, service)
                    energy = float(np.mean(energies)) if all(v is not None for v in energies) else None
                    result = {**result, "status": "MEASURED" if energy is not None else "PARTIAL",
                              "service": distribution(times), "energy_j": energy, "energy_reason": reason,
                              "forward": forward_measure(graph, ctx, samples[0], forward)}
                else:
                    result = {**result, "status": "FAILED_NUMERICAL", "reason": "Fixed 256-source FP32 allclose failed; no latency/energy measured"}
        except Exception as exc:
            write_json(output_root / "summary.json", {**result, "status": "FAILED_RUNTIME", "reason": str(exc)})
            raise
        write_json(output_root / "summary.json", result)
        write_json(output_root / "receipt.json", {"manifest_sha256": result["manifest_sha256"],
                   "files_sha256": {p.name: sha(p) for p in output_root.iterdir() if p.is_file()}})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_job(args.job_root)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] in ("MEASURED", "PARTIAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
