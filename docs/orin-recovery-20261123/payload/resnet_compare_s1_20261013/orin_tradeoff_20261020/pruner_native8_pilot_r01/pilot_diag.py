"""Short native-graph diagnosis, NOT the 256/600-request acceptance protocol."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scripts import resnet_pruner_local_measure as native


def main():
    root = Path(sys.argv[1]).resolve()
    manifest, paths = native.verify_job(root)
    with native.serial_target():
        out = root / 'pilot_result'
        out.mkdir()
        samples = native.unpack_samples(paths)
        graph, ctx = native.load_native_graph(paths)
        passed = native.numerical_check(graph, samples[:8], out)
        summary = {'protocol': 'short_diagnostic_not_full_acceptance', 'numerical_requests': 8,
                   'numerical_passed': passed, 'formal_256_completed': False,
                   'formal_600_completed': False, 'actual_backend': 'native_tvm_cuda',
                   'precision': 'FP32', 'energy_j': None, 'invocation_sha256': native.sha(__file__),
                   'manifest_sha256': native.sha(root / 'job_manifest.json'),
                   'artifacts': manifest['artifacts'], 'provenance_ready': False}
        if not passed:
            native.write_json(out / 'summary.json', {**summary, 'status': 'FAILED_NUMERICAL'})
            return 2
        for sample in samples[:5]:
            native.infer(graph, sample)
        ctx.sync()
        reader, reason = native.load_power(manifest.get('power_reader'))
        service, energies = [], []
        with (out / 'timings.jsonl').open('x') as stream:
            for window in range(3):
                for index, sample in enumerate(samples[:5]):
                    ctx.sync()
                    if reader is not None:
                        reader.start_request()
                    start = time.monotonic()
                    output = native.infer(graph, sample)
                    end = time.monotonic()
                    assert output.shape == (1, 1000) and np.isfinite(output).all()
                    evidence = native.sensor_evidence(reader.finish_request(), start, end) if reader is not None else None
                    energy = evidence['energy_j'] if evidence else None
                    latency = (end - start) * 1000
                    stream.write(json.dumps({'window': window, 'request': index, 'sample_id': sample['sample_id'], 'latency_ms': latency, 'completion': 'cpu_output_materialized', 'energy_j': energy, 'energy_evidence': evidence, 'output_sha256': __import__('hashlib').sha256(output.tobytes()).hexdigest()}) + '\n')
                    stream.flush()
                    service.append(latency)
                    energies.append(energy)
        forward = []
        graph.set_input('input0', samples[0]['input'])
        with (out / 'forward_timings.jsonl').open('x') as stream:
            for window in range(3):
                ctx.sync()
                seconds = graph.module.time_evaluator('run', ctx, number=1, repeat=5, min_repeat_ms=0)().results
                ctx.sync()
                assert len(seconds) == 5 and all(np.isfinite(x) and x > 0 for x in seconds)
                for index, value in enumerate(seconds):
                    forward.append(value * 1000)
                    stream.write(json.dumps({'window': window, 'repeat': index, 'latency_ms': value * 1000, 'sample_id': samples[0]['sample_id']}) + '\n')
                stream.flush()
        summary = {**summary, 'status': 'DIAGNOSTIC_ONLY', 'warmup_requests': 5, 'service_requests': 15,
                   'forward_requests': 15, 'forward_implicit_warmups': 3, 'forward_service_paired': False,
                   'forward_ms': native.distribution(forward), 'service_ms': native.distribution(service),
                   'energy_j': float(np.mean(energies)) if all(x is not None for x in energies) else None,
                   'energy_reason': reason, 'energy_scope': 'VIN_SYS_5V0 named rail, not whole-board or GPU-only',
                   'service_boundary': 'CPU set_input -> full graph including layout transform -> CPU output',
                   'forward_boundary': 'TVM CUDA run event, excludes set_input and output copy'}
        native.write_json(out / 'summary.json', summary)
        native.write_json(out / 'receipt.json', {'manifest_sha256': summary['manifest_sha256'], 'files_sha256': {p.name: native.sha(p) for p in out.iterdir() if p.is_file()}})
    return 0


if __name__ == '__main__':
    sys.exit(main())
