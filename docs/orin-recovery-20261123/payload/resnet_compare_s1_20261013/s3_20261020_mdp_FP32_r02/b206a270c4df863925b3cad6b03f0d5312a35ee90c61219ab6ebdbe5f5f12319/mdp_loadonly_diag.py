"""Bounded MDP load-only replay; existing engine, no rebuild or runtime edits."""
import gc
import json
from pathlib import Path
import sys
import time
import torch
from framework.resnet_compare import remote, target_entry
from framework.resnet_compare.runtime import measure_candidate
from framework.resnet_compare.trt_runtime import TensorRTRuntime


def main():
    root, original = map(Path, sys.argv[1:3])
    manifest = target_entry.verify_bundle(root)
    old = json.loads((original / 'transfer_manifest.json').read_text())
    assert root == Path(manifest['transport']['remote_root']) / manifest['candidate']['run_id'] / manifest['transport_id']
    assert all(manifest[k] == old[k] for k in ['export_sha256', 'sample_pack_sha256', 'code_sha256', 'precision'])
    keys = ['method_id', 'candidate_id', 'candidate_artifact_sha256', 'source_model_state_sha256', 'structure_digest', 'initial_weight_sha256', 'seed']
    assert all(manifest['candidate'][k] == old['candidate'][k] for k in keys)
    engine = original / 'candidate.engine'
    assert remote.sha(engine) == 'd205ed3e3c26d5164a05050fe1a74cf6190dbfb39b0383a34d84aef399ef94fa'
    def stage(name):
        with (root / 'process_stages.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': name, 'monotonic': time.monotonic()}) + '\n')
        print(name, flush=True)
    torch.set_num_threads(4)
    with target_entry.serial_target(manifest['transport']['remote_root']):
        stage('load_only_start')
        samples = target_entry.unpack_samples(root, manifest)
        adapter = TensorRTRuntime(engine)
        stage('engine_loaded')
        reader, reason = target_entry.load_power_reader(manifest['transport'].get('power_reader'), manifest['transport']['remote_root'])
        measured = target_entry.PowerRuntime(adapter, reader) if reader is not None else adapter
        result = measure_candidate(manifest['candidate'], {'target': 'orin', 'precision': 'FP32', 'adapter': measured, 'warmup': 50, 'windows': 3, 'requests_per_window': 200}, samples, root / 'result')
        rows = [json.loads(line) for line in (root / 'result/timings.jsonl').read_text().splitlines()]
        energy = [row.get('energy_j') for row in rows]
        if rows and all(value is not None for value in energy):
            result = {**result, 'energy_j': sum(energy) / len(energy), 'energy_reason': 'Actual sensor integrals'}
        else:
            result = {**result, 'energy_reason': reason, 'energy_boundary': None}
        if result['status'] == 'READY' and result['energy_j'] is None:
            result = {**result, 'status': 'PARTIAL', 'reason': 'NEEDS_POWER: load-only replay timing and numerical observations available'}
        (root / 'result/baseline_runtime.json').write_text(json.dumps(result, indent=2) + '\n')
        remote.write_json(root / 'result/build.json', {'diagnostic_load_only': True, 'reused_same_engine_sha': remote.sha(engine), 'source_remote_job': str(original), 'source_build': json.loads((original / 'result/build.json').read_text()), 'invocation_sha256': remote.sha(__file__)})
        remote.write_json(root / 'result/receipt.json', {'transport_id': manifest['transport_id'], 'manifest_sha256': remote.sha(root / 'transfer_manifest.json'), 'files_sha256': {name: remote.sha(root / 'result' / name) for name in remote.RESULT_FILES}})
        stage('measurements_and_receipt_written')
        torch.cuda.synchronize()
        del measured, adapter, samples, reader
        gc.collect()
        stage('adapter_released_and_gc_completed')
        torch.cuda.synchronize()
        stage('synchronized_after_release')
    stage('before_interpreter_exit')
    return 0 if result.get('numerical_verified') and len(rows) == 600 else 2


if __name__ == '__main__':
    sys.exit(main())
