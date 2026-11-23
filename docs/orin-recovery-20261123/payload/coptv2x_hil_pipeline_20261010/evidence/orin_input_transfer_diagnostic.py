"""One fixed real input repeated: copy-bound diagnostic, not model/HIL timing."""

import gc
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch

INPUT_ROOT = Path('/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/input')
MANIFEST = Path('/home/jichengzhi/coptv2x_hil_pipeline_20261010/evidence/worker_smoke_manifest_20.json')


def main():
    manifest = json.loads(MANIFEST.read_text())
    spec = manifest['samples'][0]['inputs']['spatial_features']
    print(json.dumps({'event': 'input_contract', 'spec': spec, 'torch_version': torch.__version__,
                      'torch_num_threads': torch.get_num_threads(),
                      'device': torch.cuda.get_device_name(0)}), flush=True)
    input_path = INPUT_ROOT / spec['path']
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == spec['sha256']
    value = np.load(str(input_path), allow_pickle=False)
    assert value.shape == (2, 64, 256, 512) and value.dtype == np.float32
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cpu_input = torch.from_numpy(value)
    pinned = [torch.empty(value.shape, dtype=torch.float32, pin_memory=True) for _ in range(2)]
    device = [torch.empty(value.shape, device='cuda', dtype=torch.float32) for _ in range(2)]
    stream = torch.cuda.Stream()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    expected = hashlib.sha256(value.tobytes()).hexdigest()
    for mode in ('pageable_blocking', 'pinned_stage_then_async_stream'):
        rows = []
        gc.collect()
        for index in range(25):
            slot = index % 2
            start = time.perf_counter_ns()
            if mode == 'pinned_stage_then_async_stream':
                pinned[slot].copy_(cpu_input)
            staged = time.perf_counter_ns()
            if mode == 'pageable_blocking':
                start_event.record()
                device[slot].copy_(cpu_input, non_blocking=False)
                end_event.record()
                torch.cuda.synchronize()
            else:
                with torch.cuda.stream(stream):
                    start_event.record(stream)
                    device[slot].copy_(pinned[slot], non_blocking=True)
                    end_event.record(stream)
                end_event.synchronize()
            end = time.perf_counter_ns()
            if index >= 5:
                rows.append({'repeat_index': index - 5, 'cpu_stage_ns': staged - start,
                             'copy_completion_wait_ns': end - staged, 'elapsed_ns': end - start,
                             'cuda_event_ms': start_event.elapsed_time(end_event)})
        actual = hashlib.sha256(device[slot].cpu().numpy().tobytes()).hexdigest()
        assert actual == expected
        print(json.dumps({'event': 'copy_mode', 'mode': mode, 'input_bytes': value.nbytes,
                          'sample_id': manifest['samples'][0]['sample_id'],
                          'unique_input_count': 1, 'warmup_repeats': 5, 'measured_repeats': 20,
                          'rows': rows, 'final_input_sha256': actual, 'input_sha_match': True,
                          'means_ms': {name: float(np.mean([row[name] for row in rows])) / 1e6
                                       for name in ('cpu_stage_ns', 'copy_completion_wait_ns', 'elapsed_ns')},
                          'scope': 'copy_only_includes_cpu_pinning_stage_no_model_no_network_no_hil',
                          'production_enabled': False}), flush=True)


if __name__ == '__main__':
    main()
