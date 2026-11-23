"""Coverage-only engine reuse; these traced timings are NOT benchmark evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from framework.resnet_compare.runtime import measure_candidate, TorchRuntime
from framework.resnet_compare.trt_runtime import TensorRTRuntime

root = Path('/home/jichengzhi/resnet_compare_s1_20261013/s1_01')
base = json.loads((root / 'baseline_runtime.json').read_text())['anchors']['D1']
samples = [dict(sample_id='coverage_only_synthetic', input=torch.from_numpy(np.load(root / 'synthetic_input.npy')),
    source_logits=torch.from_numpy(np.load(root / 'source_logits.npy')), is_synthetic=True)]
for anchor, precision in [('D1', 'FP32'), ('D2', 'FP16')]:
    adapter = TensorRTRuntime(root / (anchor + '.engine'))
    result = measure_candidate(dict(base, candidate_id=anchor + '_coverage_only'),
        dict(target='orin', adapter=adapter, precision=precision, warmup=1, windows=1, requests_per_window=2),
        samples, root / ('coverage_v2_' + anchor))
    assert result['numerical_verified'], result['reason']
    del adapter
    torch.cuda.empty_cache()
# Exercise genuine PyTorch CUDA-event handling with deterministic, nonrandom math.
class Classifier(torch.nn.Module):
    def forward(self, tensor):
        return tensor.mean((1,2,3)).reshape(1,1).repeat(1,1000)
runtime = TorchRuntime(Classifier(), 'cuda:0')
assert runtime.infer(torch.ones(1,3,224,224))['completion'] == 'cuda_event_synchronized'
