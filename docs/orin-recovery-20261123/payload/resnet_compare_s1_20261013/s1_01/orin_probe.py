"""Sequential real Orin structural probe, deliberately NOT ImageNet accuracy."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet50
from framework.resnet_compare.runtime import measure_candidate, TorchRuntime
from framework.resnet_compare.trt_runtime import TensorRTRuntime, build_engine
from framework.resnet_compare.contracts import REFERENCE_SHA256

root = Path('/home/jichengzhi/resnet_compare_s1_20261013/s1_01')
torch.set_num_threads(4)
identity = json.loads((root / 'source_identity.json').read_text())
checkpoint = root / 'resnet50-0676ba61.pth'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == REFERENCE_SHA256
assert hashlib.sha256((root / 'dense_v1.onnx').read_bytes()).hexdigest() == identity['onnx_sha256']
base = dict(run_id='s1_01', method_id='copt', seed=17,
            source_commit=identity['source_commit'], initial_weight_sha256=REFERENCE_SHA256,
            structure_digest=identity['structure_digest'],
            sample_manifest_sha256=identity['sample_manifest_sha256'])
data = torch.from_numpy(np.load(root / 'synthetic_input.npy'))
samples = [dict(sample_id='seed17_synthetic_normal_0_NOT_ImageNet', input=data,
                source_logits=torch.from_numpy(np.load(root / 'source_logits.npy')),
                is_synthetic=True)]
results = {}
started = time.perf_counter()
model = resnet50(pretrained=False, num_classes=1000)
model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
model.eval()
adapter = TorchRuntime(model, 'cuda:0')
initialization = time.perf_counter() - started
result = measure_candidate(dict(base, candidate_id='D0_dense_pytorch_fp32', precision_semantics='FP32'),
    dict(target='orin', adapter=adapter, precision='FP32'), samples, root / 'D0')
(root / 'D0' / 'initialization.json').write_text(json.dumps(dict(seconds=initialization, torch_version=torch.__version__,
    tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32), indent=2))
results['D0'] = result
print('D0', result['status'], result['reason'], flush=True)
del adapter, model
torch.cuda.empty_cache()
for anchor, precision in [('D1', 'FP32'), ('D2', 'FP16')]:
    engine = root / (anchor + '.engine')
    build = build_engine(root / 'dense_v1.onnx', engine, precision)
    (root / (anchor + '_build.json')).write_text(json.dumps(build, indent=2))
    started = time.perf_counter()
    adapter = TensorRTRuntime(engine)
    initialization = time.perf_counter() - started
    result = measure_candidate(dict(base, candidate_id=anchor + '_dense_trt_' + precision, precision_semantics=precision),
        dict(target='orin', adapter=adapter, precision=precision), samples, root / anchor)
    (root / anchor / 'engine_inspector_raw.json').write_text(json.dumps(adapter.inspection, indent=2))
    (root / anchor / 'initialization.json').write_text(json.dumps(dict(seconds=initialization), indent=2))
    results[anchor] = result
    print(anchor, result['status'], result['reason'], flush=True)
    del adapter
    torch.cuda.empty_cache()
results['D3'] = measure_candidate(dict(base, candidate_id='D3_dense_trt_INT8', precision_semantics='INT8'),
    dict(target='orin', precision='INT8'), samples, root / 'D3')
(root / 'baseline_runtime.json').write_text(json.dumps(dict(status='PARTIAL',
    scope='NOT_ImageNet_NOT_formal_accuracy_synthetic_structural_only',
    reason='NEEDS_DATA: ImageNet absent; D3 calibration unexecuted', anchors=results), indent=2))
print(json.dumps({key: dict(status=row['status'], latency_ms=row['latency_ms'],
    actual_precision=row.get('actual_precision'), numerical_verified=row['numerical_verified'])
    for key, row in results.items()}, indent=2), flush=True)
