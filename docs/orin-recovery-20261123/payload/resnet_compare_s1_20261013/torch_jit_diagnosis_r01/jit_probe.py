"""Isolated JIT optimization diagnostic; no formal measurement qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def compare(actual,reference):
    if actual.shape!=reference.shape or actual.ndim!=2:
        raise ValueError('logits shape mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('logits must be finite')
    close=np.isclose(actual,reference,atol=.01,rtol=.01)
    return {'failed_samples':np.flatnonzero(~close.all(1)).tolist(),
            'max_abs':float(np.abs(actual-reference).max()),
            'mean_abs':float(np.abs(actual-reference).mean()),
            'top1_disagreements':int((actual.argmax(1)!=reference.argmax(1)).sum()),
            'atol':.01,'rtol':.01,'allclose':bool(close.all())}


def run_inputs(module,inputs,optimized,device,validator):
    # A CLI invocation runs ONE mode: no shared graph executor cache across modes.
    with torch.jit.optimized_execution(True):
        model=torch.jit.load(str(module),map_location='cpu').eval()
        evidence=validator(model) if validator else {}
    with torch.jit.optimized_execution(optimized),torch.no_grad():
        model=model.to(device)
        rows=[]
        for x in inputs:
            y=model(torch.from_numpy(x[None]).to(device=device,dtype=torch.float16))
            if y.dtype!=torch.float16: raise ValueError('Non-half output')
            rows.append(y.float().cpu().numpy())
        graph=torch.jit.last_executed_optimized_graph()
        # Avoid incompatible Torch 1.12 graph string printer.
        evidence={**evidence,'executed_operator_kinds':[n.kind() for n in graph.nodes()]}
    return np.concatenate(rows),evidence


def main():
    p=argparse.ArgumentParser()
    for key in ('module','metadata','samples','reference','runtime-snapshot','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=('optimized','no-opt'),required=True)
    a=p.parse_args()
    sys.path.insert(0,a.runtime_snapshot)
    from framework.resnet_compare.trt_runtime import validate_script
    metadata=json.loads(Path(a.metadata).read_text())
    for path,key in ((a.module,'artifact_sha256'),(a.samples,'source_pack_sha256'),
                     (a.reference,'reference_logits_sha256')):
        if sha(path)!=metadata[key]: raise ValueError('Identity mismatch: '+key)
    inputs=np.load(a.samples)['inputs'];reference=np.load(a.reference)['logits']
    if inputs.shape!=(256,3,224,224) or reference.shape!=(256,1000):
        raise ValueError('Expected complete fixed 256 input/reference contract')
    output=Path(a.output);output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    logits,evidence=run_inputs(a.module,inputs,a.mode=='optimized','cuda:0',
        lambda model:validate_script(model,metadata['adapted_model_state_sha256']))
    np.savez(output/'logits.npz',logits=logits)
    result={'scope':'diagnostic_only_requires_independent_full_accuracy_for_new_path',
            'mode':a.mode,'torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
            'device':torch.cuda.get_device_name(0),'samples':256,'warmup':0,
            'artifact_sha256':sha(a.module),'reference_sha256':sha(a.reference),
            'samples_sha256':sha(a.samples),'diagnostic_code_sha256':sha(__file__),
            'tf32':False,'cudnn_benchmark':False,'comparison':compare(logits,reference),
            'script_evidence':evidence,'logits_sha256':sha(output/'logits.npz')}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['comparison']))

if __name__=='__main__': main()
