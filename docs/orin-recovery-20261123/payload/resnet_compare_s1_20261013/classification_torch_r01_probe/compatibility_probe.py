"""Separate 256-input target compatibility diagnostic; never formal timings."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import sys,torch
sys.path.insert(0,str(Path(__file__).parent/'runtime_snapshot'))
from framework.resnet_compare.trt_runtime import ScriptRuntime
from framework.resnet_compare.runtime import _check_target
from framework.resnet_compare.remote import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--metadata',required=True);p.add_argument('--samples',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 metadata=json.loads(Path(a.metadata).read_text())
 assert sha(a.module)==metadata['artifact_sha256']
 assert sha(a.samples)==metadata['source_pack_sha256']
 assert sha(a.reference)==metadata['reference_logits_sha256']
 output=Path(a.output);output.mkdir(exist_ok=False)
 torch.set_num_threads(4)
 runtime=ScriptRuntime(a.module,metadata['adapted_model_state_sha256']);_check_target(runtime,'orin')
 inputs=np.load(a.samples)['inputs'];reference=np.load(a.reference)['logits'];assert inputs.shape==(256,3,224,224) and reference.shape==(256,1000)
 actual=np.concatenate([runtime.infer(torch.from_numpy(x[None]))['logits'].float().numpy() for x in inputs])
 close=np.isclose(actual,reference,atol=.01,rtol=.01)
 np.savez(output/'logits.npz',logits=actual)
 (output/'result.json').write_text(json.dumps({'scope':'compatibility_only_not_formal_measurement','artifact_sha256':sha(a.module),'state_sha256':metadata['adapted_model_state_sha256'],'reference_sha256':sha(a.reference),'samples_sha256':sha(a.samples),'samples':256,'tolerance':{'atol':.01,'rtol':.01},'failed_samples':np.flatnonzero(~close.all(1)).tolist(),'max_abs':float(abs(actual-reference).max()),'top1_disagreements':int((actual.argmax(1)!=reference.argmax(1)).sum()),'torch_version':torch.__version__,'script_evidence':runtime.script_evidence},indent=2))
 print(json.dumps({'allclose':bool(close.all()),'output':str(output)}))
if __name__=='__main__':main()
