import sys,json,fcntl,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path('/home/jichengzhi/resnet_compare_s1_20261013')
BASE=ROOT/'classification_target_accuracy_r01'
sys.path.insert(0,str(BASE/'runtime_snapshot'))
from framework.resnet_compare.trt_runtime import ScriptRuntime
torch.set_num_threads(4)
with (ROOT/'.orin-runtime.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 e=json.loads((BASE/'export.json').read_text());r=ScriptRuntime(BASE/'model.pt',e['adapted_model_state_sha256'])
 x=np.load(Path(__file__).parent/'trigger_chunk.npz')['inputs']
 for i,row in enumerate(x):
  print(json.dumps({'phase':'before','index':2816+i,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'free':torch.cuda.mem_get_info()[0]}),flush=True)
  try:
   y=r.infer(torch.from_numpy(row[None].copy()))['logits']
   print(json.dumps({'phase':'after','index':2816+i,'finite':bool(torch.isfinite(y).all()),'maxabs':y.abs().max().item()}),flush=True)
  except Exception:
   traceback.print_exc();raise
