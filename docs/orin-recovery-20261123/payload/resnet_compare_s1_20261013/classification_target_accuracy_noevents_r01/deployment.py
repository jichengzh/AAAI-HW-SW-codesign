"""Authenticated metadata-only archive and no-timing-events inference path."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def check(ok,message):
 if not ok:raise ValueError(message)
def validate_deployment(a,export):
 source_sha=sha(a.source_module);actual_sha=sha(a.module)
 check(source_sha==export['artifact_sha256'],'Original export artifact changed')
 audit=json.loads(Path(a.metadata_audit).read_text())
 check(audit.get('status')=='PASS_DIAGNOSTIC_METADATA_ONLY'
       and audit.get('source_sha256')==source_sha and audit.get('diagnostic_sha256')==actual_sha
       and audit.get('state_sha256')==export['adapted_model_state_sha256']
       and audit.get('non_debug_entry_bytes_identical') is True and audit.get('module_code_equal') is True
       and audit.get('canonical_graph_equal') is True and audit.get('qualification') is False,
       'Metadata compatibility audit binding differs')
 entries=[];changed=0
 with zipfile.ZipFile(a.source_module) as old,zipfile.ZipFile(a.module) as new:
  check(old.namelist()==new.namelist() and len(set(old.namelist()))==len(old.namelist()),'Archive inventory/order differs')
  for name in old.namelist():
   original=old.read(name);adapted=new.read(name)
   if not name.endswith('.debug_pkl'):check(original==adapted,'Non-debug archive bytes changed')
   else:changed+=int(original!=adapted)
   entries.append({'name':name,'source_sha256':hashlib.sha256(original).hexdigest(),
                   'adapted_sha256':hashlib.sha256(adapted).hexdigest()})
 check(changed==audit['debug_members_changed'] and changed>0,'Debug-only change count differs')
 return {'source_export_artifact_sha256':source_sha,'artifact_sha256':actual_sha,
         'metadata_audit_sha256':sha(a.metadata_audit),'debug_members_changed':changed,
         'non_debug_entry_bytes_identical':True,
         'archive_entries_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
         'execution_path':'direct_model_CPU_to_CUDA_half_then_blocking_CPU_return',
         'explicit_cuda_timing_events':0,'metadata_audit_is_accuracy_qualification':False}

def no_events_forward(runtime,x):
 import torch
 with torch.no_grad():
  data=torch.from_numpy(x.copy()).to(device=runtime.device,dtype=torch.float16)
  output=runtime.model(data)
  check(output.dtype==torch.float16,'No-event path requires half output')
  cpu=output.to('cpu',non_blocking=False)
  result=cpu.float().numpy()
  check(result.shape==(1,1000) and np.isfinite(result).all(),'Invalid no-event logits')
  return result
