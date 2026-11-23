"""Contracts for an externally supervised full-val target; no dispatch here."""
from pathlib import Path

def check(ok,message):
    if not ok: raise ValueError(message)

def command(job):
    root=Path(job['root']);args=job['argv']
    check(args==[job['python'],str(root/'target.py'),str(root/'run/target'),str(root/'input_manifest.json')],'fixed bounded entry')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s','600s']+args

def validate_completion(job,identity,completion,maps_before,maps_after):
    check(all(identity.get(k)==job[k] for k in ('point','engine_sha256','boot_id','library_sha256')),'target identity')
    check(identity['torch_imported'] is False and identity['tensorrt_version']=='8.5.2.2','SDK execution')
    check(completion=={'status':'DIAGNOSTIC_COMPLETE','samples':2432,'library_sha256':identity['library_sha256']},'target completion')
    check(all(s and not any(x in s for x in ('libtorch','libc10','/torch/')) for s in (maps_before,maps_after)),'Torch maps')

def validate_stages(data):
    import json
    names=('trt_runtime','deserialize','inspector','load_cudart','cuda_set_device','bindings','context','malloc_input','malloc_output','stream')
    expected=[('create_runtime','begin')]+[(name,event) for name in names for event in ('begin','ok')]+[('create_runtime','ok'),('library_identity','begin'),('library_identity','ok')]
    rows=[json.loads(line) for line in data.splitlines()]
    check([(r['stage'],r['event']) for r in rows]==expected,'initialization stage sequence')
    check(len({r['pid'] for r in rows})==1 and all(type(r['time_ns']) is int for r in rows),'stage process/time')
    check(all(a['time_ns']<=b['time_ns'] for a,b in zip(rows,rows[1:])),'stage time order')
