"""Frozen three-model CPU ORT identity; no model execution."""
import hashlib,json
from pathlib import Path
from cpu_session import CONFIG
CODE_ROOT=Path(__file__).resolve().parent
FINAL='34117fa447b6cc4236b59e3f2c804f2a69fd01f221b25bcc7cf0983400e077a0'
FINAL_ROWS='98395d0907d3bee96b59ed4baa9d5aa6d0b693599d33163a3fc7aa048595d2c5'
ORT_PYBIND='4ec12754e38ce571d5d0f14d6658e9646ae7f36a9eab3abe63c1bae06886ba5c'
PINS={'dense':{'onnx':'232f666524f07b9d3971381462365774222f0488f6912e144f73c50d36d2fc21','manifest':'bded4e74c656424bf38cce70b2375ff61107ff4e2eb5e0817d84a1aa006427b2','final':FINAL},
      'copt':{'onnx':'55b4b63df303d8702dbe8d0357508266000b50168240098780c9a1009efdf6ad','manifest':'d66f3318631727964d3f0e0893dbf6d270c46117a59fab440a289aab0b25279d','final':FINAL},
      'geta':{'onnx':'38f01f97bf8d83154fedffbbd83f212fcb3437bd882b25e763723dfd1e4e7121','manifest':'31887a0d906f1e9be813c4df245678ec90290feb829950d4e324c88f678fce44','final':FINAL}}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def code_identity():
    names=('cpu_identity.py','cpu_backend.py','cpu_session.py','cpu_bootstrap.py','cpu_target.py','cpu_host.py','frozen_stream.py')
    return {n:sha(CODE_ROOT/n) for n in names}

def validate(point,args):
    if point not in PINS:raise ValueError('unknown point')
    pins=PINS[point]
    for name in ('onnx','manifest','final'):
        if sha(getattr(args,name))!=pins[name]:raise ValueError(name+' SHA mismatch')
    final=json.loads(Path(args.final).read_text())
    if final['count']!=50000 or len(final['samples'])!=50000 or final['sha256']!=FINAL_ROWS:raise ValueError('full final split mismatch')
    manifest=json.loads(Path(args.manifest).read_text())
    return {'point':point,'candidate_id':manifest['candidate']['candidate_id'],'model_sha256':pins['onnx'],
            'source_sha256':dict(pins),'manifest_identity':pins['final'],'final_sample_manifest_sha256':FINAL_ROWS,
            'execution_provider':'CPUExecutionProvider','execution_precision':'FP32','ort_version':'1.27.0',
            'ort_pybind_sha256':ORT_PYBIND,'session_config':dict(CONFIG),'code_sha256':code_identity(),
            'selection_used':False,'timing_or_energy_qualified':False}
