"""New dense SDK deployment identity; old experiments remain immutable."""
import json
from pathlib import Path
from sdk_backend import sha
ENGINE='2a7110a8ebc458170d77deba53270e25657f04c84b8b8f21400021114c0be7df'
FINAL_ROWS='98395d0907d3bee96b59ed4baa9d5aa6d0b693599d33163a3fc7aa048595d2c5'
PINS={'manifest':'bded4e74c656424bf38cce70b2375ff61107ff4e2eb5e0817d84a1aa006427b2',
      'onnx':'232f666524f07b9d3971381462365774222f0488f6912e144f73c50d36d2fc21',
      'final':'34117fa447b6cc4236b59e3f2c804f2a69fd01f221b25bcc7cf0983400e077a0',
      'layers':'6f94d9216855acf2d9e58a2adae7e7ae51e667a0bfc5a4c4d80a8d894dda0f3d'}
FILES=('sdk_backend.py','sdk_contract.py','sdk_entry.py','host_entry.py','frozen_stream.py','host_cli.py','sdk_wrapper.py','stage_logger.py','h2d_readback.py','d2h_readback.py')
def code_identity():
    root=Path(__file__).parent
    if sha(root/'frozen_stream.py')!='6214df738caa17bb1d51e9b08d7c62658c8c5a89ac21a9de4da16a4684ed9b7d':
        raise ValueError('Frozen stream changed')
    return {name:sha(root/name) for name in FILES}
def validate(a):
    for name,expected in PINS.items():
        if sha(getattr(a,name))!=expected:raise ValueError(name+' SHA mismatch')
    final=json.loads(Path(a.final).read_text())
    if final['count']!=50000 or len(final['samples'])!=50000 or final['sha256']!=FINAL_ROWS:
        raise ValueError('Frozen50000 split differs')
    manifest=json.loads(Path(a.manifest).read_text())
    return {'engine_sha256':ENGINE,'manifest_identity':PINS['final'],
            'source_sha256':dict(PINS),'candidate_id':manifest['candidate']['candidate_id'],
            'execution_path':'TensorRT8.5_CUDART_standard_memory_stream_SDK_r01',
            'final_sample_manifest_sha256':FINAL_ROWS,'code_sha256':code_identity(),
            'selection_used':False,'timing_or_energy_qualified':False}
