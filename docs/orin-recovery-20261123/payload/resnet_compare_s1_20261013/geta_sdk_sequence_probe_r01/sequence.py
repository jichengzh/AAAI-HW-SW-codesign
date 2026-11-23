"""Bounded SDK loop: one frozen sample, one fresh SDK process per segment."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np

INPUT_SHA='685664cf6003a403dba26807dcc33fc256fd2e6cf845c074a0abfec274f82136'
CODE='/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code'
ENGINE='/home/jichengzhi/resnet_compare_s1_20261013/geta_official_rebuild_r01/run_build/candidate_fp16.engine'
LAYERS='/home/jichengzhi/resnet_compare_s1_20261013/geta_official_rebuild_r01/run_build/layers.json'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def record_loop(runtime,x,count,out):
    outputs=[]
    with (out/'rows.jsonl').open('x') as rows:
        for i in range(count):
            y=runtime.infer(x)
            if y.shape!=(1,1000) or y.dtype!=np.float32 or not np.isfinite(y).all():
                np.save(out/('invalid_%05d.npy'%i),y,allow_pickle=False)
                raise ValueError('Invalid output')
            np.save(out/('output_%05d.npy'%i),y,allow_pickle=False)
            rows.write(json.dumps({'index':i,'input_sha256':digest(x.tobytes()),'output_sha256':digest(y.tobytes()),'top1':int(y.argmax())})+'\n');rows.flush();os.fsync(rows.fileno())
            outputs.append(y.reshape(1000))
    np.save(out/'outputs.npy',np.stack(outputs),allow_pickle=False)

def main(out_path,count):
    from types import SimpleNamespace
    out=Path(out_path);out.mkdir(exist_ok=False)
    if count not in (64,256):raise ValueError('unexpected segment length')
    raw=(out.parent.parent/'input2429.f32').read_bytes()
    if digest(raw)!=INPUT_SHA or len(raw)!=3*224*224*4:raise ValueError('input mismatch')
    x=np.frombuffer(raw,dtype='<f4').reshape(1,3,224,224).copy()
    sys.path.insert(0,CODE)
    from sdk_backend import assert_no_torch,library_identity
    from identity import verify_libraries
    from multi_entry import create_runtime
    from stage_logger import bind,record
    assert_no_torch();args=SimpleNamespace(point='geta',engine=ENGINE,layers=LAYERS,output=str(out))
    runtime=None
    try:
        with bind(out/'init_stages.jsonl'):
            runtime=record('create_runtime',lambda:create_runtime(args))
            before=record('library_identity',library_identity)
        verify_libraries(before)
        (out/'libraries.json').write_text(json.dumps(before,indent=2)+'\n')
        (out/'maps_before.txt').write_text(Path('/proc/self/maps').read_text())
        record_loop(runtime,x,count,out)
        after=library_identity();verify_libraries(after);assert_no_torch()
        if before!=after:raise ValueError('library identity changed')
        (out/'maps_after.txt').write_text(Path('/proc/self/maps').read_text())
    finally:
        if runtime is not None:runtime.close()
    (out/'completion.json').write_text(json.dumps({'count':count,'pid':os.getpid(),'input_sha256':INPUT_SHA,'scope':'diagnostic_only'})+'\n')

if __name__=='__main__':
    if len(sys.argv)!=3:raise ValueError('output and segment length required')
    main(sys.argv[1],int(sys.argv[2]))
