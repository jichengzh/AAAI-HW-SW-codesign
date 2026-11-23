"""Single deterministic cuBLAS GEMM. Invoked only by the locked supervisor."""
import ctypes, hashlib, json, os, struct, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gemm_probe import make_matrices, reference_gemm, check_values, CublasAPI

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_hashes(expected):
    actual={p:digest(p) for p in expected}
    if actual!=expected:raise ValueError('file identity mismatch')
    return actual

def pack(path,values):Path(path).write_bytes(struct.pack('<%df'%len(values),*values))
def unpack(path):
    raw=Path(path).read_bytes()
    if len(raw)%4:raise ValueError('bad float32 file')
    return list(struct.unpack('<%df'%(len(raw)//4),raw))
def save_inputs(out,n):
    a,b=make_matrices(n);pack(out/'a.f32',a);pack(out/'b.f32',b);pack(out/'reference.f32',reference_gemm(n,a,b))
def save_output(out,values):pack(out/'output.f32',values)
def validate_output(out,n):
    values=unpack(out/'output.f32');expected=unpack(out/'reference.f32')
    if len(values)!=n*n or len(expected)!=n*n:raise ValueError('shape mismatch')
    return check_values(values,expected)
def check_maps(maps,expected):
    if any(x in maps for x in ('libnvinfer','libtorch')):raise ValueError('unexpected TensorRT or torch mapping')
    for path in expected:
        if not any(line.rstrip().endswith(path) for line in maps.splitlines()):raise ValueError('missing mapped library '+path)
    return True

def run(out,job):
    out=Path(out); expected=job['files'];verify_hashes(expected)
    n=32;save_inputs(out,n)
    def emit(stage,state):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps({'stage':stage,'state':state,'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
    emit('load_libraries','begin')
    cuda=ctypes.CDLL(job['cudart']);blas=ctypes.CDLL(job['cublas'])
    emit('load_libraries','ok')
    api=CublasAPI(cuda,blas,emit)
    actual=api.run(n,unpack(out/'a.f32'),unpack(out/'b.f32'))
    save_output(out,actual)
    maps=Path('/proc/self/maps').read_text();(out/'maps.txt').write_text(maps)
    check_maps(maps,{p:expected[p] for p in (job['cudart'],job['cublas'],job['libcuda'])})
    verify_hashes(expected)
    result=validate_output(out,n)
    (out/'numerical.json').write_text(json.dumps(result)+'\n')
    emit('result','ok')

if __name__=='__main__':
    if len(sys.argv)!=3:raise ValueError('expected output directory and job JSON')
    run(sys.argv[1],json.loads(Path(sys.argv[2]).read_text()))
