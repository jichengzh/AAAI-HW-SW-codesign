"""Single supervised SDK initialization probe. No inference or retries."""
import fcntl, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path

LOCK='/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock'
BOOT='bb2ced1a-bad7-482f-af22-11ef1e4aca61'
ENGINE='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/geta/candidate.engine'
ENGINE_SHA='59f030a76d8565ba888ca354d666b73337309f7e4c8628ae1d6720ed31118690'
LAYERS='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/geta/prepared/contract.json'
LAYERS_SHA='76f8d96599a679b91bab87255989a4a8d2f6d2dac892343d02a952535c43062e'
TRT='/usr/src/tensorrt/bin/trtexec'
TRT_SHA='f511762a3c57a08918923586e902424b1abe40320717addafc9b6923c1ceebb4'
INPUT_SHA='ea4a6f68f067977c0aca4d1b1706fadc63b0dca6c4499cac149b9c9e3d715884'
SOURCE='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/geta/source'
SOURCE_INPUT='/home/jichengzhi/resnet_compare_s1_20261013/trt_sdk_only_diagnostic_r01/input_00000.f32'
PINS={ENGINE:ENGINE_SHA,LAYERS:LAYERS_SHA,TRT:TRT_SHA,SOURCE_INPUT:INPUT_SHA,
      SOURCE+'/transfer_manifest.json':'31887a0d906f1e9be813c4df245678ec90290feb829950d4e324c88f678fce44',
      SOURCE+'/candidate.onnx':'38f01f97bf8d83154fedffbbd83f212fcb3437bd882b25e763723dfd1e4e7121'}
LIBS={
 '/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2':'4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca',
 '/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.5.2':'5a30c8e46ba47c3a93eb825ad0a27cbe969fb6f174ea963e4d22dfe33630097a',
 '/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1':'151ad1343173e919304e08d4c6801508303111f2b1c7b3737f6b5134590808f4',
 '/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298':'6ca38be1cc6fa0b8ddad0313eea2a4483be27d2e8474437f9f7b7a536b7197a2'}


def has_gpu_fault(message):
    return bool(re.search(r'ctxsw checksum|Bad page map|nvgpu.*(?:\[ERR\]|error|fault)|Unhandled fault|BUG:|kernel panic|SError',message,re.I))

def inspect_rows(lines, cursor):
    fault=False
    for line in lines:
        data=json.loads(line)
        cursor=data['__CURSOR']
        fault = fault or has_gpu_fault(data['MESSAGE'])
    return cursor, fault

def command(args):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=5)
    if result.returncode: raise RuntimeError('command failed: '+repr(args)+' '+result.stderr)
    return result.stdout

def read_cursor():
    text=command(['journalctl','-k','-n','0','--show-cursor','--no-pager'])
    match=re.search(r'^-- cursor: (.+)$',text,re.M)
    if not match: raise ValueError('missing kernel cursor')
    return match.group(1)

def kernel_since(cursor, log):
    text=command(['journalctl','-k','--after-cursor',cursor,'-o','json','--no-pager'])
    lines=text.splitlines()
    with open(log,'a') as f:
        for line in lines:f.write(line+'\n')
        f.flush();os.fsync(f.fileno())
    return inspect_rows(lines,cursor)

def lock_free():
    with open(LOCK,'a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        return True

def members(pgid):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            data=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(data[2])==pgid and data[0]!='Z':found.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError):pass
    return sorted(found)

def cleanup(pgid):
    sent=[]
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if not members(pgid):break
        try:os.killpg(pgid,sig);sent.append(sig.name)
        except ProcessLookupError:break
        for _ in range(50):
            if not members(pgid):break
            time.sleep(.1)
    return {'signals':sent,'remaining':members(pgid)}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def model_identity():
    actual={p:digest(p) for p in PINS}
    if actual!=PINS:raise ValueError('Model identity hash mismatch')
    return actual

def signal_failure(signum, frame):
    for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, signal.SIG_IGN)
    raise RuntimeError('Supervisor signal '+str(signum))

def monitor(child, cursor, path, seconds):
    deadline=time.monotonic()+seconds
    while child.poll() is None:
        cursor,fault=kernel_since(cursor,path)
        if fault:return cursor,True,'new kernel fault'
        if time.monotonic()>=deadline:return cursor,False,'timeout'
        time.sleep(.5)
    cursor,fault=kernel_since(cursor,path)
    return cursor,fault,'new kernel fault' if fault else None

def library_identity():
    actual={p:digest(p) for p in LIBS}
    if actual!=LIBS:raise ValueError('Library file hash mismatch')
    return actual

def run(root,mode):
    if mode!='trtexec':raise ValueError('Only trtexec mode allowed')
    root=Path(root).resolve();out=root/('run_'+mode);out.mkdir(exist_ok=False)
    errors=[];child=None;code=None;clean={'signals':[],'remaining':[]};cursor=None;fault=False
    lock_holder=None;start=time.time();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    try:
        if boot!=BOOT:raise ValueError('Unexpected boot')
        if not lock_free():raise ValueError('Device busy')
        power_mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_before.log').write_text(power_mode)
        if 'MODE_30W' not in power_mode:raise ValueError('Unexpected mode')
        (out/'library_before.json').write_text(json.dumps(library_identity(),indent=2)+'\n')
        (out/'model_before.json').write_text(json.dumps(model_identity(),indent=2)+'\n')
        if digest(ENGINE)!=ENGINE_SHA:raise ValueError('Engine SHA mismatch')
        if digest(LAYERS)!=LAYERS_SHA:raise ValueError('Layer SHA mismatch')
        if digest(root/'input0.f32')!=INPUT_SHA:raise ValueError('Input raw SHA mismatch')
        if digest(TRT)!=TRT_SHA:raise ValueError('trtexec SHA mismatch')
        if mode=='trtexec':
            lock_holder=open(LOCK,'a');fcntl.flock(lock_holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cursor=read_cursor();(out/'cursor.txt').write_text(cursor+'\n')
        argv=[TRT,'--loadEngine='+ENGINE,'--loadInputs=input:'+str(root/'input0.f32'),'--iterations=1','--duration=0','--warmUp=0','--streams=1','--avgRuns=1','--exportOutput='+str(out/'trt_output.json'),'--dumpOutput']
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONNOUSERSITE']='1'
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
            (out/'launch.json').write_text(json.dumps({'argv':argv,'pgid':child.pid,'epoch':time.time(),'timeout_s':120})+'\n')
            cursor,fault,error=monitor(child,cursor,out/'kernel.jsonl',120)
            if error:errors.append(error)
    except BaseException as e:errors.append(repr(e))
    finally:
        if child is not None:
            clean=cleanup(child.pid)
            try:code=child.wait(timeout=3)
            except subprocess.TimeoutExpired:errors.append('child remained')
        if cursor is not None:
            try:
                cursor,found=kernel_since(cursor,out/'kernel.jsonl');fault |= found
                if found and 'new kernel fault' not in errors:errors.append('new kernel fault')
                (out/'library_after.json').write_text(json.dumps(library_identity(),indent=2)+'\n')
                (out/'model_after.json').write_text(json.dumps(model_identity(),indent=2)+'\n')
                mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_after.log').write_text(mode)
                if 'MODE_30W' not in mode:errors.append('Post mode mismatch')
            except BaseException as e:errors.append('Post evidence: '+repr(e))
        if lock_holder is not None:lock_holder.close()
        free=lock_free()
        after=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if not free or clean['remaining'] or after!=boot:errors.append('cleanup or boot mismatch')
        if code!=0:errors.append('child exit '+str(code))
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}
        result={'status':'PASS_PENDING_INDEPENDENT_REVIEW' if not errors else 'FAILED','boot':boot,'after_boot':after,'pgid':child.pid if child else None,'exit':code,'errors':errors,'kernel_fault':fault,'cleanup':clean,'lock_free':free,'start':start,'finish':time.time(),'sha256':files}
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_failure)
    if len(sys.argv)!=2 or sys.argv[1]!='trtexec':raise ValueError('Choose trtexec')
    result=run(Path(__file__).resolve().parent,sys.argv[1])
    print(json.dumps(result),flush=True)
    sys.exit(0 if result['status'].startswith('PASS') else 1)
