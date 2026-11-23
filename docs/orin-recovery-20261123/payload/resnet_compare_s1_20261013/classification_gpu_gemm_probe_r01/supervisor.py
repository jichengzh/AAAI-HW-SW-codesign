"""One locked, bounded cuBLAS diagnostic; never run in parallel."""
import fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner
LOCK='/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock'
BOOT='bb2ced1a-bad7-482f-af22-11ef1e4aca61'
PYTHON='/home/jichengzhi/uniad_venv/bin/python'
PYTHON_SHA='330ad6775d2db4d95524b249d38e14b9046672b8468fa6fbf7f89b7c683d1f6b'
TIMEOUT=120

def has_gpu_fault(message):
    return bool(re.search(r'ctxsw checksum|Bad page map|nvgpu.*(?:\[ERR\]|error|fault|machine check)|Unhandled fault|BUG:|kernel panic|SError',message,re.I))

def runner_argv(root,out):
    return [PYTHON,'-I',str(Path(root)/'runner.py'),str(out),str(Path(root)/'job.json')]

def exact_mode(text):
    lines=text.splitlines()
    matching=[i for i,line in enumerate(lines) if line.strip().startswith('NV Power Mode:')]
    return len(matching)==1 and lines[matching[0]].strip()=='NV Power Mode: MODE_30W' and matching[0]+1<len(lines) and lines[matching[0]+1].strip()=='2'

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

def run(root,job_sha):
    root=Path(root).resolve();out=root/'run_gemm';out.mkdir(exist_ok=False)
    errors=[];child=None;code=None;clean={'signals':[],'remaining':[]};cursor=None;fault=False;lock_holder=None;start=time.time()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    try:
        if boot!=BOOT:raise ValueError('Unexpected boot')
        if digest(root/'job.json')!=job_sha:raise ValueError('job hash mismatch')
        job=json.loads((root/'job.json').read_text())
        runner.verify_hashes(job['files'])
        if digest(PYTHON)!=PYTHON_SHA:raise ValueError('Python hash mismatch')
        if not lock_free():raise ValueError('Device busy')
        mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_before.log').write_text(mode)
        if not exact_mode(mode):raise ValueError('Unexpected mode')
        lock_holder=open(LOCK,'a');fcntl.flock(lock_holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cursor=read_cursor();(out/'cursor.txt').write_text(cursor+'\n')
        argv=runner_argv(root,out)
        env=dict(os.environ)
        for key in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH'):env.pop(key,None)
        env['PYTHONNOUSERSITE']='1'
        env['LD_LIBRARY_PATH']='/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/tegra'
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
            (out/'launch.json').write_text(json.dumps({'argv':argv,'pgid':child.pid,'epoch':time.time(),'timeout_s':TIMEOUT})+'\n')
            cursor,fault,error=monitor(child,cursor,out/'kernel.jsonl',TIMEOUT)
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
                if digest(root/'job.json')!=job_sha:raise ValueError('job hash changed')
                runner.verify_hashes(job['files'])
                mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_after.log').write_text(mode)
                if not exact_mode(mode):errors.append('Post mode mismatch')
            except BaseException as e:errors.append('Post evidence: '+repr(e))
        if lock_holder is not None:lock_holder.close()
        free=lock_free();after=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if not free or clean['remaining'] or after!=boot:errors.append('cleanup or boot mismatch')
        if code!=0:errors.append('child exit '+str(code))
        if code==0:
            try:
                a,b=runner.make_matrices(32)
                if runner.unpack(out/'a.f32')!=a or runner.unpack(out/'b.f32')!=b:raise ValueError('input mismatch')
                if runner.unpack(out/'reference.f32')!=runner.reference_gemm(32,a,b):raise ValueError('reference mismatch')
                summary=runner.validate_output(out,32)
                runner.check_maps((out/'maps.txt').read_text(),{p:job['files'][p] for p in (job['cudart'],job['cublas'],job['libcuda'])})
                (out/'supervisor_numerical.json').write_text(json.dumps(summary)+'\n')
            except BaseException as e:errors.append('Result verification: '+repr(e))
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}
        result={'status':'PASS_PENDING_INDEPENDENT_REVIEW' if not errors else 'FAILED','boot':boot,'after_boot':after,'pgid':child.pid if child else None,'exit':code,'errors':errors,'kernel_fault':fault,'cleanup':clean,'lock_free':free,'start':start,'finish':time.time(),'sha256':files}
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_failure)
    if len(sys.argv)!=2 or not re.fullmatch(r'[0-9a-f]{64}',sys.argv[1]):raise ValueError('expected frozen job sha256')
    result=run(Path(__file__).resolve().parent,sys.argv[1]);print(json.dumps(result),flush=True)
    sys.exit(0 if result['status'].startswith('PASS') else 1)
