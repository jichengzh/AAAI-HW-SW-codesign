"""Single supervised SDK initialization probe. No inference or retries."""
import fcntl, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path

LOCK='/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock'
BOOT='bb2ced1a-bad7-482f-af22-11ef1e4aca61'
ENGINE='/home/jichengzhi/resnet_compare_s1_20261013/trt_official_rebuild_diagnostic_r01/candidate_fp16.engine'
LAYERS='/home/jichengzhi/resnet_compare_s1_20261013/trt_official_rebuild_diagnostic_r01/layers.json'
CODE='/home/jichengzhi/resnet_compare_s1_20261013/sdk_ordered512_execution_r02/code'
PYTHON='/home/jichengzhi/uniad_venv/bin/python'


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

def run(root):
    root=Path(root).resolve();out=root/'run';out.mkdir(exist_ok=False)
    errors=[];child=None;code=None;clean={'signals':[],'remaining':[]};cursor=None;fault=False
    start=time.time();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    try:
        if boot!=BOOT:raise ValueError('Unexpected boot')
        if not lock_free():raise ValueError('Device busy')
        mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_before.log').write_text(mode)
        if 'MODE_30W' not in mode:raise ValueError('Unexpected mode')
        cursor=read_cursor();(out/'cursor.txt').write_text(cursor+'\n')
        argv=[PYTHON,'-I',str(root/'trace_init.py'),'--engine',ENGINE,'--layers',LAYERS,'--code',CODE,'--events',str(out/'events.jsonl')]
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
                mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_after.log').write_text(mode)
                if 'MODE_30W' not in mode:errors.append('Post mode mismatch')
            except BaseException as e:errors.append('Post evidence: '+repr(e))
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
    result=run(Path(__file__).resolve().parent)
    print(json.dumps(result),flush=True)
    sys.exit(0 if result['status'].startswith('PASS') else 1)
