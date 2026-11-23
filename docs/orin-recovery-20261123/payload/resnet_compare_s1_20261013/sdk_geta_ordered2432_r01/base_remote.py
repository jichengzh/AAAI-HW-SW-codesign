"""One fixed input0: standard-library supervisor outside the bounded GPU process."""
import ast,fcntl,hashlib,io,json,math,os,re,signal,struct,subprocess,sys,tarfile,time
from pathlib import Path

def check(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def identity(job):
    check(boot()==job['expected_boot'],'Unexpected boot; stop without reboot')
    actual={p:sha(p) for p in job['expected_files']}
    check(actual==job['expected_files'],'Pre/post file SHA mismatch')
    return {'boot':boot(),'sha256':actual,'epoch':time.time()}
def timeout_argv(job):
    args=job['argv'];root=Path(job['root'])
    check(args[:2]==[job['python'],str(root/'code/diagnostic_entry.py')],'Fixed diagnostic entry required')
    check(args.count('--mode')==1 and args[args.index('--mode')+1]=='input0','Only input0 permitted')
    check(args.count('--output')==1 and args[args.index('--output')+1]==str(root/'run/diagnostic'),'Fixed fresh output')
    check(not any(x in args for x in ('--onnx-build','--build','--saveEngine')),'Build forbidden')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s','120s']+args

def members(group,proc=Path('/proc')):
    rows=[]
    for directory in proc.iterdir():
        if not directory.name.isdigit():continue
        try:
            fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=group or fields[0]=='Z':continue
            rows.append({'pid':int(directory.name),'exe':os.readlink(directory/'exe'),
                         'argv':[a.decode(errors='replace') for a in (directory/'cmdline').read_bytes().split(b'\0') if a]})
        except (FileNotFoundError,ProcessLookupError):continue
    return rows

def owned(row,job):
    args=row['argv'];exe=row['exe']
    return (exe==os.path.realpath(job['python']) and args==job['argv']) or (
        exe==os.path.realpath('/usr/bin/timeout') and args==timeout_argv(job))
def cleanup(group,job):
    initial=members(group);sent=[]
    for sig in (signal.SIGTERM,signal.SIGKILL):
        current=members(group)
        if not current:break
        if not all(owned(row,job) for row in current):raise RuntimeError('Unknown process group member; manual cleanup required')
        try:os.killpg(group,sig);sent.append(int(sig))
        except ProcessLookupError:break
        for _ in range(60):
            if not members(group):break
            time.sleep(.05)
    return {'initial':initial,'signals':sent,'remaining':members(group)}
def lock_free(path):
    with Path(path).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        return True

def command_log(argv,path):
    with Path(path).open('xb') as out:
        try:return subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,timeout=10,check=False).returncode
        except subprocess.TimeoutExpired:return 124

def kernel_fault(path):
    return bool(re.search(r'ctxsw checksum|sm machine check|Bad page map|nvgpu.*(?:error|fault|\[ERR\])|Unhandled fault|BUG:|kernel panic|SError|segfault',Path(path).read_text(errors='replace'),re.I))

def npy(path,shape):
    data=Path(path).read_bytes();check(data[:6]==b'\x93NUMPY','NPY magic')
    version=data[6:8];check(version in (b'\x01\x00',b'\x02\x00'),'NPY version')
    width=2 if version[0]==1 else 4;offset=8+width
    size=int.from_bytes(data[8:offset],'little');header=ast.literal_eval(data[offset:offset+size].decode('latin1').strip())
    check(header=={'descr':'<f4','fortran_order':False,'shape':shape},'NPY dtype/shape/order')
    raw=data[offset+size:];count=math.prod(shape);check(len(raw)==count*4,'NPY byte size')
    values=struct.unpack('<%df'%count,raw);check(all(math.isfinite(x) for x in values),'NPY nonfinite')
    return raw,values

def wire_frames(path):
    raw=Path(path).read_bytes();frames=[];offset=0
    while offset<len(raw):
        check(len(raw)-offset>=8,'Truncated frame');nh,np=struct.unpack('!II',raw[offset:offset+8]);offset+=8
        check(0<nh<=65536 and np<=4*1024*1024 and offset+nh+np<=len(raw),'Invalid frame lengths')
        h=json.loads(raw[offset:offset+nh]);offset+=nh;p=raw[offset:offset+np];offset+=np;frames.append((h,p))
    return frames

def audit(run,job):
    out=run/'diagnostic';receipt=json.loads((out/'diagnostic_receipt.json').read_text())
    check(receipt['status']=='DIAGNOSTIC_COMPLETE' and receipt['count']==1 and receipt['mode']=='input0'
          and receipt['scope']=='DIAGNOSTIC_ONLY' and receipt['full50000_qualified'] is False and receipt['cleanup_completed'] is True,'Diagnostic completion')
    token=receipt['identity'];check(token['engine_sha256']==job['engine_sha256'],'Engine identity')
    check(token['code_sha256']==job['code_sha256'] and token['diagnostic_entry_sha256']==job['diagnostic_entry_sha256'],'Executed source identity')
    check(token['boot_id']==job['expected_boot'] and token['diagnostic_count']==1 and token['diagnostic_mode']=='input0','Actual boot/mode')
    check(token['torch_imported'] is False and token['tensorrt_version']=='8.5.2.2','SDK identity')
    libs=token['library_sha256']
    check(libs==job['expected_library_sha256'],'Actual SDK library set/hash differs from anchored baseline')
    check(any('libnvinfer.so' in p for p in libs) and any('libcudart.so' in p for p in libs),'Actual core SDK libraries')
    for p,v in libs.items():
        check(sha(p)==v,'Actual SDK library hash')
        if p in job['expected_files']:check(v==job['expected_files'][p],'SDK baseline library drift')
    for name in ('maps_before.txt','maps_after.txt'):
        text=(out/name).read_text();check(text and not any(t in text for t in ('libtorch','libc10','/torch/')),'Actual maps Torch/empty')
    inp,_=npy(out/'input_00000.npy',(1,3,224,224));output,values=npy(out/'output_00000.npy',(1,1000))
    digest=lambda b:hashlib.sha256(b).hexdigest()
    check(digest(inp)==job['input_sha256'],'Actual input0 bytes')
    rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
    check(len(rows)==1 and rows[0]['index']==0 and rows[0]['sample_id']==job['sample_id'],'One actual ordered sample')
    row=rows[0]
    check(row['input_data_sha256']==digest(inp) and row['output_data_sha256']==digest(output),'Raw data SHA')
    for kind in ('input','output'):check(row[kind+'_file_sha256']==sha(out/(kind+'_00000.npy')),'Raw file SHA')
    check(receipt['rows_sha256']==sha(out/'rows.jsonl') and receipt['wire_sha256']==sha(out/'wire.bin'),'Raw receipt SHA')
    check(sha(out/'wire.bin')==sha(run/'target.stdout.bin'),'Stdout/wire mismatch')
    frames=wire_frames(out/'wire.bin');check(len(frames)==3,'Exact ready/logits/end frames')
    ready,empty=frames[0];header,payload=frames[1];end,empty2=frames[2]
    check(not empty and not empty2 and ready=={'type':'ready','identity':token},'Ready identity')
    check(header['type']=='logits' and header['index']==0 and header['count']==1 and header['ids']==[job['sample_id']]
          and header['row_sha256']==[digest(inp)] and header['input_raw_sha256']==digest(inp)
          and header['logits_sha256']==digest(output) and payload==output,'Output frame binding')
    chain=digest(bytes(32)+json.dumps(header,sort_keys=True,separators=(',',':')).encode()+payload)
    check(end=={'type':'end','status':'COMPLETE','count':1,'chain_sha256':chain},'Final chain')
    reference=json.loads(Path(job['reference']).read_text())[0]['values']
    check(len(reference)==1000 and all(math.isfinite(x) for x in reference),'Reference finite shape')
    differences=[abs(x-y) for x,y in zip(values,reference)]
    return {'scope':'SINGLE_INPUT_DIAGNOSTIC_ONLY','finite_count':len(values),'top1':max(range(1000),key=values.__getitem__),
            'reference_top1':max(range(1000),key=reference.__getitem__),'max_abs_diff':max(differences),
            'mean_abs_diff':sum(differences)/1000,'reference_decimal_precision_limitation':True,
            'comparison_atol':.01,'comparison_rtol':.01,
            'failed_elements_original_tolerance':sum(d>.01+.01*abs(y) for d,y in zip(differences,reference)),
            'top1_matches_reference':max(range(1000),key=values.__getitem__)==max(range(1000),key=reference.__getitem__),
            'numerical_equivalence_asserted':False,'chain_sha256':chain,'pid':token['pid']}

def execute(root):
    root=Path(root);job=json.loads((root/'job.json').read_text())
    check(job['schema']=='sdk_input0_once_r01' and job['root']==str(root),'Job root/schema')
    argv=timeout_argv(job);run=root/'run';run.mkdir(exist_ok=False)
    start=time.time();child=None;code=None;clean={'remaining':[]};errors=[];before=None;analysis=None
    try:
        check(lock_free(job['global_lock']),'Global experiment lock busy')
        before=identity(job);write(run/'before.json',before)
        check(command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_before.log')==0 and 'MODE_30W' in (run/'mode_before.log').read_text(),'Expected30W mode')
        check(command_log(['journalctl','-k','-b','-n','200','--no-pager'],run/'kernel_before.log')==0,'Kernel baseline unavailable')
        with (run/'target.stdout.bin').open('xb') as out,(run/'target.stderr.log').open('xb') as err:
            env={**os.environ,'PYTHONNOUSERSITE':'1'};env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('PYTHONPATH',None)
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=env)
            write(run/'launch.json',{'argv':argv,'pgid':child.pid,'epoch':time.time(),'timeout_seconds':120,'kill_after_seconds':10})
            deadline=time.monotonic()+140;seen=set()
            while child.poll() is None and time.monotonic()<deadline:
                for row in members(child.pid):
                    if row['argv']==job['argv'] and row['pid'] not in seen:
                        try:
                            (run/('observed_pid_%s.maps'%row['pid'])).write_text(Path('/proc/%s/maps'%row['pid']).read_text())
                            write(run/('observed_pid_%s.json'%row['pid']),row);seen.add(row['pid'])
                        except FileNotFoundError:pass
                time.sleep(.05)
            code=child.poll()
            if code is None:errors.append('Outer supervision deadline140s exceeded')
    except BaseException as error:errors.append(type(error).__name__+': '+str(error))
    finally:
        if child is not None:
            try:clean=cleanup(child.pid,job)
            except BaseException as error:errors.append('Cleanup: '+str(error));clean={'remaining':members(child.pid)}
            try:code=child.wait(timeout=3)
            except subprocess.TimeoutExpired:errors.append('Timeout child still alive')
        try:
            after=identity(job);write(run/'after.json',after)
            check(before is not None and after['sha256']==before['sha256'],'Pre/post identity missing/different')
            mode_exit=command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_after.log')
            check(mode_exit==0 and 'MODE_30W' in (run/'mode_after.log').read_text(),'Post mode unavailable/different')
            kernel_exit=command_log(['journalctl','-k','--no-pager','--since','@'+str(int(start))],run/'kernel_delta.log')
            check(kernel_exit==0,'Kernel delta unavailable');check(not kernel_fault(run/'kernel_delta.log'),'New kernel fault')
        except BaseException as error:errors.append('Post evidence: '+str(error))
        free=lock_free(job['global_lock'])
        if code==0 and not errors and not clean['remaining'] and free:
            try:
                analysis=audit(run,job);write(run/'numerical_review.json',analysis)
                check(analysis['failed_elements_original_tolerance']==0 and analysis['top1_matches_reference'],'Reference numerical mismatch; no tolerance change')
            except BaseException as error:errors.append('Raw audit: '+str(error))
        result={'status':'INPUT0_DIAGNOSTIC_RECORDED_PENDING_INDEPENDENT_REVIEW' if code==0 and not errors and not clean['remaining'] and free else 'FAILED',
                'timeout_or_target_exit':code,'group_id':child.pid if child else None,'cleanup':clean,'lock_free':free,
                'cleanup_required':bool(clean['remaining']) or not free,'errors':errors,'end_epoch':time.time(),
                'scope':'INPUT0_ONLY_NO_512_NO_FULL50000','numerical_equivalence_asserted':False}
        write(run/'receipt.json',result)
    return result

def collect(root,stream):
    root=Path(root);paths=[root/'job.json']
    if (root/'run').exists():paths += [root/'run',*sorted((root/'run').rglob('*'))]
    check(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in paths),'Unsafe collect member')
    with tarfile.open(fileobj=stream,mode='w|gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(root)),recursive=False)

def signal_failure(signum,frame):
    # First termination enters execute finally; repeated soft signals cannot interrupt cleanup.
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
    raise RuntimeError("Supervisor signal "+str(signum))

def main():
    root=Path(__file__).resolve().parent
    if sys.argv[1:]==['--collect']:collect(root,sys.stdout.buffer);return 0
    check(not sys.argv[1:],'Unexpected args')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_failure)
    result=execute(root);code=0 if result['status']!='FAILED' else 1
    print('REMOTE_SDK_INPUT0_EXIT='+str(code),flush=True);return code
if __name__=='__main__':raise SystemExit(main())
