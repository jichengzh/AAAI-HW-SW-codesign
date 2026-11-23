"""Single supervised SDK initialization probe. No inference or retries."""
import fcntl, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from segments import PLAN

LOCK='/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock'
BOOT='bb2ced1a-bad7-482f-af22-11ef1e4aca61'
ENGINE='/home/jichengzhi/resnet_compare_s1_20261013/geta_official_rebuild_r01/run_build/candidate_fp16.engine'
ENGINE_SHA='fab889253af5032e828a4a6310169823799ba53d97ba4285e9ed212a6419ecc5'
LAYERS='/home/jichengzhi/resnet_compare_s1_20261013/geta_official_rebuild_r01/run_build/layers.json'
LAYERS_SHA='edc1ff61cb9472ebb55825411e4111906bf986796363c17f50c46f7fcafe4b6b'
TRT='/usr/src/tensorrt/bin/trtexec'
TRT_SHA='f511762a3c57a08918923586e902424b1abe40320717addafc9b6923c1ceebb4'
INPUT_SHA='f8a8ad06512129e835d8ee6f7237f29301e1f5a53b0f2cfb495e56706eac8d74'
SOURCE='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/geta/source'
SOURCE_INPUT='/home/jichengzhi/resnet_compare_s1_20261013/geta_sdk_segmented256_full50k_r02/input_manifest.json'
PINS={ENGINE:ENGINE_SHA,LAYERS:LAYERS_SHA,TRT:TRT_SHA,SOURCE_INPUT:INPUT_SHA,
      SOURCE+'/transfer_manifest.json':'31887a0d906f1e9be813c4df245678ec90290feb829950d4e324c88f678fce44',
      SOURCE+'/candidate.onnx':'38f01f97bf8d83154fedffbbd83f212fcb3437bd882b25e763723dfd1e4e7121'}
LIBS={
 '/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2':'4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca',
 '/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.5.2':'5a30c8e46ba47c3a93eb825ad0a27cbe969fb6f174ea963e4d22dfe33630097a',
 '/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1':'151ad1343173e919304e08d4c6801508303111f2b1c7b3737f6b5134590808f4',
 '/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298':'6ca38be1cc6fa0b8ddad0313eea2a4483be27d2e8474437f9f7b7a536b7197a2'}


def has_gpu_fault(message):
    return bool(re.search(r'ctxsw checksum|sm machine check|Bad page map|nvgpu.*(?:\[ERR\]|error|fault)|Unhandled fault|BUG:|kernel panic|SError',message,re.I))

def is_expected_power_mode(text):
    lines=text.splitlines()
    labels=[line.strip() for line in lines if line.strip().startswith('NV Power Mode:')]
    ids=[line.strip() for line in lines if re.fullmatch(r'\s*\d+\s*',line)]
    return labels == ['NV Power Mode: MODE_30W'] and ids == ['2']

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


PINS.update({'/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/frozen_stream.py': '6214df738caa17bb1d51e9b08d7c62658c8c5a89ac21a9de4da16a4684ed9b7d', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/host_entry.py': '0a432f992103de691e7d1df2cb8416bd4acbfdd0d0f2f5811e29dc0a063dd5b1', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/identity.py': '1567587b556bf7ef44f27c5db7f71f41650621953a5e2c45b5caa42e8a3d0345', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/multi_entry.py': 'a3ad49868a6cb9e4e96588544a8dd03f32d09d731bc24c170509faf33c9ae760', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/multi_host.py': '40c7bea750b600f969be584ea8855d88277b28b231d935aca6ce1fbdaf28fcab', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/sdk_backend.py': '63e7bcf178f167b20d5f9539d0afdb2650b98a1b0d93c4cf855f2c73a0c403f8', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/sdk_contract.py': '246568da4f81a263e5b51cc0e6fe6daba2fe26010f905b5d048a93f5a5ae382f', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/sdk_entry.py': 'b994002ca6b8eb1507de658959e00abd2297c88cc37e9ac1317d37029dade81c', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/sdk_wrapper.py': '6f11fc7d323010be7767855cede7e1f969d7fb19beada82df3250ff2e933b938', '/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code/stage_logger.py': 'b39f544a3d64aea7af7cc272476616a803a16afdd035ba9bd5c536fab8759941', '/home/jichengzhi/uniad_venv/bin/python': '330ad6775d2db4d95524b249d38e14b9046672b8468fa6fbf7f89b7c683d1f6b', '/usr/lib/python3.8/dist-packages/tensorrt/__init__.py': '33903e1d9a65809b28e0ef01184ff65d1c87ebdace986cd96d2bcfdaefa2c70d', '/usr/lib/python3.8/dist-packages/tensorrt/tensorrt.so': 'eb6ed22cb1895927a96093fa1778a2d0bf0120868514c841579cba2cc45aca4a'})

PINS.update({'/home/jichengzhi/resnet_compare_s1_20261013/geta_sdk_segmented256_full50k_r02/target.py': '5fc97fb8e0603bb2e45a658c87486899040cd4437af718b02d6c58895bdcb2a6', '/home/jichengzhi/resnet_compare_s1_20261013/geta_sdk_segmented256_full50k_r02/segments.py': '3b1c7f4ee4b2047f6d2fe2fc9d40a245ee9a98b87094efb076d61724c4de6177', '/home/jichengzhi/resnet_compare_s1_20261013/geta_sdk_segmented256_full50k_r02/prefix.py': '7c7ca26c7fef3c82e9cdf218b29808e1a45a62cb6b3a598625eb9a79ca2048ff', '/home/jichengzhi/resnet_compare_s1_20261013/geta_sdk_segmented256_full50k_r02/frozen_stream.py': '6214df738caa17bb1d51e9b08d7c62658c8c5a89ac21a9de4da16a4684ed9b7d'})
PYTHON='/home/jichengzhi/uniad_venv/bin/python'
TIMEOUT=7200

def run_plan(execute):
 for start,count in PLAN:
  code=execute(start,count)
  if code!=0:return code
 return 0

def verify_segment(done,rows,start,count):
 if done.get('status')!='SEGMENT_COMPLETE' or done.get('samples')!=count or done.get('segment_start')!=start or len(rows)!=count or any(r['index']!=start+i for i,r in enumerate(rows)):raise ValueError('segment completion/count')
 return True

def run(root):
 root=Path(root).resolve();out=root/'run';out.mkdir(exist_ok=False)
 errors=[];children=[];clean=[];cursor=None;lock_holder=None;code=None;fault=False;segments=[]
 start_time=time.time();deadline=time.monotonic()+TIMEOUT;boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 try:
  if boot!=BOOT:raise ValueError('Unexpected boot')
  if not lock_free():raise ValueError('Device busy')
  mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_before.log').write_text(mode)
  if not is_expected_power_mode(mode):raise ValueError('Unexpected mode')
  (out/'library_before.json').write_text(json.dumps(library_identity(),indent=2)+'\n')
  (out/'model_before.json').write_text(json.dumps(model_identity(),indent=2)+'\n')
  lock_holder=open(LOCK,'a');fcntl.flock(lock_holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cursor=read_cursor();(out/'cursor.txt').write_text(cursor+'\n')
  def execute(segment_start,count):
   nonlocal cursor,fault
   remaining=deadline-time.monotonic()
   if remaining<=0:errors.append('total timeout');return 1
   segment=out/('segment_%04d'%segment_start);fd=lock_holder.fileno()
   argv=[PYTHON,'-I',str(root/'target.py'),str(segment),str(root/'input_manifest.json'),str(segment_start),str(count),str(fd)]
   env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONNOUSERSITE']='1'
   began=time.time()
   with (out/('stderr_%04d.log'%segment_start)).open('xb') as stderr:
    child=subprocess.Popen(argv,stdin=None,stdout=None,stderr=stderr,start_new_session=True,env=env,pass_fds=(fd,));children.append(child)
    cursor,detected,error=monitor(child,cursor,out/'kernel.jsonl',remaining);fault|=detected
    if error:errors.append(error)
    clean.append(cleanup(child.pid))
    if clean[-1]['remaining']:
     errors.append('segment %d cleanup remaining %s'%(segment_start,clean[-1]['remaining']));return 1
    try:exit_code=child.wait(timeout=3)
    except subprocess.TimeoutExpired:errors.append('child remained');return 1
    segments.append({'start':segment_start,'count':count,'pid':child.pid,'begin':began,'finish':time.time(),'exit':exit_code})
    (out/'segments.json').write_text(json.dumps(segments,indent=2)+'\n')
    if error or exit_code!=0:errors.append('segment %d exit %s'%(segment_start,exit_code));return 1
    try:
     done=json.loads((segment/'completion.json').read_text());rows=[json.loads(line) for line in (segment/'rows.jsonl').read_text().splitlines()]
     verify_segment(done,rows,segment_start,count)
    except BaseException as e:errors.append('segment evidence '+repr(e));return 1
    return 0
  code=run_plan(execute)
 except BaseException as e:errors.append(repr(e))
 finally:
  for child in children:
   if child.poll() is None:clean.append(cleanup(child.pid))
  if cursor is not None:
   try:
    cursor,found=kernel_since(cursor,out/'kernel.jsonl');fault|=found
    if found:errors.append('new kernel fault')
    (out/'library_after.json').write_text(json.dumps(library_identity(),indent=2)+'\n');(out/'model_after.json').write_text(json.dumps(model_identity(),indent=2)+'\n')
    mode=command(['/usr/sbin/nvpmodel','-q']);(out/'mode_after.log').write_text(mode)
    if not is_expected_power_mode(mode):errors.append('post mode mismatch')
   except BaseException as e:errors.append('post evidence '+repr(e))
  if lock_holder is not None:lock_holder.close()
  free=lock_free();after=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  if not free or any(c['remaining'] for c in clean) or after!=boot:errors.append('cleanup or boot mismatch')
  if code!=0:errors.append('sequence incomplete')
  result={'status':'SEGMENTED_FULL50000_PENDING_INDEPENDENT_AUDIT' if not errors else 'FAILED','boot':boot,'after_boot':after,'segment_pids':[p.pid for p in children],'exit':code,'errors':errors,'kernel_fault':fault,'cleanup':clean,'lock_free':free,'start':start_time,'finish':time.time()}
  (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def collect(root,stream):
 import tarfile
 root=Path(root);paths=[root/n for n in ('input_manifest.json','supervisor.py','target.py','segments.py','prefix.py','frozen_stream.py')]
 if (root/'run').exists():paths += [root/'run',*sorted((root/'run').rglob('*'))]
 if not all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in paths):raise ValueError('unsafe collect')
 with tarfile.open(fileobj=stream,mode='w|gz') as archive:
  for p in paths:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)

if __name__=='__main__':
 root=Path(__file__).resolve().parent
 if sys.argv[1:]==['--collect']:collect(root,sys.stdout.buffer);sys.exit(0)
 if sys.argv[1:]:raise ValueError('No arguments permitted')
 for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_failure)
 result=run(root);code=1 if result['status']=='FAILED' else 0
 print('REMOTE_SDK_SEGMENT_EXIT='+str(code),file=sys.stderr,flush=True);sys.exit(code)
