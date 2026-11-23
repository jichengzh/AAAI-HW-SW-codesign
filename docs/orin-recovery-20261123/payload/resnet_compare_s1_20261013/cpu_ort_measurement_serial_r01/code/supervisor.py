"""Bounded supervisor; all GPU work remains in the locked target child."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

GLOBAL_LOCK=Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')

def validate_lock(job):
    check(Path(job['lock']).resolve()==GLOBAL_LOCK.resolve(),'wrong global experiment lock')

def check(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def command(job,root):
    root=Path(root).resolve();output=Path(job['output']).resolve()
    check(output.parent==root and output.name=='target_result','output outside fixed run root')
    check(type(job['seconds']) is int and 0<job['seconds']<=14400,'invalid timeout')
    check(Path(job['target']).resolve().is_file() and Path(job['job']).resolve().is_file(),'missing target/job')
    check(re.fullmatch('[0-9a-f]{64}',job['job_sha256']) is not None,'invalid job hash')
    expected=job.get('expected_files',{})
    check(expected.get(job['target'])==sha(job['target']) and expected.get(job['job'])==job['job_sha256']
          and sha(job['job'])==job['job_sha256'],'target/job source binding')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(job['seconds'])+'s',
            job['python'],job['target'],'--job',job['job'],'--job-sha256',job['job_sha256'],'--output',str(output)]

def mode_matches(value):
    return re.search(r'^NV Power Mode: MODE_30W\s*\n2\s*$',value,re.M) is not None

ORT_PYBIND='/home/jichengzhi/resnet_compare_s1_20261013/ort_cpu_20261026_r02/python/onnxruntime/capi/onnxruntime_pybind11_state.cpython-312-aarch64-linux-gnu.so'

def required_files(target_job,code_root):
    root=Path(code_root)
    names=('cpu_identity.py','cpu_backend.py','cpu_bootstrap.py','cpu_session.py','cpu_target.py','cpu_host.py',
           'frozen_stream.py','measurement_core.py','evidence.py','cpu_target_measure.py','ort_timing.py',
           'orin_power_reader.py','orin_power_reader_base.py','manifest.json','supervisor.py','collect.py','cpu_accuracy_freeze.json')
    paths={str(root/name) for name in names}|{target_job['inputs'],ORT_PYBIND}
    check(set(target_job['points'])=={'dense','copt','geta'},'three points required')
    for config in target_job['points'].values():
        paths.update(config[k] for k in ('onnx','manifest','final','result','audit','predictions','logits'))
    return paths

def lock_free(path):
    with Path(path).open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        return True

def kernel_fault(path):
    return re.search(r'ctxsw checksum|Bad page map|nvgpu.*(?:error|fault)|Unhandled fault|BUG:|kernel panic|SError|segfault',
                     Path(path).read_text(errors='replace'),re.I) is not None

def members(group):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0]!='Z':found.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return found

def stop(group):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if not members(group):break
        try:os.killpg(group,sig)
        except ProcessLookupError:break
        for _ in range(100):
            if not members(group):break
            time.sleep(.05)
    return members(group)

def result(code,files_match,mode_ok,kernel_ok,remaining,errors):
    good=code==0 and files_match and mode_ok and kernel_ok and not remaining and not errors
    return {'status':'MEASUREMENT_EXIT_ZERO_PENDING_INDEPENDENT_AUDIT' if good else 'FAILED',
            'target_exit':code,'files_match':files_match,'mode_ok':mode_ok,'kernel_ok':kernel_ok,
            'remaining':remaining,'errors':errors,'scope':'requires independent raw measurement audit'}

def capture(argv,path):
    with Path(path).open('wb') as out:
        try:return subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                  timeout=10,check=False).returncode
        except subprocess.TimeoutExpired:return 124

def execute(job,root):
    root=Path(root).resolve();argv=command(job,root)
    evidence=root/'supervision';evidence.mkdir(exist_ok=False)
    started=int(time.time());errors=[];remaining=[];child=None;code=None
    boot=lambda:Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    baseline={};after={};mode_ok=False;kernel_ok=False
    try:
        validate_lock(job)
        check(lock_free(job['lock']),'device lock busy')
        check(boot()==job['boot_id'],'boot mismatch')
        target_job=json.loads(Path(job['job']).read_text())
        check(required_files(target_job,Path(job['target']).parent).issubset(set(job['expected_files'])),
              'measurement sources not frozen')
        baseline={p:sha(p) for p in job['expected_files']}
        check(baseline==job['expected_files'],'source mismatch')
        check(capture(['/usr/sbin/nvpmodel','-q'],evidence/'mode_before.log')==0 and
              mode_matches((evidence/'mode_before.log').read_text()),'mode mismatch')
        with (evidence/'stdout.log').open('xb') as out,(evidence/'stderr.log').open('xb') as err:
            env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('CUDA_VISIBLE_DEVICES',None)
            env['PYTHONNOUSERSITE']='1'
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=env)
            (evidence/'launch.json').write_text(json.dumps({'argv':argv,'pgid':child.pid,'start_epoch':started})+'\n')
            deadline=time.monotonic()+job['seconds']+20
            while child.poll() is None and time.monotonic()<deadline:
                time.sleep(5)
                rc=capture(['journalctl','-k','--no-pager','--since','@'+str(started)],evidence/'kernel_live.log')
                if rc!=0 or kernel_fault(evidence/'kernel_live.log'):
                    errors.append('kernel fault or journal unavailable');break
            if child.poll() is None:
                remaining=stop(child.pid)
                if not errors:errors.append('supervisor timeout')
            code=child.wait(timeout=10)
    except BaseException as e:errors.append(repr(e))
    finally:
        if child is not None:
            try:remaining=stop(child.pid);code=child.wait(timeout=10)
            except BaseException as e:errors.append('cleanup: '+repr(e))
        try:
            after={p:sha(p) for p in job['expected_files']}
            files_match=baseline==after==job['expected_files'] and boot()==job['boot_id']
            mode_ok=(capture(['/usr/sbin/nvpmodel','-q'],evidence/'mode_after.log')==0 and
                     mode_matches((evidence/'mode_after.log').read_text()))
            kernel_ok=(capture(['journalctl','-k','--no-pager','--since','@'+str(started)],evidence/'kernel_delta.log')==0
                       and not kernel_fault(evidence/'kernel_delta.log'))
            check(lock_free(GLOBAL_LOCK),'device lock remains held')
        except BaseException as e:
            errors.append('postflight: '+repr(e));files_match=False
        receipt=result(code,files_match,mode_ok,kernel_ok,remaining,errors)
        receipt.update({'pgid':child.pid if child else None,'before':baseline,'after':after,'boot_id':boot()})
        (evidence/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def signal_failure(signum,frame):
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
    raise RuntimeError('Supervisor signal '+str(signum))

def main():
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_failure)
    check(len(sys.argv)==4,'supervisor JOB ROOT SHA required')
    check(sha(sys.argv[1])==sys.argv[3],'supervisor job SHA mismatch')
    receipt=execute(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
    code=0 if receipt['status'].startswith('MEASUREMENT_EXIT_ZERO') else 1
    print('REMOTE_CPU_ORT_MEAS_EXIT='+str(code),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
