"""Restricted persistent launcher for frozen Quant reference or measurement code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

TARGET=Path('/home/jichengzhi/resnet_compare_s1_20261013')
ENV={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved(path):
    path=Path(path).resolve();require(TARGET.resolve() in path.parents,'Path outside approved target root');return path


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def validate_job(path,expected):
    path=approved(path);require(sha(path)==expected,'Job SHA differs');job=json.loads(path.read_text())
    common={'kind','root','script','ap_file','manifest_sha256','ap_sha256','files','supervisor_sha256'}
    require(job.get('kind') in ('reference','measurement'),'Unsupported job kind')
    require(set(job)==common|({'output'} if job['kind']=='reference' else set()),'Unknown/missing job fields')
    require(job['supervisor_sha256']==sha(__file__),'Supervisor code differs')
    root=approved(job['root']);script=approved(job['script']);ap=approved(job['ap_file'])
    require(script.name==('generate.py' if job['kind']=='reference' else 'runner.py'),'Unsupported entry script')
    if job['kind']=='measurement':require(script==root/'runner.py','Measurement entry must be bundle runner')
    if job['kind']=='reference':approved(job['output'])
    required={str(script),*[str(root/n) for n in ('runner.py','qualification.py','ap_audit.py','sort_export.py')]}
    manifest=json.loads((root/'manifest.json').read_text())
    if job['kind']=='measurement':
        binding=manifest.get('target_ap',{})
        require(isinstance(binding.get('file'),str) and isinstance(binding.get('sha256'),str),'Measurement AP binding missing')
        require(approved(root/binding['file'])==ap and binding['sha256']==job['ap_sha256'],'Measurement AP binding differs')
    if 'power_reader_sha256' in manifest:
        reader=str(root/'orin_power_reader.py');required=required|{reader}
        require(job['files'].get(reader)==manifest['power_reader_sha256'],'Power reader dependency differs')
    if 'power_reader_base_sha256' in manifest:
        base_reader=str(root/'orin_power_reader_base.py');required=required|{base_reader}
        require(job['files'].get(base_reader)==manifest['power_reader_base_sha256'],'Power base reader dependency differs')
    require(required<=set(job['files']),'Frozen dependency missing')
    require(all(sha(approved(p))==digest for p,digest in job['files'].items()),'Frozen source SHA differs')
    require(sha(root/'manifest.json')==job['manifest_sha256'],'Bundle manifest differs')
    require(sha(ap)==job['ap_sha256'],'AP evidence differs')
    data=json.loads(ap.read_text());require(data.get('status')=='complete' and data.get('count')==1789,'Complete1789 AP required')
    return job


def command(job):
    if job['kind']=='measurement':return [job['script'],'run','--root',job['root'],'--expected-manifest-sha256',job['manifest_sha256']]
    return [job['script'],'--root',job['root'],'--output',job['output'],'--ap-file',job['ap_file'],
        '--expected-manifest-sha256',job['manifest_sha256'],'--expected-ap-sha256',job['ap_sha256'],
        '--expected-runner-sha256',job['files'][str(Path(job['root'])/'runner.py')],
        '--expected-code-sha256',job['files'][job['script']]]


def child(path,expected):
    job=validate_job(path,expected);prefix=TARGET/'ort_cpu_20261026_r02/python'
    for name in ('numpy','onnxruntime','onnxruntime.capi.onnxruntime_pybind11_state'):
        loaded=sys.modules.get(name)
        require(loaded is None or prefix.resolve() in Path(loaded.__file__).resolve().parents,'Runtime already loaded outside isolated prefix')
    os.environ.update(ENV);sys.path.insert(0,str(prefix));sys.path.insert(1,str(Path(job['script']).parent))
    sys.argv=command(job);runpy.run_path(job['script'],run_name='__main__')


def process_identity(pid):
    path=Path('/proc')/str(pid)
    try:return {'pid':pid,'start_ticks':int((path/'stat').read_text().rsplit(') ',1)[1].split()[19]),
        'actual_cmdline':[x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]}
    except (FileNotFoundError,ProcessLookupError):return {'pid':pid,'start_ticks':None,'actual_cmdline':None,'process_already_exited':True}


def interrupted(signum,frame):raise SystemExit(128+signum)


def supervise(path,expected,evidence):
    validate_job(path,expected);evidence=approved(evidence);evidence.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(Path(__file__).resolve()),'child','--job',str(Path(path).resolve()),'--expected-job-sha256',expected]
    proc=None;code=None;error=None;started=time.time();previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        with (evidence/'child.log').open('x') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,**ENV},start_new_session=True)
            write(evidence/'child.json',{**process_identity(proc.pid),'command':cmd,'start_epoch':started,'job_sha256':expected})
            code=proc.wait()
        return code
    except BaseException as caught:error=caught;raise
    finally:
        signal.signal(signal.SIGTERM,previous)
        observed=proc.poll() if proc is not None and error is not None else code
        write(evidence/'outer_exit.json',{'child_pid':proc.pid if proc else None,'child_exit_code':code,
            'observed_child_exit_code':observed,'child_may_still_be_running':proc is not None and observed is None,
            'child_identity_at_exit':process_identity(proc.pid) if proc else None,'error_type':type(error).__name__ if error else None,
            'error':str(error) if error else None,'elapsed_seconds':time.time()-started,'command':cmd,
            'job_sha256':expected,'metrics_success_claimed':False,'automatic_retry':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('supervise','child'));p.add_argument('--job',required=True,type=Path)
    p.add_argument('--expected-job-sha256',required=True);p.add_argument('--evidence',type=Path);a=p.parse_args()
    if a.mode=='child':require(a.evidence is None,'Child evidence argument forbidden');child(a.job,a.expected_job_sha256)
    else:
        require(a.evidence is not None,'Supervisor evidence required')
        raise SystemExit(supervise(a.job,a.expected_job_sha256,a.evidence))


if __name__=='__main__':main()
