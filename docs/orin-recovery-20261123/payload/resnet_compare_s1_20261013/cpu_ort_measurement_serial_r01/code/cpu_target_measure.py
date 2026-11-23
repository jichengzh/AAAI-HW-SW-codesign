"""Exclusive, two-phase CPU ORT target; final independent audit remains required."""
import sys,platform
from cpu_bootstrap import configure
ORT_ROOT='/home/jichengzhi/resnet_compare_s1_20261013/ort_cpu_20261026_r02/python'
configure(sys.path,sys.modules,platform.machine(),sys.version_info,ORT_ROOT)
import argparse,fcntl,gc,json,os
from pathlib import Path
from types import SimpleNamespace
from cpu_target import runtime_environment,maps_without_gpu
import numpy as np
from evidence import POINTS,check,qualification,reference_rows,sha
from cpu_identity import validate,ORT_PYBIND,code_identity
from cpu_session import create_session
from cpu_backend import Runtime
from ort_timing import TimedInference
from measurement_core import DurableLog,measure_all,audit_latency,audit_energy
from orin_power_reader import open_reader
AUDITOR_SHA='10bd3e784a856797ffd9695bce67a47dc288dab7184e572a8e785fc10ed3bd11'
FREEZE=Path(__file__).with_name('cpu_accuracy_freeze.json')
LOCK=Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')

def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def write_json(path,document):
    with Path(path).open('x') as f:
        json.dump(document,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())

def check_preflight(job):
    frozen=json.loads(FREEZE.read_text())
    check(job['auditor_sha256']==AUDITOR_SHA==frozen['audit_code_sha256'],'auditor freeze mismatch')
    check(code_identity()==frozen['cpu_code_sha256'],'cpu code freeze mismatch')
    check(set(job['points'])==set(POINTS),'three fixed points required')
    tokens={};qualified={};identities={}
    for point in POINTS:
        config=job['points'][point];args=SimpleNamespace(point=point,**config)
        token=validate(point,args)
        q=qualification(point,token,config['result'],config['audit'],config['audit_sha256'],job['auditor_sha256'])
        audit=json.loads(Path(config['audit']).read_text())
        for key,path in (('stream/predictions.jsonl',config['predictions']),('stream/logits.f32',config['logits'])):
            check(audit['artifact_sha256'].get(key)==sha(path),'audited '+key+' SHA mismatch')
        tokens[point]=token;qualified[point]=q
        identities[point]={'graph_sha256':token['model_sha256'],'source_sha256':token['source_sha256'],
                           'audit_sha256':q['audit_sha256']}
    return tokens,qualified,identities

def load_reference(job):
    inputs=np.load(job['inputs'],mmap_mode='r',allow_pickle=False)
    final=json.loads(Path(job['points']['dense']['final']).read_text())
    ids=[row['sample_id'] for row in final['samples'][:256]];refs={};rows={}
    for point in POINTS:
        config=job['points'][point]
        with Path(config['predictions']).open() as f:rows[point]=[json.loads(f.readline()) for _ in range(256)]
        check(Path(config['logits']).stat().st_size==50000*1000*4,'full logits length mismatch')
        refs[point]=np.memmap(config['logits'],mode='r',dtype='<f4',shape=(50000,1000))[:256]
    reference_rows(inputs,ids,refs,rows);return inputs,ids,refs

def raw_inventory(root):
    root=Path(root)
    names=[p+'/'+phase+'/'+name for p in POINTS for phase in ('latency','energy') for name in ('requests.jsonl','logits.f32')]
    return {name:sha(root/name) for name in names}

class Handle:
    def __init__(self,session):
        self.runtime=Runtime(session);self.timed=TimedInference(self.runtime)
    def infer(self,x):return self.timed.infer(x)
    def close(self):self.timed=None;self.runtime=None;gc.collect()

def create_runtime(args,ort):return Handle(create_session(args.onnx,ort))

def execute(job,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    runtimes={};log=None;error=None;result=None;lock=None
    try:
        lock=LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tokens,qualified,identities=check_preflight(job);inputs,ids,references=load_reference(job)
        before_boot=boot_id();ort,env=runtime_environment();env={**env,'ort_pybind_sha256':ORT_PYBIND}
        for point in POINTS:
            check(env['ort_pybind_sha256']==qualified[point]['ort_pybind_sha256'],'accuracy/runtime pybind mismatch')
            args=SimpleNamespace(point=point,**job['points'][point]);runtimes[point]=create_runtime(args,ort)
        before=maps_without_gpu();(out/'maps_before.txt').write_text(before)
        write_json(out/'identity.json',{'boot_id':before_boot,'tokens':tokens,'qualification':qualified,
                                       'environment':env,'inputs_sha256':sha(job['inputs'])})
        log=DurableLog(out/'raw')
        def current_identity(point):
            check(sha(job['points'][point]['onnx'])==identities[point]['graph_sha256'],'graph changed')
            return identities[point]
        result=measure_all(runtimes,inputs,ids,references,identities,current_identity,
              lambda p:p in qualified,open_reader,log.emit,
              latency_gate=lambda output:audit_latency(out/'raw',ids,references,output))
        check(audit_energy(out/'raw',ids,references,result['energy']),'energy audit failed')
        check(boot_id()==before_boot,'boot changed');after=maps_without_gpu()
        (out/'maps_after.txt').write_text(after)
        log.close();log=None
        for point in reversed(POINTS):runtimes[point].close()
        runtimes.clear()
        write_json(out/'target_result.json',{'status':'MEASUREMENT_RECORDED_PENDING_INDEPENDENT_AUDIT',
              'summary':result,'boot_id':before_boot,'environment':env,'qualification':qualified,
              'raw_sha256':raw_inventory(out/'raw')})
    except BaseException as exc:
        error=exc;cleanup=[]
        for resource in ([log] if log is not None else [])+list(runtimes.values()):
            try:resource.close()
            except BaseException as e:cleanup.append(repr(e))
        write_json(out/'failure.json',{'status':'FAILED','error':repr(exc),'cleanup_errors':cleanup})
    finally:
        if lock is not None:lock.close()
    if error is not None:raise error
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--job-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();check(sha(a.job)==a.job_sha256,'frozen job SHA mismatch')
    execute(json.loads(Path(a.job).read_text()),a.output)
if __name__=='__main__':main()
