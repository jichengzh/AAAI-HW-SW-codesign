"""Root-only preparation/dispatch/audit of frozen, independently qualified TRT jobs."""
import argparse,json,re,shlex,shutil,subprocess,tempfile
from pathlib import Path
import gate
from protocol import check
SOURCE=Path(__file__).resolve().parent
HOST='jichengzhi@172.16.62.222'
REMOTE='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_measurement_r01'
PYTHON='/home/jichengzhi/uniad_venv/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',HOST]
SCP=['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
PHASES=('capture','latency','energy')

def write(path,value):
 Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def select(root,point,phase):
 check(point in gate.POINTS and phase in PHASES,'Unknown point/phase')
 return Path(root).resolve()/point/phase

def remote_base(root):
 name=Path(root).resolve().name
 check(name not in ('.','..') and re.fullmatch(r'[A-Za-z0-9_.-]+',name) is not None,'Unsafe run identifier')
 return REMOTE+'/'+name

def accuracy_inputs(jobdir,job):
 check(job['code_sha256']==gate.code_identity(),'Frozen source changed')
 sources=job['accuracy_sources'];a=gate.arguments(job['point'])
 q=gate.qualification(sources['result'],sources['audit'],a,full=True)
 check(q==job['qualification'],'Full accuracy qualification changed')
 for filename,key in (('accuracy_result.json','result_sha256'),('accuracy_audit.json','audit_sha256')):
  check(gate.sha(jobdir/filename)==q[key],'Accuracy copy changed')
 return q,a

def prepare(root,point,phase,result,audit,target_python=PYTHON):
 root=Path(root).resolve();jobdir=select(root,point,phase);remote_base(root)
 interpreter=Path(target_python)
 check(interpreter.is_absolute() and str(interpreter).startswith('/home/jichengzhi/') and '..' not in interpreter.parts,'Target interpreter outside authorized home')
 result=Path(result).resolve();audit=Path(audit).resolve();a=gate.arguments(point)
 q=gate.qualification(result,audit,a,full=True)
 capture=None;latencies=None
 if phase!='capture':
  previous=root/point/'capture';audit_job(previous)
  check(gate.read(previous/'job.json')['qualification']==q,'Capture used another accuracy identity')
  capture=previous/'target'
 if phase=='energy':
  latencies=gate.latency_receipts(root)
  check(gate.read(root/point/'latency/job.json')['qualification']==q,'Latency used another engine/accuracy')
 jobdir.mkdir(parents=True,exist_ok=False)
 shutil.copy2(result,jobdir/'accuracy_result.json');shutil.copy2(audit,jobdir/'accuracy_audit.json')
 if capture is not None:shutil.copytree(capture,jobdir/'capture')
 if latencies is not None:
  for name in gate.POINTS:shutil.copytree(root/name/'latency',jobdir/'latency'/name)
 job={'point':point,'phase':phase,'target_python':str(interpreter),'qualification':q,'code_sha256':gate.code_identity(),
      'accuracy_sources':{'result':str(result),'audit':str(audit)},'all_latency':latencies or {},
      'scope':'target_repeatability_after_independent_full50000_not_host_equivalence'}
 write(jobdir/'job.json',job)
 return jobdir

def audit_job(jobdir):
 jobdir=Path(jobdir);job=gate.read(jobdir/'job.json');select(jobdir.parent.parent,job['point'],job['phase'])
 q,a=accuracy_inputs(jobdir,job);outer=gate.read(jobdir/'outer_exit.json')
 check(outer['ssh_returncode']==outer['remote_returncode']==0 and outer['remote_exit_records']==1
       and outer['job_sha256']==gate.digest(job),'Real exit evidence missing')
 if job['phase']=='capture':
  gate.capture_evidence(jobdir/'target',q,a);evidence=jobdir/'target/metadata.json';summary=None
 else:
  result=gate.validate_measurement(jobdir/'target',job,a);evidence=jobdir/'target/result.json';summary=result.get('summary')
  if job['phase']=='energy':
   receipts=gate.latency_receipts(jobdir/'latency',staged=True)
   check(receipts==job['all_latency'],'Staged three-point latency evidence changed')
 result={'status':'PASS','point':job['point'],'phase':job['phase'],'job_sha256':gate.digest(job),
         'evidence_sha256':gate.sha(evidence),'summary':summary,'scope':'local_evidence_recomputation_not_independent_reviewer'}
 write(jobdir/'host_audit.json',result)
 return result

def run(argv,stdout=None,timeout=120):
 return subprocess.run(argv,stdout=stdout if stdout is not None else subprocess.PIPE,
                       stderr=subprocess.STDOUT,text=True,timeout=timeout,check=False)

def checked(argv):
 result=run(argv)
 check(result.returncode==0,'Stage/transfer failed with exit '+str(result.returncode))

def stage_code(root):
 root=Path(root);remote=remote_base(root);receipt=root/'code_stage.json';identity=gate.code_identity()
 if receipt.exists():
  check(gate.read(receipt)=={'remote':remote,'code_sha256':identity},'Staged code identity changed')
  return remote
 command='mkdir -p '+shlex.quote(remote)+' && test ! -e '+shlex.quote(remote+'/code')
 checked(SSH+[command])
 with tempfile.TemporaryDirectory(prefix='trt_measurement_stage_') as temporary:
  code=Path(temporary)/'code';code.mkdir()
  for name in identity:
   source=SOURCE/name;dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
  checked(SCP+['-r',str(code),HOST+':'+remote+'/'])
 write(receipt,{'remote':remote,'code_sha256':identity})
 return remote

def dispatch(root,point,phase):
 root=Path(root).resolve();jobdir=select(root,point,phase);job=gate.read(jobdir/'job.json')
 check(job['point']==point and job['phase']==phase,'Prepared job selection differs')
 accuracy_inputs(jobdir,job)
 with (jobdir/'.dispatch-reserved').open('x'):pass
 remote=stage_code(root);remotejob=remote+'/jobs/'+point+'/'+phase
 checked(SSH+['mkdir -p '+shlex.quote(remote+'/jobs/'+point)+' && test ! -e '+shlex.quote(remotejob)])
 checked(SCP+['-r',str(jobdir),HOST+':'+remote+'/jobs/'+point+'/'])
 command=['timeout','--signal=TERM','--kill-after=10s','1800s',job['target_python'],remote+'/code/target.py','--job',remotejob]
 shell=shlex.join(command)+'; status=$?; printf "REMOTE_MEASUREMENT_EXIT=%s\\n" "$status" >&2; exit "$status"'
 log=jobdir/'execute.log';code=None
 try:
  with log.open('x') as output:code=run(SSH+[shell],stdout=output,timeout=1860).returncode
 finally:
  content=log.read_text() if log.exists() else ''
  markers=re.findall(r'^REMOTE_MEASUREMENT_EXIT=(-?\d+)$',content,flags=re.MULTILINE)
  write(jobdir/'outer_exit.json',{'ssh_returncode':code,'remote_returncode':int(markers[0]) if len(markers)==1 else None,
       'remote_exit_records':len(markers),'job_sha256':gate.digest(job),'remote_job':remotejob})
  downloaded=run(SCP+['-r',HOST+':'+remotejob+'/target',str(jobdir)],timeout=300)
  write(jobdir/'download_exit.json',{'returncode':downloaded.returncode})
 check(code==0,'Target execution failed; evidence retained')
 check(downloaded.returncode==0,'Target evidence download incomplete')
 return audit_job(jobdir)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','dispatch','audit'))
 p.add_argument('--root',default=str(SOURCE/'runs_r01'));p.add_argument('--point',choices=gate.POINTS,required=True)
 p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--result');p.add_argument('--audit');p.add_argument('--target-python',default=PYTHON);a=p.parse_args()
 if a.action=='prepare':
  check(a.result is not None and a.audit is not None,'Explicit full result and independent audit required')
  result={'prepared':str(prepare(a.root,a.point,a.phase,a.result,a.audit,target_python=a.target_python))}
 elif a.action=='dispatch':result=dispatch(a.root,a.point,a.phase)
 else:result=audit_job(select(a.root,a.point,a.phase))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
