"""Outside GNU timeout: supervise and clean only this build's dedicated group."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
ROOT=Path('/home/jichengzhi/resnet_compare_s1_20261013')
OBSERVER=ROOT/'trt_sdk_only_diagnostic_r01/observe_build_r01.py'
OBSERVER_SHA='463b54600d9b059026948891a53999ce9be42a2c7961480eed2afa5f73b2e9be'
OUT=ROOT/'trt_sdk_only_diagnostic_r01/build_supervisor_r01'
LOCK=ROOT/'.orin-runtime.lock'
PYTHON='/home/jichengzhi/uniad_venv/bin/python'
BINARY='/usr/src/tensorrt/bin/trtexec'
SAVE_ARG='--saveEngine='+str(ROOT/'trt_official_rebuild_diagnostic_r01/candidate_fp16.engine')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def members(group,proc=Path('/proc')):
 rows=[]
 for directory in proc.iterdir():
  if not directory.name.isdigit():continue
  try:
   fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
   if int(fields[2])!=group or fields[0]=='Z':continue
   try:exe=os.readlink(directory/'exe');argv=[a.decode(errors='replace') for a in (directory/'cmdline').read_bytes().split(b'\0') if a]
   except (FileNotFoundError,PermissionError):exe=None;argv=[]
   rows.append({'pid':int(directory.name),'pgid':group,'state':fields[0],'exe':exe,'argv':argv})
  except (FileNotFoundError,ProcessLookupError):continue
 return rows

def owned(row):
 argv=row['argv'];exe=row['exe']
 if exe==str(Path(BINARY).resolve()) and SAVE_ARG in argv:return True
 if str(OBSERVER) in argv and exe in (str(Path(PYTHON).resolve()),'/usr/bin/timeout'):return True
 return False

def cleanup(group):
 initial=members(group);sent=[]
 for sig in (signal.SIGTERM,signal.SIGKILL):
  current=members(group)
  if not current:break
  if not all(owned(row) for row in current):raise ValueError('Unrecognized member in owned experiment group: '+json.dumps(current))
  try:os.killpg(group,sig);sent.append(int(sig))
  except ProcessLookupError:break
  for _ in range(60):
   if not members(group):break
   time.sleep(.05)
 return {'initial':initial,'signals':sent,'remaining':members(group)}

def lock_free():
 with LOCK.open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  return True

def execute():
 if sha(OBSERVER)!=OBSERVER_SHA:raise ValueError('Frozen observer source changed')
 OUT.mkdir(exist_ok=False);child=None;code=None;errors=[];clean={'remaining':[]}
 argv=['/usr/bin/timeout','--signal=TERM','--kill-after=10s','300s',PYTHON,'-I',str(OBSERVER)]
 try:
  with (OUT/'observer.log').open('x') as log:
   child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(OUT/'launch.json',{'argv':argv,'pgid':child.pid,'pid':child.pid,'start_epoch':time.time(),'source_sha256':sha(__file__),
        'observer_sha256':OBSERVER_SHA,'dedicated_session':True,'timeout_foreground':False})
   try:code=child.wait(timeout=315)
   except subprocess.TimeoutExpired:errors.append('Outer timeout process exceeded315s')
 except BaseException as error:errors.append(type(error).__name__+': '+str(error))
 finally:
  if child is not None:
   try:clean=cleanup(child.pid)
   except BaseException as error:
    errors.append('Cleanup blocked: '+str(error));clean={'remaining':members(child.pid)}
   if code is None:
    try:code=child.wait(timeout=5)
    except subprocess.TimeoutExpired:errors.append('Timeout/observer still running after cleanup')
  free=lock_free()
  result={'status':'OUTER_COMPLETE_PENDING_KERNEL_REVIEW' if code==0 and not errors and not clean['remaining'] and free else 'OUTER_FAILED',
          'timeout_or_observer_returncode':code,'group_remaining':clean['remaining'],'cleanup':clean,'lock_free':free,'errors':errors,
          'group_id':child.pid if child else None,'end_epoch':time.time(),'kernel_review_required':True,'inference_authorized':False}
  write(OUT/'receipt.json',result)
 return result

if __name__=='__main__':
 result=execute();print(json.dumps(result));raise SystemExit(0 if result['status']=='OUTER_COMPLETE_PENDING_KERNEL_REVIEW' else 1)
