"""One-shot fullval supervisor; target GPU work is launched only by root."""
import importlib.util,json,os,re,signal,subprocess,sys,tarfile,time
from pathlib import Path

def expected_power(text):
    lines=text.splitlines()
    labels=[line.strip() for line in lines if line.strip().startswith('NV Power Mode:')]
    ids=[line.strip() for line in lines if re.fullmatch(r'\s*\d+\s*',line)]
    return labels==['NV Power Mode: MODE_30W'] and ids==['2']

def local_module(name):
    p=Path(__file__).resolve().parent/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,str(p));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
b=local_module('base_remote'); contracts=local_module('full_remote')

FAULT=re.compile(r'ctxsw checksum|sm machine check|Bad page map|nvgpu.*(?:error|fault|\[ERR\])|Unhandled fault|BUG:|kernel panic|SError|segfault',re.I)

def baseline_cursor(run):
    try: result=subprocess.run(['journalctl','-k','-n','1','--no-pager','--show-cursor'],capture_output=True,text=True,timeout=10,check=False)
    except subprocess.TimeoutExpired:raise ValueError('kernel baseline timeout')
    (Path(run)/'kernel_cursor_baseline.txt').write_text(result.stdout)
    b.check(result.returncode==0,'kernel baseline failure')
    matches=re.findall(r'^-- cursor: (.+)$',result.stdout,re.MULTILINE)
    b.check(len(matches)==1,'kernel baseline cursor missing')
    return matches[0]

def watch(child,run,cursor,deadline=14420,interval=2):
    limit=time.monotonic()+deadline
    b.check(bool(cursor),'kernel baseline cursor missing')
    with (Path(run)/'kernel_live.jsonl').open('x') as log:
        while child.poll() is None:
            b.check(time.monotonic()<limit,'outer timeout')
            since=['--after-cursor='+cursor]
            try: result=subprocess.run(['journalctl','-k','--no-pager','--show-cursor',*since],capture_output=True,text=True,timeout=10,check=False)
            except subprocess.TimeoutExpired:raise ValueError('kernel monitor timeout')
            log.write(json.dumps({'epoch':time.time(),'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr})+'\n');log.flush()
            b.check(result.returncode==0,'kernel monitor failure')
            b.check(not FAULT.search(result.stdout),'kernel fault')
            matches=re.findall(r'^-- cursor: (.+)$',result.stdout,re.MULTILINE)
            b.check(len(matches)==1,'kernel monitor cursor missing')
            cursor=matches[0]
            time.sleep(interval)
    return child.wait(timeout=3)

def execute(root):
    root=Path(root);job=json.loads((root/'job.json').read_text())
    b.check(job['schema']=='fullval_supervisor_r01' and job['root']==str(root),'root/schema')
    argv=contracts.command(job);run=root/'run';run.mkdir(exist_ok=False)
    start=time.time();child=None;code=None;errors=[];before=None;cursor=None;clean={'remaining':[]}
    try:
        b.check(b.lock_free(job['global_lock']),'global lock busy')
        before=b.identity(job);b.write(run/'before.json',before)
        b.check(b.command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_before.log')==0 and expected_power((run/'mode_before.log').read_text()),'mode')
        b.check(b.command_log(['journalctl','-k','-b','-n','200','--no-pager'],run/'kernel_before.log')==0,'kernel baseline')
        cursor=baseline_cursor(run)
        with (run/'target.stderr.log').open('xb') as err:
            env={**os.environ,'PYTHONNOUSERSITE':'1'};env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('PYTHONPATH',None)
            child=subprocess.Popen(argv,stdin=None,stdout=None,stderr=err,start_new_session=True,env=env)
            b.write(run/'launch.json',{'argv':argv,'pgid':child.pid,'epoch':time.time(),'timeout_seconds':14400,'kill_after_seconds':10})
            code=watch(child,run,cursor)
    except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
    finally:
        if child is not None:
            try:
                old=b.timeout_argv;b.timeout_argv=lambda _:argv
                try:clean=b.cleanup(child.pid,job)
                finally:b.timeout_argv=old
            except BaseException as exc:errors.append('cleanup: '+str(exc));clean={'remaining':b.members(child.pid)}
            try:code=child.wait(timeout=3)
            except subprocess.TimeoutExpired:errors.append('child remains')
        try:
            after=b.identity(job);b.write(run/'after.json',after)
            b.check(before is not None and before['sha256']==after['sha256'] and before['boot']==after['boot'],'identity drift')
            b.check(b.command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_after.log')==0 and expected_power((run/'mode_after.log').read_text()),'post mode')
            b.check(bool(cursor),'kernel baseline cursor missing')
            b.check(b.command_log(['journalctl','-k','--no-pager','--show-cursor','--after-cursor='+cursor],run/'kernel_delta.log')==0,'kernel delta unavailable')
            b.check(bool(re.search(r'^-- cursor: .+$',(run/'kernel_delta.log').read_text(),re.MULTILINE)) and not b.kernel_fault(run/'kernel_delta.log'),'kernel fault or cursor missing')
        except BaseException as exc:errors.append('post evidence: '+str(exc))
        free=b.lock_free(job['global_lock'])
        if code==0 and not errors and not clean['remaining'] and free:
            try:
                out=run/'target';identity=json.loads((out/'target_identity.json').read_text());completion=json.loads((out/'completion.json').read_text())
                contracts.validate_completion(job,identity,completion,(out/'maps_before.txt').read_text(),(out/'maps_after.txt').read_text())
                contracts.validate_stages((out/'init_stages.jsonl').read_text())
            except BaseException as exc:errors.append('completion: '+str(exc))
        passed=code==0 and not errors and not clean['remaining'] and free
        result={'status':'FULLVAL_RECORDED_PENDING_INDEPENDENT_REVIEW' if passed else 'FAILED','timeout_or_target_exit':code,'group_id':child.pid if child else None,'cleanup':clean,'lock_free':free,'cleanup_required':bool(clean['remaining']) or not free,'errors':errors,'end_epoch':time.time()}
        b.write(run/'receipt.json',result)
    return result

def collect(root,stream):
    root=Path(root);paths=[root/'job.json']
    if (root/'run').exists():paths += [root/'run',*sorted((root/'run').rglob('*'))]
    b.check(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in paths),'unsafe collect')
    with tarfile.open(fileobj=stream,mode='w|gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)

def main():
    root=Path(__file__).resolve().parent
    if sys.argv[1:]==['--collect']:collect(root,sys.stdout.buffer);return 0
    b.check(not sys.argv[1:],'unexpected args')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,b.signal_failure)
    result=execute(root);code=0 if result['status']!='FAILED' else 1
    print('REMOTE_SDK_FULLVAL_EXIT='+str(code),file=sys.stderr,flush=True);return code
if __name__=='__main__':raise SystemExit(main())
