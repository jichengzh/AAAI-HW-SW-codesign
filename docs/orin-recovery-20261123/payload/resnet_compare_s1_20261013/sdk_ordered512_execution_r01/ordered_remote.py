"""Fixed ordered512 target supervisor; protocol stdout is exclusively target's."""
import json, os, re, subprocess, sys, tarfile, time
from pathlib import Path
import base_remote as b

def timeout_argv(job):
    root=Path(job['root']);args=job['argv']
    b.check(args[:2]==[job['python'],str(root/'code/diagnostic_entry.py')],'Fixed diagnostic entry required')
    b.check(args.count('--mode')==1 and args[args.index('--mode')+1]=='ordered512','Only ordered512 permitted')
    b.check(args.count('--output')==1 and args[args.index('--output')+1]==str(root/'run/diagnostic'),'Fixed fresh output')
    b.check(not any(x in args for x in ('--onnx-build','--build','--saveEngine')),'Build forbidden')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s','1800s']+args

def audit(run,job):
    out=run/'diagnostic'; receipt=json.loads((out/'diagnostic_receipt.json').read_text())
    b.check(receipt['status']=='DIAGNOSTIC_COMPLETE' and receipt['count']==512 and receipt['mode']=='ordered512'
       and receipt['scope']=='DIAGNOSTIC_ONLY' and receipt['full50000_qualified'] is False and receipt['cleanup_completed'] is True,'Diagnostic receipt')
    token=receipt['identity']
    b.check(token['engine_sha256']==job['engine_sha256'] and token['code_sha256']==job['code_sha256'] and token['diagnostic_entry_sha256']==job['diagnostic_entry_sha256'],'Executed identity')
    b.check(token['boot_id']==job['expected_boot'] and token['diagnostic_count']==512 and token['diagnostic_mode']=='ordered512' and token['torch_imported'] is False and token['tensorrt_version']=='8.5.2.2','Target identity')
    b.check(token['library_sha256']==job['expected_library_sha256'],'Target libraries')
    for p,h in token['library_sha256'].items():b.check(b.sha(p)==h,'Target library drift')
    for n in ('maps_before.txt','maps_after.txt'):
        s=(out/n).read_text();b.check(s and not any(x in s for x in ('libtorch','libc10','/torch/')),'Torch maps')
    rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
    samples=json.loads(Path(job['final']).read_text())['samples']
    b.check(len(rows)==512,'Exact 512 rows')
    import hashlib
    for i,row in enumerate(rows):
        b.check(row['index']==i and row['sample_id']==samples[i]['sample_id'],'Sample order')
        for kind,shape in (('input',(1,3,224,224)),('output',(1,1000))):
            p=out/('%s_%05d.npy'%(kind,i));raw,_=b.npy(p,shape)
            b.check(row[kind+'_data_sha256']==hashlib.sha256(raw).hexdigest() and row[kind+'_file_sha256']==b.sha(p),'Raw identity')
    b.check(receipt['rows_sha256']==b.sha(out/'rows.jsonl') and receipt['wire_sha256']==b.sha(out/'wire.bin'),'Receipt identity')
    return {'status':'RAW512_VERIFIED','rows':512,'rows_sha256':b.sha(out/'rows.jsonl'),'wire_sha256':b.sha(out/'wire.bin'),'input_files_retained_on_target':True}

def execute(root):
    root=Path(root);job=json.loads((root/'job.json').read_text())
    b.check(job['schema']=='sdk_ordered512_once_r01' and job['root']==str(root),'Job root/schema')
    argv=timeout_argv(job);run=root/'run';run.mkdir(exist_ok=False)
    started=time.time();child=None;code=None;clean={'remaining':[]};errors=[];before=None;analysis=None
    try:
        b.check(b.lock_free(job['global_lock']),'Global experiment lock busy')
        before=b.identity(job);b.write(run/'before.json',before)
        b.check(b.command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_before.log')==0 and 'MODE_30W' in (run/'mode_before.log').read_text(),'Expected30W mode')
        b.check(b.command_log(['journalctl','-k','-b','-n','200','--no-pager'],run/'kernel_before.log')==0,'Kernel baseline unavailable')
        with (run/'target.stderr.log').open('xb') as err:
            env={**os.environ,'PYTHONNOUSERSITE':'1'};env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('PYTHONPATH',None)
            child=subprocess.Popen(argv,stdin=None,stdout=None,stderr=err,start_new_session=True,env=env)
            b.write(run/'launch.json',{'argv':argv,'pgid':child.pid,'epoch':time.time(),'timeout_seconds':1800,'kill_after_seconds':10})
            try:code=child.wait(timeout=1820)
            except subprocess.TimeoutExpired:errors.append('Outer supervision deadline1820s exceeded')
    except BaseException as error:errors.append(type(error).__name__+': '+str(error))
    finally:
        if child is not None:
            try:
                old=b.timeout_argv;b.timeout_argv=timeout_argv
                try:clean=b.cleanup(child.pid,job)
                finally:b.timeout_argv=old
            except BaseException as error:errors.append('Cleanup: '+str(error));clean={'remaining':b.members(child.pid)}
            try:code=child.wait(timeout=3)
            except subprocess.TimeoutExpired:errors.append('Timeout child still alive')
        try:
            after=b.identity(job);b.write(run/'after.json',after)
            b.check(before is not None and after['sha256']==before['sha256'],'Pre/post identity missing/different')
            b.check(b.command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_after.log')==0 and 'MODE_30W' in (run/'mode_after.log').read_text(),'Post mode changed')
            b.check(b.command_log(['journalctl','-k','--no-pager','--since','@'+str(int(started))],run/'kernel_delta.log')==0,'Kernel delta unavailable')
            b.check(not b.kernel_fault(run/'kernel_delta.log'),'New kernel fault')
        except BaseException as error:errors.append('Post evidence: '+str(error))
        free=b.lock_free(job['global_lock'])
        if code==0 and not errors and not clean['remaining'] and free:
            try:analysis=audit(run,job);b.write(run/'raw_audit.json',analysis)
            except BaseException as error:errors.append('Raw audit: '+str(error))
        result={'status':'ORDERED512_RECORDED_PENDING_INDEPENDENT_REVIEW' if code==0 and not errors and not clean['remaining'] and free else 'FAILED','timeout_or_target_exit':code,'group_id':child.pid if child else None,'cleanup':clean,'lock_free':free,'cleanup_required':bool(clean['remaining']) or not free,'errors':errors,'scope':'ORDERED512_ONLY_NO_FULL50000','end_epoch':time.time()}
        b.write(run/'receipt.json',result)
    return result

def collect(root,stream):
    root=Path(root);paths=[root/'job.json']
    if (root/'run').exists():paths += [root/'run',*sorted((root/'run').rglob('*'))]
    paths=[p for p in paths if not re.fullmatch(r'input_\d{5}\.npy',p.name)]
    b.check(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in paths),'Unsafe collect member')
    with tarfile.open(fileobj=stream,mode='w|gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(root)),recursive=False)

def main():
    root=Path(__file__).resolve().parent
    if sys.argv[1:]==['--collect']:collect(root,sys.stdout.buffer);return 0
    b.check(not sys.argv[1:],'Unexpected args')
    import signal
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,b.signal_failure)
    result=execute(root);code=0 if result['status']!='FAILED' else 1
    print('REMOTE_SDK_FULLVAL_EXIT='+str(code),file=sys.stderr,flush=True);return code
if __name__=='__main__':raise SystemExit(main())
