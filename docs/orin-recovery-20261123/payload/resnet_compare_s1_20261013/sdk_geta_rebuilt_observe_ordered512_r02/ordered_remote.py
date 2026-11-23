"""Fixed ordered512 target supervisor; protocol stdout is exclusively target's."""
import importlib.util, json, os, re, subprocess, sys, tarfile, time
import numpy as np
from pathlib import Path
_base_path=Path(__file__).resolve().parent/'base_remote.py'
_base_spec=importlib.util.spec_from_file_location('bundled_base_remote',str(_base_path))
if _base_spec is None or _base_spec.loader is None:raise ImportError('Missing bundled supervisor')
b=importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(b)

def expected_power(text):
    lines=text.splitlines()
    labels=[line.strip() for line in lines if line.strip().startswith('NV Power Mode:')]
    ids=[line.strip() for line in lines if re.fullmatch(r'\s*\d+\s*',line)]
    return labels==['NV Power Mode: MODE_30W'] and ids==['2']

def valid_lock(value):
    b.check(value=='/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock','Invalid global lock')
    return True

def timeout_argv(job):
    root=Path(job['root']);args=job['argv']
    b.check(args[:2]==[job['python'],str(root/'code/diagnostic_entry.py')],'Fixed diagnostic entry required')
    b.check(job['mode'] in ('input0','ordered512') and args.count('--mode')==1 and args[args.index('--mode')+1]==job['mode'],'Invalid mode')
    b.check(job['point']=='geta' and args.count('--point')==1 and args[args.index('--point')+1]==job['point'],'Invalid point')
    b.check(args.count('--output')==1 and args[args.index('--output')+1]==str(root/'run/diagnostic'),'Fixed fresh output')
    b.check(not any(x in args for x in ('--onnx-build','--build','--saveEngine')),'Build forbidden')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s','120s' if job['mode']=='input0' else '1800s']+args

def check_stages(events):
    stack=[]
    completed=[]
    for event in events:
        stage=event.get('stage');action=event.get('event')
        b.check(isinstance(stage,str) and stage,'Invalid stage')
        if action=='begin':stack.append(stage)
        elif action=='ok':
            b.check(bool(stack) and stack.pop()==stage,'Mismatched stage')
            completed.append(stage)
        else:raise ValueError('Invalid stage event')
    b.check(not stack and 'create_runtime' in completed and 'library_identity' in completed,'Incomplete stage')
    return True

def reference_status(output, reference):
    output=np.asarray(output,dtype=np.float32);reference=np.asarray(reference,dtype=np.float32)
    if output.shape!=(1000,) or reference.shape!=(1000,) or not np.isfinite(output).all() or not np.isfinite(reference).all():raise ValueError('Invalid output/reference')
    differences=np.abs(output-reference)
    failed=int(np.count_nonzero(differences > .01 + .01*np.abs(reference)))
    return {'reference_allclose':failed==0, 'reference_top1_match':bool(np.argmax(output)==np.argmax(reference)),
            'reference_failed_elements':failed,'reference_max_abs':float(differences.max())}

def audit(run,job):
    out=run/'diagnostic'; receipt=json.loads((out/'diagnostic_receipt.json').read_text())
    b.check(receipt['status']=='DIAGNOSTIC_COMPLETE' and receipt['count']==(1 if job['mode']=='input0' else 512) and receipt['mode']==job['mode']
       and receipt['scope']=='DIAGNOSTIC_ONLY' and receipt['full50000_qualified'] is False and receipt['cleanup_completed'] is True,'Diagnostic receipt')
    token=receipt['identity']
    b.check(token['point']==job['point'] and token['reference_sha256']==job['reference_sha256'],'Point/reference identity')
    check_stages([json.loads(line) for line in (out/'init_stages.jsonl').read_text().splitlines()])
    b.check(token['engine_sha256']==job['engine_sha256'] and token['code_sha256']==job['code_sha256'] and token['diagnostic_entry_sha256']==job['diagnostic_entry_sha256'],'Executed identity')
    b.check(token['boot_id']==job['expected_boot'] and token['diagnostic_count']==(1 if job['mode']=='input0' else 512) and token['diagnostic_mode']==job['mode'] and token['torch_imported'] is False and token['tensorrt_version']=='8.5.2.2','Target identity')
    b.check(token['library_sha256']==job['expected_library_sha256'],'Target libraries')
    for p,h in token['library_sha256'].items():b.check(b.sha(p)==h,'Target library drift')
    for n in ('maps_before.txt','maps_after.txt'):
        s=(out/n).read_text();b.check(s and not any(x in s for x in ('libtorch','libc10','/torch/')),'Torch maps')
    rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
    samples=json.loads(Path(job['final']).read_text())['samples']
    expected=1 if job['mode']=='input0' else 512
    b.check(len(rows)==expected,'Exact rows')
    import hashlib
    with np.load(job['reference_logits'],allow_pickle=False) as values: refs=values['logits'].copy()
    b.check(refs.shape==(expected,1000) and refs.dtype==np.float32 and np.isfinite(refs).all(),'Reference shape/values')
    reference=json.loads(Path(job['reference']).read_text())['rows']
    b.check(len(reference)==expected,'Reference sample count')
    failed_samples=0;top1_mismatches=0;failed_elements=0;max_abs=0.0
    for i,row in enumerate(rows):
        b.check(row['index']==i and row['sample_id']==samples[i]['sample_id'],'Sample order')
        for kind,shape in (('input',(1,3,224,224)),('output',(1,1000))):
            p=out/('%s_%05d.npy'%(kind,i));raw,values=b.npy(p,shape)
            b.check(row[kind+'_data_sha256']==hashlib.sha256(raw).hexdigest() and row[kind+'_file_sha256']==b.sha(p),'Raw identity')
            if kind=='input':b.check(reference[i]['sample_id']==row['sample_id'] and reference[i]['input_sha256']==hashlib.sha256(raw).hexdigest(),'Reference input identity')
            if kind=='output':
                actual=reference_status(values,refs[i])
                b.check(all(row.get(key)==value for key,value in actual.items()),'Reference calculation mismatch')
                failed_samples+=int(not actual['reference_allclose']);top1_mismatches+=int(not actual['reference_top1_match'])
                failed_elements+=actual['reference_failed_elements'];max_abs=max(max_abs,actual['reference_max_abs'])
    b.check(receipt['rows_sha256']==b.sha(out/'rows.jsonl') and receipt['wire_sha256']==b.sha(out/'wire.bin'),'Receipt identity')
    return {'status':'RAW_DIAGNOSTIC_VERIFIED','rows':expected,'rows_sha256':b.sha(out/'rows.jsonl'),'wire_sha256':b.sha(out/'wire.bin'),'input_files_retained_on_target':True,'reference_allclose_failed_samples':failed_samples,'reference_top1_mismatch_samples':top1_mismatches,'reference_failed_elements':failed_elements,'reference_max_abs':max_abs,'reference_atol':.01,'reference_rtol':.01,'reference_scope':'cross_engine_and_cross_backend','original_reference_qualification_pass':failed_samples==0 and top1_mismatches==0,'same_engine_comparison_scope':'input0_only_not_this_512_audit'}

def reference_receipt(analysis):
    keys=('reference_allclose_failed_samples','reference_top1_mismatch_samples','original_reference_qualification_pass')
    return {key:analysis.get(key) if analysis else None for key in keys}

def monitor(child,cursor,path,seconds):
    deadline=time.monotonic()+seconds
    while child.poll() is None:
        cursor,fault=b.kernel_poll(cursor,path)
        if fault:return None,'new kernel fault'
        if time.monotonic()>=deadline:return None,'outer supervision timeout'
        time.sleep(.5)
    cursor,fault=b.kernel_poll(cursor,path)
    return child.poll(), 'new kernel fault' if fault else None

def final_kernel(cursor,path):
    b.check(bool(cursor),'Kernel cursor unavailable')
    _,fault=b.kernel_poll(cursor,path)
    b.check(not fault,'New kernel fault')

def execute(root):
    root=Path(root);job=json.loads((root/'job.json').read_text())
    b.check(job['schema']=='sdk_multi_diagnostic_once_r01' and job['root']==str(root),'Job root/schema')
    valid_lock(job['global_lock'])
    argv=timeout_argv(job);run=root/'run';run.mkdir(exist_ok=False)
    started=time.time();child=None;code=None;clean={'remaining':[]};errors=[];before=None;analysis=None;cursor=None
    try:
        b.check(b.lock_free(job['global_lock']),'Global experiment lock busy')
        before=b.identity(job);b.write(run/'before.json',before)
        b.check(b.command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_before.log')==0 and expected_power((run/'mode_before.log').read_text()),'Expected30W mode')
        b.check(b.command_log(['journalctl','-k','-b','-n','200','--no-pager'],run/'kernel_before.log')==0,'Kernel baseline unavailable')
        cursor=b.cursor_start();(run/'kernel_cursor.txt').write_text(cursor+'\n')
        with (run/'target.stderr.log').open('xb') as err:
            env={**os.environ,'PYTHONNOUSERSITE':'1'};env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('PYTHONPATH',None)
            child=subprocess.Popen(argv,stdin=None,stdout=None,stderr=err,start_new_session=True,env=env)
            b.write(run/'launch.json',{'argv':argv,'pgid':child.pid,'epoch':time.time(),'timeout_seconds':120 if job['mode']=='input0' else 1800,'kill_after_seconds':10})
            code,monitor_error=monitor(child,cursor,run/'kernel_events.jsonl',140 if job['mode']=='input0' else 1820)
            if monitor_error:errors.append(monitor_error)
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
            b.check(b.command_log(['/usr/sbin/nvpmodel','-q'],run/'mode_after.log')==0 and expected_power((run/'mode_after.log').read_text()),'Post mode changed')
            final_kernel(cursor,run/'kernel_final.jsonl')
        except BaseException as error:errors.append('Post evidence: '+str(error))
        free=b.lock_free(job['global_lock'])
        if code==0 and not errors and not clean['remaining'] and free:
            try:analysis=audit(run,job);b.write(run/'raw_audit.json',analysis)
            except BaseException as error:errors.append('Raw audit: '+str(error))
        result={'status':'DIAGNOSTIC_RECORDED_PENDING_INDEPENDENT_REVIEW' if code==0 and not errors and not clean['remaining'] and free else 'FAILED','timeout_or_target_exit':code,'group_id':child.pid if child else None,'cleanup':clean,'lock_free':free,'cleanup_required':bool(clean['remaining']) or not free,'errors':errors,'scope':'DIAGNOSTIC_ONLY_NO_FULL50000','end_epoch':time.time(),**reference_receipt(analysis)}
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
