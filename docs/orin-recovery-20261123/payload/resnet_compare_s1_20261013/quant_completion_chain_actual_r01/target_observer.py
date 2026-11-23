"""One fixed, ordinary-user measurement observation; no model imports or retry."""
import argparse,collections,fcntl,hashlib,json,os,re,subprocess,sys,time,traceback,tarfile
from pathlib import Path
BASE=Path('/home/jichengzhi/resnet_compare_s1_20261013')
BUNDLE=BASE/'quant_measurement_actual_r01'
CHAIN=BASE/'quant_completion_chain_actual_r01'
PYTHON='/home/jichengzhi/miniforge3/bin/python3.12'
BINARY=BASE/'ort_cpu_20261026_r02/python/onnxruntime/capi/onnxruntime_pybind11_state.cpython-312-aarch64-linux-gnu.so'
LOCK=BASE/'.orin-runtime.lock'

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def relative(name):
    p=Path(name);require(isinstance(name,str) and bool(name) and not p.is_absolute() and '..' not in p.parts and str(p)==name,'Unsafe relative file');return name

def kernel_faults(before,after):
    lines=(collections.Counter(after.splitlines())-collections.Counter(before.splitlines())).elements()
    pattern=re.compile(r'\b(?:xid|segfault|oom|oops|panic|bug:)\b|out of memory|bad page|kernel bug|memory corruption|serror|unhandled fault|ctxsw|notifier|sm error|nvgpu|gpu.*(?:fault|error)|nvrm.*(?:fault|error)|smmu.*fault|tegra.*(?:fault|error)',re.I)
    return [s for s in lines if pattern.search(s)]

def lock_free(path=LOCK):
    with Path(path).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);stat=os.fstat(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)
    return {'path':str(path),'inode':stat.st_ino,'device':stat.st_dev,'nonblocking_exclusive_acquired':True,'released':True}

def sources(config):
    values={str(BUNDLE/relative(n)):v for n,v in config['bundle_files'].items()}
    values={**values,**config['target_extra_files']}
    for path,digest in values.items():
        p=Path(path);require(p.is_absolute() and BASE in p.resolve().parents and not p.is_symlink() and sha(p)==digest,'Frozen target file differs: '+path)
    return values

def journal(path):
    cp=subprocess.run(['journalctl','-k','-b','--no-pager','-o','short-monotonic'],capture_output=True,text=True,timeout=30)
    Path(path).write_text(cp.stdout);Path(str(path)+'.stderr').write_text(cp.stderr)
    require(cp.returncode==0,'Kernel journal unavailable');return cp.stdout

def wait_supervisor(proc,evidence,observations,config,proc_root=Path('/proc')):
    mapped=False
    while proc.poll() is None:
        child_file=evidence/'child.json'
        if child_file.exists() and not mapped:
            child=read(child_file);p=proc_root/str(child['pid'])
            try:
                stat=(p/'stat').read_text().rsplit(') ',1)[1].split();maps=(p/'maps').read_text()
                require(int(stat[19])==child['start_ticks'],'Measurement child PID reused')
                if str(BINARY) in maps:
                    (observations/'runtime_child_maps.txt').write_text(maps);write(observations/'runtime_child_identity.json',child);mapped=True
            except FileNotFoundError:pass
        time.sleep(5)
    code=proc.wait();child=read(evidence/'child.json') if (evidence/'child.json').exists() else {}
    write(observations/'supervisor_exit.json',{'supervisor_pid':proc.pid,'supervisor_exit_code':code,'collector_observed_job_sha256':config['job_sha256'],'child_pid':child.get('pid'),'target_root':str(BUNDLE)})
    require(code==0,'Supervisor nonzero exit');require(mapped,'Actual child ORT binary mapping unobserved')
    return read(evidence/'child.json'),read(evidence/'outer_exit.json')

def observe(config,expected):
    require(sha(CHAIN/'config.json')==expected,'Target config SHA differs')
    require(sha(__file__)==config['target_observer_sha256'],'Observer source differs')
    require(sha(BUNDLE/'manifest.json')==config['manifest_sha256'],'Target manifest differs')
    job=read(config['target_job']);require(job['kind']=='measurement' and job['root']==str(BUNDLE) and job['script']==str(BUNDLE/'runner.py'),'Unexpected fixed measurement job')
    require(sha(config['target_job'])==config['job_sha256'] and sha(config['target_supervisor'])==config['supervisor_sha256'],'Job/supervisor differs')
    observations=BUNDLE/'observations';observations.mkdir(exist_ok=False);status={'status':'FAILED','automatic_retry':False};process=None
    try:
        (observations/'frozen_job.json').write_bytes(Path(config['target_job']).read_bytes());(observations/'frozen_supervisor.py').write_bytes(Path(config['target_supervisor']).read_bytes())
        before=sources(config);require(before.get(str(BUNDLE/'manifest.json'))==config['manifest_sha256'],'Manifest absent from source inventory')
        require(before.get(config['target_job'])==config['job_sha256'] and before.get(config['target_supervisor'])==config['supervisor_sha256'],'Executor absent from source inventory')
        binary_before=sha(BINARY);require(binary_before==config['runtime_binary_sha256'],'ORT binary differs');write(observations/'preflight.json',{'sources':before,'runtime_binary_file':str(BINARY),'runtime_binary_sha256':binary_before,'lock':lock_free(LOCK)})
        kernel_before=journal(observations/'kernel_before.txt');started=time.monotonic()
        cmd=[PYTHON,config['target_supervisor'],'supervise','--job',config['target_job'],'--expected-job-sha256',config['job_sha256'],'--evidence',str(observations/'supervisor')]
        with (observations/'supervisor.log').open('x') as f:
            process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'})
            write(observations/'launch.json',{'supervisor_pid':process.pid,'command':cmd,'started_monotonic_s':started})
            child,outer=wait_supervisor(process,observations/'supervisor',observations,config)
        ended=time.monotonic();require(outer['child_pid']==child['pid'] and outer['job_sha256']==config['job_sha256'] and outer['command']==child['command']==[PYTHON,config['target_supervisor'],'child','--job',config['target_job'],'--expected-job-sha256',config['job_sha256']],'Actual child/job command differs')
        require(outer['child_exit_code']==outer['observed_child_exit_code']==0 and outer['child_may_still_be_running'] is False and outer['error_type'] is None,'Child exit incomplete')
        after=sources(config);binary_after=sha(BINARY);require(before==after and binary_after==binary_before,'Frozen files/runtime changed')
        kernel_after=journal(observations/'kernel_after.txt');faults=kernel_faults(kernel_before,kernel_after);lock=lock_free(LOCK);require(not faults,'New kernel fault')
        names=['requests.jsonl','summary.json','runtime_versions.json','execution.json']
        profiles=sorted(BUNDLE.glob('profile_*.json'));require(len(profiles)==4,'Exactly four actual profiles required');names += [p.name for p in profiles]
        artifacts={n:sha(BUNDLE/n) for n in names}
        raw={'manifest_sha256':config['manifest_sha256'],'target_root':str(BUNDLE),'job':{'file':'observations/frozen_job.json','sha256':config['job_sha256'],'target_path':config['target_job']},'child':{'pid':child['pid'],'command':outer['command'],'supervisor_file':'observations/frozen_supervisor.py','runner_command':[str(BUNDLE/'runner.py'),'run','--root',str(BUNDLE),'--expected-manifest-sha256',config['manifest_sha256']],'started_monotonic_s':started,'ended_monotonic_s':ended,'command_scope':'runner_sys_argv_from_frozen_job_and_supervisor','actual_os_command':child['command']},'source_before':before,'source_after':after,'runtime_binary_before_sha256':binary_before,'runtime_binary_after_sha256':binary_after,'artifacts':artifacts,'postflight':{'lock':lock,'kernel':{'before_file':'observations/kernel_before.txt','after_file':'observations/kernel_after.txt','before_sha256':sha(observations/'kernel_before.txt'),'after_sha256':sha(observations/'kernel_after.txt'),'new_fault_lines':faults}}}
        write(observations/'observation.json',raw);status={'status':'MEASUREMENT_EXIT_ZERO_PENDING_INDEPENDENT_AUDIT','automatic_retry':False}
    except BaseException as error:
        (observations/'error.txt').write_text(traceback.format_exc());status={**status,'error_type':type(error).__name__,'error':str(error),'supervisor_pid':process.pid if process else None,'supervisor_may_still_be_running':process is not None and process.poll() is None}
    finally:
        files=[p for p in observations.rglob('*') if p.is_file()]
        for name in ('requests.jsonl','summary.json','runtime_versions.json','execution.json'):
            if (BUNDLE/name).exists():files.append(BUNDLE/name)
        files += list(BUNDLE.glob('profile_*.json'))
        inventory={str(p.relative_to(BUNDLE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
        try:archive=build_archive(inventory)
        except Exception as error:archive=None;status={**status,'status':'FAILED','archive_error':str(error)}
        write(observations/'terminal.json',{**status,'archive':archive,'files':inventory,'observer_pid':os.getpid(),'config_sha256':expected,'job_sha256':config['job_sha256'],'manifest_sha256':config['manifest_sha256']})
    return status

def build_archive(files):
    archive=CHAIN/'measurement_outputs_r03.tar.gz';partial=CHAIN/'measurement_outputs_r03.tar.gz.part'
    require(not archive.exists() and not partial.exists(),'Archive/partial exists; no retry')
    with partial.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as packed:
            for name,spec in files.items():
                relative(name);require(not name.endswith(('.onnx','.engine','.plan')),'Model graph forbidden in evidence archive')
                path=BUNDLE/name;require(path.is_file() and not path.is_symlink() and path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256'],'Original evidence changed before archiving')
                packed.add(path,arcname=name,recursive=False)
        stream.flush();os.fsync(stream.fileno())
    require(all((BUNDLE/n).stat().st_size==v['bytes'] and sha(BUNDLE/n)==v['sha256'] for n,v in files.items()),'Original evidence changed during archiving')
    partial.rename(archive);return {'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'encoding':'gzip_tar_original_bytes'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config-sha256',required=True);a=p.parse_args();return 0 if observe(read(CHAIN/'config.json'),a.config_sha256)['status']=='MEASUREMENT_EXIT_ZERO_PENDING_INDEPENDENT_AUDIT' else 1
if __name__=='__main__':raise SystemExit(main())
