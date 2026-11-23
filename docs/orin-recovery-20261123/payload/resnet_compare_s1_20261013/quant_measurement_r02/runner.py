"""Two-phase CPU ORT measurement; fixed explicit randomness and initial EMA replay."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

TARGET_ROOT = Path('/home/jichengzhi/resnet_compare_s1_20261013')
ORT_PATH = TARGET_ROOT/'ort_cpu_20261026_r02/python'
sys.path.insert(0,str(ORT_PATH))
import numpy as np
HOST_REFERENCE_OPTIONS = dict(version='1.19.2', provider='CPUExecutionProvider', intra_op_threads=4,
               inter_op_threads=1, graph_optimization='ORT_DISABLE_ALL')
TARGET_OPTIONS = {**HOST_REFERENCE_OPTIONS,'version':'1.27.0','numpy_version':'1.26.4','python_major_minor':'3.12'}
WHEEL_INVENTORY = Path('/data/jichengzhi_v2x/resnet_compare_20261013/s7_hardware_adaptation_20261026/ort_wheels/ort127_cp312/wheel_inventory.json')
BOUNDARY = dict(backend='ONNX Runtime CPU; no TensorRT or GPU', independent_samples=2,
    random='fixed explicit native uniform arrays replayed; no argmax substitution',
    state='each request independently replays its saved initial EMA; final EMA verified, no cross-request state carry',
    forward='synchronous CPU session.run including output allocation',
    service='CPU input array copies through CPU output copies; excludes NPZ loading, preprocessing, uniform generation, postprocessing and communication',
    energy='VIN_SYS_5V0 designated rail integral; includes reader start/stop boundary overhead; not whole-board or GPU energy',
    geometry='full saved dynamic voxel count, actual FP64 transforms; geometry and codebook remain float')


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def array_spec(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def pack_spec(path):
    with np.load(path,allow_pickle=False) as p:
        arrays={k:array_spec(p[k]) for k in p.files}
    return dict(sha256=sha(path),arrays=arrays)


def load_pack(path,spec):
    require(sha(path)==spec['sha256'],'NPZ SHA mismatch')
    with np.load(path,allow_pickle=False) as p: result={k:p[k] for k in p.files}
    require({k:array_spec(v) for k,v in result.items()}==spec['arrays'],'Array identity mismatch')
    require(all(np.isfinite(v).all() for v in result.values()),'Nonfinite input/reference')
    return result


def compare(outputs,refs):
    require(len(outputs)==len(refs),'Output count mismatch')
    differences=[]
    for output,reference in zip(outputs,refs):
        require(output.shape==reference.shape and output.dtype==reference.dtype,'Output shape/dtype mismatch')
        require(np.isfinite(output).all(),'Nonfinite output')
        require(np.allclose(output,reference,rtol=1e-4,atol=1e-5),'Output differs from host reference')
        differences.append(float(np.max(np.abs(output-reference))) if output.size else 0.)
    return dict(passed=True,max_abs_per_output=differences,rtol=1e-4,atol=1e-5)


def verify_profile(events,integer):
    kernels=[e['args'] for e in events if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time')]
    require(bool(kernels),'No execution kernels in profile')
    require(all(k.get('provider')=='CPUExecutionProvider' for k in kernels),'Provider fallback')
    count=sum(k.get('op_name')=='ConvInteger' for k in kernels)
    require(count==(122 if integer else 0),'Actual ConvInteger execution differs')
    return dict(convinteger=count,providers=sorted(set(k['provider'] for k in kernels)),kernel_count=len(kernels))


def runtime_versions(ort_version):
    python_version='.'.join(str(v) for v in sys.version_info[:3])
    require(ort_version==TARGET_OPTIONS['version'],'Target ORT version differs')
    require(np.__version__==TARGET_OPTIONS['numpy_version'],'Target NumPy version differs')
    require('.'.join(str(v) for v in sys.version_info[:2])==TARGET_OPTIONS['python_major_minor'],'Target Python version differs')
    return dict(python=python_version,numpy=np.__version__,onnxruntime=ort_version)


def session(path,prefix):
    sys.path.insert(0,str(ORT_PATH))
    import onnxruntime as ort
    versions=runtime_versions(ort.__version__)
    o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
    o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.enable_profiling=True;o.profile_file_prefix=str(prefix)
    s=ort.InferenceSession(str(path),o,providers=['CPUExecutionProvider'])
    s.disable_fallback()
    require(s.get_providers()==['CPUExecutionProvider'],'Provider differs')
    s._measurement_versions=versions
    return s


def append(path,row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def stats(values):
    return dict(mean=float(np.mean(values)),p50=float(np.percentile(values,50)),
                p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)))


def schedule(phase='latency'):
    for i in range(256):yield 'numerical',-1,1+i%2,i
    for i in range(50):yield 'warmup',-1,1+i%2,i
    for w in range(3):
        for i in range(200):yield phase,w,1+i%2,i


def request(s,feeds,reader=None):
    if reader: reader.start_request()
    service_start=time.monotonic()
    start=time.perf_counter()
    copied={k:v.copy() for k,v in feeds.items()}
    before=time.perf_counter();outputs=s.run(None,copied);after=time.perf_counter()
    result=[v.copy() for v in outputs]
    end=time.perf_counter()
    service_end=time.monotonic()
    energy=reader.finish_request() if reader else None
    return dict(forward_ms=(after-before)*1000,service_ms=(end-start)*1000,outputs=result,energy=energy,service_start_monotonic_s=service_start,service_end_monotonic_s=service_end)


@contextmanager
def serial_target(root):
    """Same persistent inode and flock protocol as existing target_entry.serial_target."""
    require(TARGET_ROOT in Path(root).resolve().parents,'Run outside approved target root')
    with (TARGET_ROOT/'.orin-runtime.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)


def power_reader(root,manifest):
    path=root/'orin_power_reader.py'
    require(sha(path)==manifest['power_reader_sha256'],'Power reader changed')
    spec=importlib.util.spec_from_file_location('approved_power',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reader=module.open_reader()
    require(reader.probe().get('available') is True,'Authorized rail unavailable')
    return reader


def verify_identity(manifest,identity):
    expected={v+'_a'+str(a) for v in ('control','integer') for a in (1,2)}
    points=manifest['points']
    require(len(points)==4 and {p['key'] for p in points}==expected,'Exactly four graph points required')
    require(identity['options']==HOST_REFERENCE_OPTIONS and set(identity['points'])==expected,'Reference options/points differ')
    for p in points:
        require({k:identity['points'][p['key']][k] for k in ('graph_sha256','input_sha256','reference_sha256')}==dict(graph_sha256=p['graph_sha256'],
             input_sha256=p['input_spec']['sha256'],reference_sha256=p['reference_spec']['sha256']),
             'Reference graph/input/output binding mismatch')
    for a in (1,2):
        pair=[p for p in points if p['key'].endswith('_a'+str(a))]
        require(pair[0]['input_spec']==pair[1]['input_spec'],'Methods must use identical inputs')


def validate_energy(row):
    e=row['energy']
    require(e['method']=='sensor_integral' and e['rail_label']=='VIN_SYS_5V0' and
        e['sensor_id']=='ina3221:/sys/class/hwmon/hwmon3:VIN_SYS_5V0' and
        e['scope']=='VIN_SYS_5V0_rail_not_GPU_or_whole_board' and
        e['integration']=='trapezoidal_actual_sensor_samples','Wrong energy provenance')
    for key in ('energy_j','sampled_power_min_w','sampled_power_max_w','start_monotonic_s','end_monotonic_s','sample_interval_requested_s'):
        require(math.isfinite(e[key]) and e[key]>0,'Invalid energy numeric evidence: '+key)
    require(type(e['sample_count']) is int and e['sample_count']>=2,'Insufficient power samples')
    require(e['sampled_power_max_w']>=e['sampled_power_min_w'],'Invalid power range')
    require(.001<=e['sample_interval_requested_s']<=.02,'Power poll interval differs')
    require(e['start_monotonic_s']<=row['service_start_monotonic_s']<row['service_end_monotonic_s']<=e['end_monotonic_s'],'Energy does not enclose service')
    duration=e['end_monotonic_s']-e['start_monotonic_s']
    require(e['sampled_power_min_w']*duration<=e['energy_j']*(1+1e-9) and
            e['energy_j']<=e['sampled_power_max_w']*duration*(1+1e-9),'Energy outside sampled-power duration bounds')


def execute(root,expected_manifest_sha256):
    root=Path(root)
    require(sha(root/'manifest.json')==expected_manifest_sha256,'External manifest SHA mismatch')
    m=json.loads((root/'manifest.json').read_text())
    require(m['target_options']==TARGET_OPTIONS,'Session options changed')
    require(sha(root/'wheel_inventory.json')==m['wheel_inventory_sha256'],'Wheel inventory changed')
    require(m['host_reference_options']==HOST_REFERENCE_OPTIONS,'Host reference options changed')
    require(sha(root/'reference_identity.json')==m['reference_identity_sha256'],'Reference identity changed')
    verify_identity(m,json.loads((root/'reference_identity.json').read_text()))
    require(sha(__file__)==m['runner_sha256'],'Runner changed')
    require(not (root/'requests.jsonl').exists(),'Refusing to overwrite/resume incomplete request series')
    sessions={};feeds={};refs={};executions={}
    for p in m['points']:
        key=p['key'];graph=root/p['graph']
        require(sha(graph)==p['graph_sha256'],'Graph SHA changed')
        feeds[key]=load_pack(root/p['inputs'],p['input_spec'])
        refs[key]=load_pack(root/p['reference'],p['reference_spec'])
        sessions[key]=session(graph,root/('profile_'+key))
        require(set(feeds[key])=={v.name for v in sessions[key].get_inputs()},'Input contract differs')
        require(set(refs[key])=={v.name for v in sessions[key].get_outputs()},'Reference contract differs')
    versions={key:s._measurement_versions for key,s in sessions.items()}
    write(root/'runtime_versions.json',dict(sessions=versions,wheel_inventory_sha256=m['wheel_inventory_sha256'],host_reference_options=HOST_REFERENCE_OPTIONS,target_options=TARGET_OPTIONS))
    reader=power_reader(root,m)
    rows=[]
    # Sessions are created once, profiled once, then reused unchanged across both phases.
    for phase in ('latency','energy'):
        for variant in ('control','integer'):
            for stage,window,a,index in schedule(phase):
                key=variant+'_a'+str(a);s=sessions[key]
                row=request(s,feeds[key],reader if phase=='energy' and stage=='energy' else None)
                if row['energy'] is not None:validate_energy(row)
                output=row.pop('outputs')
                reference=[refs[key][v.name] for v in s.get_outputs()]
                check=compare(output,reference)
                if key not in executions:
                    profile=s.end_profiling()
                    executions[key]=verify_profile(json.loads(Path(profile).read_text()),variant=='integer')
                    write(root/'execution.json',executions)
                entry=dict(row,phase=phase,stage=stage,window=window,agent_count=a,index=index,
                           variant=variant,graph_sha256=next(p['graph_sha256'] for p in m['points'] if p['key']==key),
                           numerical=check,wall_time=time.time())
                append(root/'requests.jsonl',entry);rows.append(entry)
                if index%10==0:print(json.dumps({k:entry[k] for k in ('phase','stage','window','agent_count','index','variant')}),flush=True)
    summaries={}
    for phase in ('latency','energy'):
        for variant in ('control','integer'):
            selected=[r for r in rows if r['stage']==phase and r['variant']==variant]
            require(len(selected)==600,'Request count mismatch')
            summaries[phase+'_'+variant]={k:stats([r[k] for r in selected]) for k in ('forward_ms','service_ms')}
            if phase=='energy':summaries[phase+'_'+variant]['rail_energy_j']=stats([r['energy']['energy_j'] for r in selected])
    write(root/'summary.json',dict(status='complete',boundary=BOUNDARY,host_reference_options=HOST_REFERENCE_OPTIONS,target_options=TARGET_OPTIONS,runtime_versions=versions,wheel_inventory_sha256=m['wheel_inventory_sha256'],measurements=summaries,
                                  manifest_sha256=sha(root/'manifest.json'),same_sessions_both_phases=True))


def prepare(source,destination,references):
    source=Path(source);root=Path(destination);references=Path(references)
    identity=json.loads((references/'reference_identity.json').read_text())
    require(identity['options']==HOST_REFERENCE_OPTIONS,'Reference options differ')
    root.mkdir(parents=True,exist_ok=False)
    points=[]
    for variant in ('control','integer'):
        for a in (1,2):
            key=variant+'_a'+str(a)
            stem='calibrated_w8a8_explicit_random' if variant=='control' else 'hardware_adapted_mixed_uint8'
            name=stem+('_a1' if a==1 else '')+'.onnx'
            inp='first_validation_full_inputs_a'+str(a)+'.npz'
            ref=key+'_outputs.npz'
            for src,dst in ((source/name,root/name),(source/inp,root/inp),(references/ref,root/ref)):
                if not dst.exists():shutil.copyfile(src,dst)
            require({k:identity['points'][key][k] for k in ('graph_sha256','input_sha256','reference_sha256')}==dict(graph_sha256=sha(root/name),input_sha256=sha(root/inp),reference_sha256=sha(root/ref)), 'Reference graph/input/output binding mismatch')
            points.append(dict(key=key,graph=name,graph_sha256=sha(root/name),inputs=inp,
                               input_spec=pack_spec(root/inp),reference=ref,reference_spec=pack_spec(root/ref)))
    shutil.copyfile(WHEEL_INVENTORY,root/'wheel_inventory.json')
    shutil.copyfile(references/'reference_identity.json',root/'reference_identity.json')
    shutil.copyfile(__file__,root/'runner.py')
    reader=Path('/data/jichengzhi_v2x/resnet_compare_20261013/s1_real_validation_20261016/scripts/orin_power_reader.py')
    shutil.copyfile(reader,root/reader.name)
    write(root/'manifest.json',dict(points=points,host_reference_options=HOST_REFERENCE_OPTIONS,target_options=TARGET_OPTIONS,boundary=BOUNDARY,wheel_inventory_sha256=sha(root/'wheel_inventory.json'),
          reference_identity_sha256=sha(root/'reference_identity.json'),runner_sha256=sha(root/'runner.py'),power_reader_sha256=sha(root/reader.name)))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--references',required=True)
    p=sub.add_parser('run');p.add_argument('--root',required=True);p.add_argument('--expected-manifest-sha256',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.source,args.destination,args.references)
    else:
        with serial_target(args.root):execute(args.root,args.expected_manifest_sha256)

if __name__=='__main__': main()
