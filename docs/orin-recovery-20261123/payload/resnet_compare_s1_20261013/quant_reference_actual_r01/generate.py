"""Four fixed-input target references, generated only after complete target AP.
No training, random generation, state carry, performance or energy measurement.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

KEYS={'control_a1','control_a2','integer_a1','integer_a2'}
TARGET=Path('/home/jichengzhi/resnet_compare_s1_20261013')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def source_identity(root,r,manifest,ap):
    paths=[Path(__file__),Path(r.__file__),root/'manifest.json',ap]
    paths += [Path(r.__file__).parent/n for n in ('qualification.py','ap_audit.py','sort_export.py') if (Path(r.__file__).parent/n).exists()]
    for p in manifest['points']:
        for key in ('graph','inputs','reference'):
            path=(root/p[key]).resolve();require(root.resolve() in path.parents,'Path outside bundle');paths.append(path)
    return {str(p.resolve()):sha(p) for p in paths}


def runtime_identity(r,s,ap):
    v=s._measurement_versions
    require(v['onnxruntime']==r.TARGET_OPTIONS['version'] and v['numpy']==r.TARGET_OPTIONS['numpy_version'] and v['python'].startswith(r.TARGET_OPTIONS['python_major_minor']+'.'),'Runtime versions differ')
    paths=[Path(v[k]).resolve() for k in ('numpy_module','ort_module','runtime_binary')]
    require(all(r.ORT_PATH.resolve() in p.parents for p in paths),'Runtime outside isolated prefix')
    actual={str(p):sha(p) for p in paths}
    require(sha(v['runtime_binary'])==v['runtime_binary_sha256']==ap['runtime_binary_sha256'],'Actual binary/AP identity differs')
    require(s.get_providers()==['CPUExecutionProvider'],'Provider fallback')
    o=s.get_session_options()
    require(o.intra_op_num_threads==4 and o.inter_op_num_threads==1 and int(o.graph_optimization_level)==0,'Actual session options differ')
    return {'versions':v,'actual_files':actual,'target_options':r.TARGET_OPTIONS}


def point(root,out,r,p,ap):
    import numpy as np
    require(sha(root/p['graph'])==p['graph_sha256'],'Graph differs')
    feed=r.load_pack(root/p['inputs'],p['input_spec']);refs=r.load_pack(root/p['reference'],p['reference_spec'])
    session=r.session(root/p['graph'],out/('profile_'+p['key']))
    before=runtime_identity(r,session,ap)
    require(set(feed)=={v.name for v in session.get_inputs()},'Input names differ')
    names=[v.name for v in session.get_outputs()]
    require(len(names)==6 and len(set(names))==6 and set(names)==set(refs),'Six output contract differs')
    started=time.time();values=session.run(None,feed)
    require(len(values)==6,'Output count differs')
    differences={}
    for name,value in zip(names,values):
        ref=refs[name]
        require(value.shape==ref.shape and value.dtype==ref.dtype,'Output shape/dtype differs')
        require(np.isfinite(value).all() and np.isfinite(ref).all(),'Nonfinite output/reference')
        delta=np.abs(value-ref)
        differences[name]={'allclose':bool(np.allclose(value,ref,rtol=1e-4,atol=1e-5)),
            'max_abs_error':float(delta.max()) if delta.size else 0.,'outside_count':int(np.count_nonzero(delta>1e-5+1e-4*np.abs(ref))),
            'shape':list(value.shape),'dtype':str(value.dtype),'rtol':1e-4,'atol':1e-5}
    after=runtime_identity(r,session,ap);require(before==after,'Runtime identity changed during request')
    profile=Path(session.end_profiling());execution=r.verify_profile(json.loads(profile.read_text()),p['key'].startswith('integer'))
    output=out/(p['key']+'_outputs.npz');np.savez(output,**dict(zip(names,values)))
    receipt={'graph_sha256':p['graph_sha256'],'input_sha256':sha(root/p['inputs']),'reference_sha256':sha(output),
        'host_reference_sha256':sha(root/p['reference']),'outputs':r.pack_spec(output),'request_count':1,
        'runtime_before':before,'runtime_after':after,'profile_file':profile.name,'profile_sha256':sha(profile),'started_epoch':started}
    probe={'graph_sha256':p['graph_sha256'],'versions':before['versions'],'differences':differences,'execution':execution}
    return receipt,probe


def generate(root,output,r,expected_manifest_sha,ap_file,expected_ap_sha,lock_path):
    root=Path(root);output=Path(output);ap_file=Path(ap_file)
    require(sha(root/'manifest.json')==expected_manifest_sha,'Frozen manifest differs')
    require(sha(ap_file)==expected_ap_sha,'Frozen AP file differs')
    manifest=json.loads((root/'manifest.json').read_text());ap=json.loads(ap_file.read_text())
    require(ap.get('status')=='complete' and ap.get('count')==1789,'Complete1789 AP required')
    require(manifest['target_options']==r.TARGET_OPTIONS,'Session options differ')
    require(ap.get('target_options')==r.TARGET_OPTIONS,'AP target options differ')
    require(ap.get('graph_sha256')=={p['key']:p['graph_sha256'] for p in manifest['points']},'AP graph identities differ')
    require(len(manifest['points'])==4 and {p['key'] for p in manifest['points']}==KEYS,'Four fixed graphs required')
    with Path(lock_path).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        try:
            before=source_identity(root,r,manifest,ap_file);points={};probe={}
            for p in manifest['points']:
                points[p['key']],probe[p['key']]=point(root,output,r,p,ap)
                require(source_identity(root,r,manifest,ap_file)==before,'Source/input identity changed')
            after=source_identity(root,r,manifest,ap_file)
            write(output/'probe_results.json',probe)
            identity={'status':'complete','kind':'independent_target_probe_fixed_input_replay','request_count':4,
                'points':points,'target_options':r.TARGET_OPTIONS,'runtime_binary_sha256':ap['runtime_binary_sha256'],
                'probe_results':{'file':'probe_results.json','sha256':sha(output/'probe_results.json')},
                'ap_evidence_sha256':expected_ap_sha,'source_manifest_sha256':expected_manifest_sha,
                'identity_before':before,'identity_after':after,'host_equivalence':False,
                'state':'saved fixed initial EMA independently for every point; no state carry','time':time.time()}
            write(output/'target_reference_identity.json',identity);return identity
        except BaseException as error:
            write(output/'failure.json',{'type':type(error).__name__,'message':str(error),'time':time.time()});raise


def install_runtime_prefix(prefix):
    prefix=Path(prefix).resolve()
    for name in ('numpy','onnxruntime','onnxruntime.capi.onnxruntime_pybind11_state'):
        loaded=sys.modules.get(name)
        require(loaded is None or prefix in Path(loaded.__file__).resolve().parents,'Runtime already loaded outside isolated prefix')
    sys.path.insert(0,str(prefix))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','output','ap-file'):p.add_argument('--'+name,required=True,type=Path)
    for name in ('expected-manifest-sha256','expected-ap-sha256','expected-runner-sha256','expected-code-sha256'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    require(all(TARGET.resolve() in v.resolve().parents for v in (args.root,args.output,args.ap_file)),'Paths must be in approved target root')
    require(sha(__file__)==args.expected_code_sha256,'Generator code differs')
    require(sha(args.root/'runner.py')==args.expected_runner_sha256,'Runner code differs')
    install_runtime_prefix(TARGET/'ort_cpu_20261026_r02/python')
    sys.path.insert(1,str(args.root))
    spec=importlib.util.spec_from_file_location('reference_source_runner',args.root/'runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    generate(args.root,args.output,r,args.expected_manifest_sha256,args.ap_file,args.expected_ap_sha256,TARGET/'.orin-runtime.lock')


if __name__=='__main__':main()
