import os,sys,json,hashlib,importlib,importlib.util,importlib.metadata,subprocess,time
from pathlib import Path
root=Path('/home/jichengzhi/resnet_compare_s1_20261013/isolated_torch_compat_r01');site=root/'venv/lib/python3.8/site-packages'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONNOUSERSITE')=='1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
assert importlib.util.find_spec('yaml') is None,'Expected documented missing dependency before repair'
red={'status':'EXPECTED_MISSING_DEPENDENCY','module':'yaml','gpu_probe_performed':False};(root/'yaml_import_red.json').write_text(json.dumps(red,indent=2))
sources=[Path('/usr/lib/python3/dist-packages/yaml'),Path('/usr/lib/python3/dist-packages/PyYAML-5.3.1.egg-info'),Path('/usr/lib/python3/dist-packages/_yaml.cpython-38-aarch64-linux-gnu.so')]
files=[]
for src in sources:
 assert src.exists();files.extend([f for f in src.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'] if src.is_dir() else [src])
before={str(f):{'sha256':sha(f),'bytes':f.stat().st_size} for f in files}
for src in sources:
 dest=site/src.name;assert not dest.exists();dest.symlink_to(src)
importlib.invalidate_caches()
import torch,yaml,_yaml
torch.set_num_threads(2);torch.set_num_interop_threads(1)
def forbidden(*a,**kw):raise RuntimeError('GPU API is forbidden during CPU import validation')
for name in ('is_available','device_count','_lazy_init'):setattr(torch.cuda,name,forbidden)
a=Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01');sys.path.insert(0,str(a/'runtime_snapshot'));sys.path.insert(0,str(a))
modules={}
for name in ('framework.resnet_compare.contracts','framework.resnet_compare.runtime','framework.resnet_compare.remote','framework.resnet_compare.trt_runtime','framework.resnet_compare.target_entry','contract','stream_accuracy','run_accuracy','target_wrapper'):
 m=importlib.import_module(name);modules[name]={'origin':m.__file__,'sha256':sha(Path(m.__file__))}
assert yaml.__version__=='5.3.1' and yaml.safe_load('probe: true')=={'probe':True}
assert yaml.load('probe: true',Loader=yaml.CSafeLoader)=={'probe':True}
assert all(sha(Path(f))==v['sha256'] for f,v in before.items())
original=json.loads((root/'original_files_before.json').read_text());changed=[f for f,v in original['files'].items() if not Path(f).is_file() or sha(Path(f))!=v['sha256']];assert not changed
verification=subprocess.run(['dpkg','-V','python3-yaml'],capture_output=True,text=True,check=True);assert not verification.stdout
m=json.loads((root/'installation_manifest.json').read_text());supplement={**m,'revision':'r02_yaml_dependency_completion','prior_manifest_sha256':sha(root/'installation_manifest.json'),'additional_existing_package':{'name':'PyYAML','python_version':'5.3.1','dpkg_package':'python3-yaml','dpkg_version':'5.3.1-1ubuntu0.1','mechanism':'new_venv_symlinks_to_unchanged_system_files','source_files':before,'symlinks':{str(site/src.name):str(src) for src in sources},'dpkg_verify_exit':verification.returncode,'dpkg_verify_output':verification.stdout}}
(root/'installation_manifest_r02.json').write_text(json.dumps(supplement,indent=2))
pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True);(root/'pip_check_r02.json').write_text(json.dumps({'exit_code':pip.returncode,'output':pip.stdout,'python':sys.executable,'gpu_probe_performed':False},indent=2))
r={'status':'PASS_CPU_FULL_RUNTIME_IMPORT_ONLY','python':sys.executable,'torch_version':torch.__version__,'yaml_version':yaml.__version__,'yaml_origin':yaml.__file__,'yaml_native_origin':_yaml.__file__,'yaml_libyaml':yaml.__with_libyaml__,'modules':modules,'original_files_verified_unchanged':len(original['files']),'original_changed_files':changed,'system_yaml_files_verified_unchanged':len(before),'no_gpu_probe_performed':True,'cuda_query_and_initialization_blocked_in_process':True,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'script_sha256':sha(Path(__file__)),'pip_check_exit':pip.returncode}
(root/'cpu_full_runtime_import_r02.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
