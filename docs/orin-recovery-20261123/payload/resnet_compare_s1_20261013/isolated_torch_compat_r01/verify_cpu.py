import os,sys,json,hashlib,importlib.metadata,subprocess,time,zipfile
from pathlib import Path
root=Path('/home/jichengzhi/resnet_compare_s1_20261013/isolated_torch_compat_r01');site=root/'venv/lib/python3.8/site-packages'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONNOUSERSITE')=='1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
import torch,numpy as np,tensorrt as trt
torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert Path(torch.__file__).resolve().is_relative_to(site) if hasattr(Path('.'),'is_relative_to') else str(Path(torch.__file__).resolve()).startswith(str(site)+'/')
assert str(Path(np.__file__).resolve()).startswith(str(site)+'/')
assert str(Path(trt.__file__).resolve()).startswith('/usr/lib/python3.8/dist-packages/tensorrt/')
assert torch.__version__=='2.1.0a0+41361538.nv23.06',torch.__version__
assert np.__version__=='1.22.4' and trt.__version__=='8.5.2.2'
x=torch.arange(12,dtype=torch.float32).reshape(3,4);y=x@x.T;assert y.device.type=='cpu';assert np.array_equal(y.numpy(),np.arange(12,dtype=np.float32).reshape(3,4)@np.arange(12,dtype=np.float32).reshape(3,4).T)
modules={n:importlib.metadata.version(n) for n in ['torch','numpy','tensorrt','filelock','fsspec','Jinja2','networkx','sympy','typing_extensions','mpmath','MarkupSafe']}
loaded=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line and any(s in line for s in ('libtorch','libc10','libcudart','libcudnn','libnvinfer','libopenblas'))))
libs={f:{'sha256':sha(Path(f)),'bytes':Path(f).stat().st_size} for f in loaded if Path(f).is_file()}
m=json.loads((root/'installation_manifest.json').read_text());wheel=root/'wheels'/m['torch']['filename'];native=[]
with zipfile.ZipFile(wheel) as z:
 for item in z.infolist():
  if item.filename.startswith('torch/') and ('.so' in Path(item.filename).name):
   dest=site/item.filename;h=hashlib.sha256(z.read(item)).hexdigest();assert sha(dest)==h;native.append({'path':item.filename,'sha256':h,'bytes':dest.stat().st_size})
prev=json.loads((root/'original_files_before.json').read_text());changed=[f for f,v in prev['files'].items() if not Path(f).is_file() or sha(Path(f))!=v['sha256']]
assert not changed,changed
r={'status':'PASS_CPU_IMPORT_ONLY_NOT_GPU_QUALIFIED','python':sys.executable,'python_version':sys.version,'torch_version':torch.__version__,'torch_cuda_build':torch.version.cuda,'numpy_version':np.__version__,'tensorrt_version':trt.__version__,'torch_origin':torch.__file__,'numpy_origin':np.__file__,'tensorrt_origin':trt.__file__,'module_versions':modules,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'no_gpu_probe_performed':True,'cpu_matmul_exact':True,'loaded_libraries':libs,'wheel_native_files_verified':native,'original_files_verified_unchanged':len(prev['files']),'original_changed_files':changed,'verification_script_sha256':sha(Path(__file__))}
(root/'cpu_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ('status','python','torch_version','torch_cuda_build','numpy_version','tensorrt_version','original_files_verified_unchanged')}))
