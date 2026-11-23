import json,sys,hashlib,subprocess,os,time,sysconfig
from pathlib import Path
root=Path('/home/jichengzhi/resnet_compare_s1_20261013/isolated_torch_compat_r01')
m=json.loads((root/'installation_manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
entries=[m['torch']]+m['dependencies']
for row in entries:
 p=root/'wheels'/row['filename'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
assert sys.version_info[:2]==(3,8),sys.version
venv=root/'venv';assert not venv.exists(),'Never overwrite existing environment'
start=time.time();subprocess.run(['/usr/bin/python3','-m','venv',str(venv)],check=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
argv=[str(venv/'bin/python'),'-m','pip','install','--no-cache-dir','--no-index','--no-deps','--ignore-installed']+[str(root/'wheels'/r['filename']) for r in entries]
with (root/'install.log').open('x') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
site=venv/'lib/python3.8/site-packages';trt=Path('/usr/lib/python3.8/dist-packages/tensorrt');assert trt.is_dir();(site/'tensorrt').symlink_to(trt)
metadata=list(trt.parent.glob('tensorrt-*.dist-info'))+list(trt.parent.glob('tensorrt-*.egg-info'))
for item in metadata:(site/item.name).symlink_to(item)
(root/'install_receipt.json').write_text(json.dumps({'status':'INSTALLED_CPU_VERIFICATION_PENDING','python':str(venv/'bin/python'),'argv':argv,'elapsed_seconds':time.time()-start,'system_site_packages':False,'tensorrt_symlink':str(trt),'tensorrt_metadata':[str(x) for x in metadata],'cuda_visible_devices':''},indent=2))
print('isolated wheel installation completed; GPU not queried')
