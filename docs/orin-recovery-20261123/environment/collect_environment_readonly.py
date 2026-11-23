#!/usr/bin/env python3
"""Read-only remote metadata collector; run locally with SSH_AUTH_SOCK already set."""
import json, os, pathlib, subprocess
REMOTE = r'''
import json, os, pathlib, platform, re, subprocess, datetime, ast, hashlib
H=pathlib.Path('/home/jichengzhi')
def run(args, **kw):
 p=subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
 return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
def read(p):
 p=pathlib.Path(p)
 return p.read_text(errors='replace').replace('\x00','') if p.exists() else None
pkgs=run(['dpkg-query','-W','-f=${Package}\t${Version}\t${Architecture}\t${db:Status-Status}\n'])
lines=[x.split('\t') for x in pkgs['stdout'].splitlines()]
packages=[{'name':p[0],'version':p[1],'arch':p[2]} for p in lines if len(p)==4 and p[3]=='installed' and re.match(r'^(nvidia|cuda|libcud|libcublas|libcufft|libcurand|libcusolver|libcusparse|libnvinfer|libnvonnx|libnvparsers|tensorrt|ros-|python3|libopencv|libopenblas|libopenmpi|openmpi|libgomp|cmake|gcc|g\+\+|ffmpeg)',p[0])]
probe="import sys,json,importlib.metadata as m; print(json.dumps({'executable':sys.executable,'version':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'sys_path':sys.path,'packages':sorted([{'name':d.metadata['Name'],'version':d.version,'location':str(d.locate_file(''))} for d in m.distributions()],key=lambda d:(d['name'] or '').lower())}))"
interpreters={'system':'/usr/bin/python3','uniad_venv':str(H/'uniad_venv/bin/python'),'miniforge_base':str(H/'miniforge3/bin/python'),'miniforge_uniad':str(H/'miniforge3/envs/uniad/bin/python'),'isolated_torch':str(H/'resnet_compare_s1_20261013/isolated_torch_compat_r01/venv/bin/python')}
envs={}
for name,py in interpreters.items():
 result=run([py,'-B','-c',probe]) if pathlib.Path(py).exists() else {'returncode':127,'stderr':'interpreter missing'}
 if result['returncode']==0:
  try: result=json.loads(result['stdout'])
  except Exception: pass
 root=pathlib.Path(py).parent.parent
 result['pyvenv_cfg']=read(root/'pyvenv.cfg')
 envs[name]=result
# ORT --target directories have no own interpreter; inspect only dist-info metadata.
import importlib.metadata as md
ort={}
for dirname in ['ort_cpu_20261026_r01/python','ort_cpu_20261026_r02/python','ort127_cp312']:
 p=H/'resnet_compare_s1_20261013'/dirname
 ort[dirname]={'path':str(p),'packages':[{'name':d.metadata['Name'],'version':d.version} for d in md.distributions(path=[str(p)])], 'wheel_files':[q.name for q in p.glob('*.whl')]}
# Conda package manifest fields are allowlisted; omit channel/auth URLs.
conda={}
for root in [H/'miniforge3',H/'miniforge3/envs/uniad']:
 conda[str(root)]=[{k:data.get(k) for k in ['name','version','build','build_number','subdir']} for f in sorted((root/'conda-meta').glob('*.json')) for data in [json.loads(f.read_text())]]
wheel_paths=[H/'resnet_compare_s1_20261013/isolated_torch_compat_r01/wheels',H/'resnet_compare_s1_20261013/ort_cpu_20261026_r01/wheels',H/'resnet_compare_s1_20261013/ort127_cp312']
wheels=[{'path':str(f),'bytes':f.stat().st_size} for p in wheel_paths for f in sorted(p.glob('*.whl'))]
# Source dependency extraction, never import project code.
source={}
source_paths=['UniAD/requirements.txt','UniAD/requirements2.0.txt','ROS1/wheeltec_robot/src/uniad_bridge/package.xml','ROS1/wheeltec_robot/src/uniad_bridge/CMakeLists.txt']
for p in source_paths: source[str(H/p)]=read(H/p)
source_revisions={}
for rel in ['mmcv','mmcv_build/mmcv','mmcv_build/mmdetection']:
 source_revisions[str(H/rel)]={'commit':run(['git','-C',str(H/rel),'rev-parse','HEAD'])['stdout'],'tag':run(['git','-C',str(H/rel),'describe','--tags','--always'])['stdout'],'dirty':bool(run(['git','-C',str(H/rel),'status','--porcelain'])['stdout'])}
dependency_imports={}
for rel in ['ditl_rsu_edge_v1/staging2/ditl_trt_backend.py','ditl_rsu_edge_v1/staging2/ditl_ros2_edge_bridge.py','coptv2x_hil_serial_opt_stage2_20261009/runtime/scripts/coptv2x_hil_orin_worker.py','ROS1/wheeltec_robot/src/uniad_bridge/scripts/uniad_cmdvel_bridge.py']:
 p=H/rel
 if not p.exists():continue
 raw=p.read_bytes();tree=ast.parse(raw.decode())
 dependency_imports[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'imports':sorted(set([alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]+[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module]))}
for rel in ['ditl_rsu_edge_v1/Dockerfile.ros2trace','ditl_rsu_edge_v1/Dockerfile.ros2trace-enabled']:
 source[str(H/rel)]=read(H/rel)
docker_images=run(['docker','images','--format','{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'])
print(json.dumps({'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'read-only metadata; no inference/import of torch/TRT/ORT; no credentials or environment dump','machine':platform.machine(),'kernel':platform.release(),'os_release':read('/etc/os-release'),'l4t_release':read('/etc/nv_tegra_release'),'model':read('/proc/device-tree/model'),'cuda_nvcc':run(['/usr/local/cuda/bin/nvcc','--version']),'cuda_realpath':os.path.realpath('/usr/local/cuda'),'ros_dirs':[str(p) for p in pathlib.Path('/opt/ros').glob('*')],'dpkg':packages,'python':envs,'ort_targets':ort,'conda_packages':conda,'wheel_inventory':wheels,'source_dependencies':source,'source_revisions':source_revisions,'dependency_imports':dependency_imports,'docker_images':docker_images},ensure_ascii=False,indent=2))
'''
ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jichengzhi@172.16.62.222','python3 -B -']
p=subprocess.run(ssh,input=REMOTE,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
data=json.loads(p.stdout)
root=pathlib.Path(__file__).resolve().parent
(root/'environment_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(root/'dpkg_versions.tsv').write_text('package\tversion\tarchitecture\n'+''.join(f"{x['name']}\t{x['version']}\t{x['arch']}\n" for x in data['dpkg']))
for name,info in data['python'].items():
 if 'packages' in info:
  (root/f'python_{name}_versions.tsv').write_text('package\tversion\tlocation\n'+''.join(f"{x['name']}\t{x['version']}\t{x['location']}\n" for x in info['packages']))
print('Snapshot saved:',root,'; Python environments:',list(data['python']))
