"""Target-only CPU daemon; root launches after granting exclusive device window."""
import sys,os
from pathlib import Path
APPROVED=Path('/home/jichengzhi/resnet_compare_s1_20261013')
ORT_PREFIX=Path('/home/jichengzhi/resnet_compare_s1_20261013/ort_cpu_20261026_r02/python')
sys.path.insert(0,str(ORT_PREFIX))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time,traceback,fcntl
import numpy as np
import protocol as p
from worker import Worker
from exchange import step


def validate_identity(root,manifest):
    import onnxruntime as ort
    import onnxruntime.capi.onnxruntime_pybind11_state as binary
    options=manifest['target_options']
    p.require(ort.__version__==options['version'] and np.__version__==options['numpy_version'],'runtime package version differs')
    p.require('.'.join(map(str,sys.version_info[:2]))==options['python_major_minor'],'Python version differs')
    p.require(p.sha(binary.__file__)==manifest['runtime_binary_sha256'],'runtime binary SHA differs')
    for name,expected in manifest['code_sha256'].items():
        p.require(Path(name).name==name and p.sha(Path(root)/name)==expected,'frozen execution source differs')
    return dict(runtime_binary_file=binary.__file__,runtime_binary_sha256=p.sha(binary.__file__),target_options=options,
        code_sha256=manifest['code_sha256'],manifest_sha256=p.canonical(manifest),python=sys.version,ort_file=ort.__file__,numpy_file=np.__file__)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--graph-root',required=True,type=Path);args=parser.parse_args()
    approved=APPROVED
    p.require(args.root.resolve().is_relative_to(approved) and args.graph_root.resolve().is_relative_to(approved),'outside approved experiment roots')
    manifest=json.loads((args.root/'manifest.json').read_text());p.require(len(manifest['frame_ids'])==1789 and len(set(manifest['frame_ids']))==1789,'production frozen 1789 frame list required')
    with (approved/'.orin-runtime.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            identity=validate_identity(args.root,manifest)
            p.write_json(args.root/f'run_identity_{time.time_ns()}.json',identity)
            worker=Worker(args.root,manifest,args.graph_root)
            while not (args.root/'completed.json').exists():step(worker);time.sleep(.2)
        except Exception as error:
            traceback.print_exc();p.write_json(args.root/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),overwrite=True)
            raise
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
