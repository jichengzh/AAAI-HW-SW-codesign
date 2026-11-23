from pathlib import Path
import sys,json,fcntl
root=Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_complete_classifier_build_probe_r04')
sys.path.insert(0,str(root))
from framework.resnet_compare.trt_runtime import build_engine
from framework.resnet_compare.remote import sha
with open('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 result=build_engine(Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_20261026_latency_dense_FP16_r03/43a8afb90fd2670355ef256cfac4c5ffd1a3a43c94108e3629d340b94770fc53')/'candidate.onnx',root/'dense.engine','FP16')
 import tensorrt as trt
 logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine((root/'dense.engine').read_bytes());inspector=engine.create_engine_inspector()
 result['inspector']=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
 result['source_graph']='/home/jichengzhi/resnet_compare_s1_20261013/s7_20261026_latency_dense_FP16_r03/43a8afb90fd2670355ef256cfac4c5ffd1a3a43c94108e3629d340b94770fc53'
 result['runtime_sha256']=sha(root/'framework/resnet_compare/trt_runtime.py')
 (root/'dense_build.json').write_text(json.dumps(result,indent=2))
 del inspector;del engine;del runtime
