import tensorrt as trt,numpy as np,json,time,fcntl
from pathlib import Path
ROOT=Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_fc_interface_matmul_probe_r04');ROOT.mkdir(exist_ok=True)
logger=trt.Logger(trt.Logger.INFO);results=[]
with open('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for mode in ('split_fp16_matmul_fp32_bias','complete_fp16_classifier'):
  builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  inp=network.add_input('input',trt.float32,(1,8));weights=network.add_constant((4,8),np.arange(32,dtype=np.float32)/32);weights.name='fc.weight'
  mm=network.add_matrix_multiply(inp,trt.MatrixOperation.NONE,weights.get_output(0),trt.MatrixOperation.TRANSPOSE);mm.name='/fc/Gemm'
  bias=network.add_constant((4,),np.ones(4,dtype=np.float32)/10);bias.name='fc.bias';reshape=network.add_shuffle(bias.get_output(0));reshape.reshape_dims=(1,4)
  add=network.add_elementwise(mm.get_output(0),reshape.get_output(0),trt.ElementWiseOperation.SUM);add.name='/fc/Gemm/bias_add'
  tensor=add.get_output(0);tensor.name='output';tensor.dtype=trt.float32;network.mark_output(tensor)
  config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<26);config.set_flag(trt.BuilderFlag.FP16);config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);config.clear_flag(trt.BuilderFlag.TF32);config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
  mm.precision=trt.float16;mm.set_output_type(0,trt.float16);dtype=trt.float32 if mode.startswith('split') else trt.float16;add.precision=dtype;add.set_output_type(0,dtype)
  network.unmark_output(tensor);tensor.name='before_cast';cast=network.add_identity(tensor);cast.name='explicit_fp32_output_cast';cast.set_output_type(0,trt.float32);cast.get_output(0).dtype=trt.float32;cast.get_output(0).name='output';network.mark_output(cast.get_output(0))
  print('PROBE_MODE',mode,flush=True);start=time.time();serialized=builder.build_serialized_network(network,config)
  record={'mode':mode,'success':serialized is not None,'seconds':time.time()-start,'trt':trt.__version__}
  if serialized is not None:
   (ROOT/(mode+'.engine')).write_bytes(bytes(serialized));runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized);inspector=engine.create_engine_inspector();record['inspector']=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON));del inspector;del engine;del runtime
  results.append(record)
 (ROOT/'results.json').write_text(json.dumps(results,indent=2))
