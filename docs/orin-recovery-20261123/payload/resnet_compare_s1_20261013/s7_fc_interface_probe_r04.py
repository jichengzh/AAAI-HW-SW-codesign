import tensorrt as trt,numpy as np,json,time,fcntl
from pathlib import Path
ROOT=Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_fc_interface_probe_r04')
ROOT.mkdir(exist_ok=True)
logger=trt.Logger(trt.Logger.INFO)
results=[]
with open('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for mode in ('r03_cast','half_internal','half_io'):
  builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  inp=network.add_input('input',trt.float32,(1,8,1,1))
  conv=network.add_convolution_nd(inp,4,(1,1),np.arange(32,dtype=np.float32)/32,np.zeros(4,dtype=np.float32));conv.name='/fc/Gemm'
  reshape=network.add_shuffle(conv.get_output(0));reshape.name='reshape_after_/fc/Gemm';reshape.reshape_dims=(1,4)
  tensor=reshape.get_output(0);tensor.name='output';tensor.dtype=trt.float32;network.mark_output(tensor)
  config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<26)
  config.set_flag(trt.BuilderFlag.FP16);config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);config.clear_flag(trt.BuilderFlag.TF32);config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
  conv.precision=trt.float16;conv.set_output_type(0,trt.float16)
  if mode!='half_io':
   network.unmark_output(tensor);tensor.name='before_cast'
   if mode=='half_internal':tensor.dtype=trt.float16
   cast=network.add_identity(tensor);cast.name='explicit_fp32_output_cast';cast.set_output_type(0,trt.float32)
   cast.get_output(0).dtype=trt.float32;cast.get_output(0).name='output';network.mark_output(cast.get_output(0))
  else:tensor.dtype=trt.float16
  print('PROBE_MODE',mode,'RESHAPE_DTYPE',tensor.dtype,'CONSTRAINT',reshape.precision_is_set,flush=True)
  start=time.time();serialized=builder.build_serialized_network(network,config)
  record={'mode':mode,'success':serialized is not None,'seconds':time.time()-start,'trt':trt.__version__,'reshape_tensor_dtype':str(tensor.dtype),'reshape_precision_is_set':reshape.precision_is_set}
  if serialized is not None:
   (ROOT/(mode+'.engine')).write_bytes(bytes(serialized));runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized);inspector=engine.create_engine_inspector()
   record['inspector']=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
  results.append(record)
 (ROOT/'results.json').write_text(json.dumps(results,indent=2))
