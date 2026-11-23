from pathlib import Path
import sys,json,fcntl,hashlib,numpy as np,torch,tensorrt as trt,time
root=Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_fc_diagnosis_r05')
sys.path.insert(0,'/home/jichengzhi/resnet_compare_s1_20261013/s7_complete_classifier_build_probe_r04b')
from framework.resnet_compare.trt_runtime import apply_fc_only_policy,add_fp32_output_cast
source=Path(sys.argv[1]);root.mkdir(exist_ok=True)
with open('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,logger);assert parser.parse((source/'candidate.onnx').read_bytes())
 config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30);config.set_flag(trt.BuilderFlag.FP16);config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);config.clear_flag(trt.BuilderFlag.TF32);config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
 policy=apply_fc_only_policy(network,config,trt);casts=add_fp32_output_cast(network,trt)
 flatten=[network.get_layer(i) for i in range(network.num_layers) if network.get_layer(i).name=='/Flatten'];assert len(flatten)==1
 pooled=flatten[0].get_output(0);pooled.name='diagnostic_pooled';pooled.dtype=trt.float32;network.mark_output(pooled)
 serialized=builder.build_serialized_network(network,config);assert serialized is not None
 enginepath=root/'dense_debug.engine';enginepath.write_bytes(bytes(serialized));runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized);context=engine.create_execution_context()
 buffers=[torch.empty(tuple(engine.get_binding_shape(i)),dtype=torch.float32 if engine.get_binding_dtype(i)==trt.float32 else torch.float16,device='cuda') for i in range(engine.num_bindings)];bindings=[b.data_ptr() for b in buffers];names=[engine.get_binding_name(i) for i in range(engine.num_bindings)];inputindex=next(i for i in range(engine.num_bindings) if engine.binding_is_input(i));outputs={n:[] for i,n in enumerate(names) if i!=inputindex};stream=torch.cuda.Stream()
 samples=np.load(source/'samples.npz')['inputs']
 with torch.cuda.stream(stream):
  for sample in samples:
   buffers[inputindex].copy_(torch.from_numpy(sample[None]),non_blocking=False)
   assert context.execute_async_v2(bindings,stream.cuda_stream);stream.synchronize()
   for i,n in enumerate(names):
    if i!=inputindex:outputs[n].append(buffers[i].cpu().numpy().copy())
 np.savez(root/'captured.npz',**{n:np.concatenate(v,axis=0) for n,v in outputs.items()})
 inspector=engine.create_engine_inspector();inspector.execution_context=context
 (root/'evidence.json').write_text(json.dumps({'policy':policy,'casts':casts,'engine_sha256':hashlib.sha256(enginepath.read_bytes()).hexdigest(),'source_onnx_sha256':hashlib.sha256((source/'candidate.onnx').read_bytes()).hexdigest(),'sample_pack_sha256':hashlib.sha256((source/'samples.npz').read_bytes()).hexdigest(),'bindings':names,'inspector':json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))},indent=2))
 del inspector;del context;del engine;del runtime
