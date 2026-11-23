from pathlib import Path
import sys,json,fcntl,numpy as np,torch,tensorrt as trt
root=Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_fc_diagnosis_r05');sys.path.insert(0,'/home/jichengzhi/resnet_compare_s1_20261013/s7_complete_classifier_build_probe_r04b')
from framework.resnet_compare.trt_runtime import apply_fc_only_policy,add_fp32_output_cast
with open('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pack=np.load(root/'host_dense_pooled.npz');debug=np.load(root/'captured.npz');logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
 inp=network.add_input('features',trt.float32,(1,2048));w=network.add_constant((1000,2048),np.ascontiguousarray(pack['weight']));w.name='fc.weight';mm=network.add_matrix_multiply(inp,trt.MatrixOperation.NONE,w.get_output(0),trt.MatrixOperation.TRANSPOSE);mm.name='/fc/Gemm';b=network.add_constant((1000,),np.ascontiguousarray(pack['bias']));b.name='fc.bias';sh=network.add_shuffle(b.get_output(0));sh.reshape_dims=(1,1000);add=network.add_elementwise(mm.get_output(0),sh.get_output(0),trt.ElementWiseOperation.SUM);out=add.get_output(0);out.name='logits';network.mark_output(out)
 config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30);config.set_flag(trt.BuilderFlag.FP16);config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);config.clear_flag(trt.BuilderFlag.TF32);config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED;policy=apply_fc_only_policy(network,config,trt);casts=add_fp32_output_cast(network,trt)
 serialized=builder.build_serialized_network(network,config);assert serialized is not None;(root/'head.engine').write_bytes(bytes(serialized));runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized);context=engine.create_execution_context();stream=torch.cuda.Stream();buffers=[torch.empty(tuple(engine.get_binding_shape(i)),dtype=torch.float32,device='cuda') for i in range(engine.num_bindings)];bindings=[b.data_ptr() for b in buffers];ii=next(i for i in range(engine.num_bindings) if engine.binding_is_input(i));oi=1-ii
 weight=torch.from_numpy(pack['weight']).cuda();bias=torch.from_numpy(pack['bias']).cuda();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 values={}
 for label,features in [('host',pack['features']),('target',debug['diagnostic_pooled'])]:
  trtrows=[];halfrows=[];quantfp32=[]
  for x in features:
   tensor=torch.from_numpy(x[None]).cuda()
   with torch.cuda.stream(stream):
    buffers[ii].copy_(tensor);assert context.execute_async_v2(bindings,stream.cuda_stream)
   stream.synchronize();trtrows.append(buffers[oi].cpu().numpy().copy())
   with torch.no_grad():
    halfrows.append(torch.nn.functional.linear(tensor.half(),weight.half(),bias.half()).float().cpu().numpy())
    quantfp32.append(torch.nn.functional.linear(tensor.half().float(),weight.half().float(),bias.half().float()).half().float().cpu().numpy())
  values[label+'_trt']=np.concatenate(trtrows);values[label+'_torch_half']=np.concatenate(halfrows);values[label+'_quantized_fp32_accum']=np.concatenate(quantfp32)
 np.savez(root/'head_comparison.npz',**values)
 inspector=engine.create_engine_inspector();inspector.execution_context=context;(root/'head_evidence.json').write_text(json.dumps({'policy':policy,'inspector':json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))},indent=2));del inspector;del context;del engine;del runtime
