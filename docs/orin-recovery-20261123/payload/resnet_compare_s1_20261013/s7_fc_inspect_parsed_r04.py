import sys,json,tensorrt as trt
from pathlib import Path
sys.path.insert(0,'/home/jichengzhi/resnet_compare_s1_20261013/s7_20261026_latency_dense_FP16_r03/43a8afb90fd2670355ef256cfac4c5ffd1a3a43c94108e3629d340b94770fc53')
from framework.resnet_compare.trt_runtime import apply_fc_only_policy,add_fp32_output_cast
logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,logger)
assert parser.parse((Path('/home/jichengzhi/resnet_compare_s1_20261013/s7_20261026_latency_dense_FP16_r03/43a8afb90fd2670355ef256cfac4c5ffd1a3a43c94108e3629d340b94770fc53')/'candidate.onnx').read_bytes())
config=builder.create_builder_config();policy=apply_fc_only_policy(network,config,trt);casts=add_fp32_output_cast(network,trt)
rows=[]
for i in range(network.num_layers):
 layer=network.get_layer(i)
 if i<network.num_layers-12:continue
 rows.append({'name':layer.name,'type':str(layer.type),'precision_is_set':layer.precision_is_set,'precision':str(layer.precision),'outputs':[{'name':layer.get_output(j).name,'dtype':str(layer.get_output(j).dtype),'shape':list(layer.get_output(j).shape),'output_type_is_set':layer.output_type_is_set(j),'output_type':str(layer.get_output_type(j))} for j in range(layer.num_outputs)],'inputs':[{'name':layer.get_input(j).name,'dtype':str(layer.get_input(j).dtype),'shape':list(layer.get_input(j).shape)} for j in range(layer.num_inputs) if layer.get_input(j) is not None]})
print(json.dumps({'policy':policy,'casts':casts,'layers':rows},indent=2))
