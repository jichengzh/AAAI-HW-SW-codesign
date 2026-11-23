import ctypes, tensorrt as trt
ctypes.CDLL("./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so", mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER=trt.Logger(trt.Logger.ERROR)
builder=trt.Builder(TRT_LOGGER)
network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser=trt.OnnxParser(network, TRT_LOGGER)
with open("./onnx/uniad_tiny_imgx0.25_cp.noaten_simp.onnx","rb") as f:
    data=f.read()
if not parser.parse(data):
    for i in range(parser.num_errors): print("PARSE ERR:", parser.get_error(i))
    raise SystemExit("parse failed")
print("network layers:", network.num_layers)
keep=set(l.strip() for l in open("tools_acc/keep.txt") if l.strip())
head=set(l.strip() for l in open("tools_acc/head_float.txt") if l.strip())
names=[network.get_layer(i).name for i in range(network.num_layers)]
nset=set(names)
print("total layers:", len(names), "unique:", len(nset))
print("layers matching keep:", len(nset & keep), "/", len(keep))
print("layers matching head:", len(nset & head), "/", len(head))
print("layers in neither:", len(nset - keep - head))
print("keep names NOT found as layers:", len(keep - nset))
import itertools
print("sample neither layers:", list(itertools.islice((x for x in nset-keep-head),15)))
# dynamic inputs
print("--- network inputs ---")
for i in range(network.num_inputs):
    t=network.get_input(i); print(t.name, t.shape)
