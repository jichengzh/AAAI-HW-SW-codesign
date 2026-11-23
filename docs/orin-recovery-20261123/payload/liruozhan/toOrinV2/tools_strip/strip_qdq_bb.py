import onnx
from onnx import checker
SRC="/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx"
DST="/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_bb.onnx"
m=onnx.load(SRC); g=m.graph
prod={}
for n in g.node:
    for o in n.output: prod[o]=n
def keep(n):
    # 仅 backbone 卷积区保留 INT8;且剔除 Reshape/Cast 后的 Q(8.5 QDQToCopy 断言)
    if "img_backbone" not in n.name: return False
    src=n.input[0]
    if src in prod and prod[src].op_type in ("Reshape","Cast"): return False
    return True
targets=[n for n in g.node if n.op_type=="QuantizeLinear" and not keep(n)]
print("Q to strip:",len(targets),"of",sum(1 for n in g.node if n.op_type=="QuantizeLinear"))
remove=set(); rewire={}; drop_init=set()
for q in targets:
    q_in=q.input[0]; q_out=q.output[0]
    for dq in [x for x in g.node if x.op_type=="DequantizeLinear" and x.input[0]==q_out]:
        rewire[dq.output[0]]=q_in; remove.add(dq.name); drop_init.update(dq.input[1:])
    remove.add(q.name); drop_init.update(q.input[1:])
for n in g.node:
    for i in range(len(n.input)):
        t=n.input[i]
        while t in rewire: t=rewire[t]
        n.input[i]=t
new=[n for n in g.node if n.name not in remove]
del g.node[:]; g.node.extend(new)
used=set()
for n in g.node: used.update(n.input)
keepi=[i for i in g.initializer if not (i.name in drop_init and i.name not in used)]
del g.initializer[:]; g.initializer.extend(keepi)
vi=[v for v in g.value_info if v.name not in rewire]
del g.value_info[:]; g.value_info.extend(vi)
print("Q remaining:",sum(1 for n in g.node if n.op_type=="QuantizeLinear"))
checker.check_model(m); onnx.save(m,DST); print("saved",DST)
