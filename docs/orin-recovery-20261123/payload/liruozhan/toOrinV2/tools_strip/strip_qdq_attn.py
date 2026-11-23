import onnx, re
from onnx import checker
SRC="/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx"
DST="/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx"
PAT=re.compile(r"(attentions|/attn/|_attn|attn_module|interaction_transformer|/transformer/encoder|/transformer/decoder)")
m=onnx.load(SRC); g=m.graph
targets=[n for n in g.node if n.op_type=="QuantizeLinear" and PAT.search(n.name)]
print("attention/transformer Q nodes to strip:",len(targets))
print("total Q before:",sum(1 for n in g.node if n.op_type=="QuantizeLinear"))
remove=set(); rewire={}; drop_init=set()
for q in targets:
    q_in=q.input[0]; q_out=q.output[0]
    dqs=[n for n in g.node if n.op_type=="DequantizeLinear" and n.input[0]==q_out]
    assert dqs, q.name
    for dq in dqs:
        rewire[dq.output[0]]=q_in; remove.add(dq.name); drop_init.update(dq.input[1:])
    remove.add(q.name); drop_init.update(q.input[1:])
for n in g.node:
    for i in range(len(n.input)):
        t=n.input[i]
        while t in rewire: t=rewire[t]
        n.input[i]=t
new=[n for n in g.node if n.name not in remove]
print("nodes:",len(g.node),"->",len(new))
del g.node[:]; g.node.extend(new)
used=set()
for n in g.node: used.update(n.input)
keep=[i for i in g.initializer if not (i.name in drop_init and i.name not in used)]
print("inits:",len(g.initializer),"->",len(keep))
del g.initializer[:]; g.initializer.extend(keep)
vi=[v for v in g.value_info if v.name not in rewire]
del g.value_info[:]; g.value_info.extend(vi)
print("Q remaining:",sum(1 for n in g.node if n.op_type=="QuantizeLinear"))
checker.check_model(m)
onnx.save(m,DST); print("saved",DST)
