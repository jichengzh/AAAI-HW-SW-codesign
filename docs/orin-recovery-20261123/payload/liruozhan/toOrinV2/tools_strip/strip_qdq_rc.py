import onnx
from onnx import checker
SRC="/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx"
DST="/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt/UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_noRC.onnx"
m=onnx.load(SRC); g=m.graph
prod={}
for n in g.node:
    for o in n.output: prod[o]=n
targets=[n for n in g.node if n.op_type=="QuantizeLinear" and n.input[0] in prod and prod[n.input[0]].op_type in ("Reshape","Cast")]
print("target Q nodes:",len(targets))
remove=set()
rewire={}   # dq_out -> q_in
drop_init=set()
for q in targets:
    q_in=q.input[0]; q_out=q.output[0]
    dqs=[n for n in g.node if n.op_type=="DequantizeLinear" and n.input[0]==q_out]
    assert dqs, q.name
    for dq in dqs:
        rewire[dq.output[0]]=q_in
        remove.add(dq.name)
        drop_init.update(dq.input[1:])
    remove.add(q.name)
    drop_init.update(q.input[1:])
# apply rewire to all node inputs
for n in g.node:
    for i in range(len(n.input)):
        t=n.input[i]
        while t in rewire: t=rewire[t]
        n.input[i]=t
# drop nodes
new_nodes=[n for n in g.node if n.name not in remove]
print("nodes before/after:",len(g.node),len(new_nodes))
del g.node[:]
g.node.extend(new_nodes)
# drop now-unused initializers (scale/zp)
used=set()
for n in g.node: used.update(n.input)
keep=[i for i in g.initializer if not (i.name in drop_init and i.name not in used)]
print("initializers before/after:",len(g.initializer),len(keep))
del g.initializer[:]
g.initializer.extend(keep)
# drop stale value_info for removed tensors
removed_outs=set()
for n in list(remove):
    pass
vi=[v for v in g.value_info if v.name not in set(rewire.keys())]
del g.value_info[:]
g.value_info.extend(vi)
checker.check_model(m)
onnx.save(m,DST)
print("saved",DST)
