"""TorchScript-only runtime; legacy filename retained solely for transport inventory."""
import hashlib,json
from pathlib import Path
import torch
from .runtime import TorchRuntime

def model_state_sha256(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        header=json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':'))
        digest.update(header.encode()+b'\0')
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes()+b'\0')
    return digest.hexdigest()

def validate_script(model,expected_state):
    state=model.state_dict()
    floating=[(name,value) for name,value in state.items() if value.is_floating_point()]
    if not floating or any(value.dtype!=torch.float16 for name,value in floating):
        raise ValueError('Every floating parameter/buffer must be FP16; no empty/frozen state')
    if model_state_sha256(model)!=expected_state:
        raise ValueError('TorchScript state hash differs from authenticated export')
    allowed={'prim::GetAttr','prim::Constant','prim::ListConstruct','prim::TupleConstruct',
             'prim::ListUnpack','aten::_convolution','aten::conv2d','aten::batch_norm',
             'aten::relu','aten::relu_','aten::max_pool2d','aten::adaptive_avg_pool2d',
             'aten::add','aten::add_','aten::flatten','aten::linear','aten::size',
             'aten::Int','aten::view','aten::reshape','aten::contiguous'}
    graph=model.inlined_graph
    nodes=list(graph.nodes())
    ops=[]
    canonical=[]
    for node in nodes:
        if node.kind() not in allowed or list(node.blocks()):
            raise ValueError('Unsupported or dtype-changing TorchScript operator: '+node.kind())
        ops.append(node.kind())
        canonical.append({'kind':node.kind(),'inputs':[v.debugName() for v in node.inputs()],
                          'outputs':[{'name':v.debugName(),'dtype':v.type().scalarType() if isinstance(v.type(),torch.TensorType) else None} for v in node.outputs()],
                          'attributes':{name:node.kindOf(name) for name in node.attributeNames()}})
        for output in node.outputs():
            if isinstance(output.type(),torch.TensorType):
                dtype=output.type().scalarType()
                if dtype not in (None,'Half','Long','Int','Bool'):
                    raise ValueError('Non-FP16 floating tensor in scripted graph')
                constant=output.toIValue()
                if isinstance(constant,torch.Tensor) and constant.is_floating_point() and constant.dtype!=torch.float16:
                    raise ValueError('Non-FP16 embedded floating constant')
    if not any(op in ops for op in ('aten::_convolution','aten::conv2d','aten::linear')):
        raise ValueError('No actual floating computation in scripted graph')
    text=json.dumps(canonical,sort_keys=True,separators=(',',':'))
    return {'floating_state_dtype':'torch.float16','state_sha256':expected_state,
            'floating_state_entries':len(floating),'operators':sorted(set(ops)),
            'graph':canonical,'graph_format':'canonical_nodes_v1','graph_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'evidence_scope':'all floating state and dtype-preserving graph operators; accumulator precision not asserted'}

class ScriptRuntime(TorchRuntime):
    """Native TorchScript/CUDA half execution; no TensorRT or FP32 substitution."""
    def __init__(self,path,expected_state,device='cuda:0'):
        model=torch.jit.load(str(path),map_location='cpu').eval()
        self.script_evidence=validate_script(model,expected_state)
        self.input_dtype=torch.float16
        super().__init__(model,device)
        self.layer_precision=[{'name':'TorchScript_full_FP16','precision':'FP16',**self.script_evidence}]
    def infer(self,tensor):
        result=super().infer(tensor)
        if result['logits'].dtype!=torch.float16:
            raise ValueError('TorchScript output silently changed from FP16')
        return result
