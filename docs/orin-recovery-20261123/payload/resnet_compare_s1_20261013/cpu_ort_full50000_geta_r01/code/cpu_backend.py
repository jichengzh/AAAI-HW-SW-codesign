"""Fixed CPU ORT batch-one FP32 classification execution; no fallback."""
import numpy as np

class Runtime:
    def __init__(self,session):
        self.session=session
        session.disable_fallback()
        self.check_provider()
        inputs=session.get_inputs();outputs=session.get_outputs()
        if len(inputs)!=1 or (inputs[0].name,inputs[0].type,inputs[0].shape)!=('input','tensor(float)',[1,3,224,224]):raise ValueError('input metadata')
        if len(outputs)!=1 or (outputs[0].name,outputs[0].type,outputs[0].shape)!=('logits','tensor(float)',[1,1000]):raise ValueError('output metadata')
    def check_provider(self):
        if self.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('provider fallback')
    def infer(self,x):
        if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.shape!=(1,3,224,224) or not x.flags.c_contiguous or not np.isfinite(x).all():raise ValueError('invalid input')
        self.check_provider();values=self.session.run(['logits'],{'input':x});self.check_provider()
        if len(values)!=1 or not isinstance(values[0],np.ndarray) or values[0].dtype!=np.float32 or values[0].shape!=(1,1000) or not np.isfinite(values[0]).all():raise ValueError('invalid output')
        return values[0]
