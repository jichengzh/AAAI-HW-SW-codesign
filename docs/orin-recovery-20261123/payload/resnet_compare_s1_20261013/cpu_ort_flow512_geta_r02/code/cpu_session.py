"""One frozen ORT CPU session policy shared by accuracy and timing."""
CONFIG={'provider':'CPUExecutionProvider','intra_op_num_threads':4,'inter_op_num_threads':1,
        'execution_mode':'ORT_SEQUENTIAL','graph_optimization_level':'ORT_ENABLE_ALL','profiling':False}

def create_session(path,ort):
    if ort.__version__!='1.27.0':raise ValueError('ORT version mismatch')
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_profiling=False
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
