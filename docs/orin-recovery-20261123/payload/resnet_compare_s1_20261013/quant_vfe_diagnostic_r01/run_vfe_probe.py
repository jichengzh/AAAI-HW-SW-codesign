"""Read-only first-divergence probe; never changes final deployment graphs."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--probe-root',type=Path,required=True)
parser.add_argument('--input-a1',type=Path,required=True)
parser.add_argument('--input-a2',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--ort-root',type=Path)
args=parser.parse_args()
if args.ort_root:sys.path.insert(0,str(args.ort_root))
import numpy as np
import onnxruntime as ort


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


manifest=json.loads((args.probe_root/'vfe_probe_manifest.json').read_text())
path=args.probe_root/'vfe_prefix.onnx'
assert sha(path)==manifest['probe_graph_sha256']
options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;options.log_severity_level=3
session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
result=dict(diagnostic_only=True,python=sys.version,numpy=np.__version__,onnxruntime=ort.__version__,
    probe_graph_sha256=sha(path),source_graph_sha256=manifest['source_graph_sha256'],points={})
for agents,input_path in ((1,args.input_a1),(2,args.input_a2)):
    row=manifest['points'][str(agents)];assert sha(input_path)==row['input_sha256']
    reference_path=args.probe_root/f'vfe_reference_a{agents}.npz'
    assert sha(reference_path)==row['reference_sha256']
    with np.load(input_path) as pack:feeds={v.name:pack[v.name] for v in session.get_inputs()}
    arrays=session.run(None,feeds)
    actual={v.name:value for v,value in zip(session.get_outputs(),arrays)}
    comparisons={}
    with np.load(reference_path) as reference:
        for name,value in actual.items():
            error=np.abs(value-reference[name]);indices=np.unravel_index(np.argmax(error),error.shape)
            comparisons[name]=dict(shape=list(value.shape),bitwise_equal=bool(np.array_equal(value,reference[name])),
                max_abs_error=float(error.max()),mean_abs_error=float(error.mean()),
                differing_elements=int(np.count_nonzero(error)),max_error_index=[int(x) for x in indices],
                actual_at_max=float(value[indices]),reference_at_max=float(reference[name][indices]),
                per_feature_max=[float(x) for x in error.max(axis=(0,1))])
        error=np.abs(actual['points_mean']-reference['points_mean'])
        index=np.unravel_index(np.argmax(error),error.shape);voxel,_,coordinate=index
        count=int(feeds['voxel_num_points'][voxel])
        values=feeds['voxel_features'][voxel,:,coordinate]
        diagnostic=dict(voxel_index=int(voxel),coordinate=int(coordinate),actual_num_points=count,
            all_32_original_values=[float(x) for x in values],
            double_sum_divide_only_diagnostic=float(values.astype(np.float64).sum()/count))
    result['points'][str(agents)]=dict(input_sha256=sha(input_path),comparisons=comparisons,
        max_mean_error_case=diagnostic)
args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
