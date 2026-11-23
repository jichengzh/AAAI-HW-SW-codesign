"""Run ONLY with original official NumPy1.22.4 to freeze exact default argsort ties."""
import hashlib,json,sys
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(value.tobytes()).hexdigest()

def main(root,verify=False):
    root=Path(root)
    if np.__version__!='1.22.4':raise ValueError('Original NumPy1.22.4 required')
    binary=Path(np.core._multiarray_umath.__file__)
    if digest(binary)!='674f8b0f4c7dec1adb4232dcdde3ec3a54fae75ea3c0984822489271b3226c40':raise ValueError('Original NumPy binary differs')
    evidence=json.loads((root/'ap_evidence.json').read_text());orders={};points={}
    for method,spec in evidence['methods'].items():
        frame=root/spec['frames_file']
        if digest(frame)!=spec['frames_sha256']:raise ValueError('Frozen frame trace changed')
        rows=[json.loads(line) for line in frame.read_text().splitlines()]
        for iou in ('0.3','0.5','0.7'):
            scores=np.asarray([v for row in rows for v in row['statistics'][iou]['score']])
            order=np.argsort(-scores);key=method+'_'+iou;orders[key]=order
            points[key]=dict(score_sha256=array_sha(scores),score_dtype=str(scores.dtype),count=len(scores),order_sha256=array_sha(order),order_dtype=str(order.dtype),frames_sha256=spec['frames_sha256'])
    identity=dict(numpy=np.__version__,python=sys.version,executable=sys.executable,binary_path=str(binary),binary_sha256=digest(binary),exporter_sha256=digest(__file__),ap_evidence_sha256=digest(root/'ap_evidence.json'),points=points)
    if verify:
        saved=json.loads((root/'sort_identity.json').read_text())
        if saved['identity']!=identity:raise ValueError('Original sort identity differs')
        if digest(root/'sort_indices.npz')!=saved['indices_sha256']:raise ValueError('Original sort pack changed')
        with np.load(root/'sort_indices.npz',allow_pickle=False) as data:
            if set(data.files)!=set(orders) or not all(np.array_equal(data[k],v) for k,v in orders.items()):raise ValueError('Original argsort order differs')
    else:
        if (root/'sort_identity.json').exists() or (root/'sort_indices.npz').exists():raise ValueError('Refusing overwrite of frozen sorts')
        np.savez(root/'sort_indices.npz',**orders)
        (root/'sort_identity.json').write_text(json.dumps(dict(identity=identity,indices_sha256=digest(root/'sort_indices.npz')),indent=2)+'\n')

if __name__=='__main__':main(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='--verify')
