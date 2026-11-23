"""Bounded atomic data/state protocol, shared by host and target CPU programs."""
import hashlib,json,os
from pathlib import Path
import numpy as np

BASE_INPUTS=('voxel_features','voxel_coords','voxel_num_points','pairwise_t_matrix','record_len')
HEADS=('cls_preds','reg_preds','dir_preds')
OUTPUTS=HEADS+tuple(f'final_freqEMA_{i}' for i in range(3))
MAX_BYTES=100_000_000
OUTPUT_RESERVE=6_000_000


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_spec(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest())


def state_sha(values):return canonical([array_spec(value) for value in values])


def write_json(path,value,overwrite=False):
    path=Path(path)
    if path.exists() and not overwrite:raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def write_pack(path,arrays):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    return dict(sha256=sha(path),bytes=path.stat().st_size,arrays={key:array_spec(value) for key,value in arrays.items()})


def load_pack(path,identity):
    require(sha(path)==identity['sha256'],'NPZ SHA mismatch')
    with np.load(path,allow_pickle=False) as source:arrays={key:source[key] for key in source.files}
    require({key:array_spec(value) for key,value in arrays.items()}==identity['arrays'],'Array SHA/type/shape mismatch')
    require(all(np.isfinite(value).all() for value in arrays.values()),'Nonfinite array')
    return arrays


def uniforms(seed,shape):
    rng=np.random.RandomState(seed)
    return tuple((rng.randint(0,2**32,size=shape,dtype=np.uint32)&np.uint32(0xffffff)).astype(np.float32)*np.float32(2**-24) for _ in range(3))


def check_uniforms(values,hashes):
    require(len(values)==len(hashes)==3,'random level count differs')
    require([array_spec(value)['sha256'] for value in values]==hashes,'random byte SHA differs from native host Torch')


def validate_batch(batch,manifest,next_index,max_bytes=MAX_BYTES,output_reserve=OUTPUT_RESERVE):
    rows=batch['frames']
    require(bool(rows) and len(rows)<=8,'empty/oversized frame batch')
    require(sum(row['bytes']+output_reserve for row in rows)<=max_bytes,'compressed input/output cache bound exceeded')
    for offset,row in enumerate(rows):
        index=next_index+offset
        require(row['index']==index and index<len(manifest['frame_ids']),'missing/repeated/reordered frame')
        require(row['sample_id']==manifest['frame_ids'][index],'wrong frozen sample identity')
        name=row['file']
        require(Path(name).name==name and name.endswith('.npz'),'unsafe filename')


def validate_inputs(values,agents):
    require(set(values)==set(BASE_INPUTS),'input names differ')
    vf,coords,points,pairwise,record=(values[key] for key in BASE_INPUTS)
    require(agents in (1,2) and record.dtype==np.int64 and record.tolist()==[agents],'native agent route differs')
    n=vf.shape[0]
    require(vf.ndim==3 and vf.shape[1:]==(32,4) and vf.dtype==np.float32 and 0<n<=140000,'full voxel features shape/dtype/count differs')
    require(coords.shape==(n,4) and coords.dtype==np.int32 and points.shape==(n,) and points.dtype==np.int32,'voxel row/dtype mismatch')
    require(set(np.unique(coords[:,0]))==set(range(agents)),'agent coordinates differ')
    require(pairwise.shape==(1,2,2,4,4) and pairwise.dtype==np.float64,'real geometry shape/dtype differs')
    require(all(np.isfinite(value).all() for value in values.values()),'nonfinite inputs')


def validate_outputs(outputs,previous,shapes):
    require(set(outputs)==set(OUTPUTS)==set(shapes),'output names differ')
    for key,value in outputs.items():
        require(value.dtype==np.float32 and list(value.shape)==shapes[key] and np.isfinite(value).all(),'output dtype/shape/finiteness differs: '+key)
    for i in range(3):
        expected=np.float32(.9)*previous[i].sum(-1)+np.float32(.1)
        require(np.allclose(outputs[f'final_freqEMA_{i}'].sum(-1),expected,atol=2e-7,rtol=2e-6),'EMA mass update differs')
