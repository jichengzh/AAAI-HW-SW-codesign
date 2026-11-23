"""Separate bounded diagnostics; cannot produce full50000 qualification."""
import argparse,fcntl,io,json,os,sys
from pathlib import Path
import numpy as np
from sdk_backend import assert_no_torch,library_identity,sha
from identity import validate, verify_libraries
from multi_entry import create_runtime,protocol_output,LOCK
from frozen_stream import serve,send,receive,encode_chunk,digest
from stage_logger import bind, record
REFERENCE_PINS={'copt':'60654021686ee4e65eedc3250a2dd12e5b4a423c9d85e7e192cbb88ace033274',
                'geta':'4a0b2b37c7ed972ccbb18bca8984c5a870d78420213e71428e9a19103428e486'}
REFERENCE_ENGINE_SHA='59f030a76d8565ba888ca354d666b73337309f7e4c8628ae1d6720ed31118690'
INPUT0_SHA='ea4a6f68f067977c0aca4d1b1706fadc63b0dca6c4499cac149b9c9e3d715884'

def diagnostic_identity(a):
    token=validate(a.point,a)  # Always the complete frozen50000 source contract.
    if a.mode not in ('input0','ordered512'):raise ValueError('Unsupported diagnostic mode')
    count=1 if a.mode=='input0' else 512
    rows=json.loads(Path(a.final).read_text())['samples'][:count]
    if len(rows)!=count:raise ValueError('Diagnostic prefix incomplete')
    reference=reference_identity(a,token,rows)
    return {**token,**reference,'scope':'DIAGNOSTIC_ONLY','diagnostic_count':count,'diagnostic_mode':a.mode,
            'diagnostic_entry_sha256':sha(__file__)},rows

def reference_identity(a,token,rows):
    if REFERENCE_PINS.get(a.point)!=a.reference_sha256:
        raise ValueError('reference not registered for point')
    if sha(a.reference)!=a.reference_sha256:
        raise ValueError('reference metadata SHA mismatch')
    meta=json.loads(Path(a.reference).read_text())
    if (meta.get('point')!=a.point or meta.get('engine_sha256')!=REFERENCE_ENGINE_SHA or
        meta.get('final_sha256')!=token['source_sha256']['final'] or not meta.get('source') or
        sha(a.reference_logits)!=meta.get('logits_sha256')):
        raise ValueError('reference model/source/logits mismatch')
    rr=meta.get('rows',[])
    if not rr or len(rr)>512 or (a.mode!='input0' and len(rr)>len(rows)) or any(r.get('sample_id')!=x['sample_id'] for r,x in zip(rr,rows)):
        raise ValueError('reference sample IDs differ')
    if any(not isinstance(r.get('input_sha256'),str) or len(r['input_sha256'])!=64 for r in rr):
        raise ValueError('reference input hashes invalid')
    with np.load(a.reference_logits,allow_pickle=False) as values:
        logits=values['logits']
        if logits.shape!=(len(rr),1000) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
            raise ValueError('reference logits invalid')
    return {'reference_sha256':a.reference_sha256,'reference_logits_sha256':meta['logits_sha256'],
            'reference_engine_sha256':REFERENCE_ENGINE_SHA,'reference_scope':'cross_engine_and_cross_backend','reference_count':min(len(rr),len(rows)),'reference_available_count':len(rr),'reference_source':meta['source']}

def check_reference_input(row,x):
    if digest(x.tobytes())!=row['input_sha256']:
        raise ValueError('reference input hash differs')

class RawRecorder:
    def __init__(self,out,rows,infer,reference=None,logits=None):
        self.out=Path(out);self.rows=rows;self.infer=infer;self.index=0;self.record=None
        self.reference=reference or [];self.logits=logits
    def __enter__(self):
        self.record=(self.out/'rows.jsonl').open('x');return self
    def __exit__(self,*args):self.record.close()
    def __call__(self,x):
        index=self.index
        if index>=len(self.rows):raise ValueError('Extra diagnostic request')
        inp=self.out/('input_%05d.npy'%index);output=self.out/('output_%05d.npy'%index)
        np.save(inp,x,allow_pickle=False)  # Preserve real input before SDK execution.
        if index<len(self.reference):check_reference_input(self.reference[index],x)
        logits=self.infer(x)
        if logits.dtype!=np.float32 or logits.shape!=(1,1000) or not np.isfinite(logits).all():raise ValueError('Invalid output')
        np.save(output,logits,allow_pickle=False)
        covered=index<len(self.reference)
        close=bool(np.allclose(logits.reshape(-1),self.logits[index],atol=.01,rtol=.01)) if covered else None
        top1=bool(np.argmax(logits)==np.argmax(self.logits[index])) if covered else None
        failed=int(np.count_nonzero(np.abs(logits.reshape(-1)-self.logits[index]) > .01 + .01*np.abs(self.logits[index]))) if covered else None
        self.record.write(json.dumps({'index':index,'sample_id':self.rows[index]['sample_id'],
            'input_data_sha256':digest(x.tobytes()),'output_data_sha256':digest(logits.tobytes()),
            'input_file_sha256':sha(inp),'output_file_sha256':sha(output),
            'reference_max_abs':float(np.max(np.abs(logits.reshape(-1)-self.logits[index]))) if covered else None,
            'reference_allclose':close,'reference_top1_match':top1,'reference_failed_elements':failed})+'\n')
        self.record.flush();self.index+=1
        return logits

def input0_reader(a,rows):
    if sha(a.input0)!=INPUT0_SHA:raise ValueError('Fixed real input0 SHA mismatch')
    raw=Path(a.input0).read_bytes()
    if len(raw)!=3*224*224*4:raise ValueError('input0 byte size')
    x=np.frombuffer(raw,dtype='<f4').reshape(1,3,224,224)
    h,p=encode_chunk(0,rows,x);reader=io.BytesIO();send(reader,h,p);send(reader,{'type':'end','count':1});reader.seek(0)
    return reader

class TeeWriter:
    def __init__(self,writer,raw):self.writer=writer;self.raw=raw
    def write(self,data):self.raw.write(data);return self.writer.write(data)
    def flush(self):self.raw.flush();self.writer.flush()

def target(a,reader,writer):
    assert_no_torch();token,rows=diagnostic_identity(a)
    if a.mode=='input0':reader=input0_reader(a,rows)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=None
        try:
            with bind(out/'init_stages.jsonl'):
                runtime=record('create_runtime',lambda:create_runtime(a))
                before=record('library_identity',library_identity)
            verify_libraries(before)
            sdk={'tensorrt_version':'8.5.2.2','torch_imported':False}
            token={**token,**sdk,'library_sha256':before,'pid':os.getpid(),
                   'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            (out/'identity.json').write_text(json.dumps(token,indent=2)+'\n')
            (out/'maps_before.txt').write_text(Path('/proc/self/maps').read_text())
            reference=json.loads(Path(a.reference).read_text())['rows']
            with np.load(a.reference_logits,allow_pickle=False) as refdata: ref_logits=refdata['logits'].copy()
            with RawRecorder(out,rows,runtime.infer,reference,ref_logits) as recorder,(out/'wire.bin').open('xb') as raw:
                serve(reader,TeeWriter(writer,raw),rows,recorder,token)
                count=recorder.index
            assert_no_torch();after=library_identity();verify_libraries(after)
            (out/'maps_after.txt').write_text(Path('/proc/self/maps').read_text())
            if before!=after:raise RuntimeError('SDK libraries changed')
            if count!=len(rows):raise ValueError('Diagnostic count mismatch')
        finally:
            if runtime is not None:runtime.close()
        (out/'diagnostic_receipt.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE',
            'scope':'DIAGNOSTIC_ONLY','count':count,'mode':a.mode,'identity':token,
            'rows_sha256':sha(out/'rows.jsonl'),'wire_sha256':sha(out/'wire.bin'),
            'full50000_qualified':False,'cleanup_completed':True},indent=2)+'\n')
def main():
    p=argparse.ArgumentParser()
    for name in ('engine','manifest','onnx','final','layers','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--point',required=True,choices=('geta',))
    for name in ('reference','reference-logits','reference-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',required=True,choices=('input0','ordered512'));p.add_argument('--input0')
    a=p.parse_args()
    if a.mode=='input0' and not a.input0:p.error('--input0 required')
    with protocol_output() as wire:target(a,sys.stdin.buffer,wire)
if __name__=='__main__':main()
