"""Separate bounded diagnostics; cannot produce full50000 qualification."""
import argparse,fcntl,io,json,os,sys
from pathlib import Path
import numpy as np
from sdk_backend import assert_no_torch,library_identity,sha
from sdk_contract import validate
from sdk_entry import create_runtime,protocol_output,LOCK
from frozen_stream import serve,send,receive,encode_chunk,digest
INPUT0_SHA='ea4a6f68f067977c0aca4d1b1706fadc63b0dca6c4499cac149b9c9e3d715884'

def diagnostic_identity(a):
    token=validate(a)  # Always the original, complete, frozen50000 source contract.
    if a.mode not in ('input0','ordered512'):raise ValueError('Unsupported diagnostic mode')
    count=1 if a.mode=='input0' else 512
    rows=json.loads(Path(a.final).read_text())['samples'][:count]
    if len(rows)!=count:raise ValueError('Diagnostic prefix incomplete')
    return {**token,'scope':'DIAGNOSTIC_ONLY','diagnostic_count':count,'diagnostic_mode':a.mode,
            'diagnostic_entry_sha256':sha(__file__)},rows

class RawRecorder:
    def __init__(self,out,rows,infer):
        self.out=Path(out);self.rows=rows;self.infer=infer;self.index=0;self.record=None
    def __enter__(self):
        self.record=(self.out/'rows.jsonl').open('x');return self
    def __exit__(self,*args):self.record.close()
    def __call__(self,x):
        index=self.index
        if index>=len(self.rows):raise ValueError('Extra diagnostic request')
        inp=self.out/('input_%05d.npy'%index);output=self.out/('output_%05d.npy'%index)
        np.save(inp,x,allow_pickle=False)  # Preserve real input before SDK execution.
        logits=self.infer(x)
        np.save(output,logits,allow_pickle=False)
        self.record.write(json.dumps({'index':index,'sample_id':self.rows[index]['sample_id'],
            'input_data_sha256':digest(x.tobytes()),'output_data_sha256':digest(logits.tobytes()),
            'input_file_sha256':sha(inp),'output_file_sha256':sha(output)})+'\n')
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
            runtime,sdk=create_runtime(a);before=library_identity()
            token={**token,**sdk,'library_sha256':before,'pid':os.getpid(),
                   'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            (out/'identity.json').write_text(json.dumps(token,indent=2)+'\n')
            (out/'maps_before.txt').write_text(Path('/proc/self/maps').read_text())
            with RawRecorder(out,rows,runtime.infer) as recorder,(out/'wire.bin').open('xb') as raw:
                serve(reader,TeeWriter(writer,raw),rows,recorder,token)
                count=recorder.index
            assert_no_torch();after=library_identity()
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
    p.add_argument('--mode',required=True,choices=('input0','ordered512'));p.add_argument('--input0')
    a=p.parse_args()
    if a.mode=='input0' and not a.input0:p.error('--input0 required')
    with protocol_output() as wire:target(a,sys.stdin.buffer,wire)
if __name__=='__main__':main()
