"""Frozen host loader and bounded prefix/full stream to CPU ORT target."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
from frozen_stream import evaluate
from cpu_identity import validate,sha
SOURCE={'scripts/resnet_frozen_validate.py':'ed615ac0d5e08bfbb4dae4d1ab20d62de97db798ac5a4164bd5e094c105ae9dd',
        'framework/resnet_compare/data.py':'e842f35d1d084c1cc8e453d0fe86aed5563c02e4fedf70927235343de5354035'}
class DatasetView:
    def __init__(self,base):self.base=base;self.rows=base.rows;self.manifest_identity=base.manifest_sha256
    def __len__(self):return len(self.base)
    def __getitem__(self,i):return self.base[i]

class Prefix:
    def __init__(self,base,count):self.base=base;self.rows=base.rows[:count]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.base[i]
class RawReader:
    def __init__(self,reader,raw):self.reader=reader;self.raw=raw
    def read(self,n=-1):
        data=self.reader.read(n);self.raw.write(data);self.raw.flush();return data

def settle(process):
    try:process.stdin.close()
    except (OSError,ValueError):pass
    for action,timeout in ((None,120),(process.terminate,30),(process.kill,30)):
        try:
            if action is not None:action()
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,ProcessLookupError):continue
    return None

def read_marker(path):
    markers=re.findall(r'^REMOTE_CPU_ORT_EXIT=(-?\d+)$',Path(path).read_text(errors='replace'),re.MULTILINE)
    return (int(markers[0]) if len(markers)==1 else None,len(markers))

def run_host(dataset,expected,command,output,count,source_hashes):
    if count not in (1,512,50000):raise ValueError('unsupported count')
    if len(dataset)!=50000 or len(dataset.rows)!=50000 or dataset.manifest_identity!=expected['manifest_identity']:raise ValueError('full dataset contract')
    if not isinstance(command,list) or not command or not all(isinstance(s,str) and s and '\x00' not in s for s in command):raise ValueError('command')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);scope='FULL50000' if count==50000 else 'DIAGNOSTIC_ONLY'
    token={**expected,'scope':scope,'count':count};process=None;error=None;result=None;exitcode=None
    with (out/'target.stderr.log').open('xb') as stderr,(out/'target.stdout.raw').open('xb') as raw:
        try:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,shell=False)
            result=evaluate(RawReader(process.stdout,raw),process.stdin,Prefix(dataset,count),out/'stream',token)
            if result['samples']!=count:raise ValueError('count mismatch')
        except BaseException as exc:error=exc
        finally:
            if process is not None:
                exitcode=settle(process);process.stdout.close()
    remote,markers=read_marker(out/'target.stderr.log')
    if error is None and (exitcode!=0 or remote!=0 or markers!=1):error=ValueError('remote/SSH exit mismatch')
    names=('target.stdout.raw','target.stderr.log','stream/logits.f32','stream/predictions.jsonl')
    receipt={'status':'FAILED' if error else 'COMPLETE','scope':scope,'count':count,'accuracy':result,
        'source_hashes':dict(source_hashes),'expected_identity':token,'command':command,'ssh_exit':exitcode,'remote_exit':remote,
        'remote_marker_count':markers,'cleanup_required':remote is None,'error':repr(error) if error else None,
        'artifact_sha256':{n:sha(out/n) for n in names if (out/n).is_file()}}
    (out/'result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    if error:raise error
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('dense','copt','geta'),required=True)
    for name in ('onnx','manifest','final','source','data','command-json','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,choices=(1,512,50000),required=True);a=p.parse_args()
    token=validate(a.point,a);actual={n:sha(Path(a.source)/n) for n in SOURCE}
    if actual!=SOURCE:raise ValueError('frozen loader source mismatch')
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    sys.path.insert(0,a.source)
    from scripts.resnet_frozen_validate import load_final_dataset
    base=load_final_dataset(Path(a.data),Path(a.final).parent)
    if base.manifest_sha256!=token['manifest_identity']:raise ValueError('loader final identity')
    run_host(DatasetView(base),token,json.loads(a.command_json),a.output,a.count,actual)
if __name__=='__main__':main()
