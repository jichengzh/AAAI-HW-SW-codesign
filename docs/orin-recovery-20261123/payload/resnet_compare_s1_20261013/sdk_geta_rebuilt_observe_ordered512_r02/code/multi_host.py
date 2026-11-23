"""Host-only frozen preprocessing/topk adapter; never imported by SDK target."""
import argparse,json,sys
from pathlib import Path
from host_entry import run_host
from identity import validate
from sdk_backend import sha
SOURCE={'scripts/resnet_frozen_validate.py':'ed615ac0d5e08bfbb4dae4d1ab20d62de97db798ac5a4164bd5e094c105ae9dd',
        'framework/resnet_compare/data.py':'e842f35d1d084c1cc8e453d0fe86aed5563c02e4fedf70927235343de5354035'}
class DatasetView:
    def __init__(self,base,identity):self.base=base;self.rows=base.rows;self.manifest_identity=identity
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return self.base[index]
def verify_source(root):
    actual={name:sha(Path(root)/name) for name in SOURCE}
    if actual!=SOURCE:raise ValueError('Frozen loader source changed')
    return actual
def load_dataset(a):
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    sys.path.insert(0,a.source)
    from scripts.resnet_frozen_validate import load_final_dataset
    return load_final_dataset(Path(a.data),Path(a.final).parent)
def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--point', choices=('geta',), required=True)
    for name in ('engine','manifest','onnx','final','layers','source','data','command-json','output'):
        p.add_argument('--'+name,required=True)
    return p.parse_args()
def main():
    a=arguments();token=validate(a.point,a);source=verify_source(a.source);base=load_dataset(a)
    if base.manifest_sha256!=token['manifest_identity']:raise ValueError('Loader manifest changed')
    dataset=DatasetView(base,token['manifest_identity'])
    return run_host(dataset,token,json.loads(a.command_json),a.output,source)
if __name__=='__main__':main()
