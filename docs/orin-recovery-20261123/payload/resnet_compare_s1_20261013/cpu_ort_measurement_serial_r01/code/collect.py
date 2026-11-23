"""Read-only tar evidence capture; never starts or retries a model."""
import sys,tarfile
from pathlib import Path
ROOT=Path('/home/jichengzhi/resnet_compare_s1_20261013')

def validate_root(root):
    path=Path(root)
    if path.is_symlink() or ROOT.resolve() not in path.resolve().parents:raise ValueError('outside approved root')
    return path

def collect(root,stream):
    root=Path(root);paths=[]
    for name in ('supervision','target_result'):
        base=root/name
        if base.exists():paths.extend([base,*sorted(base.rglob('*'))])
    if not paths:raise ValueError('missing evidence')
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):raise ValueError('unsafe evidence member')
    with tarfile.open(fileobj=stream,mode='w|gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(root)),recursive=False)

def main():
    if len(sys.argv)!=2:raise ValueError('collect ROOT required')
    collect(validate_root(sys.argv[1]),sys.stdout.buffer)
if __name__=='__main__':main()
