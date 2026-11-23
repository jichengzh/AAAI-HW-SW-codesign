"""Single bounded batch exchange; ACK removes transient data only after host commit."""
import json,shutil,time,subprocess,shlex
from pathlib import Path
import protocol as p


def step(worker):
    root=worker.root;(root/'incoming').mkdir(exist_ok=True);(root/'acks').mkdir(exist_ok=True)
    (root/'acks_done').mkdir(exist_ok=True);(root/'cleanup_started').mkdir(exist_ok=True)
    for ack in sorted((root/'acks').glob('batch_*.json')):
        folder=root/'outgoing'/ack.stem;checkpoint=root/'checkpoints'/ack.name
        p.require(checkpoint.exists(),'acknowledgement without committed checkpoint')
        saved=json.loads(checkpoint.read_text());value=json.loads(ack.read_text())
        p.require(value['result_sha256']==saved['result_manifest_sha256'],'acknowledgement of different/uncommitted result')
        if folder.exists():
            marker=root/'cleanup_started'/ack.name
            identity=dict(result_sha256=value['result_sha256'],checkpoint_sha256=p.sha(checkpoint))
            if marker.exists():p.require(json.loads(marker.read_text())==identity,'ACK cleanup identity changed')
            else:
                p.require(p.sha(folder/'done.json')==value['result_sha256'],'ack result manifest changed')
                p.write_json(marker,identity)
        batch=saved['input_batch'];ready=root/'incoming'/(ack.stem+'.ready.json')
        if ready.exists():p.require(p.canonical(json.loads(ready.read_text()))==p.canonical(batch),'ack input identity differs')
        for row in batch['frames']:
            path=root/'incoming'/row['file']
            if path.exists():p.require(p.sha(path)==row['sha256'],'ack input file changed');path.unlink()
        ready.unlink(missing_ok=True)
        if folder.exists():shutil.rmtree(folder)
        destination=root/'acks_done'/ack.name
        if destination.exists():p.require(p.sha(destination)==p.sha(ack),'repeated acknowledgement differs');ack.unlink()
        else:ack.rename(destination)
    ready=root/'incoming'/f'batch_{worker.last_batch+1:05d}.ready.json'
    if ready.exists():worker.process_batch(json.loads(ready.read_text()),root/'incoming')
    status=dict(next_index=worker.next_index,total=len(worker.manifest['frame_ids']),last_batch=worker.last_batch,time=time.time())
    p.write_json(root/'heartbeat.json',status,overwrite=True)
    if worker.next_index==len(worker.manifest['frame_ids']) and not list((root/'outgoing').glob('batch_*')):
        path=root/'completed.json'
        if not path.exists():p.write_json(path,{**status,'manifest_sha256':p.canonical(worker.manifest)})


class LocalTransport:
    def __init__(self,root):self.root=Path(root)
    def put(self,source,relative):
        target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():p.require(p.sha(source)==p.sha(target),'immutable transmitted file differs');return
        temporary=target.with_name('.'+target.name+'.part');shutil.copyfile(source,temporary);temporary.replace(target)
    def send(self,batch,input_root,batch_path):
        for row in batch['frames']:self.put(Path(input_root)/row['file'],'incoming/'+row['file'])
        self.put(batch_path,f"incoming/batch_{batch['batch_index']:05d}.ready.json")
    def receive(self,index,local_root):
        source=self.root/'outgoing'/f'batch_{index:05d}';p.require((source/'done.json').exists(),'batch result not ready')
        target=Path(local_root)/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copytree(source,target)
        return json.loads((target/'done.json').read_text()),target
    def ack(self,index,result_sha):
        p.write_json(self.root/'acks'/f'batch_{index:05d}.json',dict(result_sha256=result_sha))


class SSHTransport:
    def __init__(self,remote_root,host='jichengzhi@172.16.62.222'):
        p.require(str(remote_root).startswith('/home/jichengzhi/resnet_compare_s1_20261013/'),'outside approved target experiment root')
        self.root=str(remote_root);self.host=host;self.options=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4']
    def command(self,args):
        return subprocess.run(['ssh',*self.options,self.host,shlex.join(args)],check=True,capture_output=True,text=True)
    def put(self,source,relative):
        p.require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'unsafe remote relative path')
        target=self.root+'/'+relative
        existing=self.command(['sh','-c','if test -f '+shlex.quote(target)+'; then sha256sum '+shlex.quote(target)+'; fi'])
        if existing.stdout.strip():
            p.require(existing.stdout.split()[0]==p.sha(source),'remote immutable file differs; original retained')
            return
        self.command(['mkdir','-p',str(Path(target).parent)])
        temporary=target+'.part'
        subprocess.run(['scp',*self.options,str(source),self.host+':'+shlex.quote(temporary)],check=True,capture_output=True)
        # Final path is immutable. sha256sum is evaluated remotely before atomic rename.
        script='set -eu; actual=$(sha256sum '+shlex.quote(temporary)+'); test "${actual%% *}" = '+shlex.quote(p.sha(source))+'; if test -f '+shlex.quote(target)+'; then cmp '+shlex.quote(target)+' '+shlex.quote(temporary)+'; rm '+shlex.quote(temporary)+'; else mv '+shlex.quote(temporary)+' '+shlex.quote(target)+'; fi'
        self.command(['sh','-c',script])
    def send(self,batch,input_root,batch_path):
        for row in batch['frames']:self.put(Path(input_root)/row['file'],'incoming/'+row['file'])
        self.put(batch_path,f"incoming/batch_{batch['batch_index']:05d}.ready.json")
    def receive(self,index,local_root,timeout=1800):
        source=self.root+f'/outgoing/batch_{index:05d}';deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            result=self.command(['sh','-c','if test -f '+shlex.quote(self.root+'/failure.json')+'; then cat '+shlex.quote(self.root+'/failure.json')+'; exit 3; fi; if test -f '+shlex.quote(source+'/done.json')+'; then echo ready; fi'])
            if result.stdout.strip()=='ready':break
            time.sleep(2)
        else:raise TimeoutError('target batch result timed out; files preserved')
        target=Path(local_root)/f'batch_{index:05d}';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary=target.with_name('.'+target.name+'.part');temporary.mkdir(exist_ok=True)
            # SFTP scp rejects a directory basename of '.'. Copy its real basename.
            subprocess.run(['scp',*self.options,'-r',self.host+':'+shlex.quote(source),str(temporary)],check=True,capture_output=True)
            downloaded=temporary/target.name
            p.require((downloaded/'done.json').is_file(),'SFTP result directory contract differs')
            downloaded.rename(target);temporary.rmdir()
        return json.loads((target/'done.json').read_text()),target
    def ack(self,index,result_sha):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ack.json';p.write_json(path,dict(result_sha256=result_sha))
            self.put(path,f'acks/batch_{index:05d}.json')
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            result=self.command(['sh','-c','if test -d '+shlex.quote(self.root+f'/outgoing/batch_{index:05d}')+'; then echo pending; fi'])
            if not result.stdout.strip():return
            time.sleep(.2)
        raise TimeoutError('target ACK cleanup timed out; stop before next input batch')
