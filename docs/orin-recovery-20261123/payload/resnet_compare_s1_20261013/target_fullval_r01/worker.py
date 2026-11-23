"""Four persistent CPU sessions, atomic bounded batches and independent EMA chains."""
import json,shutil,uuid
from pathlib import Path
import numpy as np
import protocol as p


class Worker:
    def __init__(self,root,manifest,graph_root):
        self.root=Path(root);self.manifest=manifest;self.graph_root=Path(graph_root)
        for name in ('outgoing','checkpoints','profiles'):(self.root/name).mkdir(parents=True,exist_ok=True)
        initial=p.load_pack(self.root/'initial_state.npz',manifest['initial_state'])
        self.states={m:[initial[f'{m}__ema_{i}'] for i in range(3)] for m in ('control','integer')}
        self.next_index=0;self.last_batch=-1;self.checkpoint_sha=None
        for path in sorted((self.root/'checkpoints').glob('batch_*.json')):
            value=json.loads(path.read_text())
            p.require(value['batch_index']==self.last_batch+1 and value['previous_checkpoint_sha256']==self.checkpoint_sha,'checkpoint sequence/chain differs')
            p.require(value['manifest_sha256']==p.canonical(manifest),'checkpoint model/data manifest differs')
            data=p.load_pack(path.with_suffix('.npz'),value['state'])
            self.states={m:[data[f'{m}__ema_{i}'] for i in range(3)] for m in self.states}
            self.next_index=value['next_index'];self.last_batch=value['batch_index'];self.checkpoint_sha=p.sha(path)
        self.sessions={point:self.session(point,entry) for point,entry in manifest['graphs'].items()}
        p.require(set(self.sessions)=={'control_a1','control_a2','integer_a1','integer_a2'},'four sessions required')
        self.profiled=set()

    def session(self,point,entry):
        import onnxruntime as ort
        graph=self.graph_root/entry['file'];p.require(p.sha(graph)==entry['sha256'],'frozen graph SHA differs')
        p.require(ort.__version__==self.manifest['target_options']['version'],'ORT version differs')
        p.require(np.__version__==self.manifest['target_options']['numpy_version'],'NumPy version differs')
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;options.log_severity_level=3
        options.enable_profiling=True;options.profile_file_prefix=str(self.root/'profiles'/(point+'_'+uuid.uuid4().hex))
        session=ort.InferenceSession(str(graph),options,providers=['CPUExecutionProvider']);session.disable_fallback()
        p.require(session.get_providers()==['CPUExecutionProvider'],'provider fallback')
        p.require(tuple(v.name for v in session.get_outputs())==p.OUTPUTS,'output order differs')
        return session

    def profile(self,point):
        if point in self.profiled:return
        path=self.sessions[point].end_profiling();events=json.loads(Path(path).read_text())
        kernels=[e['args'] for e in events if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time')]
        p.require(bool(kernels) and all(k.get('provider')=='CPUExecutionProvider' for k in kernels),'missing CPU execution evidence/fallback')
        count=sum(k.get('op_name')=='ConvInteger' for k in kernels)
        p.require(count==self.manifest['graphs'][point]['expected_integer_kernels'],'silent integer fallback')
        p.write_json(Path(path).with_name(Path(path).stem+'_receipt.json'),dict(graph_sha256=self.manifest['graphs'][point]['sha256'],
            convinteger=count,provider='CPUExecutionProvider',profile_file=path,profile_sha256=p.sha(path)))
        self.profiled.add(point)

    def checkpoint(self,result,folder):
        index=result['batch_index'];path=self.root/'checkpoints'/f'batch_{index:05d}.json'
        if path.exists():
            old=json.loads(path.read_text());p.require(old['result_manifest_sha256']==p.sha(folder/'done.json'),'committed result identity changed')
            return
        state_path=path.with_suffix('.npz')
        if not state_path.exists():
            temporary=state_path.with_suffix('.part');shutil.copyfile(folder/'state.npz',temporary);temporary.replace(state_path)
        data=p.load_pack(state_path,result['state'])
        value=dict(batch_index=index,next_index=result['next_index'],manifest_sha256=p.canonical(self.manifest),
            previous_checkpoint_sha256=self.checkpoint_sha,result_manifest_sha256=p.sha(folder/'done.json'),state=result['state'],input_batch=result['input_batch'])
        p.write_json(path,value)
        self.states={m:[data[f'{m}__ema_{i}'] for i in range(3)] for m in self.states}
        self.next_index=result['next_index'];self.last_batch=index;self.checkpoint_sha=p.sha(path)

    def process_batch(self,batch,input_root):
        index=batch['batch_index'];folder=self.root/'outgoing'/f'batch_{index:05d}'
        if folder.exists():
            result=json.loads((folder/'done.json').read_text())
            p.require(result['input_batch_sha256']==p.canonical(batch),'repeated batch payload differs')
            if index>self.last_batch:self.checkpoint(result,folder)
            return result
        p.require(index==self.last_batch+1,'batch missing/repeated/reordered')
        p.validate_batch(batch,self.manifest,self.next_index)
        staging=folder.with_name('.'+folder.name+'.tmp')
        if staging.exists():
            failed=self.root/'failed_attempts';failed.mkdir(exist_ok=True)
            audit=failed/(folder.name+'_'+uuid.uuid4().hex+'.json')
            p.write_json(audit,dict(reason='interrupted uncommitted outputs; all original inputs and previous checkpoint retained for exact replay',previous_checkpoint_sha256=self.checkpoint_sha,
                manifest_sha256=p.canonical(self.manifest),input_batch=batch,files={str(path.relative_to(staging)):dict(sha256=p.sha(path),bytes=path.stat().st_size) for path in staging.rglob('*') if path.is_file()}))
            # Only uncommitted, reproducible outputs are cleared after their audit is durable.
            shutil.rmtree(staging)
        staging.mkdir(exist_ok=False)
        local={m:list(values) for m,values in self.states.items()};rows=[]
        for source in batch['frames']:
            values=p.load_pack(Path(input_root)/source['file'],source);agents=source['agents']
            p.validate_inputs(values,agents)
            random=p.uniforms(17+source['index'],self.manifest['uniform_shape'][str(agents)])
            p.check_uniforms(random,source['uniform_sha256'])
            shared={**values,**{f'gumbel_uniform_{i}':array for i,array in enumerate(random)}}
            shared_specs={key:p.array_spec(value) for key,value in shared.items()}
            shared_sha=p.canonical(shared_specs)
            p.require(shared_sha==source['shared_input_sha256'],'complete shared input identity differs')
            for value in shared.values():value.flags.writeable=False
            combined={};methods={}
            for method in ('control','integer'):
                point=f'{method}_a{agents}';session=self.sessions[point];previous=local[method]
                feeds={**shared,**{f'initial_freqEMA_{i}':array for i,array in enumerate(previous)}}
                p.require(set(feeds)=={v.name for v in session.get_inputs()},'input contract differs')
                arrays=session.run(None,feeds);outputs=dict(zip(p.OUTPUTS,arrays))
                p.validate_outputs(outputs,previous,self.manifest['output_shapes']);self.profile(point)
                following=[outputs[f'final_freqEMA_{i}'].copy() for i in range(3)]
                methods[method]=dict(graph_sha256=self.manifest['graphs'][point]['sha256'],
                    initial_state_sha256=p.state_sha(previous),final_state_sha256=p.state_sha(following),
                    outputs={key:p.array_spec(value) for key,value in outputs.items()})
                local={**local,method:following}
                combined.update({f'{method}__{key}':value for key,value in outputs.items()})
            name=f"frame_{source['index']:05d}.npz";identity=p.write_pack(staging/name,combined)
            p.require(identity['bytes']<=p.OUTPUT_RESERVE,'output reserve bound exceeded')
            rows.append(dict(index=source['index'],sample_id=source['sample_id'],agents=agents,input_sha256=shared_sha,
                uniform_sha256=source['uniform_sha256'],source_pack_sha256=source['sha256'],file=name,
                output=identity,methods=methods,completed=True))
        state=p.write_pack(staging/'state.npz',{f'{m}__ema_{i}':value for m,values in local.items() for i,value in enumerate(values)})
        result=dict(batch_index=index,input_batch=batch,input_batch_sha256=p.canonical(batch),manifest_sha256=p.canonical(self.manifest),
            next_index=self.next_index+len(rows),frames=rows,state=state)
        p.require(sum(row['bytes'] for row in batch['frames'])+sum(row['output']['bytes'] for row in rows)+state['bytes']<p.MAX_BYTES,'actual cache bound exceeded')
        p.write_json(staging/'done.json',result);staging.rename(folder)
        self.checkpoint(result,folder)
        return result
