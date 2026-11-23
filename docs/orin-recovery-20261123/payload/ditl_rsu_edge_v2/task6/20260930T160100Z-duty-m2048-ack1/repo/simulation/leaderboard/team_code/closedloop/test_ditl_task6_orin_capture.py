"""CPU-only contracts for isolated Orin capture."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).parent))

def load(name):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / 'scripts' / (name + '.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.c = load('ditl_task6_orin_capture')
        self.g = load('ditl_task6_gpu_load')

    def plan(self):
        return self.c.build_capture_plan('unit', 2048, 'cal-1')

    def summaries(self):
        return [dict(profile=p['profile'], measured=100, remote_input_tensor_bytes=0,
                     source_sha256='a'*64, model_id='model', engine_sha256='b'*64,
                     input_sha256='c'*64, output_sha256='d'*64,
                     calibration_id='cal-1', inference_p95_ms=(i+1)*50)
                for i,p in enumerate(self.plan())]

    def test_exact_profiles_and_commands(self):
        plan = self.plan()
        self.assertEqual([p['duty_cycle'] for p in plan], [0, .25, .5, .85])
        self.assertEqual([p['profile'] for p in plan], ['idle','low','medium','high'])
        for p in plan:
            self.assertEqual((p['period_ms'],p['matrix_size']), (100,2048))
            commands = self.c.build_commands(p)
            native = commands['native']
            for flag, value in [('--count','120'),('--warmup-count','20'),('--compression','none'),('--result-host','127.0.0.1'),('--result-port','55021')]:
                self.assertEqual(native[native.index(flag)+1],value)
            self.assertIn('ROS_LOCALHOST_ONLY=1', commands['sink'])
            for role in ('sink', 'publisher'):
                self.assertIn('--ack-topic /ditl/rsu/perception_result_ack', commands[role][-1])
            self.assertIn('--ack-timeout-s 30', commands['publisher'][-1])
            self.assertEqual(commands['load'] is None, p['profile']=='idle')

    def test_invalid_plans_paths_and_shell_tokens(self):
        for cohort in ['', '..', 'a;b', 'a b', '/tmp/x', 'x$(id)']:
            with self.assertRaises(ValueError): self.c.build_capture_plan(cohort,2048,'cal')
        for size in [0, True, 1, 1025, 9000]:
            with self.assertRaises(ValueError): self.c.build_capture_plan('unit',size,'cal')
        for key,value in [('engine','/tmp/x'),('data_root','/home/jichengzhi/ditl_rsu_edge_v2/../x'),('manifest','/home/jichengzhi/ditl_rsu_edge_v2/x;bad')]:
            with self.assertRaises(ValueError): self.c.build_commands(dict(self.plan()[0],**{key:value}))

    def test_acceptance_rejects_every_invalid_axis(self):
        good = self.summaries()
        self.assertEqual(self.c.validate_capture_set(good), tuple(good))
        for key,value in [('measured',99),('remote_input_tensor_bytes',1),('source_sha256','e'*64),('model_id','other'),('engine_sha256','e'*64),('input_sha256','e'*64),('output_sha256','e'*64),('calibration_id','other'),('inference_p95_ms',50),('inference_p95_ms',float('nan')),('profile','idle')]:
            bad=copy.deepcopy(good); bad[1][key]=value
            with self.assertRaises(ValueError): self.c.validate_capture_set(bad)
        with self.assertRaises(ValueError): self.c.validate_capture_set(good[:-1])

    def test_worker_boundary_validation(self):
        self.g.validate_options(.25,100,2048,30)
        for values in [(float('nan'),100,2048,30),(1.1,100,2048,30),(-.1,100,2048,30),(.5,0,2048,30),(.5,100,1,30),(.5,100,2048,0),(.5,100,2048,float('inf'))]:
            with self.assertRaises(ValueError): self.g.validate_options(*values)

    def test_worker_records_metrics_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready=Path(tmp)/'ready'; metrics=Path(tmp)/'metrics'
            gpu=mock.Mock(); gpu.cuda.is_available.return_value=True
            now=[0.0]
            def clock(): now[0]+=.01; return now[0]
            self.g.run_worker(.25,100,256,.3,ready,metrics,torch_module=gpu,clock=clock,sleep=lambda n:None)
            self.assertTrue(ready.is_file())
            self.assertTrue(metrics.read_text())
            self.assertTrue(gpu.mm.called)
            with self.assertRaises(FileExistsError): self.g.run_worker(.25,100,256,.3,ready,metrics,torch_module=gpu)


    def fixture(self, tmp):
        import numpy as np
        base=Path(tmp)
        (base/'repo').mkdir(); (base/'inputs').mkdir()
        (base/'source_sha256sums.txt').write_text('committed code hashes')
        (base/'SOURCE_COMMIT').write_text('a'*40)
        (base/'inputs/model.engine').write_bytes(b'engine')
        np.save(str(base/'inputs/input.npy'),np.ones((2,)))
        (base/'manifest.json').write_text('{}')
        directory=base/'idle'; directory.mkdir()
        model=dict(model_id='fixed-model',engine_sha256=self.c.sha_file(base/'inputs/model.engine'))
        records=[dict(model=model,network=dict(remote_input_tensor_bytes=0),warmup=i<20) for i in range(120)]
        (directory/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        summary=dict(samples=dict(measured=100,warmup=20),stages=dict(gpu_inference=dict(p95_ms=50)))
        (directory/'summary.json').write_text(json.dumps(summary))
        (directory/'device_snapshot.txt').write_text('MODE_30W')
        for name in ['output.npz','native_last_output.npz']:
            np.savez(str(directory/name),pred=np.ones((2,)))
        return dict(self.plan()[0],directory=str(directory),repo=str(base/'repo'),
                    data_root=str(base/'inputs'),engine=str(base/'inputs/model.engine'),manifest=str(base/'manifest.json'))

    def test_real_capture_summary_and_preserved_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.fixture(tmp)
            result=self.c.summarize_capture(p)
            self.assertEqual(result['measured'],100)
            self.assertEqual(result['inference_p95_ms'],50)
            self.assertEqual(len(result['output_sha256']),64)
            self.assertEqual(set(x.name for x in (Path(p['directory'])/'source').iterdir()),
                {'trace.jsonl','command_manifest.txt','device_snapshot.txt','model_sha256.txt','source_sha256sums.txt'})
            provenance=json.loads((Path(p['directory'])/'source'/'command_manifest.txt').read_text())
            self.assertEqual(provenance['result_ack_topic'], '/ditl/rsu/perception_result_ack')
            self.assertEqual(provenance['result_ack_timeout_s'], 30)
            with self.assertRaises(FileExistsError): self.c.summarize_capture(p)

    def test_summary_rejects_short_remote_or_corrupt_outputs(self):
        import numpy as np
        for issue in ['short','remote','output','model']:
            with tempfile.TemporaryDirectory() as tmp:
                p=self.fixture(tmp); d=Path(p['directory'])
                if issue=='output': np.savez(str(d/'output.npz'),pred=np.zeros((2,)))
                else:
                    rows=[json.loads(x) for x in (d/'trace.jsonl').read_text().splitlines()]
                    if issue=='short': rows=rows[:-1]
                    if issue=='remote': rows[0]['network']['remote_input_tensor_bytes']=1
                    if issue=='model': rows[1]['model']['model_id']='other'
                    (d/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError): self.c.summarize_capture(p)

    def test_readiness_and_plan_cli_are_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'ready'; f.write_text('{}')
            self.c.wait_ready(f)
            with self.assertRaises(RuntimeError): self.c.wait_ready(Path(tmp)/'missing',timeout=0)
            self.c.write_new(Path(tmp)/'new',{'ok':True})
            with self.assertRaises(FileExistsError): self.c.write_new(Path(tmp)/'new',{})
        with mock.patch.object(self.c.subprocess,'Popen',side_effect=AssertionError('GPU in plan')), mock.patch('sys.stdout'):
            self.assertEqual(self.c.main(['--cohort','unit','--matrix-size','2048','--calibration-id','cal']),0)

    def test_capture_failure_cleans_only_owned_ids(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            plan=self.c.build_capture_plan('unit',2048,'cal')[1]
            self.c.ensure_cohort(plan)
            actions=[]
            cid='a'*64
            def fake_run(cmd,**kwargs):
                actions.append(cmd)
                if cmd[:3]==['docker','run','-d']: return cid
                return ''
            process=mock.Mock(); process.poll.return_value=None
            with mock.patch.object(self.c,'run',side_effect=fake_run), mock.patch.object(self.c.subprocess,'Popen',return_value=process), mock.patch.object(self.c,'wait_ready',side_effect=RuntimeError('not ready')):
                with self.assertRaisesRegex(RuntimeError,'not ready'): self.c.capture_profile(plan)
            self.assertEqual(actions[-1],['docker','rm','-f',cid])
            self.assertTrue((Path(plan['directory'])/'plan.json').exists())
            process.terminate.assert_called_once()
            with self.assertRaises(FileExistsError): self.c.capture_profile(plan)

    def test_worker_cli_and_missing_cuda(self):
        with mock.patch.object(self.g,'run_worker') as worker:
            self.assertEqual(self.g.main(['--duty-cycle','.25','--period-ms','100','--matrix-size','2048','--duration-s','30','--ready-file','r','--metrics-jsonl','m']),0)
        with tempfile.TemporaryDirectory() as tmp:
            gpu=mock.Mock(); gpu.cuda.is_available.return_value=False
            with self.assertRaisesRegex(RuntimeError,'CUDA'):
                self.g.run_worker(.5,100,256,1,Path(tmp)/'ready',Path(tmp)/'metrics',torch_module=gpu)


    def test_successful_runtime_keeps_diagnostics_and_load_identity(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            fixture_root=Path(tmp)/'fixture'; fixture_root.mkdir()
            fixture=self.fixture(str(fixture_root))
            plan=self.c.build_capture_plan('unit',2048,'cal')[1]
            base=Path(plan['directory']).parent
            self.c.ensure_cohort(plan)
            for name in ['repo','inputs']: shutil.copytree(str(fixture_root/name),str(base/name))
            for name in ['manifest.json','SOURCE_COMMIT','source_sha256sums.txt']: shutil.copyfile(str(fixture_root/name),str(base/name))
            commands=[]
            def fake_run(cmd,**kwargs):
                commands.append(cmd)
                if cmd[:3]==['docker','run','-d']: return ('a' if 'result-sink' in cmd[-1] else 'b')*64
                if cmd[:2]==['docker','wait']: return '0'
                return 'MODE_30W'
            def native(cmd,**kwargs):
                for name in ['trace.jsonl','summary.json','output.npz','native_last_output.npz']:
                    shutil.copyfile(str(Path(fixture['directory'])/name),str(Path(plan['directory'])/name))
            process=mock.Mock(); process.poll.return_value=None
            with mock.patch.object(self.c,'run',side_effect=fake_run), mock.patch.object(self.c.subprocess,'Popen',return_value=process), mock.patch.object(self.c.subprocess,'run',side_effect=native), mock.patch.object(self.c,'wait_ready'):
                result=self.c.capture_profile(plan)
            self.assertEqual(result['profile'],'low')
            self.assertEqual(result['measured'],100)
            self.assertEqual([cmd for cmd in commands if cmd[:3]==['docker','rm','-f']],
                             [['docker','rm','-f','a'*64],['docker','rm','-f','b'*64]])

    def test_explicit_calibration_pins_matrix_once(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            p=self.c.build_capture_plan('unit',2048,'cal')[0]
            with mock.patch.object(self.c,'run',return_value=''):
                d=self.c.calibrate(p)
            data=json.loads((d/'calibration.json').read_text())
            self.assertEqual(data['matrix_size'],2048)
            self.assertEqual(data['command'][data['command'].index('--duration-s')+1],'10')
            with self.assertRaises(FileExistsError): self.c.calibrate(p)

    def test_register_rejects_nonincreasing_before_any_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=self.summaries(); rows[2]['inference_p95_ms']=50
            (root/'orin_capture_candidate.json').write_text(json.dumps(dict(profiles=rows)))
            with self.assertRaises(ValueError): self.c.register_local_capture(root,root/'library')


    def test_four_real_bundles_register_with_unambiguous_mapping(self):
        with tempfile.TemporaryDirectory(prefix='task6-orin-test-',dir='/var/tmp/codriving-ditl') as tmp:
            root=Path(tmp); library=root/'library'; (library/'v1/bundles').mkdir(parents=True)
            rows=self.summaries()
            fixture=ROOT/'simulation/leaderboard/team_code/closedloop/fixtures/ditl_library/source'
            record=json.loads((fixture/'trace.jsonl').read_text().splitlines()[0])
            for row in rows:
                row['model_id']=record['model']['model_id']; row['engine_sha256']=record['model']['engine_sha256']
                row['capture_time_utc']='2026-09-24T10:00:00Z'
                source=root/row['profile']/'source'; source.mkdir(parents=True)
                records=[]
                for i in range(120):
                    r=copy.deepcopy(record); r['warmup']=i<20
                    r['trace_id']='test:'+str(i); r['message']['sequence_id']=i; r['message']['frame_id']=i
                    r['timestamps_ns']={k:v+i*1000000000 for k,v in r['timestamps_ns'].items()}
                    records.append(r)
                (source/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                (source/'command_manifest.txt').write_text(json.dumps(dict(row,source_commit='a'*40)))
                for name in ['device_snapshot.txt','model_sha256.txt','source_sha256sums.txt']:
                    (source/name).write_text('fixture')
            (root/'orin_capture_candidate.json').write_text(json.dumps(dict(profiles=rows)))
            import ditl_trace_library as bundle_library
            unrelated={'e'*64:dict(checksums_sha256='f'*64,profile_id='unrelated',relative_path='v1/bundles/'+'e'*64)}
            old=json.dumps(dict(library_schema_version='ditl.feature-library.v1',features=unrelated)).encode()
            (library/'catalog.json').write_bytes(old)
            replace=bundle_library._replace_catalog
            visible=[]
            def fail_after_publish(fd,path,before,after):
                fsync=bundle_library.os.fsync
                def sync(handle):
                    current=(library/'catalog.json').read_bytes()
                    if current!=old and not visible:
                        visible.append(len(json.loads(current)['features']))
                        raise OSError('failure after catalog replacement')
                    return fsync(handle)
                with mock.patch.object(bundle_library.os,'fsync',side_effect=sync):
                    return replace(fd,path,before,after)
            with mock.patch.object(self.c,'LIBRARY_ROOT',library), mock.patch.object(bundle_library,'_replace_catalog',side_effect=fail_after_publish):
                with self.assertRaises(OSError): self.c.register_local_capture(root,library)
            self.assertEqual(visible,[5],'all four additions must become visible together')
            self.assertEqual((library/'catalog.json').read_bytes(),old)
            self.assertFalse((root/'orin_capture_report.json').exists())
            publications=[]
            def publish(fd,path,before,after):
                publications.append(json.loads(after)['features'])
                return replace(fd,path,before,after)
            with mock.patch.object(self.c,'LIBRARY_ROOT',library), mock.patch.object(bundle_library,'_replace_catalog',side_effect=publish):
                self.assertEqual(self.c.main(['--mode','finalize','--mirror-root',str(root)]),0)
                report=json.loads((root/'orin_capture_report.json').read_text())
            self.assertEqual(len(publications),1,'four profiles must publish in one catalog transaction')
            self.assertEqual(set(publications[0]),set(unrelated)|{v['feature_id'] for v in report['profile_bundles'].values()})
            self.assertEqual(report['status'],'verified')
            self.assertEqual(len(report['profile_bundles']),4)
            self.assertEqual(len(json.loads((library/'catalog.json').read_text())['features']),5)
            for profile,bundle in report['profile_bundles'].items():
                manifest=json.loads((Path(bundle['bundle_path'])/'manifest.json').read_text())
                self.assertEqual(manifest['load_label'],profile)
                self.assertEqual(manifest['accepted_count'],100)

    def test_cli_requires_matching_calibration(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            p=self.c.build_capture_plan('unit',2048,'cal')[0]
            self.c.ensure_cohort(p)
            d=Path(p['directory']).parent/'calibration-cal'; d.mkdir()
            (d/'calibration.json').write_text(json.dumps(dict(matrix_size=4096)))
            with self.assertRaisesRegex(ValueError,'matrix'):
                self.c.main(['--cohort','unit','--matrix-size','2048','--calibration-id','cal','--mode','capture'])


    def test_input_identity_binds_manifest_selection(self):
        hashes=[]
        for manifest in ['{"frame_index":0}','{"frame_index":1}']:
            with tempfile.TemporaryDirectory() as tmp:
                p=self.fixture(tmp)
                Path(p['manifest']).write_text(manifest)
                hashes.append(self.c.summarize_capture(p)['input_sha256'])
        self.assertNotEqual(hashes[0],hashes[1])

    def test_prepare_cli_creates_clean_tree_exclusively_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            argv=['--cohort','unit','--matrix-size','2048','--calibration-id','cal','--mode','prepare']
            self.assertEqual(self.c.main(argv),0)
            marker=Path(tmp)/'task6/unit/owner.json'
            self.assertEqual(json.loads(marker.read_text())['matrix_size'],2048)
            with self.assertRaises(FileExistsError): self.c.main(argv)
            plan=self.c.build_capture_plan('unit',2048,'cal')[0]
            self.assertEqual(self.c.ensure_cohort(plan),marker.parent)
            with self.assertRaises(ValueError): self.c.ensure_cohort(dict(plan,calibration_id='other'))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            outside=Path(tmp)/'outside'; outside.mkdir()
            (Path(tmp)/'task6').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError): self.c.main(argv)
            self.assertEqual(list(outside.iterdir()),[])

    def test_signal_unwinds_cleanup_without_signaling_foreign_processes(self):
        import signal
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.c,'ROOT',Path(tmp)):
            plan=self.c.build_capture_plan('unit',2048,'cal')[1]
            self.c.ensure_cohort(plan)
            handlers={}
            def install(sig,handler):
                previous=handlers.get(sig,signal.SIG_DFL); handlers[sig]=handler; return previous
            commands=[]
            def command(cmd):
                commands.append(cmd)
                if cmd[:3]==['docker','run','-d']: return 'a'*64
                return ''
            process=mock.Mock(); process.poll.return_value=None
            def interrupted(*args,**kwargs): handlers[signal.SIGTERM](signal.SIGTERM,None)
            with mock.patch.object(self.c.signal,'signal',side_effect=install), mock.patch.object(self.c,'run',side_effect=command), mock.patch.object(self.c.subprocess,'Popen',return_value=process), mock.patch.object(self.c.subprocess,'run',side_effect=interrupted), mock.patch.object(self.c,'wait_ready'):
                with self.assertRaisesRegex(RuntimeError,'signal'): self.c.capture_profile(plan)
            self.assertEqual(process.terminate.call_count,2)
            self.assertEqual([c for c in commands if c[:3]==['docker','rm','-f']],
                             [['docker','rm','-f','a'*64],['docker','rm','-f','a'*64]])
            self.assertEqual(handlers[signal.SIGTERM],signal.SIG_DFL)

    def test_finalize_cli_invokes_full_registration(self):
        with mock.patch.object(self.c,'register_local_capture',return_value={'status':'verified'}) as finalize:
            self.assertEqual(self.c.main(['--mode','finalize','--mirror-root','/var/tmp/codriving-ditl/task6-orin-unit']),0)
            self.assertEqual(finalize.call_args[0][0],Path('/var/tmp/codriving-ditl/task6-orin-unit'))

if __name__ == '__main__': unittest.main()
