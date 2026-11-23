#!/usr/bin/env python3
"""Plan/capture isolated RSU-local GPU profiles; CPU plan mode has no side effects."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import signal
import socket
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/jichengzhi/ditl_rsu_edge_v2')
LIBRARY_ROOT = Path('/home/jichengzhi/V2X/multi_agent/v2x_rts/ditl_trace_library')
PYTHON = '/home/jichengzhi/miniforge3/envs/uniad/bin/python'
PROFILES = (('idle', 0.0), ('low', .25), ('medium', .50), ('high', .85))
IDENTITIES = ('source_sha256', 'model_id', 'engine_sha256', 'input_sha256', 'output_sha256', 'calibration_id')


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('invalid task6 identifier')
    return value


def local_path(value):
    path = Path(value)
    if '..' in path.parts or not re.fullmatch(r'[/A-Za-z0-9_.-]+', str(path)) or ROOT not in path.parents:
        raise ValueError('path must stay inside deployment root without shell characters')
    if path.resolve() != path:
        raise ValueError('symlink paths are forbidden')
    return str(path)


def build_capture_plan(cohort, matrix_size, calibration_id):
    token(cohort); token(calibration_id)
    if type(matrix_size) is not int or not 256 <= matrix_size <= 8192 or matrix_size % 256:
        raise ValueError('invalid calibrated matrix size')
    base = ROOT / 'task6' / cohort
    return tuple(dict(cohort=cohort, profile=profile, duty_cycle=duty, period_ms=100,
                      matrix_size=matrix_size, calibration_id=calibration_id,
                      directory=str(base/profile), repo=str(base/'repo'),
                      data_root=str(base/'inputs'), manifest=str(base/'manifest.json'),
                      engine=str(base/'inputs/model.engine')) for profile,duty in PROFILES)



def ensure_cohort(plan, exclusive=False):
    """Create only the approved tree, or verify a previously prepared owner marker."""
    expected=build_capture_plan(plan['cohort'],plan['matrix_size'],plan['calibration_id'])[0]
    cohort=Path(expected['directory']).parent
    local_path(cohort)
    cohort.parent.mkdir(exist_ok=True)
    owner={k:plan[k] for k in ('cohort','matrix_size','calibration_id')}
    if exclusive or not cohort.exists():
        cohort.mkdir()
        write_new(cohort/'owner.json',owner)
    elif (cohort/'owner.json').is_symlink() or json.loads((cohort/'owner.json').read_text())!=owner:
        raise ValueError('cohort ownership or calibration mismatch')
    return cohort


@contextmanager
def capture_signals():
    """Turn interruption into unwinding; ignore repeated signals during cleanup."""
    def interrupted(signum, frame):
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,signal.SIG_IGN)
        raise RuntimeError('capture interrupted by signal %s' % signum)
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        yield
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


def build_commands(plan):
    p = dict(plan)
    for key in ('directory','repo','data_root','manifest','engine'):
        p[key] = local_path(p[key])
    token(p['cohort']); token(p['profile'])
    expected = {x['profile']: x for x in build_capture_plan(p['cohort'],p['matrix_size'],p['calibration_id'])}
    if p['profile'] not in expected or any(p[k] != expected[p['profile']][k] for k in ('directory','repo','duty_cycle','period_ms')):
        raise ValueError('profile does not match frozen plan')
    d, repo = p['directory'], p['repo']
    def container(role, args):
        cmd = ['python3','/workspace/scripts/ditl_ros2_edge_bridge.py',role]+args
        return ['docker','run','-d','--name','ditl-task6-'+p['cohort']+'-'+p['profile']+'-'+role,
                '--network','host','--ipc','host','-e','ROS_DOMAIN_ID=91','-e','ROS_LOCALHOST_ONLY=1',
                '-v',repo+':/workspace:ro','-v',d+':/artifacts',
                'ditl/ros-foxy-tracing-enabled:arm64','bash','-lc',
                'source /opt/ros/foxy/setup.bash && exec '+' '.join(shlex.quote(s) for s in cmd)]
    return dict(
        sink=container('result-sink',['--max-results','120','--timeout-s','600',
            '--trace-jsonl','/artifacts/trace.jsonl','--output-npz','/artifacts/output.npz',
            '--ready-file','/artifacts/sink_ready.json']),
        publisher=container('result-publisher',['--bind','127.0.0.1','--result-port','55021',
            '--max-results','120','--discovery-timeout-s','60','--socket-timeout-s','600',
            '--ready-file','/artifacts/publisher_ready.json']),
        native=[PYTHON,repo+'/scripts/ditl_rsu_edge_publisher.py','--data-root',p['data_root'],
            '--manifest',p['manifest'],'--engine',p['engine'],'--model-id','pyramid_dair_m1_collab_n2_fp16_orin','--run-id',p['cohort']+'-'+p['profile'],
            '--count','120','--warmup-count','20','--compression','none','--result-host','127.0.0.1',
            '--result-port','55021','--output-npz',d+'/native_last_output.npz'],
        load=None if p['profile']=='idle' else [PYTHON,repo+'/scripts/ditl_task6_gpu_load.py',
            '--duty-cycle',str(p['duty_cycle']),'--period-ms','100','--matrix-size',str(p['matrix_size']),
            '--duration-s','600','--ready-file',d+'/load_ready.json','--metrics-jsonl',d+'/load_metrics.jsonl'],
        telemetry=['tegrastats','--interval','100'],
        summarize=[PYTHON,repo+'/scripts/ditl_summarize_stages.py','--trace-jsonl',d+'/trace.jsonl',
                   '--output-json',d+'/summary.json','--warmup-count','20'])


def validate_capture_set(summaries):
    rows = tuple(dict(row) for row in summaries)
    if tuple(r.get('profile') for r in rows) != tuple(p for p,d in PROFILES):
        raise ValueError('require exactly idle/low/medium/high in order')
    previous = 0
    for row in rows:
        if row.get('measured') != 100 or row.get('remote_input_tensor_bytes') != 0:
            raise ValueError('require 100 measured and zero remote input bytes')
        for key in IDENTITIES:
            if not row.get(key) or row[key] != rows[0].get(key):
                raise ValueError('capture identity mismatch: '+key)
        p95 = row.get('inference_p95_ms')
        if type(p95) not in (float,int) or not math.isfinite(p95) or p95 <= previous:
            raise ValueError('inference p95 must increase strictly')
        previous = p95
    return rows


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wait_ready(path, timeout=60):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if Path(path).is_file() and Path(path).stat().st_size:
            return
        time.sleep(.1)
    raise RuntimeError('readiness timeout: '+str(path))


def run(command):
    return subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=650)


def capture_profile(plan):
    with capture_signals():
        return _capture_profile(plan)


def _capture_profile(plan):
    ensure_cohort(plan)
    commands = build_commands(plan)
    directory = Path(plan['directory'])
    directory.mkdir()  # exclusive: never reuse or erase a capture
    write_new(directory/'plan.json', plan)
    write_new(directory/'commands.json',commands)
    containers, processes, logs = [], [], []
    try:
        telemetry = (directory/'tegrastats.log').open('x'); logs.append(telemetry)
        processes.append(subprocess.Popen(commands['telemetry'],stdout=telemetry,stderr=subprocess.STDOUT))
        for role in ('sink','publisher'):
            cid = run(commands[role]).strip()
            if not re.fullmatch('[0-9a-f]{64}',cid):
                raise RuntimeError('invalid created container ID')
            containers.append(cid)
            write_new(directory/(role+'_container_id.json'),cid)
            wait_ready(directory/(role+'_ready.json'))
        if commands['load']:
            log=(directory/'load.log').open('x'); logs.append(log)
            processes.append(subprocess.Popen(commands['load'],stdout=log,stderr=subprocess.STDOUT))
            wait_ready(directory/'load_ready.json')
        with (directory/'native.log').open('x') as log:
            subprocess.run(commands['native'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        for role,cid in zip(('sink','publisher'),containers):
            status=run(['docker','wait',cid]).strip()
            with (directory/(role+'.log')).open('x') as log:
                log.write(run(['docker','logs',cid]))
            if status != '0':
                raise RuntimeError(role+' failed: '+status)
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('telemetry or load ended before capture completed')
        run(commands['summarize'])
    finally:
        # Popen retains child identity; Docker cleanup uses only IDs returned by our run.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
        for cid in containers:
            run(['docker','rm','-f',cid])
        for log in logs: log.close()
    with (directory/'device_snapshot.txt').open('x') as snapshot:
        snapshot.write(run(['uname','-a'])+run(['cat','/etc/nv_tegra_release'])+run(['/usr/sbin/nvpmodel','-q']))
    return summarize_capture(plan)


def summarize_capture(plan):
    import numpy as np
    directory=Path(plan['directory'])
    records=[json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
    summary=json.loads((directory/'summary.json').read_text())
    if len(records)!=120 or summary['samples']['measured']!=100 or summary['samples']['warmup']!=20:
        raise ValueError('incomplete capture')
    if any(r.get('network', {}).get('remote_input_tensor_bytes') != 0 for r in records):
        raise ValueError('remote input detected')
    digest=hashlib.sha256()
    with np.load(str(directory/'output.npz'),allow_pickle=False) as actual, np.load(str(directory/'native_last_output.npz'),allow_pickle=False) as native:
        if set(actual.files)!=set(native.files): raise ValueError('output names mismatch')
        for name in sorted(actual.files):
            if not np.array_equal(actual[name],native[name]): raise ValueError('handoff output mismatch')
            digest.update(name.encode()); digest.update(str(actual[name].dtype).encode())
            digest.update(str(actual[name].shape).encode()); digest.update(actual[name].tobytes())
    # Source, engine, and input identities are taken from pinned files, not labels.
    repo=Path(plan['repo'])
    source=sha_file(repo.parent/'source_sha256sums.txt')
    inputs=hashlib.sha256()
    inputs.update(sha_file(plan['manifest']).encode())
    for path in sorted(Path(plan['data_root']).rglob('*.npy')):
        inputs.update(path.name.encode()); inputs.update(sha_file(path).encode())
    model=records[0]['model']
    result=dict(profile=plan['profile'], measured=100,remote_input_tensor_bytes=0,
        source_sha256=source,model_id=model['model_id'],engine_sha256=sha_file(plan['engine']),
        input_sha256=inputs.hexdigest(),output_sha256=digest.hexdigest(),
        calibration_id=plan['calibration_id'],inference_p95_ms=summary['stages']['gpu_inference']['p95_ms'],
        capture_time_utc=datetime.utcnow().isoformat()+'Z',directory=str(directory))
    if any(r['model'] != model for r in records) or model['engine_sha256']!=result['engine_sha256']:
        raise ValueError('model identity changed during capture')
    write_new(directory/'capture_summary.json',result)
    source_directory=directory/'source'
    source_directory.mkdir()
    for name in ('trace.jsonl','device_snapshot.txt'):
        shutil.copyfile(str(directory/name),str(source_directory/name))
    shutil.copyfile(str(repo.parent/'source_sha256sums.txt'),str(source_directory/'source_sha256sums.txt'))
    with (source_directory/'model_sha256.txt').open('x') as output:
        output.write(result['engine_sha256']+'  '+plan['engine']+'\n')
    write_new(source_directory/'command_manifest.txt',dict(plan,
        source_commit=(repo.parent/'SOURCE_COMMIT').read_text().strip(), **result))
    return result


def calibrate(plan):
    ensure_cohort(plan)
    command=build_commands(dict(plan,profile='high',duty_cycle=.85,
                                directory=str(Path(plan['directory']).parent/'high')))['load']
    directory=Path(plan['directory']).parent/('calibration-'+plan['calibration_id'])
    directory.mkdir()
    command=[str(directory/'ready.json') if x.endswith('/load_ready.json') else
             str(directory/'metrics.jsonl') if x.endswith('/load_metrics.jsonl') else
             '10' if x=='600' else x for x in command]
    result=run(command)
    write_new(directory/'calibration.json',dict(matrix_size=plan['matrix_size'],period_ms=100,
              calibration_id=plan['calibration_id'],command=command,stdout=result))
    return directory



def register_local_capture(root, library_root):
    """Build all four candidates before atomically registering each verified bundle."""
    root, library_root = Path(root).absolute(), Path(library_root).absolute()
    candidate=json.loads((root/'orin_capture_candidate.json').read_text())
    rows=validate_capture_set(candidate['profiles'])
    if not str(root).startswith('/var/tmp/codriving-ditl/task6-orin-'):
        raise ValueError('mirror root must be an authorized task6-orin cohort')
    if library_root != LIBRARY_ROOT:
        raise ValueError('unexpected trace library root')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'simulation/leaderboard/team_code/closedloop'))
    from ditl_trace_library import (BuildRequest, build_feature_bundle, parse_capture_condition,
                                    verify_feature_bundle)
    results=[]
    for row in rows:
        profile='raw_samehost_idle' if row['profile']=='idle' else 'raw_samehost_gpu_'+row['profile']
        source=root/row['profile']/'source'
        records=[json.loads(line) for line in (source/'trace.jsonl').read_text().splitlines()]
        first=records[0]
        provenance=json.loads((source/'command_manifest.txt').read_text())
        if any(provenance.get(k)!=row.get(k) for k in IDENTITIES):
            raise ValueError('mirrored capture identity differs')
        condition=dict(profile_id=profile,workload_label='fixed_input_tensor_replay',
            ros_placement='same_host',link_type='loopback',qos='reliable',publish_frequency_hz=None,
            payload_semantics=first['source']['payload_semantics'],codec='none',
            raw_payload_bytes=first['codec']['raw_payload_bytes'],wire_payload_bytes=first['codec']['wire_payload_bytes'],
            clock_domain='monotonic_raw',clock_comparable=True,source_commit=provenance['source_commit'],
            model_id=row['model_id'],engine_sha256=row['engine_sha256'],device_identity='orin-agx-01',
            jetpack_l4t=(source/'device_snapshot.txt').read_text(),power_mode='MODE_30W',load_label=profile,
            capture_time_utc=row['capture_time_utc'],capture_time_source='UTC at capture summarization')
        result=build_feature_bundle(BuildRequest(source,library_root,root,parse_capture_condition(condition)))
        verified=verify_feature_bundle(result.bundle_path,library_root)
        if result.sample_count!=100 or verified.feature_id!=result.feature_id:
            raise ValueError('bundle verification gate failed')
        results.append((profile,result))
    register_capture_bundles(results,library_root)
    mapping={profile:dict(feature_id=result.feature_id,bundle_path=str(result.bundle_path),
                          manifest_sha256=result.manifest_sha256) for profile,result in results}
    report=dict(status='verified',profiles=list(rows),profile_bundles=mapping)
    write_new(root/'orin_capture_report.json',report)
    return report



def register_capture_bundles(results, library_root):
    """Merge all verified entries under one lock and publish the catalog once."""
    from ditl_trace_library import (_directory, _catalog_lock, _read_file, _parse_catalog,
        _canonical, _replace_catalog, _verify_directory_binding, verify_feature_bundle,
        LIBRARY_SCHEMA_VERSION)
    catalog_path=library_root/'catalog.json'
    with _directory(library_root) as fd, _catalog_lock(fd):
        try: old=_read_file(fd,'catalog.json')
        except FileNotFoundError: old=None
        catalog=(_parse_catalog(old) if old is not None else
                 dict(library_schema_version=LIBRARY_SCHEMA_VERSION,features={}))
        entries=dict(catalog['features'])
        for profile,result in results:
            verified=verify_feature_bundle(result.bundle_path,library_root)
            if verified.feature_id!=result.feature_id or verified.checksums_sha256!=result.checksums_sha256:
                raise ValueError('bundle identity changed before publication')
            entry=dict(checksums_sha256=result.checksums_sha256,profile_id=profile,
                       relative_path='v1/bundles/'+result.feature_id)
            if result.feature_id in entries and entries[result.feature_id]!=entry:
                raise ValueError('existing catalog entry conflicts')
            entries=dict(entries,**{result.feature_id:entry})
        _verify_directory_binding(library_root,fd)
        _replace_catalog(fd,catalog_path,old,_canonical(dict(catalog,features=entries)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort')
    parser.add_argument('--matrix-size',type=int)
    parser.add_argument('--calibration-id')
    parser.add_argument('--mode',choices=('plan','prepare','calibrate','capture','finalize'),default='plan')
    parser.add_argument('--mirror-root',type=Path)
    args=parser.parse_args(argv)
    if args.mode=='finalize':
        if args.mirror_root is None: parser.error('--mirror-root is required')
        register_local_capture(args.mirror_root,LIBRARY_ROOT); return 0
    plans=build_capture_plan(args.cohort,args.matrix_size,args.calibration_id)
    if args.mode=='plan':
        print(json.dumps([dict(p,commands=build_commands(p)) for p in plans],indent=2)); return 0
    if args.mode=='prepare':
        ensure_cohort(plans[0],exclusive=True); return 0
    if args.mode=='calibrate':
        with capture_signals(): calibrate(plans[0])
        return 0
    ensure_cohort(plans[0])
    calibration=Path(plans[0]['directory']).parent/('calibration-'+args.calibration_id)/'calibration.json'
    if json.loads(calibration.read_text())['matrix_size']!=args.matrix_size:
        raise ValueError('matrix differs from explicit calibration')
    with socket.socket() as guard: guard.bind(('127.0.0.1',55021))
    rows=[capture_profile(plan) for plan in plans]
    write_new(Path(plans[0]['directory']).parent/'orin_capture_candidate.json',
              dict(status='candidate',profiles=rows))
    validate_capture_set(rows)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
