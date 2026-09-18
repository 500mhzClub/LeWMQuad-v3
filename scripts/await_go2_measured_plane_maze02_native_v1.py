"""Queue one fresh measured-plane episode after the exact existing owner ends."""
import argparse
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
import psutil
import numpy as np
from scripts import run_go2_measured_plane_maze02_pilot_v1 as native
from scripts.startup_source_inventory_development import discover_sources

run,inputs=native.run,native.inputs
SOURCE='scripts/await_go2_measured_plane_maze02_native_v1.py'
TEST='lewm/tests/test_measured_plane_native_wait_development.py'
PROTOCOL='docs/go2_measured_plane_maze02_native_wait_v1_2026-09-11.md'
OUTPUT=run.BASE/'go2_measured_plane_maze02_native_wait_v1_attempt_001'


def wait_for_queue(sources,event,*,sleep=time.sleep):
    while run.owner_live(inputs.QUEUE_OWNER):
        run.verify_artifacts(inputs.queued.OUTPUT,{'launch.json':inputs.QUEUE_LAUNCH_SHA})
        event('EXACT_ORIGINAL_CHAINED_ANCHOR_QUEUE_OWNER_LIVE',owner=inputs.QUEUE_OWNER)
        sleep(30)
    sha=run.digest(inputs.queued.OUTPUT/'result.json')
    inputs.admit_queue(sha,sources)
    run.verify(sources)
    return sha


def authenticate(sources,queue_sha):
    root=native.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve original measured-plane native failure')
    sha=run.digest(root/'result.json');result=run.read_json(root,'result.json')
    ids=result['artifact_sha256'] | {'result.json':sha}
    run.verify_artifacts(root,ids);launch=run.read_json(root,'launch.json')
    if (result['status'] != 'MEASURED_PLANE_MAZE02_PILOT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(k) != v for k,v in result['source_sha256'].items())
            or result['chained_wait_result_sha256'] != queue_sha
            or result['controller_completion_sha256'] != native.prefix.COMPLETION_SHA
            or result['automatic_retry'] is not False or len(result['conditions']) != 1
            or run.owner_live(launch['owner'])):
        raise ValueError('same ended original native child and complete result required')
    record=result['conditions'][0];name=native.CASE[0]
    expected={name+'/'+n for n in native.pipeline.artifacts(2,record['collection'])}
    expected.update(name+s for s in ('_worker_terminal.json','_worker.log','_audit.json','_prefix_comparison.json','_readout.json'))
    expected.update(('launch.json','resource_monitor.jsonl','result.json'))
    if (set(ids) != expected or record != run.read_json(root,name+'_worker_terminal.json')
            or record['collection'] != run.read_json(root,name+'/result.json')
            or record['prefix_comparison'] != run.read_json(root,name+'_prefix_comparison.json')
            or record['readout'] != run.read_json(root,name+'_readout.json')
            or record['worker_log_sha256'] != ids[name+'_worker.log']
            or any(ids.get(k) != v for k,v in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])):
        raise ValueError('complete actual worker receipts and raw artifact roster required')
    audit=run.read_json(root,name+'_audit.json');native.require_worker(record,audit)
    with np.load(root/name/'physics_trace.npz',allow_pickle=False) as raw:
        if native.case_readout(audit,record['collection'],raw['physics_contact']) != record['readout']:
            raise ValueError('physical/contact/timing readout must reconstruct')
    if native.prefix_result(record['collection'],launch['input_admission']['prefix_report']) != record['prefix_comparison']:
        raise ValueError('actual new command and physical/public prefix must reconstruct')
    native.verify_inputs(launch);run.verify(sources);run.verify_artifacts(root,ids)
    return dict(native_result_sha256=sha,measured_round_trip_successes=result['measured_round_trip_successes'],
        complete_native_worker_and_artifact_roster_verified=True,actual_physical_prefix_reconstructed=record['prefix_comparison']['actual_paired_execution_compared'],
        scientific_success_required=False,navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False)


def main(preflight=False):
    for root in (OUTPUT,native.OUTPUT):
        run.validate_root(root,must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched child required')
    inherited=inputs.prepared_sources((native.SOURCE,native.PROTOCOL,*native.TESTS))
    sources=discover_sources((SOURCE,TEST,PROTOCOL),inherited);run.verify(sources)
    hw=native.resources();native.require_environment(run.read_json(native.old.OUTPUT,'launch.json'))
    live=run.owner_live(inputs.QUEUE_OWNER)
    if not live: inputs.admit_queue(run.digest(inputs.queued.OUTPUT/'result.json'),sources)
    if preflight:
        print('MEASURED_PLANE_NATIVE_WAITER_PREFLIGHT',len(sources),'original_queue_live',live,flush=True);return
    run.create_output(OUTPUT);process=psutil.Process()
    run.write_json(OUTPUT/'launch.json',dict(source_sha256=sources,boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline()),
        original_queue_owner=inputs.QUEUE_OWNER,original_queue_launch_sha256=inputs.QUEUE_LAUNCH_SHA,
        controller_completion_sha256=native.prefix.COMPLETION_SHA,planned_case=list(native.CASE),
        environment=run.ENV,hardware=hw,poll_seconds=30,automatic_retry=False,
        native_workers_while_waiting=0,native_workers_after_original_completion=1))
    print('MEASURED_PLANE_NATIVE_WAITER_LAUNCHED',run.digest(OUTPUT/'launch.json'),flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status,**details):
                events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),status=status,**details))+'\n');events.flush()
                print(status,flush=True)
            queue_sha=wait_for_queue(sources,event)
            run.write_json(OUTPUT/'input_completion.json',dict(chained_wait_result_sha256=queue_sha))
            native.resources();native.require_native_idle();run.verify(sources)
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('native root appeared outside owner')
            command=[sys.executable,'-B',native.SOURCE,'--chained-wait-result-sha256',queue_sha]
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                event('MEASURED_PLANE_NATIVE_CHILD_STARTED',pid=child.pid,command=command)
                while child.poll() is None:
                    event('MEASURED_PLANE_NATIVE_CHILD_LIVE',pid=child.pid);time.sleep(30)
            event('MEASURED_PLANE_NATIVE_CHILD_EXITED',pid=child.pid,returncode=child.returncode)
            if child.returncode != 0: raise ValueError('original native child failed; no retry')
            report=authenticate(sources,queue_sha);run.write_json(OUTPUT/'native_completion.json',report)
        ids={n:run.digest(OUTPUT/n) for n in ('launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json')}
        run.verify(sources);run.verify_artifacts(OUTPUT,ids)
        run.write_json(OUTPUT/'result.json',dict(status='MEASURED_PLANE_MAZE02_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,automatic_retry=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('MEASURED_PLANE_NATIVE_WAITER_COMPLETE',run.digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MEASURED_PLANE_NATIVE_WAIT_FAILURE',reason=repr(error),automatic_retry=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args().preflight)
