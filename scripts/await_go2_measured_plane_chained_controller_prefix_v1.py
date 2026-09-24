"""Wait for the reserved timing job, then execute one causal controller replay."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

import psutil

from scripts import replay_go2_measured_plane_chained_controller_prefix_v1 as job

run = job.run
SOURCE = 'scripts/await_go2_measured_plane_chained_controller_prefix_v1.py'
TEST = 'lewm/tests/test_measured_plane_chained_controller_wait_development.py'
PROTOCOL = 'docs/go2_measured_plane_chained_controller_wait_v1_2026-09-12.md'
OUTPUT = run.BASE/'go2_measured_plane_chained_controller_wait_v1_attempt_001'


def wait_for_predecessor(event):
    while True:
        try:
            live = run.owner_live(job.CPU_OWNER)
            run.verify_artifacts(job.timing_waiter.OUTPUT, {'launch.json':job.CPU_LAUNCH_SHA})
        except (OSError,psutil.AccessDenied) as error:
            event('CPU_OWNER_OBSERVATION_RETRY',reason=repr(error));time.sleep(30);continue
        if not live:break
        event('EXACT_RESERVED_TIMING_OWNER_LIVE');time.sleep(30)
    root = job.timing_waiter.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve preceding timing failure without automatic dispatch')
    return run.digest(job.native.OUTPUT/'result.json'),run.digest(root/'result.json')


def wait_for_resources(event):
    while True:
        try: hardware = run.hardware()
        except (OSError,psutil.AccessDenied) as error:
            event('RESOURCE_OBSERVATION_RETRY',reason=repr(error));time.sleep(30);continue
        if hardware['memory_available_bytes'] >= 64*1024**3 and hardware['artifact_free_bytes'] >= 43*1024**3:
            return hardware
        event('PAIRED_CONTROLLER_RESOURCE_WAIT',memory_available_bytes=hardware['memory_available_bytes'],
            artifact_free_bytes=hardware['artifact_free_bytes']);time.sleep(30)


def completed_child(sources,native_sha,cpu_sha):
    root=job.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed controller replay without retry')
    launch=run.read_json(root,'launch.json')
    if (run.owner_live(launch['owner'])
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('original controller replay owner must end on its recorded boot')
    sha=run.digest(root/'result.json');result=run.read_json(root,'result.json')
    if (result['status'] != 'MEASURED_PLANE_CHAINED_CONTROLLER_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(k) != v for k,v in result['source_sha256'].items())
            or result['complete_output_and_consumed_public_packets_rechecked'] is not True
            or result['original_raw_inputs_reauthenticated_before_and_after'] is not True
            or any(result[k] is not False for k in ('native_execution','navigation_qualified','goal_achieved'))
            or launch['stop_at_first_changed_command_or_terminal'] is not True
            or launch['automatic_retry'] is not False
            or set(result['artifact_sha256']) != {'launch.json','context_decisions.jsonl.gz','resource_monitor.jsonl','report.json'}):
        raise ValueError('complete exact causal controller replay with diagnostic scope required')
    ids=result['artifact_sha256']|{'result.json':sha};run.verify_artifacts(root,ids)
    admission=job.admit(native_sha,cpu_sha,sources)
    if launch['input_admission'] != admission or result['report'] != run.read_json(root,'report.json'):
        raise ValueError('same complete controller replay inputs and saved report required')
    job.check_output(result['report'],admission)
    run.verify(sources);run.verify_artifacts(root,ids)
    report=result['report']
    return dict(replay_result_sha256=sha,native_result_sha256=native_sha,timing_waiter_result_sha256=cpu_sha,
        original_owner_ended=True,complete_consumed_rows_and_public_packets_reconstructed=True,
        original_raw_artifacts_reauthenticated=True,controller_replay_reexecuted=False,
        frames=report['frames'],boundary_comparison=report['boundary_comparison'],
        actual_model_forward_calls=report['actual_model_forward_calls'],
        following_changed_command_outcome_consumed=False,changed_command_executed=False,
        native_execution=False,navigation_recovered=False,goal_achieved=False)


def main(preflight=False):
    for root in (OUTPUT,job.OUTPUT):
        run.validate_root(root,must_exist=False)
        if root.exists() or root.is_symlink():raise ValueError('exclusive waiter and unlaunched controller replay required')
    if not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items()):
        raise ValueError('original deterministic CPU environment required')
    sources=job.prepared_sources((SOURCE,TEST,PROTOCOL));hardware=job.original.resources()
    live=run.owner_live(job.CPU_OWNER)
    if preflight:
        print('MEASURED_PLANE_CHAINED_WAITER_PREFLIGHT',len(sources),'preceding_owner_live',live,flush=True);return
    run.create_output(OUTPUT);process=psutil.Process()
    run.write_json(OUTPUT/'launch.json',dict(source_sha256=sources,hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline()),
        predecessor_owner=job.CPU_OWNER,predecessor_launch_sha256=job.CPU_LAUNCH_SHA,
        poll_seconds=30,full_replays_while_waiting=0,maximum_full_cpu_replays=1,
        native_execution=False,automatic_retry=False))
    print('MEASURED_PLANE_CHAINED_WAITER_LAUNCHED',run.digest(OUTPUT/'launch.json'),flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status,**details):
                events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),status=status,**details))+'\n');events.flush()
            native_sha,cpu_sha=wait_for_predecessor(event)
            job.admit(native_sha,cpu_sha,sources);wait_for_resources(event);run.verify(sources)
            if job.OUTPUT.exists() or job.OUTPUT.is_symlink():raise ValueError('controller replay root appeared outside this waiter')
            command=[sys.executable,'-B',job.SOURCE,'--learned-result-sha256',native_sha,'--timing-waiter-result-sha256',cpu_sha]
            with (OUTPUT/'replay_stdout.log').open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                event('CHAINED_CONTROLLER_REPLAY_STARTED',pid=child.pid,command=command)
                while child.poll() is None:
                    event('CHAINED_CONTROLLER_REPLAY_LIVE',pid=child.pid);time.sleep(30)
            event('CHAINED_CONTROLLER_REPLAY_EXITED',pid=child.pid,returncode=child.returncode)
            if child.returncode != 0:raise ValueError('controller replay failed; preserve without retry')
            report=completed_child(sources,native_sha,cpu_sha);run.write_json(OUTPUT/'completion.json',report)
        ids={name:run.digest(OUTPUT/name) for name in ('launch.json','events.jsonl','replay_stdout.log','completion.json')}
        run.verify(sources);run.verify_artifacts(OUTPUT,ids)
        run.write_json(OUTPUT/'result.json',dict(status='MEASURED_PLANE_CHAINED_CONTROLLER_WAIT_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,automatic_retry=False,
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('MEASURED_PLANE_CHAINED_WAITER_COMPLETE',run.digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MEASURED_PLANE_CHAINED_CONTROLLER_WAIT_FAILURE',
            reason=repr(error),automatic_retry=False,original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args().preflight)
