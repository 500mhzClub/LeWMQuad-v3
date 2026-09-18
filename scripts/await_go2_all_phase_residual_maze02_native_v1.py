"""Own one six-case native launch after both original prerequisites finish."""
import json
import os
import subprocess
import sys
import time
from scripts import run_go2_all_phase_residual_maze02_matched_native_v1 as native
from scripts import all_phase_residual_maze02_native_inputs_development as inputs
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json,verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_all_phase_residual_maze02_native_wait_v1_attempt_001'
SOURCE='scripts/await_go2_all_phase_residual_maze02_native_v1.py'
TEST='lewm/tests/test_all_phase_residual_maze02_native_wait_development.py'
PROTOCOL='docs/go2_all_phase_residual_maze02_native_wait_v1_2026-09-10.md'
WAIT_SECONDS=48*3600


def completed_identity(root,launch_sha,status,sources):
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original prerequisite failed; preserve it and do not launch')
    if not (root/'result.json').is_file():raise ValueError('original owner ended without complete result')
    sha=digest(root/'result.json');result,launch,ids=inputs.completed(root,sha,launch_sha,sources)
    if result['status']!=status:raise ValueError('complete original prerequisite status required')
    return sha


def wait_for_inputs(sources,event,*,sleep=time.sleep,clock=time.monotonic):
    started=clock()
    while True:
        native_live=inputs.corrected_wait.owner_live(inputs.NATIVE_OWNER)
        correction_live=inputs.corrected_wait.owner_live(inputs.CORRECTION_OWNER)
        # A process that has ended requires its original complete output even
        # while the other prerequisite continues. No PID/file-based restart.
        native_sha=correction_sha=None
        if not native_live:
            native_sha=completed_identity(inputs.native.OUTPUT,inputs.NATIVE_LAUNCH_SHA,
                'RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE',sources)
        if not correction_live:
            correction_sha=completed_identity(inputs.corrected_wait.OUTPUT,inputs.WAIT_SHA,
                'ALL_PHASE_TRANSLATION_BIAS_WAIT_COMPLETE',sources)
        if not native_live and not correction_live:
            verify(sources)
            return dict(native_result_sha256=native_sha,correction_wait_result_sha256=correction_sha)
        if clock()-started>=WAIT_SECONDS:
            raise ValueError('bounded original-owner wait expired; no native child launched or restarted')
        event('WAITING_FOR_ORIGINAL_INPUTS',native_owner_live=native_live,correction_owner_live=correction_live)
        sleep(30)


def authenticate_completed(sources,receipt):
    if (native.OUTPUT/'failure.json').exists():raise ValueError('native cohort failure retained')
    sha=digest(native.OUTPUT/'result.json');result=read_json(native.OUTPUT,'result.json')
    ids=result['artifact_sha256']|{'result.json':sha};verify_artifacts(native.OUTPUT,ids)
    launch=read_json(native.OUTPUT,'launch.json')
    if (result['status']!='ALL_PHASE_RESIDUAL_MAZE02_MATCHED_NATIVE_V1_COMPLETE'
            or result['source_sha256']!=launch['source_sha256']
            or any(sources.get(n)!=h for n,h in result['source_sha256'].items())
            or result['correction_wait_result_sha256']!=receipt['correction_wait_result_sha256']
            or result['predecessor_native_result_sha256']!=receipt['native_result_sha256']):
        raise ValueError('complete unchanged original six-case native result required')
    native.verify_inputs(launch)
    records=result['conditions'];reports=[]
    if len(records)!=6:raise ValueError('all six native cases required')
    for case,record in zip(native.CASES,records,strict=True):
        for suffix in ('_audit.json','_startup_comparison.json','_worker_terminal.json','_worker.log','_readout.json'):
            if case[0]+suffix not in ids:raise ValueError('complete bound case evidence required')
        if (record!=read_json(native.OUTPUT,case[0]+'_worker_terminal.json')
                or record['startup_comparison']!=read_json(native.OUTPUT,case[0]+'_startup_comparison.json')
                or record['readout']!=read_json(native.OUTPUT,case[0]+'_readout.json')
                or any(ids.get(n)!=h for n,h in record['artifact_sha256'].items())):
            raise ValueError('unchanged complete worker receipts and artifacts required')
        reports.append(read_json(native.OUTPUT,case[0]+'_audit.json'))
    summary=native.complete_cohort(records,reports)
    if any(result[k]!=v for k,v in summary.items()):raise ValueError('all six native outcomes must reproduce exactly')
    verify(sources);verify_artifacts(native.OUTPUT,ids)
    return dict(native_result_sha256=sha,all_fixed_cases_executed=True,
        measured_round_trip_successes=summary['measured_round_trip_successes'],
        all_completion_receipts_and_artifacts_verified=True,scientific_success_required=False)


def main():
    for root in (OUTPUT,native.OUTPUT):
        validate_root(root,must_exist=False)
        if root.exists() or root.is_symlink():raise ValueError('exclusive waiter and previously unlaunched cohort required')
    sources=inputs.prepared_sources((SOURCE,TEST,PROTOCOL,native.SOURCE,native.PROTOCOL,*native.TESTS))
    if not inputs.corrected_wait.owner_live(inputs.NATIVE_OWNER):
        raise ValueError('original maze3 owner must be live when registering this waiter')
    resources=hardware();native.resources_for(resources,6)
    verify(sources);create_output(OUTPUT)
    launch=dict(source_sha256=sources,original_native_owner=inputs.NATIVE_OWNER,
        original_correction_owner=inputs.CORRECTION_OWNER,boot_id=inputs.corrected_wait.BOOT,
        native_launch_sha256=inputs.NATIVE_LAUNCH_SHA,correction_wait_launch_sha256=inputs.WAIT_SHA,
        waiter_pid=os.getpid(),planned_cases=[list(c) for c in native.CASES],hardware=resources,
        maximum_wait_s=WAIT_SECONDS,automatic_retry=False,source_changes_permitted=False,
        native_workers_while_waiting=0,native_workers_after_original_owner=1)
    write_json(OUTPUT/'launch.json',launch);started=time.monotonic()
    print('ALL_PHASE_RESIDUAL_MAZE02_WAITER_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status,**details):
                events.write(json.dumps(dict(status=status,elapsed_s=time.monotonic()-started,**details))+'\n');events.flush()
                print(status,details,flush=True)
            receipt=wait_for_inputs(sources,event);write_json(OUTPUT/'input_completion.json',receipt)
            verify(sources);resources=hardware();native.resources_for(resources,6);native.require_native_idle()
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink():raise ValueError('cohort appeared outside waiter ownership')
            command=[sys.executable,native.SOURCE,'--correction-wait-result-sha256',receipt['correction_wait_result_sha256'],
                '--native-result-sha256',receipt['native_result_sha256']]
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                event('MATCHED_NATIVE_CHILD_STARTED',pid=process.pid,command=command,hardware=resources)
                while process.poll() is None:
                    event('MATCHED_NATIVE_CHILD_LIVE',pid=process.pid);time.sleep(30)
            event('MATCHED_NATIVE_CHILD_EXITED',pid=process.pid,returncode=process.returncode)
            if process.returncode!=0:raise ValueError('original six-case native child failed; no retry')
            report=authenticate_completed(sources,receipt);write_json(OUTPUT/'native_completion.json',report)
        files=('launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json')
        verify(sources);ids={n:digest(OUTPUT/n) for n in files};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='ALL_PHASE_RESIDUAL_MAZE02_NATIVE_WAIT_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,automatic_retry=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('ALL_PHASE_RESIDUAL_MAZE02_WAITER_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ALL_PHASE_RESIDUAL_MAZE02_NATIVE_WAIT_FAILURE',
            reason=repr(error),automatic_retry=False,original_work_retained=True));raise


if __name__=='__main__':main()
