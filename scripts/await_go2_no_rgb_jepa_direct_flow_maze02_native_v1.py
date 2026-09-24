"""Append one fixed recovery test after its exact replay and the original queue."""
import json
import os
import re
import subprocess
import sys
import time
import numpy as np
from scripts import run_go2_no_rgb_jepa_direct_flow_maze02_pilot_v1 as native
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = BASE/'go2_no_rgb_jepa_direct_flow_maze02_native_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_no_rgb_jepa_direct_flow_maze02_native_v1.py'
TEST = 'lewm/tests/test_no_rgb_jepa_direct_flow_native_wait_development.py'
PROTOCOL = 'docs/go2_no_rgb_jepa_direct_flow_maze02_native_wait_v1_2026-09-10.md'
PREPARATION = 'docs/go2_no_rgb_jepa_direct_flow_native_launcher_preparation_2026-09-10.json'
PREPARATION_SHA = '3e783e4223bfd48944aa8e58a1aa7960dce6a0688d35712e3b704a92e3f9f260'
WAIT_SECONDS = 48*3600
PREFIX_BINDINGS = {
    'launch.json':'324dacf5bf45ca09faf83985a3f940e7999e40b6cf3688c15e249e88a60e0486',
    'report.json':'77c335b55d3b0f225b26c925bf07b6ff9035d8cce9694582f599fe5614e639ff',
    'context_decisions.jsonl.gz':'2748089cb1e2ea48661e44595c042bdbeb1d081d506e8379990eb5483d70bc50',
}
BATCH_OWNER = dict(pid=2659758,created=1789030196.29,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',native.inputs.replay.batch.SOURCE,
    '--correction-wait-result-sha256','4bf13d2e00fb318fa836d02bad93784fbb1a9c5ccef8792bd19dcfd503f657a0',
    '--native-result-sha256','330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723'])
PREREQUISITES = (
    ('prefix',native.inputs.REPLAY_OWNER,native.inputs.replay.OUTPUT,PREFIX_BINDINGS['launch.json'],
     'NO_RGB_JEPA_DIRECT_FLOW_CONTROLLER_PREFIX_V1_COMPLETE'),
    ('batch',BATCH_OWNER,native.inputs.replay.batch.OUTPUT,
     '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a',
     'ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE'),
    *((key,dict(pid=pid,created=created,command=['.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',waiter.SOURCE]),
        waiter.OUTPUT,sha,status) for key,waiter,sha,status,pid,created in native.inputs.queue.JOBS),
)


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    preparation=json.loads((ROOT/PREPARATION).read_text())
    if preparation['status'] != 'DIRECT_FLOW_ANCHORED_NATIVE_LAUNCHER_SOURCE_CHECKED_NOT_EXECUTED':
        raise ValueError('checked exact native launcher required')
    verify(preparation['source_sha256'])
    sources=discover_sources((SOURCE,TEST,PROTOCOL,PREPARATION),preparation['source_sha256']);verify(sources)
    return sources


def completion_identity(spec,sources):
    """Light completion identity only; full artifact/model admission is later."""
    key,_,root,launch_sha,status=spec
    if (root/'failure.json').exists() or not (root/'result.json').is_file():
        raise ValueError('original '+key+' owner ended without successful completion; no restart')
    sha=digest(root/'result.json');verify_artifacts(root,{'launch.json':launch_sha,'result.json':sha})
    result=read_json(root,'result.json');launch=read_json(root,'launch.json')
    if (result['status'] != status or result['artifact_sha256'].get('launch.json') != launch_sha
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())):
        raise ValueError('exact completed original '+key+' source and launch identity required')
    if key=='prefix':
        if result['artifact_sha256'] != PREFIX_BINDINGS:
            raise ValueError('exact reviewed controller boundary artifacts required')
        verify_artifacts(root,{'report.json':PREFIX_BINDINGS['report.json']})
        if result['report'] != read_json(root,'report.json'):
            raise ValueError('completed replay must retain the reviewed boundary')
        native.boundary(result['report'])
        if result['report']['boundary_requested_command'] != [0.,0.,-.45]:
            raise ValueError('reviewed right-turn intervention required')
    elif key=='batch':
        if result.get('all_fixed_cases_executed') is not True or len(result['conditions']) != 6:
            raise ValueError('complete original six-case batch required')
    elif result.get('automatic_retry') is not False:
        raise ValueError('original queue completion without retry required')
    return sha


def wait_for_inputs(sources,event,*,sleep=time.sleep,clock=time.monotonic):
    start=clock();completed={}
    while True:
        live={};ids={}
        for spec in PREREQUISITES:
            key,owner,_,_,_=spec;live[key]=owner_live(owner)
            if not live[key]:
                ids[key]=completion_identity(spec,sources)
                if key in completed and completed[key] != ids[key]:
                    raise ValueError('previously completed original identity changed: '+key)
                completed[key]=ids[key]
        if not any(live.values()):verify(sources);return ids
        if clock()-start >= WAIT_SECONDS:raise ValueError('original prerequisite wait expired; no retry or replacement')
        event('WAITING_FOR_ORIGINAL_REPLAY_AND_NATIVE_QUEUE',original_processes_live=live,completed_result_sha256=dict(completed))
        sleep(30)


def command_for(ids):
    if (type(ids) is not dict or set(ids) != {spec[0] for spec in PREREQUISITES}
            or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}',v) is None for v in ids.values())):
        raise ValueError('all five exact completed original SHA-256 prerequisites required')
    return [sys.executable,native.SOURCE,'--controller-prefix-result-sha256',ids['prefix'],
        '--adapter-batch-result-sha256',ids['batch'],'--frontier-wait-result-sha256',ids['frontier'],
        '--hold-wait-result-sha256',ids['hold'],'--contact-wait-result-sha256',ids['contact']]


def authenticate_completed(sources,ids):
    root=native.OUTPUT
    if (root/'failure.json').exists():raise ValueError('native tracking-recovery failure preserved')
    sha=digest(root/'result.json');result=read_json(root,'result.json')
    bindings=result['artifact_sha256']|{'result.json':sha};verify_artifacts(root,bindings)
    launch=read_json(root,'launch.json')
    if (result['status'] != 'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_PILOT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or result['prospective_prefix_result_sha256'] != ids['prefix']
            or result['original_adapter_batch_result_sha256'] != ids['batch']
            or result['original_ordered_waiter_result_sha256'] != {k:ids[k] for k in ('frontier','hold','contact')}
            or len(result['conditions']) != 1):
        raise ValueError('complete native test tied to all exact original prerequisites required')
    native.verify_inputs(launch)
    name=native.CASE[0];record=result['conditions'][0]
    required=[name+'/'+n for n in native.artifacts(2,record['collection'])]
    required += [name+s for s in ('_worker_terminal.json','_worker.log','_audit.json','_prefix_comparison.json','_readout.json')]
    if any(n not in bindings for n in required):raise ValueError('complete collection and worker artifact roster required')
    if (record != read_json(root,name+'_worker_terminal.json')
            or record['collection'] != read_json(root,name+'/result.json')
            or record['prefix_comparison'] != read_json(root,name+'_prefix_comparison.json')
            or record['readout'] != read_json(root,name+'_readout.json')
            or record['worker_log_sha256'] != bindings[name+'_worker.log']
            or any(bindings.get(n) != h for n,h in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])):
        raise ValueError('complete exact native result and worker receipts required')
    audit=read_json(root,name+'_audit.json');native.require_worker(record,audit,launch['input_admission']['prefix_report'])
    with np.load(root/name/'physics_trace.npz',allow_pickle=False) as raw:
        if native.case_readout(audit,record['collection'],raw['physics_contact']) != record['readout']:
            raise ValueError('native contact/timing readout must reproduce')
    verify(sources);verify_artifacts(root,bindings)
    return dict(native_result_sha256=sha,measured_round_trip_successes=result['measured_round_trip_successes'],
        complete_native_worker_and_artifact_roster_verified=True,scientific_success_required=False)


def main():
    for root in (OUTPUT,native.OUTPUT):
        validate_root(root,must_exist=False)
        if root.exists() or root.is_symlink():raise ValueError('exclusive waiter and unlaunched native child required')
    sources=prepared_sources()
    for spec in PREREQUISITES:
        if not owner_live(spec[1]):completion_identity(spec,sources)
    resources=native.hardware();native.cohort_resources(resources,1);verify(sources);create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,boot_id=BOOT,waiter_pid=os.getpid(),
        original_owners={s[0]:s[1] for s in PREREQUISITES},reviewed_prefix_artifact_sha256=PREFIX_BINDINGS,
        planned_case=list(native.CASE),hardware=resources,maximum_wait_s=WAIT_SECONDS,
        automatic_retry=False,source_changes_permitted=False,native_workers_while_waiting=0,
        native_workers_after_original_completion=1,light_completion_identity_checks_while_waiting=True,
        full_input_admission_deferred_to_native_launcher=True))
    print('DIRECT_FLOW_ANCHORED_NATIVE_WAITER_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    start=time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status,**details):
                events.write(json.dumps(dict(status=status,elapsed_s=time.monotonic()-start,**details))+'\n')
                events.flush();print(status,details,flush=True)
            ids=wait_for_inputs(sources,event);write_json(OUTPUT/'input_completion.json',ids)
            native.inputs.owners_ended();verify(sources);native.cohort_resources(native.hardware(),1);native.require_native_idle()
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink():raise ValueError('native root appeared outside waiter ownership')
            command=command_for(ids)
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                event('DIRECT_FLOW_ANCHORED_NATIVE_CHILD_STARTED',pid=child.pid,command=command)
                while child.poll() is None:
                    event('DIRECT_FLOW_ANCHORED_NATIVE_CHILD_LIVE',pid=child.pid);time.sleep(30)
            event('DIRECT_FLOW_ANCHORED_NATIVE_CHILD_EXITED',pid=child.pid,returncode=child.returncode)
            if child.returncode != 0:raise ValueError('original native child failed; no retry')
            report=authenticate_completed(sources,ids);write_json(OUTPUT/'native_completion.json',report)
        names=('launch.json','events.jsonl','input_completion.json','native_stdout.log','native_completion.json')
        verify(sources);bindings={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='NO_RGB_JEPA_DIRECT_FLOW_MAZE02_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,report=report,automatic_retry=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('DIRECT_FLOW_ANCHORED_NATIVE_WAITER_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DIRECT_FLOW_ANCHORED_NATIVE_WAIT_FAILURE',
            reason=repr(error),automatic_retry=False,original_work_retained=True));raise


if __name__=='__main__':main()
