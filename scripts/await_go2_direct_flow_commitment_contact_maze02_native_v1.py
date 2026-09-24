"""Own one contact-plus-flow native child after the original replay and full queue."""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

from scripts import run_go2_direct_flow_commitment_contact_maze02_pilot_v1 as native
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/await_go2_direct_flow_commitment_contact_maze02_native_v1.py'
TEST = 'lewm/tests/test_direct_flow_commitment_contact_native_wait_development.py'
PROTOCOL = 'docs/go2_direct_flow_commitment_contact_maze02_native_wait_v1_2026-09-11.md'
PREPARATION = 'docs/go2_direct_flow_commitment_contact_native_launcher_preparation_2026-09-11.json'
PREPARATION_SHA = 'fb04ba82edaf23be053ec2add9078b8db16e9cabbfe07ff93bbc714ac46e5945'
OUTPUT = BASE/'go2_direct_flow_commitment_contact_maze02_native_wait_v1_attempt_001'
WAIT_SECONDS = 48*3600
RAW_RESULT_SHA = 'a4404e737a29f2499ee5313fcb116bc7a980f6fff5c2f75122ba6be6df437560'
PREREQUISITES = (
    ('raw', native.inputs.prefix.OWNER, native.inputs.replay.OUTPUT, native.inputs.prefix.LAUNCH_SHA,
        'CONTACT_ANCHORED_DIRECT_FLOW_CONTROLLER_PREFIX_V1_COMPLETE'),
    ('sustained', native.inputs.SUSTAINED_OWNER, native.inputs.sustained.OUTPUT, native.inputs.SUSTAINED_LAUNCH,
        'SUSTAINED_HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE'))



def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    prep = json.loads((ROOT/PREPARATION).read_text())
    if prep['status'] != 'DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_LAUNCHER_PREPARED':
        raise ValueError('tested exact contact-plus-flow native launcher required')
    verify(prep['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), prep['source_sha256'])
    verify(sources)
    return sources


def completion_identity(spec, sources):
    """Light identity check; full raw/model/queue admission belongs to the child."""
    key, _, root, launch_sha, status = spec
    if ((root/'failure.json').exists() or (root/'failure.json').is_symlink()
            or not (root/'result.json').is_file()):
        raise ValueError('original '+key+' owner ended without complete result; no restart')
    sha = digest(root/'result.json')
    verify_artifacts(root, {'launch.json':launch_sha, 'result.json':sha})
    result = read_json(root, 'result.json'); launch = read_json(root, 'launch.json')
    if (result['status'] != status or result['artifact_sha256'].get('launch.json') != launch_sha
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())):
        raise ValueError('exact original '+key+' completion source and launch identity required')
    if key == 'raw':
        if sha != RAW_RESULT_SHA: raise ValueError('exact verified completed contact controller replay required')
        native.boundary(result['report'])
        if result['report']['native_execution'] is not False:
            raise ValueError('original prospective raw prefix must not claim native execution')
    elif result.get('automatic_retry') is not False:
        raise ValueError('original sustained-turn waiter completion without retry required')
    return sha


def wait_for_inputs(sources, event, *, sleep=time.sleep, clock=time.monotonic):
    start = clock(); completed = {}
    while True:
        live = {}; ids = {}
        for spec in PREREQUISITES:
            key, owner, _, _, _ = spec; live[key] = owner_live(owner)
            if not live[key]:
                ids[key] = completion_identity(spec, sources)
                if key in completed and completed[key] != ids[key]:
                    raise ValueError('previously completed original identity changed: '+key)
                completed[key] = ids[key]
        if not any(live.values()): verify(sources); return ids
        if clock()-start >= WAIT_SECONDS:
            raise ValueError('original prerequisite wait expired; no retry or replacement')
        event('WAITING_FOR_CONTACT_FLOW_RAW_AND_ORIGINAL_SUSTAINED_QUEUE', original_processes_live=live,
            completed_result_sha256=dict(completed))
        sleep(30)


def command_for(ids):
    if (type(ids) is not dict or set(ids) != {'raw', 'sustained'}
            or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in ids.values())):
        raise ValueError('exact completed raw replay and original sustained-turn waiter SHA-256 identities required')
    return [sys.executable, '-B', native.SOURCE, '--raw-prefix-result-sha256', ids['raw'],
        '--sustained-wait-result-sha256', ids['sustained']]


def authenticate_completed(sources, ids):
    root = native.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original contact-plus-flow native failure must be preserved')
    sha = digest(root/'result.json'); result = read_json(root, 'result.json')
    bindings = result['artifact_sha256'] | {'result.json':sha}; verify_artifacts(root, bindings)
    launch = read_json(root, 'launch.json'); admission = launch['input_admission']
    if (result['status'] != 'DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_PILOT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())
            or len(result['conditions']) != 1
            or result['raw_prefix_result_sha256'] != ids['raw']
            or admission['raw_prefix_result_sha256'] != ids['raw']
            or result['prospective_prefix_result_sha256'] != ids['raw']
            or result['sustained_wait_result_sha256'] != ids['sustained']
            or admission['sustained_wait_result_sha256'] != ids['sustained']):
        raise ValueError('complete contact-plus-flow native result tied to both original completions required')
    native.verify_inputs(launch)
    name = native.CASE[0]; record = result['conditions'][0]
    required = [name+'/'+n for n in native.artifacts(2, record['collection'])]
    required += [name+s for s in ('_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json')]
    required += ['launch.json', 'resource_monitor.jsonl']
    if any(n not in bindings for n in required):
        raise ValueError('complete raw collection and worker artifact roster required')
    if (record != read_json(root, name+'_worker_terminal.json')
            or record['collection'] != read_json(root, name+'/result.json')
            or record['prefix_comparison'] != read_json(root, name+'_prefix_comparison.json')
            or record['readout'] != read_json(root, name+'_readout.json')
            or record['worker_log_sha256'] != bindings[name+'_worker.log']
            or any(bindings.get(n) != h for n, h in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])):
        raise ValueError('complete actual contact-plus-flow worker receipts and outcome accounting required')
    audit = read_json(root, name+'_audit.json')
    native.require_worker(record, audit, admission['prefix_report'])
    with np.load(root/name/'physics_trace.npz', allow_pickle=False) as raw:
        if native.case_readout(audit, record['collection'], raw['physics_contact']) != record['readout']:
            raise ValueError('actual physical/contact/timing readout must reconstruct')
    replay = native.inputs.replay
    receipt = native.compare(replay.native.OUTPUT/replay.native.CASE[0], root/name,
        replay.OUTPUT, admission['prefix_report'])
    if receipt != record['prefix_comparison']:
        raise ValueError('actual contact-plus-flow physical and public prefix must reconstruct')
    verify(sources); verify_artifacts(root, bindings)
    return dict(native_result_sha256=sha, measured_round_trip_successes=result['measured_round_trip_successes'],
        complete_native_worker_and_artifact_roster_verified=True, actual_physical_prefix_reconstructed=True,
        scientific_success_required=False, final_independent_population_policy_review_performed=False)


def main():
    for root in (OUTPUT, native.OUTPUT):
        validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched contact-plus-flow child required')
    sources = prepared_sources()
    for spec in PREREQUISITES:
        if not owner_live(spec[1]): completion_identity(spec, sources)
    resources = native.hardware(); native.cohort_resources(resources, 1); verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, boot_id=BOOT, waiter_pid=os.getpid(),
        original_owners={s[0]:s[1] for s in PREREQUISITES}, planned_case=list(native.CASE),
        prerequisite_launch_sha256={s[0]:s[3] for s in PREREQUISITES},
        hardware=resources, maximum_wait_s=WAIT_SECONDS, automatic_retry=False, source_changes_permitted=False,
        native_workers_while_waiting=0, native_workers_after_original_completion=1,
        light_completion_identity_checks_while_waiting=True, full_input_admission_deferred_to_native_launcher=True))
    print('CONTACT_FLOW_NATIVE_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-start, **details))+'\n')
                events.flush(); print(status, details, flush=True)
            ids = wait_for_inputs(sources, event); write_json(OUTPUT/'input_completion.json', ids)
            native.inputs.owners_ended()
            verify(sources); native.cohort_resources(native.hardware(), 1); native.require_native_idle()
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('native root appeared outside this owner')
            command = command_for(ids)
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('CONTACT_FLOW_NATIVE_CHILD_STARTED', pid=child.pid, command=command)
                while child.poll() is None:
                    event('CONTACT_FLOW_NATIVE_CHILD_LIVE', pid=child.pid); time.sleep(30)
            event('CONTACT_FLOW_NATIVE_CHILD_EXITED', pid=child.pid, returncode=child.returncode)
            if child.returncode != 0: raise ValueError('original contact-plus-flow child failed; no retry')
            report = authenticate_completed(sources, ids); write_json(OUTPUT/'native_completion.json', report)
        names = ('launch.json', 'events.jsonl', 'input_completion.json', 'native_stdout.log', 'native_completion.json')
        verify(sources); bindings = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, automatic_retry=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('CONTACT_FLOW_NATIVE_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CONTACT_FLOW_NATIVE_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False, original_work_retained=True))
        raise


if __name__ == '__main__': main()
