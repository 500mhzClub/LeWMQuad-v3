"""Own one native ordinary commitment-contact experiment after both exact predecessors."""
import json
import os
import subprocess
import sys
import time
from scripts import run_go2_commitment_contact_anchored_maze02_pilot_v1 as native
from scripts import commitment_contact_anchored_native_inputs_development as inputs
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT = BASE/'go2_commitment_contact_anchored_maze02_native_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_commitment_contact_anchored_maze02_native_v1.py'
TEST = 'lewm/tests/test_commitment_contact_anchored_native_wait_development.py'
PROTOCOL = 'docs/go2_commitment_contact_anchored_maze02_native_wait_v1_2026-09-10.md'
WAIT_SECONDS = 48*3600
RAW_OWNER = dict(pid=2700744, created=1789047270.48, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', inputs.raw_wait.SOURCE])
PRIOR_OWNER = dict(pid=2671835, created=1789036731.41, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', inputs.prior_wait.SOURCE])
PREREQUISITES = (
    ('raw_prefix_wait_result_sha256', RAW_OWNER, inputs.raw_wait.OUTPUT, inputs.RAW_WAIT_LAUNCH,
        'COMMITMENT_CONTACT_ANCHORED_RAW_PREFIX_WAIT_V1_COMPLETE'),
    ('prior_native_wait_result_sha256', PRIOR_OWNER, inputs.prior_wait.OUTPUT, inputs.PRIOR_WAIT_LAUNCH,
        'HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE'))


def completed_identity(root, launch_sha, status, sources):
    if not (root/'result.json').is_file(): raise ValueError('original owner ended without complete result; no retry')
    sha = digest(root/'result.json'); result, _, _ = inputs.completed(root, sha, launch_sha, sources)
    if result['status'] != status: raise ValueError('exact completed original prerequisite required')
    return sha


def wait_for_inputs(sources, event, *, sleep=time.sleep, clock=time.monotonic):
    start = clock(); completed_ids = {}
    while True:
        live = {}; ids = {}
        for key, owner, root, launch_sha, status in PREREQUISITES:
            live[key] = owner_live(owner)
            if not live[key]:
                ids[key] = completed_identity(root, launch_sha, status, sources)
                if key in completed_ids and completed_ids[key] != ids[key]:
                    raise ValueError('previously completed prerequisite identity changed')
                completed_ids[key] = ids[key]
        if not any(live.values()): verify(sources); return ids
        if clock()-start >= WAIT_SECONDS:
            raise ValueError('original-owner wait expired; no original or replacement run restarted')
        event('WAITING_FOR_RAW_REPLAY_AND_SCHEDULED_HOLD_NATIVE', original_processes_live=live)
        sleep(30)


def authenticate_completed(sources, receipt):
    root = native.OUTPUT
    if (root/'failure.json').exists(): raise ValueError('native ordinary commitment-contact failure retained')
    sha = digest(root/'result.json'); result = read_json(root, 'result.json')
    ids = result['artifact_sha256'] | {'result.json': sha}; verify_artifacts(root, ids)
    launch = read_json(root, 'launch.json')
    if (result['status'] != 'COMMITMENT_CONTACT_ANCHORED_MAZE02_PILOT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or result['raw_prefix_wait_result_sha256'] != receipt['raw_prefix_wait_result_sha256']
            or result['prior_native_wait_result_sha256'] != receipt['prior_native_wait_result_sha256']
            or result['prospective_prefix_result_sha256'] != launch['input_admission']['prefix_result_sha256']
            or len(result['conditions']) != 1):
        raise ValueError('complete exact native ordinary commitment-contact result required')
    native.verify_inputs(launch)
    name = native.CASE[0]; record = result['conditions'][0]
    for suffix in ('_worker_terminal.json', '_audit.json', '_prefix_comparison.json', '_readout.json', '_worker.log'):
        if name+suffix not in ids: raise ValueError('every completed native worker artifact must be bound')
    if (record != read_json(root, name+'_worker_terminal.json')
            or record['prefix_comparison'] != read_json(root, name+'_prefix_comparison.json')
            or record['readout'] != read_json(root, name+'_readout.json')
            or any(ids.get(n) != h for n,h in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])):
        raise ValueError('complete native receipts and outcome count required')
    native.require_worker(record, read_json(root, name+'_audit.json'), launch['input_admission']['prefix_report'])
    verify(sources); verify_artifacts(root, ids)
    return dict(native_result_sha256=sha, measured_round_trip_successes=result['measured_round_trip_successes'],
        all_completion_artifacts_and_receipts_verified=True, scientific_success_required=False)


def main():
    for root in (OUTPUT, native.OUTPUT):
        validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive native waiter and unlaunched child required')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, TEST, native.SOURCE, native.PROTOCOL, *native.TESTS))
    for _, owner, root, launch_sha, status in PREREQUISITES:
        if not owner_live(owner): completed_identity(root, launch_sha, status, sources)
    resources = native.hardware(); native.cohort_resources(resources, 1); verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, boot_id=BOOT,
        original_raw_replay_owner=RAW_OWNER, original_prior_native_owner=PRIOR_OWNER,
        original_raw_wait_launch_sha256=inputs.RAW_WAIT_LAUNCH,
        original_prior_native_wait_launch_sha256=inputs.PRIOR_WAIT_LAUNCH,
        planned_case=list(native.CASE), waiter_pid=os.getpid(), hardware=resources, maximum_wait_s=WAIT_SECONDS,
        automatic_retry=False, source_changes_permitted=False, native_workers_while_waiting=0,
        native_workers_after_original_completion=1))
    print('COMMITMENT_CONTACT_ANCHORED_NATIVE_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-start, **details))+'\n')
                events.flush(); print(status, details, flush=True)
            receipt = wait_for_inputs(sources, event); write_json(OUTPUT/'input_completion.json', receipt)
            verify(sources); native.cohort_resources(native.hardware(), 1); native.require_native_idle()
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('native output appeared outside waiter ownership')
            command = [sys.executable, native.SOURCE,
                '--raw-prefix-wait-result-sha256', receipt['raw_prefix_wait_result_sha256'],
                '--prior-native-wait-result-sha256', receipt['prior_native_wait_result_sha256']]
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('COMMITMENT_CONTACT_ANCHORED_NATIVE_CHILD_STARTED', pid=child.pid, command=command)
                while child.poll() is None:
                    event('COMMITMENT_CONTACT_ANCHORED_NATIVE_CHILD_LIVE', pid=child.pid); time.sleep(30)
            event('COMMITMENT_CONTACT_ANCHORED_NATIVE_CHILD_EXITED', pid=child.pid, returncode=child.returncode)
            if child.returncode != 0: raise ValueError('original native child failed; no retry')
            report = authenticate_completed(sources, receipt); write_json(OUTPUT/'native_completion.json', report)
        names = ('launch.json', 'events.jsonl', 'input_completion.json', 'native_stdout.log', 'native_completion.json')
        verify(sources); ids = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='COMMITMENT_CONTACT_ANCHORED_MAZE02_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, automatic_retry=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('COMMITMENT_CONTACT_ANCHORED_NATIVE_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMMITMENT_CONTACT_ANCHORED_NATIVE_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False, original_work_retained=True)); raise


if __name__ == '__main__': main()
