"""Own one frontier simulation after both exact original processes complete."""
import json
import os
import subprocess
import sys
import time
from scripts import run_go2_reached_frontier_maze03_pilot_v1 as native
from scripts import reached_frontier_native_inputs_development as inputs
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT = BASE/'go2_reached_frontier_maze03_native_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_reached_frontier_maze03_native_v1.py'
TEST = 'lewm/tests/test_reached_frontier_native_wait_development.py'
PROTOCOL = 'docs/go2_reached_frontier_maze03_native_wait_v1_2026-09-10.md'
WAIT_SECONDS = 48*3600
BATCH_OWNER = dict(pid=2659758, created=1789030196.29, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', inputs.batch.SOURCE,
    '--correction-wait-result-sha256', '4bf13d2e00fb318fa836d02bad93784fbb1a9c5ccef8792bd19dcfd503f657a0',
    '--native-result-sha256', '330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723'])
PREFIX_OWNER = dict(pid=2660458, created=1789030447.53, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', inputs.replay.SOURCE])
PREREQUISITES = (
    ('prefix_result_sha256', PREFIX_OWNER, inputs.replay.OUTPUT, inputs.PREFIX_LAUNCH,
        'REACHED_FRONTIER_MAZE03_PREFIX_V1_COMPLETE'),
    ('adapter_batch_result_sha256', BATCH_OWNER, inputs.batch.OUTPUT, inputs.BATCH_LAUNCH,
        'ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE'))


def completed_identity(root, launch_sha, status, sources):
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original prerequisite failed; preserve it and leave frontier simulation unlaunched')
    if not (root/'result.json').is_file():
        raise ValueError('original owner ended without a complete result; no retry')
    sha = digest(root/'result.json')
    result, _, _ = inputs.completed(root, sha, launch_sha, sources)
    if result['status'] != status: raise ValueError('complete exact original prerequisite status required')
    return sha


def wait_for_inputs(sources, event, *, sleep=time.sleep, clock=time.monotonic):
    start = clock(); completed_ids = {}
    while True:
        identities = {}; live = {}
        for key, owner, root, launch_sha, status in PREREQUISITES:
            live[key] = owner_live(owner)
            if not live[key]:
                identities[key] = completed_identity(root, launch_sha, status, sources)
                if key in completed_ids and completed_ids[key] != identities[key]:
                    raise ValueError('already completed original prerequisite identity changed')
                completed_ids[key] = identities[key]
        if not any(live.values()):
            verify(sources); return identities
        if clock()-start >= WAIT_SECONDS:
            raise ValueError('bounded original-owner wait expired; no original or replacement child restarted')
        event('WAITING_FOR_ORIGINAL_FRONTIER_INPUTS', original_processes_live=live)
        sleep(30)


def authenticate_completed(sources, receipt):
    root = native.OUTPUT
    if (root/'failure.json').exists(): raise ValueError('frontier native failure retained')
    sha = digest(root/'result.json'); result = read_json(root, 'result.json')
    ids = result['artifact_sha256'] | {'result.json': sha}; verify_artifacts(root, ids)
    launch = read_json(root, 'launch.json')
    if (result['status'] != 'REACHED_FRONTIER_MAZE03_PILOT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())
            or result['prospective_prefix_result_sha256'] != receipt['prefix_result_sha256']
            or result['adapter_batch_result_sha256'] != receipt['adapter_batch_result_sha256']
            or len(result['conditions']) != 1):
        raise ValueError('complete exact frontier native result required')
    native.verify_inputs(launch)
    name = native.CASE[0]; record = result['conditions'][0]
    for suffix in ('_worker_terminal.json', '_audit.json', '_prefix_comparison.json', '_readout.json', '_worker.log'):
        if name+suffix not in ids: raise ValueError('all frontier worker artifacts must be bound')
    if (record != read_json(root, name+'_worker_terminal.json')
            or record['prefix_comparison'] != read_json(root, name+'_prefix_comparison.json')
            or record['readout'] != read_json(root, name+'_readout.json')
            or any(ids.get(n) != h for n, h in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])):
        raise ValueError('complete frontier worker receipts and outcome count required')
    native.require_worker(record, read_json(root, name+'_audit.json'), launch['input_admission']['prefix_report'])
    verify(sources); verify_artifacts(root, ids)
    return dict(native_result_sha256=sha, all_completion_receipts_and_artifacts_verified=True,
        measured_round_trip_successes=result['measured_round_trip_successes'], scientific_success_required=False)


def main():
    for root in (OUTPUT, native.OUTPUT):
        validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched frontier simulation required')
    sources = inputs.prepared_sources((SOURCE, TEST, PROTOCOL, native.SOURCE, native.PROTOCOL, *native.TESTS))
    if not owner_live(BATCH_OWNER): raise ValueError('original adapter batch must still be live at waiter registration')
    if not owner_live(PREFIX_OWNER):
        completed_identity(inputs.replay.OUTPUT, inputs.PREFIX_LAUNCH, PREREQUISITES[0][4], sources)
    resources = native.hardware(); native.cohort_resources(resources, 1); verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, boot_id=BOOT,
        original_batch_owner=BATCH_OWNER, original_prefix_owner=PREFIX_OWNER,
        original_batch_launch_sha256=inputs.BATCH_LAUNCH, original_prefix_launch_sha256=inputs.PREFIX_LAUNCH,
        planned_case=list(native.CASE), waiter_pid=os.getpid(), hardware=resources,
        maximum_wait_s=WAIT_SECONDS, automatic_retry=False, source_changes_permitted=False,
        native_workers_while_waiting=0, native_workers_after_original_completion=1))
    print('REACHED_FRONTIER_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True); start = time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-start, **details))+'\n'); events.flush()
                print(status, details, flush=True)
            receipt = wait_for_inputs(sources, event); write_json(OUTPUT/'input_completion.json', receipt)
            verify(sources); resources = native.hardware(); native.cohort_resources(resources, 1); native.require_native_idle()
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('frontier output appeared outside waiter ownership')
            command = [sys.executable, native.SOURCE, '--prefix-result-sha256', receipt['prefix_result_sha256'],
                '--adapter-batch-result-sha256', receipt['adapter_batch_result_sha256']]
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('FRONTIER_NATIVE_CHILD_STARTED', pid=process.pid, command=command)
                while process.poll() is None:
                    event('FRONTIER_NATIVE_CHILD_LIVE', pid=process.pid); time.sleep(30)
            event('FRONTIER_NATIVE_CHILD_EXITED', pid=process.pid, returncode=process.returncode)
            if process.returncode != 0: raise ValueError('original frontier child failed; no retry')
            report = authenticate_completed(sources, receipt); write_json(OUTPUT/'native_completion.json', report)
        names = ('launch.json', 'events.jsonl', 'input_completion.json', 'native_stdout.log', 'native_completion.json')
        verify(sources); ids = {n: digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='REACHED_FRONTIER_MAZE03_NATIVE_WAIT_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, automatic_retry=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('REACHED_FRONTIER_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACHED_FRONTIER_NATIVE_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False, original_work_retained=True)); raise


if __name__ == '__main__': main()
