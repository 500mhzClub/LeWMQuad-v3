"""Queue exactly one budget diagnostic after the original four native stages."""
import json
import os
import re
import subprocess
import sys
import time
from types import FunctionType

import numpy as np

from scripts import run_go2_no_rgb_direct_extended_budget_maze02_pilot_v1 as native
from scripts import await_go2_no_rgb_jepa_direct_flow_maze02_native_v1 as original
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/await_go2_no_rgb_direct_extended_budget_maze02_native_v1.py'
TEST = 'lewm/tests/test_no_rgb_direct_extended_budget_native_wait_development.py'
PROTOCOL = 'docs/go2_no_rgb_direct_extended_budget_maze02_native_wait_v1_2026-09-11.md'
PREPARATION = 'docs/go2_no_rgb_direct_extended_budget_native_preparation_2026-09-11.json'
PREPARATION_SHA = '24592143a89b2ee9c04abaf533791bf99eb831e0ee949a11b92588791328fc94'
OUTPUT = BASE/'go2_no_rgb_direct_extended_budget_maze02_native_wait_v1_attempt_001'
WAIT_SECONDS = 48*3600
PREREQUISITES = tuple(spec for spec in original.PREREQUISITES if spec[0] != 'prefix') + (
    ('tracking', native.inputs.queue.OWNER, original.OUTPUT, native.inputs.queue.LAUNCH_SHA,
        'NO_RGB_JEPA_DIRECT_FLOW_MAZE02_NATIVE_WAIT_V1_COMPLETE'),)


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    prep = json.loads((ROOT/PREPARATION).read_text())
    if prep['status'] != 'NO_RGB_DIRECT_EXTENDED_BUDGET_NATIVE_SOURCE_TESTED_PREFLIGHT_PASSED':
        raise ValueError('tested exact extended-budget native launcher required')
    verify(prep['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), prep['source_sha256'])
    verify(sources)
    return sources


# The original bounded loop checks every exact PID/create-time/argv owner,
# preserves completed identities across polls and never restarts a process.
# Its completion helper's special replay-prefix branch is unreachable with
# this frozen batch/frontier/hold/contact/tracking prerequisite roster.
completion_identity = original.completion_identity
wait_for_inputs = FunctionType(original.wait_for_inputs.__code__, original.wait_for_inputs.__globals__ | dict(
    PREREQUISITES=PREREQUISITES, WAIT_SECONDS=WAIT_SECONDS, completion_identity=completion_identity),
    original.wait_for_inputs.__name__, original.wait_for_inputs.__defaults__)
wait_for_inputs.__kwdefaults__ = original.wait_for_inputs.__kwdefaults__


def command_for(ids):
    if (type(ids) is not dict or set(ids) != {s[0] for s in PREREQUISITES}
            or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in ids.values())):
        raise ValueError('exact completed batch and all four waiter SHA-256 identities required')
    command = [sys.executable, '-B', native.SOURCE]
    for key, option in [('batch', 'adapter-batch'), ('frontier', 'frontier-wait'), ('hold', 'hold-wait'),
                        ('contact', 'contact-wait'), ('tracking', 'tracking-wait')]:
        command.extend(['--'+option+'-result-sha256', ids[key]])
    return command


def authenticate_completed(sources, ids):
    root = native.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original budget diagnostic failure must be preserved')
    sha = digest(root/'result.json'); result = read_json(root, 'result.json')
    bindings = result['artifact_sha256'] | {'result.json':sha}; verify_artifacts(root, bindings)
    launch = read_json(root, 'launch.json'); admission = launch['input_admission']
    if (result['status'] != 'NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_PILOT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())
            or len(result['conditions']) != 1 or result['automatic_retry'] is not False
            or admission['batch_result_sha256'] != ids['batch']
            or admission['extended_queue_completion']['tracking_wait_result_sha256'] != ids['tracking']
            or admission['extended_queue_completion']['original_queue_admission']['ordered_waiter_result_sha256']
                != {k:ids[k] for k in ('frontier', 'hold', 'contact')}):
        raise ValueError('complete budget diagnostic tied to all original queued completions required')
    native.verify_inputs(launch)
    name = native.CASE[0]; record = result['conditions'][0]
    required = [name+'/'+n for n in native.pipeline.artifacts(2, record['collection'])]
    required += [name+s for s in ('_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json')]
    if any(n not in bindings for n in required): raise ValueError('complete raw collection and worker artifact roster required')
    if (record != read_json(root, name+'_worker_terminal.json')
            or record['collection'] != read_json(root, name+'/result.json')
            or record['prefix_comparison'] != read_json(root, name+'_prefix_comparison.json')
            or record['readout'] != read_json(root, name+'_readout.json')
            or record['worker_log_sha256'] != bindings[name+'_worker.log']
            or any(bindings.get(n) != h for n, h in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])
            or result['budget_only_preboundary_execution_supported']
                != record['prefix_comparison']['budget_only_preboundary_execution_supported']):
        raise ValueError('complete actual worker receipts and outcome accounting required')
    audit = read_json(root, name+'_audit.json'); native.require_worker(record, audit)
    with np.load(root/name/'physics_trace.npz', allow_pickle=False) as raw:
        if native.case_readout(audit, record['collection'], raw['physics_contact']) != record['readout']:
            raise ValueError('actual physical/contact/timing readout must reconstruct')
    if native.prefix_result(record['collection'], launch, record['artifact_sha256']) != record['prefix_comparison']:
        raise ValueError('actual budget-only prefix finding must reconstruct, including negative outcomes')
    verify(sources); verify_artifacts(root, bindings)
    return dict(native_result_sha256=sha, measured_round_trip_successes=result['measured_round_trip_successes'],
        budget_only_preboundary_execution_supported=result['budget_only_preboundary_execution_supported'],
        complete_native_worker_and_artifact_roster_verified=True, actual_prefix_finding_reconstructed=True,
        scientific_success_required=False, final_independent_population_policy_review_performed=False)


def main():
    for root in (OUTPUT, native.OUTPUT):
        validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched budget child required')
    sources = prepared_sources()
    for spec in PREREQUISITES:
        if not owner_live(spec[1]): completion_identity(spec, sources)
    resources = native.hardware(); native.resource_admission(resources); verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, boot_id=BOOT, waiter_pid=os.getpid(),
        original_owners={s[0]:s[1] for s in PREREQUISITES}, planned_case=list(native.CASE),
        hardware=resources, maximum_wait_s=WAIT_SECONDS, automatic_retry=False, source_changes_permitted=False,
        native_workers_while_waiting=0, native_workers_after_original_completion=1,
        light_completion_identity_checks_while_waiting=True, full_input_admission_deferred_to_native_launcher=True))
    print('EXTENDED_BUDGET_NATIVE_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-start, **details))+'\n')
                events.flush(); print(status, details, flush=True)
            ids = wait_for_inputs(sources, event); write_json(OUTPUT/'input_completion.json', ids)
            native.inputs.queue.owners_ended(); native.inputs.queue.original.owners_ended()
            verify(sources); native.resource_admission(native.hardware()); native.require_native_idle()
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('native root appeared outside this owner')
            command = command_for(ids)
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('EXTENDED_BUDGET_NATIVE_CHILD_STARTED', pid=child.pid, command=command)
                while child.poll() is None:
                    event('EXTENDED_BUDGET_NATIVE_CHILD_LIVE', pid=child.pid); time.sleep(30)
            event('EXTENDED_BUDGET_NATIVE_CHILD_EXITED', pid=child.pid, returncode=child.returncode)
            if child.returncode != 0: raise ValueError('original budget child failed; no retry')
            report = authenticate_completed(sources, ids); write_json(OUTPUT/'native_completion.json', report)
        names = ('launch.json', 'events.jsonl', 'input_completion.json', 'native_stdout.log', 'native_completion.json')
        verify(sources); bindings = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, automatic_retry=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('EXTENDED_BUDGET_NATIVE_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_EXTENDED_BUDGET_NATIVE_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False, original_work_retained=True))
        raise


if __name__ == '__main__': main()
