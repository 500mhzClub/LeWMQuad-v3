"""Wait for the original queue, then execute exactly one frozen native pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts import run_go2_supervised_commitment_contact_maze01_pilot_v1 as native
from scripts.supervised_commitment_contact_queue_gate_development import (
    QUEUE, QUEUE_LAUNCH_SHA, verify_queue_completion, require_native_idle)
from scripts.run_go2_prepared_native_queue_v1 import ENVIRONMENT, PYTHON, competitors
from scripts.navigation_artifact_root_development import BASE, artifact_path, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit, merge_sources

OUTPUT = BASE/'go2_supervised_commitment_contact_native_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_supervised_commitment_contact_native_v1.py'
PROTOCOL = 'docs/go2_supervised_commitment_contact_native_wait_v1_2026-09-09.md'
TEST = 'lewm/tests/test_supervised_commitment_contact_native_wait_development.py'
OWNER_PID = 2551088
OWNER_START_TICKS = 130406252
BOOT_ID = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
WAIT_SECONDS = 48*3600
MAX_OUTPUT_BYTES = 64*1024**2
BINDINGS = {
    'scripts/supervised_commitment_contact_maze01_episode_development.py':'580885b34612a0d7caf6cfa21d58055e6381b08a927b98a35057149f263b48d9',
    'scripts/supervised_commitment_contact_maze01_audit_development.py':'b08c4c56c6dc5a5708027ce970a1925f1131dff84a3eed3a64731fcf1f70dd52',
    'scripts/supervised_commitment_contact_native_prefix_development.py':'b148e1d4fc157fb23e99c676b31f38448b574e64d1a975f9146ef1673748f1de',
    'scripts/supervised_commitment_contact_queue_gate_development.py':'e773705a9c6c96c86bcf424840c0517ced60863eee50c56897a4e9bf17d87bf6',
    native.SOURCE:'9cf3a4d103dce745d1909548e99316ba7ea95c4c6aa8ed989df83f1a7de75439',
    'lewm/tests/test_supervised_commitment_contact_native_development.py':'c4a2dae9b853dd5e6538f5c51412512f7bd02e5d71dcf8de9094e270207a8c2b',
    'lewm/tests/test_supervised_commitment_contact_native_launcher_development.py':'22f5b6048b2186e3c2cb7722693996ac8798db53837e54f1447e19a1c4cca799',
    native.PROTOCOL:'88e12b7784bb95056900af4346ce73ec33d65628b06ad6a23450efc8e5f8b2e7',
}


def owner_state(stat, boot):
    if boot != BOOT_ID: raise ValueError('original machine boot required; no waiter resume')
    if stat is None: return False
    fields = stat.rsplit(')', 1)[1].split()
    if int(fields[19]) != OWNER_START_TICKS: raise ValueError('queue PID reused; original owner identity required')
    return fields[0] not in ('Z', 'X')


def owner_live():
    try: stat = Path(f'/proc/{OWNER_PID}/stat').read_text()
    except FileNotFoundError: stat = None
    return owner_state(stat, Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def prepared_sources():
    verify_artifacts(QUEUE, {'launch.json':QUEUE_LAUNCH_SHA})
    verify_artifacts(native.PREFIX, {'result.json':native.PREFIX_SHA})
    queue = read_json(QUEUE, 'launch.json'); prefix = read_json(native.PREFIX, 'result.json')
    verify_artifacts(native.PREFIX, prefix['artifact_sha256']); native.admit_prefix(native.PREFIX, prefix)
    inherited = merge_sources(queue['source_sha256'], prefix['source_sha256'], BINDINGS); verify(inherited)
    sources = discover_sources((SOURCE, PROTOCOL, TEST,
        'docs/go2_supervised_commitment_contact_prefix_result_2026-09-09.md'), inherited)
    verify(sources); return sources


def wait_for_queue(sources, event, *, sleep=time.sleep, clock=time.monotonic):
    started = clock()
    while True:
        if (QUEUE/'failure.json').exists() or (QUEUE/'failure.json').is_symlink():
            raise ValueError('original queue terminal failure; no bypass or retry')
        live = owner_live(); busy = competitors()
        if not live:
            # A missing handle is terminal only after checking its exact output.
            sha = digest(artifact_path(QUEUE, 'result.json'))
            receipt = verify_queue_completion(sha, sources)
            if not busy:
                event('ORIGINAL_QUEUE_COMPLETED', queue_result_sha256=sha)
                return receipt
        if clock()-started >= WAIT_SECONDS: raise ValueError('bounded prelaunch wait expired; no native child started')
        event('WAITING', original_queue_live=live, competing_processes=busy)
        sleep(30)


def authenticate_native(sources):
    root = native.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink(): raise ValueError('native terminal failure retained')
    sha = digest(artifact_path(root, 'result.json')); result = read_json(root, 'result.json')
    verify_artifacts(root, result['artifact_sha256']); launch = read_json(root, 'launch.json')
    if (result['status'] != 'SUPERVISED_COMMITMENT_CONTACT_MAZE01_PILOT_V1_COMPLETE'
            or len(result['conditions']) != 1 or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(k) != v for k, v in result['source_sha256'].items())):
        raise ValueError('complete unchanged fixed native output required')
    native.verify_inputs(launch)
    record = result['conditions'][0]; name = native.CASE[0]
    if any(n not in result['artifact_sha256'] for n in ('launch.json', name+'_audit.json',
            name+'_prefix_comparison.json', name+'_worker_terminal.json', name+'_worker.log')):
        raise ValueError('all original completion and audit files must be bound')
    audit = read_json(root, name+'_audit.json'); prefix = read_json(root, name+'_prefix_comparison.json')
    terminal = read_json(root, name+'_worker_terminal.json')
    if (record != terminal or record['case'] != name or record['layout_index'] != 1
            or record['status'] != 'SUPERVISED_COMMITMENT_CONTACT_MAZE01_COLLECTED_AND_RAW_AUDITED'
            or 'failure' in record or record['prefix_comparison'] != prefix
            or record['model_state_unchanged'] is not True or record['model_state_sha256'] != native.SUPERVISED_STATE
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])
            or result['queue_result_sha256'] != launch['queue_result_sha256']):
        raise ValueError('complete original worker, audit and actual prefix required')
    require_raw_audit(record, audit, learned=True)
    for key in ('physical_and_public_prefix_exact', 'all_preintervention_requested_commands_exact',
            'complete_candidate_decisions_match_prospective_prefix', 'candidate_intervention_command_completed'):
        if prefix[key] is not True: raise ValueError('native physical prefix failed: '+key)
    if (prefix['common_prefix_frames'], prefix['first_intervention_frame'], prefix['physical_prefix_samples'],
            prefix['raw_model_forecast_comparisons']) != (4, 3, 900, 1):
        raise ValueError('fixed four-observation actual intervention required')
    if any(result['artifact_sha256'].get(k) != v for k, v in record['artifact_sha256'].items()):
        raise ValueError('all worker artifact bindings must remain in the final native result')
    verify(sources); verify_artifacts(root, result['artifact_sha256']|{'result.json':sha})
    return dict(result_sha256=sha, queue_result_sha256=launch['queue_result_sha256'],
        measured_round_trip_successes=int(record['verified_round_trip']), scientific_success_required=False,
        all_raw_audits_pass=True, actual_physical_prefix_pass=True, original_native_verifier_reexecuted=True)


def execute(sources, receipt, event, *, popen=subprocess.Popen):
    validate_root(native.OUTPUT, must_exist=False)
    if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('fresh native output required; no skip or retry')
    verify(sources)
    if verify_queue_completion(receipt['queue_result_sha256'], sources) != receipt:
        raise ValueError('queue completion changed before native launch')
    resources = hardware()
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 51*1024**3:
        raise ValueError('native resource admission failed')
    require_native_idle()
    environment = dict(os.environ, **ENVIRONMENT); environment.pop('PYTHONOPTIMIZE', None)
    command = [str(PYTHON), native.SOURCE, '--prefix-result-sha256', native.PREFIX_SHA,
        '--queue-result-sha256', receipt['queue_result_sha256']]
    with (OUTPUT/'native_stdout.log').open('x') as log:
        child = popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        event('NATIVE_CHILD_STARTED', pid=child.pid, command=command, hardware=resources)
        while True:
            try: code = child.wait(timeout=30); break
            except subprocess.TimeoutExpired: event('NATIVE_CHILD_LIVE', pid=child.pid)
    event('NATIVE_CHILD_EXITED', pid=child.pid, returncode=code, stdout_sha256=digest(OUTPUT/'native_stdout.log'))
    if code != 0: raise ValueError('native child failed; no automatic retry')
    report = authenticate_native(sources)
    if report['queue_result_sha256'] != receipt['queue_result_sha256']: raise ValueError('same waited original queue required')
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive one-shot waiter; no resume')
    sources = prepared_sources(); live = owner_live(); resources = hardware()
    if resources['memory_available_bytes'] < 40*1024**3 or resources['artifact_free_bytes'] < 51*1024**3+MAX_OUTPUT_BYTES:
        raise ValueError('8GiB waiting/verification plus32GiB native RAM and storage allowances required')
    if not live: raise ValueError('this initial waiter requires the observed original queue still live')
    if args.preflight_only:
        print('COMMITMENT_CONTACT_WAITER_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            original_queue_live=True, native_execution=False, output_created=False)), flush=True); return
    create_output(OUTPUT)
    launch = dict(source_sha256=sources, queue_launch_sha256=QUEUE_LAUNCH_SHA, queue_owner_pid=OWNER_PID,
        queue_owner_start_ticks=OWNER_START_TICKS, boot_id=BOOT_ID, fixed_native_runner=native.SOURCE,
        native_prefix_result_sha256=native.PREFIX_SHA, hardware=resources, maximum_prelaunch_wait_seconds=WAIT_SECONDS,
        output_allowance_bytes=MAX_OUTPUT_BYTES, native_scene_workers_while_waiting=0,
        native_scene_workers_after_original_queue=1, automatic_retry=False, navigation_qualified=False, goal_achieved=False)
    write_json(OUTPUT/'launch.json', launch); started = time.monotonic()
    print('COMMITMENT_CONTACT_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as stream:
            def event(status, **details):
                row = json.dumps(dict(status=status, elapsed_s=time.monotonic()-started, **details))+'\n'
                if stream.tell()+len(row.encode()) > MAX_OUTPUT_BYTES//2: raise ValueError('waiter event allowance exceeded')
                stream.write(row); stream.flush()
                if status not in ('WAITING', 'NATIVE_CHILD_LIVE'): print(status, json.dumps(details), flush=True)
            receipt = wait_for_queue(sources, event); write_json(OUTPUT/'queue_completion.json', receipt)
            report = execute(sources, receipt, event); write_json(OUTPUT/'native_completion.json', report)
        names = ('launch.json', 'events.jsonl', 'queue_completion.json', 'native_completion.json', 'native_stdout.log')
        bindings = {n:digest(OUTPUT/n) for n in names}; verify(sources); verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SUPERVISED_COMMITMENT_CONTACT_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, automatic_retry=False,
            navigation_qualified=False, goal_achieved=False))
        print('COMMITMENT_CONTACT_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMMITMENT_CONTACT_WAITER_FAILURE', reason=repr(error),
            original_queue_and_native_artifacts_retained=True, automatic_retry=False)); raise


if __name__ == '__main__': main()
