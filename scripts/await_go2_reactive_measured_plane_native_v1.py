"""Wait for the exact nominal comparison, then dispatch one reactive episode."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

import numpy as np
import psutil

from scripts import run_go2_reactive_measured_plane_maze02_v1 as native

run = native.run
SOURCE = 'scripts/await_go2_reactive_measured_plane_native_v1.py'
TEST = 'lewm/tests/test_reactive_measured_plane_native_wait_development.py'
PROTOCOL = 'docs/go2_reactive_measured_plane_native_wait_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_reactive_measured_plane_native_wait_v1_attempt_001'


def wait_for_nominal(event):
    while True:
        try:
            live = run.owner_live(native.inputs.NOMINAL_WAIT_OWNER)
            run.verify_artifacts(native.inputs.nominal_wait.OUTPUT,
                {'launch.json': native.inputs.NOMINAL_WAIT_LAUNCH_SHA})
        except (OSError, psutil.AccessDenied) as error:
            event('NOMINAL_OWNER_OBSERVATION_RETRY', reason=repr(error)); time.sleep(30); continue
        if not live: break
        event('EXACT_NOMINAL_WAITER_LIVE'); time.sleep(30)
    root = native.inputs.nominal_wait.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed nominal comparison; no automatic reactive dispatch')
    return run.digest(root/'result.json')


def completed_child(sources, wait_sha):
    root = native.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed reactive native attempt without retry')
    launch = run.read_json(root, 'launch.json')
    if (run.owner_live(launch['owner'])
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('exact reactive parent must have ended on this boot')
    sha = run.digest(root/'result.json'); result = run.read_json(root, 'result.json')
    admission = launch['input_admission']
    if (result['status'] != 'REACTIVE_MEASURED_PLANE_MAZE02_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(k) != v for k, v in result['source_sha256'].items())
            or result['nominal_wait_result_sha256'] != wait_sha
            or admission['nominal_wait_result_sha256'] != wait_sha
            or result['learned_result_sha256'] != admission['learned_result_sha256']
            or result['nominal_result_sha256'] != admission['nominal_result_sha256']
            or result['reactive_prefix_result_sha256'] != native.prefix.RESULT_SHA
            or len(result['conditions']) != 1 or result['automatic_retry'] is not False
            or result['high_level_world_model_loaded'] is not False
            or result['fully_nonpredictive_controller'] is not True
            or result['reactive_is_whole_method_comparison'] is not True
            or result['isolated_prediction_ranking_ablation'] is not False):
        raise ValueError('complete exact reactive result tied to the original comparison required')
    ids = result['artifact_sha256'] | {'result.json': sha}; run.verify_artifacts(root, ids)
    record = result['conditions'][0]; name = native.CASE[0]
    required = [name+'/'+n for n in native.pipeline.artifacts(2, record['collection'])]
    required += [name+s for s in ('_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json')]
    required += ['launch.json', 'resource_monitor.jsonl']
    if any(n not in ids for n in required): raise ValueError('complete reactive raw artifact roster required')
    if ids['launch.json'] != run.digest(root/'launch.json'):
        raise ValueError('complete result must bind its actual reactive launch')
    if (record != run.read_json(root, name+'_worker_terminal.json')
            or record['collection'] != run.read_json(root/name, 'result.json')
            or record['readout'] != run.read_json(root, name+'_readout.json')
            or record['prefix_comparison'] != run.read_json(root, name+'_prefix_comparison.json')
            or record['worker_log_sha256'] != ids[name+'_worker.log']
            or any(ids.get(k) != v for k, v in record['artifact_sha256'].items())
            or result['measured_round_trip_successes'] != int(record['verified_round_trip'])):
        raise ValueError('same complete reactive worker receipts and outcome accounting required')
    audit = run.read_json(root, name+'_audit.json'); native.require_worker(record, audit)
    with np.load(root/name/'physics_trace.npz', allow_pickle=False) as raw:
        if native.original.case_readout(audit, record['collection'], raw['physics_contact']) != record['readout']:
            raise ValueError('actual reactive contact, progress and timing readout must reconstruct')
    if native.prefix_result(record['collection'], admission['prefix_report']) != record['prefix_comparison']:
        raise ValueError('actual learned-versus-reactive physical prefix must reconstruct')
    native.verify_inputs(launch); run.verify(sources); run.verify_artifacts(root, ids)
    return dict(reactive_result_sha256=sha, nominal_wait_result_sha256=wait_sha,
        learned_result_sha256=admission['learned_result_sha256'], nominal_result_sha256=admission['nominal_result_sha256'],
        complete_native_worker_and_artifact_roster_verified=True, physical_prefix_accounting_reconstructed=True,
        actual_physical_prefix_reconstructed=record['prefix_comparison']['actual_paired_execution_compared'],
        scientific_success_required=False, raw_controller_audit_reexecuted=False,
        high_level_world_model_loaded=False, reactive_is_whole_method_comparison=True,
        reactive_measured_round_trip_successes=result['measured_round_trip_successes'])


def main(preflight=False):
    for root in (OUTPUT, native.OUTPUT):
        run.validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched reactive child required')
    if not __debug__ or any(run.os.environ.get(k) != v for k, v in run.ENV.items()):
        raise ValueError('original deterministic CPU environment required')
    sources = native.inputs.prepared_sources((SOURCE, TEST, PROTOCOL, native.SOURCE, native.PROTOCOL, *native.TESTS))
    hardware = native.resources(); live = run.owner_live(native.inputs.NOMINAL_WAIT_OWNER)
    if preflight:
        print('REACTIVE_MEASURED_PLANE_WAITER_PREFLIGHT', len(sources), 'nominal_waiter_live', live, flush=True); return
    run.create_output(OUTPUT); process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        original_owner=native.inputs.NOMINAL_WAIT_OWNER,
        original_launch_sha256=native.inputs.NOMINAL_WAIT_LAUNCH_SHA,
        planned_case=list(native.CASE), poll_seconds=30, automatic_retry=False,
        native_workers_while_waiting=0, native_workers_after_original_completion=1))
    print('REACTIVE_MEASURED_PLANE_WAITER_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), status=status, **details))+'\n')
                events.flush()
            wait_sha = wait_for_nominal(event)
            native.inputs.admit(wait_sha, sources); native.resources(); native.wait_for_idle(); run.verify(sources)
            if native.OUTPUT.exists() or native.OUTPUT.is_symlink(): raise ValueError('reactive root appeared outside this waiter')
            command = [sys.executable, '-B', native.SOURCE, '--nominal-wait-result-sha256', wait_sha]
            with (OUTPUT/'native_stdout.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('REACTIVE_NATIVE_CHILD_STARTED', pid=child.pid, command=command)
                while child.poll() is None:
                    event('REACTIVE_NATIVE_CHILD_LIVE', pid=child.pid); time.sleep(30)
            event('REACTIVE_NATIVE_CHILD_EXITED', pid=child.pid, returncode=child.returncode)
            if child.returncode != 0: raise ValueError('reactive child failed; preserve the attempt without retry')
            report = completed_child(sources, wait_sha); run.write_json(OUTPUT/'native_completion.json', report)
        ids = {n: run.digest(OUTPUT/n) for n in ('launch.json', 'events.jsonl', 'native_stdout.log', 'native_completion.json')}
        run.verify(sources); run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='REACTIVE_MEASURED_PLANE_NATIVE_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, automatic_retry=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('REACTIVE_MEASURED_PLANE_WAITER_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_MEASURED_PLANE_NATIVE_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args().preflight)
