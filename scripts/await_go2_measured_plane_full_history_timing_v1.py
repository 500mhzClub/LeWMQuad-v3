"""Observe exact learned completion, run one full CPU pair and verify its end."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

import psutil

from scripts import replay_go2_measured_plane_single_pass_full_history_v1 as job

run = job.run
SOURCE = 'scripts/await_go2_measured_plane_full_history_timing_v1.py'
TEST = 'lewm/tests/test_measured_plane_full_history_timing_wait_development.py'
PROTOCOL = 'docs/go2_measured_plane_full_history_timing_wait_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_full_history_timing_wait_v1_attempt_001'


def wait_for_native(event):
    while True:
        try:
            live = run.owner_live(job.inputs.LEARNED_OWNER)
            run.verify_artifacts(job.native.OUTPUT, {'launch.json': job.inputs.LEARNED_LAUNCH_SHA})
        except (OSError, psutil.AccessDenied) as error:
            event('NATIVE_OWNER_OBSERVATION_RETRY', reason=repr(error)); time.sleep(30); continue
        if not live: break
        event('EXACT_LEARNED_NATIVE_OWNER_LIVE'); time.sleep(30)
    if (job.native.OUTPUT/'failure.json').exists() or (job.native.OUTPUT/'failure.json').is_symlink():
        raise ValueError('preserve failed native execution; do not dispatch a full-history replay')
    return run.digest(job.native.OUTPUT/'result.json')


def wait_for_resources(event):
    while True:
        try:
            hw = run.hardware()
        except (OSError, psutil.AccessDenied) as error:
            event('RESOURCE_OBSERVATION_RETRY', reason=repr(error)); time.sleep(30); continue
        if hw['memory_available_bytes'] >= 64*1024**3 and hw['artifact_free_bytes'] >= 43*1024**3:
            return hw
        event('FULL_REPLAY_RESOURCE_WAIT', memory_available_bytes=hw['memory_available_bytes'],
            artifact_free_bytes=hw['artifact_free_bytes']); time.sleep(30)


def completed_child(sources, native_sha):
    root = job.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed full-history replay without retry')
    launch = run.read_json(root, 'launch.json')
    if (run.owner_live(launch['owner'])
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('original replay owner must end on its recorded boot')
    sha = run.digest(root/'result.json'); result = run.read_json(root, 'result.json')
    if (result['status'] != 'MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(k) != v for k, v in result['source_sha256'].items())
            or result['complete_output_and_public_packets_rechecked'] is not True
            or result['original_raw_inputs_reauthenticated_before_and_after'] is not True
            or any(result[k] is not False for k in
                ('native_execution', 'navigation_qualified', 'real_time_qualified', 'goal_achieved'))
            or launch['actual_complete_population_required'] is not True
            or launch['automatic_retry'] is not False):
        raise ValueError('exact complete full-history comparison with diagnostic scope required')
    expected = {'launch.json', 'comparison.jsonl', 'state_checks.json', 'resource_monitor.jsonl', 'report.json'}
    if set(result['artifact_sha256']) != expected:
        raise ValueError('entire exact comparison, state, resource and report artifact set required')
    ids = result['artifact_sha256'] | {'result.json': sha}; run.verify_artifacts(root, ids)
    admission = job.admit(native_sha, sources)
    if (launch['input_admission'] != admission
            or launch['state_frames'] != job.comparison.state_frames(admission['frames'])
            or result['report'] != run.read_json(root, 'report.json')):
        raise ValueError('same complete original input population and saved report required')
    job.check_output(result['report'], admission)
    run.verify(sources); run.verify_artifacts(root, ids)
    report = result['report']; timing = report['timing']['all_observations']
    return dict(replay_result_sha256=sha, native_result_sha256=native_sha, complete_actual_native_history=True,
        frames=report['frames'], actual_model_forward_calls=report['actual_model_forward_calls'],
        complete_rows_states_and_public_packets_reconstructed=True, original_owner_ended=True,
        original_raw_artifacts_reauthenticated=True, controller_replay_reexecuted=False,
        observed_state_checks=report['observed_state_checks'], timing=report['timing'],
        all_observation_time_reduction_percent=100*(1-timing['candidate_total_s']/timing['baseline_total_s']),
        native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False)


def main(preflight=False):
    for root in (OUTPUT, job.OUTPUT):
        run.validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched full replay required')
    if not __debug__ or any(run.os.environ.get(k) != v for k, v in run.ENV.items()):
        raise ValueError('original deterministic CPU environment required')
    sources = job.prepared_sources((SOURCE, TEST, PROTOCOL))
    # The waiting process needs the ordinary native reserve. The full pair's
    # larger requirement is observed again immediately before dispatch.
    hardware = job.original.resources(); live = run.owner_live(job.inputs.LEARNED_OWNER)
    if preflight:
        print('MEASURED_PLANE_FULL_TIMING_WAITER_PREFLIGHT', len(sources), 'native_live', live, flush=True); return
    run.create_output(OUTPUT); process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        original_owner=job.inputs.LEARNED_OWNER, original_launch_sha256=job.inputs.LEARNED_LAUNCH_SHA,
        poll_seconds=30, full_replays_while_waiting=0, maximum_full_cpu_replays=1,
        native_execution=False, automatic_retry=False))
    print('MEASURED_PLANE_FULL_TIMING_WAITER_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), status=status, **details))+'\n')
                events.flush()
            native_sha = wait_for_native(event)
            job.admit(native_sha, sources); wait_for_resources(event); run.verify(sources)
            if job.OUTPUT.exists() or job.OUTPUT.is_symlink(): raise ValueError('full replay root appeared outside this waiter')
            command = [sys.executable, '-B', job.SOURCE, '--learned-result-sha256', native_sha]
            with (OUTPUT/'replay_stdout.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('FULL_HISTORY_REPLAY_STARTED', pid=child.pid, command=command)
                while child.poll() is None:
                    event('FULL_HISTORY_REPLAY_LIVE', pid=child.pid); time.sleep(30)
            event('FULL_HISTORY_REPLAY_EXITED', pid=child.pid, returncode=child.returncode)
            if child.returncode != 0: raise ValueError('full replay failed; preserve without retry')
            report = completed_child(sources, native_sha); run.write_json(OUTPUT/'completion.json', report)
        ids = {n: run.digest(OUTPUT/n) for n in ('launch.json', 'events.jsonl', 'replay_stdout.log', 'completion.json')}
        run.verify(sources); run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_FULL_HISTORY_TIMING_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, automatic_retry=False,
            native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_FULL_TIMING_WAITER_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_FULL_HISTORY_TIMING_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args().preflight)
