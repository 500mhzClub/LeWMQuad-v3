"""Start one CPU raw replay after the exact original native worker completes."""
import json
import os
import subprocess
import sys
import time
import psutil
from scripts import replay_go2_hold_reorientation_maze02_prefix_v1 as replay
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.await_go2_reached_frontier_maze03_native_v1 import BATCH_OWNER
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT = BASE/'go2_hold_reorientation_raw_prefix_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_hold_reorientation_raw_prefix_v1.py'
PROTOCOL = 'docs/go2_hold_reorientation_raw_prefix_wait_v1_2026-09-10.md'
TEST = 'lewm/tests/test_hold_reorientation_raw_wait_development.py'
WAIT_SECONDS = 48*3600
WORKER = dict(pid=2662101, created=1789031235.91, command=[
    '/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',
    '-B', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=9, pipe_handle=13)',
    '--multiprocessing-fork'])


def completed_worker_identity():
    root = replay.original.OUTPUT; name = replay.CASE[0]; terminal = name+'_worker_terminal.json'
    verify_artifacts(root, {'launch.json': replay.LAUNCH_SHA})
    if not (root/terminal).is_file(): raise ValueError('original worker ended without terminal evidence; no retry')
    sha = digest(root/terminal); record = read_json(root, terminal)
    report_name = name+'_audit.json'
    if report_name not in record['artifact_sha256']:
        raise ValueError('original worker has no bound raw audit')
    verify_artifacts(root, {terminal: sha, report_name: record['artifact_sha256'][report_name]})
    replay.require_case(replay.CASE, record, read_json(root, report_name))
    return sha


def wait_for_worker(event, *, sleep=time.sleep, clock=time.monotonic):
    start = clock()
    while owner_live(WORKER):
        if not owner_live(BATCH_OWNER): raise ValueError('original worker lost its recorded parent owner')
        if clock()-start >= WAIT_SECONDS:
            raise ValueError('original-worker wait expired; no process restarted')
        event('WAITING_FOR_ORIGINAL_ADAPTER_JEPA_WORKER', original_worker_pid=WORKER['pid'])
        sleep(30)
    return completed_worker_identity()


def authenticate_replay(worker_sha, sources):
    root = replay.OUTPUT
    if (root/'failure.json').exists(): raise ValueError('raw replay failure retained; no retry')
    sha = digest(root/'result.json'); result = read_json(root, 'result.json')
    verify_artifacts(root, result['artifact_sha256'] | {'result.json': sha})
    launch = read_json(root, 'launch.json'); report = result['report']
    if (result['status'] != 'HOLD_REORIENTATION_MAZE02_RAW_PREFIX_V1_COMPLETE'
            or result['original_worker_terminal_sha256'] != worker_sha
            or launch['input_admission']['original_worker_terminal_sha256'] != worker_sha
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or launch['saved_selection_result_sha256'] != replay.SAVED_SHA
            or report['frames'] != 406 or report['first_changed_command_frame'] != 405
            or report['raw_model_forecast_comparisons'] != 403
            or report['model_state_sha256'] != replay.MODEL_SHA
            or report['candidate_requested_command'] != [0., 0., .45]
            or report['original_requested_command'] != [0., 0., 0.]
            or report['no_observation_after_changed_request_consumed'] is not True
            or result['native_execution'] is not False):
        raise ValueError('complete exact original-worker raw replay required')
    verify(sources)
    verify_artifacts(replay.original.OUTPUT, {replay.CASE[0]+'_worker_terminal.json': worker_sha})
    return dict(raw_replay_result_sha256=sha, frames=406, first_changed_command_frame=405,
        native_execution=False, original_worker_terminal_sha256=worker_sha)


def main():
    for root in (OUTPUT, replay.OUTPUT):
        validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched raw replay required')
    saved_result, _, _ = replay.saved_inputs()
    sources = discover_sources((SOURCE, PROTOCOL, TEST, replay.SOURCE, replay.PROTOCOL, replay.TEST,
        'lewm/tests/test_hold_reorientation_raw_runner_development.py'), saved_result['source_sha256'])
    verify(sources); verify_artifacts(replay.original.OUTPUT, {'launch.json': replay.LAUNCH_SHA})
    if owner_live(WORKER):
        if not owner_live(BATCH_OWNER) or psutil.Process(WORKER['pid']).ppid() != BATCH_OWNER['pid']:
            raise ValueError('exact live original worker-parent ownership required')
    else: completed_worker_identity()
    resources = replay.hardware(); replay.resources_for(resources)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, original_worker_owner=WORKER,
        original_parent_owner=BATCH_OWNER, boot_id=BOOT, waiter_pid=os.getpid(),
        original_launch_sha256=replay.LAUNCH_SHA, hardware=resources, maximum_wait_s=WAIT_SECONDS,
        native_execution=False, automatic_retry=False, source_changes_permitted=False))
    print('HOLD_REORIENTATION_RAW_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-start, **details))+'\n')
                events.flush(); print(status, details, flush=True)
            worker_sha = wait_for_worker(event)
            write_json(OUTPUT/'input_completion.json', {'original_worker_terminal_sha256': worker_sha})
            verify(sources); replay.resources_for(replay.hardware())
            if replay.OUTPUT.exists() or replay.OUTPUT.is_symlink(): raise ValueError('raw replay output appeared outside waiter ownership')
            command = [sys.executable, replay.SOURCE, '--original-worker-terminal-sha256', worker_sha]
            with (OUTPUT/'replay_stdout.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('HOLD_REORIENTATION_RAW_CHILD_STARTED', pid=child.pid, command=command)
                while child.poll() is None:
                    event('HOLD_REORIENTATION_RAW_CHILD_LIVE', pid=child.pid); time.sleep(30)
            event('HOLD_REORIENTATION_RAW_CHILD_EXITED', pid=child.pid, returncode=child.returncode)
            if child.returncode != 0: raise ValueError('original raw replay child failed; no retry')
            report = authenticate_replay(worker_sha, sources)
            write_json(OUTPUT/'replay_completion.json', report)
        names = ('launch.json', 'events.jsonl', 'input_completion.json', 'replay_stdout.log', 'replay_completion.json')
        verify(sources); ids = {n: digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='HOLD_REORIENTATION_RAW_PREFIX_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, automatic_retry=False,
            native_execution=False, goal_achieved=False))
        print('HOLD_REORIENTATION_RAW_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_HOLD_REORIENTATION_RAW_WAITER_FAILURE',
            reason=repr(error), original_work_retained=True, automatic_retry=False)); raise


if __name__ == '__main__': main()
