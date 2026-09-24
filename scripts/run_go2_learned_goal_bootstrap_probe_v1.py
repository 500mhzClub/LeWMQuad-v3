"""Exclusive two-episode online learned-goal development probe."""
import contextlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
import cv2
import torch

from scripts.geometry_progress_family_runtime_development import preflight, verify
from scripts.learned_goal_probe_episode_development import collect, artifacts
from scripts.learned_goal_probe_audit_development import audit
from scripts.learned_goal_probe_checkpoint_development import authenticate_study, load_model
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE / 'go2_learned_goal_bootstrap_probe_v1_attempt_001'
PROTOCOL = 'docs/go2_learned_goal_bootstrap_probe_v1_2026-09-08.md'
TRIALS = ('family_episode_052', 'family_episode_039')
SEEDS = ('scripts/run_go2_learned_goal_bootstrap_probe_v1.py',
    'lewm/tests/test_learned_goal_probe_development.py')


def worker(trial, launch_sha):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    terminal = dict(trial=trial, status='LEARNED_GOAL_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(trial+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha})
            launch = read_json(OUTPUT, 'launch.json')
            assert launch['planned_trials'] == list(TRIALS) and launch['output_root'] == str(OUTPUT)
            verify(launch)
            result = collect(trial, launch['source_sha256'][PROTOCOL], output=OUTPUT)
            bindings = {trial+'/'+n: digest(OUTPUT/trial/n) for n in artifacts(trial, result)}
            verify_artifacts(OUTPUT, bindings)
            report = audit(trial, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT)
            name = trial+'_audit.json'
            write_json(OUTPUT/name, report)
            bindings[name] = digest(OUTPUT/name)
            verify(launch)
            load_model()  # Reauthenticate final checkpoint bytes, with no optimization.
            verify_artifacts(OUTPUT, bindings)
            terminal.update(status='LEARNED_GOAL_COLLECTED_AND_RAW_AUDITED', artifact_sha256=bindings,
                goal=report['goal'], collection=result, selection_count=report['selection_count'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'])
        except Exception as error:
            import traceback
            traceback.print_exc()
            terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started, worker_log_sha256=digest(OUTPUT/(trial+'_worker.log')))
    write_json(OUTPUT/(trial+'_worker_terminal.json'), terminal)
    return terminal


def main():
    if not __debug__:
        raise ValueError('audit assertions required')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive attempt; no retry/resume')
    checkpoint = authenticate_study()
    load_model()
    launch = preflight(output=OUTPUT, protocol=PROTOCOL, seed_paths=SEEDS,
        planned_trials=list(TRIALS), workers=1, storage_bytes=4*1024**3)
    launch.update(checkpoint=checkpoint, experiment='online learned goal bootstrap V1',
        data_scope='two known mirrored development panels; no training or independent mazes',
        randomized_assignment_used_for_commands=False,
        concurrency_reason='serial fresh native workers for uncontended control-loop timing')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', launch)
    print('LEARNED_GOAL_PROBE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    records = []
    started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as executor:
                for trial in TRIALS:
                    future = executor.submit(worker, trial, digest(OUTPUT/'launch.json'))
                    while True:
                        monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, trial=trial, **hardware()))+'\n')
                        monitor.flush()
                        done, _ = wait([future], timeout=15)
                        if done:
                            break
                    record = future.result()
                    records.append(record)
                    print('LEARNED_GOAL_WORKER_TERMINAL', trial, record['status'], record.get('goal'), flush=True)
                    if record['status'] != 'LEARNED_GOAL_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('worker infrastructure or raw audit failure; remaining trials unlaunched')
        bindings = {n: h for r in records for n, h in r['artifact_sha256'].items()}
        for name in ('launch.json', 'resource_monitor.jsonl', *(t+s for t in TRIALS for s in ('_worker.log', '_worker_terminal.json'))):
            bindings[name] = digest(OUTPUT/name)
        verify(launch)
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='LEARNED_GOAL_BOOTSTRAP_PROBE_COMPLETE',
            trials=list(TRIALS), conditions=records, artifact_sha256=bindings,
            source_sha256=launch['source_sha256'], wall_s=time.perf_counter()-started,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and not r['hard_measurement_failed_frames'] for r in records),
            model_training=False, checkpoint_selection_performed=False, independent_maze_evaluation=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('LEARNED_GOAL_PROBE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_LEARNED_GOAL_PROBE_FAILURE',
            reason=repr(error), completed_trials=[r['trial'] for r in records]))
        raise


if __name__ == '__main__':
    main()
