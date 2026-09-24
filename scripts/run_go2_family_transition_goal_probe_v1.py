"""Fresh native goal probe of fixed family-JEPA model and corner RGB-D observer."""
import argparse
import contextlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait
import cv2
import torch

from lewm.pulse_timed_training_runner_development import state_digest
from scripts.geometry_progress_family_runtime_development import preflight, verify
from scripts.family_transition_goal_episode_development import collect, artifacts
from scripts.family_transition_goal_audit_development import audit
from scripts.family_transition_model_admission_development import admit
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_family_transition_goal_probe_v1_attempt_001'
OBSERVER = BASE/'go2_corner_support_observer_replay_v1_attempt_001'
OBSERVER_IDS = {'launch.json': '162ad020a1de3781259dafb284e9724174363985e4bcc75ea0595082faef163c',
    'result.json': '97517b61fc33d2cdc484d96f0f47aeef0e730defb866444b08cc0ac41cb97375'}
PROTOCOL = 'docs/go2_family_transition_goal_probe_v1_2026-09-08.md'
TRIALS = ('family_episode_052', 'family_episode_039')


def worker(trial, launch_sha):
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(trial=trial, status='FAMILY_TRANSITION_GOAL_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(trial+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha})
            launch = read_json(OUTPUT, 'launch.json'); verify(launch)
            if launch['planned_trials'] != list(TRIALS) or launch['output_root'] != str(OUTPUT):
                raise ValueError('exact prospective native assignment required')
            model, receipt = admit(launch['checkpoint']['study_result_sha256'])
            if receipt != launch['checkpoint']: raise ValueError('exact prebound final model required')
            before = state_digest(model.state_dict())
            result = collect(trial, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model)
            if state_digest(model.state_dict()) != before:
                raise ValueError('native inference changed checkpoint state')
            bindings = {trial+'/'+n: digest(OUTPUT/trial/n) for n in artifacts(trial, result)}
            verify_artifacts(OUTPUT, bindings)
            replay_model, _ = admit(launch['checkpoint']['study_result_sha256'])
            report = audit(trial, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT, model=replay_model)
            name = trial+'_audit.json'; write_json(OUTPUT/name, report); bindings[name] = digest(OUTPUT/name)
            verify(launch); verify_artifacts(OUTPUT, bindings)
            terminal.update(status='FAMILY_TRANSITION_GOAL_COLLECTED_AND_RAW_AUDITED',
                artifact_sha256=bindings, collection=result, goal=report['goal'],
                selection_count=report['selection_count'], hard_measurement_failed_frames=report['hard_measurement_failed_frames'])
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started, worker_log_sha256=digest(OUTPUT/(trial+'_worker.log')))
    write_json(OUTPUT/(trial+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--study-result-sha256', required=True); args = parser.parse_args()
    if not __debug__: raise ValueError('audit assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new native attempt; no retry/resume')
    verify_artifacts(OBSERVER, OBSERVER_IDS)
    observer = read_json(OBSERVER, 'result.json')
    if observer['candidate_eligible_for_separate_native_probe'] is not True:
        raise ValueError('complete passing corner-observer replay required')
    verify_artifacts(OBSERVER, observer['artifact_sha256'])
    _, checkpoint = admit(args.study_result_sha256)
    launch = preflight(output=OUTPUT, protocol=PROTOCOL,
        seed_paths=('scripts/run_go2_family_transition_goal_probe_v1.py', 'lewm/tests/test_family_transition_goal_probe_development.py',
            *observer['source_sha256']),
        planned_trials=list(TRIALS), workers=1, storage_bytes=4*1024**3)
    for name, sha in observer['source_sha256'].items():
        if launch['source_sha256'].get(name) != sha:
            raise ValueError('native observer must retain all replay-bound sources')
    launch.update(checkpoint=checkpoint, observer_sha256=OBSERVER_IDS,
        experiment='family transition JEPA plus corner observer native goal probe V1',
        data_scope='two known mirrored integration panels; not independent-maze evaluation',
        randomized_assignment_used_for_commands=False, concurrency_reason='serial fresh workers for complete-loop timing')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('FAMILY_TRANSITION_GOAL_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    records = []; started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as executor:
                for trial in TRIALS:
                    future = executor.submit(worker, trial, digest(OUTPUT/'launch.json'))
                    while True:
                        monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, trial=trial, **hardware()))+'\n'); monitor.flush()
                        done, _ = wait([future], timeout=15)
                        if done: break
                    record = future.result(); records.append(record)
                    print('FAMILY_TRANSITION_GOAL_TERMINAL', trial, record['status'], record.get('goal'), flush=True)
                    if record['status'] != 'FAMILY_TRANSITION_GOAL_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('infrastructure/raw audit failure; later native cases unlaunched')
        bindings = {n: h for r in records for n, h in r['artifact_sha256'].items()}
        for name in ('launch.json', 'resource_monitor.jsonl', *(t+s for t in TRIALS for s in ('_worker.log', '_worker_terminal.json'))):
            bindings[name] = digest(OUTPUT/name)
        verify(launch); verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='FAMILY_TRANSITION_GOAL_PROBE_COMPLETE',
            trials=list(TRIALS), conditions=records, artifact_sha256=bindings, source_sha256=launch['source_sha256'],
            wall_s=time.perf_counter()-started, checkpoint=checkpoint,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and not r['hard_measurement_failed_frames'] for r in records),
            model_training=False, checkpoint_selection_performed=False, independent_maze_evaluation=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('FAMILY_TRANSITION_GOAL_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FAMILY_TRANSITION_GOAL_FAILURE',
            reason=repr(error), completed_trials=[r['trial'] for r in records]))
        raise


if __name__ == '__main__': main()
