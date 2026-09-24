"""Four matched fresh native cases: current-frame versus persistent surfaces."""
import argparse
import contextlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait
import cv2
import torch

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.geometry_progress_family_runtime_development import preflight, verify
from scripts.surface_memory_goal_episode_development import collect, artifacts
from scripts.surface_memory_goal_audit_development import audit
from scripts.family_transition_model_admission_development import admit
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_surface_memory_goal_probe_v1_attempt_001'
MEMORY = BASE/'go2_joint_visual_surface_memory_v1_attempt_001'
MEMORY_IDS = {'launch.json': '21b63a30376c25fcc20dbfd86dc035eb4d411d576a20b10a9869b078d6b298c0',
    'result.json': 'd19b8254779c7aafb7ef4d0bee612a0d266da5236f1bea831f376d775c79dcfe'}
PROTOCOL = 'docs/go2_surface_memory_goal_probe_v1_2026-09-08.md'
TRIALS = ('family_episode_052', 'family_episode_039')
# Fixed counterbalanced order; no branch depends on scientific outcomes.
CASES = [(v+'_'+t, t, v == 'persistent') for t, variants in
    ((TRIALS[0], ('current_frame', 'persistent')), (TRIALS[1], ('persistent', 'current_frame')))
    for v in variants]


def worker(case, launch_sha):
    name, trial, persistent = case
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, trial=trial, persistent=persistent,
        status='SURFACE_MEMORY_GOAL_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha})
            launch = read_json(OUTPUT, 'launch.json'); verify(launch)
            assert launch['planned_cases'] == [list(c) for c in CASES] and tuple(case) in CASES
            assert launch['output_root'] == str(OUTPUT) and digest(URDF) == launch['robot_urdf_sha256']
            model, receipt = admit(launch['checkpoint']['study_result_sha256'])
            assert receipt == launch['checkpoint']
            before = state_digest(model.state_dict())
            geometry = ArticulatedCollisionGeometry(URDF)
            result = collect(trial, launch['source_sha256'][PROTOCOL], output=OUTPUT,
                model=model, geometry=geometry, persistent=persistent, episode_name=name)
            assert state_digest(model.state_dict()) == before
            bindings = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(trial, result)}
            verify_artifacts(OUTPUT, bindings)
            replay_model, _ = admit(launch['checkpoint']['study_result_sha256'])
            report = audit(trial, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                model=replay_model, robot_geometry=ArticulatedCollisionGeometry(URDF),
                persistent=persistent, episode_name=name)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name, report)
            bindings[audit_name] = digest(OUTPUT/audit_name)
            verify(launch); verify_artifacts(OUTPUT, bindings)
            assert digest(URDF) == launch['robot_urdf_sha256']
            terminal.update(status='SURFACE_MEMORY_GOAL_COLLECTED_AND_RAW_AUDITED',
                artifact_sha256=bindings, collection=result, goal=report['goal'],
                selection_count=report['selection_count'], hard_measurement_failed_frames=report['hard_measurement_failed_frames'])
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started, worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--study-result-sha256', required=True); args = parser.parse_args()
    if not __debug__: raise ValueError('audit assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new native attempt')
    verify_artifacts(MEMORY, MEMORY_IDS)
    memory = read_json(MEMORY, 'result.json')
    assert memory['status'] == 'JOINT_VISUAL_SURFACE_MEMORY_REPLAY_COMPLETE'
    verify_artifacts(MEMORY, memory['artifact_sha256'])
    _, checkpoint = admit(args.study_result_sha256)
    launch = preflight(output=OUTPUT, protocol=PROTOCOL,
        seed_paths=('scripts/run_go2_surface_memory_goal_probe_v1.py',
            'lewm/tests/test_surface_memory_goal_probe_development.py', *memory['source_sha256']),
        planned_trials=list(TRIALS), workers=1, storage_bytes=8*1024**3)
    for name, sha in memory['source_sha256'].items():
        assert launch['source_sha256'].get(name) == sha, name
    ArticulatedCollisionGeometry(URDF)
    launch.update(checkpoint=checkpoint, memory_sha256=MEMORY_IDS, planned_cases=CASES,
        robot_urdf_path=str(URDF), robot_urdf_sha256=digest(URDF),
        experiment='matched current-frame versus persistent surface conflict filter V1',
        data_scope='two known mirrored integration panels; not independent-maze evaluation',
        randomized_assignment_used_for_commands=False,
        concurrency_reason='one fresh worker per case, serial to measure complete-loop timing without contention')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('SURFACE_MEMORY_GOAL_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    records = []; started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as executor:
                for case in CASES:
                    future = executor.submit(worker, case, digest(OUTPUT/'launch.json'))
                    while True:
                        monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, case=case[0], **hardware()))+'\n'); monitor.flush()
                        done, _ = wait([future], timeout=15)
                        if done: break
                    record = future.result(); records.append(record)
                    print('SURFACE_MEMORY_GOAL_TERMINAL', case[0], record['status'], record.get('goal'), flush=True)
                    if record['status'] != 'SURFACE_MEMORY_GOAL_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('infrastructure/raw audit failure; later cases unlaunched')
        bindings = {n:h for r in records for n,h in r['artifact_sha256'].items()}
        for name in ('launch.json', 'resource_monitor.jsonl', *(c[0]+s for c in CASES for s in ('_worker.log', '_worker_terminal.json'))):
            bindings[name] = digest(OUTPUT/name)
        verify(launch); verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SURFACE_MEMORY_GOAL_PROBE_COMPLETE',
            cases=CASES, conditions=records, artifact_sha256=bindings, source_sha256=launch['source_sha256'],
            wall_s=time.perf_counter()-started, checkpoint=checkpoint,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and not r['hard_measurement_failed_frames'] for r in records),
            model_training=False, checkpoint_selection_performed=False, independent_maze_evaluation=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('SURFACE_MEMORY_GOAL_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SURFACE_MEMORY_GOAL_FAILURE',
            reason=repr(error), completed_cases=[r['case'] for r in records]))
        raise


if __name__ == '__main__': main()
