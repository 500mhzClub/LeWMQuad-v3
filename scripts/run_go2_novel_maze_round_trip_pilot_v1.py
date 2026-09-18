"""First fixed prospective maze execution with a learned outbound/return policy."""
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission, LAYOUT_COUNT
from lewm.novel_maze_round_trip_contract_development import (
    NAVIGATION_TICKS, RESERVE_BYTES, COLLECTION_ALLOWANCE_BYTES, PERSISTENCE_HEADROOM_BYTES)
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_round_trip_episode_development import collect, artifacts
from scripts.novel_maze_round_trip_audit_development import audit
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_round_trip_controller_compatibility_v1 import OUTPUT as INTEGRATION
from scripts.run_go2_causal_residual_final_goal_probe_v1 import CORRECTION, FITS
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_novel_maze_round_trip_pilot_v1_attempt_001'
PROTOCOL = 'docs/go2_novel_maze_round_trip_pilot_v1_2026-09-08.md'
INTEGRATION_SHA = '5c9d1d60e789329ccb7d2a035388e9fa315f7d702aa7d7dfa438c3b49164792f'
CASE = ('full_jepa_novel_maze_00', 0, 'full', 'jepa', 'seed_2026091001_full_jepa')


def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, layout_index=index, model_name=model_name,
        status='NOVEL_MAZE_ROUND_TRIP_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json'); verify(launch)
            assert launch['planned_case'] == list(CASE) and launch['output_root'] == str(OUTPUT)
            assert launch['scene_specification'] == specification(index) and launch['public_mission'] == public_mission(index)
            assert digest(URDF) == launch['robot_urdf_sha256']
            model, c, v = load_assigned(launch['correction_admission'], model_name); assert (c, v) == (condition, variant)
            before = state_digest(model.state_dict())
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            assert state_digest(model.state_dict()) == before
            bindings = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(index, result)}
            verify_artifacts(OUTPUT, bindings)
            replay_model, c, v = load_assigned(launch['correction_admission'], model_name)
            assert (c, v) == (condition, variant)
            report = audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT, model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name, report); bindings[audit_name] = digest(OUTPUT/audit_name)
            verify(launch); verify_artifacts(OUTPUT, bindings)
            verify_artifacts(FITS, launch['correction_admission']['base_admission']['fit_artifact_sha256'])
            verify_artifacts(CORRECTION, launch['correction_admission']['correction_artifact_sha256'])
            terminal.update(status='NOVEL_MAZE_ROUND_TRIP_COLLECTED_AND_RAW_AUDITED', artifact_sha256=bindings,
                collection=result, verified_round_trip=report['verified_round_trip'],
                native_evaluation=report['native_evaluation'], strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'], model_state_unchanged=True)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), terminal)
    return terminal


def main():
    if not __debug__: raise ValueError('audit assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive first-maze execution; no retry/resume')
    verify_artifacts(INTEGRATION, {'result.json': INTEGRATION_SHA}); integration = read_json(INTEGRATION, 'result.json')
    assert integration['status'] == 'ROUND_TRIP_CONTROLLER_COMPATIBILITY_COMPLETE'
    assert len(integration['conditions']) == 2 and sum(r['frames'] for r in integration['conditions']) == 311
    assert all(r['all_original_decision_fields_except_controller_identity_exact']
        and r['actual_recorded_commands_exact'] and r['model_state_unchanged'] for r in integration['conditions'])
    ids = {'result.json': INTEGRATION_SHA, **integration['artifact_sha256']}; verify_artifacts(INTEGRATION, ids)
    old = read_json(INTEGRATION, 'launch.json'); verify(old)
    admission = old['correction_admission']
    verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_novel_maze_round_trip_pilot_v1.py',
        'docs/go2_round_trip_controller_compatibility_result_2026-09-08.md',
        'docs/go2_novel_maze_round_trip_scene_source_2026-09-08.md',
        'docs/go2_novel_maze_round_trip_evaluator_source_2026-09-08.md',
        'lewm/tests/test_novel_maze_round_trip_scene_development.py',
        'lewm/tests/test_novel_maze_round_trip_evaluation_development.py',
        'lewm/tests/test_novel_maze_round_trip_native_scope_development.py',
        'lewm/tests/test_maze_decision_stream_development.py'), integration['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 32*1024**3:
        raise ValueError('32 GiB available RAM required')
    if resources['artifact_free_bytes'] < RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES:
        raise ValueError('collection/persistence allowance above original reserve required')
    launch = {k: old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), planned_case=CASE,
        scene_specification=specification(CASE[1]), public_mission=public_mission(CASE[1]),
        declared_layout_sequence=list(range(LAYOUT_COUNT)), first_maze_only=True,
        not_yet_launched_matched_direct_or_other_layouts=True,
        correction_admission=admission, integration_artifact_sha256=ids,
        robot_urdf_path=str(URDF), robot_urdf_sha256=digest(URDF), hardware=resources,
        navigation_ticks=NAVIGATION_TICKS, shared_outbound_return_budget=True,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        minimum_free_bytes=RESERVE_BYTES, planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,
        memory_admission_bytes=32*1024**3, os_resource_limits_enforced=False,
        concurrency_reason='first longer-maze workload is unmeasured; one native scene fits available storage allowances',
        native_execution=True, model_training=False, checkpoint_selection_performed=False,
        physics_paused_during_compute=True, real_time_qualified=False,
        data_scope='first fixed prospective development maze; source-disjoint from the 32 reviewed prior maze generators',
        prior_failed_outcomes_unchanged=True, navigation_qualified=False, goal_achieved=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    launch_sha = digest(OUTPUT/'launch.json'); started = time.perf_counter()
    print('NOVEL_MAZE_ROUND_TRIP_LAUNCHED', launch_sha, flush=True)
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('NOVEL_MAZE_ROUND_TRIP_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != 'NOVEL_MAZE_ROUND_TRIP_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('first native maze infrastructure/raw audit failure; all generated evidence retained')
        bindings = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            bindings[name] = digest(OUTPUT/name)
        verify(launch); verify_artifacts(OUTPUT, bindings); verify_artifacts(INTEGRATION, ids)
        verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
        verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])
        write_json(OUTPUT/'result.json', dict(status='NOVEL_MAZE_ROUND_TRIP_PILOT_COMPLETE',
            conditions=[record], source_sha256=sources, artifact_sha256=bindings,
            wall_s=time.perf_counter()-started, measured_round_trip_successes=int(record['verified_round_trip']),
            independent_layout_development_executions=1, model_training=False, matched_baselines_completed=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('NOVEL_MAZE_ROUND_TRIP_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_NOVEL_MAZE_ROUND_TRIP_PILOT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
