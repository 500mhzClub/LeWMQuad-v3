"""Prospective later-measured floor contact resolution on reused development maze 0."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.later_floor_resolution_maze_episode_development import collect, artifacts
from scripts.later_floor_resolution_maze_audit_development import audit
from scripts.later_floor_resolution_native_prefix_comparison_development import compare
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_later_floor_resolution_maze_prefix_v1 import (
    OUTPUT as INTEGRATION, INPUT as PREVIOUS,
    READOUT as PREVIOUS_READOUT, CASE, CORRECTION, FITS, RESULTS)
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_later_floor_resolution_maze_pilot_v1_attempt_001'
PROTOCOL = 'docs/go2_later_floor_resolution_maze_pilot_v1_2026-09-09.md'


def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, layout_index=index, model_name=model_name,
        status='LATER_FLOOR_RESOLUTION_MAZE_WORKER_FAILED', artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json'); verify(launch)
            assert launch['planned_case'] == list(CASE) and launch['output_root'] == str(OUTPUT)
            assert launch['scene_specification'] == specification(index) and launch['public_mission'] == public_mission(index)
            assert digest(URDF) == launch['robot_urdf_sha256']
            model, c, v = load_assigned(launch['correction_admission'], model_name); assert (c, v) == (condition, variant)
            before = state_digest(model.state_dict()); assert before == launch['prefix_report']['model_state_sha256']
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            assert state_digest(model.state_dict()) == before
            bindings = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(index, result)}
            verify_artifacts(OUTPUT, bindings); terminal.update(collection=result, artifact_sha256=dict(bindings))
            replay_model, c, v = load_assigned(launch['correction_admission'], model_name); assert (c, v) == (condition, variant)
            report = audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT, model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name, report); bindings[audit_name] = digest(OUTPUT/audit_name)
            terminal['artifact_sha256'] = dict(bindings)
            prefix = compare(PREVIOUS/name, OUTPUT/name, INTEGRATION, launch['prefix_report'])
            prefix_name = name+'_prefix_comparison.json'; write_json(OUTPUT/prefix_name, prefix); bindings[prefix_name] = digest(OUTPUT/prefix_name)
            verify(launch); verify_artifacts(OUTPUT, bindings)
            verify_artifacts(FITS, launch['correction_admission']['base_admission']['fit_artifact_sha256'])
            verify_artifacts(CORRECTION, launch['correction_admission']['correction_artifact_sha256'])
            terminal.update(status='LATER_FLOOR_RESOLUTION_MAZE_COLLECTED_AND_RAW_AUDITED', artifact_sha256=bindings,
                verified_round_trip=report['verified_round_trip'], native_evaluation=report['native_evaluation'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                model_state_unchanged=True, prefix_comparison=prefix, reused_development_layout=True)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive prospective later-floor-resolution experiment; no retry/resume')
    verify_artifacts(INTEGRATION, {'result.json': args.prefix_result_sha256}); integration = read_json(INTEGRATION, 'result.json')
    assert integration['status'] == 'LATER_FLOOR_RESOLUTION_PREFIX_COMPLETE'
    prefix = integration['report']
    assert prefix['first_requested_command_difference'] is not None
    assert prefix['first_terminal_policy_difference'] in (None, prefix['first_requested_command_difference'])
    assert prefix['frames'] == prefix['first_requested_command_difference']+1
    assert prefix['completed_predecessor_raw_audit_reused'] and prefix['predecessor_controller_rerun'] is False
    assert prefix['original_visual_pose_map_mission_exact']
    assert prefix['raw_forecasts_and_original_contact_queries_exact']
    assert prefix['full_nominal_path_checks_unchanged'] and prefix['contact_interpretation_is_declared_intervention']
    assert prefix['model_state_unchanged'] and prefix['final_terminal'] is None
    assert prefix['final_requested_command'] != prefix['prior_requested_command']
    assert prefix['stopped_before_unexecuted_outcome'] and not prefix['unexecuted_outcomes_inferred']
    assert not prefix['new_native_navigation_verified'] and not integration['native_execution']
    ids = {'result.json': args.prefix_result_sha256, **integration['artifact_sha256']}; verify_artifacts(INTEGRATION, ids)
    old = read_json(INTEGRATION, 'launch.json'); verify(old); admission = old['correction_admission']
    previous_bindings = old['replay_input_bindings']
    assert set(previous_bindings) == {str(root) for root in RESULTS}
    for root, sha in RESULTS.items(): assert previous_bindings[str(root)]['result.json'] == sha
    assert read_json(PREVIOUS, 'result.json')['status'] == 'JOINT_FLOOR_REGISTERED_MAZE_PILOT_COMPLETE'
    assert read_json(PREVIOUS_READOUT, 'result.json')['native_result_sha256'] == previous_bindings[str(PREVIOUS)]['result.json']
    for root, input_bindings in previous_bindings.items(): verify_artifacts(root, input_bindings)
    verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_later_floor_resolution_maze_pilot_v1.py',
        'lewm/tests/test_later_floor_resolution_native_scope_development.py',
        'lewm/tests/test_later_floor_resolution_native_prefix_comparison_development.py',
        'docs/go2_later_floor_resolution_maze_prefix_result_2026-09-09.md'), integration['source_sha256'])
    resources = hardware()
    if args.preflight_only:
        print('LATER_FLOOR_RESOLUTION_MAZE_PREFLIGHT', json.dumps(dict(
            completed_inputs_and_sources_verified=True, source_count=len(sources), hardware=resources,
            required_memory_bytes=32*1024**3,
            required_artifact_free_bytes=RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES,
            memory_admission_pass=resources['memory_available_bytes'] >= 32*1024**3,
            storage_admission_pass=resources['artifact_free_bytes'] >= RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES,
            output_created=False, native_execution=False)), flush=True)
        return
    if resources['memory_available_bytes'] < 32*1024**3:
        raise ValueError('32 GiB available RAM required')
    if resources['artifact_free_bytes'] < RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES:
        raise ValueError('declared later-floor-resolution storage/persistence envelope unavailable')
    launch = {k: old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), planned_case=CASE,
        scene_specification=specification(CASE[1]), public_mission=public_mission(CASE[1]),
        correction_admission=admission, integration_artifact_sha256=ids, prefix_report=prefix,
        floor_plane_measurement_model='common_plane_all_current_camera_candidates',
        contact_interpretation='strictly_later_single_view_complete_floor_evidence_nominal_feet',
        implementation_class='LaterFloorResolutionRoundTripController',
        original_partitions_retained=True, full_nominal_horizon_unchanged=True,
        previous_evidence_bindings=previous_bindings,
        robot_urdf_path=str(URDF), robot_urdf_sha256=digest(URDF), hardware=resources,
        navigation_ticks=NAVIGATION_TICKS, shared_outbound_return_budget=True,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        minimum_free_bytes=RESERVE_BYTES, planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,
        predecessor_collection_allowance_bytes=10*1024**3, storage_allowance_unchanged=True,
        memory_admission_bytes=32*1024**3, os_resource_limits_enforced=False,
        concurrency_reason='one existing-maze later-floor-resolution intervention; storage envelope excludes parallel scenes',
        native_execution=True, model_training=False, checkpoint_selection_performed=False,
        physics_paused_during_compute=True, real_time_qualified=False,
        data_scope='prospective policy intervention on reused development maze 0; not a new independent layout',
        prior_failed_outcomes_unchanged=True, predecessor_strict_visibility_failed_frames=[909],
        strict_visibility_gate_unchanged=True, predecessor_failure_in_unchanged_prefix=prefix['frames'] > 909,
        navigation_qualified=False, goal_achieved=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    launch_sha = digest(OUTPUT/'launch.json'); started = time.perf_counter()
    print('LATER_FLOOR_RESOLUTION_MAZE_LAUNCHED', launch_sha, flush=True)
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('LATER_FLOOR_RESOLUTION_MAZE_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != 'LATER_FLOOR_RESOLUTION_MAZE_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('native later-floor-resolution infrastructure/raw audit/prefix comparison failure; evidence retained')
        bindings = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            bindings[name] = digest(OUTPUT/name)
        verify(launch); verify_artifacts(OUTPUT, bindings); verify_artifacts(INTEGRATION, ids)
        for root, input_bindings in previous_bindings.items(): verify_artifacts(root, input_bindings)
        verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
        verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])
        write_json(OUTPUT/'result.json', dict(status='LATER_FLOOR_RESOLUTION_MAZE_PILOT_COMPLETE', conditions=[record],
            source_sha256=sources, artifact_sha256=bindings, wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, model_training=False, matched_baselines_completed=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('LATER_FLOOR_RESOLUTION_MAZE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_LATER_FLOOR_RESOLUTION_MAZE_PILOT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
