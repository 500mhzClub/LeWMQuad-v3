"""Prospective nearer-route connector baseline with independent raw auditing."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.reactive_nominal_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.reactive_connector_maze_episode_development import collect, artifacts
from scripts.reactive_connector_maze_audit_development import audit
from scripts.reactive_nominal_native_prefix_comparison_development import compare
from scripts.replay_go2_reactive_connector_route_prefix_v1 import OUTPUT as PREFIX, INPUT as PREVIOUS, CASE as PRIOR_CASE
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_reactive_connector_maze_pilot_v1_attempt_001'
PROTOCOL = 'docs/go2_reactive_connector_maze_pilot_v1_2026-09-08.md'
CASE = 'reactive_connector_maze_00'


def worker(launch_sha):
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=CASE, layout_index=0, controller='reactive_connector_round_trip_controller_v1',
        status='REACTIVE_CONNECTOR_MAZE_WORKER_FAILED', artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(CASE+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json'); verify(launch)
            assert launch['planned_case'] == CASE and launch['output_root'] == str(OUTPUT)
            assert launch['scene_specification'] == specification(0) and launch['public_mission'] == public_mission(0)
            assert digest(URDF) == launch['robot_urdf_sha256']
            result = collect(0, launch['source_sha256'][PROTOCOL], output=OUTPUT,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=CASE)
            bindings = {CASE+'/'+n: digest(OUTPUT/CASE/n) for n in artifacts(0, result)}
            verify_artifacts(OUTPUT, bindings); terminal.update(collection=result, artifact_sha256=dict(bindings))
            report = audit(0, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=CASE)
            name = CASE+'_audit.json'; write_json(OUTPUT/name, report); bindings[name] = digest(OUTPUT/name)
            terminal['artifact_sha256'] = dict(bindings)
            comparisons = dict(original_reactive=compare(PREVIOUS/PRIOR_CASE, OUTPUT/CASE, launch['prefix_report']))
            name = CASE+'_prefix_comparisons.json'; write_json(OUTPUT/name, comparisons); bindings[name] = digest(OUTPUT/name)
            verify(launch); verify_artifacts(OUTPUT, bindings)
            terminal.update(status='REACTIVE_CONNECTOR_MAZE_COLLECTED_AND_RAW_AUDITED', artifact_sha256=bindings,
                verified_round_trip=report['verified_round_trip'], native_evaluation=report['native_evaluation'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                high_level_world_model_used=False, same_pretrained_locomotion_policy=True,
                prefix_comparisons=comparisons, reused_development_layout=True,
                predictive_geometry_gates_matched=False)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(CASE+'_worker.log')))
    write_json(OUTPUT/(CASE+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive prospective reactive baseline required')
    bound = []
    for root, sha, status in ((PREFIX, args.prefix_result_sha256, 'REACTIVE_CONNECTOR_ROUTE_PREFIX_COMPLETE'),):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json'); assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids, result))
    integration = bound[0][2]; prefix = integration['report']
    changed = prefix['first_requested_command_difference']
    assert type(changed) is int and changed >= 0 and prefix['frames'] == changed+1
    assert prefix['stopped_before_unexecuted_outcome'] and prefix['original_nominal_radius_preserved']
    assert prefix['exact_other_decision_fields_before_intervention'] and prefix['measured_route_and_current_geometry_exact']
    assert prefix['first_terminal_policy_difference'] is None and prefix['final_terminal'] is None
    assert prefix['causal_observations_maps_and_mission_exact'] and prefix['original_command_outputs_before_intervention_exact']
    assert not prefix['learned_model_used'] and not prefix['candidate_future_outcomes_evaluated']
    changed_row = next(islice(read_rows(PREFIX), changed, changed+1))
    assert changed_row['tick'] == changed
    assert changed_row['decision']['requested_command'] == prefix['final_requested_command']
    assert changed_row['decision']['new_selection']['nearer_observed_route_target'] is True
    assert prefix['prior_requested_command'] == [0., 0., 0.]
    old = read_json(PREFIX, 'launch.json'); verify(old)
    previous_bindings = old['replay_input_bindings']
    assert str(PREVIOUS) in previous_bindings
    from pathlib import Path
    for name, ids in previous_bindings.items(): verify_artifacts(Path(name), ids)
    merged = {}
    for _, _, result in bound:
        for name, sha in result['source_sha256'].items():
            if name in merged: assert merged[name] == sha, ('frozen source disagreement', name)
            merged[name] = sha
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_reactive_connector_maze_pilot_v1.py',
        'lewm/tests/test_reactive_connector_native_scope_development.py',
        'lewm/tests/test_reactive_nominal_native_prefix_comparison_development.py',
        'docs/go2_reactive_connector_route_prefix_result_2026-09-08.md'), merged)
    resources = hardware()
    if args.preflight_only:
        print('REACTIVE_CONNECTOR_PREFLIGHT', json.dumps(dict(
            source_count=len(sources), inputs_and_source_union_verified=True, output_created=False,
            memory_admitted=resources['memory_available_bytes'] >= 32*1024**3,
            storage_admitted=resources['artifact_free_bytes'] >= RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES,
            required_artifact_free_bytes=RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES,
            hardware=resources)), flush=True)
        return
    if resources['memory_available_bytes'] < 32*1024**3:
        raise ValueError('32 GiB available RAM required')
    if resources['artifact_free_bytes'] < RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES:
        raise ValueError('same full-length 10+1 GiB envelope over 40 GiB reserve required')
    launch = {k: old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), planned_case=CASE,
        scene_specification=specification(0), public_mission=public_mission(0), prefix_report=prefix,
        previous_replay_input_bindings=previous_bindings,
        completed_input_bindings={str(p): ids for p, ids, _ in bound},
        robot_urdf_path=str(URDF), robot_urdf_sha256=digest(URDF), hardware=resources,
        navigation_ticks=NAVIGATION_TICKS, shared_outbound_return_budget=True,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        minimum_free_bytes=RESERVE_BYTES, planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES, memory_admission_bytes=32*1024**3,
        os_resource_limits_enforced=False, concurrency_reason='one ordered full-length baseline; preserve matched resources and independent process ownership',
        native_execution=True, model_training=False, high_level_world_model_used=False,
        same_pretrained_locomotion_policy=True, checkpoint_selection_performed=False,
        historical_world_model_bindings_provenance_only=True, predictive_geometry_gates_matched=False,
        physics_paused_during_compute=True, real_time_qualified=False,
        data_scope='reactive nearer-route connector baseline on reused development maze 0',
        prior_failed_outcomes_unchanged=True, navigation_qualified=False, goal_achieved=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    launch_sha = digest(OUTPUT/'launch.json'); started = time.perf_counter()
    print('REACTIVE_CONNECTOR_MAZE_LAUNCHED', launch_sha, flush=True)
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('REACTIVE_CONNECTOR_MAZE_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != 'REACTIVE_CONNECTOR_MAZE_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('baseline infrastructure/raw audit/prefix comparison failed; evidence retained')
        bindings = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE+'_worker.log', CASE+'_worker_terminal.json'):
            bindings[name] = digest(OUTPUT/name)
        verify(launch); verify_artifacts(OUTPUT, bindings)
        for root, ids, _ in bound: verify_artifacts(root, ids)
        for name, ids in previous_bindings.items(): verify_artifacts(Path(name), ids)
        write_json(OUTPUT/'result.json', dict(status='REACTIVE_CONNECTOR_MAZE_PILOT_COMPLETE', conditions=[record],
            source_sha256=sources, artifact_sha256=bindings, wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, model_training=False, high_level_world_model_used=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('REACTIVE_CONNECTOR_MAZE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_CONNECTOR_MAZE_PILOT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
