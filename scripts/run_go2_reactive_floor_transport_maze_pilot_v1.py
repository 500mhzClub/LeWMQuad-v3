"""Fresh reactive method comparison with current sensing, memory and mission."""
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
from lewm.independent_floor_transport_study_development import admit_predecessor
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.reactive_floor_transport_maze_episode_development import collect, artifacts
from scripts.reactive_floor_transport_maze_audit_development import audit
from scripts.reactive_floor_transport_native_prefix_development import admit_prefix, compare
from scripts.replay_go2_reactive_floor_transport_prefix_v1 import OUTPUT as PREFIX, verify_inputs as verify_prefix
from scripts.run_go2_measured_floor_transport_maze_pilot_v1 import OUTPUT as LEARNED, CASE as LEARNED_CASE, verify_inputs as verify_learned
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_reactive_floor_transport_maze_pilot_v1_attempt_001'
CASE = 'reactive_floor_transport_novel_maze_00'
PROTOCOL = 'docs/go2_reactive_floor_transport_maze_pilot_v1_2026-09-09.md'
PREFIX_SHA = '71a5ecd8486d6d8354762c5dc249307cc7d1bf7dc57fe6d1f2df76372d5aaba9'


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(PREFIX, launch['prefix_artifact_sha256'])
    verify_artifacts(LEARNED, launch['learned_artifact_sha256'])
    verify_prefix(read_json(PREFIX, 'launch.json'))
    verify_learned(read_json(LEARNED, 'launch.json'))
    if (launch['planned_case'] != CASE or launch['scene_specification'] != specification(0)
            or launch['public_mission'] != public_mission(0)):
        raise ValueError('fixed reactive maze0 definition required')


def worker(launch_sha):
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=CASE, layout_index=0, status='REACTIVE_FLOOR_TRANSPORT_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(CASE+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json':launch_sha}); launch = read_json(OUTPUT, 'launch.json')
            verify_inputs(launch)
            if digest(URDF) != launch['robot_urdf_sha256']: raise ValueError('frozen robot geometry required')
            result = collect(0, launch['source_sha256'][PROTOCOL], output=OUTPUT,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=CASE)
            bindings = {CASE+'/'+n:digest(OUTPUT/CASE/n) for n in artifacts(0,result)}
            verify_artifacts(OUTPUT, bindings)
            terminal.update(collection=result, artifact_sha256=dict(bindings))
            report = audit(0, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=CASE)
            name = CASE+'_audit.json'; write_json(OUTPUT/name, report); bindings[name] = digest(OUTPUT/name)
            terminal['artifact_sha256'] = dict(bindings)
            prefix = compare(LEARNED/LEARNED_CASE[0], OUTPUT/CASE, PREFIX, launch['prefix_report'])
            name = CASE+'_prefix_comparison.json'; write_json(OUTPUT/name, prefix); bindings[name] = digest(OUTPUT/name)
            verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
            terminal.update(status='REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED', artifact_sha256=bindings,
                verified_round_trip=report['verified_round_trip'], native_evaluation=report['native_evaluation'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                renderer_capture_audit=report['renderer_capture_audit'], prefix_comparison=prefix,
                high_level_world_model_used=False, reused_development_layout=True)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(CASE+'_worker.log')))
    write_json(OUTPUT/(CASE+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--learned-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive reactive native attempt required')
    verify_artifacts(PREFIX, {'result.json':PREFIX_SHA}); prefix = read_json(PREFIX, 'result.json')
    prefix_ids = {'result.json':PREFIX_SHA, **prefix['artifact_sha256']}; verify_artifacts(PREFIX, prefix_ids)
    prefix_report = admit_prefix(PREFIX, prefix)
    verify_artifacts(LEARNED, {'result.json':args.learned_result_sha256})
    learned = read_json(LEARNED, 'result.json')
    learned_ids = {'result.json':args.learned_result_sha256, **learned['artifact_sha256']}
    verify_artifacts(LEARNED, learned_ids); old = read_json(LEARNED, 'launch.json')
    admission = admit_predecessor(learned, read_json(LEARNED, LEARNED_CASE[0]+'_audit.json'), old, LEARNED_CASE)
    inherited = dict(prefix['source_sha256'])
    for name, sha in old['source_sha256'].items():
        if inherited.get(name) != sha: raise ValueError('incompatible frozen learned/reactive sources: '+name)
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_reactive_floor_transport_maze_pilot_v1.py',
        'lewm/tests/test_reactive_floor_transport_native_development.py',
        'docs/go2_reactive_floor_transport_prefix_result_2026-09-09.md'), inherited)
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')
    launch = {k:old[k] for k in keys}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), planned_case=CASE,
        scene_specification=specification(0), public_mission=public_mission(0),
        implementation_class='ReactiveFloorTransportController', robot_urdf_sha256=digest(URDF),
        learned_result_sha256=args.learned_result_sha256, learned_artifact_sha256=learned_ids,
        learned_predecessor_admission=admission, prefix_artifact_sha256=prefix_ids, prefix_report=prefix_report,
        navigation_ticks=NAVIGATION_TICKS, shared_outbound_return_budget=True,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        minimum_free_bytes=RESERVE_BYTES, planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES, memory_admission_bytes=32*1024**3,
        os_resource_limits_enforced=False, physics_paused_during_compute=True,
        native_execution=True, high_level_world_model_loaded=False, model_training=False,
        shared_observer_registration_map_and_settling_mission=True,
        measured_floor_transport_enabled=True, renderer_capture_witnesses_enabled=True,
        candidate_future_outcomes_evaluated=False, learned_residual_used=False,
        isolated_prediction_ranking_ablation=False, predictive_feasibility_gates_matched=False,
        data_scope='reactive method comparison on reused development maze0',
        independent_layout_development_execution=False, reused_development_layout=True,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    memory_ok = resources['memory_available_bytes'] >= 32*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES
    if args.preflight_only:
        print('REACTIVE_FLOOR_TRANSPORT_NATIVE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok,
            input_and_source_bindings_verified=True, output_created=False, native_execution=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('reactive native resources unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('REACTIVE_FLOOR_TRANSPORT_NATIVE_LAUNCHED', launch_sha, flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('REACTIVE_FLOOR_TRANSPORT_NATIVE_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != 'REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('reactive collection/raw audit/prefix failed; all evidence retained')
        bindings = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE+'_worker.log', CASE+'_worker_terminal.json'):
            bindings[name] = digest(OUTPUT/name)
        verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='REACTIVE_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE',
            conditions=[record], source_sha256=sources, artifact_sha256=bindings,
            learned_result_sha256=args.learned_result_sha256, wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, high_level_world_model_used=False, model_training=False,
            matched_method_execution_completed=True, isolated_prediction_ranking_ablation=False,
            navigation_qualified=False, hardware_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('REACTIVE_FLOOR_TRANSPORT_NATIVE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_FLOOR_TRANSPORT_NATIVE_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
