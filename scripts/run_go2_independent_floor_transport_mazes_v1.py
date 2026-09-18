"""Execute the same frozen controller on development layouts1,2,3, sequentially."""
import argparse
import contextlib
import multiprocessing
import resource
import time
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.independent_floor_transport_study_development import (
    MODEL_STATE, LAYOUTS, planned_cases, admit_predecessor, independent_scope, require_resources)
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.measured_floor_transport_maze_episode_development import collect, artifacts
from scripts.measured_floor_transport_maze_audit_development import audit
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_measured_floor_transport_maze_pilot_v1 import (
    OUTPUT as PREVIOUS, CASE, verify_inputs as verify_predecessor)
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_independent_floor_transport_mazes_v1_attempt_001'
PROTOCOL = 'docs/go2_independent_floor_transport_mazes_v1_2026-09-09.md'


def resources_for(resources, remaining):
    return require_resources(resources, remaining, reserve=RESERVE_BYTES,
        collection=COLLECTION_ALLOWANCE_BYTES, persistence=PERSISTENCE_HEADROOM_BYTES)


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(PREVIOUS, launch['predecessor_artifact_sha256'])
    verify_predecessor(read_json(PREVIOUS, 'launch.json'))
    if launch['planned_cases'] != [list(c) for c in planned_cases(CASE)]:
        raise ValueError('fixed all-three-layout cohort required')
    if launch['scene_specifications'] != [specification(i) for i in LAYOUTS]:
        raise ValueError('fixed independent scene definitions required')
    if launch['public_missions'] != [public_mission(i) for i in LAYOUTS]:
        raise ValueError('coordinate-only missions required')


def worker(case, launch_sha):
    name, index, variant, condition, model_name = case
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, layout_index=index, model_name=model_name,
        status='INDEPENDENT_FLOOR_TRANSPORT_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha})
            launch = read_json(OUTPUT, 'launch.json'); verify_inputs(launch)
            if list(case) not in launch['planned_cases'] or digest(URDF) != launch['robot_urdf_sha256']:
                raise ValueError('frozen case and robot required')
            model, c, v = load_assigned(launch['correction_admission'], model_name)
            if (c, v) != (condition, variant) or state_digest(model.state_dict()) != MODEL_STATE:
                raise ValueError('fresh unchanged assigned model required')
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            if state_digest(model.state_dict()) != MODEL_STATE:
                raise ValueError('model changed during native execution')
            bindings = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(index, result)}
            verify_artifacts(OUTPUT, bindings)
            terminal.update(collection=result, artifact_sha256=dict(bindings))
            replay_model, c, v = load_assigned(launch['correction_admission'], model_name)
            if (c, v) != (condition, variant) or state_digest(replay_model.state_dict()) != MODEL_STATE:
                raise ValueError('fresh unchanged replay model required')
            report = independent_scope(audit(index, result, launch['source_sha256'][PROTOCOL],
                input_root=OUTPUT, model=replay_model, robot_geometry=ArticulatedCollisionGeometry(URDF),
                episode_name=name, condition=condition, variant=variant), index)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name, report)
            bindings[audit_name] = digest(OUTPUT/audit_name)
            terminal['artifact_sha256'] = dict(bindings)
            verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
            terminal.update(status='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED',
                verified_round_trip=report['verified_round_trip'], native_evaluation=report['native_evaluation'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                renderer_capture_audit=report['renderer_capture_audit'], model_state_unchanged=True,
                independent_layout_development_execution=True, reused_development_layout=False)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive independent cohort attempt required')
    verify_artifacts(PREVIOUS, {'result.json': args.native_result_sha256})
    previous = read_json(PREVIOUS, 'result.json')
    bindings = {'result.json': args.native_result_sha256, **previous['artifact_sha256']}
    verify_artifacts(PREVIOUS, bindings)
    old = read_json(PREVIOUS, 'launch.json'); verify_predecessor(old)
    admission = admit_predecessor(previous, read_json(PREVIOUS, CASE[0]+'_audit.json'), old, CASE)
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_independent_floor_transport_mazes_v1.py',
        'lewm/tests/test_independent_floor_transport_study_development.py'), old['source_sha256'])
    resources = hardware(); allowances = resources_for(resources, len(LAYOUTS))
    # Retain the predecessor's exact native/model/runtime bindings, but remove
    # maze0-specific execution claims from the new launch definition.
    inherited_keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'correction_admission',
        'renderer_environment', 'robot_urdf_path', 'robot_urdf_sha256')
    launch = {k:old[k] for k in inherited_keys}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT),
        predecessor_artifact_sha256=bindings, predecessor_admission=admission,
        planned_cases=[list(c) for c in planned_cases(CASE)],
        scene_specifications=[specification(i) for i in LAYOUTS], public_missions=[public_mission(i) for i in LAYOUTS],
        implementation_class='MeasuredFloorTransportController', model_state_sha256=MODEL_STATE,
        hardware=resources, resource_admission=allowances, navigation_ticks=NAVIGATION_TICKS,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        native_execution=True, model_training=False, fresh_controller_and_memory_per_case=True,
        controller_or_model_changes_between_cases=False, outcome_based_case_selection=False,
        case_order=list(LAYOUTS), physics_paused_during_compute=True,
        independent_layout_development_execution=True, reused_development_layout=False,
        prior_maze0_prefix_comparison_applicable=False, strict_visibility_gate_unchanged=True,
        measured_floor_transport_enabled=True, renderer_capture_witnesses_enabled=True,
        matched_baselines_completed=False, real_time_qualified=False, navigation_qualified=False,
        hardware_qualified=False, goal_achieved=False)
    verify_inputs(launch)
    # Recheck admission after potentially lengthy source/input validation.
    resources = hardware(); launch['hardware'] = resources
    launch['resource_admission'] = resources_for(resources, len(LAYOUTS))
    if args.preflight_only:
        print('INDEPENDENT_FLOOR_TRANSPORT_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            allowances=launch['resource_admission'], planned_cases=launch['planned_cases'],
            completed_inputs_and_sources_verified=True, output_created=False, native_execution=False)), flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('INDEPENDENT_FLOOR_TRANSPORT_LAUNCHED', launch_sha, flush=True)
    started = time.perf_counter(); records = []; artifacts_bound = {}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                for position, case in enumerate(planned_cases(CASE)):
                    resources = hardware(); allowance = resources_for(resources, len(LAYOUTS)-position)
                    write_json(OUTPUT/(case[0]+'_admission.json'), dict(hardware=resources, allowances=allowance))
                    future = pool.submit(worker, case, launch_sha)
                    while True:
                        monitor.write(json.dumps(dict(case=case[0], elapsed_s=time.perf_counter()-started, **hardware()))+'\n')
                        monitor.flush()
                        done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                        if done: record = future.result(); break
                    records.append(record)
                    artifacts_bound.update(record['artifact_sha256'])
                    for suffix in ('_admission.json', '_worker.log', '_worker_terminal.json'):
                        name = case[0]+suffix; artifacts_bound[name] = digest(OUTPUT/name)
                    progress_name = f'cohort_progress_after_{position+1:02d}.json'
                    write_json(OUTPUT/progress_name, dict(completed_conditions=records,
                        remaining_layouts=list(LAYOUTS[position+1:]), original_case_order=list(LAYOUTS)))
                    artifacts_bound[progress_name] = digest(OUTPUT/progress_name)
                    print('INDEPENDENT_FLOOR_TRANSPORT_CASE_TERMINAL', case[0], record['status'],
                        record.get('verified_round_trip'), record.get('failure'), flush=True)
                    if record['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('native collection/raw audit failed; partial cohort retained without retry')
        for name in ('launch.json', 'resource_monitor.jsonl'):
            artifacts_bound[name] = digest(OUTPUT/name)
        verify_inputs(launch); verify_artifacts(OUTPUT, artifacts_bound)
        write_json(OUTPUT/'result.json', dict(status='INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE',
            conditions=records, source_sha256=sources, artifact_sha256=artifacts_bound,
            predecessor_result_sha256=args.native_result_sha256, wall_s=time.perf_counter()-started,
            measured_round_trip_successes=sum(int(r['verified_round_trip']) for r in records),
            new_independent_layout_executions=len(records), reused_layout_executions=0,
            all_fixed_cases_executed=True, original_case_order=list(LAYOUTS), model_training=False,
            matched_baselines_completed=False, navigation_qualified=False, hardware_qualified=False,
            real_time_qualified=False, goal_achieved=False))
        print('INDEPENDENT_FLOOR_TRANSPORT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_INDEPENDENT_FLOOR_TRANSPORT_STUDY_FAILURE',
            reason=repr(error), completed_conditions=records, artifact_sha256=artifacts_bound,
            original_case_order=list(LAYOUTS), automatic_retry=False))
        raise


if __name__ == '__main__': main()
