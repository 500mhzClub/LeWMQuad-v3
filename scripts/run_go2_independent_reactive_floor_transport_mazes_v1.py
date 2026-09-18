"""Execute the predeclared current reactive controller on fixed layouts1,2,3."""
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
from lewm.independent_floor_transport_study_development import LAYOUTS, independent_scope, require_resources
from lewm.independent_reactive_floor_transport_study_development import (
    MATCHED_KEYS, OUTCOME_KEYS, planned_cases, merge_sources, admit_inputs, paired_outcomes)
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.reactive_floor_transport_maze_episode_development import collect, artifacts
from scripts.reactive_floor_transport_maze_audit_development import audit
from scripts.run_go2_independent_floor_transport_mazes_v1 import OUTPUT as LEARNED, verify_inputs as verify_learned
from scripts.run_go2_reactive_floor_transport_maze_pilot_v1 import OUTPUT as PILOT, CASE as PILOT_CASE, verify_inputs as verify_pilot
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_independent_reactive_floor_transport_mazes_v1_attempt_001'
PROTOCOL = 'docs/go2_independent_reactive_floor_transport_mazes_v1_2026-09-09.md'


def resources_for(resources, remaining):
    return require_resources(resources, remaining, reserve=RESERVE_BYTES,
        collection=COLLECTION_ALLOWANCE_BYTES, persistence=PERSISTENCE_HEADROOM_BYTES)


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(LEARNED, launch['learned_artifact_sha256'])
    verify_artifacts(PILOT, launch['reactive_pilot_artifact_sha256'])
    learned = read_json(LEARNED, 'launch.json'); pilot = read_json(PILOT, 'launch.json')
    verify_learned(learned); verify_pilot(pilot)
    if (launch['planned_cases'] != [list(c) for c in planned_cases()]
            or launch['scene_specifications'] != [specification(i) for i in LAYOUTS]
            or launch['public_missions'] != [public_mission(i) for i in LAYOUTS]
            or launch['scene_specifications'] != learned['scene_specifications']
            or launch['public_missions'] != learned['public_missions']):
        raise ValueError('identical fixed independent scenes and coordinate missions required')
    for key in MATCHED_KEYS:
        if not launch[key] == learned[key] == pilot[key]:
            raise ValueError('matched execution configuration differs: '+key)


def worker(case, launch_sha):
    name, index = case
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, layout_index=index,
        status='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_WORKER_FAILED', artifact_sha256={})
    started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json':launch_sha})
            launch = read_json(OUTPUT, 'launch.json'); verify_inputs(launch)
            if list(case) not in launch['planned_cases'] or digest(URDF) != launch['robot_urdf_sha256']:
                raise ValueError('frozen case and robot required')
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name)
            bindings = {name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(index,result)}
            verify_artifacts(OUTPUT, bindings)
            terminal.update(collection=result, artifact_sha256=dict(bindings))
            report = independent_scope(audit(index, result, launch['source_sha256'][PROTOCOL],
                input_root=OUTPUT, robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=name), index)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name, report)
            bindings[audit_name] = digest(OUTPUT/audit_name)
            terminal['artifact_sha256'] = dict(bindings)
            verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
            terminal.update(status='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED',
                **{k:report[k] for k in OUTCOME_KEYS}, high_level_world_model_used=False,
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
    parser = argparse.ArgumentParser(); parser.add_argument('--learned-cohort-result-sha256', required=True)
    parser.add_argument('--reactive-pilot-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive reactive cohort attempt required')
    records_in = []; launches = []; all_bindings = []
    for root, sha in ((LEARNED,args.learned_cohort_result_sha256), (PILOT,args.reactive_pilot_result_sha256)):
        verify_artifacts(root, {'result.json':sha}); record = read_json(root, 'result.json')
        bindings = dict(record['artifact_sha256']); bindings['result.json'] = sha
        verify_artifacts(root, bindings)
        records_in.append(record); launches.append(read_json(root, 'launch.json')); all_bindings.append(bindings)
    learned, pilot = records_in; old, reactive = launches
    admission = admit_inputs(learned, old,
        [read_json(LEARNED, f'full_jepa_novel_maze_{i:02d}_audit.json') for i in LAYOUTS],
        pilot, reactive, read_json(PILOT, PILOT_CASE+'_audit.json'))
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_independent_reactive_floor_transport_mazes_v1.py',
        'lewm/tests/test_independent_reactive_floor_transport_study_development.py'),
        merge_sources(old['source_sha256'], reactive['source_sha256']))
    resources = hardware(); allowances = resources_for(resources, len(LAYOUTS))
    launch = {k:reactive[k] for k in MATCHED_KEYS}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT),
        learned_cohort_result_sha256=args.learned_cohort_result_sha256,
        reactive_pilot_result_sha256=args.reactive_pilot_result_sha256,
        learned_artifact_sha256=all_bindings[0], reactive_pilot_artifact_sha256=all_bindings[1],
        predecessor_admission=admission, planned_cases=[list(c) for c in planned_cases()],
        scene_specifications=[specification(i) for i in LAYOUTS], public_missions=[public_mission(i) for i in LAYOUTS],
        implementation_class='ReactiveFloorTransportController', hardware=resources, resource_admission=allowances,
        shared_outbound_return_budget=True, minimum_free_bytes=RESERVE_BYTES,
        planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES, persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,
        native_execution=True, model_training=False, high_level_world_model_loaded=False,
        candidate_future_outcomes_evaluated=False, learned_residual_used=False,
        fresh_controller_and_memory_per_case=True, controller_changes_between_cases=False,
        case_order=list(LAYOUTS), outcome_based_case_selection=False,
        independent_layout_development_execution=True, reused_development_layout=False,
        actual_cross_method_prefix_equality_claimed=False, strict_visibility_gate_unchanged=True,
        measured_floor_transport_enabled=True, isolated_prediction_ranking_ablation=False,
        predictive_feasibility_gates_matched=False, real_time_qualified=False,
        navigation_qualified=False, hardware_qualified=False, goal_achieved=False)
    verify_inputs(launch)
    resources = hardware(); launch['hardware'] = resources
    launch['resource_admission'] = resources_for(resources, len(LAYOUTS))
    if args.preflight_only:
        print('INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_PREFLIGHT', json.dumps(dict(source_count=len(sources),
            hardware=resources, allowances=launch['resource_admission'], planned_cases=launch['planned_cases'],
            completed_inputs_and_sources_verified=True, output_created=False, native_execution=False)), flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_LAUNCHED', launch_sha, flush=True)
    started = time.perf_counter(); records = []; artifacts_bound = {}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                for position, case in enumerate(planned_cases()):
                    resources = hardware(); allowance = resources_for(resources, len(LAYOUTS)-position)
                    write_json(OUTPUT/(case[0]+'_admission.json'), dict(hardware=resources, allowances=allowance))
                    future = pool.submit(worker, case, launch_sha)
                    while True:
                        monitor.write(json.dumps(dict(case=case[0], elapsed_s=time.perf_counter()-started, **hardware()))+'\n')
                        monitor.flush(); done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                        if done: record = future.result(); break
                    records.append(record); artifacts_bound.update(record['artifact_sha256'])
                    for suffix in ('_admission.json', '_worker.log', '_worker_terminal.json'):
                        name = case[0]+suffix; artifacts_bound[name] = digest(OUTPUT/name)
                    name = f'cohort_progress_after_{position+1:02d}.json'
                    write_json(OUTPUT/name, dict(completed_conditions=records,
                        remaining_layouts=list(LAYOUTS[position+1:]), original_case_order=list(LAYOUTS)))
                    artifacts_bound[name] = digest(OUTPUT/name)
                    print('INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_CASE_TERMINAL', case[0], record['status'],
                        record.get('verified_round_trip'), record.get('failure'), flush=True)
                    if record['status'] != 'INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('reactive raw collection/audit failed; partial cohort retained without retry')
        for name in ('launch.json', 'resource_monitor.jsonl'): artifacts_bound[name] = digest(OUTPUT/name)
        pairs = paired_outcomes(learned['conditions'], records)
        verify_inputs(launch); verify_artifacts(OUTPUT, artifacts_bound)
        write_json(OUTPUT/'result.json', dict(status='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_MAZES_V1_COMPLETE',
            conditions=records, paired_native_outcomes=pairs, source_sha256=sources, artifact_sha256=artifacts_bound,
            learned_cohort_result_sha256=args.learned_cohort_result_sha256,
            reactive_pilot_result_sha256=args.reactive_pilot_result_sha256, wall_s=time.perf_counter()-started,
            measured_round_trip_successes=sum(int(r['verified_round_trip']) for r in records),
            new_independent_layout_executions=len(records), reused_layout_executions=0,
            all_fixed_cases_executed=True, original_case_order=list(LAYOUTS), model_training=False,
            matched_reactive_method_executions_completed=True, isolated_prediction_ranking_ablation=False,
            jepa_training_advantage_established=False, memory_advantage_established=False,
            statistical_reliability_established=False, navigation_qualified=False,
            hardware_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_STUDY_FAILURE',
            reason=repr(error), completed_conditions=records, artifact_sha256=artifacts_bound,
            original_case_order=list(LAYOUTS), automatic_retry=False))
        raise


if __name__ == '__main__': main()
