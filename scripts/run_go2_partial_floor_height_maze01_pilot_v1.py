"""Fresh physical maze1 height intervention after completed prospective replay."""
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
from scripts.partial_floor_height_maze01_episode_development import collect, artifacts
from scripts.partial_floor_height_maze01_audit_development import audit
from scripts.partial_floor_height_maze01_native_prefix_development import admit_prefix, compare
from scripts.replay_go2_partial_floor_height_prefix_v1 import OUTPUT as PREFIX, verify_inputs as verify_prefix, CASE as PRIOR_CASE
from scripts.run_go2_direct_flow_maze01_pilot_v1 import OUTPUT as PRIOR, verify_inputs as verify_prior
from scripts.diagnose_go2_direct_flow_maze01_floor_conflict_v1 import admit as admit_prior
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_partial_floor_height_maze01_pilot_v1_attempt_001'
CASE = ('full_jepa_partial_floor_height_maze_01', *PRIOR_CASE[1:])
PROTOCOL = 'docs/go2_partial_floor_height_maze01_pilot_v1_2026-09-09.md'
PRIOR_SHA = 'd6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de'


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(PREFIX,launch['prefix_artifact_sha256'])
    verify_artifacts(PRIOR,launch['prior_artifact_sha256'])
    verify_prefix(read_json(PREFIX,'launch.json')); verify_prior(read_json(PRIOR,'launch.json'))
    if (launch['planned_case'] != list(CASE) or launch['scene_specification'] != specification(1)
            or launch['public_mission'] != public_mission(1)):
        raise ValueError('fixed partial floor height maze1 definition required')


def worker(launch_sha):
    name,index,variant,condition,model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name,layout_index=index,model_name=model_name,
        status='PARTIAL_FLOOR_HEIGHT_MAZE01_WORKER_FAILED',artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT,{'launch.json':launch_sha}); launch = read_json(OUTPUT,'launch.json'); verify_inputs(launch)
            if digest(URDF) != launch['robot_urdf_sha256']: raise ValueError('frozen robot required')
            model,c,v = load_assigned(launch['correction_admission'],model_name)
            before = state_digest(model.state_dict())
            if (c,v) != (condition,variant) or before != launch['prefix_report']['model_state_sha256']:
                raise ValueError('same assigned learned model required')
            result = collect(index,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                geometry=ArticulatedCollisionGeometry(URDF),episode_name=name,condition=condition,variant=variant)
            if state_digest(model.state_dict()) != before: raise ValueError('collection changed model weights')
            bindings = {name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(index,result)}
            verify_artifacts(OUTPUT,bindings); terminal.update(collection=result,artifact_sha256=dict(bindings))
            replay_model,c,v = load_assigned(launch['correction_admission'],model_name)
            if (c,v) != (condition,variant) or state_digest(replay_model.state_dict()) != before:
                raise ValueError('fresh identical audit model required')
            report = audit(index,result,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF),episode_name=name,condition=condition,variant=variant)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name,report); bindings[audit_name] = digest(OUTPUT/audit_name)
            terminal['artifact_sha256'] = dict(bindings)
            prefix = compare(PRIOR/PRIOR_CASE[0],OUTPUT/name,PREFIX,launch['prefix_report'])
            prefix_name = name+'_prefix_comparison.json'; write_json(OUTPUT/prefix_name,prefix)
            bindings[prefix_name] = digest(OUTPUT/prefix_name)
            verify_inputs(launch); verify_artifacts(OUTPUT,bindings)
            terminal.update(status='PARTIAL_FLOOR_HEIGHT_MAZE01_COLLECTED_AND_RAW_AUDITED',artifact_sha256=bindings,
                verified_round_trip=report['verified_round_trip'],native_evaluation=report['native_evaluation'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                renderer_capture_audit=report['renderer_capture_audit'],prefix_comparison=prefix,
                model_state_unchanged=True,reused_development_layout=True,
                direct_corner_flow_missingness_fallback_enabled=True)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'),terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix-result-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive partial floor height native attempt required')
    verify_artifacts(PREFIX, {'result.json':args.prefix_result_sha256}); prefix = read_json(PREFIX, 'result.json')
    prefix_ids = dict(prefix['artifact_sha256']); prefix_ids['result.json'] = args.prefix_result_sha256
    verify_artifacts(PREFIX, prefix_ids); prefix_report = admit_prefix(PREFIX, prefix)
    verify_artifacts(PRIOR, {'result.json':PRIOR_SHA}); prior = read_json(PRIOR, 'result.json')
    prior_ids = dict(prior['artifact_sha256']); prior_ids['result.json'] = PRIOR_SHA
    verify_artifacts(PRIOR, prior_ids); old = read_json(PRIOR, 'launch.json')
    admit_prior(prior, old, read_json(PRIOR, PRIOR_CASE[0]+'_audit.json'))
    inherited = dict(prefix['source_sha256'])
    for name, sha in prior['source_sha256'].items():
        if inherited.get(name) != sha: raise ValueError('incompatible frozen source: '+name)
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_partial_floor_height_maze01_pilot_v1.py',
        'lewm/tests/test_partial_floor_height_maze01_native_development.py',
        'lewm/tests/test_partial_floor_height_maze01_native_prefix_development.py'), inherited)
    keys = ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules','renderer_environment','correction_admission')
    launch = {k:old[k] for k in keys}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),planned_case=list(CASE),
        scene_specification=specification(1),public_mission=public_mission(1),
        implementation_class='PartialHeightDirectFlowController',robot_urdf_sha256=digest(URDF),
        prior_native_result_sha256=PRIOR_SHA,prior_artifact_sha256=prior_ids,
        prospective_prefix_result_sha256=args.prefix_result_sha256,prefix_artifact_sha256=prefix_ids,prefix_report=prefix_report,
        navigation_ticks=NAVIGATION_TICKS,shared_outbound_return_budget=True,
        native_scene_workers=1,opencv_threads=1,blas_threads=1,maximum_tasks_per_process=1,
        minimum_free_bytes=RESERVE_BYTES,planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,memory_admission_bytes=32*1024**3,
        os_resource_limits_enforced=False,physics_paused_during_compute=True,native_execution=True,model_training=False,
        measured_floor_transport_enabled=True,renderer_capture_witnesses_enabled=True,
        direct_corner_flow_missingness_fallback_enabled=True,
        preintervention_original_decisions_exact_required=True,
        correspondence_association_rule_changed=False,partial_floor_height_constraint_enabled=True,
        rigid_and_temporal_thresholds_unchanged=True,
        reference_or_pose_history_reset=False,
        independent_layout_development_execution=False,reused_development_layout=True,
        navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False)
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    memory_ok = resources['memory_available_bytes'] >= 32*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES
    if args.preflight_only:
        print('PARTIAL_FLOOR_HEIGHT_MAZE01_NATIVE_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,
            input_and_source_bindings_verified=True,output_created=False,native_execution=False)),flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('partial floor height native resources unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); launch_sha = digest(OUTPUT/'launch.json')
    print('PARTIAL_FLOOR_HEIGHT_MAZE01_NATIVE_LAUNCHED',launch_sha,flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                future = pool.submit(worker,launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,**hardware()))+'\n'); monitor.flush()
                    done,_ = wait([future],timeout=15,return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('PARTIAL_FLOOR_HEIGHT_MAZE01_NATIVE_TERMINAL',record['status'],record.get('verified_round_trip'),record.get('failure'),flush=True)
        if record['status'] != 'PARTIAL_FLOOR_HEIGHT_MAZE01_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('collection/raw audit/prefix failed; all evidence retained')
        bindings = dict(record['artifact_sha256'])
        for name in ('launch.json','resource_monitor.jsonl',CASE[0]+'_worker.log',CASE[0]+'_worker_terminal.json'):
            bindings[name] = digest(OUTPUT/name)
        verify_inputs(launch); verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='PARTIAL_FLOOR_HEIGHT_MAZE01_PILOT_V1_COMPLETE',
            conditions=[record],source_sha256=sources,artifact_sha256=bindings,
            prior_native_result_sha256=PRIOR_SHA,prospective_prefix_result_sha256=args.prefix_result_sha256,wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']),reused_layout_executions=1,
            new_independent_layout_executions=0,model_training=False,
            direct_corner_flow_missingness_fallback_enabled=True,
            physical_clearance_certified=False,
            navigation_qualified=False,hardware_qualified=False,real_time_qualified=False,goal_achieved=False))
        print('PARTIAL_FLOOR_HEIGHT_MAZE01_NATIVE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_PARTIAL_FLOOR_HEIGHT_MAZE01_NATIVE_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
