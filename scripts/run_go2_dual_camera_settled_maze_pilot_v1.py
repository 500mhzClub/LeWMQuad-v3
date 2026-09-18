"""Prospective dual-camera tracking on reused development maze 0."""
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
from scripts.dual_camera_settled_maze_episode_development import collect, artifacts
from scripts.dual_camera_settled_maze_audit_development import audit
from scripts.dual_camera_native_prefix_comparison_development import compare
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_dual_camera_controller_prefix_v2 import OUTPUT as INTEGRATION, verify_all as verify_prefix
from scripts.run_go2_settled_boundary_maze_pilot_v2 import OUTPUT as PREVIOUS, CASE, CORRECTION, FITS
from lewm.dual_camera_native_admission_development import admit_prefix, admit_predecessor
from scripts.dual_camera_intervention_witness_development import admit_intervention
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_dual_camera_settled_maze_pilot_v1_attempt_001'
PROTOCOL = 'docs/go2_dual_camera_settled_maze_pilot_v1_2026-09-09.md'


def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, layout_index=index, model_name=model_name,
        status='DUAL_CAMERA_SETTLED_MAZE_WORKER_FAILED', artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json'); verify_inputs(launch)
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
            verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
            verify_artifacts(FITS, launch['correction_admission']['base_admission']['fit_artifact_sha256'])
            verify_artifacts(CORRECTION, launch['correction_admission']['correction_artifact_sha256'])
            terminal.update(status='DUAL_CAMERA_SETTLED_MAZE_COLLECTED_AND_RAW_AUDITED', artifact_sha256=bindings,
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


PREVIOUS_READOUT = BASE/'go2_settled_boundary_maze_readout_v2_attempt_001'
NATIVE_RESULT = 'a7a02db120b4b662cd66efee01f10784edb6f2ed6984bdc420007773a1b1b6fb'
READOUT_RESULT = 'a5523b729310d39a81baf9ccf9cf94fa6fb8a82077e322d4fbe394a45748a70d'


def verify_inputs(launch):
    verify(launch)
    for path,bindings in launch['previous_evidence_bindings'].items():
        verify_artifacts(path,bindings)
    verify_artifacts(INTEGRATION,launch['integration_artifact_sha256'])
    verify_prefix(read_json(INTEGRATION,'launch.json'))
    admission=launch['correction_admission']
    verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION,admission['correction_artifact_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix-result-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fresh settled-boundary native attempt required')
    verify_artifacts(INTEGRATION,{'result.json':args.prefix_result_sha256})
    integration=read_json(INTEGRATION,'result.json');admit_prefix(integration)
    ids={'result.json':args.prefix_result_sha256,**integration['artifact_sha256']};verify_artifacts(INTEGRATION,ids)
    old=read_json(INTEGRATION,'launch.json');verify_prefix(old);admission=old['correction_admission']
    intervention=admit_intervention(INTEGRATION,integration)
    verify_artifacts(PREVIOUS,{'result.json':NATIVE_RESULT});previous=read_json(PREVIOUS,'result.json')
    verify_artifacts(PREVIOUS,previous['artifact_sha256'])
    predecessor_launch=read_json(PREVIOUS,'launch.json')
    audit=read_json(PREVIOUS,CASE[0]+'_audit.json')
    predecessor_admission=admit_predecessor(previous,audit,old,case=CASE[0])
    if (admission!=predecessor_launch['correction_admission']
            or integration['model_state_sha256']!=predecessor_launch['prefix_report']['model_state_sha256']):
        raise ValueError('unchanged assigned model and admission required')
    verify_artifacts(PREVIOUS_READOUT,{'result.json':READOUT_RESULT});readout=read_json(PREVIOUS_READOUT,'result.json')
    if (readout['status']!='SETTLED_BOUNDARY_MAZE_READOUT_V2_COMPLETE'
            or readout['native_result_sha256']!=NATIVE_RESULT or not readout['original_outcome_unchanged']):
        raise ValueError('completed unchanged tenth execution readout required')
    previous_bindings={str(PREVIOUS):{'result.json':NATIVE_RESULT,**previous['artifact_sha256']},
        str(PREVIOUS_READOUT):{'result.json':READOUT_RESULT,'launch.json':readout['launch_sha256']}}
    inherited=dict(integration['source_sha256'])
    for name,h in readout['source_sha256'].items():
        if name in inherited and inherited[name]!=h:raise ValueError('incompatible frozen readout source: '+name)
        inherited[name]=h
    sources=discover_sources((PROTOCOL,'scripts/run_go2_dual_camera_settled_maze_pilot_v1.py',
        'lewm/tests/test_dual_camera_native_preparation_development.py',
        'lewm/tests/test_dual_camera_native_execution_scope_development.py',
        'docs/go2_settled_boundary_maze_pilot_result_2026-09-09.md'),inherited)
    prefix={k:v for k,v in integration.items() if k not in ('source_sha256','artifact_sha256','hardware_after')}
    resources=hardware()
    launch={k:predecessor_launch[k] for k in ('input_sha256','native_sha256','native_scene_sha256',
        'native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),planned_case=CASE,
        scene_specification=specification(CASE[1]),public_mission=public_mission(CASE[1]),
        correction_admission=admission,integration_artifact_sha256=ids,prefix_report=prefix,
        previous_evidence_bindings=previous_bindings,implementation_class='DualCameraSettledController',
        predecessor_admission=predecessor_admission,intervention_admission=intervention,
        robot_urdf_path=str(URDF),robot_urdf_sha256=digest(URDF),hardware=resources,
        navigation_ticks=NAVIGATION_TICKS,shared_outbound_return_budget=True,
        native_scene_workers=1,opencv_threads=1,blas_threads=1,maximum_tasks_per_process=1,
        minimum_free_bytes=RESERVE_BYTES,planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,memory_admission_bytes=32*1024**3,
        os_resource_limits_enforced=False,physics_paused_during_compute=True,
        concurrency_reason='one native scene after completed raw-controller prefix; no parallel native scenes',
        native_execution=True,model_training=False,checkpoint_selection_performed=False,
        model_map_contact_selector_memory_unchanged=True,measured_quiet_boundary_required_before_dwell=True,
        additional_auxiliary_rgb_for_motion=True,primary_first_tracking=True,
        continuous_speed_bound_from_visual_displacement=False,strict_visibility_gate_unchanged=True,
        predecessor_strict_visibility_failed_frames=[909],predecessor_failure_in_unchanged_prefix=True,
        data_scope='prospective dual-camera tracking intervention on reused development maze 0',
        prior_failed_outcomes_unchanged=True,navigation_qualified=False,hardware_qualified=False,
        real_time_qualified=False,goal_achieved=False)
    verify_inputs(launch)
    memory_ok=resources['memory_available_bytes']>=32*1024**3
    storage_ok=resources['artifact_free_bytes']>=RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES
    if args.preflight_only:
        print('DUAL_CAMERA_SETTLED_MAZE_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            completed_inputs_and_sources_verified=True,memory_admission_pass=memory_ok,
            storage_admission_pass=storage_ok,output_created=False,native_execution=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('native settled-boundary resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);launch_sha=digest(OUTPUT/'launch.json')
    print('DUAL_CAMERA_SETTLED_MAZE_LAUNCHED',launch_sha,flush=True);started=time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                future=pool.submit(worker,launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,**hardware()))+'\n');monitor.flush()
                    done,_=wait([future],timeout=15,return_when=FIRST_COMPLETED)
                    if done:record=future.result();break
        print('DUAL_CAMERA_SETTLED_MAZE_TERMINAL',record['status'],record.get('verified_round_trip'),record.get('failure'),flush=True)
        if record['status']!='DUAL_CAMERA_SETTLED_MAZE_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('native collection/raw audit/prefix failed; all evidence retained')
        bindings=dict(record['artifact_sha256'])
        for name in ('launch.json','resource_monitor.jsonl',CASE[0]+'_worker.log',CASE[0]+'_worker_terminal.json'):
            bindings[name]=digest(OUTPUT/name)
        verify_inputs(launch);verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='DUAL_CAMERA_SETTLED_MAZE_PILOT_V1_COMPLETE',conditions=[record],
            source_sha256=sources,artifact_sha256=bindings,wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']),reused_layout_executions=1,
            new_independent_layout_executions=0,model_training=False,matched_baselines_completed=False,
            navigation_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('DUAL_CAMERA_SETTLED_MAZE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DUAL_CAMERA_SETTLED_MAZE_PILOT_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()

