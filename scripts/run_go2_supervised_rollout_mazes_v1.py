"""Fresh matched supervised-rollout executions on fixed development layouts1–3."""
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
from lewm.independent_floor_transport_study_development import LAYOUTS, require_resources
from lewm.independent_reactive_floor_transport_study_development import merge_sources, require_raw_audit
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE, MATCHED_KEYS, OUTCOME_KEYS, planned_cases, paired_outcomes
from lewm.matched_rollout_objective_admission_development import admit_objective_pair, NAMES
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.measured_floor_transport_maze_episode_development import collect, artifacts
from scripts.measured_floor_transport_maze_audit_development import audit
from scripts.supervised_rollout_native_prefix_development import admit_prefixes, compare
from scripts.replay_go2_matched_objective_prefixes_v1 import (
    OUTPUT as PREFIX, INPUT as LEARNED, verify_inputs as verify_prefix, admit_cases, FITS)
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_supervised_rollout_mazes_v1_attempt_001'
PROTOCOL='docs/go2_supervised_rollout_mazes_v1_2026-09-09.md'
PREFIX_SHA='edf680896e11ef22096cf56323094be2de24b712405466149cdeeb59bab2a6fd'


def resources_for(resources,remaining):
    return require_resources(resources,remaining,reserve=RESERVE_BYTES,
        collection=COLLECTION_ALLOWANCE_BYTES,persistence=PERSISTENCE_HEADROOM_BYTES)


def verify_inputs(launch):
    verify(launch); verify_artifacts(PREFIX,launch['prefix_artifact_sha256'])
    prefix_launch=read_json(PREFIX,'launch.json')
    # The existing prefix verifier authenticates this exact full cohort map and
    # its upstream evidence. Equality is required before reusing that check.
    if (launch['learned_artifact_sha256']!=prefix_launch['replay_input_bindings']
            or launch['correction_admission']!=prefix_launch['correction_admission']):
        raise ValueError('identical original cohort and correction admission required')
    verify_prefix(prefix_launch)
    old=read_json(LEARNED,'launch.json')
    if (launch['planned_cases']!=[list(c) for c in planned_cases()]
            or launch['model_state_sha256']!=SUPERVISED_STATE
            or launch['implementation_class']!='MeasuredFloorTransportController'
            or launch['scene_specifications']!=[specification(i) for i in LAYOUTS]
            or launch['public_missions']!=[public_mission(i) for i in LAYOUTS]
            or launch['scene_specifications']!=old['scene_specifications']
            or launch['public_missions']!=old['public_missions']):
        raise ValueError('same fixed scenes, public missions and exact supervised assignment required')
    for key in MATCHED_KEYS:
        if launch[key]!=old[key]: raise ValueError('matched execution setting differs: '+key)


def worker(case,launch_sha):
    name,index,variant,condition,model_name=case
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal=dict(case=name,layout_index=index,model_name=model_name,
        status='SUPERVISED_ROLLOUT_WORKER_FAILED',artifact_sha256={}); started=time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT,{'launch.json':launch_sha}); launch=read_json(OUTPUT,'launch.json'); verify_inputs(launch)
            if list(case) not in launch['planned_cases'] or digest(URDF)!=launch['robot_urdf_sha256']:
                raise ValueError('frozen supervised case and robot required')
            model,c,v=load_assigned(launch['correction_admission'],model_name)
            if (c,v)!=(condition,variant) or state_digest(model.state_dict())!=SUPERVISED_STATE:
                raise ValueError('exact assigned supervised corrected model required')
            result=collect(index,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                geometry=ArticulatedCollisionGeometry(URDF),episode_name=name,condition=condition,variant=variant)
            if state_digest(model.state_dict())!=SUPERVISED_STATE: raise ValueError('collection changed model state')
            bindings={name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(index,result)}
            verify_artifacts(OUTPUT,bindings); terminal.update(collection=result,artifact_sha256=dict(bindings))
            replay_model,c,v=load_assigned(launch['correction_admission'],model_name)
            if (c,v)!=(condition,variant) or state_digest(replay_model.state_dict())!=SUPERVISED_STATE:
                raise ValueError('fresh identical supervised audit model required')
            report=audit(index,result,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF),episode_name=name,condition=condition,variant=variant)
            audit_name=name+'_audit.json'; write_json(OUTPUT/audit_name,report); bindings[audit_name]=digest(OUTPUT/audit_name)
            terminal['artifact_sha256']=dict(bindings)
            prefix_report=launch['prefix_reports'][index-1]
            witness=compare(LEARNED/prefix_report['case'],OUTPUT/name,PREFIX,prefix_report,launch['correction_admission'])
            prefix_name=name+'_prefix_comparison.json'; write_json(OUTPUT/prefix_name,witness)
            bindings[prefix_name]=digest(OUTPUT/prefix_name)
            terminal.update(artifact_sha256=dict(bindings),prefix_comparison=witness)
            verify_inputs(launch); verify_artifacts(OUTPUT,bindings)
            terminal.update(status='SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED',
                **{k:report[k] for k in OUTCOME_KEYS},model_state_sha256=SUPERVISED_STATE,model_state_unchanged=True,
                condition=condition,variant=variant,head='rollout_outcomes',
                reused_development_layout=True,independent_layout_development_execution=False)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure']=repr(error)
    terminal.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'),terminal)
    return terminal


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--learned-cohort-result-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true'); args=parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed supervised cohort required')
    verify_artifacts(PREFIX,{'result.json':PREFIX_SHA}); prefix=read_json(PREFIX,'result.json')
    prefix_ids={'result.json':PREFIX_SHA,**prefix['artifact_sha256']}; verify_artifacts(PREFIX,prefix_ids)
    prefix_launch=read_json(PREFIX,'launch.json'); verify_prefix(prefix_launch)
    verify_artifacts(LEARNED,{'result.json':args.learned_cohort_result_sha256}); learned=read_json(LEARNED,'result.json')
    learned_ids={'result.json':args.learned_cohort_result_sha256,**learned['artifact_sha256']}
    if learned_ids!=prefix_launch['replay_input_bindings']:
        raise ValueError('completed exact learned cohort in the matched replay required')
    old=read_json(LEARNED,'launch.json'); admit_cases(learned,old)
    for record in learned['conditions']:
        require_raw_audit(record,read_json(LEARNED,record['case']+'_audit.json'),learned=True)
    admission=old['correction_admission']
    pair=admit_objective_pair(admission,{n:read_json(FITS,n+'_fit.json') for n in NAMES})
    if pair!=prefix_launch['objective_pair_admission']: raise ValueError('unchanged matched training-objective definition required')
    reports=admit_prefixes(PREFIX,prefix,admission)
    inherited=merge_sources(old['source_sha256'],prefix['source_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/run_go2_supervised_rollout_mazes_v1.py',
        'lewm/tests/test_supervised_rollout_native_prefix_development.py',
        'lewm/tests/test_supervised_rollout_maze_study_development.py',
        'docs/go2_matched_objective_prefixes_result_2026-09-09.md'),inherited)
    launch={k:old[k] for k in MATCHED_KEYS}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),
        learned_cohort_result_sha256=args.learned_cohort_result_sha256,learned_artifact_sha256=learned_ids,
        prefix_result_sha256=PREFIX_SHA,prefix_artifact_sha256=prefix_ids,prefix_reports=reports,
        correction_admission=admission,objective_pair_admission=pair,planned_cases=[list(c) for c in planned_cases()],
        scene_specifications=[specification(i) for i in LAYOUTS],public_missions=[public_mission(i) for i in LAYOUTS],
        implementation_class='MeasuredFloorTransportController',model_state_sha256=SUPERVISED_STATE,
        condition='supervised_rollout',variant='full',head='rollout_outcomes',
        navigation_ticks=NAVIGATION_TICKS,shared_outbound_return_budget=True,
        minimum_free_bytes=RESERVE_BYTES,planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,native_execution=True,model_training=False,
        fresh_controller_and_memory_per_case=True,controller_or_model_changes_between_cases=False,
        case_order=list(LAYOUTS),outcome_based_case_selection=False,
        measured_floor_transport_enabled=True,strict_visibility_gate_unchanged=True,
        reused_development_layout=True,independent_layout_development_execution=False,
        same_predictive_controller_and_rollout_head=True,jepa_training_advantage_established=False,
        memory_advantage_established=False,real_time_qualified=False,navigation_qualified=False,
        hardware_qualified=False,goal_achieved=False)
    verify_inputs(launch); resources=hardware(); launch['hardware']=resources
    launch['resource_admission']=resources_for(resources,len(LAYOUTS))
    if args.preflight_only:
        print('SUPERVISED_ROLLOUT_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            allowances=launch['resource_admission'],planned_cases=launch['planned_cases'],
            all_three_prefixes_admitted=True,input_and_source_bindings_verified=True,
            output_created=False,native_execution=False)),flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); launch_sha=digest(OUTPUT/'launch.json')
    print('SUPERVISED_ROLLOUT_LAUNCHED',launch_sha,flush=True)
    started=time.perf_counter(); records=[]; bindings={}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                for position,case in enumerate(planned_cases()):
                    resources=hardware(); allowance=resources_for(resources,len(LAYOUTS)-position)
                    write_json(OUTPUT/(case[0]+'_admission.json'),dict(hardware=resources,allowances=allowance))
                    future=pool.submit(worker,case,launch_sha)
                    while True:
                        monitor.write(json.dumps(dict(case=case[0],elapsed_s=time.perf_counter()-started,**hardware()))+'\n'); monitor.flush()
                        done,_=wait([future],timeout=15,return_when=FIRST_COMPLETED)
                        if done: record=future.result(); break
                    records.append(record); bindings.update(record['artifact_sha256'])
                    for suffix in ('_admission.json','_worker.log','_worker_terminal.json'):
                        name=case[0]+suffix; bindings[name]=digest(OUTPUT/name)
                    name=f'cohort_progress_after_{position+1:02d}.json'
                    write_json(OUTPUT/name,dict(completed_conditions=records,
                        remaining_layouts=list(LAYOUTS[position+1:]),original_case_order=list(LAYOUTS)))
                    bindings[name]=digest(OUTPUT/name)
                    print('SUPERVISED_ROLLOUT_CASE_TERMINAL',case[0],record['status'],record.get('verified_round_trip'),record.get('failure'),flush=True)
                    if record['status']!='SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('raw audit or physical-prefix failure; partial cohort retained without retry')
        for name in ('launch.json','resource_monitor.jsonl'): bindings[name]=digest(OUTPUT/name)
        pairs=paired_outcomes(learned['conditions'],records)
        verify_inputs(launch); verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='SUPERVISED_ROLLOUT_MAZES_V1_COMPLETE',conditions=records,
            paired_native_outcomes=pairs,source_sha256=sources,artifact_sha256=bindings,
            learned_cohort_result_sha256=args.learned_cohort_result_sha256,prefix_result_sha256=PREFIX_SHA,
            wall_s=time.perf_counter()-started,all_fixed_cases_executed=True,original_case_order=list(LAYOUTS),
            measured_round_trip_successes=sum(int(r['verified_round_trip']) for r in records),
            new_independent_layout_executions=0,reused_layout_executions=len(records),model_training=False,
            same_predictive_controller_and_rollout_head=True,jepa_training_advantage_established=False,
            memory_advantage_established=False,statistical_reliability_established=False,
            navigation_qualified=False,hardware_qualified=False,real_time_qualified=False,goal_achieved=False))
        print('SUPERVISED_ROLLOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SUPERVISED_ROLLOUT_COHORT_FAILURE',reason=repr(error),
            completed_conditions=records,artifact_sha256=bindings,original_case_order=list(LAYOUTS),automatic_retry=False)); raise


if __name__=='__main__': main()
