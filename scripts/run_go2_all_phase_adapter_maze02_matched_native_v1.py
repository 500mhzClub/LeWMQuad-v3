"""Execute the six fixed models with the prospectively replayed planner adapter."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import numpy as np
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_study_development import (
    CASES, WORKER_STATUS, IMPLEMENTATION, resources_for, require_case, complete_cohort)
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS
from lewm.all_phase_residual_maze02_readout_development import case_readout
from scripts.all_phase_residual_maze02_native_inputs_development import (
    admit_inputs, verify_bound_inputs, reference, corrected_wait, NATIVE_OWNER, CORRECTION_OWNER)
from scripts.all_phase_adapter_native_startup_development import compare as compare_adapter_startup
from scripts import all_phase_adapter_native_evidence_development as adapter_evidence
from scripts.all_phase_adapter_native_evidence_development import prepared_sources
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from scripts.all_phase_planner_model_admission_development import load_assigned
from scripts.residual_anchored_continuation_maze_episode_development import collect, artifacts
from scripts.residual_anchored_continuation_maze_audit_development import audit
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.supervised_commitment_contact_queue_gate_development import require_native_idle
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
SOURCE='scripts/run_go2_all_phase_adapter_maze02_matched_native_v1.py'
PROTOCOL='docs/go2_all_phase_adapter_maze02_matched_native_v1_2026-09-10.md'
TESTS=('lewm/tests/test_all_phase_adapter_native_development.py',
    'lewm/tests/test_all_phase_planner_model_adapter_development.py',
    'lewm/tests/test_all_phase_residual_maze02_study_development.py',
    'lewm/tests/test_all_phase_residual_maze02_startup_development.py',
    'lewm/tests/test_all_phase_residual_maze02_native_launcher_development.py')


def verify_inputs(launch, *, full=False):
    verify_ordered_launch(launch)
    expected=dict(planned_cases=[list(c) for c in CASES],implementation_class=IMPLEMENTATION,
        scene_specification=specification(2),public_mission=public_mission(2),navigation_ticks=NAVIGATION_TICKS,
        output_root=str(OUTPUT),native_scene_workers=1,opencv_threads=1,blas_threads=1,
        maximum_tasks_per_process=1,physics_paused_during_compute=True,
        renderer_capture_witnesses_enabled=True,fresh_controller_and_memory_per_case=True,
        unplanned_controller_or_model_changes_between_cases=False,checkpoint_selection_performed=False,
        measured_floor_transport_enabled=True,residual_first_interval_feasibility_fallback_enabled=True,
        residual_hold_feasibility_enabled=True,residual_anchored_continuation_enabled=True,
        independent_layout_development_execution=False,reused_development_layout=True,
        native_execution=True,model_training=False,planner_interface_adapter_enabled=True,
        frontier_transition_enabled=False,adapter_prefix_result_sha256=adapter_evidence.PREFIX_SHA)
    if any(launch[k]!=v for k,v in expected.items()):
        raise ValueError('fixed six-model native definition required')
    if (set(launch['assigned_model_states'])!={c[4] for c in CASES}
            or launch['robot_urdf_sha256']!=digest(URDF)):
        raise ValueError('all exact assigned models and unchanged robot required')
    adapter_evidence.verify_admission(launch['adapter_admission'],launch['source_sha256'])
    if launch['assigned_model_states']!=launch['adapter_admission']['assigned_model_states']:
        raise ValueError('original assigned model states must remain exact')
    verify_bound_inputs(launch['input_admission'],launch['source_sha256'])
    if full:
        a=launch['input_admission']
        if admit_inputs(a['correction_wait_result_sha256'],a['native_result_sha256'],launch['source_sha256'])!=a:
            raise ValueError('complete original input admission changed')
        if adapter_evidence.admit(launch['source_sha256'])!=launch['adapter_admission']:
            raise ValueError('complete adapter evidence admission changed')


def assigned_model(launch,case):
    if tuple(case) not in CASES:raise ValueError('one of the six preassigned native cases required')
    model,condition,variant=load_assigned(launch['input_admission']['correction_admission'],case[4])
    if (type(model) is not AllPhasePlannerModel or model.training
            or (condition,variant)!=(case[3],case[2])
            or state_digest(model.state_dict())!=launch['assigned_model_states'][case[4]]):
        raise ValueError('exact assigned corrected model state required')
    return model


def worker(case,launch_sha):
    name,index,variant,condition,model_name=case
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    record=dict(case=name,layout_index=index,variant=variant,condition=condition,model_name=model_name,
        status='ALL_PHASE_RESIDUAL_MAZE02_WORKER_FAILED',artifact_sha256={});started=time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT,{'launch.json':launch_sha});launch=read_json(OUTPUT,'launch.json');verify_inputs(launch)
            model=assigned_model(launch,case);before=state_digest(model.state_dict())
            result=collect(index,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                geometry=ArticulatedCollisionGeometry(URDF),episode_name=name,condition=condition,variant=variant)
            if state_digest(model.state_dict())!=before:raise ValueError('collection changed assigned model')
            bindings={name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(index,result)}
            verify_artifacts(OUTPUT,bindings);record.update(collection=result,artifact_sha256=dict(bindings))
            replay_model=assigned_model(launch,case)
            report=audit(index,result,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF),episode_name=name,condition=condition,variant=variant)
            audit_name=name+'_audit.json';write_json(OUTPUT/audit_name,report);bindings[audit_name]=digest(OUTPUT/audit_name)
            record['artifact_sha256']=dict(bindings)
            startup=compare_adapter_startup(case,OUTPUT/name)
            startup_name=name+'_startup_comparison.json';write_json(OUTPUT/startup_name,startup)
            bindings[startup_name]=digest(OUTPUT/startup_name);record['artifact_sha256']=dict(bindings)
            with np.load(OUTPUT/name/'physics_trace.npz',allow_pickle=False) as saved:
                readout=case_readout(report,result,saved['physics_contact'])
            readout_name=name+'_readout.json';write_json(OUTPUT/readout_name,readout)
            bindings[readout_name]=digest(OUTPUT/readout_name);record['artifact_sha256']=dict(bindings)
            verify_inputs(launch);verify_artifacts(OUTPUT,bindings)
            record.update(status=WORKER_STATUS,**{k:report[k] for k in OUTCOME_KEYS},
                model_state_sha256=before,model_state_unchanged=True,startup_comparison=startup,readout=readout,
                reused_development_layout=True,independent_layout_development_execution=False,
                head='direct_outcomes' if condition=='direct' else 'rollout_outcomes')
            require_case(case,record,report)
        except Exception as error:
            import traceback
            traceback.print_exc();record.update(status='ALL_PHASE_RESIDUAL_MAZE02_WORKER_FAILED',failure=repr(error))
    record.update(wall_s=time.perf_counter()-started,maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'),record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--correction-wait-result-sha256');parser.add_argument('--native-result-sha256')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--source-preflight-only',action='store_true');modes.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive complete cohort; no retry/resume')
    sources=prepared_sources((SOURCE,PROTOCOL,*TESTS));resources=hardware();resources_for(resources,6)
    if args.source_preflight_only:
        print('ALL_PHASE_RESIDUAL_MAZE02_SOURCE_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            original_native_live=corrected_wait.owner_live(NATIVE_OWNER),
            original_correction_live=corrected_wait.owner_live(CORRECTION_OWNER),
            native_execution=False,output_created=False,complete_input_admission_performed=False)),flush=True);return
    if not args.correction_wait_result_sha256 or not args.native_result_sha256:
        raise ValueError('both exact completed original result identities required')
    print('ALL_PHASE_ADAPTER_INPUT_ADMISSION_STARTED',len(sources),flush=True)
    adapter_admission=adapter_evidence.admit(sources)
    admission=admit_inputs(args.correction_wait_result_sha256,args.native_result_sha256,sources)
    old=read_json(reference.OUTPUT,'launch.json')
    keys=('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules','renderer_environment')
    launch={k:old[k] for k in keys};states={}
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for case in CASES:
        model,c,v=load_assigned(admission['correction_admission'],case[4])
        if (c,v)!=(case[3],case[2]):raise ValueError('all assigned model treatments required')
        states[case[4]]=state_digest(model.state_dict());del model
    launch.update(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),input_admission=admission,
        adapter_admission=adapter_admission,planner_interface_adapter_enabled=True,
        frontier_transition_enabled=False,adapter_prefix_result_sha256=adapter_evidence.PREFIX_SHA,
        planned_cases=[list(c) for c in CASES],implementation_class=IMPLEMENTATION,
        scene_specification=specification(2),public_mission=public_mission(2),robot_urdf_sha256=digest(URDF),
        assigned_model_states=states,navigation_ticks=NAVIGATION_TICKS,shared_outbound_return_budget=True,
        native_scene_workers=1,opencv_threads=1,blas_threads=1,maximum_tasks_per_process=1,
        physics_paused_during_compute=True,renderer_capture_witnesses_enabled=True,
        fresh_controller_and_memory_per_case=True,unplanned_controller_or_model_changes_between_cases=False,
        checkpoint_selection_performed=False,native_execution=True,model_training=False,
        measured_floor_transport_enabled=True,residual_first_interval_feasibility_fallback_enabled=True,
        residual_hold_feasibility_enabled=True,residual_anchored_continuation_enabled=True,
        independent_layout_development_execution=False,reused_development_layout=True,
        model_rgb_ablation_keeps_controller_rgbd=True,direct_is_nonpredictive_baseline=False,
        navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False)
    verify_inputs(launch);resources=hardware();launch['hardware']=resources
    launch['resource_admission']=resources_for(resources,6)
    if args.preflight_only:
        print('ALL_PHASE_RESIDUAL_MAZE02_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            complete_input_admission_performed=True,native_execution=False,output_created=False)),flush=True);return
    require_native_idle();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);launch_sha=digest(OUTPUT/'launch.json')
    print('ALL_PHASE_RESIDUAL_MAZE02_LAUNCHED',launch_sha,flush=True);started=time.perf_counter();records=[];bindings={}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for slot,case in enumerate(CASES):
                verify_inputs(launch);resources=hardware();resources_for(resources,len(CASES)-slot);require_native_idle()
                monitor.write(json.dumps(dict(case=case[0],stage='before_case',**resources))+'\n');monitor.flush()
                with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                    future=pool.submit(worker,case,launch_sha)
                    while True:
                        monitor.write(json.dumps(dict(case=case[0],elapsed_s=time.perf_counter()-started,**hardware()))+'\n');monitor.flush()
                        done,_=wait([future],timeout=15,return_when=FIRST_COMPLETED)
                        if done:record=future.result();break
                print('ALL_PHASE_RESIDUAL_MAZE02_CASE_TERMINAL',case[0],record['status'],
                    record.get('verified_round_trip'),record.get('failure'),flush=True)
                if record['status']!=WORKER_STATUS:raise ValueError('incomplete collection/raw audit; preserve all evidence')
                require_case(case,record,read_json(OUTPUT,case[0]+'_audit.json'))
                records.append(record);bindings.update(record['artifact_sha256'])
                for suffix in ('_worker.log','_worker_terminal.json'):bindings[case[0]+suffix]=digest(OUTPUT/(case[0]+suffix))
                write_json(OUTPUT/(case[0]+'_parent_completion.json'),dict(case=case[0],
                    worker_terminal_sha256=bindings[case[0]+'_worker_terminal.json'],
                    verified_round_trip=record['verified_round_trip'],scientific_success_required=False))
                n=case[0]+'_parent_completion.json';bindings[n]=digest(OUTPUT/n)
        reports=[read_json(OUTPUT,c[0]+'_audit.json') for c in CASES]
        summary=complete_cohort(records,reports)
        verify_inputs(launch,full=True)
        for n in ('launch.json','resource_monitor.jsonl'):bindings[n]=digest(OUTPUT/n)
        verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,conditions=records,wall_s=time.perf_counter()-started,
            correction_wait_result_sha256=args.correction_wait_result_sha256,
            predecessor_native_result_sha256=args.native_result_sha256,
            original_failed_cohort_result_sha256=adapter_evidence.ORIGINAL_SHA,
            adapter_prefix_result_sha256=adapter_evidence.PREFIX_SHA,
            planner_interface_adapter_enabled=True,frontier_transition_enabled=False,**summary))
        print('ALL_PHASE_RESIDUAL_MAZE02_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_FAILURE',
            reason=repr(error),completed_cases=[r['case'] for r in records],automatic_retry=False));raise


if __name__=='__main__':main()
