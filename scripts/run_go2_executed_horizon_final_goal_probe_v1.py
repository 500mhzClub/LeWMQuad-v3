"""Prospective native execution with complete primary-plus-auxiliary sensing."""
import argparse
import contextlib
import json
import multiprocessing
import random
import resource
import time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import cv2
import numpy as np
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.geometry_progress_family_runtime_development import preflight,verify
from scripts.executed_horizon_final_goal_episode_development import collect,artifacts
from scripts.executed_horizon_final_goal_audit_development import audit
from scripts.training_translation_bias_model_admission_development import admit,load_assigned
from scripts.fit_go2_training_translation_bias_v1 import OUTPUT as CORRECTION,FIT_SHA
from scripts.run_go2_observation_horizon_fits_v1 import OUTPUT as FITS
from scripts.replay_go2_executed_horizon_final_goal_prefix_v1 import OUTPUT as INTEGRATION
from scripts.capture_go2_auxiliary_downward45_depth_prefix_v1 import OUTPUT as SENSOR_PREFIX
from scripts.read_go2_training_translation_bias_v1 import OUTPUT as READOUT
from scripts.probe_go2_moving_action_switch_scaling_v1 import OUTPUT as SCALING
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

PREVIOUS=BASE/'go2_exact_mission_target_goal_probe_v1_attempt_001'
DIAGNOSTIC=BASE/'go2_exact_mission_target_goal_readout_v1_attempt_001'
PREVIOUS_SHA='d9e0cef6a7e66459a5cc6d21cc6cf6e638666a33a12cdcfb8b2364f1477bd125'
DIAGNOSTIC_SHA='d13c20027d16f48e146a542c443af61e205e02420bcf27e7e7cfa2ace7d61a43'
OUTPUT=BASE/'go2_executed_horizon_final_goal_probe_v1_attempt_001'
PRIOR=BASE/'go2_overlap_retention_goal_probe_v1_attempt_001'
PRIOR_READOUT=BASE/'go2_overlap_retention_goal_readout_v1_attempt_001'
PRIOR_SHA='fa67b645ae0439b803adce6d1a73c90e29862daeb8f8cd9b037183acd1477765'
PRIOR_READOUT_SHA='4f38b4fc8cca26808eb248db5d6484c5b68f29c73c2e14ea2f2a5dd82bc76799'
SENSOR_PREFIX_SHA='92f32aad3d2bde466a073df6410bf7060dfc9a1a1cdf5b9520fabcf8eb4268e4'
SCALING_SHA='b593dfe4b8924d9bf8cdd9e6e382a3772575e332444d56c7dbb8aed20031ef79'
PROTOCOL='docs/go2_executed_horizon_final_goal_probe_v1_2026-09-08.md'
TRIAL='family_episode_039'
CASES=[(f'full_{condition}_{TRIAL}',TRIAL,'full',condition,f'seed_2026091001_full_{condition}')
    for condition in ('jepa','direct')]
random.Random(2026091501).shuffle(CASES)


def causal_prefix(directory):
    reader=IntentReturnRGBDReplay(directory)
    if len(reader.frames)<4:raise ValueError('complete actual four-frame warmup prefix required')
    with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
        if len(archive['timestamp_s'])<900:raise ValueError('complete native warmup samples required')
        values={'native/'+k:fingerprint(archive[k][:900]) for k in archive.files}
    for i in range(4):
        values['packet/'+str(i)]=fingerprint(reader.packet(i))
        with np.load(directory/f'auxiliary_depth_{i:04d}.npz',allow_pickle=False) as z:
            values['auxiliary/'+str(i)]=fingerprint({k:z[k] for k in ('depth_m','valid')})
    return values


def worker(case,launch_sha):
    name,trial,variant,condition,model_name=case
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    terminal=dict(case=name,trial=trial,variant=variant,condition=condition,model_name=model_name,
        status='EXECUTED_HORIZON_FINAL_GOAL_WORKER_FAILED',artifact_sha256={});started=time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT,{'launch.json':launch_sha});launch=read_json(OUTPUT,'launch.json');verify(launch)
            assert launch['planned_cases']==[list(c) for c in CASES] and tuple(case) in CASES
            assert launch['output_root']==str(OUTPUT) and digest(URDF)==launch['robot_urdf_sha256']
            model,c,v=load_assigned(launch['correction_admission'],model_name);assert (c,v)==(condition,variant)
            before=state_digest(model.state_dict())
            result=collect(trial,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                geometry=ArticulatedCollisionGeometry(URDF),persistent=True,episode_name=name,condition=condition,variant=variant)
            assert state_digest(model.state_dict())==before
            bindings={name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(trial,result)};verify_artifacts(OUTPUT,bindings)
            replay_model,c,v=load_assigned(launch['correction_admission'],model_name);assert (c,v)==(condition,variant)
            report=audit(trial,result,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF),persistent=True,episode_name=name,condition=condition,variant=variant)
            audit_name=name+'_audit.json';write_json(OUTPUT/audit_name,report);bindings[audit_name]=digest(OUTPUT/audit_name)
            prefix=causal_prefix(OUTPUT/name)
            if prefix!=launch['prior_causal_prefix']:raise ValueError('physical/public prefix changed before model control')
            verify(launch);verify_artifacts(OUTPUT,bindings)
            verify_artifacts(FITS,{'result.json':launch['fit_result_sha256'],**launch['correction_admission']['base_admission']['fit_artifact_sha256']})
            assert digest(URDF)==launch['robot_urdf_sha256']
            terminal.update(status='EXECUTED_HORIZON_FINAL_GOAL_COLLECTED_AND_RAW_AUDITED',artifact_sha256=bindings,
                collection=result,goal=report['goal'],selection_count=report['selection_count'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                original_physical_and_public_warmup_prefix_exact=True,model_state_unchanged=True)
        except Exception as error:
            import traceback
            traceback.print_exc();terminal['failure']=repr(error)
    terminal.update(wall_s=time.perf_counter()-started,maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'),terminal);return terminal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--correction-result-sha256',required=True)
    parser.add_argument('--correction-readout-sha256',required=True)
    parser.add_argument('--integration-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive two-case native probe; no retry/resume')
    verify_artifacts(PREVIOUS,{'result.json':PREVIOUS_SHA});previous=read_json(PREVIOUS,'result.json')
    assert previous['status']=='EXACT_MISSION_TARGET_GOAL_PROBE_COMPLETE' and previous['cases']==[list(c) for c in CASES]
    previous_ids={'result.json':PREVIOUS_SHA,**previous['artifact_sha256']};verify_artifacts(PREVIOUS,previous_ids)
    verify_artifacts(DIAGNOSTIC,{'result.json':DIAGNOSTIC_SHA});diagnostic=read_json(DIAGNOSTIC,'result.json')
    assert diagnostic['status']=='EXACT_MISSION_TARGET_GOAL_READOUT_COMPLETE'
    diagnostic_ids={'result.json':DIAGNOSTIC_SHA,'launch.json':diagnostic['launch_sha256']};verify_artifacts(DIAGNOSTIC,diagnostic_ids)
    verify(read_json(PREVIOUS,'launch.json'));verify(read_json(DIAGNOSTIC,'launch.json'))
    verify_artifacts(INTEGRATION,{'result.json':args.integration_result_sha256});integration=read_json(INTEGRATION,'result.json')
    assert integration['status']=='EXECUTED_HORIZON_FINAL_GOAL_PREFIX_COMPLETE' and len(integration['conditions'])==2
    assert all(r['causal_observations_maps_forecasts_and_nominal_checks_exact'] and r['surface_checks_exact'] and r['only_final_goal_pose_scoring_horizon_changed'] and not r['controller_failures'] and r['model_state_unchanged'] and r['exact_fresh_replay_passes']==2 for r in integration['conditions'])
    assert any(r['first_requested_command_difference'] is not None for r in integration['conditions'])
    integration_ids={'result.json':args.integration_result_sha256,**integration['artifact_sha256']};verify_artifacts(INTEGRATION,integration_ids)
    verify_artifacts(SENSOR_PREFIX,{'result.json':SENSOR_PREFIX_SHA});sensor_prefix=read_json(SENSOR_PREFIX,'result.json')
    assert sensor_prefix['status']=='AUXILIARY_DOWNWARD45_DEPTH_PREFIX_COMPLETE' and sensor_prefix['raw_audit_pass']
    sensor_ids={'result.json':SENSOR_PREFIX_SHA,**sensor_prefix['artifact_sha256']};verify_artifacts(SENSOR_PREFIX,sensor_ids)
    admitted=admit(args.correction_result_sha256)
    integration_launch=read_json(INTEGRATION,'launch.json');verify(integration_launch)
    assert integration_launch['correction_admission']==admitted
    verify_artifacts(READOUT,{'result.json':args.correction_readout_sha256});readout=read_json(READOUT,'result.json')
    if readout['status']!='TRAINING_TRANSLATION_BIAS_READOUT_COMPLETE' or readout['models']!=18 or readout['trained_heads']!=30 or readout['correction_result_sha256']!=args.correction_result_sha256:
        raise ValueError('complete full-fit readout required')
    readout_ids={'result.json':args.correction_readout_sha256,**readout['artifact_sha256']};verify_artifacts(READOUT,readout_ids)
    fit_readout_launch=read_json(READOUT,'launch.json');verify(fit_readout_launch)
    if fit_readout_launch['correction_admission']!=admitted:raise ValueError('unchanged complete fit admission required')
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA});prior=read_json(PRIOR,'result.json')
    if prior['status']!='OVERLAP_RETENTION_GOAL_PROBE_COMPLETE' or len(prior['conditions'])!=1:raise ValueError('exact original affected probe required')
    prior_ids={'result.json':PRIOR_SHA,**prior['artifact_sha256']};verify_artifacts(PRIOR,prior_ids);verify(read_json(PRIOR,'launch.json'))
    verify_artifacts(PRIOR_READOUT,{'result.json':PRIOR_READOUT_SHA});prior_readout=read_json(PRIOR_READOUT,'result.json')
    prior_readout_ids={'result.json':PRIOR_READOUT_SHA,'launch.json':prior_readout['launch_sha256'],
        **prior_readout.get('artifact_sha256',{})}
    verify_artifacts(PRIOR_READOUT,prior_readout_ids)
    verify(read_json(PRIOR_READOUT,'launch.json'))
    verify_artifacts(SCALING,{'result.json':SCALING_SHA});scaling=read_json(SCALING,'result.json')
    if scaling['selected_workers']!=4 or not scaling['all_execution_and_pixel_signatures_equal']:
        raise ValueError('passing exact four-process native collection benchmark required')
    scaling_ids={'result.json':SCALING_SHA,'launch.json':scaling['launch_sha256']}
    verify_artifacts(SCALING,scaling_ids);verify(read_json(SCALING,'launch.json'))
    scaling_phase_ids={}
    for root,ids in scaling['phase_result_sha256'].items():
        verify_artifacts(Path(root),ids)
        phase=read_json(Path(root),'result.json')
        scaling_phase_ids[root]={**ids,**phase['artifact_sha256']}
        verify_artifacts(Path(root),scaling_phase_ids[root])
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,
        seed_paths=('scripts/run_go2_executed_horizon_final_goal_probe_v1.py',
            'lewm/tests/test_executed_horizon_final_goal_native_scope_development.py',
            'docs/go2_executed_horizon_final_goal_prefix_result_2026-09-08.md',
            *integration['source_sha256'],*sensor_prefix['source_sha256'],
            'docs/go2_eight_step_planning_goal_probe_result_2026-09-08.md',
            *previous['source_sha256'],*diagnostic['source_sha256'],
            'docs/go2_training_translation_bias_result_2026-09-08.md',
            *prior['source_sha256'],*prior_readout['source_sha256'],*readout['source_sha256']),
        planned_trials=[TRIAL],workers=4,storage_bytes=8*1024**3)
    for source in (prior,prior_readout,readout,previous,diagnostic,integration,sensor_prefix):
        for name,h in source['source_sha256'].items():assert launch['source_sha256'].get(name)==h,name
    launch.update(integration_artifact_sha256=integration_ids,sensor_prefix_artifact_sha256=sensor_ids,
        previous_probe_artifact_sha256=previous_ids,previous_readout_artifact_sha256=diagnostic_ids,
        correction_admission=admitted,fit_result_sha256=FIT_SHA,correction_readout_sha256=readout_ids,
        correction_result_sha256=args.correction_result_sha256,correction_artifact_sha256=admitted['correction_artifact_sha256'],
        prior_probe_sha256=prior_ids,prior_readout_sha256=PRIOR_READOUT_SHA,scaling_result_sha256=SCALING_SHA,
        prior_readout_artifact_sha256=prior_readout_ids,scaling_artifact_sha256=scaling_ids,
        scaling_phase_artifact_sha256=scaling_phase_ids,
        prior_causal_prefix=causal_prefix(SENSOR_PREFIX/'sensor_prefix'),planned_cases=CASES,
        order_seed=2026091501,robot_urdf_path=str(URDF),robot_urdf_sha256=digest(URDF),
        preflight_capacity_workers=4,native_scene_workers=2,model_training=False,native_execution=True,
        experiment='exact final-goal potential scored at the executed 100-ms pose horizon; original 800-ms contact score and all surface/path checks retained; intermediate/view scoring, models, arrival and budgets unchanged',
        concurrency_reason='two independent one-thread scenes; prior four-process scaling, current visible-robot capture and RAM/storage envelopes checked',
        data_scope='one reused integration layout; zero independent novel mazes',final_checkpoint_per_method_fixed=True)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('EXECUTED_HORIZON_FINAL_GOAL_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    records=[];started=time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                pending={pool.submit(worker,case,digest(OUTPUT/'launch.json')):case for case in CASES}
                while pending:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,**hardware()))+'\n');monitor.flush()
                    done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                    for future in done:
                        case=pending.pop(future)
                        try:row=future.result()
                        except Exception as error:row=dict(case=case[0],status='EXECUTED_HORIZON_FINAL_GOAL_PROCESS_FAILURE',failure=repr(error))
                        records.append(row);print('EXECUTED_HORIZON_FINAL_GOAL_TERMINAL',case[0],row['status'],row.get('goal'),flush=True)
        if len(records)!=2 or any(r['status']!='EXECUTED_HORIZON_FINAL_GOAL_COLLECTED_AND_RAW_AUDITED' for r in records):
            raise ValueError('native infrastructure/raw audit failure; both launched cases retained')
        records.sort(key=lambda r:[c[0] for c in CASES].index(r['case']))
        bindings={n:h for r in records for n,h in r['artifact_sha256'].items()}
        for name in ('launch.json','resource_monitor.jsonl',*(c[0]+s for c in CASES for s in ('_worker.log','_worker_terminal.json'))):bindings[name]=digest(OUTPUT/name)
        verify(launch);verify_artifacts(OUTPUT,bindings);verify_artifacts(PRIOR,prior_ids);verify_artifacts(READOUT,readout_ids)
        verify_artifacts(PREVIOUS,previous_ids);verify_artifacts(DIAGNOSTIC,diagnostic_ids)
        verify_artifacts(INTEGRATION,integration_ids);verify_artifacts(SENSOR_PREFIX,sensor_ids)
        verify_artifacts(PRIOR_READOUT,prior_readout_ids);verify_artifacts(SCALING,scaling_ids)
        for root,ids in scaling_phase_ids.items():verify_artifacts(Path(root),ids)
        verify_artifacts(FITS,{'result.json':FIT_SHA,**admitted['base_admission']['fit_artifact_sha256']})
        verify_artifacts(CORRECTION,admitted['correction_artifact_sha256'])
        result=dict(status='EXECUTED_HORIZON_FINAL_GOAL_PROBE_COMPLETE',cases=CASES,conditions=records,
            artifact_sha256=bindings,source_sha256=launch['source_sha256'],wall_s=time.perf_counter()-started,
            all_measurement_gates_pass=all(r['strict_physical_visibility_pass'] and not r['hard_measurement_failed_frames'] for r in records),
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and r['strict_physical_visibility_pass']
                and not r['hard_measurement_failed_frames'] for r in records),model_training=False,checkpoint_selection_performed=False,
            independent_maze_evaluation=False,navigation_qualified=False,hardware_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print('EXECUTED_HORIZON_FINAL_GOAL_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_EXECUTED_HORIZON_FINAL_GOAL_FAILURE',reason=repr(error),
            completed_cases=[r['case'] for r in records]));raise


if __name__=='__main__':main()
