"""Fixed JEPA/supervised-rollout prefixes on all three completed independent mazes."""
import argparse
import gc
from itertools import islice
import json
import shutil
import time
import cv2
import numpy as np
import torch
from lewm.matched_rollout_objective_admission_development import admit_objective_pair, NAMES, CONDITIONS, BASE_STATE
from lewm.matched_objective_prefix_development import compare_step, MAX_FRAMES
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.run_go2_independent_floor_transport_mazes_v1 import OUTPUT as INPUT, verify_inputs as verify_cohort
from scripts.training_translation_bias_model_admission_development import load_assigned, FITS
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_matched_objective_prefixes_v1_attempt_001'
PROTOCOL='docs/go2_matched_objective_prefixes_v1_2026-09-09.md'
INPUT_SHA='a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720'
LAYOUTS=(1,2,3)
MAX_OUTPUT_BYTES=512*1024**2


def verify_inputs(launch):
    verify(launch); verify_artifacts(INPUT,launch['replay_input_bindings'])
    verify_cohort(read_json(INPUT,'launch.json')); verify_artifacts(FITS,launch['paired_fit_bindings'])


def admit_cases(result,old):
    if (result['status']!='INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE'
            or result['all_fixed_cases_executed'] is not True
            or [r['layout_index'] for r in result['conditions']]!=list(LAYOUTS)
            or old['model_state_sha256']!=MODEL_STATE
            or old['implementation_class']!='MeasuredFloorTransportController'):
        raise ValueError('complete original fixed three-maze controller cohort required')
    for index,record in zip(LAYOUTS,result['conditions'],strict=True):
        name=f'full_jepa_novel_maze_{index:02d}'
        if (record['case']!=name or record['status']!='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
                or [name,index,'full','jepa',NAMES[0]] not in old['planned_cases']):
            raise ValueError('fixed completed JEPA case required')
        audit=read_json(INPUT,name+'_audit.json')
        for key in ('raw_sensor_reconstruction_pass','additional_auxiliary_rgb_reconstructed',
                    'raw_model_command_replay_pass','raw_command_audit_pass','model_state_unchanged'):
            if audit[key] is not True: raise ValueError('original complete raw audit required: '+key)


def load_pair(admission):
    models=[]; identities=[]
    for i,(name,condition) in enumerate(zip(NAMES,CONDITIONS,strict=True)):
        model,c,v=load_assigned(admission,name)
        if (c,v)!=(condition,'full') or model.training or model.base.training:
            raise ValueError('exact assigned full-RGB evaluation-only objective model required')
        if state_digest(model.base.state_dict())!=BASE_STATE[i]:
            raise ValueError('exact objective-specific base neural state required')
        heads=admission['coefficients'][name]['heads']
        if set(model.corrected_heads)!=set(heads): raise ValueError('same trained heads required')
        for head,row in heads.items():
            actual=getattr(model,head+'_xy_bias').cpu().numpy()
            if actual.dtype!=np.float32 or not np.array_equal(actual,np.asarray(row['applied_bias_xy_m'],np.float32)):
                raise ValueError('each assigned training-only intercept must match exactly')
        state=state_digest(model.state_dict())
        if i==0 and state!=MODEL_STATE: raise ValueError('original corrected JEPA state required')
        models.append(model); identities.append(dict(name=name,condition=condition,base_state_sha256=BASE_STATE[i],
            corrected_state_sha256=state,head='rollout_outcomes',model_training=False))
    return models,identities


def replay_case(launch,index):
    models,identities=load_pair(launch['correction_admission'])
    controllers=[MeasuredFloorTransportController(model,ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS,public_mission=public_mission(index),condition=condition,
        variant='full',persistent=True) for model,condition in zip(models,CONDITIONS,strict=True)]
    if any(c.selector.head!='rollout_outcomes' for c in controllers): raise ValueError('same rollout inference head required')
    corrections={c:launch['correction_admission']['coefficients'][n]['heads']['rollout_outcomes']['applied_bias_xy_m']
        for n,c in zip(NAMES,CONDITIONS,strict=True)}
    name=f'full_jepa_novel_maze_{index:02d}'; source=INPUT/name; output=OUTPUT/name; output.mkdir()
    reader=IntentReturnRGBDReplay(source); acquisitions=read_json(source,'auxiliary_camera_audit.json')
    tape=read_json(source,'command_tape.json')
    if not len(reader.frames)==len(acquisitions)==len(tape)+1 or len(tape)<MAX_FRAMES[index]:
        raise ValueError('complete actual paired observations and command tape required')
    count=banks=0; first_prediction=None; last=check=None
    with writer(output) as append:
        for i,original in enumerate(islice(read_rows(source),MAX_FRAMES[index])):
            if original['observation_index']!=i or original['pre_sample_index']!=749+50*i or not tape[i]['completed']:
                raise ValueError('exact completed original observation/command endpoints required')
            if shutil.disk_usage(BASE).free<RESERVE_BYTES+MAX_OUTPUT_BYTES: raise ValueError('prefix reserve unavailable')
            policy,depth,fast,now=reader.packet(i)
            image,auxiliary=packet(source,i,policy,public_acquisition(acquisitions[i]),now_ns=now)
            inputs=fingerprint((policy,depth,fast,auxiliary,image,now)); decisions=[]
            for controller in controllers:
                decisions.append(json.loads(json.dumps(controller.observe(policy,depth,fast,now_ns=now,
                    auxiliary_depth=auxiliary,auxiliary_rgb=image),allow_nan=False)))
                if fingerprint((policy,depth,fast,auxiliary,image,now))!=inputs:
                    raise ValueError('model arm mutated public observation inputs')
            row=dict(tick=i,jepa_decision=decisions[0],decision=decisions[1],
                original_requested_command=tape[i]['requested_command'],public_input_arrays_unchanged=True)
            try:
                check=compare_step(original['decision'],*decisions,tape[i]['requested_command'],
                    frame=i,layout=index,corrections=corrections)
            except Exception as error:
                append(row|dict(comparison_failure=repr(error))); raise
            append(row|dict(comparison=check)); count+=1; last=decisions
            banks+=int(check['both_full_forecast_banks_present'])
            if check['raw_prediction_changed'] and first_prediction is None: first_prediction=i
            if (output/DECISIONS).stat().st_size>MAX_OUTPUT_BYTES//(2*len(LAYOUTS)):
                raise ValueError('bounded per-case receipt output exceeded')
            if i%32==0: print('MATCHED_OBJECTIVE_PREFIX_FRAME',index,i,flush=True)
            if check['stop']: break
    if not check or not check['stop']: raise ValueError('must stop at first command/terminal change or original terminal')
    for model,identity in zip(models,identities,strict=True):
        if (state_digest(model.state_dict())!=identity['corrected_state_sha256']
                or any(p.grad is not None for p in model.parameters())):
            raise ValueError('both objective states unchanged and no gradients required')
    return dict(case=name,layout_index=index,frames=count,maximum_frames=MAX_FRAMES[index],models=identities,
        first_prediction_difference=first_prediction,paired_forecast_banks=banks,
        first_requested_command_difference=count-1 if check['requested_command_changed'] else None,
        first_terminal_difference=count-1 if check['terminal_changed'] else None,
        jepa_final_requested_command=last[0]['requested_command'],supervised_final_requested_command=last[1]['requested_command'],
        jepa_terminal=last[0]['terminal'],supervised_terminal=last[1]['terminal'],
        complete_original_jepa_decisions_exact=True,shared_observed_state_exact=True,
        prior_actual_commands_exact=True,prior_commands_compared=count-1,
        stopped_before_following_a_changed_command=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,model_states_unchanged=True,model_training=False,
        native_execution=False,unexecuted_outcomes_inferred=False,jepa_advantage_established=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight-only',action='store_true'); args=parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive matched objective replay required')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA}); result=read_json(INPUT,'result.json')
    ids={'result.json':INPUT_SHA,**result['artifact_sha256']}; verify_artifacts(INPUT,ids)
    old=read_json(INPUT,'launch.json'); admit_cases(result,old); admission=old['correction_admission']
    fit_ids={n+'_fit.json':admission['base_admission']['fit_artifact_sha256'][n+'_fit.json'] for n in NAMES}
    verify_artifacts(FITS,fit_ids); paired=admit_objective_pair(admission,{n:read_json(FITS,n+'_fit.json') for n in NAMES})
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_matched_objective_prefixes_v1.py',
        'lewm/tests/test_matched_objective_prefix_development.py',
        'lewm/tests/test_matched_rollout_objective_admission_development.py'),old['source_sha256'])
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),replay_input_bindings=ids,
        correction_admission=admission,paired_fit_bindings=fit_ids,objective_pair_admission=paired,
        fixed_layouts=list(LAYOUTS),maximum_case_frames=MAX_FRAMES,one_fresh_model_pair_per_case=True,
        same_controller_class='MeasuredFloorTransportController',same_inference_head='rollout_outcomes',
        model_training=False,native_execution=False,replay_workers=1,opencv_threads=1,blas_threads=1,
        memory_admission_bytes=12*1024**3,output_allowance_bytes=MAX_OUTPUT_BYTES,minimum_free_bytes=RESERVE_BYTES,
        concurrency_reason='one sequential CPU comparison beside at most one separately owned native scene')
    verify_inputs(launch); resources=hardware(); launch['hardware']=resources
    if resources['memory_available_bytes']<12*1024**3 or resources['artifact_free_bytes']<RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('matched replay resources unavailable')
    if args.preflight_only:
        print('MATCHED_OBJECTIVE_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            input_and_source_bindings_verified=True,model_loaded=False,output_created=False,native_execution=False)),flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); started=time.perf_counter(); reports=[]
    print('MATCHED_OBJECTIVE_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        for index in LAYOUTS:
            reports.append(replay_case(launch,index)); gc.collect()
            print('MATCHED_OBJECTIVE_PREFIX_CASE_COMPLETE',index,reports[-1]['frames'],flush=True)
        verify_inputs(launch)
        bindings={'launch.json':digest(OUTPUT/'launch.json')}
        for report in reports:
            n=report['case']+'/'+DECISIONS; bindings[n]=digest(OUTPUT/n)
        verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='MATCHED_OBJECTIVE_PREFIXES_V1_COMPLETE',source_sha256=sources,
            artifact_sha256=bindings,conditions=reports,all_fixed_cases_executed=True,model_loaded=True,
            model_training=False,native_execution=False,hardware_after=hardware(),wall_s=time.perf_counter()-started,
            jepa_advantage_established=False,navigation_qualified=False,goal_achieved=False))
        print('MATCHED_OBJECTIVE_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MATCHED_OBJECTIVE_PREFIX_FAILURE',reason=repr(error),
            completed_cases=reports)); raise


if __name__=='__main__': main()
