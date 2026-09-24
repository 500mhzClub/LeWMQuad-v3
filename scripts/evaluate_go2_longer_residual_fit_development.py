"""Compare fixed final fits on identical causal inputs from all six missions."""
from functools import lru_cache
import json
import math
import sys
import time
import cv2
import numpy as np
import torch

from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.terminal_translation_pulse_development import command_sequences
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import train_go2_longer_residual_fit_development as train
from scripts import nominal_motion_residual_snapshot_development as snapshot
from scripts import read_go2_action_forecast_bias_development as bias

OUTPUT=train.OUTPUT/'prediction_evaluation'
VARIANTS=('jepa_1200','jepa_6000','supervised_rollout_1200','supervised_rollout_6000')


def metrics(rows):
    if not rows:return dict(windows=0)
    return dict(windows=len(rows),models={name:dict(parts={part:dict(
        rmse_mm=float(np.sqrt(np.mean([np.dot(r['errors_mm'][name][part],r['errors_mm'][name][part]) for r in rows]))),
        mean_signed_xy_error_mm=np.mean([r['errors_mm'][name][part] for r in rows],axis=0).tolist())
        for part in bias.PARTS},yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean([
            r['yaw_error_rad'][name]**2 for r in rows]))))) for name in VARIANTS})


@torch.inference_mode()
def main():
    if OUTPUT.exists():raise ValueError('preserve complete or partial evaluation')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    plan=json.loads(train.PLAN.read_text());models={};identities={}
    for condition in train.CONDITIONS:
        for updates in (1200,6000):
            directory=(train.data.BASE/'go2_short_pulse_residual_matched_fits_v1_attempt_001'
                if updates==1200 else train.OUTPUT/condition)
            record=plan['old_models'][condition] if updates==1200 else json.loads((directory/'fit.json').read_text())
            assert record['configuration']['updates']==updates
            name=f'{condition}_{updates}'
            models[name]=snapshot.load_snapshot(directory,record['filename'],sha256=record['sha256'],
                expected_binding=record['binding'],expected_config=record['configuration']).model
            identities[name]=dict(directory=str(directory),sha256=record['sha256'],model_sha256=record['model_sha256'])
    OUTPUT.mkdir();began=time.monotonic()
    labels=json.loads((bias.OUTPUT/'result.json').read_text())
    assert labels['sources']==[str(train.BASE/r) for r in plan['evaluation_roots']]
    all_rows=[];runs=[];conflicts=[];largest_reference_difference=0.
    for number,root in enumerate(bias.SOURCES,1):
        read=lambda name:json.loads((root/name).read_text())
        windows={r['frame']:r for r in labels['rows'] if r['run']==number}
        plans={r['frame']:r for r in read('planning.json') if 'selection' in r}
        old_arm=read('launch.json')['training_condition'];reader=NoisyPublicReplay(root/'native')
        packet=lru_cache(maxsize=8)(reader.policy_packet)
        camera={r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
        with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:physics=data['base_pose_world'].copy()
        extra=(504,508,512,516,520,524,528) if number==6 else ()
        rows=[]
        for ordinal,frame in enumerate(sorted(set(windows)|set(extra)),1):
            p=plans[frame];c=p['motion_correction']
            history=causal_history_tensors([packet(i) for i in range(frame-3,frame+1)],p['measured_ns'])
            inputs=delayed_candidate_inputs(history,p['committed_prefix'],delay_ticks=3,commit_ticks=4)
            if c['terminal_translation_pulse']:
                inputs['known_action_blocks']=torch.as_tensor(command_sequences(p['committed_prefix'],pulse=True)[:,:,None],
                    dtype=torch.float32)/torch.tensor([.3,1.,.5])
            predictions={name:model(**inputs)['rollout_outcomes'].cpu().numpy() for name,model in models.items()}
            assert all(value.shape==(6,8,5) and np.isfinite(value).all() for value in predictions.values())
            reference=np.asarray(c['upstream_prediction_for_yaw_ablation'])[:,:,:4]
            original=predictions[f'{old_arm}_1200'][:,:,:4]
            np.testing.assert_allclose(original,reference,rtol=0,atol=1e-6)
            largest_reference_difference=max(largest_reference_difference,float(np.max(np.abs(original-reference))))
            if frame in extra:conflicts.append(dict(frame=frame,predictions={k:v[:,:,:4].tolist() for k,v in predictions.items()}))
            if frame in windows:
                w=windows[frame];index=ACTIONS.index(w['action']);errors={};yaw_errors={}
                endpoints=[rotation_xyzw(physics[camera[f]['physical_sample_index'],3:]) for f in (frame,frame+7)]
                actual_yaw=math.atan2(endpoints[1][1,0],endpoints[1][0,0])-math.atan2(endpoints[0][1,0],endpoints[0][0,0])
                for name,prediction in predictions.items():
                    parts=bias.components(prediction[index,:,:2])
                    errors[name]={part:(1000*(np.asarray(parts[part])-w['actual_xy_m'][part])).tolist() for part in bias.PARTS}
                    delta=math.atan2(prediction[index,6,2],prediction[index,6,3])-actual_yaw
                    yaw_errors[name]=math.atan2(math.sin(delta),math.cos(delta))
                rows.append(dict(run=number,frame=frame,action=w['action'],prefix_group=w['prefix_group'],
                    recorded_arm=old_arm,errors_mm=errors,yaw_error_rad=yaw_errors))
            if ordinal%300==0:print('LONGER_EVALUATION',number,ordinal,flush=True)
        runs.append(dict(run=number,root=str(root),recorded_arm=old_arm,total=metrics(rows),
            by_action={action:metrics([r for r in rows if r['action']==action]) for action in ACTIONS}))
        all_rows.extend(rows);train.write(OUTPUT/f'run_{number:02d}.json',dict(summary=runs[-1],rows=rows))
        print('LONGER_EVALUATION_RUN_COMPLETE',number,len(rows),flush=True)
    result=dict(schema='longer_residual_prediction_evaluation.v1',model_identities=identities,
        runs=runs,total=metrics(all_rows),by_action={a:metrics([r for r in all_rows if r['action']==a]) for a in ACTIONS},
        maximum_original_forecast_difference=largest_reference_difference,wall_s=time.monotonic()-began,
        original_forecasts_reproduced_within_1e_6=True,all_six_preselected_runs_included=True,
        matched_input_windows_for_all_four_models=True,matched_requested_command_sequence_ms=700,
        original_training_data_only=True,navigation_reexecuted=False,
        RGB_and_body_and_command_history_only_model_inputs=True,depth_not_loaded=True,
        native_state_evaluator_only=True,overlapping_windows_not_independent=True,
        exposed_development_evaluation=True,independent_generalization_established=False,
        unexecuted_conflict_forecasts=conflicts)
    train.write(OUTPUT/'result.json',result)
    print(json.dumps(result['total'],indent=2),flush=True)


def compare_conflict_forecasts():
    from lewm.fine_stored_obstacle_routing_development import cached_clearance
    result=json.loads((OUTPUT/'result.json').read_text());root=bias.SOURCES[-1]
    probe=json.loads((root/'view_recovery_forecast_clearance_probe_v1.json').read_text())
    references={r['frame']:r for r in probe['rows']}
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    rows=[]
    for record in result['unexecuted_conflict_forecasts']:
        frame=record['frame'];reference=references[frame]
        snap=probe['snapshots'][str(reference['map_frame'])]
        B=np.asarray(snap['map_from_initial']);p=B@np.asarray(poses[frame]['position_initial_body_m'])
        R=B@np.asarray(poses[frame]['rotation_initial_body_from_current_body'])
        clearance=cached_clearance(snap['fine_occupied']);models={}
        for name,prediction in record['predictions'].items():
            models[name]={}
            for action in ('left_turn','right_turn'):
                index=ACTIONS.index(action)
                points=p[:2]+np.vstack((np.zeros(2),np.asarray(prediction)[index,:,:2]))@R[:2,:2].T
                minimum=min(clearance.minimum(a,b) for a,b in zip(points[:-1],points[1:]))
                if name=='jepa_1200':
                    expected=next(c for c in reference['forecasts']['neural'] if c['action']==action)['minimum_m']
                    assert abs(minimum-expected)<1e-10
                models[name][action]=dict(minimum_m=minimum,full_48cm_reserve_clear=minimum>.48+1e-12)
        rows.append(dict(frame=frame,models=models))
    output=dict(schema='longer_fit_saved_conflict_clearance.v1',rows=rows,
        same_reconstructed_observed_maps=True,native_state_or_geometry_used=False,
        original_JEPA_clearances_reproduced=True,full_reserve_only=True,
        complete_selector_reexecuted=False,alternative_actions_executed=False,
        newly_eligible_action_safety_or_navigation_success_proven=False)
    train.write(OUTPUT/'saved_conflict_clearance.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['--conflicts']:compare_conflict_forecasts()
    elif not sys.argv[1:]:main()
    else:raise ValueError('use no arguments or --conflicts')
