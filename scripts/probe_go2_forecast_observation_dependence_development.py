"""Fixed RGB perturbations and command coverage on recent development inputs."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import time
import cv2
import numpy as np
import torch

from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.terminal_translation_pulse_development import command_sequences
from lewm.geometry_progress_pilot_development import ACTIONS
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import train_go2_longer_residual_fit_development as train
from scripts import nominal_motion_residual_snapshot_development as snapshot
from scripts import read_go2_action_forecast_bias_development as bias

OUTPUT=train.BASE/'go2_forecast_observation_dependence_v1_attempt_001'
PERTURBATIONS=('full','repeat_current_rgb','half_brightness','grayscale','zero_rgb')
STRIDE=10


def commands_key(commands):
    return tuple(tuple(round(float(v),6) for v in c) for c in commands)


def command_coverage(labels):
    training=train.data.load_training_rows();schedule=json.loads(train.SCHEDULE.read_text())
    counts={h:Counter() for h in (3,7)}
    for row in training:
        for horizon in counts:
            if len(row['known_commands'])>=horizon and all(t['motion_valid'] for t in row['targets'][:horizon]):
                counts[horizon][commands_key(row['known_commands'][:horizon])]+=schedule['context_draw_counts'][row['sample_id']]
    rows=[]
    for number,root in enumerate(bias.SOURCES,1):
        plans={p['frame']:p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p}
        for window in (r for r in labels if r['run']==number):
            p=plans[window['frame']]
            commands=command_sequences(p['committed_prefix'],pulse=p['motion_correction']['terminal_translation_pulse'])[ACTIONS.index(window['action'])]
            rows.append(dict(run=number,frame=window['frame'],action=window['action'],
                matching_training_draws={str(h):counts[h][commands_key(commands[:h])] for h in counts}))
    return dict(training_contexts=len(training),training_draws=7200,command_rounding_decimals=6,
        full_recent_matched_population_windows=len(rows),rows=rows,
        by_action={a:dict(windows=sum(r['action']==a for r in rows),
            covered_prefix_3=sum(r['action']==a and r['matching_training_draws']['3']>0 for r in rows),
            covered_sequence_7=sum(r['action']==a and r['matching_training_draws']['7']>0 for r in rows)) for a in ACTIONS},
        matches_require_motion_labels_for_every_compared_horizon=True,
        identical_command_sequence_does_not_establish_observation_state_coverage=True)


def summarize(rows):
    if not rows:return dict(windows=0)
    return dict(windows=len(rows),conditions={name:{variant:dict(
        xy_rmse_mm=float(np.sqrt(np.mean([r['errors_mm'][name][variant]**2 for r in rows]))),
        rms_forecast_change_mm=float(np.sqrt(np.mean([r['shift_mm'][name][variant]**2 for r in rows]))))
        for variant in PERTURBATIONS} for name in rows[0]['errors_mm']})


@torch.inference_mode()
def main():
    if OUTPUT.exists():raise ValueError('preserve complete or partial diagnostic')
    OUTPUT.mkdir();began=time.monotonic()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    labels=json.loads((bias.OUTPUT/'result.json').read_text())['rows']
    coverage=command_coverage(labels);train.write(OUTPUT/'command_coverage.json',coverage)
    plan=json.loads(train.PLAN.read_text());models={}
    for condition in train.CONDITIONS:
        for updates in (1200,6000):
            directory=(train.data.BASE/'go2_short_pulse_residual_matched_fits_v1_attempt_001'
                if updates==1200 else train.OUTPUT/condition)
            record=plan['old_models'][condition] if updates==1200 else json.loads((directory/'fit.json').read_text())
            models[f'{condition}_{updates}']=snapshot.load_snapshot(directory,record['filename'],sha256=record['sha256'],
                expected_binding=record['binding'],expected_config=record['configuration']).model
    all_rows=[];runs=[]
    for number,root in enumerate(bias.SOURCES,1):
        selected=sorted([r for r in labels if r['run']==number],key=lambda r:r['frame'])[::STRIDE]
        plans={p['frame']:p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p}
        arm=json.loads((root/'launch.json').read_text())['training_condition']
        reader=NoisyPublicReplay(root/'native');packet=lru_cache(maxsize=8)(reader.policy_packet);rows=[]
        for window in selected:
            frame=window['frame'];p=plans[frame]
            history=causal_history_tensors([packet(i) for i in range(frame-3,frame+1)],p['measured_ns'])
            inp=delayed_candidate_inputs(history,p['committed_prefix'],delay_ticks=3,commit_ticks=4)
            if p['motion_correction']['terminal_translation_pulse']:
                inp['known_action_blocks']=torch.as_tensor(command_sequences(p['committed_prefix'],pulse=True)[:,:,None],
                    dtype=torch.float32)/torch.tensor([.3,1.,.5])
            rgb=inp['observation_history']['rgb'];gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,None,:,None,None]).sum(2,keepdim=True).expand_as(rgb)
            variants=dict(full=rgb,repeat_current_rgb=rgb[:,-1:].expand_as(rgb),
                half_brightness=rgb*.5,grayscale=gray,zero_rgb=torch.zeros_like(rgb))
            predictions={name:{} for name in models}
            for variant,pixels in variants.items():
                changed=inp | dict(observation_history=inp['observation_history'] | dict(rgb=pixels))
                for name,model in models.items():
                    value=model(**changed)['rollout_outcomes'].cpu().numpy()
                    assert value.shape==(6,8,5) and np.isfinite(value).all()
                    predictions[name][variant]=value
            np.testing.assert_array_equal(predictions[f'{arm}_1200']['full'][:,:,:4],
                np.asarray(p['motion_correction']['upstream_prediction_for_yaw_ablation'])[:,:,:4])
            index=ACTIONS.index(window['action']);actual=np.asarray(window['actual_xy_m']['whole_700ms'])
            errors={name:{v:float(1000*np.linalg.norm(pred[index,6,:2]-actual)) for v,pred in variants.items()}
                for name,variants in predictions.items()}
            shifts={name:{v:float(1000*np.linalg.norm(pred[index,6,:2]-variants['full'][index,6,:2])) for v,pred in variants.items()}
                for name,variants in predictions.items()}
            rows.append(dict(run=number,frame=frame,action=window['action'],errors_mm=errors,shift_mm=shifts))
        runs.append(dict(run=number,root=str(root),summary=summarize(rows),
            by_action={a:summarize([r for r in rows if r['action']==a]) for a in ACTIONS}))
        all_rows.extend(rows);print('RGB_DEPENDENCE_RUN',number,len(rows),flush=True)
    result=dict(schema='forecast_observation_dependence.v1',stride_per_run=STRIDE,perturbations=PERTURBATIONS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=runs,rows=all_rows,total=summarize(all_rows),
        by_action={a:summarize([r for r in all_rows if r['action']==a]) for a in ACTIONS},
        sample_selected_without_errors_or_outcomes=True,all_original_full_forecasts_reproduced=True,
        body_and_control_and_future_command_tensors_unchanged=True,
        model_weights_unchanged=True,depth_not_read=True,navigation_reexecuted=False,
        zero_rgb_and_repeated_frames_are_out_of_distribution_diagnostics=True,
        trained_no_rgb_baseline_not_evaluated=True,visual_causal_navigation_benefit_not_established=True,
        wall_s=time.monotonic()-began)
    train.write(OUTPUT/'result.json',result)
    print(json.dumps(result['total'],indent=2));print('COVERAGE',json.dumps(coverage['by_action']))


if __name__=='__main__':
    from pathlib import Path
    main()
