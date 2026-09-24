"""Common observed-goal pose evaluation of old and bidirectional readouts."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw
from scripts import train_go2_direct_visual_goal_readout_development as fit
from scripts import evaluate_go2_visual_servo_readout_development as previous

OUTPUT = fit.OUTPUT/'fixed_observed_goal_evaluation'
RESULT = Path('docs/go2_direct_visual_goal_readout_evaluation_2026-09-17.json')


def planar_relative(current,goal):
    a,b = rotation_xyzw(current[3:]),rotation_xyzw(goal[3:])
    yaw_a,yaw_b = np.arctan2(a[1,0],a[0,0]),np.arctan2(b[1,0],b[0,0])
    delta = goal[:2]-current[:2]; c,s = np.cos(yaw_a),np.sin(yaw_a)
    return np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],
        np.arctan2(np.sin(yaw_b-yaw_a),np.cos(yaw_b-yaw_a))])


@torch.inference_mode()
def run():
    assert not OUTPUT.exists() and not RESULT.exists()
    terminal = json.loads(fit.RESULT.read_text()); assert terminal['status'] == 'COMPLETE'
    OUTPUT.mkdir(); started = time.monotonic(); torch.set_num_threads(4)
    models = dict(direct_goal=fit.load().cuda(),previous_500ms=previous.fitted.load().cuda())
    base = fit.metric.parent.reference
    encoder = base.encoders.VJepa21Arm(); encoder.build(torch.device('cuda:0'),torch.float32)
    cache = {}
    def encode(path):
        key = fit.digest(path)
        if key not in cache:
            z = encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float()
            cache[key] = pool_tokens(F.layer_norm(z,(1024,)))
        return cache[key]
    plan = dict(source_sha256=fit.digest(__file__),fit_sha256=fit.digest(fit.RESULT),
        previous_readout_sha256=fit.digest(previous.fitted.OUTPUT/'readout.pt'),
        cases=[0,3],frames=previous.FRAMES,pair_directions=['current_to_goal','goal_to_current'],
        goal_images='unchanged supplied task images',common_feature_precision='GPU float32, batch one',
        targets='common planar body-frame XY and wrapped world-yaw difference',
        no_training=True,no_commands=True,no_navigation=True,
        selection='same 20 observed pairs as preceding diagnostic, plus their reversals',
        limitations=['two exposed related local tasks; correlated frames',
            'new readout changes pair coverage, direction augmentation, architecture and loss together',
            'goal-state prediction is not prospective controller performance'])
    fit.save(OUTPUT/'plan.json',plan)
    predictions = []
    for case in (0,3):
        directory = previous.pilot.OUTPUT/f'case_{case:02d}'
        goal_trial = previous.pilot.previous.CASES[case][1]
        goal = encode(previous.pilot.previous.GOAL_ROOT/goal_trial/'rgb_0023.png')
        for frame in previous.FRAMES:
            current = encode(directory/f'rgb_{frame:04d}.png')
            for direction,first,second in (('current_to_goal',current,goal),('goal_to_current',goal,current)):
                estimates = {name:model(first,second)[0].cpu().numpy() for name,model in models.items()}
                assert all(np.isfinite(v).all() for v in estimates.values())
                predictions.append(dict(case=case,frame=frame,direction=direction,
                    estimates={name:value.tolist() for name,value in estimates.items()}))
            assert torch.equal(models['direct_goal'](current,current),torch.zeros(1,3,device='cuda'))
    fit.save(OUTPUT/'visual_predictions_complete.json',dict(rows=predictions,physical_targets_loaded=False))
    rows = []
    for case in (0,3):
        directory = previous.pilot.OUTPUT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text()); goal_pose = np.asarray(result['goal_pose_evaluator_only'])
        camera = json.loads((directory/'camera_audit.json').read_text())
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            poses = archive['base_pose_world'].copy()
        for entry in [r for r in predictions if r['case']==case]:
            pose = poses[camera[entry['frame']]['physical_sample_index']]
            target = planar_relative(pose,goal_pose) if entry['direction']=='current_to_goal' else planar_relative(goal_pose,pose)
            truth_within = bool(np.linalg.norm(target[:2])<=.03 and abs(target[2])<=np.deg2rad(5))
            models_scored = {}
            for name,values in entry['estimates'].items():
                values = np.asarray(values); error = values-target
                yaw_error = np.arctan2(np.sin(error[2]),np.cos(error[2]))
                predicted_within = bool(np.linalg.norm(values[:2])<=.03 and abs(values[2])<=np.deg2rad(5))
                models_scored[name] = dict(predicted_motion=values.tolist(),xy_error_cm=float(np.linalg.norm(error[:2])*100),
                    yaw_error_deg=float(abs(np.rad2deg(yaw_error))),predicted_within_goal=predicted_within,
                    predicted_normalized_goal_cost=float(np.sum((values/np.array([.03,.03,np.deg2rad(5)]))**2)))
            rows.append(dict(case=case,frame=entry['frame'],direction=entry['direction'],
                target_motion=target.tolist(),actual_within_goal=truth_within,
                native_goal_tolerance_label=result['camera_goal_errors'][entry['frame']]['within_goal'],models=models_scored))
    summaries = {}
    for direction in ('current_to_goal','goal_to_current'):
        selected = [r for r in rows if r['direction']==direction]; summaries[direction] = {}
        for name in models:
            summaries[direction][name] = dict(pairs=len(selected),
                mean_xy_error_cm=float(np.mean([r['models'][name]['xy_error_cm'] for r in selected])),
                mean_yaw_error_deg=float(np.mean([r['models'][name]['yaw_error_deg'] for r in selected])),
                true_goal_detections=sum(r['actual_within_goal'] and r['models'][name]['predicted_within_goal'] for r in selected),
                missed_goal_detections=sum(r['actual_within_goal'] and not r['models'][name]['predicted_within_goal'] for r in selected),
                false_goal_detections=sum(not r['actual_within_goal'] and r['models'][name]['predicted_within_goal'] for r in selected),
                true_outside_detections=sum(not r['actual_within_goal'] and not r['models'][name]['predicted_within_goal'] for r in selected))
    old = json.loads(previous.RESULT.read_text()); discrepancies = []
    for entry in old['rows']:
        new = next(r for r in rows if r['case']==entry['case'] and r['frame']==entry['frame'] and r['direction']=='current_to_goal')
        discrepancies.append(np.max(np.abs(np.array(entry['predicted_goal_motion'])-new['models']['previous_500ms']['predicted_motion'])))
    report = dict(status='COMPLETE',summaries=summaries,rows=rows,wall_s=time.monotonic()-started,
        unique_images=len(cache),identical_pair_output_exact_zero=True,
        native_vs_planar_goal_label_disagreements=sum(r['actual_within_goal'] != r['native_goal_tolerance_label']
            for r in rows if r['direction']=='current_to_goal'),
        native_tolerance_confusion_current_to_goal={name:dict(
            false_goal_detections=sum(not r['native_goal_tolerance_label'] and r['models'][name]['predicted_within_goal']
                for r in rows if r['direction']=='current_to_goal'),
            missed_goal_detections=sum(r['native_goal_tolerance_label'] and not r['models'][name]['predicted_within_goal']
                for r in rows if r['direction']=='current_to_goal')) for name in models},
        previous_cpu_vs_current_gpu_max_abs_displacement_radian_difference=float(max(discrepancies)),
        plan_sha256=fit.digest(OUTPUT/'plan.json'),new_navigation=False,
        limitations=plan['limitations'])
    fit.save(OUTPUT/'result.json',report); fit.save(RESULT,report)
    print('DIRECT_GOAL_READOUT_EVALUATION_COMPLETE',json.dumps(summaries),flush=True)


if __name__ == '__main__':
    try:run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            fit.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
