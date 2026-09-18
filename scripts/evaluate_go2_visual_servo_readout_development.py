"""Can the existing 500-ms motion probe estimate current-to-goal displacement?

CPU-only diagnostic of a possible reactive baseline. No training, commands,
future-state prediction or navigation. Readout inputs are observed RGB pairs.
"""
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw
from scripts import train_go2_dense_visual_motion_readout_development as fitted
from scripts import run_go2_dense_metric_goal_pilot_development as pilot

OUTPUT = fitted.OUTPUT.parent/'go2_visual_servo_readout_diagnostic_v1_attempt_001'
RESULT = Path('docs/go2_visual_servo_readout_diagnostic_2026-09-17.json')
FRAMES = (10,15,20,25,30,35,40,55,80,110)


@torch.inference_mode()
def run():
    assert not OUTPUT.exists() and not RESULT.exists()
    OUTPUT.mkdir(); started = time.monotonic(); torch.set_num_threads(4)
    model = fitted.load()
    assert next(model.parameters()).device.type == 'cpu'
    plan = dict(cases=[0,3],frames=FRAMES,source_sha256=fitted.digest(__file__),
        readout_sha256=fitted.digest(fitted.OUTPUT/'readout.pt'),device='cpu',
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        question='transfer of existing 500-ms observed-pair motion head to direct visual goal displacement',
        no_training=True,no_navigation=True,goal_images_are_task_inputs=True,
        labels_used_only_after_readout_predictions=True,
        limitations=['two exposed related tasks','goal separations can exceed the probe training horizon',
            'CPU float32 encoder rather than live GPU execution; no policy comparison'])
    fitted.save(OUTPUT/'plan.json',plan)
    encoder = fitted.parent.reference.encoders.VJepa21Arm(); encoder.build(torch.device('cpu'),torch.float32)
    cache = {}
    def encode(path):
        key = fitted.digest(path)
        if key not in cache:
            z = F.layer_norm(encoder.tokens(encoder.preprocess(str(path))[None]).float(),(1024,))
            cache[key] = pool_tokens(z)
        return cache[key]
    predictions = []
    for case in (0,3):
        directory = pilot.OUTPUT/f'case_{case:02d}'
        goal_trial = pilot.previous.CASES[case][1]
        goal = encode(pilot.previous.GOAL_ROOT/goal_trial/'rgb_0023.png')
        for frame in FRAMES:
            current = encode(directory/f'rgb_{frame:04d}.png')
            prediction = model(current,goal)[0].numpy()
            identity = model(current,current)[0].numpy()
            assert np.isfinite(prediction).all()
            predictions.append(dict(case=case,frame=frame,predicted_goal_motion=prediction.tolist(),
                identical_image_prediction=identity.tolist()))
        print('VISUAL_SERVO_READOUT_CASE',case,'seconds',round(time.monotonic()-started,1),flush=True)
    fitted.save(OUTPUT/'visual_predictions_complete.json',dict(rows=predictions,physical_targets_loaded=False))
    rows = []
    for case in (0,3):
        directory = pilot.OUTPUT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text())
        goal_pose = np.asarray(result['goal_pose_evaluator_only']); goal_rotation = rotation_xyzw(goal_pose[3:])
        cameras = json.loads((directory/'camera_audit.json').read_text())
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            poses = archive['base_pose_world'].copy()
        for row in [r for r in predictions if r['case'] == case]:
            pose = poses[cameras[row['frame']]['physical_sample_index']]
            rotation = rotation_xyzw(pose[3:]); delta = rotation.T @ (goal_pose[:3]-pose[:3])
            relative = rotation.T @ goal_rotation
            target = np.array([delta[0],delta[1],np.arctan2(relative[1,0],relative[0,0])])
            prediction = np.asarray(row['predicted_goal_motion']); error = prediction-target
            yaw_error = np.arctan2(np.sin(error[2]),np.cos(error[2]))
            rows.append(row|dict(target_goal_motion=target.tolist(),
                xy_error_cm=float(np.linalg.norm(error[:2])*100),yaw_error_deg=float(abs(np.rad2deg(yaw_error))),
                predicted_xy_distance_cm=float(np.linalg.norm(prediction[:2])*100),
                target_xy_distance_cm=float(np.linalg.norm(target[:2])*100),
                predicted_within_goal=bool(np.linalg.norm(prediction[:2]) <= .03 and abs(prediction[2]) <= np.deg2rad(5)),
                actual_within_goal=result['camera_goal_errors'][row['frame']]['within_goal']))
    report = dict(status='COMPLETE',rows=rows,wall_s=time.monotonic()-started,unique_images=len(cache),
        mean_xy_error_cm=float(np.mean([r['xy_error_cm'] for r in rows])),
        mean_yaw_error_deg=float(np.mean([r['yaw_error_deg'] for r in rows])),
        false_goal_classifications=sum(r['predicted_within_goal'] and not r['actual_within_goal'] for r in rows),
        missed_goal_classifications=sum(not r['predicted_within_goal'] and r['actual_within_goal'] for r in rows),
        plan_sha256=fitted.digest(OUTPUT/'plan.json'),navigation_tested=False,model_changed=False,
        independent_samples=False,interpretation='component suitability diagnostic; not a reactive baseline performance result')
    fitted.save(OUTPUT/'result.json',report); fitted.save(RESULT,report)
    print('VISUAL_SERVO_READOUT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    try: run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            fitted.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
