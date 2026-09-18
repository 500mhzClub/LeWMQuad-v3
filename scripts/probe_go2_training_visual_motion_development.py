"""Feasibility of causal motion features from one retained training recording."""
import hashlib
import json
import time
import cv2
import torch
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.read_go2_training_execution_coverage_development import BASE

SOURCE=BASE/'go2_moving_action_switch_family_v1_attempt_001/switch_episode_000'
OUTPUT=BASE/'go2_training_visual_motion_probe_v1_attempt_001'


def main():
    if OUTPUT.exists():raise ValueError('preserve motion probe')
    branch=json.loads((SOURCE/'branch_specification.json').read_text())
    if branch['data_role']!='train':raise ValueError('training-only feasibility probe')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    count=len(json.loads((SOURCE/'policy_observations.json').read_text())['frames'])
    observer=MultiReferenceVisualLedMotion();rows=[];identities={};started=time.monotonic()
    OUTPUT.mkdir()
    for name in ('branch_specification.json','policy_observations.json','policy_histories.npz',
                 'depth_observations.json','fast_gyro_histories.npz'):
        identities[name]=hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
    try:
        for frame in range(count):
            for name in (f'rgb_{frame:04d}.png',f'depth_{frame:04d}.npz'):
                identities[name]=hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
            policy,depth=load_rgbd_observation(SOURCE,frame)
            fast=load_fast_packet(SOURCE,frame)
            value=observer.observe(policy,depth,fast,now_ns=policy['sensor_state']['decision_ns'])
            rows.append(dict(frame=frame,**value))
            if value['current_pose'] is None:
                break
        result=dict(status='COMPLETE',source=str(SOURCE),data_role='train',recorded_frames=count,
            attempted_frames=len(rows),accepted_poses=sum(r['current_pose'] is not None for r in rows),
            terminal_failure=rows[-1]['terminal_failure'],
            all_frames_tracked=len(rows)==count and rows[-1]['current_pose'] is not None,
            usable_four_pose_histories=max(0,sum(r['current_pose'] is not None for r in rows)-3),
            observer='MultiReferenceVisualLedMotion_primary_RGBD_gyro',
            deployed_dual_camera_registered_observer_equivalence_established=False,
            native_state_read=False,native_accuracy_evaluated=False,
            ideal_recorded_depth_and_gyro=True,hardware_validated=False,
            input_sha256=identities,wall_s=time.monotonic()-started)
        (OUTPUT/'poses.json').write_text(json.dumps(rows,indent=2)+'\n')
        (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
    except Exception as error:
        (OUTPUT/'failure.json').write_text(json.dumps(dict(reason=repr(error),attempted_frames=len(rows))))
        raise


if __name__=='__main__':main()
