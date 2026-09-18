"""Inspect failed image-pair associations without admitting poses or moving."""
import json
from pathlib import Path

import cv2
import numpy as np

from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.feature_budget_150_tracker_development import FeatureFrame150
from lewm.eligible_floor_registration_development import bind
from lewm.gyro_initial_camera_consensus_development import gyro_core
from lewm.joint_camera_registration_development import project_views
from lewm.joint_rgbd_rigid_pose_development import inliers
from lewm.local_feature_depth_consensus_development import matched_points, tracked_points
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import run_go2_route_turn_memory_transfer_development as run


def main():
    root=run.BASE/run.root_name(1)
    output=root/'failure_pair_correspondence_probe_v1.json'
    if output.exists():raise ValueError('preserve completed diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    read=lambda name:json.loads((root/name).read_text())
    replay=read('tracking_failure_replay_v1/result.json')
    assert replay['all_recorded_raw_poses_matched'] and replay['failure']['frame']==1267
    poses={r['frame']:r['raw_pose'] for r in read('poses.json')}
    gyro={f:np.asarray(poses[f]['gyro_rotation_initial_body_from_current_body']) for f in (1265,1266)}
    gyro[1267]=np.asarray(replay['failure']['view_probe']['current_gyro'])
    reader=NoisyPublicReplay(root/'native');packets={f:reader.packet(f) for f in (1265,1266,1267)}
    A,offset=body_from_reference();rows=[]
    for budget,constructor in ((150,FeatureFrame150),(600,CornerSupportFeatureFrame)):
        features={}
        for frame,(policy,depth,_,rgb,auxiliary,_) in packets.items():
            features[frame]=[constructor(policy['image']['rgb'],depth),constructor(rgb['rgb'],auxiliary)]
        if budget==150:
            assert [len(f.keypoints) for f in features[1266]]==[59,37]
        for frame in (1266,1267):
            for method in ('descriptor','direct_flow'):
                views=[];receipts=[]
                for left,right in zip(features[frame-1],features[frame]):
                    if method=='descriptor':
                        values=matched_points(left,right);receipt=None
                    else:values,receipt=tracked_points(left,right)
                    views.append(values);receipts.append(receipt)
                (pa,pb,pu,pv),(aa,ab,au,av)=views;split=len(pa)
                a=np.concatenate((pa,aa@A.T+offset));b=np.concatenate((pb,ab@A.T+offset))
                ua,ub=np.concatenate((pu,au)),np.concatenate((pv,av))
                check=bind(inliers,project=lambda points:project_views(points,split))
                record=dict(frame=frame,feature_budget=budget,method=method,
                    reference_features=[f.witness() for f in features[frame-1]],
                    current_features=[f.witness() for f in features[frame]],
                    camera_matches=[len(pa),len(aa)],association_receipts=receipts,
                    pose_admitted=False)
                try:
                    R,t,mask,receipt=bind(gyro_core,inliers=check)(a,b,ua,ub,
                        gyro_rotation=gyro[frame-1].T@gyro[frame],mode='joint',frame=frame)
                    counts=[int(mask[:split].sum()),int(mask[split:].sum())]
                    record.update(core_fit_passed=True,camera_inliers=counts,
                        per_camera_three_inlier_gate_passed=min(counts)>=3,
                        strict_majority_passed=2*int(mask.sum())>len(mask),
                        translation_reference_body_m=t.tolist(),fit_receipt=receipt)
                except SensorContractError as error:
                    record.update(core_fit_passed=False,fit_failure=str(error))
                rows.append(record)
                print('PAIR_CORRESPONDENCES',budget,frame,method,record['camera_matches'],
                    record.get('camera_inliers',record.get('fit_failure')),flush=True)
    output.write_text(json.dumps(dict(rows=rows,delivered_noisy_depth_digests_verified=True,
        diagnostic_only=True,tracker_not_restarted=True,poses_admitted=False,
        floor_refinement_and_temporal_admission_executed=False,native_state_used=False,
        alternative_navigation_outcome_proven=False),indent=2)+'\n')


if __name__=='__main__':main()
