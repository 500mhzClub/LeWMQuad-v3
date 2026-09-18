"""Four-observation local kinematics, independent of global pose tracking.

Every translation requires adjacent measured RGB-D correspondences. Failed
pairs invalidate the affected histories; there is no command/native fallback.
This component exposes no global pose and never resets a navigation map.
"""
from collections import deque
import numpy as np
from lewm.causal_depth_observation_development import validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.keyframe_rgbd_pose_development import FeatureFrame,matched_points
from lewm.joint_rgbd_rigid_pose_development import register,angle,RIGID_RULES


def local_history_features(pairs):
    if len(pairs)!=3 or any(p is None for p in pairs):
        return None
    positions=[np.zeros(3)];rotations=[np.eye(3)]
    for pair in pairs:
        R=np.asarray(pair['rotation']);t=np.asarray(pair['translation'])
        positions.append(positions[-1]+rotations[-1]@t)
        rotations.append(rotations[-1]@R)
    result=[]
    for p,R in zip(positions[:3],rotations[:3]):
        delta=rotations[-1].T@(p-positions[-1]);relative=rotations[-1].T@R
        yaw=np.arctan2(relative[1,0],relative[0,0])
        result.extend([*delta[:2],np.sin(yaw),np.cos(yaw)-1.])
    return result


class CausalLocalRGBDMotion:
    def __init__(self):
        self.previous=None;self.pairs=deque(maxlen=3)

    def observe(self,policy,depth,fast,*,now_ns):
        validate_depth(depth,policy,now_ns=now_ns)
        current=dict(policy=policy,fast=fast,now=now_ns,
                     features=FeatureFrame(policy['image']['rgb'],depth))
        pair=None;reason=None;quality=None
        if self.previous is not None:
            previous=self.previous
            if now_ns-previous['now']!=100_000_000:
                raise SensorContractError('consecutive local motion observations required')
            try:
                gyro=FastRelativeOrientation()
                gyro.begin(previous['policy'],previous['fast'],now_ns=previous['now'])
                G=np.asarray(gyro.step(policy,fast,now_ns=now_ns)['rotation_initial_body_from_current_body'])
                a,b,ua,ub=matched_points(previous['features'],current['features'])
                # Pair-local deterministic proposals do not depend on episode age.
                R,t,_,quality=register(a,b,ua,ub,gyro_rotation=G,mode='gyro',frame=1)
                if (np.linalg.norm(t)>RIGID_RULES['maximum_increment_translation_m']
                        or angle(R)>RIGID_RULES['maximum_increment_rotation_rad']):
                    raise SensorContractError('local increment envelope rejected')
                pair=dict(rotation=R.tolist(),translation=t.tolist())
            except SensorContractError as error:
                reason=[];cause=error
                while cause is not None:
                    reason.append(str(cause));cause=cause.__cause__
            self.pairs.append(pair)
        self.previous=current
        features=local_history_features(self.pairs)
        return dict(measured_ns=now_ns,pair=pair,pair_failure=reason,pair_quality=quality,
            history_available=features is not None,history_features=features,
            history_offsets_ns=[-300_000_000,-200_000_000,-100_000_000],
            history_feature_fields=['past_x_in_current_body_m','past_y_in_current_body_m',
                                    'sin_past_relative_yaw','cos_past_relative_yaw_minus_one'],
            global_pose_estimated=False,native_state_input=False,command_motion_substitution=False)
