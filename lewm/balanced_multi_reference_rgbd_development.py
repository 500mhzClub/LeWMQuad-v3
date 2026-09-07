"""Balanced frontend with inherited primary-first reference selection/gates."""
from copy import deepcopy
import hashlib
import cv2
import numpy as np
from lewm.causal_depth_observation_development import validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose,MultiReferenceVisualLedMotion
from lewm.joint_rgbd_rigid_pose_development import proper,angle
from lewm.support_aware_rgbd_pose_development import support_near_limit
from lewm.balanced_rgbd_features_development import BalancedFeatureFrame


class BalancedMultiReferenceRGBDPose(MultiReferenceRGBDPose):
    def observe(self,policy,depth,fast,*,now_ns):
        if self.failed:raise SensorContractError('multi-reference observer terminal; no recovery or reinitialization')
        self.last_selection=dict(status='VALIDATING_CURRENT_INPUT',decision_ns=now_ns)
        try:
            validate_depth(depth,policy,now_ns=now_ns)
            attitude=(self.gyro.begin(policy,fast,now_ns=now_ns) if self.reference is None
                      else self.gyro.step(policy,fast,now_ns=now_ns))
            G=np.asarray(attitude['rotation_initial_body_from_current_body']);current=BalancedFeatureFrame(policy['image']['rgb'],depth)
            self.feature_summary=deepcopy(current.feature_summary)
            self.frame+=1;ref=self.anchor_frame;reg=None;reason=None
            if self.reference is None:
                R=np.eye(3);p=np.zeros(3);self._remember(current,R,G,p,now_ns)
                self.nodes.append(dict(frame=0,measured_ns=now_ns,parent_frame=None,position_initial_body_m=p.tolist(),
                    rotation_initial_body_from_current_body=R.tolist(),pose_error_bound=None))
                self.last_selection=dict(status='INITIAL_REFERENCE',selected_reference=0,retained_references=1)
            else:
                c,fallback=self._choose(current,G);R,p,reg=c['R'],c['p'],c['registration'];ref=c['reference'].frame
                motion=np.linalg.norm(c['t'])>=.4 or angle(c['local_R'])>=.35
                reason=('qualified_recent_reference' if fallback else 'motion_threshold' if motion
                        else 'accepted_support_margin' if support_near_limit(reg) else None)
                if reason is not None:
                    self.nodes.append(dict(frame=self.frame,measured_ns=now_ns,parent_frame=ref,
                        position_initial_body_m=p.tolist(),rotation_initial_body_from_current_body=R.tolist(),pose_error_bound=None))
                    self._remember(current,R,G,p,now_ns)
            proper(R);self.last_R=R.copy();self.last_p=p.copy()
            return dict(frame=self.frame,measured_ns=now_ns,mode='gyro',reference_frame=ref,
                position_initial_body_m=p.tolist(),rotation_initial_body_from_current_body=R.tolist(),
                gyro_rotation_initial_body_from_current_body=G.tolist(),global_image_gyro_disagreement_rad=angle(G.T@R),
                registration=reg,promoted_keyframe=reason is not None,promotion_reason=reason,keyframe_count=len(self.nodes),
                rgb_sha256=depth['rgb_sha256'],depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
                position_error_bound=None,orientation_error_bound=None,uncertainty_model_validated=False,
                gyro_role='rotation_estimator',native_pose_input=False,global_history_reset=False,navigation_qualified=False)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,cv2.error,np.linalg.LinAlgError) as error:
            self.failed=True
            raise SensorContractError('multi-reference RGBD pose unavailable; terminal failure') from error


class BalancedVisualLedMotion(MultiReferenceVisualLedMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=BalancedMultiReferenceRGBDPose()

    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs)
        return result|dict(feature_selection=deepcopy(getattr(self.model,"feature_summary",None)))
