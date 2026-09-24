"""Primary-first, bounded recent-reference tracking with unchanged pair gates.

No pose reset, native input, weaker matching rule, extrapolation or calibrated
uncertainty. Alternative references are correlated hypotheses, not independent
measurements. Agreement below a diagnostic threshold is not an error bound.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import cv2
import numpy as np
from lewm.causal_depth_observation_development import validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.keyframe_rgbd_pose_development import FeatureFrame,matched_points
from lewm.joint_rgbd_rigid_pose_development import RigidRGBDKeyframePose,register,proper,angle,RIGID_RULES
from lewm.support_aware_rgbd_pose_development import support_near_limit
from lewm.visual_led_motion_development import VisualLedMotion

REFERENCE_RULES=dict(maximum_references=8,maximum_candidate_disagreement_m=.02,
                     maximum_candidate_disagreement_rad=.10,primary_first=True)


@dataclass(frozen=True)
class Reference:
    frame: int
    measured_ns: int
    features: object
    rotation: np.ndarray
    gyro: np.ndarray
    position: np.ndarray


class MultiReferenceRGBDPose(RigidRGBDKeyframePose):
    def __init__(self):
        super().__init__('gyro');self.references=[];self.last_selection=None

    def _remember(self,current,R,G,p,now):
        self.references.append(Reference(self.frame,now,current,R.copy(),G.copy(),p.copy()))
        self.references=self.references[-8:]
        self.reference=current;self.anchor_frame=self.frame;self.anchor_ns=now
        self.anchor_R=R.copy();self.anchor_G=G.copy();self.anchor_p=p.copy()

    def _candidate(self,ref,current,G):
        a,b,ua,ub=matched_points(ref.features,current);relative_gyro=ref.gyro.T@G
        local_R,t,mask,reg=register(a,b,ua,ub,gyro_rotation=relative_gyro,mode='gyro',frame=self.frame)
        R=ref.rotation@local_R;p=ref.position+ref.rotation@t
        if (np.linalg.norm(p-self.last_p)>RIGID_RULES['maximum_increment_translation_m']
                or angle(self.last_R.T@R)>RIGID_RULES['maximum_increment_rotation_rad']):
            raise SensorContractError('consecutive rigid-pose displacement envelope rejected')
        reg|=dict(reference_frame=ref.frame,reference_measured_ns=ref.measured_ns,
            translation_reference_body_m=t.tolist(),relative_rotation=local_R.tolist(),gyro_relative_rotation=relative_gyro.tolist(),
            reference_inlier_pixels=ua[mask].tolist(),current_inlier_pixels=ub[mask].tolist(),
            reference_inlier_points_body_m=a[mask].tolist(),current_inlier_points_body_m=b[mask].tolist())
        return dict(reference=ref,R=R,p=p,local_R=local_R,t=t,registration=reg)

    def _choose(self,current,G):
        attempts=[];self.last_selection=dict(status='SEARCHING',attempts=attempts,retained_references=len(self.references))
        primary=self.references[-1]
        try:
            candidate=self._candidate(primary,current,G)
            attempts.append(dict(reference_frame=primary.frame,status='ACCEPTED_PRIMARY'))
            self.last_selection.update(status='PRIMARY_ACCEPTED',selected_reference=primary.frame)
            return candidate,False
        except SensorContractError as error:
            attempts.append(dict(reference_frame=primary.frame,status='REJECTED',reason=str(error)))
        candidates=[]
        for ref in reversed(self.references[:-1]):
            try:
                c=self._candidate(ref,current,G);candidates.append(c)
                q=c['registration'];attempts.append(dict(reference_frame=ref.frame,status='QUALIFIED_ALTERNATIVE',
                    inlier_fraction=q['inlier_fraction'],inliers=q['inliers'],reference_grid_cells=q['reference_grid_cells'],
                    current_grid_cells=q['current_grid_cells'],residual_rms_m=q['residual_rms_m']))
            except SensorContractError as error:attempts.append(dict(reference_frame=ref.frame,status='REJECTED',reason=str(error)))
        if not candidates:
            self.last_selection['status']='NO_QUALIFIED_REFERENCE'
            raise SensorContractError('no recent reference passes unchanged registration and increment gates')
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                if (np.linalg.norm(a['p']-b['p'])>.02 or angle(a['R'].T@b['R'])>.10):
                    self.last_selection['status']='CONFLICTING_ALTERNATIVES'
                    raise SensorContractError('qualified recent-reference hypotheses conflict')
        def rank(c):
            q=c['registration']
            return (min(q['reference_grid_cells'],q['current_grid_cells']),q['inlier_fraction'],q['inliers'],
                    -q['residual_rms_m'],c['reference'].frame)
        chosen=max(candidates,key=rank)
        self.last_selection.update(status='RECENT_REFERENCE_ACCEPTED',selected_reference=chosen['reference'].frame,
                                   qualified_alternatives=len(candidates),uncertainty_calibrated=False)
        return chosen,True

    def observe(self,policy,depth,fast,*,now_ns):
        if self.failed:raise SensorContractError('multi-reference observer terminal; no recovery or reinitialization')
        self.last_selection=dict(status='VALIDATING_CURRENT_INPUT',decision_ns=now_ns)
        try:
            validate_depth(depth,policy,now_ns=now_ns)
            attitude=(self.gyro.begin(policy,fast,now_ns=now_ns) if self.reference is None
                      else self.gyro.step(policy,fast,now_ns=now_ns))
            G=np.asarray(attitude['rotation_initial_body_from_current_body']);current=FeatureFrame(policy['image']['rgb'],depth)
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


class MultiReferenceVisualLedMotion(VisualLedMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__('gyro',identity=identity);self.model=MultiReferenceRGBDPose()

    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs)
        return result|dict(reference_selection=deepcopy(self.model.last_selection))
