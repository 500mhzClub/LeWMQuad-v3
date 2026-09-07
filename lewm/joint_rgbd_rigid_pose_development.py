"""Matched robust rigid-pose / gyro-conditioned translation development models.

Point-cloud scatter is conditioning evidence, not calibrated pose uncertainty.
Gyro is an independent consistency monitor in joint mode, not a pose reset.
"""
import hashlib

import cv2
import numpy as np

from lewm.causal_depth_observation_development import validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.keyframe_rgbd_pose_development import FeatureFrame,matched_points
from lewm.rgbd_correspondence_motion_development import RULES,project,cells
from lewm.support_aware_rgbd_pose_development import support_near_limit

RIGID_RULES=dict(proposals=128,seed=2026090631,minimum_second_scatter_rms_m=.02,
    minimum_tangent_scatter_ratio=.05,maximum_gyro_disagreement_rad=.10,
    maximum_reference_translation_m=3.,maximum_increment_translation_m=.15,
    maximum_increment_rotation_rad=.20,promote_translation_m=.4,promote_rotation_rad=.35)


def angle(R):
    R=np.asarray(R,float)
    return float(np.arctan2(np.linalg.norm([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2,
                            np.clip((np.trace(R)-1)/2,-1,1)))


def proper(R):
    R=np.asarray(R,float)
    if (R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0)
            or abs(np.linalg.det(R)-1)>1e-8):raise SensorContractError('proper finite rotation required')
    return R


def scatter(points):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<3 or not np.isfinite(p).all():
        raise SensorContractError('at least three finite points required')
    rms=np.linalg.svd(p-p.mean(0),compute_uv=False)/np.sqrt(len(p))
    if (rms[1]<RIGID_RULES['minimum_second_scatter_rms_m']
            or rms[1]<RIGID_RULES['minimum_tangent_scatter_ratio']*rms[0]):
        raise SensorContractError('noncollinear well-spread point support required')
    return rms


def fit(a,b,*,gyro_rotation=None):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape:raise SensorContractError('paired point shapes required')
    sa,sb=scatter(a),scatter(b)
    if gyro_rotation is None:
        U,_,Vt=np.linalg.svd((b-b.mean(0)).T@(a-a.mean(0)))
        sign=1. if np.linalg.det(Vt.T@U.T)>=0 else -1.
        R=Vt.T@np.diag([1.,1.,sign])@U.T
    else:R=proper(gyro_rotation)
    proper(R);t=a.mean(0)-R@b.mean(0)
    return R,t,dict(reference_scatter_rms_m=sa.tolist(),current_scatter_rms_m=sb.tolist())


def inliers(a,b,ua,ub,R,t):
    residual=np.linalg.norm(a-b@R.T-t,axis=1)
    current,cvalid=project((a-t)@R);reference,rvalid=project(b@R.T+t)
    keep=(residual<=RULES['residual_m'])&cvalid&rvalid
    keep&=np.linalg.norm(current-ub,axis=1)<=RULES['reprojection_pixels']
    keep&=np.linalg.norm(reference-ua,axis=1)<=RULES['reprojection_pixels']
    return keep,residual


def register(a,b,ua,ub,*,gyro_rotation,mode,frame):
    a,b,ua,ub=[np.asarray(x,float) for x in (a,b,ua,ub)];G=proper(gyro_rotation)
    if (mode not in ('joint','gyro') or type(frame) is not int or frame<0 or a.ndim!=2 or a.shape[1:]!=(3,)
            or b.shape!=a.shape or ua.shape!=(len(a),2) or ub.shape!=ua.shape
            or not all(np.isfinite(x).all() for x in (a,b,ua,ub))):
        raise SensorContractError('paired points, pixels and declared fitting mode required')
    if len(a)<RULES['minimum_matches']:raise SensorContractError('insufficient rigid-pose matches')
    fixed=None if mode=='joint' else G
    rng=np.random.default_rng(np.random.SeedSequence([RIGID_RULES['seed'],frame]))
    subsets=[np.arange(len(a))]+[rng.choice(len(a),3,replace=False) for _ in range(RIGID_RULES['proposals'])]
    best=None;rank=None;valid_candidates=0
    for indices in subsets:
        try:R,t,_=fit(a[indices],b[indices],gyro_rotation=fixed)
        except SensorContractError:continue
        valid_candidates+=1;mask,residual=inliers(a,b,ua,ub,R,t)
        count=int(mask.sum());candidate=(count,-float(np.sum(residual[mask]**2)))
        if best is None or candidate>rank:best=mask;rank=candidate
    if best is None:raise SensorContractError('no conditioned rigid-pose proposal')
    mask=best.copy();initial_count=int(mask.sum());rounds=0
    # Monotonic pruning terminates: every nonfinal step removes >=1 point.
    while True:
        if mask.sum()<RULES['minimum_matches']:raise SensorContractError('insufficient rigid consensus after pruning')
        R,t,conditioning=fit(a[mask],b[mask],gyro_rotation=fixed)
        good,residual=inliers(a,b,ua,ub,R,t);use=mask&good;rounds+=1
        if np.array_equal(use,mask):break
        mask=use
    if (mask.mean()<RULES['minimum_inlier_fraction']
            or min(cells(ua[mask]),cells(ub[mask]))<RULES['minimum_grid_cells']
            or np.linalg.norm(t)>RIGID_RULES['maximum_reference_translation_m']):
        raise SensorContractError('rigid consensus fraction, grid support or displacement rejected')
    disagreement=angle(G.T@R)
    if disagreement>RIGID_RULES['maximum_gyro_disagreement_rad']:
        raise SensorContractError('image and gyro reference rotations disagree beyond diagnostic envelope')
    return R,t,mask,conditioning|dict(mode=mode,lifted_matches=len(a),inliers=int(mask.sum()),
        inlier_fraction=float(mask.mean()),reference_grid_cells=cells(ua[mask]),current_grid_cells=cells(ub[mask]),
        residual_rms_m=float(np.sqrt(np.mean(residual[mask]**2))),valid_proposals=valid_candidates,
        initial_consensus_points=initial_count,pruning_rounds=rounds,gyro_disagreement_rad=disagreement,
        matched_consensus_rules=True,pose_error_bound=None,conditioning_is_not_covariance=True)


class RigidRGBDKeyframePose:
    def __init__(self,mode):
        if mode not in ('joint','gyro'):raise SensorContractError('joint or matched gyro mode required')
        self.mode=mode;self.gyro=FastRelativeOrientation();self.failed=False;self.frame=-1
        self.reference=None;self.anchor_frame=0;self.anchor_ns=None
        self.anchor_R=np.eye(3);self.anchor_G=np.eye(3);self.anchor_p=np.zeros(3)
        self.last_R=np.eye(3);self.last_p=np.zeros(3);self.nodes=[]

    def observe(self,policy,depth,fast,*,now_ns):
        if self.failed:raise SensorContractError('rigid RGBD observer terminal; no recovery or reinitialization')
        try:
            validate_depth(depth,policy,now_ns=now_ns)
            attitude=(self.gyro.begin(policy,fast,now_ns=now_ns) if self.reference is None
                      else self.gyro.step(policy,fast,now_ns=now_ns))
            G=np.asarray(attitude['rotation_initial_body_from_current_body']);current=FeatureFrame(policy['image']['rgb'],depth)
            self.frame+=1;ref=self.anchor_frame;reg=None;reason=None
            if self.reference is None:
                R=np.eye(3);p=np.zeros(3);self.reference=current;self.anchor_ns=now_ns
                self.nodes.append(dict(frame=0,measured_ns=now_ns,parent_frame=None,position_initial_body_m=p.tolist(),
                    rotation_initial_body_from_current_body=R.tolist(),pose_error_bound=None))
            else:
                a,b,ua,ub=matched_points(self.reference,current);relative_gyro=self.anchor_G.T@G
                local_R,t,mask,reg=register(a,b,ua,ub,gyro_rotation=relative_gyro,mode=self.mode,frame=self.frame)
                R=self.anchor_R@local_R;p=self.anchor_p+self.anchor_R@t
                if (np.linalg.norm(p-self.last_p)>RIGID_RULES['maximum_increment_translation_m']
                        or angle(self.last_R.T@R)>RIGID_RULES['maximum_increment_rotation_rad']):
                    raise SensorContractError('consecutive rigid-pose displacement envelope rejected')
                reg|=dict(reference_frame=ref,reference_measured_ns=self.anchor_ns,
                    translation_reference_body_m=t.tolist(),relative_rotation=local_R.tolist(),gyro_relative_rotation=relative_gyro.tolist(),
                    reference_inlier_pixels=ua[mask].tolist(),current_inlier_pixels=ub[mask].tolist(),
                    reference_inlier_points_body_m=a[mask].tolist(),current_inlier_points_body_m=b[mask].tolist())
                motion=np.linalg.norm(t)>=.4 or angle(local_R)>=.35
                near=support_near_limit(reg)
                reason='motion_threshold' if motion else 'accepted_support_margin' if near else None
                if reason is not None:
                    self.nodes.append(dict(frame=self.frame,measured_ns=now_ns,parent_frame=ref,
                        position_initial_body_m=p.tolist(),rotation_initial_body_from_current_body=R.tolist(),pose_error_bound=None))
                    self.reference=current;self.anchor_frame=self.frame;self.anchor_ns=now_ns
                    self.anchor_R=R.copy();self.anchor_G=G.copy();self.anchor_p=p.copy()
            proper(R);self.last_R=R.copy();self.last_p=p.copy()
            return dict(frame=self.frame,measured_ns=now_ns,mode=self.mode,reference_frame=ref,
                position_initial_body_m=p.tolist(),rotation_initial_body_from_current_body=R.tolist(),
                gyro_rotation_initial_body_from_current_body=G.tolist(),global_image_gyro_disagreement_rad=angle(G.T@R),
                registration=reg,promoted_keyframe=reason is not None,promotion_reason=reason,keyframe_count=len(self.nodes),
                rgb_sha256=depth['rgb_sha256'],depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
                position_error_bound=None,orientation_error_bound=None,uncertainty_model_validated=False,
                gyro_role='consistency_monitor_only' if self.mode=='joint' else 'rotation_estimator',
                native_pose_input=False,global_history_reset=False,navigation_qualified=False)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,cv2.error,np.linalg.LinAlgError) as error:
            self.failed=True
            raise SensorContractError('rigid RGBD pose unavailable; terminal failure') from error
