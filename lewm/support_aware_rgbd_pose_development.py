"""Accepted-view promotion and separately conditional contaminated-point bounds."""
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.keyframe_rgbd_pose_development import (KeyframeRGBDPose,FeatureFrame,KeyframeHypotheses,
    point_radius,rotation_distance_bound,compose_radius)
from lewm.rgbd_correspondence_motion_development import RULES

OUTLIER_FRACTION=.2
SUPPORT_MARGIN_CELLS=1


def support_near_limit(registration):
    counts=[registration[k] for k in ('reference_grid_cells','current_grid_cells')]
    if any(type(v) is not int or v<RULES['minimum_grid_cells'] for v in counts):
        raise SensorContractError('promotion requires an already accepted support population')
    return min(counts)<=RULES['minimum_grid_cells']+SUPPORT_MARGIN_CELLS


class SupportAwareRGBDPose(KeyframeRGBDPose):
    """Same fit/acceptance/gyro as V1; additionally promote accepted low-margin views."""
    def observe(self,policy,depth,fast,*,now_ns):
        row=super().observe(policy,depth,fast,now_ns=now_ns)
        row['promotion_reason']='motion_threshold' if row['promoted_keyframe'] else None
        try:
            if (row['registration'] is not None and not row['promoted_keyframe']
                    and support_near_limit(row['registration'])):
                current=FeatureFrame(policy['image']['rgb'],depth)
                self.nodes.append(dict(frame=self.frame,measured_ns=now_ns,parent_frame=row['reference_frame'],
                    position_initial_body_m=row['position_initial_body_m'],
                    conditional_global_position_radius_m=row['conditional_global_position_radius_m']))
                self.reference=current;self.anchor_ns=now_ns
                self.anchor_R=np.asarray(row['rotation_initial_body_from_current_body']).copy()
                self.anchor_p=np.asarray(row['position_initial_body_m']).copy()
                self.anchor_radius=row['conditional_global_position_radius_m']
                row['promoted_keyframe']=True;row['promotion_reason']='accepted_support_margin'
                row['keyframe_count']=len(self.nodes)
            row['legacy_all_inlier_radius_is_not_validated']=True
            return row
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True
            raise SensorContractError('support-aware promotion failed; terminal') from error


def contaminated_box(deltas,radii,*,maximum_outliers):
    """Outer box for a translation lying in at least N-f supplied balls.

For each coordinate, at least N-f lower endpoints are <=truth and N-f upper
endpoints >=truth. Thus L_(N-f)<=truth<=U_(f+1). Different coordinates may use
different subsets, so this is an enclosure, NOT proof of a joint consensus.
"""
    d,e=np.asarray(deltas,float),np.asarray(radii,float)
    if (d.ndim!=2 or d.shape[1:]!=(3,) or not len(d) or e.shape!=(len(d),)
            or not np.isfinite(d).all() or not np.isfinite(e).all() or np.any(e<0)
            or type(maximum_outliers) is not int or not 0<=maximum_outliers<len(d)/2):
        raise SensorContractError('finite balls and a strict-majority good-point hypothesis required')
    f=maximum_outliers;n=len(d)
    lo=np.nextafter(d-e[:,None]-1e-12,-np.inf);hi=np.nextafter(d+e[:,None]+1e-12,np.inf)
    lower=np.sort(lo,axis=0)[n-f-1];upper=np.sort(hi,axis=0)[f]
    available=bool(np.all(lower<=upper))
    return dict(status='CONDITIONAL_CONTAMINATED_BOX' if available else 'INCONSISTENT_CONTAMINATION_HYPOTHESIS',
        lower=lower.tolist(),upper=upper.tolist(),maximum_outliers=f,points=n,required_good_points=n-f,
        joint_consensus_proven=False,calibrated=False)


class RobustReferenceEvidence:
    """Independent uncertainty accounting around the unchanged point-mean poses.

If an unbounded view becomes a keyframe, its global uncertainty remains unknown
in all descendants. Local evidence can still be inspected, never used as global
permission. This object estimates no pose and performs no feature selection.
"""
    def __init__(self):
        self.anchor_frame=0;self.anchor_radius=0.;self.start_ns=None;self.last_frame=-1

    def observe(self,row):
        if row['frame']!=self.last_frame+1:raise SensorContractError('complete ordered pose history required')
        self.last_frame=row['frame']
        if row['frame']==0:
            self.start_ns=row['measured_ns']
            return dict(status='COORDINATE_ANCHOR',local_radius_m=0.,global_radius_m=0.,maximum_outliers=0)
        reg=row['registration']
        if reg['reference_frame']!=self.anchor_frame:raise SensorContractError('same retained reference chain required')
        a,b,ua,ub=[np.asarray(reg[k]) for k in ('reference_inlier_points_body_m','current_inlier_points_body_m',
            'reference_inlier_pixels','current_inlier_pixels')]
        h=KeyframeHypotheses(**row['hypotheses']);R=np.asarray(reg['relative_rotation']);t=np.asarray(reg['translation_reference_body_m'])
        radii=point_radius(a,ua,h)+point_radius(b,ub,h)
        radii+=rotation_distance_bound(reg['conditional_local_angle_radius_rad'])*np.linalg.norm(b,axis=1)
        box=contaminated_box(a-b@R.T,radii,maximum_outliers=int(np.floor(OUTLIER_FRACTION*len(a))))
        local=global_radius=None
        if box['status']=='CONDITIONAL_CONTAMINATED_BOX':
            local=float(np.linalg.norm(np.maximum(np.abs(np.asarray(box['lower'])-t),np.abs(np.asarray(box['upper'])-t))))
            if self.anchor_radius is not None:
                angle=(reg['reference_measured_ns']-self.start_ns)*1e-9*h.gyro_and_integration_error_rad_s
                global_radius=compose_radius(self.anchor_radius,local,angle,t)
        result=box|dict(local_radius_m=local,global_radius_m=global_radius,reference_global_radius_m=self.anchor_radius,
            outlier_fraction_hypothesis=OUTLIER_FRACTION,all_inliers_correct_not_assumed=True,
            gyro_camera_and_majority_point_bounds_still_assumed=True,navigation_qualified=False)
        if row['promoted_keyframe']:
            self.anchor_frame=row['frame'];self.anchor_radius=global_radius
        return result
