"""Direct RGB-D keyframe constraints with explicit correlated error hypotheses.

No native pose, inertial translation fallback, covariance independence, global
reset, relocalization after failure, floor exemption or navigation permission.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib

import cv2
import numpy as np

from lewm.causal_depth_observation_development import FOCAL, validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.rgbd_correspondence_motion_development import RULES, lift, project, cells


@dataclass(frozen=True)
class KeyframeHypotheses:
    pixel_coordinate_error: float = .5
    lifted_depth_error_m: float = .00025
    gyro_and_integration_error_rad_s: float = .001
    promote_translation_m: float = .4
    promote_rotation_rad: float = .35

    def __post_init__(self):
        v = np.asarray(list(asdict(self).values()), float)
        if not np.isfinite(v).all() or np.any(v < 0) or min(self.promote_translation_m,self.promote_rotation_rad)<=0:
            raise SensorContractError('finite nonnegative error and positive keyframe thresholds required')


def rotation_distance_bound(angle):
    if not np.isfinite(angle) or angle < 0: raise SensorContractError('nonnegative angular error required')
    return float(2*np.sin(min(angle,np.pi)/2))


def point_radius(points, pixels, hypotheses):
    """Bound lifting error without a 1/sqrt(N) independence assumption.

The depth hypothesis includes interpolation error at the TRUE matched feature;
the pixel hypothesis is per-coordinate correspondence/localization error.
Exact intrinsics/extrinsics and correct static correspondences are assumed.
"""
    p, uv = np.asarray(points,float), np.asarray(pixels,float)
    if p.ndim!=2 or p.shape[1:]!=(3,) or uv.shape!=(len(p),2) or not np.isfinite(p).all() or not np.isfinite(uv).all():
        raise SensorContractError('finite paired lifted points and pixels required')
    # Body x = optical depth + the camera's 0.326-m forward lever arm.
    z = p[:,0]-.326
    if np.any(z<=0): raise SensorContractError('positive optical depth required')
    ez, ep = hypotheses.lifted_depth_error_m, hypotheses.pixel_coordinate_error
    ray = np.abs((uv+[.5,.5]-[320.,240.])/FOCAL)
    transverse = ez*ray+(z[:,None]+ez)*ep/FOCAL
    return np.sqrt(np.sum(transverse**2,axis=1)+ez**2)


def translation_radius(a,b,ua,ub,angle_error,hypotheses):
    if len(a)==0: raise SensorContractError('nonempty matched support required')
    # Triangle inequality applies even if every point error has the same sign.
    each = point_radius(a,ua,hypotheses)+point_radius(b,ub,hypotheses)
    each += rotation_distance_bound(angle_error)*np.linalg.norm(b,axis=1)
    return float(np.mean(each))


def compose_radius(anchor_radius, local_radius, anchor_angle_error, local_translation):
    if not np.isfinite([anchor_radius,local_radius]).all() or min(anchor_radius,local_radius)<0:
        raise SensorContractError('finite nonnegative composed radii required')
    t = np.asarray(local_translation,float)
    if t.shape!=(3,) or not np.isfinite(t).all(): raise SensorContractError('finite local translation required')
    return float(anchor_radius+local_radius+rotation_distance_bound(anchor_angle_error)*np.linalg.norm(t))


class FeatureFrame:
    def __init__(self, rgb, depth):
        self.rgb = rgb.copy(); self.depth = deepcopy(depth)
        self.gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.keypoints,self.descriptors = cv2.SIFT_create(nfeatures=RULES['sift_features']).detectAndCompute(self.gray,None)


def matched_points(reference,current):
    """Same mutual SIFT-ratio, duplicate-location and LK checks as predecessor."""
    kp0,kp1 = reference.keypoints,current.keypoints; d0,d1 = reference.descriptors,current.descriptors
    empty = (np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,2)))
    if d0 is None or d1 is None or min(len(d0),len(d1))<2: return empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratios(a,b):
        return {x.queryIdx:x.trainIdx for pair in matcher.knnMatch(a,b,k=2) if len(pair)==2
                for x,y in [pair] if x.distance<RULES['ratio']*y.distance}
    forward,backward = ratios(d0,d1),ratios(d1,d0); pairs=[]; seen0=set(); seen1=set()
    for i,j in sorted(forward.items()):
        if backward.get(j)!=i: continue
        left=tuple(np.rint(np.asarray(kp0[i].pt)*2).astype(int)); right=tuple(np.rint(np.asarray(kp1[j].pt)*2).astype(int))
        if left in seen0 or right in seen1: continue
        pairs.append((i,j)); seen0.add(left); seen1.add(right)
    if not pairs: return empty
    p=np.float32([kp0[i].pt for i,j in pairs]).reshape(-1,1,2)
    initial=np.float32([kp1[j].pt for i,j in pairs]).reshape(-1,1,2)
    options=dict(winSize=(RULES['lk_window'],)*2,maxLevel=RULES['lk_levels'],
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    q,ok,_=cv2.calcOpticalFlowPyrLK(reference.gray,current.gray,p,initial.copy(),**options)
    back,reverse_ok,_=cv2.calcOpticalFlowPyrLK(current.gray,reference.gray,q,p.copy(),**options)
    good=ok.ravel().astype(bool)&reverse_ok.ravel().astype(bool)
    good &= np.isfinite(q).all((1,2))&np.isfinite(back).all((1,2))
    good &= np.linalg.norm(back[:,0]-p[:,0],axis=1)<=RULES['fb_pixels']
    good &= np.linalg.norm(q[:,0]-initial[:,0],axis=1)<=RULES['reprojection_pixels']
    ua,ub=p[good,0],q[good,0]; a,va=lift(reference.depth,ua); b,vb=lift(current.depth,ub); use=va&vb
    return a[use],b[use],ua[use],ub[use]


def register(a,b,ua,ub,rotation):
    """Direct keyframe displacement, not the predecessor's <=0.15-m increment.

Same residual/reprojection/coverage requirements; accepted mean and reported
inlier set must be a fixed point so the conditional radius bounds THAT mean.
"""
    a,b,ua,ub,R = [np.asarray(x,float) for x in (a,b,ua,ub,rotation)]
    if (a.ndim!=2 or a.shape[1:]!=(3,) or b.shape!=a.shape or ua.shape!=(len(a),2) or ub.shape!=ua.shape
            or R.shape!=(3,3) or not all(np.isfinite(x).all() for x in (a,b,ua,ub,R))
            or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(R)-1)>1e-8):
        raise SensorContractError('finite paired points and proper rotation required')
    if len(a)<RULES['minimum_matches']: raise SensorContractError('insufficient keyframe matches')
    deltas=a-b@R.T; t=np.median(deltas,axis=0)
    mask=np.linalg.norm(deltas-t,axis=1)<=RULES['residual_m']
    stable=False
    for _ in range(10):
        if mask.sum()<RULES['minimum_matches']: break
        t=deltas[mask].mean(0)
        forward,fg=project((a-t)@R); reverse,rg=project(b@R.T+t)
        selected=(np.linalg.norm(deltas-t,axis=1)<=RULES['residual_m'])&fg&rg
        selected &= (np.linalg.norm(forward-ub,axis=1)<=RULES['reprojection_pixels'])
        selected &= (np.linalg.norm(reverse-ua,axis=1)<=RULES['reprojection_pixels'])
        if np.array_equal(selected,mask): stable=True; break
        mask=selected
    if (not stable or mask.sum()<RULES['minimum_matches'] or mask.mean()<RULES['minimum_inlier_fraction']
            or min(cells(ua[mask]),cells(ub[mask]))<RULES['minimum_grid_cells'] or np.linalg.norm(t)>3.):
        raise SensorContractError('keyframe consistency or reference displacement rejected')
    return t,mask,dict(lifted_matches=len(a),inliers=int(mask.sum()),inlier_fraction=float(mask.mean()),
        reference_grid_cells=cells(ua[mask]),current_grid_cells=cells(ub[mask]),
        residual_rms_m=float(np.sqrt(np.mean(np.sum((deltas[mask]-t)**2,axis=1)))),stable_mean_inliers=True)


class KeyframeRGBDPose:
    def __init__(self, hypotheses=KeyframeHypotheses()):
        if not isinstance(hypotheses,KeyframeHypotheses): raise SensorContractError('explicit keyframe hypotheses required')
        self.hypotheses=hypotheses; self.orientation=FastRelativeOrientation(); self.failed=False
        self.reference=None; self.anchor_ns=None; self.anchor_R=np.eye(3); self.anchor_p=np.zeros(3)
        self.anchor_radius=0.; self.last_position=np.zeros(3); self.nodes=[]; self.frame=-1

    def observe(self,policy,depth,fast,*,now_ns):
        if self.failed: raise SensorContractError('keyframe observer fault latched; no reinitialization')
        try:
            validate_depth(depth,policy,now_ns=now_ns)
            attitude=(self.orientation.begin(policy,fast,now_ns=now_ns) if self.reference is None
                      else self.orientation.step(policy,fast,now_ns=now_ns))
            R=np.asarray(attitude['rotation_initial_body_from_current_body']); current=FeatureFrame(policy['image']['rgb'],depth)
            self.frame+=1; start=self.orientation.start_ns; h=self.hypotheses
            global_angle=(now_ns-start)*1e-9*h.gyro_and_integration_error_rad_s
            if self.reference is None:
                # Initial observation is a coordinate anchor, not a measured displacement.
                self.reference=current; self.anchor_ns=now_ns
                self.nodes.append(dict(frame=self.frame,measured_ns=now_ns,parent_frame=None,
                    position_initial_body_m=[0.,0.,0.],conditional_global_position_radius_m=0.))
                p=np.zeros(3); radius=local_radius=0.; reference_frame=0; diagnostics=None; promoted=False
            else:
                relative=self.anchor_R.T@R; a,b,ua,ub=matched_points(self.reference,current)
                t,mask,diagnostics=register(a,b,ua,ub,relative)
                local_angle=(now_ns-self.anchor_ns)*1e-9*h.gyro_and_integration_error_rad_s
                local_radius=translation_radius(a[mask],b[mask],ua[mask],ub[mask],local_angle,h)
                radius=compose_radius(self.anchor_radius,local_radius,
                    (self.anchor_ns-start)*1e-9*h.gyro_and_integration_error_rad_s,t)
                p=self.anchor_p+self.anchor_R@t
                if np.linalg.norm(p-self.last_position)>.15: raise SensorContractError('consecutive physical displacement envelope rejected')
                reference_frame=self.nodes[-1]['frame']
                diagnostics |= dict(reference_frame=reference_frame,reference_measured_ns=self.anchor_ns,
                    translation_reference_body_m=t.tolist(),relative_rotation=relative.tolist(),
                    conditional_local_angle_radius_rad=local_angle,
                    reference_inlier_pixels=ua[mask].tolist(),current_inlier_pixels=ub[mask].tolist(),
                    reference_inlier_points_body_m=a[mask].tolist(),current_inlier_points_body_m=b[mask].tolist())
                angle=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
                promoted=bool(np.linalg.norm(t)>=h.promote_translation_m or angle>=h.promote_rotation_rad)
                if promoted:
                    self.nodes.append(dict(frame=self.frame,measured_ns=now_ns,parent_frame=reference_frame,
                        position_initial_body_m=p.tolist(),conditional_global_position_radius_m=radius))
                    self.reference=current; self.anchor_ns=now_ns; self.anchor_R=R.copy(); self.anchor_p=p.copy(); self.anchor_radius=radius
            self.last_position=p.copy()
            return dict(frame=self.frame,measured_ns=now_ns,position_initial_body_m=p.tolist(),
                rotation_initial_body_from_current_body=R.tolist(),
                conditional_global_position_radius_m=radius,conditional_local_translation_radius_m=local_radius,
                conditional_global_angle_radius_rad=global_angle,reference_frame=reference_frame,
                promoted_keyframe=promoted,keyframe_count=len(self.nodes),registration=diagnostics,
                rgb_sha256=depth['rgb_sha256'],depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
                hypotheses=asdict(h),all_inlier_correspondences_static_and_correct_assumed=True,
                exact_camera_geometry_assumed=True,conditional_bounds_calibrated=False,
                native_pose_input=False,global_history_reset=False,navigation_qualified=False)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,cv2.error) as error:
            self.failed=True
            raise SensorContractError('keyframe RGBD pose unavailable; terminal failure') from error
