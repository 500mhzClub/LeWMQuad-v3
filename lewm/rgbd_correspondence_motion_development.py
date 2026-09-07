"""Causal RGB correspondences with depth lifting and gyro-conditioned translation.

An accepted estimate is conditional on static, correct point correspondences.
It does not relabel plane-depth rank, calibrate uncertainty or authorize motion.
"""
from copy import deepcopy
import hashlib

import cv2
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation

RULES=dict(sift_features=600,ratio=.7,lk_window=21,lk_levels=3,fb_pixels=.5,
    minimum_matches=12,minimum_grid_cells=6,grid_columns=4,grid_rows=3,
    depth_spread_base_m=.02,depth_spread_fraction=.01,residual_m=.02,
    minimum_inlier_fraction=.6,reprojection_pixels=1.,maximum_translation_m=.15)
T=np.asarray(BODY_FROM_OPTICAL)


def lift(depth,uv):
    """Bilinear optical depth; reject invalid neighbours and surface jumps."""
    uv=np.asarray(uv,float)
    if uv.ndim!=2 or uv.shape[1:]!=(2,) or not np.isfinite(uv).all():
        raise SensorContractError('finite image-coordinate pairs required')
    x,y=uv.T; ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    inside=(ix>=0)&(iy>=0)&(ix<639)&(iy<479)
    ix=np.clip(ix,0,638);iy=np.clip(iy,0,478)
    z=np.stack([depth['depth_m'][iy+dy,ix+dx] for dx,dy in ((0,0),(1,0),(0,1),(1,1))],1)
    valid=np.stack([depth['valid'][iy+dy,ix+dx] for dx,dy in ((0,0),(1,0),(0,1),(1,1))],1).all(1)&inside
    valid&=np.ptp(z,axis=1)<=RULES['depth_spread_base_m']+RULES['depth_spread_fraction']*np.mean(z,axis=1)
    a=x-ix;b=y-iy; weights=np.column_stack(((1-a)*(1-b),a*(1-b),(1-a)*b,a*b))
    distance=np.sum(z*weights,axis=1)
    optical=np.column_stack((distance*(x+.5-320)/FOCAL,distance*(y+.5-240)/FOCAL,distance))
    return optical@T[:3,:3].T+T[:3,3],valid


def project(points):
    optical=(np.asarray(points)-T[:3,3])@T[:3,:3]
    z=optical[:,2];good=z>.2
    return optical[:,:2]/np.maximum(z[:,None],1e-12)*FOCAL+[319.5,239.5],good


def cells(uv):
    uv=np.asarray(uv)
    return len(set((int(x//160),int(y//160)) for x,y in uv))


def solve_translation(previous,current,uv_previous,uv_current,rotation):
    """Robust known-rotation point registration; no native position input."""
    a,b,up,uc,R=[np.asarray(x,float) for x in (previous,current,uv_previous,uv_current,rotation)]
    if (a.ndim!=2 or a.shape[1:]!=(3,) or b.shape!=a.shape or up.shape!=(len(a),2) or uc.shape!=up.shape
            or R.shape!=(3,3) or not all(np.isfinite(x).all() for x in (a,b,up,uc,R))
            or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(R)-1)>1e-8):
        raise SensorContractError('finite paired points/pixels and proper gyro rotation required')
    result=dict(status='INSUFFICIENT_POINT_SUPPORT',translation_previous_body_m=None,
        conditional_point_correspondence_rank=0,lifted_matches=len(a),inliers=0,inlier_fraction=0.,
        previous_grid_cells=0,current_grid_cells=0,residual_rms_m=None,
        calibrated_uncertainty=False,plane_depth_rank_modified=False,navigation_qualified=False)
    if len(a)<RULES['minimum_matches']:return result
    deltas=a-b@R.T;t=np.median(deltas,axis=0)
    good=np.linalg.norm(deltas-t,axis=1)<=RULES['residual_m']
    for _ in range(3):
        if good.sum()<RULES['minimum_matches']:return result
        t=deltas[good].mean(0)
        forward,fvalid=project((a-t)@R);reverse,rvalid=project(b@R.T+t)
        good=(np.linalg.norm(deltas-t,axis=1)<=RULES['residual_m'])&fvalid&rvalid
        good&=(np.linalg.norm(forward-uc,axis=1)<=RULES['reprojection_pixels'])
        good&=(np.linalg.norm(reverse-up,axis=1)<=RULES['reprojection_pixels'])
    count=int(good.sum());fraction=count/len(a);nprev=cells(up[good]);ncur=cells(uc[good])
    rms=float(np.sqrt(np.mean(np.sum((deltas[good]-t)**2,axis=1)))) if count else None
    accepted=(count>=RULES['minimum_matches'] and fraction>=RULES['minimum_inlier_fraction']
        and min(nprev,ncur)>=RULES['minimum_grid_cells'] and np.linalg.norm(t)<=RULES['maximum_translation_m'])
    return result|dict(status='CONDITIONAL_RGBD_POINT_TRANSLATION' if accepted else 'POINT_CONSISTENCY_REJECTED',
        translation_previous_body_m=t.tolist() if accepted else None,
        conditional_point_correspondence_rank=3 if accepted else 0,inliers=count,inlier_fraction=fraction,
        previous_grid_cells=nprev,current_grid_cells=ncur,residual_rms_m=rms)


def track(previous_rgb,current_rgb,previous_depth,current_depth,rotation):
    for rgb in (previous_rgb,current_rgb):
        if not isinstance(rgb,np.ndarray) or rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
            raise SensorContractError('calibrated RGB uint8 frames required')
    gray=[cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) for rgb in (previous_rgb,current_rgb)]
    detector=cv2.SIFT_create(nfeatures=RULES['sift_features'])
    kp0,d0=detector.detectAndCompute(gray[0],None);kp1,d1=detector.detectAndCompute(gray[1],None)
    counts=dict(previous_keypoints=len(kp0),current_keypoints=len(kp1),mutual_ratio_matches=0,forward_backward_matches=0)
    empty=solve_translation(np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,2)),rotation)
    if d0 is None or d1 is None or min(len(d0),len(d1))<2:return empty|counts
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def ratios(left,right):
        return {a.queryIdx:a.trainIdx for pair in matcher.knnMatch(left,right,k=2) if len(pair)==2
                for a,b in [pair] if a.distance<RULES['ratio']*b.distance}
    forward=ratios(d0,d1);backward=ratios(d1,d0)
    pairs=[];seen_previous=set();seen_current=set()
    for i,j in sorted(forward.items()):
        if backward.get(j)!=i:continue
        # Multiple SIFT orientations at one location are not new observed points.
        left=tuple(np.rint(np.asarray(kp0[i].pt)*2).astype(int))
        right=tuple(np.rint(np.asarray(kp1[j].pt)*2).astype(int))
        if left in seen_previous or right in seen_current:continue
        pairs.append((i,j));seen_previous.add(left);seen_current.add(right)
    counts['mutual_ratio_matches']=len(pairs)
    if not pairs:return empty|counts
    p=np.float32([kp0[i].pt for i,j in pairs]).reshape(-1,1,2)
    initial=np.float32([kp1[j].pt for i,j in pairs]).reshape(-1,1,2)
    options=dict(winSize=(RULES['lk_window'],)*2,maxLevel=RULES['lk_levels'],
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    q,ok,_=cv2.calcOpticalFlowPyrLK(gray[0],gray[1],p,initial.copy(),**options)
    back,reverse_ok,_=cv2.calcOpticalFlowPyrLK(gray[1],gray[0],q,p.copy(),**options)
    good=ok.ravel().astype(bool)&reverse_ok.ravel().astype(bool)
    good&=np.isfinite(q).all((1,2))&np.isfinite(back).all((1,2))
    good&=np.linalg.norm(back[:,0]-p[:,0],axis=1)<=RULES['fb_pixels']
    # LK refinement must remain close to the independently ratio-matched feature.
    good&=np.linalg.norm(q[:,0]-initial[:,0],axis=1)<=RULES['reprojection_pixels']
    p=p[good,0];q=q[good,0];counts['forward_backward_matches']=len(p)
    a,va=lift(previous_depth,p);b,vb=lift(current_depth,q);valid=va&vb
    return solve_translation(a[valid],b[valid],p[valid],q[valid],rotation)|counts


class RGBDCorrespondenceMotion:
    def __init__(self):
        self.orientation=FastRelativeOrientation();self.previous=None;self.previous_rotation=None;self.failed=False

    def observe(self,policy,depth,fast,*,now_ns):
        if self.failed:raise SensorContractError('RGBD correspondence observer fault latched')
        try:
            validate_depth(depth,policy,now_ns=now_ns)
            attitude=(self.orientation.begin(policy,fast,now_ns=now_ns) if self.previous is None
                      else self.orientation.step(policy,fast,now_ns=now_ns))
            R=np.asarray(attitude['rotation_initial_body_from_current_body'])
            if self.previous is None:
                motion=dict(status='INITIAL_RGBD_ANCHOR',translation_previous_body_m=None,
                    conditional_point_correspondence_rank=0)
            else:
                rgb,old_depth=self.previous
                motion=track(rgb,policy['image']['rgb'],old_depth,depth,self.previous_rotation.T@R)
            result=dict(measured_ns=now_ns,identity=list(depth['identity']),rgb_sha256=depth['rgb_sha256'],
                depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
                previous_rgb_sha256=self.previous[1]['rgb_sha256'] if self.previous else None,
                relative_orientation=attitude,motion=motion,static_point_correspondences_assumed=True,
                native_pose_input=False,plane_depth_rank_modified=False,navigation_qualified=False)
            self.previous=(policy['image']['rgb'].copy(),deepcopy(depth));self.previous_rotation=R.copy()
            return result
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,cv2.error) as error:
            self.failed=True
            raise SensorContractError('invalid causal RGBD correspondence observations') from error
