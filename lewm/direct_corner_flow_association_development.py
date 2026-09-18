"""Alternative short-interval association; no pose admission or controller use.

Track existing reference corners directly instead of requiring a separately
detected current corner and mutual descriptor match. Keep the original0.5pixel
reverse-flow check and depth lift; add an explicit local photometric check.
The caller must still apply all rigid-fit and temporal-continuity gates.
"""
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_correspondence_motion_development import RULES, lift

FLOW_RULES = dict(maximum_reference_corners=600, patch_width=11,
                  minimum_patch_std=2., minimum_patch_zncc=.90)


def patch_agrees(left, right, p, q):
    width=FLOW_RULES['patch_width']; half=width//2
    if any(not (half <= x <= 639-half and half <= y <= 479-half) for x,y in (p,q)):
        return False
    a=cv2.getRectSubPix(left,(width,width),tuple(map(float,p))).astype(float)
    b=cv2.getRectSubPix(right,(width,width),tuple(map(float,q))).astype(float)
    a-=a.mean(); b-=b.mean()
    if min(float(a.std()),float(b.std())) < FLOW_RULES['minimum_patch_std']: return False
    zncc=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
    return zncc >= FLOW_RULES['minimum_patch_zncc']


def tracked_points(reference, current):
    """Return candidate lifted pairs and gate counts; fit quality is separate."""
    for f in (reference,current):
        if not isinstance(f.gray,np.ndarray) or f.gray.shape!=(480,640) or f.gray.dtype!=np.uint8:
            raise SensorContractError('exact finite uint8 camera grayscale required')
    if len(reference.keypoints)>FLOW_RULES['maximum_reference_corners']:
        raise SensorContractError('bounded original reference-corner population required')
    points=[]; seen=set()
    for keypoint in reference.keypoints:
        p=np.asarray(keypoint.pt,float)
        if p.shape!=(2,) or not np.isfinite(p).all(): raise SensorContractError('finite reference corners required')
        location=tuple(np.rint(p*2).astype(int))
        if location not in seen: points.append(p); seen.add(location)
    counts=dict(reference_corners=len(reference.keypoints),distinct_reference_corners=len(points),
        forward_finite=0,reverse_finite=0,forward_backward=0,photometric=0,distinct_current=0,valid_depth_pair=0)
    receipt=dict(counts=counts,association='direct_corner_lk_photometric_v1',
        original_descriptor_gates_applied=False,initial_flow='same_pixel_hypothesis',
        flow_rules=dict(FLOW_RULES),forward_backward_limit_pixels=RULES['fb_pixels'],
        rigid_geometry_evaluated=False,pose_admitted=False,native_state_used=False)
    empty=(np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,2)))
    if not points: return empty,receipt
    p=np.asarray(points,np.float32).reshape(-1,1,2)
    options=dict(winSize=(RULES['lk_window'],)*2,maxLevel=RULES['lk_levels'],
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
    q,ok,_=cv2.calcOpticalFlowPyrLK(reference.gray,current.gray,p,None,**options)
    if q is None or ok is None or q.shape!=p.shape or ok.shape!=(len(p),1):
        raise SensorContractError('complete forward flow response required')
    good=ok.ravel().astype(bool)&np.isfinite(q).all((1,2)); counts['forward_finite']=int(good.sum())
    p=p[good]; q=q[good]
    if not len(p): return empty,receipt
    back,ok,_=cv2.calcOpticalFlowPyrLK(current.gray,reference.gray,q,None,**options)
    if back is None or ok is None or back.shape!=p.shape or ok.shape!=(len(p),1):
        raise SensorContractError('complete reverse flow response required')
    good=ok.ravel().astype(bool)&np.isfinite(back).all((1,2)); counts['reverse_finite']=int(good.sum())
    good &= np.linalg.norm(back[:,0]-p[:,0],axis=1)<=RULES['fb_pixels']
    counts['forward_backward']=int(good.sum()); p=p[good,0]; q=q[good,0]
    good=np.asarray([patch_agrees(reference.gray,current.gray,a,b) for a,b in zip(p,q,strict=True)],bool)
    counts['photometric']=int(good.sum()); p=p[good]; q=q[good]
    keep=[]; seen=set()
    for i,point in enumerate(q):
        location=tuple(np.rint(point*2).astype(int))
        if location not in seen: keep.append(i); seen.add(location)
    p=p[keep]; q=q[keep]; counts['distinct_current']=len(p)
    a,va=lift(reference.depth,p); b,vb=lift(current.depth,q)
    valid=va&vb&np.isfinite(a).all(1)&np.isfinite(b).all(1)
    counts['valid_depth_pair']=int(valid.sum())
    return (a[valid],b[valid],p[valid],q[valid]),receipt
