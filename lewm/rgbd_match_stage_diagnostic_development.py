"""Observe frozen correspondence gates without changing their output arrays."""
import cv2
import numpy as np
from lewm.rgbd_correspondence_motion_development import RULES, lift
from lewm.keyframe_rgbd_pose_development import matched_points


def diagnose_matches(reference, current):
    kp0, kp1 = reference.keypoints, current.keypoints
    d0, d1 = reference.descriptors, current.descriptors
    counts = dict(reference_features=len(kp0), current_features=len(kp1),
        forward_ratio=0, backward_ratio=0, mutual=0, distinct_locations=0,
        flow_status=0, finite_flow=0, forward_backward=0, descriptor_location_agreement=0,
        valid_depth_pair=0)
    values = (np.empty((0,3)), np.empty((0,3)), np.empty((0,2)), np.empty((0,2)))
    if d0 is not None and d1 is not None and min(len(d0),len(d1)) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        def ratios(a,b):
            return {x.queryIdx:x.trainIdx for pair in matcher.knnMatch(a,b,k=2) if len(pair)==2
                    for x,y in [pair] if x.distance<RULES['ratio']*y.distance}
        forward, backward = ratios(d0,d1), ratios(d1,d0)
        counts.update(forward_ratio=len(forward), backward_ratio=len(backward))
        pairs=[]; seen0=set(); seen1=set()
        for i,j in sorted(forward.items()):
            if backward.get(j) != i: continue
            counts['mutual'] += 1
            left=tuple(np.rint(np.asarray(kp0[i].pt)*2).astype(int))
            right=tuple(np.rint(np.asarray(kp1[j].pt)*2).astype(int))
            if left in seen0 or right in seen1: continue
            pairs.append((i,j)); seen0.add(left); seen1.add(right)
        counts['distinct_locations'] = len(pairs)
        if pairs:
            p=np.float32([kp0[i].pt for i,j in pairs]).reshape(-1,1,2)
            initial=np.float32([kp1[j].pt for i,j in pairs]).reshape(-1,1,2)
            options=dict(winSize=(RULES['lk_window'],)*2,maxLevel=RULES['lk_levels'],
                criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            q,ok,_=cv2.calcOpticalFlowPyrLK(reference.gray,current.gray,p,initial.copy(),**options)
            back,reverse_ok,_=cv2.calcOpticalFlowPyrLK(current.gray,reference.gray,q,p.copy(),**options)
            good=ok.ravel().astype(bool)&reverse_ok.ravel().astype(bool)
            counts['flow_status']=int(good.sum())
            good &= np.isfinite(q).all((1,2))&np.isfinite(back).all((1,2))
            counts['finite_flow']=int(good.sum())
            good &= np.linalg.norm(back[:,0]-p[:,0],axis=1)<=RULES['fb_pixels']
            counts['forward_backward']=int(good.sum())
            good &= np.linalg.norm(q[:,0]-initial[:,0],axis=1)<=RULES['reprojection_pixels']
            counts['descriptor_location_agreement']=int(good.sum())
            ua,ub=p[good,0],q[good,0];a,va=lift(reference.depth,ua);b,vb=lift(current.depth,ub);use=va&vb
            counts['valid_depth_pair']=int(use.sum())
            values=a[use],b[use],ua[use],ub[use]
    original=matched_points(reference,current)
    for actual, expected in zip(values,original,strict=True):
        if actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes()!=expected.tobytes():
            raise ValueError('stage instrumentation changed frozen correspondence arrays')
    sequence=[counts[k] for k in ('mutual','distinct_locations','flow_status','finite_flow',
        'forward_backward','descriptor_location_agreement','valid_depth_pair')]
    if any(a<b for a,b in zip(sequence,sequence[1:])):
        raise ValueError('sequential gate populations must be nonincreasing')
    return dict(counts=counts, all_four_frozen_output_arrays_byte_exact=True,
                gates_changed=False, pose_admitted=False)
