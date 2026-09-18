"""Diagnostic copy of descriptor matching with stage counters; exact arrays checked."""
import cv2
import numpy as np
from lewm.rgbd_correspondence_motion_development import RULES, lift


def counted_descriptor_matches(reference,current):
    """Same mutual SIFT-ratio, duplicate-location and LK checks as predecessor."""
    counts = dict(reference_features=len(reference.keypoints), current_features=len(current.keypoints),
        forward_ratio=0, backward_ratio=0, mutual_pairs=0, unique_pairs=0,
        bidirectional_seed_consistent=0, paired_depth=0)
    kp0,kp1 = reference.keypoints,current.keypoints; d0,d1 = reference.descriptors,current.descriptors
    empty = (np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,2)))
    if d0 is None or d1 is None or min(len(d0),len(d1))<2: return (*empty, counts)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratios(a,b):
        return {x.queryIdx:x.trainIdx for pair in matcher.knnMatch(a,b,k=2) if len(pair)==2
                for x,y in [pair] if x.distance<RULES['ratio']*y.distance}
    forward,backward = ratios(d0,d1),ratios(d1,d0); pairs=[]; seen0=set(); seen1=set()
    counts.update(forward_ratio=len(forward), backward_ratio=len(backward))
    for i,j in sorted(forward.items()):
        if backward.get(j)!=i: continue
        counts['mutual_pairs'] += 1
        left=tuple(np.rint(np.asarray(kp0[i].pt)*2).astype(int)); right=tuple(np.rint(np.asarray(kp1[j].pt)*2).astype(int))
        if left in seen0 or right in seen1: continue
        pairs.append((i,j)); seen0.add(left); seen1.add(right)
    counts['unique_pairs'] = len(pairs)
    if not pairs: return (*empty, counts)
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
    counts['bidirectional_seed_consistent'] = int(good.sum())
    ua,ub=p[good,0],q[good,0]; a,va=lift(reference.depth,ua); b,vb=lift(current.depth,ub); use=va&vb
    counts['paired_depth'] = int(use.sum())
    return a[use],b[use],ua[use],ub[use],counts
