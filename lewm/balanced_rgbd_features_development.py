"""Spatially balanced SIFT descriptors; unchanged matching/registration gates.

Detect candidates without a global response cap, then allocate the same600
descriptor budget across occupied160px cells. One descriptor orientation per
half-pixel location; matched_points already rejects duplicate locations. This
is a distinct frontend hypothesis, not a calibrated uncertainty model.
"""
from copy import deepcopy
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_correspondence_motion_development import RULES


def select_keypoints(keypoints):
    groups={};seen=set()
    order=sorted(enumerate(keypoints),key=lambda item:(-item[1].response,item[0]))
    for _,keypoint in order:
        x,y=keypoint.pt
        if not np.isfinite([x,y,keypoint.response]).all() or not (0<=x<640 and 0<=y<480):
            raise SensorContractError('finite in-frame feature required')
        location=tuple(np.rint(np.asarray([x,y])*2).astype(int))
        if location in seen:continue
        seen.add(location);cell=(int(x//160),int(y//160))
        groups.setdefault(cell,[]).append(keypoint)
    selected=[];level=0
    while len(selected)<RULES['sift_features']:
        added=False
        for cell in sorted(groups):
            if level<len(groups[cell]):
                selected.append(groups[cell][level]);added=True
                if len(selected)==RULES['sift_features']:break
        if not added:break
        level+=1
    return selected


class BalancedFeatureFrame:
    def __init__(self,rgb,depth):
        if not isinstance(rgb,np.ndarray) or rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
            raise SensorContractError('calibrated RGB uint8 frame required')
        self.rgb=rgb.copy();self.depth=deepcopy(depth)
        self.gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        detector=cv2.SIFT_create(nfeatures=0)
        candidates=detector.detect(self.gray,None);selected=select_keypoints(candidates)
        self.keypoints,self.descriptors=detector.compute(self.gray,selected)
        if len(self.keypoints)>RULES['sift_features']:raise SensorContractError('descriptor budget exceeded')
        def counts(points):
            result=np.zeros((3,4),int)
            for p in points:result[int(p.pt[1]//160),int(p.pt[0]//160)]+=1
            return result.tolist()
        self.feature_summary=dict(candidate_count=len(candidates),descriptor_count=len(self.keypoints),
            candidate_grid_counts=counts(candidates),descriptor_grid_counts=counts(self.keypoints),
            descriptor_budget=RULES['sift_features'],detector='spatially_balanced_sift',uncertainty_calibrated=False)
