"""Bounded spatial SIFT selection from current RGB and valid public depth.

Select up to50 strongest liftable features per4x3 image cell, max600 total.
Unchanged mutual descriptors, LK, depth lifting and rigid gates apply later.
No native geometry, labels, gyro or commands participate in feature selection.
"""
from copy import deepcopy
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_correspondence_motion_development import lift, RULES

FEATURE_RULES = dict(columns=4, rows=3, per_cell=50, maximum_selected=600,
    maximum_detected=50000, depth_validity_before_selection=True,
    descriptor='SIFT_default', detector_contrast_threshold=.04)


def cell_index(x, y):
    return int(y//160)*4+int(x//160)


class SpatialSupportFeatureFrame:
    def __init__(self, rgb, depth):
        if not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise SensorContractError('exact current RGB frame required')
        self.rgb = rgb.copy()
        self.depth = deepcopy(depth)
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detector = cv2.SIFT_create(nfeatures=0)
        detected = detector.detect(self.gray, depth['valid'].astype(np.uint8)*255)
        if len(detected) > FEATURE_RULES['maximum_detected']:
            raise SensorContractError('bounded SIFT detection population exceeded')
        self.detected_count = len(detected)
        self.cell_counts = [0]*12
        self.liftable_count = 0
        selected = []
        if detected:
            _, valid = lift(depth, np.asarray([k.pt for k in detected], float))
            candidates = [k for k, v in zip(detected, valid, strict=True) if v]
            self.liftable_count = len(candidates)
            candidates.sort(key=lambda k: (-k.response, k.pt[1], k.pt[0], k.size, k.octave, k.angle))
            for keypoint in candidates:
                cell = cell_index(*keypoint.pt)
                if self.cell_counts[cell] < FEATURE_RULES['per_cell']:
                    selected.append(keypoint)
                    self.cell_counts[cell] += 1
        if selected:
            self.keypoints, self.descriptors = detector.compute(self.gray, selected)
        else:
            self.keypoints, self.descriptors = [], None
        if len(self.keypoints) != sum(self.cell_counts) or len(self.keypoints) > FEATURE_RULES['maximum_selected']:
            raise SensorContractError('descriptor population changed after bounded selection')
        assert RULES['grid_columns'] == 4 and RULES['grid_rows'] == 3

    def witness(self):
        return dict(detected_features=self.detected_count, liftable_features=self.liftable_count,
            selected_features=len(self.keypoints), selected_per_cell=self.cell_counts.copy(),
            feature_selection='spatial_public_depth_sift_v1', native_state_used=False)
