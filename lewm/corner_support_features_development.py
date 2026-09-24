"""Measured Shi-Tomasi corners with upright SIFT descriptors and frozen matching.

Large low-frequency block corners need not be SIFT scale-space extrema.
Descriptor ratio, mutual pairing, LK and geometry remain downstream gates.
This is a new unqualified detector, not an accepted pose or visibility bound.
"""
from copy import deepcopy
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_correspondence_motion_development import lift
from lewm.spatial_support_features_development import cell_index

CORNER_RULES = dict(quality_level=.01, minimum_distance_pixels=5., block_size=3,
    per_cell=50, maximum_selected=600, maximum_detected=50000,
    descriptor_size_pixels=8., descriptor_angle_degrees=0.)


class CornerSupportFeatureFrame:
    def __init__(self, rgb, depth):
        if not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise SensorContractError('exact current RGB frame required')
        self.rgb = rgb.copy(); self.depth = deepcopy(depth)
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=0,
            qualityLevel=CORNER_RULES['quality_level'], minDistance=CORNER_RULES['minimum_distance_pixels'],
            mask=depth['valid'].astype(np.uint8)*255, blockSize=CORNER_RULES['block_size'], useHarrisDetector=False)
        points = np.empty((0, 2), np.float32) if points is None else points[:, 0]
        if len(points) > CORNER_RULES['maximum_detected']:
            raise SensorContractError('bounded corner population exceeded')
        self.detected_count = len(points); self.cell_counts = [0]*12
        selected = []
        _, valid = lift(depth, points)
        points = points[valid]
        self.liftable_count = len(points)
        response = cv2.cornerMinEigenVal(self.gray, blockSize=CORNER_RULES['block_size'])
        order = sorted(points.tolist(), key=lambda p: (-float(response[int(p[1]), int(p[0])]), p[1], p[0]))
        for x, y in order:
            cell = cell_index(x, y)
            if self.cell_counts[cell] < CORNER_RULES['per_cell']:
                selected.append(cv2.KeyPoint(x, y, CORNER_RULES['descriptor_size_pixels'],
                    CORNER_RULES['descriptor_angle_degrees']))
                self.cell_counts[cell] += 1
        if selected:
            self.keypoints, self.descriptors = cv2.SIFT_create().compute(self.gray, selected)
        else:
            self.keypoints, self.descriptors = [], None
        if len(self.keypoints) != sum(self.cell_counts) or len(self.keypoints) > CORNER_RULES['maximum_selected']:
            raise SensorContractError('exact bounded corner descriptor population required')

    def witness(self):
        return dict(detected_features=self.detected_count, liftable_features=self.liftable_count,
            selected_features=len(self.keypoints), selected_per_cell=self.cell_counts.copy(),
            feature_selection='corner_upright_sift_support_v1', native_state_used=False)
