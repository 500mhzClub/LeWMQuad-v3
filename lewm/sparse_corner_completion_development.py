"""Fill spare sparse-view slots with weaker measured corners, up to 150.

Preserve the stronger sparse-view features and descriptors. Pose admission,
depth estimation, matching and temporal checks retain their existing rules.
"""
import cv2
import numpy as np

from lewm.sparse_feature_budget_tracking_development import SparseFeatureFrame150
from lewm.corner_support_features_development import CORNER_RULES
from lewm.conditioned_support_150_tracker_development import _BudgetDual
from lewm.development_support_tracker_development import use
from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitPose
from lewm.rgbd_correspondence_motion_development import lift
from lewm.spatial_support_features_development import cell_index

COMPLETION_QUALITY = .001


class CompletedSparseFeatureFrame150(SparseFeatureFrame150):
    def __init__(self, rgb, depth):
        super().__init__(rgb, depth)
        self.added_corners = 0
        self.completion_detected = None
        if self.liftable_count > 150 or len(self.keypoints) == 150:
            return
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=0,
            qualityLevel=COMPLETION_QUALITY,
            minDistance=CORNER_RULES['minimum_distance_pixels'],
            mask=depth['valid'].astype(np.uint8)*255,
            blockSize=CORNER_RULES['block_size'], useHarrisDetector=False)
        points = np.empty((0, 2), np.float32) if points is None else points[:, 0]
        self.completion_detected = len(points)
        _, valid = lift(depth, points)
        points = points[valid]
        response = cv2.cornerMinEigenVal(self.gray, blockSize=CORNER_RULES['block_size'])
        ordered = sorted(points.tolist(), key=lambda p:
            (-float(response[int(p[1]), int(p[0])]), p[1], p[0]))
        existing = {tuple(k.pt) for k in self.keypoints}
        extra = [cv2.KeyPoint(x, y, CORNER_RULES['descriptor_size_pixels'],
            CORNER_RULES['descriptor_angle_degrees']) for x, y in ordered
            if (x, y) not in existing][:150-len(self.keypoints)]
        if extra:
            keys, descriptors = cv2.SIFT_create().compute(self.gray, extra)
            self.added_corners = len(keys)
            self.keypoints = list(self.keypoints) + list(keys)
            self.descriptors = (descriptors if self.descriptors is None else
                np.concatenate((self.descriptors, descriptors)))
            self.descriptors_evaluated += len(keys)
            for key in keys:
                self.cell_counts[cell_index(*key.pt)] += 1
        assert len(self.keypoints) <= 150

    def witness(self):
        return super().witness() | dict(
            feature_selection='sparse_corner_completion150_v1',
            original_quality_level=CORNER_RULES['quality_level'],
            completion_quality_level=COMPLETION_QUALITY,
            completion_detected_features=self.completion_detected,
            added_weaker_corners=self.added_corners,
            maximum_features_per_cell=150 if self.liftable_count <= 150 else 13)


class _CompletedBudgetDual(_BudgetDual):
    observe = use(_BudgetDual.observe, CornerSupportFeatureFrame=CompletedSparseFeatureFrame150)


class SparseCornerCompletionPose(CadencedViewRevisitPose, _CompletedBudgetDual):
    pass
