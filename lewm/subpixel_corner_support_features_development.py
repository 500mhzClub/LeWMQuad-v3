"""Separate measured subpixel corner candidate; downstream matching is unchanged."""
import cv2
import numpy as np
from lewm.corner_support_features_development import CornerSupportFeatureFrame, CORNER_RULES
from lewm.rgbd_correspondence_motion_development import lift
from lewm.spatial_support_features_development import cell_index

SUBPIXEL_RULES = dict(window_radius_pixels=5, iterations=30, epsilon_pixels=.01,
    maximum_refinement_displacement_pixels=2.)


class SubpixelCornerSupportFeatureFrame(CornerSupportFeatureFrame):
    def __init__(self, rgb, depth):
        super().__init__(rgb, depth)
        self.integer_selected_count = len(self.keypoints)
        self.refinement_rejected = 0
        self.maximum_refinement_pixels = 0.
        if not self.keypoints:
            return
        original = np.float32([k.pt for k in self.keypoints]).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(self.gray, original.copy(),
            (SUBPIXEL_RULES['window_radius_pixels'],) * 2, (-1, -1),
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
             SUBPIXEL_RULES['iterations'], SUBPIXEL_RULES['epsilon_pixels']))[:, 0]
        finite = np.isfinite(refined).all(1)
        safe = np.where(finite[:, None], refined, original[:, 0])
        displacement = np.linalg.norm(safe - original[:, 0], axis=1)
        _, valid = lift(self.depth, safe)
        valid &= finite & (displacement <= SUBPIXEL_RULES['maximum_refinement_displacement_pixels'])
        self.maximum_refinement_pixels = float(displacement[finite].max(initial=0.))
        self.cell_counts = [0] * 12
        selected, seen = [], set()
        for i, (x, y) in enumerate(safe):
            if not valid[i]:
                continue
            location = tuple(np.rint(np.asarray([x, y]) * 2).astype(int))
            cell = cell_index(x, y)
            if location in seen or self.cell_counts[cell] >= CORNER_RULES['per_cell']:
                continue
            selected.append(cv2.KeyPoint(float(x), float(y), CORNER_RULES['descriptor_size_pixels'],
                CORNER_RULES['descriptor_angle_degrees']))
            seen.add(location); self.cell_counts[cell] += 1
        self.refinement_rejected = self.integer_selected_count - len(selected)
        if selected:
            self.keypoints, self.descriptors = cv2.SIFT_create().compute(self.gray, selected)
        else:
            self.keypoints, self.descriptors = [], None
        if len(self.keypoints) != sum(self.cell_counts):
            raise ValueError('exact refined descriptor population required')

    def witness(self):
        return super().witness() | dict(feature_selection='subpixel_corner_upright_sift_support_v1',
            integer_selected_features=self.integer_selected_count,
            refinement_rejected=self.refinement_rejected,
            maximum_refinement_pixels=self.maximum_refinement_pixels,
            matching_thresholds_unchanged=True, candidate_adopted=False)
