"""Direct image association across the actual bounded camera interval.

The predecessor kept a fixed-100-ms fallback guard after accepting variable
camera intervals. This successor permits that fallback on the last observed
image pair. Pixel, depth, rigid-fit, disagreement and displacement limits stay
unchanged. No intermediate images or poses are synthesized.
"""
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.gapped_camera_plane_tracker_development import (
    GappedCameraPlaneTracker, _interval, _with_interval)


class _ActualIntervalDirectFlow(DirectFlowDualCameraAnchorPose):
    def _candidate(self, ref, current, G):
        now = current['primary'].depth['measured_ns']
        return _with_interval(DirectFlowDualCameraAnchorPose._candidate, self,
            _interval(self, now), ref, current, G)


class GappedDirectFlowTracker(GappedCameraPlaneTracker, _ActualIntervalDirectFlow):
    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        return result | dict(direct_flow_uses_actual_visual_interval=True,
            direct_flow_image_and_geometry_limits_unchanged=True,
            intermediate_camera_images_synthesized=False)
