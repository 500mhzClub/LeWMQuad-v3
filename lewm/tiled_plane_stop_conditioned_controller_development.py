"""Reuse the existing floor kernel inside measured-plane visual tracking.

Only the candidate-extraction implementation changes. The original image
registration, plane fit, temporal admission and stopping mission are inherited.
This separate composition is not installed into existing experiment runners.
"""
from lewm.eligible_floor_registration_development import bind
from lewm.tiled_density_floor_registration_development import measured_candidates
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.measured_plane_chained_anchor_development import (
    MeasuredPlaneChainedAnchorPose, MeasuredPlaneChainedAnchorVisualMotion)
from lewm.stop_conditioned_settling_development import StopConditionedSettlingController


class TiledPlaneChainedPose(MeasuredPlaneChainedAnchorPose):
    _prepare_plane = bind(MeasuredPlaneDualCameraPose._prepare_plane,
        measured_candidates=measured_candidates)


class TiledPlaneChainedVisualMotion(MeasuredPlaneChainedAnchorVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = TiledPlaneChainedPose()


class TiledPlaneStopConditionedController(StopConditionedSettlingController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = TiledPlaneChainedVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            tiled_plane_tracking_kernel_enabled=True)
