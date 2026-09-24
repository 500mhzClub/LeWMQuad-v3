"""Separate measured-plane plus retained-image-chain observer candidate.

The measured-plane wrapper encloses every original/direct/chained image fit.
The existing chained observer owns fallback, conflict checks, reference rules
and bounded image history. Neither parent implementation is modified. This
candidate has no native execution or navigation qualification result.
"""
from copy import deepcopy

from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController


class MeasuredPlaneChainedAnchorPose(MeasuredPlaneDualCameraPose, ChainedAnchorDualCameraPose):
    """Cooperative MRO refines complete image fits before temporal admission."""


class MeasuredPlaneChainedAnchorVisualMotion(MeasuredPlaneVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = MeasuredPlaneChainedAnchorPose()

    def snapshot(self, *, now_ns):
        result = super().snapshot(now_ns=now_ns)
        for field, key in (('last_direct_flow_fallback', 'direct_corner_flow_fallback'),
                ('last_chained_anchor_fallback', 'chained_anchor_fallback')):
            value = getattr(self.model, field)
            if value is not None: result[key] = deepcopy(value)
        return result


class MeasuredPlaneChainedAnchorController(MeasuredPlaneResidualController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneChainedAnchorVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='measured_plane_chained_anchor_controller_v1',
            chained_anchor_reacquisition_enabled=True,
            direct_corner_flow_missingness_fallback_enabled=True)
