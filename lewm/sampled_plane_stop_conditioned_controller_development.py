"""Use sampled plane extraction in a fresh development controller only."""
from lewm.eligible_floor_registration_development import bind
from lewm.sampled_plane_candidates_development import measured_candidates
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.measured_plane_chained_anchor_development import (
    MeasuredPlaneChainedAnchorPose, MeasuredPlaneChainedAnchorVisualMotion)
from lewm.extended_return_budget_transport_development import ExtendedReturnBudgetFloorRegistration
from lewm.stop_conditioned_settling_development import StopConditionedSettlingController


class SampledPlaneChainedPose(MeasuredPlaneChainedAnchorPose):
    _prepare_plane = bind(MeasuredPlaneDualCameraPose._prepare_plane,
        measured_candidates=measured_candidates)


class SampledPlaneChainedVisualMotion(MeasuredPlaneChainedAnchorVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = SampledPlaneChainedPose()


class SampledPlaneFloorRegistration(ExtendedReturnBudgetFloorRegistration):
    observe = bind(ExtendedReturnBudgetFloorRegistration.observe,
        measured_candidates=measured_candidates)


class SampledPlaneStopConditionedController(StopConditionedSettlingController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = SampledPlaneChainedVisualMotion(identity=(0, 0, 0))
        self.registration = SampledPlaneFloorRegistration(identity=self.registration.identity)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(sampled_plane_candidates_enabled=True)
