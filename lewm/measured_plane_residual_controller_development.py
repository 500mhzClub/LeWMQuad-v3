"""Independent measured-plane observer inside the existing learned planner."""
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion


class MeasuredPlaneResidualController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='measured_plane_residual_continuation_controller_v1',
            measured_plane_constrained_estimator=True)
