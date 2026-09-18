"""Combine measured-plane perception with the completed single-pass controller.

The existing verified geometry, bounds and receipt optimizations are retained.
No optimization whose full replay is still pending is included.
"""
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from lewm.single_pass_body_projected_controller_development import SinglePassBodyProjectedController

CONTROLLER = 'measured_plane_single_pass_controller_v1'


class MeasuredPlaneSinglePassController(SinglePassBodyProjectedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller=CONTROLLER, measured_plane_constrained_estimator=True)
