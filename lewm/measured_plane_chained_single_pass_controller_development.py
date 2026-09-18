"""Source-only composition of chained tracking and the single-pass controller.

Neither existing controller is modified. This composition needs its own
recorded-history equivalence evidence before adoption in a native experiment.
"""
from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorVisualMotion
from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController

CONTROLLER = 'measured_plane_chained_single_pass_controller_v1'


class MeasuredPlaneChainedSinglePassController(MeasuredPlaneSinglePassController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneChainedAnchorVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller=CONTROLLER,
            chained_anchor_reacquisition_enabled=True,
            direct_corner_flow_missingness_fallback_enabled=True)
