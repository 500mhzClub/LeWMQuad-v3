"""Existing direct-flow observer inside the unchanged anchored planning policy."""
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion

CONTROLLER = 'direct_flow_residual_anchored_continuation_controller_v1'
FLAG = 'direct_corner_flow_missingness_fallback_enabled'


class DirectFlowResidualAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = DirectFlowDualCameraVisualMotion(identity=(0,0,0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
