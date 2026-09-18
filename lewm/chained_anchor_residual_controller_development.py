"""Separate chained-anchor observer inside the original learned planning policy."""
from lewm.direct_flow_residual_anchored_controller_development import DirectFlowResidualAnchoredController
from lewm.chained_anchor_visual_motion_development import ChainedAnchorVisualMotion

CONTROLLER = 'chained_anchor_residual_continuation_controller_v1'
FLAG = 'chained_retained_anchor_reacquisition_enabled'


class ChainedAnchorResidualController(DirectFlowResidualAnchoredController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = ChainedAnchorVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
