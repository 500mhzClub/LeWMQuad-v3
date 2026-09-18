"""Original anchored selection with temporary read-only footprint evidence."""
from lewm.frozen_footprint_receipts_development import FootprintReceiptMap, detach_receipts
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)

CONTROLLER = 'frozen_footprint_residual_anchored_continuation_controller_v1'
FLAG = 'invocation_frozen_footprint_receipts_enabled'


class FrozenFootprintAnchoredSelector(ResidualAnchoredContinuationSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        result = super().choose(model, history, FootprintReceiptMap(mapper), geometry, now_ns=now_ns)
        return detach_receipts(result)


class FrozenFootprintAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = FrozenFootprintAnchoredSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
