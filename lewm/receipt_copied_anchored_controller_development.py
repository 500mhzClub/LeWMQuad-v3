"""Same anchored-continuation policy with an isolated receipt-copy provider."""
from lewm.residual_first_interval_controller_development import ResidualFirstIntervalSelector
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.receipt_copied_anchored_selection_development import (
    reconsider_with_receipt_copy as reconsider_anchored_continuation)


class ReceiptCopiedAnchoredSelector(ResidualFirstIntervalSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        original = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        return reconsider_anchored_continuation(
            original, self.residual.snapshot(), mapper, geometry, now_ns=now_ns)


class ReceiptCopiedAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ReceiptCopiedAnchoredSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='receipt_copied_residual_anchored_continuation_controller_v1',
            anchored_selection_receipt_copy_enabled=True)
