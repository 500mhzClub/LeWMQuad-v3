"""Original anchored controller with invocation-local surface receipt sharing."""
from lewm.residual_first_interval_controller_development import ResidualFirstIntervalSelector
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.shared_surface_anchored_selection_development import (
    reconsider_with_shared_surface as reconsider_anchored_continuation)

CONTROLLER = 'shared_surface_residual_anchored_continuation_controller_v1'
FLAG = 'anchored_surface_receipt_sharing_enabled'


class SharedSurfaceAnchoredSelector(ResidualFirstIntervalSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        original = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        return reconsider_anchored_continuation(
            original, self.residual.snapshot(), mapper, geometry, now_ns=now_ns)


class SharedSurfaceAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = SharedSurfaceAnchoredSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | { 'controller': CONTROLLER, FLAG: True }
