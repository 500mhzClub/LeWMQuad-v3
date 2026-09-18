"""Original anchored policy with selection-local exact footprint reuse."""
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.frozen_footprint_receipts_development import FootprintReceiptMap, detach_receipts
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)
from lewm.scoped_footprint_reuse_development import ScopedFootprintReuse, ScopedFootprintMap

CONTROLLER = 'scoped_footprint_reuse_residual_anchored_continuation_controller_v1'
FLAG = 'selection_scoped_exact_footprint_reuse_enabled'


class ScopedFootprintAnchoredSelector(ResidualAnchoredContinuationSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        # The reviewed production chain does not modify observed memory or
        # geometry during footprint filtering and the two recovery passes.
        # Unknown subclasses/proxies/instance overrides retain uncached calls.
        supported = (type(mapper.surface) is MeasuredFloorTransportMemory
                     and type(geometry) is ArticulatedCollisionGeometry
                     and 'footprint' not in vars(mapper.surface)
                     and 'supports' not in vars(geometry))
        if not supported:
            return detach_receipts(super().choose(
                model, history, FootprintReceiptMap(mapper), geometry, now_ns=now_ns))
        with ScopedFootprintReuse(mapper.surface, geometry) as memory:
            result = super().choose(model, history, ScopedFootprintMap(mapper, memory),
                                    geometry, now_ns=now_ns)
        return detach_receipts(result)


class ScopedFootprintAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ScopedFootprintAnchoredSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
