"""Combined controller with cheaper owned footprint receipt construction."""
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.frozen_footprint_receipts_development import FootprintReceiptMap, detach_receipts
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationSelector
from lewm.scoped_footprint_reuse_development import ScopedFootprintMap
from lewm.scoped_batched_footprint_controller_development import (
    ScopedBatchedFootprintController, CONTROLLER as BASELINE)
from lewm.fused_scoped_footprint_development import FusedScopedFootprintReuse

CONTROLLER = 'fused_receipt_scoped_batched_anchored_controller_v1'
FLAG = 'fused_footprint_receipt_construction_enabled'


class FusedScopedBatchedSelector(ResidualAnchoredContinuationSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        supported = (type(mapper.surface) is MeasuredFloorTransportMemory
                     and type(geometry) is ArticulatedCollisionGeometry
                     and 'footprint' not in vars(mapper.surface)
                     and 'supports' not in vars(geometry))
        if not supported:
            return detach_receipts(super().choose(
                model, history, FootprintReceiptMap(mapper), geometry, now_ns=now_ns))
        with FusedScopedFootprintReuse(mapper.surface, geometry) as memory:
            result = super().choose(model, history, ScopedFootprintMap(mapper, memory),
                                    geometry, now_ns=now_ns)
        return detach_receipts(result)


class FusedScopedBatchedController(ScopedBatchedFootprintController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = FusedScopedBatchedSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}


def normalize_to_combined(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit fused receipt controller required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = BASELINE
    return result
