"""Combine exact selection-local reuse with existing batched patch queries."""
from lewm.batched_patch_anchored_controller_development import BatchedPatchAnchoredController
from lewm.scoped_footprint_anchored_controller_development import ScopedFootprintAnchoredSelector

CONTROLLER = 'scoped_batched_footprint_residual_anchored_continuation_controller_v1'
FLAG = 'scoped_and_batched_footprint_queries_enabled'


class ScopedBatchedFootprintController(BatchedPatchAnchoredController):
    def __init__(self, *args, **kwargs):
        # The existing batched constructor enforces fresh primary/auxiliary
        # stores and preserves the exact memory type required by scoped reuse.
        super().__init__(*args, **kwargs)
        self.selector = ScopedFootprintAnchoredSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {
            'controller': CONTROLLER, FLAG: True,
            'selection_scoped_exact_footprint_reuse_enabled': True}


def normalize_to_scoped(decision):
    """Remove only declared composition metadata for an incremental comparison."""
    from lewm.scoped_footprint_anchored_controller_development import CONTROLLER as baseline
    if (decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True
            or decision.get('batched_retained_floor_queries_enabled') is not True
            or decision.get('selection_scoped_exact_footprint_reuse_enabled') is not True):
        raise ValueError('explicit combined scoped and batched query implementation required')
    result = decision.copy()
    result.pop(FLAG); result.pop('batched_retained_floor_queries_enabled')
    result['controller'] = baseline
    return result
