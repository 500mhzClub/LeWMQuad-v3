"""Original anchored policy with batched historical floor-patch queries only."""
from lewm.batched_retained_floor_patch_development import BatchedRetainedFloorPatches
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController

CONTROLLER = 'batched_patch_residual_anchored_continuation_controller_v1'
FLAG = 'batched_retained_floor_queries_enabled'
PATCH_FIELDS = ('patches', 'auxiliary_patches')


class BatchedPatchAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self.memory) is not MeasuredFloorTransportMemory or self.memory is not self.mapper.surface:
            raise ValueError('original measured floor memory and map alias required')
        for name in PATCH_FIELDS:
            previous = getattr(self.memory, name)
            if type(previous) is not RetainedFloorPatches or vars(previous) != {'frames': []}:
                raise ValueError('fresh original primary and auxiliary patch stores required')
        for name in PATCH_FIELDS:
            setattr(self.memory, name, BatchedRetainedFloorPatches())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
