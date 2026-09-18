"""Preserve immediate patch witnesses while amortizing long historical searches."""
from lewm.density_routed_floor_controller_development import DensityRoutedFloorController, CONTROLLER as BASELINE
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.visibility_batched_footprint_controller_development import PATCH_FIELDS
from lewm.visibility_batched_retained_floor_patch_development import VisibilityBatchedRetainedFloorPatches
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches

CONTROLLER='progressive_batched_density_routed_floor_controller_v1'
FLAG='progressive_retained_floor_patch_batching_enabled'


def install_fresh_progressive_patches(memory):
    if type(memory) is not MeasuredFloorTransportMemory:
        raise ValueError('exact original measured floor memory required')
    patches=[getattr(memory,name) for name in PATCH_FIELDS]
    if (len({id(p) for p in patches})!=2 or len({id(p.frames) for p in patches})!=2
            or any(type(p) is not VisibilityBatchedRetainedFloorPatches or set(vars(p))!={'frames'}
                or type(p.frames) is not list or p.frames for p in patches)):
        raise ValueError('two distinct completely empty visibility-batched patch stores required')
    for name in PATCH_FIELDS:setattr(memory,name,ProgressiveBatchedRetainedFloorPatches())


class ProgressiveBatchedFloorController(DensityRoutedFloorController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.memory is not self.mapper.surface:raise ValueError('original memory/map alias required')
        install_fresh_progressive_patches(self.memory)

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs)|{'controller':CONTROLLER,FLAG:True}


def normalize_to_density_routed(decision):
    if decision.get('controller')!=CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit progressive-batched controller identity required')
    result=decision.copy();result.pop(FLAG);result['controller']=BASELINE
    return result
