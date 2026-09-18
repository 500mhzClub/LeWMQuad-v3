"""Original receipt-copy controller with empty visibility enumeration skipped."""
from lewm.receipt_copied_footprint_development import ReceiptCopiedFootprintController, CONTROLLER as BASELINE
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.batched_retained_floor_patch_development import BatchedRetainedFloorPatches
from lewm.visibility_batched_retained_floor_patch_development import VisibilityBatchedRetainedFloorPatches

PATCH_FIELDS=('patches','auxiliary_patches')
CONTROLLER='visibility_batched_receipt_copied_footprint_controller_v1'
FLAG='empty_retained_patch_visibility_enumeration_skipped'


def install_empty_visibility_patches(memory):
    if type(memory) is not MeasuredFloorTransportMemory:
        raise ValueError('exact original measured floor memory required')
    patches=[getattr(memory,name) for name in PATCH_FIELDS]
    if (len({id(p) for p in patches})!=2 or len({id(p.frames) for p in patches})!=2
            or any(type(p) is not BatchedRetainedFloorPatches or set(vars(p))!={'frames'}
                or type(p.frames) is not list or p.frames for p in patches)):
        raise ValueError('two distinct completely empty original batched patch stores required')
    for name in PATCH_FIELDS:setattr(memory,name,VisibilityBatchedRetainedFloorPatches())


class VisibilityBatchedFootprintController(ReceiptCopiedFootprintController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.memory is not self.mapper.surface:raise ValueError('original memory/map alias required')
        install_empty_visibility_patches(self.memory)

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs)|{'controller':CONTROLLER,FLAG:True}


def normalize_to_receipt_copied(decision):
    if decision.get('controller')!=CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit visibility-batched controller identity required')
    result=decision.copy();result.pop(FLAG);result['controller']=BASELINE
    return result
