"""Change the copier in the same two pure footprint evidence paths only."""
from lewm.deferred_atomic_memo_copy_development import copy_receipt
from lewm.confirmed_auxiliary_floor_memory_development import ConfirmedAuxiliaryFloorMemory,confirmed_contact_check
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMemory
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm import receipt_copied_footprint_development as original
from lewm.single_pass_body_projected_controller_development import SinglePassBodyProjectedController,CONTROLLER as BASELINE

CONTROLLER='deferred_memo_single_pass_controller_v1'
FLAG='deferred_atomic_memo_receipt_copy_enabled'
copied_contact_check=original.fork(confirmed_contact_check,deepcopy=copy_receipt)


class DeferredMemoConfirmedMemory(ConfirmedAuxiliaryFloorMemory):
    footprint=original.fork(ConfirmedAuxiliaryFloorMemory.footprint,confirmed_contact_check=copied_contact_check)


class DeferredMemoLaterMemory(LaterResolvedFloorMemory,DeferredMemoConfirmedMemory):
    footprint=original.fork(LaterResolvedFloorMemory.footprint,deepcopy=copy_receipt)


class DeferredMemoFootprintView(MeasuredFloorTransportMemory,DeferredMemoLaterMemory):
    __setattr__=original.ReceiptCopiedFootprintView.__setattr__
    __delattr__=original.ReceiptCopiedFootprintView.__delattr__


footprint_view=original.fork(original.footprint_view,ReceiptCopiedFootprintView=DeferredMemoFootprintView)


class DeferredMemoFootprintScope(original.ReceiptCopiedFootprintScope):
    __init__=original.fork(original.ReceiptCopiedFootprintScope.__init__,footprint_view=footprint_view)


class DeferredMemoFootprintSelector(original.ReceiptCopiedFootprintSelector):
    choose=original.fork(original.ReceiptCopiedFootprintSelector.choose,
        FusedScopedFootprintReuse=DeferredMemoFootprintScope)


def install_fresh_selector(controller):
    previous=controller.selector
    if (type(previous) is not original.ReceiptCopiedFootprintSelector or previous.residual is not controller.residual
            or controller.memory is not controller.mapper.surface or controller.memory.route
            or controller.last_ns is not None or controller.memory.failed or controller.mapper.failed):
        raise ValueError('fresh unchanged single-pass controller and original selector aliases required')
    selector=object.__new__(DeferredMemoFootprintSelector)
    selector.__dict__=vars(previous).copy()
    controller.selector=selector


class DeferredMemoSinglePassController(SinglePassBodyProjectedController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        install_fresh_selector(self)

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs) | {'controller':CONTROLLER,FLAG:True}


def normalize_to_single_pass(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit deferred-memo controller identity required')
    result=decision.copy();result.pop(FLAG);result['controller']=BASELINE
    return result
