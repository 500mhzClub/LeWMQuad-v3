"""Use the existing copier in two pure footprint evidence-building paths.

Only invocation-local views and private function globals differ. The original
memory type, owned fields, observation methods and selector guards remain.
"""
from copy import deepcopy
from types import FunctionType

from lewm.receipt_copy_development import copy_receipt
from lewm.confirmed_auxiliary_floor_memory_development import (
    ConfirmedAuxiliaryFloorMemory, confirmed_contact_check)
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMemory
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.fused_scoped_footprint_development import FusedScopedFootprintReuse
from lewm.fused_scoped_batched_controller_development import FusedScopedBatchedSelector
from lewm.packed_fused_scoped_controller_development import PackedFusedScopedController, CONTROLLER as BASELINE

CONTROLLER = 'receipt_copied_footprint_packed_fused_controller_v1'
FLAG = 'footprint_evidence_receipt_copy_enabled'


def fork(original, **replacements):
    if any(name not in original.__globals__ for name in replacements):
        raise ValueError('only existing explicit global bindings may change')
    if 'deepcopy' in replacements and original.__globals__['deepcopy'] is not deepcopy:
        raise ValueError('original standard copier required')
    result = FunctionType(original.__code__, original.__globals__ | replacements,
                          original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    result.__annotations__ = original.__annotations__
    result.__qualname__ = original.__qualname__
    result.__doc__ = original.__doc__
    return result


copied_contact_check = fork(confirmed_contact_check, deepcopy=copy_receipt)


class CopiedConfirmedMemory(ConfirmedAuxiliaryFloorMemory):
    footprint = fork(ConfirmedAuxiliaryFloorMemory.footprint,
                     confirmed_contact_check=copied_contact_check)


class CopiedLaterMemory(LaterResolvedFloorMemory, CopiedConfirmedMemory):
    footprint = fork(LaterResolvedFloorMemory.footprint, deepcopy=copy_receipt)


class ReceiptCopiedFootprintView(MeasuredFloorTransportMemory, CopiedLaterMemory):
    def __setattr__(self, name, value):
        raise TypeError('selection footprint view does not assign memory fields')

    def __delattr__(self, name):
        raise TypeError('selection footprint view does not delete memory fields')


def footprint_view(memory):
    if type(memory) is not MeasuredFloorTransportMemory or 'footprint' in vars(memory):
        raise ValueError('exact original memory without instance query override required')
    view = object.__new__(ReceiptCopiedFootprintView)
    # Share the original field dictionary so stale/failure checks also see
    # changes made during the scope. The two footprint paths only read it.
    # No view is stored in the memory, mapper or returned public receipts.
    object.__setattr__(view, '__dict__', vars(memory))
    return view


class ReceiptCopiedFootprintScope(FusedScopedFootprintReuse):
    def __init__(self, memory, geometry):
        super().__init__(footprint_view(memory), geometry)


class ReceiptCopiedFootprintSelector(FusedScopedBatchedSelector):
    choose = fork(FusedScopedBatchedSelector.choose,
                  FusedScopedFootprintReuse=ReceiptCopiedFootprintScope)


class ReceiptCopiedFootprintController(PackedFusedScopedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ReceiptCopiedFootprintSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}


def normalize_to_packed(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit footprint receipt copier identity required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = BASELINE
    return result
