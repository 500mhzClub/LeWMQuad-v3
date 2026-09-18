"""Exercise original geometry through the composed receipt copier and cache."""
from copy import deepcopy
from types import FunctionType
import pytest

from lewm import receipt_copied_footprint_development as candidate
from lewm.packed_fused_scoped_controller_development import PackedFusedScopedController
from lewm.fused_scoped_batched_controller_development import FusedScopedBatchedSelector
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.confirmed_auxiliary_floor_memory_development import ConfirmedAuxiliaryFloorMemory, confirmed_contact_check
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMemory
from lewm.tests import test_scoped_batched_footprint_controller_development as original
from lewm.tests.test_packed_fused_scoped_controller_development import state
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None, geometry, **kwargs()) for kind in
            (PackedFusedScopedController, candidate.ReceiptCopiedFootprintController)]


@pytest.mark.parametrize('old,new,bindings', [
    (confirmed_contact_check, candidate.copied_contact_check, {'deepcopy':candidate.copy_receipt}),
    (ConfirmedAuxiliaryFloorMemory.footprint, candidate.CopiedConfirmedMemory.footprint,
     {'confirmed_contact_check':candidate.copied_contact_check}),
    (LaterResolvedFloorMemory.footprint, candidate.CopiedLaterMemory.footprint,
     {'deepcopy':candidate.copy_receipt}),
    (FusedScopedBatchedSelector.choose, candidate.ReceiptCopiedFootprintSelector.choose,
     {'FusedScopedFootprintReuse':candidate.ReceiptCopiedFootprintScope}),
])
def test_only_explicit_globals_change_with_exact_code_closure_and_defaults(old,new,bindings):
    assert old.__code__ is new.__code__ and old.__closure__ is new.__closure__
    assert old.__defaults__ is new.__defaults__ and old.__kwdefaults__ is new.__kwdefaults__
    assert old.__globals__ is not new.__globals__
    assert set(old.__globals__) == set(new.__globals__)
    changed = {k:v for k,v in new.__globals__.items() if old.__globals__[k] is not v}
    assert changed == bindings
    if 'deepcopy' in old.__globals__: assert old.__globals__['deepcopy'] is deepcopy


def test_original_owned_memory_and_observation_methods_unchanged():
    old,new = controllers()
    assert type(new.memory) is type(old.memory) is MeasuredFloorTransportMemory
    assert new.memory is new.mapper.surface and new.selector.residual is new.residual
    for name in ('observe','advance'):
        assert getattr(type(new),name) is getattr(type(old),name)
    assert fingerprint(state(old)) == fingerprint(state(new))
    view = candidate.footprint_view(new.memory)
    assert vars(view) is vars(new.memory)
    assert view.footprint.__func__ is candidate.CopiedLaterMemory.footprint
    mro = type(view).__mro__
    assert mro.index(LaterResolvedFloorMemory) < mro.index(candidate.CopiedConfirmedMemory)
    assert mro.index(candidate.CopiedConfirmedMemory) < mro.index(ConfirmedAuxiliaryFloorMemory)
    with pytest.raises(TypeError): view.failed = True
    with pytest.raises(TypeError): del view.failed
    new.memory.failed = True
    assert view.failed is True


def test_scope_observes_latched_failure_and_closes_owned_references():
    old,new = controllers()
    scope = candidate.ReceiptCopiedFootprintScope(new.memory, None)
    with scope:
        assert vars(scope._memory) is vars(new.memory)
        new.memory.failed = True
        with pytest.raises(ValueError): scope.footprint(None,[0.,0.],0.,now_ns=1)
    assert scope._memory is None and scope._geometry is None and scope._closed


@pytest.mark.parametrize('fault',['subclass','instance_override'])
def test_query_view_rejects_unsupported_memory(fault):
    _,new=controllers()
    memory = new.memory
    if fault == 'subclass': memory = type('Custom',(MeasuredFloorTransportMemory,),{})(identity=(0,0,0))
    else: memory.footprint = lambda *a,**k: None
    with pytest.raises(ValueError):candidate.footprint_view(memory)


def test_original_stop_and_metadata():
    old,new=controllers()
    expected=old.observe({}, {}, {}, now_ns=1)
    actual=new.observe({}, {}, {}, now_ns=1)
    assert candidate.normalize_to_packed(actual)==expected
    assert actual['terminal']=='SENSOR_OR_MODEL_FAILURE'
    with pytest.raises(ValueError):candidate.normalize_to_packed(expected)


# Runs the original public-packet observation and articulated-footprint test.
# It compares state/receipts, exercises cache misses and hits, mutates one
# public result to prove ownership, and checks both retained patch stores.
_function=original.test_actual_observation_and_scoped_batched_robot_footprints_match
_namespace=dict(_function.__globals__,controllers=controllers,state=state,
                normalize_to_scoped=candidate.normalize_to_packed)
test_actual_observation_and_owned_footprint_receipts=FunctionType(
    _function.__code__,_namespace,'test_actual_observation_and_owned_footprint_receipts',_function.__defaults__)
