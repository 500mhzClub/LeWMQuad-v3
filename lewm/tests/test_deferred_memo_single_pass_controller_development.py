"""Unchanged memory, actual public geometry and detached footprint receipts."""
from types import FunctionType
import pytest
from lewm import deferred_memo_single_pass_controller_development as candidate
from lewm.single_pass_body_projected_controller_development import SinglePassBodyProjectedController
from lewm.receipt_copied_footprint_development import ReceiptCopiedFootprintSelector
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
from lewm.tests.test_single_pass_body_projected_controller_development import state
from lewm.tests import test_scoped_batched_footprint_controller_development as packets
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def controllers(geometry=None):
    return [kind(None,geometry,**kwargs()) for kind in
        (SinglePassBodyProjectedController,candidate.DeferredMemoSinglePassController)]


def test_only_fresh_selector_is_replaced():
    old=SinglePassBodyProjectedController(None,None,**kwargs())
    fields=vars(old).copy();selector_fields=vars(old.selector).copy()
    before=fingerprint(state(old))
    candidate.install_fresh_selector(old)
    assert type(old.selector) is candidate.DeferredMemoFootprintSelector
    assert old.selector is not fields['selector']
    assert all(vars(old)[k] is v for k,v in fields.items() if k != 'selector')
    assert all(vars(old.selector)[k] is v for k,v in selector_fields.items())
    assert fingerprint(state(old)) == before
    baseline,revised=controllers()
    for method in ('observe','advance'):
        assert getattr(type(baseline),method) is getattr(type(revised),method)
    assert type(baseline.memory) is type(revised.memory)
    assert type(baseline.mapper) is type(revised.mapper)
    assert type(baseline.registration) is type(revised.registration)


@pytest.mark.parametrize('fault',['route','time','memory_alias','residual_alias','failed','already'])
def test_nonfresh_or_misaliased_installation_is_rejected_without_mutation(fault):
    old=SinglePassBodyProjectedController(None,None,**kwargs())
    if fault == 'route': old.memory.route.append({})
    elif fault == 'time': old.last_ns=1
    elif fault == 'memory_alias': old.memory=object()
    elif fault == 'residual_alias': old.selector.residual=object()
    elif fault == 'failed': old.memory.failed=True
    else: candidate.install_fresh_selector(old)
    previous=old.selector
    with pytest.raises(ValueError): candidate.install_fresh_selector(old)
    assert old.selector is previous


def test_actual_observation_and_owned_footprints_match(monkeypatch):
    original=packets.test_actual_observation_and_scoped_batched_robot_footprints_match
    check=FunctionType(original.__code__,original.__globals__ | dict(controllers=controllers,state=state,
        normalize_to_scoped=candidate.normalize_to_single_pass,
        BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches),
        'deferred_memo_packet_check',original.__defaults__)
    check(monkeypatch)


def test_original_query_code_and_scope_guards_remain_unchanged():
    source=candidate.original
    assert candidate.DeferredMemoFootprintSelector.choose.__code__ is source.ReceiptCopiedFootprintSelector.choose.__code__
    assert candidate.DeferredMemoFootprintScope.__init__.__code__ is source.ReceiptCopiedFootprintScope.__init__.__code__
    assert candidate.DeferredMemoFootprintScope.footprint is source.ReceiptCopiedFootprintScope.footprint
    assert candidate.footprint_view.__code__ is source.footprint_view.__code__
    assert candidate.DeferredMemoLaterMemory.footprint.__globals__['deepcopy'] is candidate.copy_receipt
    assert candidate.copied_contact_check.__globals__['deepcopy'] is candidate.copy_receipt
    assert candidate.DeferredMemoFootprintView.footprint is candidate.DeferredMemoLaterMemory.footprint


def test_original_failure_and_nested_evidence_are_preserved():
    old,new=controllers()
    expected=old.observe({},{},{},now_ns=1)
    actual=new.observe({},{},{},now_ns=1)
    assert actual['terminal']=='SENSOR_OR_MODEL_FAILURE'
    assert candidate.normalize_to_single_pass(actual)==expected
    with pytest.raises(ValueError): candidate.normalize_to_single_pass(expected)
    actual['new_selection']={'changed':True}
    assert candidate.normalize_to_single_pass(actual)['new_selection']=={'changed':True}
