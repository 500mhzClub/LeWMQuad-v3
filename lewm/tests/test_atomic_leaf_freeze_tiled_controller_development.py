"""Freeze-only controller composition with actual public packets and geometry."""
from types import FunctionType

import pytest

from lewm import atomic_leaf_freeze_tiled_controller_development as candidate
from lewm import fused_scoped_footprint_development as fused
from lewm.receipt_copied_footprint_development import ReceiptCopiedFootprintScope, ReceiptCopiedFootprintSelector
from lewm.tiled_density_progressive_floor_controller_development import TiledDensityProgressiveFloorController
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from scripts.progressive_batched_floor_state_development import normalized_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests import test_scoped_batched_footprint_controller_development as original
from lewm.tests.test_frame_floor_cache_development import equal


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None, geometry, **kwargs()) for kind in
        (TiledDensityProgressiveFloorController, candidate.AtomicLeafFreezeTiledController)]


def state(controller):
    return dict(retained=normalized_state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
        occupied=controller.mapper.occupied, residual=controller.residual, history=controller.history)),
        registration=vars(controller.registration), cache_counts=controller.mapper.last_cache_counts)


def test_only_private_freeze_and_scope_bindings_change_original_clone_is_retained():
    pairs = [(fused.FusedScopedFootprintReuse.footprint, candidate.AtomicLeafFreezeFootprintScope.footprint,
        'freeze_ordinary_footprint', candidate.freeze_ordinary_footprint),
        (ReceiptCopiedFootprintSelector.choose, candidate.AtomicLeafFreezeFootprintSelector.choose,
        'FusedScopedFootprintReuse', candidate.AtomicLeafFreezeFootprintScope)]
    for old,new,key,replacement in pairs:
        assert old.__code__ is new.__code__ and old.__closure__ is new.__closure__
        assert old.__defaults__ is new.__defaults__ and old.__kwdefaults__ is new.__kwdefaults__
        assert old.__globals__ is not new.__globals__
        for name,value in old.__globals__.items(): assert new.__globals__[name] is (replacement if name == key else value)
    assert candidate.AtomicLeafFreezeFootprintScope.footprint.__globals__['_clone_cached_receipt'] is fused._clone_cached_receipt
    assert candidate.AtomicLeafFreezeFootprintScope.__init__ is ReceiptCopiedFootprintScope.__init__


def test_constructor_retains_every_original_selector_field_and_persistent_state():
    old,new = controllers()
    assert type(new.selector) is candidate.AtomicLeafFreezeFootprintSelector
    assert type(old.mapper) is type(new.mapper) and type(old.registration) is type(new.registration)
    assert type(old.memory) is type(new.memory) and new.memory is new.mapper.surface
    assert new.selector.residual is new.residual
    equal({k:v for k,v in vars(old.selector).items() if k != 'residual'},
          {k:v for k,v in vars(new.selector).items() if k != 'residual'})
    assert fingerprint(state(old)) == fingerprint(state(new))
    for method in ('observe','advance'): assert getattr(type(old),method) is getattr(type(new),method)


_function = original.test_actual_observation_and_scoped_batched_robot_footprints_match
test_actual_public_packets_registration_map_and_scoped_robot_footprints = FunctionType(_function.__code__,
    _function.__globals__ | dict(controllers=controllers, state=state,
        normalize_to_scoped=candidate.normalize_to_tiled, BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches),
    'test_actual_public_packets_registration_map_and_scoped_robot_footprints', _function.__defaults__)


def test_original_sensor_failure_and_complete_evidence_are_preserved():
    old,new = controllers()
    expected = old.observe({}, {}, {}, now_ns=1)
    actual = new.observe({}, {}, {}, now_ns=1)
    assert candidate.normalize_to_tiled(actual) == expected
    assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    actual['new_selection'] = {'nested_evidence':'unchanged'}
    assert candidate.normalize_to_tiled(actual)['new_selection'] is actual['new_selection']


@pytest.mark.parametrize('field', ['controller', candidate.FLAG])
def test_normalization_requires_both_new_identities(field):
    _,new = controllers(); actual = new.observe({}, {}, {}, now_ns=1)
    actual.pop(field)
    with pytest.raises(ValueError, match='explicit'): candidate.normalize_to_tiled(actual)
