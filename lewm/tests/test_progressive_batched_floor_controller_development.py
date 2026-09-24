"""Keep original model/map/selector behavior and all retained state bytes."""
from types import FunctionType
import numpy as np
import pytest
from lewm.density_routed_floor_controller_development import DensityRoutedFloorController
from lewm.progressive_batched_floor_controller_development import (
    ProgressiveBatchedFloorController, install_fresh_progressive_patches, normalize_to_density_routed)
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from lewm.visibility_batched_retained_floor_patch_development import VisibilityBatchedRetainedFloorPatches
from scripts.progressive_batched_floor_state_development import normalized_state_tree,STATE_TYPE_PATHS
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests import test_scoped_batched_footprint_controller_development as original


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None,geometry,**kwargs()) for kind in (DensityRoutedFloorController,ProgressiveBatchedFloorController)]


def state(controller):
    return normalized_state_tree(dict(memory=controller.memory,floor=controller.mapper.floor,
        occupied=controller.mapper.occupied,residual=controller.residual,history=controller.history))


def test_only_two_empty_patch_stores_change_and_memory_aliases_remain():
    old,new=controllers()
    assert type(old.memory) is type(new.memory) and type(old.mapper) is type(new.mapper)
    assert type(old.selector) is type(new.selector)
    assert new.memory is new.mapper.surface and new.selector.residual is new.residual
    for field in ('patches','auxiliary_patches'):
        assert type(getattr(old.memory,field)) is VisibilityBatchedRetainedFloorPatches
        assert type(getattr(new.memory,field)) is ProgressiveBatchedRetainedFloorPatches
    for method in ('observe','advance'):assert getattr(type(old),method) is getattr(type(new),method)
    assert len(STATE_TYPE_PATHS)==10 and fingerprint(state(old))==fingerprint(state(new))


@pytest.mark.parametrize('fault',['nonempty','extra','alias','frames_alias','custom','mixed'])
def test_incompatible_storage_is_rejected_before_any_replacement(fault):
    old,_=controllers();m=old.memory
    if fault=='nonempty':m.auxiliary_patches.frames.append({'x':1})
    if fault=='extra':m.auxiliary_patches.extra=None
    if fault=='alias':m.auxiliary_patches=m.patches
    if fault=='frames_alias':m.auxiliary_patches.frames=m.patches.frames
    if fault=='custom':m.auxiliary_patches=type('Custom',(VisibilityBatchedRetainedFloorPatches,),{})()
    if fault=='mixed':m.auxiliary_patches=ProgressiveBatchedRetainedFloorPatches()
    before=(m.patches,m.auxiliary_patches)
    with pytest.raises(ValueError):install_fresh_progressive_patches(m)
    assert m.patches is before[0] and m.auxiliary_patches is before[1]


def test_state_normalization_cannot_hide_changed_witness_or_pixel_bytes():
    from lewm.tests.test_batched_retained_floor_patch_development import frame
    old,new=controllers()
    for controller in (old,new):
        for name in ('patches','auxiliary_patches'):getattr(controller.memory,name).frames.append(frame(0))
    assert fingerprint(state(old))==fingerprint(state(new))
    new.memory.patches.frames[0]['witness']['evidence'].append('changed')
    assert fingerprint(state(old))!=fingerprint(state(new))
    new.memory.patches.frames[0]['witness']['evidence'].pop()
    pixels=new.memory.patches.frames[0]['prefix'].copy();pixels[1,1]=1;pixels.flags.writeable=False
    new.memory.patches.frames[0]['prefix']=pixels
    assert fingerprint(state(old))!=fingerprint(state(new))
    new.memory.auxiliary_patches=VisibilityBatchedRetainedFloorPatches()
    with pytest.raises(ValueError):state(new)


_function=original.test_actual_observation_and_scoped_batched_robot_footprints_match
test_actual_public_packets_maps_and_robot_footprints=FunctionType(_function.__code__,
    _function.__globals__|dict(controllers=controllers,state=state,normalize_to_scoped=normalize_to_density_routed,
        BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches),
    'test_actual_public_packets_maps_and_robot_footprints',_function.__defaults__)


def test_original_failure_stop_and_metadata_are_retained():
    old,new=controllers()
    expected=old.observe({},{},{},now_ns=1);actual=new.observe({},{},{},now_ns=1)
    assert normalize_to_density_routed(actual)==expected and actual['terminal']=='SENSOR_OR_MODEL_FAILURE'
    with pytest.raises(ValueError):normalize_to_density_routed(expected)
