"""Check exact mapping, cache lifetime, registration and robot-footprint behavior."""
from types import FunctionType
import numpy as np
import pytest
from lewm import tiled_density_progressive_floor_controller_development as routed
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMap, RecordingFloorGeometry
from lewm.density_routed_floor_controller_development import DensityRoutedFloorMap
from lewm.progressive_batched_floor_controller_development import ProgressiveBatchedFloorController
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from scripts.progressive_batched_floor_state_development import normalized_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests import test_scoped_batched_footprint_controller_development as original


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None,geometry,**kwargs()) for kind in (ProgressiveBatchedFloorController,routed.TiledDensityProgressiveFloorController)]


def state(controller):
    return dict(retained=normalized_state_tree(dict(memory=controller.memory,floor=controller.mapper.floor,
        occupied=controller.mapper.occupied,residual=controller.residual,history=controller.history)),
        registration=vars(controller.registration),cache_counts=controller.mapper.last_cache_counts)


def test_only_private_floor_kernel_bindings_change():
    pairs=((FrameFloorIndexCache.index,routed.TiledDensityRecordingFloorGeometry.index,'observed_floor_cell_index'),
           (LaterResolvedFloorMap.observe,routed.TiledDensityFloorMap.observe,'RecordingFloorGeometry'))
    for old,new,changed in pairs:
        assert old.__code__ is new.__code__
        assert old.__globals__[changed] is not new.__globals__[changed]
        for key in old.__globals__:
            if key!=changed:assert old.__globals__[key] is new.__globals__[key]
    assert LaterResolvedFloorMap.observe.__globals__['RecordingFloorGeometry'] is RecordingFloorGeometry


def test_constructor_retains_all_existing_packed_memory_and_patch_fields():
    old,new=controllers()
    assert type(old.mapper) is DensityRoutedFloorMap
    assert type(new.mapper) is routed.TiledDensityFloorMap
    assert type(old.memory) is type(new.memory)
    assert new.memory is new.mapper.surface and new.selector.residual is new.residual
    assert type(old.selector) is type(new.selector)
    assert fingerprint(state(old))==fingerprint(state(new))
    for method in ('observe','advance'):assert getattr(type(old),method) is getattr(type(new),method)


@pytest.mark.parametrize('fault',['route','map_failed','map_scope','memory_scope','registration','alias'])
def test_live_or_incompatible_consumer_cannot_be_replaced(fault):
    old,_=controllers()
    if fault=='route':old.memory.route.append({})
    if fault=='map_failed':old.mapper.failed=True
    if fault=='map_scope':old.mapper.frame_geometry=object()
    if fault=='memory_scope':old.memory.frame_geometry=object()
    if fault=='registration':old.registration.frame=0
    if fault=='alias':old.memory=object()
    before=(old.mapper,old.registration)
    with pytest.raises(ValueError):routed.install_fresh_tiled_consumers(old)
    assert old.mapper is before[0] and old.registration is before[1]


def test_cache_keeps_exact_key_readonly_bytes_and_closed_lifetime():
    depth=np.zeros((480,640));valid=np.zeros_like(depth,dtype=bool);up=np.array([0.,0.,1.])
    context=routed.TiledDensityRecordingFloorGeometry({}, {},0,1_500_000_000)
    first=context.index(depth,valid,up)
    assert context.index(depth.copy(),valid.copy(),up.copy()) is first
    assert context.counts()==dict(hits=1,misses=1,uncached=0)
    for value in first.values():
        with pytest.raises(ValueError):value.flags.writeable=True
    context.close();assert context._entries=={} and context.closed is True
    with pytest.raises(ValueError,match='closed'):context.index(depth,valid,up)


_function=original.test_actual_observation_and_scoped_batched_robot_footprints_match
test_actual_public_packets_registration_map_cache_and_robot_footprints=FunctionType(_function.__code__,
    _function.__globals__|dict(controllers=controllers,state=state,
        normalize_to_scoped=routed.normalize_to_progressive,
        BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches),
    'test_actual_public_packets_registration_map_cache_and_robot_footprints',_function.__defaults__)


def test_original_sensor_failure_and_metadata_retained():
    old,new=controllers()
    expected=old.observe({},{},{},now_ns=1);actual=new.observe({},{},{},now_ns=1)
    assert routed.normalize_to_progressive(actual)==expected
    assert actual['terminal']=='SENSOR_OR_MODEL_FAILURE'
    with pytest.raises(ValueError):routed.normalize_to_progressive(expected)
