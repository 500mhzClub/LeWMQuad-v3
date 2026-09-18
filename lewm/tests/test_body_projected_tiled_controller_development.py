"""Real public packets, robot footprints, aliases and observation cleanup."""
from types import FunctionType
import numpy as np
import pytest
from lewm import body_projected_tiled_controller_development as candidate
from lewm.body_projected_floor_geometry_development import BodyProjectedFloorGeometry
from lewm.tiled_density_progressive_floor_controller_development import (
    TiledDensityProgressiveFloorController, TiledDensityRecordingFloorGeometry)
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMap, RecordingFloorGeometry
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from lewm.tests import test_scoped_batched_footprint_controller_development as original
from lewm.tests.test_tiled_density_progressive_floor_controller_development import state
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None, geometry, **kwargs()) for kind in
        (TiledDensityProgressiveFloorController, candidate.BodyProjectedTiledController)]


def test_original_map_body_and_recording_order_retained():
    old, new = LaterResolvedFloorMap.observe, candidate.BodyProjectedTiledFloorMap.observe
    assert old.__code__ is new.__code__
    for key in old.__globals__:
        if key != 'RecordingFloorGeometry': assert old.__globals__[key] is new.__globals__[key]
    assert old.__globals__['RecordingFloorGeometry'] is RecordingFloorGeometry
    assert new.__globals__['RecordingFloorGeometry'] is candidate.BodyProjectedRecordingFloorGeometry
    kind = candidate.BodyProjectedRecordingFloorGeometry
    assert kind.index is TiledDensityRecordingFloorGeometry.index
    assert kind.floor_coverage is RecordingFloorGeometry.floor_coverage
    assert kind.primary_floor_plane is BodyProjectedFloorGeometry.primary_floor_plane
    assert kind.close is BodyProjectedFloorGeometry.close
    assert kind.__mro__.index(RecordingFloorGeometry) < kind.__mro__.index(BodyProjectedFloorGeometry)


def test_install_retains_every_existing_field_object_and_memory_alias():
    controller, _ = controllers()
    old_map = controller.mapper
    fields = vars(old_map).copy()
    other = {k:v for k,v in vars(controller).items() if k != 'mapper'}
    candidate.install_fresh_body_projected_map(controller)
    assert type(controller.mapper) is candidate.BodyProjectedTiledFloorMap
    assert vars(controller.mapper).keys() == fields.keys()
    for key, value in fields.items(): assert vars(controller.mapper)[key] is value
    for key, value in other.items(): assert vars(controller)[key] is value
    assert controller.memory is controller.mapper.surface
    baseline, revised = controllers()
    assert fingerprint(state(baseline)) == fingerprint(state(revised))
    for name in ('observe', 'advance'):
        assert getattr(type(baseline),name) is getattr(type(revised),name)


@pytest.mark.parametrize('fault', ['route','map_failed','map_scope','memory_scope','alias','already_installed'])
def test_live_or_incompatible_map_cannot_be_replaced(fault):
    controller, _ = controllers()
    if fault == 'route': controller.memory.route.append({})
    if fault == 'map_failed': controller.mapper.failed = True
    if fault == 'map_scope': controller.mapper.frame_geometry = object()
    if fault == 'memory_scope': controller.memory.frame_geometry = object()
    if fault == 'alias': controller.memory = object()
    if fault == 'already_installed': candidate.install_fresh_body_projected_map(controller)
    before = controller.mapper
    with pytest.raises(ValueError): candidate.install_fresh_body_projected_map(controller)
    assert controller.mapper is before


def test_real_public_packets_and_robot_footprints_match_and_release_projection(monkeypatch):
    closed = []
    original_close = candidate.BodyProjectedRecordingFloorGeometry.close
    def close(context):
        counts = context.body_projection_cache.counts().copy()
        original_close(context)
        closed.append((context, counts))
    monkeypatch.setattr(candidate.BodyProjectedRecordingFloorGeometry, 'close', close)
    function = original.test_actual_observation_and_scoped_batched_robot_footprints_match
    check = FunctionType(function.__code__, function.__globals__ | dict(
        controllers=controllers, state=state, normalize_to_scoped=candidate.normalize_to_tiled,
        BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches),
        'actual_public_packet_comparison', function.__defaults__)
    check(monkeypatch)
    assert len(closed) == 1
    context, counts = closed[0]
    assert 1 <= counts['misses'] <= 2 and counts['hits'] >= 1 and counts['uncached'] == 0
    assert len(context.observations) == 2
    assert context.closed and context.body_projection_cache.closed
    assert context._entries == {} and context.body_projection_cache._entries == {}


def test_late_map_exception_closes_both_caches_and_releases_memory_context(monkeypatch):
    _, controller = controllers()
    mapper = controller.mapper
    contexts = []
    def fail(*args, **kwargs):
        context = mapper.frame_geometry
        assert context is mapper.surface.frame_geometry
        depth = np.zeros((480,640)); valid = np.zeros_like(depth,dtype=bool)
        context.index(depth,valid,np.array([0.,0.,1.]))
        context.body_projection(depth)
        contexts.append(context)
        raise RuntimeError('injected late mapping failure after projection')
    monkeypatch.setattr(mapper,'_observe_both',fail)
    with pytest.raises(RuntimeError,match='injected late mapping failure'):
        mapper.observe({}, {}, {}, auxiliary_depth={}, now_ns=1_500_000_000)
    assert mapper.failed and mapper.surface.failed
    assert mapper.frame_geometry is None and mapper.surface.frame_geometry is None
    assert len(contexts) == 1
    context = contexts[0]
    assert context.closed and context.body_projection_cache.closed
    assert context._entries == {} and context.body_projection_cache._entries == {}


def test_original_sensor_failure_and_nested_evidence_preserved():
    old, new = controllers()
    expected = old.observe({}, {}, {}, now_ns=1)
    actual = new.observe({}, {}, {}, now_ns=1)
    assert candidate.normalize_to_tiled(actual) == expected
    assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    with pytest.raises(ValueError): candidate.normalize_to_tiled(expected)
    actual['new_selection'] = {'changed_evidence': True}
    assert candidate.normalize_to_tiled(actual)['new_selection'] == {'changed_evidence': True}
