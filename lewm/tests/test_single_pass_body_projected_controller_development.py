"""Composition checks exercise actual public packets and unchanged evidence."""
from types import FunctionType

import numpy as np
import pytest

from lewm import single_pass_body_projected_controller_development as candidate
from lewm.body_projected_tiled_controller_development import BodyProjectedTiledController
from lewm.packed_fused_scoped_controller_development import index_owners
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
from lewm.tests import test_scoped_batched_footprint_controller_development as packets_test
from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
from scripts.single_pass_body_projected_state_development import normalized_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def controllers(geometry=None):
    return [kind(None, geometry, **kwargs()) for kind in
        (BodyProjectedTiledController, candidate.SinglePassBodyProjectedController)]


def state(controller):
    return normalized_state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
        occupied=controller.mapper.occupied, residual=controller.residual, history=controller.history))


def indices(controller):
    return [getattr(owner, name) for owner, name in index_owners(controller.memory)]


def test_composition_preserves_all_other_objects_and_methods():
    controller, _ = controllers()
    other = vars(controller).copy()
    memory_fields = vars(controller.memory).copy()
    old_indices = indices(controller)
    candidate.install_empty_single_pass_indices(controller)
    for name, value in other.items():
        assert vars(controller)[name] is value
    for name, value in memory_fields.items():
        if name not in ('index', 'auxiliary_index'):
            assert vars(controller.memory)[name] is value
    assert all(type(i) is SinglePassMeasuredSampleBoundsIndex for i in indices(controller))
    assert all(not any(vars(i).values()) for i in old_indices)
    assert len({id(i) for i in indices(controller)}) == 8
    assert SinglePassMeasuredSampleBoundsIndex.insert is PackedOwnedMeasuredSampleBoundsIndex.insert
    baseline, revised = controllers()
    for name in ('observe', 'advance'):
        assert getattr(type(baseline), name) is getattr(type(revised), name)
    assert type(baseline.mapper) is type(revised.mapper)
    assert type(baseline.selector) is type(revised.selector)
    assert type(baseline.registration) is type(revised.registration)
    assert fingerprint(state(baseline)) == fingerprint(state(revised))


@pytest.mark.parametrize('fault', ['cells', 'bounds', 'sample_counts', 'latest_frames',
    'extra_field', 'index_alias', 'route', 'last_ns', 'failed', 'map_failed',
    'memory_scope', 'map_scope', 'memory_alias', 'already_installed'])
def test_invalid_installation_never_partially_replaces_indices(fault):
    controller, _ = controllers()
    if fault in ('cells', 'bounds', 'sample_counts', 'latest_frames'):
        getattr(indices(controller)[-1], fault)[(0, 0, 0)] = 1
    elif fault == 'extra_field': indices(controller)[-1].unexpected = True
    elif fault == 'index_alias': controller.memory.auxiliary_index = controller.memory.index
    elif fault == 'route': controller.memory.route.append({})
    elif fault == 'last_ns': controller.memory.last_ns = 1
    elif fault == 'failed': controller.memory.failed = True
    elif fault == 'map_failed': controller.mapper.failed = True
    elif fault == 'memory_scope': controller.memory.frame_geometry = object()
    elif fault == 'map_scope': controller.mapper.frame_geometry = object()
    elif fault == 'memory_alias': controller.mapper.surface = object()
    elif fault == 'already_installed': candidate.install_empty_single_pass_indices(controller)
    before = indices(controller)
    with pytest.raises(ValueError): candidate.install_empty_single_pass_indices(controller)
    assert all(a is b for a, b in zip(before, indices(controller), strict=True))


def test_actual_public_observation_and_robot_footprints(monkeypatch):
    original = packets_test.test_actual_observation_and_scoped_batched_robot_footprints_match
    check = FunctionType(original.__code__, original.__globals__ | dict(
        controllers=controllers, state=state,
        normalize_to_scoped=candidate.normalize_to_body_projected,
        BatchedRetainedFloorPatches=ProgressiveBatchedRetainedFloorPatches),
        'composed_packet_check', original.__defaults__)
    check(monkeypatch)


def test_accumulated_evidence_and_every_index_query_remain_equal():
    old, new = controllers()
    rng = np.random.default_rng(2026091131)
    for frame in range(3):
        cloud = rng.uniform(-.1, .1, (100, 3))
        for first, second in zip(indices(old), indices(new), strict=True):
            for index in (first, second): index.insert(cloud, {'frame': frame, 'proof': [frame]})
            assert first.intersect([-.04]*3, [.04]*3) == second.intersect([-.04]*3, [.04]*3)
            assert first.intersect_sphere([0.]*3, .04) == second.intersect_sphere([0.]*3, .04)
        assert fingerprint(state(old)) == fingerprint(state(new))
    indices(new)[-1].sample_counts[next(iter(indices(new)[-1].sample_counts))] += 1
    assert fingerprint(state(old)) != fingerprint(state(new))


def test_state_comparison_rejects_mixed_or_aliased_indices():
    _, controller = controllers()
    controller.memory.index = PackedOwnedMeasuredSampleBoundsIndex()
    with pytest.raises(ValueError): state(controller)
    _, controller = controllers()
    controller.memory.auxiliary_index = controller.memory.index
    with pytest.raises(ValueError): state(controller)


def test_original_failure_and_nested_receipts_are_preserved():
    old, new = controllers()
    expected = old.observe({}, {}, {}, now_ns=1)
    actual = new.observe({}, {}, {}, now_ns=1)
    assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert candidate.normalize_to_body_projected(actual) == expected
    with pytest.raises(ValueError): candidate.normalize_to_body_projected(expected)
    actual['new_selection'] = {'changed': True}
    assert candidate.normalize_to_body_projected(actual)['new_selection'] == {'changed': True}
