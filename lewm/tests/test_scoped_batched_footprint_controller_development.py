"""Composition equivalence with real observed geometry, without model inference."""
from copy import deepcopy
from functools import partial
import numpy as np
import pytest

from lewm.scoped_batched_footprint_controller_development import (
    ScopedBatchedFootprintController, normalize_to_scoped, FLAG)
from lewm.scoped_footprint_anchored_controller_development import (
    ScopedFootprintAnchoredController, ScopedFootprintAnchoredSelector)
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationSelector
from lewm.scoped_footprint_reuse_development import ScopedFootprintReuse
from lewm.batched_retained_floor_patch_development import BatchedRetainedFloorPatches
from lewm.retained_floor_patch_development import RetainedFloorPatches
from scripts.replay_go2_batched_patch_anchored_prefix_v1 import normalized_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests.test_batched_retained_floor_patch_development import frame, GOOD, BAD


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None, geometry, **kwargs()) for kind in
        (ScopedFootprintAnchoredController, ScopedBatchedFootprintController)]


def state(controller):
    return normalized_state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
        occupied=controller.mapper.occupied, residual=controller.residual, history=controller.history))


def test_only_two_patch_store_types_differ_from_scoped_baseline():
    old, new = controllers()
    assert new.memory is new.mapper.surface and new.selector.residual is new.residual
    assert type(new.memory) is type(old.memory) and type(new.mapper) is type(old.mapper)
    assert type(new.selector) is type(old.selector) is ScopedFootprintAnchoredSelector
    for name in ('patches', 'auxiliary_patches'):
        assert type(getattr(new.memory, name)) is BatchedRetainedFloorPatches
        assert type(getattr(old.memory, name)) is RetainedFloorPatches
    assert new.memory.patches.frames is not new.memory.auxiliary_patches.frames
    for name in ('observe','advance'):
        assert getattr(type(new), name) is getattr(type(old), name)
    assert BatchedRetainedFloorPatches.append is RetainedFloorPatches.append
    assert fingerprint(state(old)) == fingerprint(state(new))


def test_late_history_retains_earliest_exact_witness_and_all_frames():
    old, new = controllers()
    for name in ('patches','auxiliary_patches'):
        first = getattr(old.memory, name); second = getattr(new.memory, name)
        for i in range(1428):
            first.frames.append(frame(i, GOOD if i == 1418 else BAD))
            second.frames.append(frame(i, GOOD if i == 1418 else BAD))
        result = second.coverage([[1.,0.],[0.,0.]])
        assert result == first.coverage([[1.,0.],[0.,0.]])
        assert result[0]['coverage_witness']['witness']['frame'] == 1418
        assert all(r['retained_frames'] == 1428 for r in result)
        assert result[1]['coverage_witness'] is None


def test_actual_observation_and_scoped_batched_robot_footprints_match(monkeypatch):
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.tests import test_joint_pulse_execution_development as pulse
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.tests.test_frame_floor_cache_development import equal
    from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
    from lewm.fast_gyro_development import FastGyroBuffer
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    monkeypatch.setattr(pulse, 'visual', partial(visual, origin=1_500_000_000))
    geometry = ArticulatedCollisionGeometry(URDF); old, new = controllers(geometry)
    p, d, auxiliary, _, now = packets()
    image = from_captured_rgb(p['image']['rgb'], auxiliary, p, measured_ns=now, available_ns=now, now_ns=now)
    gyro = FastGyroBuffer((0,0,0))
    for t in range(now-100_000_000, now+1, 2_000_000):
        gyro.append(np.zeros(3), np.ones(3,bool), measured_ns=t, available_ns=t)
    args = dict(now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
    expected = old.observe(p,d,gyro.packet(now_ns=now),**args)
    actual = new.observe(p,d,gyro.packet(now_ns=now),**args)
    assert expected['terminal'] is None
    equal(expected, normalize_to_scoped(actual)); equal(state(old), state(new))
    before = fingerprint(state(new)); calls = []; scopes = []
    original_coverage = BatchedRetainedFloorPatches.coverage
    def coverage(self, *a, **k):
        calls.append(id(self)); return original_coverage(self, *a, **k)
    monkeypatch.setattr(BatchedRetainedFloorPatches, 'coverage', coverage)
    def choose(self, model, history, mapper, g, *, now_ns):
        assert isinstance(mapper.surface, ScopedFootprintReuse)
        scopes.append(mapper.surface)
        return dict(surface_checks=[mapper.surface.footprint(g,[.02,0.],0.,now_ns=now_ns) for _ in range(2)])
    monkeypatch.setattr(ResidualAnchoredContinuationSelector, 'choose', choose)
    baseline = old.selector.choose(None,None,old.mapper,geometry,now_ns=now)
    result = new.selector.choose(None,None,new.mapper,geometry,now_ns=now)
    equal(result,baseline)
    assert {id(new.memory.patches),id(new.memory.auxiliary_patches)} <= set(calls)
    for scope in scopes:
        assert scope.counts() == dict(requests=2,computations=1,hits=1,retained_entries=0,scope_closed=True)
    saved_second = deepcopy(result['surface_checks'][1])
    result['surface_checks'][0]['possible_intersection'] = not result['surface_checks'][0]['possible_intersection']
    equal(result['surface_checks'][1],saved_second)
    assert fingerprint(state(new)) == before
    equal(state(old),state(new))


def test_failed_sensor_stop_matches_scoped_controller():
    old,new = controllers()
    expected=old.observe({}, {}, {}, now_ns=1)
    actual=new.observe({}, {}, {}, now_ns=1)
    assert actual['terminal']=='SENSOR_OR_MODEL_FAILURE' and actual['requested_command']==[0.,0.,0.]
    assert normalize_to_scoped(actual)==expected


@pytest.mark.parametrize('field',['controller',FLAG,'batched_retained_floor_queries_enabled',
    'selection_scoped_exact_footprint_reuse_enabled'])
def test_normalization_requires_all_declared_identities(field):
    _,new=controllers();result=new.observe({}, {}, {}, now_ns=1)
    result.pop(field)
    with pytest.raises(ValueError,match='explicit combined'):normalize_to_scoped(result)


def test_normalization_preserves_nested_evidence_changes():
    _,new=controllers();result=new.observe({}, {}, {}, now_ns=1)
    result['new_selection']={'modified_nested_evidence':True}
    assert normalize_to_scoped(result)['new_selection']=={'modified_nested_evidence':True}
