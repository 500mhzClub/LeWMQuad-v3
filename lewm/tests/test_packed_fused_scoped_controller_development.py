"""Preserve real observed geometry and cache behavior across eight replacements."""
from types import FunctionType
import pytest

from lewm.fused_scoped_batched_controller_development import FusedScopedBatchedController, FusedScopedBatchedSelector
from lewm.packed_fused_scoped_controller_development import (
    PackedFusedScopedController, install_empty_packed_indices, index_owners, normalize_to_fused)
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from scripts.packed_fused_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests import test_scoped_batched_footprint_controller_development as original


def controllers(geometry=None):
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    return [kind(None, geometry, **kwargs()) for kind in (FusedScopedBatchedController, PackedFusedScopedController)]


def state(controller):
    return normalized_state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
        occupied=controller.mapper.occupied, residual=controller.residual, history=controller.history))


def test_exact_empty_state_eight_indices_and_unchanged_runtime_methods():
    baseline, candidate = controllers()
    assert len(STATE_TYPE_PATHS) == len(set(STATE_TYPE_PATHS)) == 10
    assert type(candidate.memory) is type(baseline.memory)
    assert type(candidate.mapper) is type(baseline.mapper)
    assert type(candidate.selector) is type(baseline.selector) is FusedScopedBatchedSelector
    assert candidate.memory is candidate.mapper.surface
    assert candidate.selector.residual is candidate.residual
    for controller, kind in ((baseline, MeasuredSampleBoundsIndex), (candidate, PackedOwnedMeasuredSampleBoundsIndex)):
        indices = [getattr(owner, name) for owner, name in index_owners(controller.memory)]
        assert len({id(index) for index in indices}) == 8
        assert all(type(index) is kind and not index.cells for index in indices)
    for name in ('observe', 'advance'):
        assert getattr(type(candidate), name) is getattr(type(baseline), name)
    assert PackedOwnedMeasuredSampleBoundsIndex._intersect is MeasuredSampleBoundsIndex._intersect
    assert fingerprint(state(baseline)) == fingerprint(state(candidate))


@pytest.mark.parametrize('fault', ['nonempty', 'extra', 'alias', 'custom', 'mixed'])
def test_installation_rejects_incompatible_state_before_any_replacement(fault):
    baseline, _ = controllers()
    memory = baseline.memory
    if fault == 'nonempty': memory.confirmed_auxiliary_partition.other.bounds[(0,0,0)] = [1]
    elif fault == 'extra': memory.confirmed_auxiliary_partition.other.extra = None
    elif fault == 'alias': memory.auxiliary_index = memory.index
    elif fault == 'mixed': memory.auxiliary_index = PackedOwnedMeasuredSampleBoundsIndex()
    else: memory.auxiliary_index = type('Custom', (MeasuredSampleBoundsIndex,), {})()
    before = [getattr(owner, name) for owner, name in index_owners(memory)]
    with pytest.raises(ValueError): install_empty_packed_indices(memory)
    after = [getattr(owner, name) for owner, name in index_owners(memory)]
    assert all(a is b for a,b in zip(before, after, strict=True))


def test_state_comparison_rejects_mixed_indices_and_retains_changed_bound_bytes():
    baseline, candidate = controllers()
    for controller in (baseline, candidate):
        for owner, name in index_owners(controller.memory):
            getattr(owner, name).insert([[.1, .2, .3]], {'frame': 0, 'nested': [1]})
    assert fingerprint(state(baseline)) == fingerprint(state(candidate))
    key = next(iter(candidate.memory.index.bounds))
    candidate.memory.index.bounds[key][0,0] -= .0001
    assert fingerprint(state(baseline)) != fingerprint(state(candidate))
    candidate.memory.auxiliary_index = MeasuredSampleBoundsIndex()
    with pytest.raises(ValueError): state(candidate)


# This original integration test observes real public primary/auxiliary
# packets, checks all state, then queries articulated robot footprints through
# the actual scoped cache and both historical patch stores. Only test-class
# choices and the exact declared normalizers change; module globals do not.
_function = original.test_actual_observation_and_scoped_batched_robot_footprints_match
_namespace = dict(_function.__globals__, controllers=controllers, state=state,
                  normalize_to_scoped=normalize_to_fused)
test_actual_observation_and_scoped_packed_footprints = FunctionType(
    _function.__code__, _namespace, 'test_actual_observation_and_scoped_packed_footprints', _function.__defaults__)


def test_original_failure_stop_and_declared_metadata():
    baseline, candidate = controllers()
    expected = baseline.observe({}, {}, {}, now_ns=1)
    actual = candidate.observe({}, {}, {}, now_ns=1)
    assert normalize_to_fused(actual) == expected
    assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    with pytest.raises(ValueError): normalize_to_fused(expected)
