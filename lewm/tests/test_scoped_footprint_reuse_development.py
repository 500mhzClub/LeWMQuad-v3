"""Exact query keys, owned receipts, original recovery and real robot geometry."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.frozen_footprint_receipts_development import detach_receipts
from lewm.scoped_footprint_reuse_development import ScopedFootprintReuse, ScopedFootprintMap
from lewm.scoped_footprint_anchored_controller_development import (
    ScopedFootprintAnchoredController, ScopedFootprintAnchoredSelector, FLAG)
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)
from lewm.residual_anchored_continuation_development import reconsider_anchored_continuation
from lewm.tests.test_residual_anchored_continuation_development import fixture
from lewm.tests.test_residual_first_interval_feasibility_development import NOW


class Memory:
    def __init__(self):
        self.calls = 0
        self.failed = False
        self.last_ns = NOW

    def _current(self, now_ns):
        if self.failed or now_ns != self.last_ns:
            raise ValueError('original current-memory failure')

    def footprint(self, geometry, xy, yaw, *, now_ns, persistent=True):
        self._current(now_ns)
        self.calls += 1
        if not np.isfinite(xy).all():
            raise ValueError('original bad displacement')
        shared = {'witness': [1, 2]}
        return dict(possible_intersection=False, shapes=[shared, shared])


def test_query_reuse_preserves_internal_aliases_and_separate_public_ownership():
    m = Memory(); geometry = object()
    with ScopedFootprintReuse(m, geometry) as view:
        first = view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
        second = view.footprint(geometry, np.zeros(2), 0., now_ns=NOW)
        assert first == second and first is not second and m.calls == 1
        assert deepcopy(first) is first and deepcopy(second) is second
        result = detach_receipts({'first': first, 'second': second})
        result['first']['shapes'][0]['witness'].append(3)
        assert result['first']['shapes'][1]['witness'] == [1, 2, 3]
        assert result['second']['shapes'][0]['witness'] == [1, 2]
        assert view.footprint(geometry, [0., 0.], 0., now_ns=NOW) == second
        assert view.counts()['hits'] == 2
    assert view.counts()['scope_closed'] and view.counts()['retained_entries'] == 0
    assert view._memory is view._geometry is None
    with pytest.raises(ValueError): view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
    with pytest.raises(ValueError): view.__enter__()


@pytest.mark.parametrize('change', ['signed_x', 'signed_yaw', 'tiny_x', 'tiny_yaw', 'persistent', 'geometry'])
def test_distinct_exact_inputs_are_never_conflated(change):
    m = Memory(); geometry = object()
    with ScopedFootprintReuse(m, geometry) as view:
        view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
        xy, yaw, persistent, g = [0., 0.], 0., True, geometry
        if change == 'signed_x': xy[0] = -0.
        if change == 'signed_yaw': yaw = -0.
        if change == 'tiny_x': xy[0] = 1e-14
        if change == 'tiny_yaw': yaw = 1e-14
        if change == 'persistent': persistent = False
        if change == 'geometry': g = object()
        view.footprint(g, xy, yaw, now_ns=NOW, persistent=persistent)
        assert m.calls == 2 and view.counts()['hits'] == 0


def test_non_python_float_yaw_retains_original_numeric_dispatch():
    m = Memory(); geometry = object()
    with ScopedFootprintReuse(m, geometry) as view:
        for yaw in (float(np.float32(.1)), np.float32(.1), np.float32(.1)):
            view.footprint(geometry, [0., 0.], yaw, now_ns=NOW)
        assert m.calls == 3 and view.counts()['hits'] == 0


def test_capacity_is_bounded_and_overflow_recomputes_without_rejection():
    m = Memory(); geometry = object()
    with ScopedFootprintReuse(m, geometry) as view:
        for i in range(20): view.footprint(geometry, [i*.001, 0.], 0., now_ns=NOW)
        assert view.counts()['retained_entries'] == 18
        view.footprint(geometry, [19*.001, 0.], 0., now_ns=NOW)
        view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
        assert m.calls == 21 and view.counts()['hits'] == 1


@pytest.mark.parametrize('kind', ['custom', 'cycle', 'tuple'])
def test_unsupported_graphs_always_forward(kind):
    class OtherMemory(Memory):
        def footprint(self, *args, **kwargs):
            self.calls += 1
            result = {'x': []}
            if kind == 'custom': result['x'] = object()
            elif kind == 'tuple': result['x'] = (1, 2)
            else: result['x'].append(result)
            return result
    m = OtherMemory(); geometry = object()
    with ScopedFootprintReuse(m, geometry) as view:
        for _ in range(2): view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
        assert m.calls == 2 and view.counts()['retained_entries'] == 0


def test_original_errors_stale_time_and_latched_failure_are_not_hidden_by_hits():
    m = Memory(); geometry = object(); view = ScopedFootprintReuse(m, geometry)
    with pytest.raises(ValueError, match='current-memory'):
        with view:
            for _ in range(2):
                with pytest.raises(ValueError, match='bad displacement'):
                    view.footprint(geometry, [np.nan, 0.], 0., now_ns=NOW)
            assert m.calls == 2 and view.counts()['retained_entries'] == 0
            view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
            with pytest.raises(ValueError, match='current-memory'):
                view.footprint(geometry, [0., 0.], 0., now_ns=NOW+1)
            m.failed = True
            view.footprint(geometry, [0., 0.], 0., now_ns=NOW)
    assert view.counts()['scope_closed'] and view.counts()['retained_entries'] == 0


@pytest.mark.parametrize('expensive,blocked', [(False, False), (True, False), (False, True)])
def test_original_recovery_chain_and_gates_with_exact_repeated_queries(expensive, blocked):
    selection, residual, original = fixture(expensive=expensive)
    original.surface.block = blocked
    expected = reconsider_anchored_continuation(selection, residual, original, object(), now_ns=NOW)
    selection2, residual2, mapper = fixture(expensive=expensive)
    mapper.surface.block = blocked
    mapper.surface.calls.clear()
    geometry = object()
    with ScopedFootprintReuse(mapper.surface, geometry) as memory:
        actual = reconsider_anchored_continuation(selection2, residual2,
            ScopedFootprintMap(mapper, memory), geometry, now_ns=NOW)
        assert detach_receipts(actual) == expected
        assert memory.counts()['hits'] >= 6
        assert len(mapper.surface.calls) < memory.requests
    assert memory.counts()['retained_entries'] == 0


def test_unknown_stateful_surface_retains_uncached_selector_calls(monkeypatch):
    class Stateful(Memory):
        def footprint(self, *args, **kwargs):
            self.calls += 1
            return {'possible_intersection': self.calls > 1}
    memory = Stateful(); mapper = SimpleNamespace(surface=memory)
    def original(self, model, history, m, geometry, *, now_ns):
        return {'checks': [m.surface.footprint(geometry, [0., 0.], 0., now_ns=now_ns) for _ in range(2)]}
    monkeypatch.setattr(ResidualAnchoredContinuationSelector, 'choose', original)
    selector = ScopedFootprintAnchoredSelector(residual=object(), condition='jepa', variant='full',
                                              goal_initial_body_xy_m=[1., 0.])
    result = selector.choose(None, None, mapper, object(), now_ns=NOW)
    assert result == {'checks': [{'possible_intersection': False}, {'possible_intersection': True}]}
    assert memory.calls == 2


def test_actual_observed_memory_and_robot_footprints_remain_exact(monkeypatch):
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.tests import test_joint_pulse_execution_development as pulse
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
    from lewm.tests.test_frame_floor_cache_development import equal
    from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
    from lewm.fast_gyro_development import FastGyroBuffer
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    monkeypatch.setattr(pulse, 'visual', partial(visual, origin=1_500_000_000))
    geometry = ArticulatedCollisionGeometry(URDF)
    controller = ResidualAnchoredContinuationController(None, geometry, **kwargs())
    p, d, auxiliary, _, now = packets()
    image = from_captured_rgb(p['image']['rgb'], auxiliary, p, measured_ns=now, available_ns=now, now_ns=now)
    gyro = FastGyroBuffer((0, 0, 0))
    for t in range(now-100_000_000, now+1, 2_000_000):
        gyro.append(np.zeros(3), np.ones(3, bool), measured_ns=t, available_ns=t)
    result = controller.observe(p, d, gyro.packet(now_ns=now), auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
    assert result['terminal'] is None
    snapshot = deepcopy(vars(controller.memory))
    with ScopedFootprintReuse(controller.memory, geometry) as view:
        for x, y, yaw in [(0., 0., 0.), (.02, 0., 0.), (.016, .002, .045), (.016, -.002, -.045)]:
            expected = controller.memory.footprint(geometry, [x, y], yaw, now_ns=now)
            first = view.footprint(geometry, [x, y], yaw, now_ns=now)
            second = view.footprint(geometry, [x, y], yaw, now_ns=now)
            equal(expected, detach_receipts(first)); equal(expected, detach_receipts(second))
        assert view.counts()['computations'] == view.counts()['hits'] == 4
    # Compare nested state through the existing recursive state normalizer.
    from scripts.replay_go2_receipt_copied_anchored_prefix_v1 import state_tree
    equal(state_tree(snapshot), state_tree(vars(controller.memory)))
    # Exercise the supported selector branch with this real observed memory;
    # only the enclosing selector is substituted, not the contact computation.
    scopes = []
    def query_pair(self, model, history, mapper, g, *, now_ns):
        scopes.append(mapper.surface)
        assert isinstance(mapper.surface, ScopedFootprintReuse)
        return {'surface_checks': [mapper.surface.footprint(g, [.02, 0.], 0., now_ns=now_ns)
                                   for _ in range(2)]}
    monkeypatch.setattr(ResidualAnchoredContinuationSelector, 'choose', query_pair)
    selector = ScopedFootprintAnchoredSelector(residual=controller.residual, condition='jepa',
        variant='full', goal_initial_body_xy_m=[1., 0.])
    first = selector.choose(None, None, controller.mapper, geometry, now_ns=now)
    second = selector.choose(None, None, controller.mapper, geometry, now_ns=now)
    assert first == second and scopes[0] is not scopes[1]
    for scope in scopes:
        assert scope.counts() == dict(requests=2, computations=1, hits=1,
                                     retained_entries=0, scope_closed=True)
    assert first['surface_checks'][0] is not first['surface_checks'][1]
    assert not any(isinstance(v, ScopedFootprintReuse) for v in vars(selector).values())
    equal(state_tree(snapshot), state_tree(vars(controller.memory)))


def test_controller_keeps_original_observation_state_and_failure_stop():
    for name in ('observe', 'advance'):
        assert getattr(ScopedFootprintAnchoredController, name) is getattr(ResidualAnchoredContinuationController, name)
    c = ScopedFootprintAnchoredController(None, None,
        public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
                            require_return_after_goal=True), navigation_ticks=40,
        condition='jepa', variant='full', persistent=True)
    assert c.memory is c.mapper.surface and c.selector.residual is c.residual
    r = c.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    assert r[FLAG] is True
