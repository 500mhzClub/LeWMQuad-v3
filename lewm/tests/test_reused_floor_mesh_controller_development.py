from collections import deque
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.frozen_footprint_anchored_controller_development import (
    FrozenFootprintAnchoredController, CONTROLLER as ORIGINAL_CONTROLLER)
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap
from lewm.reused_floor_mesh_controller_development import (
    ReusedFloorMeshController, ReusedMeshFloorMap, ReusedMeshRecordingFloorGeometry, FLAG)
from lewm.tests.test_reused_floor_mesh_development import equal
from lewm.tests.test_current_primary_floor_plane_development import depth_plane
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_measured_floor_transport_development import item


def same_state(a, b, path=()):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same_state(a[k], b[k], path+(k,))
    elif isinstance(a, (list, tuple, deque)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same_state(x, y, path+(i,))
    elif hasattr(a, '__dict__'):
        if path == ():
            assert type(a) is FrozenFootprintAnchoredController and type(b) is ReusedFloorMeshController
        elif path == ('mapper',):
            assert type(a) is MeasuredFloorTransportMap and type(b) is ReusedMeshFloorMap
        else:
            assert type(a) is type(b), path
        same_state(vars(a), vars(b), path)
    else:
        assert type(a) is type(b) and a == b, path


def test_recording_geometry_uses_new_cache_and_keeps_original_lifetime():
    g = ReusedMeshRecordingFloorGeometry({}, {}, 0, 1_500_000_000)
    d, v = depth_plane()
    one = g.index(d, v, [0., 0., 1.])
    assert g.index(d.copy(), v.copy(), [0., 0., 1.]) is one
    g.index(d, v, [1e-12, 0., 1.])
    assert len(g._meshes) == 1 and len(g._entries) == 2
    assert g.counts() == dict(hits=1, misses=2, uncached=0)
    g.close()
    assert g.closed and not g._meshes and not g._entries


@pytest.mark.parametrize('narrow', [False, True])
def test_complete_synthetic_warmup_decisions_and_retained_state_match(narrow, monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    options = dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],
        return_initial_body_xy_m=[0.,0.], require_return_after_goal=True),
        navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    old = FrozenFootprintAnchoredController(None, None, **options)
    new = ReusedFloorMeshController(None, None, **options)
    same_state(old, new)
    previous = None
    for frame in range(3):
        p, d, a, raw, now, image = item(frame, previous, narrow=narrow and frame>0)
        before = [(x['depth_m'].tobytes(), x['valid'].tobytes()) for x in (d,a)]
        for c in (old,new):
            c.motion = SimpleNamespace(observe=lambda *args, **kwargs: deepcopy(raw))
        expected = old.observe(p,d,None,auxiliary_depth=a,auxiliary_rgb=image,now_ns=now)
        actual = new.observe(p,d,None,auxiliary_depth=a,auxiliary_rgb=image,now_ns=now)
        assert expected['terminal'] is None, expected['failure']
        assert actual.pop(FLAG) is True
        actual['controller'] = ORIGINAL_CONTROLLER
        same_state(expected, actual)
        same_state(old, new)
        assert before == [(x['depth_m'].tobytes(),x['valid'].tobytes()) for x in (d,a)]
        assert new.memory is new.mapper.surface and new.selector.residual is new.residual
        assert new.mapper.frame_geometry is None and new.memory.frame_geometry is None
        assert new.mapper.last_cache_counts == old.mapper.last_cache_counts
        assert new.mapper.last_cache_counts['misses'] > 0
        previous = raw


def test_exception_closes_both_caches_and_preserves_original_failure_latches(monkeypatch):
    m = ReusedMeshFloorMap(); contexts=[]; d,v=depth_plane()
    def fail(*args, **kwargs):
        context=m.frame_geometry;contexts.append(context)
        context.index(d,v,[0.,0.,1.])
        assert len(context._meshes)==1
        raise SensorContractError('injected observed map failure')
    monkeypatch.setattr(m,'_observe_both',fail)
    with pytest.raises(SensorContractError,match='injected'):
        m.observe({}, {}, {}, auxiliary_depth={}, now_ns=1_500_000_000)
    assert m.failed and m.surface.failed
    assert m.frame_geometry is None and m.surface.frame_geometry is None
    assert m.last_cache_counts == dict(hits=0,misses=1,uncached=0)
    assert contexts[0].closed and not contexts[0]._meshes and not contexts[0]._entries


def test_nested_observation_is_rejected_before_touching_active_context():
    m=ReusedMeshFloorMap(); active=object();m.frame_geometry=active
    with pytest.raises(SensorContractError,match='nested'):
        m.observe({}, {}, {}, auxiliary_depth={}, now_ns=1_500_000_000)
    assert m.frame_geometry is active and not m.failed and not m.surface.failed
