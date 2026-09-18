"""Exact support receipts, ownership, invalidation and actual contact evidence."""
from copy import deepcopy
from functools import partial
import numpy as np
import pytest
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.scoped_support_query_cache_development import SupportQueryCache
from lewm.support_cached_single_pass_controller_development import SupportCachedSinglePassController
from lewm.single_pass_receipt_copied_controller_development import SinglePassReceiptCopiedController
from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
from lewm.tests.test_frame_floor_cache_development import equal
from scripts.analyze_go2_ground_plane_development_v1 import URDF


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_exact_robot_support_and_mutable_return_ownership(seed):
    geometry = ArticulatedCollisionGeometry(URDF); rng = np.random.default_rng(seed)
    q = rng.uniform(-.5, .5, 12); directions = rng.normal(size=(5, 3))
    expected = geometry.supports(q, directions); inputs = deepcopy((q, directions))
    cache = SupportQueryCache(geometry)
    with cache:
        a = cache.supports(q, directions); b = cache.supports(q.copy(), directions.copy())
        assert a == b == expected and a is not b and a['shapes'] is not b['shapes']
        a['shapes'][0]['lower'][0] = 999
        assert cache.supports(q, directions) == b == expected
        assert cache.counts()['computations'] == 1 and cache.counts()['hits'] == 2
        assert cache._shapes is geometry._shapes
    equal((q, directions), inputs)
    assert cache.counts()['retained_entries'] == 0 and cache.counts()['scope_closed']
    with pytest.raises(ValueError): cache.supports(q, directions)
    with pytest.raises(ValueError): cache.__enter__()


def test_exact_key_changes_and_bounded_capacity_do_not_change_results():
    geometry = ArticulatedCollisionGeometry(URDF); q = np.zeros(12); R = np.eye(3)
    with SupportQueryCache(geometry) as cache:
        for i in range(12):
            q[0] = i*1e-8
            assert cache.supports(q, R) == geometry.supports(q, R)
        assert cache.counts()['retained_entries'] == 8 and cache.counts()['computations'] == 12
        q[0] = 0.; cache.supports(q, R)
        assert cache.counts()['hits'] == 1
        R[0, 0] += 1e-8
        assert cache.supports(q, R) == geometry.supports(q, R)
        assert cache.counts()['computations'] == 13


@pytest.mark.parametrize('q, normals', [(np.zeros(11), np.eye(3)), (np.full(12, np.nan), np.eye(3)),
    (np.zeros(12), np.zeros((3, 3))), (np.zeros(12), np.ones((3, 2)))])
def test_original_errors_are_preserved_and_never_cached(q, normals):
    geometry = ArticulatedCollisionGeometry(URDF)
    with pytest.raises(Exception) as expected: geometry.supports(q, normals)
    cache = SupportQueryCache(geometry)
    with cache:
        for _ in range(2):
            with pytest.raises(type(expected.value)) as actual: cache.supports(q, normals)
            assert str(actual.value) == str(expected.value)
        assert cache.counts()['computations'] == 2 and cache.counts()['retained_entries'] == 0
    assert cache.counts()['scope_closed']


def test_scope_closes_after_original_exception():
    class Geometry:
        def supports(self, *args): raise RuntimeError('original support failure')
    cache = SupportQueryCache(Geometry())
    with pytest.raises(RuntimeError, match='original support failure'):
        with cache: cache.supports(np.zeros(12), np.eye(3))
    assert cache.counts() == dict(requests=1, computations=1, hits=0, retained_entries=0, scope_closed=True)


def test_actual_dual_camera_contact_receipts_and_later_observation_stops_are_exact(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
    from lewm.fast_gyro_development import FastGyroBuffer
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    geometry = ArticulatedCollisionGeometry(URDF)
    old, new = [cls(None, geometry, **kwargs()) for cls in (SinglePassReceiptCopiedController, SupportCachedSinglePassController)]
    p, d, a, _, now = packets()
    image = from_captured_rgb(p['image']['rgb'], a, p, measured_ns=now, available_ns=now, now_ns=now)
    gyro = FastGyroBuffer((0, 0, 0))
    for t in range(now-100_000_000, now+1, 2_000_000):
        gyro.append(np.zeros(3), np.ones(3, bool), measured_ns=t, available_ns=t)
    f = gyro.packet(now_ns=now); rows = []
    for controller in (old, new):
        rows.append(controller.observe(*deepcopy((p, d, f)), auxiliary_rgb=deepcopy(image), auxiliary_depth=deepcopy(a), now_ns=now))
    equal(*rows); assert rows[0]['terminal'] is None
    for dx, dy, yaw in [(0., 0., 0.), (.02, 0., 0.), (.016, .002, .045), (.016, -.002, -.045), (0., 0., .045), (0., 0., -.045)]:
        arow = old.memory.footprint(geometry, [dx, dy], yaw, now_ns=now)
        brow = new.memory.footprint(geometry, [dx, dy], yaw, now_ns=now)
        equal(arow, brow)
        counts = new.memory.last_support_query_counts
        assert counts['requests'] >= 5 and counts['computations'] == 1 and counts['scope_closed']
        assert counts['retained_entries'] == 0
    rows = [c.observe(p, d, f, auxiliary_rgb=None, auxiliary_depth=a, now_ns=now+100_000_000) for c in (old, new)]
    equal(*rows); assert rows[0]['terminal'] == 'SENSOR_OR_MODEL_FAILURE'


def test_controller_science_methods_and_shared_residual_are_unchanged():
    for method in ('observe', 'advance', '_result'):
        assert getattr(SupportCachedSinglePassController, method) is getattr(SinglePassReceiptCopiedController, method)
    old, new = [cls(None, None, **kwargs()) for cls in (SinglePassReceiptCopiedController, SupportCachedSinglePassController)]
    assert new.memory is new.mapper.surface and new.residual is new.selector.residual
    for name in ('motion', 'registration', 'mission', 'residual', 'selector'):
        assert type(getattr(new, name)) is type(getattr(old, name))
