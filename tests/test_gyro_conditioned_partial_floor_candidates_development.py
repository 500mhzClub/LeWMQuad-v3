import numpy as np
import pytest
from lewm import joint_measured_floor_plane_development as planes
from lewm.partial_floor_height_development import fit_gyro_height
from lewm.robust_height_floor_candidates_development import select_clouds as original
from lewm.gyro_conditioned_partial_floor_candidates_development import select_clouds


@pytest.fixture(autouse=True)
def original_navigation_extent(monkeypatch):
    monkeypatch.setattr(planes, 'MINIMUM_SECOND_EXTENT_M', .02)


def assert_unchanged(clouds):
    expected, before = original(clouds, [0., 0., 1.])
    actual, after = select_clouds(clouds, [0., 0., 1.])
    assert before == after
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_fully_observable_floor_is_unchanged():
    rng = np.random.default_rng(2026091561)
    points = np.column_stack((rng.uniform(-.4, .4, (400, 2)),
        rng.uniform(-.3005, -.2995, 400)))
    assert_unchanged([points[:200], points[200:]])


def test_weak_extent_already_consistent_with_gyro_is_unchanged():
    x = np.linspace(-.005, .005, 400)
    points = np.column_stack((x, np.sin(np.arange(400))*.4, -.3+.1*x))
    assert_unchanged([np.empty((0, 3)), points])


def test_weak_extent_removes_only_inliers_that_fail_the_existing_gyro_residual():
    x = np.r_[np.linspace(-.005, .005, 400), np.full(5, .035)]
    points = np.column_stack((x, np.sin(np.arange(len(x)))*.4, -.3+.1*x))
    clouds = [np.empty((0, 3)), points]
    masks, before = original(clouds, [0., 0., 1.])
    original_plane = fit_gyro_height(*(p[m] for p, m in zip(clouds, masks)), [0., 0., 1.])
    assert not original_plane['available']
    assert original_plane['partial_height_maximum_residual_m'] > .003
    revised, after = select_clouds(clouds, [0., 0., 1.])
    plane = fit_gyro_height(*(p[m] for p, m in zip(clouds, revised)), [0., 0., 1.])
    assert plane['available'] and not plane['full_plane_qualification']
    assert plane['partial_height_maximum_residual_m'] <= .003
    assert after['selected_count'] == 400 < before['selected_count']
    assert after['selected_count'] >= max(100, .25*after['raw_pool_count'])
    assert after['only_original_candidates_removed']
    assert all(not np.any(a & ~b) for a, b in zip(revised, masks))


def test_missing_support_cannot_be_created():
    assert_unchanged([np.empty((0, 3)), np.empty((0, 3))])
    assert_unchanged([np.empty((0, 3)), np.tile([.4, 0., -.3], (50, 1))])
